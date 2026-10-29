"""`meetwise run FILE [FILE...]`: the whole pipeline on local files, with the Linux runners.

Stages: decode, quality probe, VAD chunking, speaker hints (pyannote 3.1), language ID per chunk
(VoxLingua107) and the router, ASR with guards (faster-whisper), glossary correction and the cited summary
(Qwen3-8B through llama.cpp). `--language` skips language ID and sends every chunk to that language's
provider. Each model is released when its last stage finishes, so the stages share a small GPU instead of
needing room for all models at once.

Writes `transcript.md`, `summary.md` and `summary.json` into `--out` (default `<file>.meetwise/` next to the
recording). Stage results are kept in `--work` (default `$XDG_CACHE_HOME/meetwise`), so a rerun of the same
file skips finished stages. Needs the `runners` extra and the models from scripts/fetch_models.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from meetwise.asr import Asr
from meetwise.checkpoint import ArtifactStore
from meetwise.decode import Decode
from meetwise.diarize import Diarize
from meetwise.glossary import Correct
from meetwise.langid import LangId, Router
from meetwise.pipeline import StageEvent, run_pipeline
from meetwise.probe import Probe
from meetwise.report import write_report
from meetwise.summary import Summarise
from meetwise.vad import Vad

log = logging.getLogger("meetwise")


def work_root() -> Path:
    cache = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(cache) / "meetwise"


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="meetwise")
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="transcribe and summarise recordings")
    run.add_argument("files", nargs="+", type=Path)
    run.add_argument("--out", type=Path, default=None, help="output directory (one file only)")
    run.add_argument("--work", type=Path, default=None, help="stage cache; default $XDG_CACHE_HOME/meetwise")
    run.add_argument("--language", default=None, help="force one language (BCP-47); default: detect")
    run.add_argument("--summary-language", default="en", help="language of the summary (BCP-47)")
    run.add_argument(
        "--glossary", type=Path, default=None, help='JSON {"<lang>": ["term", ...]}; default: no corrections'
    )
    run.add_argument("--device", choices=("cuda", "cpu"), default=None, help="default: cuda when present")
    run.add_argument("--gpu-layers", type=int, default=-1, help="LLM layers on the GPU; -1 all, 0 none")
    run.add_argument("--context", type=int, default=8192, help="LLM context in tokens")
    run.add_argument("--max-mb", type=int, default=2048, help="largest recording accepted")
    return ap


def main(argv: list[str] | None = None) -> None:
    a = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.out and len(a.files) > 1:
        raise SystemExit("--out takes one file; without it each result goes next to its recording")
    # Heavy imports here, so `meetwise --help` works without the runners extra.
    from meetwise.runners.diarize_pyannote import PyannoteDiarizer
    from meetwise.runners.langid_voxlingua import VoxLingua
    from meetwise.runners.llm_llama import LlamaCpp
    from meetwise.runners.stt_ct2 import FasterWhisper

    router = Router.load()
    diarizer = PyannoteDiarizer(device=a.device)
    if a.language:
        whisper = {"forced": FasterWhisper(router.routes.get(a.language, router.fallback), device=a.device)}
        routing, langid = [], None
        asr = Asr(whisper["forced"], language=a.language)
    else:
        whisper = {n: FasterWhisper(n, device=a.device) for n in {*router.routes.values(), router.fallback}}
        langid = VoxLingua()
        routing = [LangId(langid, router)]
        asr = Asr(whisper[router.fallback], routes={k: whisper[n] for k, n in router.routes.items()})
    llm = LlamaCpp(gpu_layers=0 if a.device == "cpu" else a.gpu_layers, context=a.context)
    glossary = json.loads(a.glossary.read_text()) if a.glossary else {}
    stages = [
        Decode(),
        Probe(),
        Vad(),
        Diarize(diarizer),
        *routing,
        asr,
        Correct(llm, {k: frozenset(v) for k, v in glossary.items()}),
        Summarise(llm, a.summary_language),
    ]
    # The last stage that uses each model frees it.
    release = {
        "diarize": [diarizer],
        "langid": [langid] if langid else [],
        "asr": list(whisper.values()),
        "summary": [llm],
    }

    def on_event(e: StageEvent) -> None:
        log.info("%s %s", e.stage, e.status)
        if e.status in ("done", "failed"):
            for runner in release.get(e.stage, ()):
                runner.release()

    store = ArtifactStore(a.work or work_root())
    for f in a.files:
        rec = store.ingest(f, max_bytes=a.max_mb << 20)
        outputs = run_pipeline(rec, stages, store, on_event)
        dest = a.out or f.with_name(f.name + ".meetwise")
        for p in write_report(outputs, dest):
            print(p)


if __name__ == "__main__":
    main()
