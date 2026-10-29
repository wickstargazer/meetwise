"""The readable output: `transcript.md` and `summary.md` from the finished stages.

The transcript is one line per chunk, `[mm:ss] Speaker N (lang): text`. The speaker is the diarization label
that overlaps the chunk most, renumbered by first appearance; it is a hint, so a chunk with no overlap has
no speaker. The summary lists decisions, action items and key points, each with its `[mm:ss]` citations,
and the quality gate with its reasons when it is not green.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path

from meetwise.asr import AsrResult
from meetwise.diarize import Diarization
from meetwise.glossary import Correction
from meetwise.types import SpeakerHint, TranscriptChunk, mmss

SECTIONS = (("decisions", "Decisions"), ("action_items", "Action items"), ("key_points", "Key points"))


def speaker_of(chunk: TranscriptChunk, hints: Sequence[SpeakerHint]) -> str | None:
    overlap: dict[str, int] = {}
    for h in hints:
        n = min(chunk.span.end, h.span.end) - max(chunk.span.start, h.span.start)
        if n > 0:
            overlap[h.label] = overlap.get(h.label, 0) + n
    return max(overlap, key=lambda k: overlap[k]) if overlap else None


def transcript_md(chunks: Sequence[TranscriptChunk], hints: Sequence[SpeakerHint] = ()) -> str:
    names: dict[str, str] = {}
    lines = ["# Transcript", ""]
    for c in chunks:
        if not c.text:
            continue
        label = speaker_of(c, hints)
        who = names.setdefault(label, f"Speaker {len(names) + 1}") + " " if label else ""
        lines.append(f"[{mmss(c.span.start)}] {who}({c.language}): {c.text}")
    return "\n".join(lines) + "\n"


def summary_md(doc: dict) -> str:
    s = doc["summary"]
    lines = ["# Summary", ""]
    if doc["gate"] != "green":
        lines += [f"Quality: {doc['gate'].upper()}. " + "; ".join(doc.get("reasons", ())), ""]
    for key, title in SECTIONS:
        lines += [f"## {title}", ""]
        items = s.get(key, [])
        for it in items:
            cites = " ".join(f"[{mmss(c)}]" for c in it["citations"])
            owner = f" (owner: {it['owner'] or 'unassigned'})" if key == "action_items" else ""
            lines.append(f"- {it['text']}{owner} {cites}")
        lines += [*([] if items else ["- none"]), ""]
    return "\n".join(lines).rstrip() + "\n"


def write_report(outputs: dict[str, Path], dest: Path) -> list[Path]:
    """transcript.md, summary.md and the JSON they come from, into `dest`. Returns the files written."""
    dest.mkdir(parents=True, exist_ok=True)
    chunks = (
        Correction.load_chunks(outputs["correct"])
        if "correct" in outputs
        else AsrResult.load(outputs["asr"]).chunks
    )
    hints = Diarization.load(outputs["diarize"]).hints if "diarize" in outputs else ()
    written = [dest / "transcript.md"]
    written[0].write_text(transcript_md(chunks, hints))
    if "summary" in outputs:
        src = outputs["summary"] / "summary.json"
        (dest / "summary.md").write_text(summary_md(json.loads(src.read_text())))
        shutil.copyfile(src, dest / "summary.json")
        written += [dest / "summary.md", dest / "summary.json"]
    return written
