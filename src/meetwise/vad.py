"""Stage 5: VAD chunking.

silero-vad scores every 32 ms frame. Chunks are cut only inside silence of at least 300 ms, aim for 20-30 s
and never exceed 30 s. If speech runs past 30 s with no such silence, the chunk is cut at the least
speech-like frame in its last 10 s and flagged `forced_cut`, so the guards and the reviewer know a word may
be split.

VAD reads the enhanced signal when an `enhance` stage ran (stage 4), else raw-norm. With `split_channels`
from decode, every non-silent channel is scored too and a frame counts as speech if any source says so.
Chunk spans always index raw-norm, which is what ASR reads (`audio_chunks`).

The model is the vendored silero_vad.onnx (models/silero_vad/SOURCE), run by onnxruntime: no PyTorch and
no download.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from meetwise.decode import RAW_NORM, DecodeInfo, load_samples
from meetwise.pipeline import StageContext
from meetwise.types import SAMPLE_RATE, AudioChunk, SampleRange

FRAME = 512  # 32 ms at 16 kHz, fixed by the model
_CONTEXT = 64
MODEL = Path(__file__).parent / "models" / "silero_vad" / "silero_vad.onnx"
MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
ENHANCED = "enhanced.f32"  # what a stage-4 `enhance` stage will write


def _ms(ms: int) -> int:
    return ms * SAMPLE_RATE // 1000


@dataclass(frozen=True, slots=True)
class ChunkParams:
    threshold: float = 0.5
    neg_threshold: float = 0.35  # hysteresis: speech ends only below this
    min_speech: int = _ms(250)
    min_silence: int = _ms(100)  # shorter dips stay inside one speech segment
    min_cut_silence: int = _ms(300)
    target: int = 20 * SAMPLE_RATE
    max_len: int = 30 * SAMPLE_RATE
    pad: int = _ms(100)


@dataclass(frozen=True, slots=True)
class VadChunk:
    chunk_id: int
    start: int
    end: int
    speech_samples: int
    forced_cut: bool = False

    @property
    def span(self) -> SampleRange:
        return SampleRange(self.start, self.end)


class SileroVad:
    def __init__(self, model: Path = MODEL) -> None:
        import onnxruntime as ort  # heavy import; only when the stage runs

        blob = model.read_bytes()
        if hashlib.sha256(blob).hexdigest() != MODEL_SHA256:
            raise RuntimeError("silero_vad.onnx does not match the pinned sha256")
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(blob, sess_options=opts, providers=["CPUExecutionProvider"])

    def probs(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Speech probability per FRAME; a short last frame is zero-padded."""
        n = -(-len(x) // FRAME)
        padded = np.zeros(n * FRAME + _CONTEXT, dtype=np.float32)
        padded[_CONTEXT : _CONTEXT + len(x)] = x
        state = np.zeros((2, 1, 128), dtype=np.float32)
        sr = np.array(SAMPLE_RATE, dtype=np.int64)
        out = np.empty(n, dtype=np.float32)
        for i in range(n):
            frame = padded[i * FRAME : i * FRAME + FRAME + _CONTEXT][None, :]
            p, state = self._session.run(None, {"input": frame, "state": state, "sr": sr})
            out[i] = p[0, 0]
        return out


def speech_segments(probs: NDArray[np.float32], total: int, p: ChunkParams) -> list[SampleRange]:
    """Hysteresis over frame probabilities, gaps under min_silence closed, runs under min_speech dropped."""
    raw: list[list[int]] = []
    on = False
    for i, pr in enumerate(probs):
        if not on and pr >= p.threshold:
            on = True
            raw.append([i * FRAME, total])
        elif on and pr < p.neg_threshold:
            on = False
            raw[-1][1] = i * FRAME
    merged: list[list[int]] = []
    for s, e in raw:
        if merged and s - merged[-1][1] < p.min_silence:
            merged[-1][1] = e
        else:
            merged.append([s, min(e, total)])
    return [SampleRange(s, e) for s, e in merged if e - s >= p.min_speech]


def plan_chunks(
    segments: Sequence[SampleRange], probs: NDArray[np.float32], total: int, p: ChunkParams
) -> list[VadChunk]:
    # Islands: speech separated by at least min_cut_silence. Only the gaps between islands may be cut.
    islands: list[tuple[int, int, list[SampleRange]]] = []
    for seg in segments:
        if islands and seg.start - islands[-1][1] < p.min_cut_silence:
            s, _, segs = islands[-1]
            islands[-1] = (s, seg.end, [*segs, seg])
        else:
            islands.append((seg.start, seg.end, [seg]))

    # Pieces: islands, with any island longer than max_len force-cut at its quietest frame.
    pieces: list[tuple[int, int, int, bool, bool]] = []  # start, end, speech, cut_before, cut_after
    for s, e, segs in islands:
        cuts = [s]
        while e - cuts[-1] > p.max_len - 2 * p.pad:
            lo, hi = cuts[-1] + p.max_len - 2 * p.pad - 10 * SAMPLE_RATE, cuts[-1] + p.max_len - 2 * p.pad
            f_lo, f_hi = -(-lo // FRAME), hi // FRAME
            cuts.append(int(f_lo + np.argmin(probs[f_lo:f_hi])) * FRAME)
        cuts.append(e)
        for k in range(len(cuts) - 1):
            a, b = cuts[k], cuts[k + 1]
            speech = sum(max(0, min(g.end, b) - max(g.start, a)) for g in segs)
            pieces.append((a, b, speech, k > 0, k < len(cuts) - 2))

    # Greedy packing into chunks of at most max_len including padding; never across a forced cut.
    groups: list[list[tuple[int, int, int, bool, bool]]] = []
    for piece in pieces:
        g = groups[-1] if groups else None
        if g and not piece[3] and piece[1] + p.pad - (g[0][0] - p.pad) <= p.max_len and _short(g, p):
            g.append(piece)
        else:
            groups.append([piece])

    chunks: list[VadChunk] = []
    for i, g in enumerate(groups):
        start = g[0][0] if g[0][3] else g[0][0] - p.pad
        end = g[-1][1] if g[-1][4] else g[-1][1] + p.pad
        # Padding never reaches past the middle of the silence shared with a neighbouring chunk.
        lo = (groups[i - 1][-1][1] + g[0][0]) // 2 if i > 0 and not g[0][3] else 0
        hi = (g[-1][1] + groups[i + 1][0][0]) // 2 if i + 1 < len(groups) and not g[-1][4] else total
        start, end = max(start, lo), min(end, hi)
        forced = g[0][3] or g[-1][4]
        chunks.append(VadChunk(i, start, end, sum(x[2] for x in g), forced))
    return chunks


def _short(group: list[tuple[int, int, int, bool, bool]], p: ChunkParams) -> bool:
    """Keep filling while the chunk is under target; past it, close at the next cuttable gap."""
    return group[-1][1] - group[0][0] < p.target


@dataclass
class Vad:
    name: str = "vad"
    version: str = "1"
    params: ChunkParams = field(default_factory=ChunkParams)

    def run(self, ctx: StageContext, out: Path) -> None:
        dec_dir = ctx.outputs["decode"]
        info = DecodeInfo.load(dec_dir)
        enh = ctx.outputs.get("enhance")
        sources = [(enh, ENHANCED) if enh else (dec_dir, RAW_NORM)]
        if info.split_channels:
            sources += [
                (dec_dir, f"ch{k}.f32") for k in range(info.num_channels) if k not in info.silent_channels
            ]

        model = SileroVad()
        per_source = [model.probs(np.asarray(load_samples(d, n))) for d, n in sources]
        probs = np.max(np.stack(per_source), axis=0)
        np.save(out / "probs.npy", np.stack(per_source).astype(np.float16))

        segs = speech_segments(probs, info.num_samples, self.params)
        chunks = plan_chunks(segs, probs, info.num_samples, self.params)
        doc = {
            "frame": FRAME,
            "params": asdict(self.params),
            "sources": [n for _, n in sources],
            "speech": [[s.start, s.end] for s in segs],
            "chunks": [asdict(c) for c in chunks],
        }
        (out / "vad.json").write_text(json.dumps(doc, indent=1))


def load_chunks(vad_dir: Path) -> list[VadChunk]:
    return [VadChunk(**c) for c in json.loads((vad_dir / "vad.json").read_text())["chunks"]]


def audio_chunks(recording_sha: str, decode_dir: Path, vad_dir: Path) -> Iterator[AudioChunk]:
    """What ASR gets: raw-norm samples for each VAD chunk, never the enhanced signal."""
    x = load_samples(decode_dir)
    for c in load_chunks(vad_dir):
        yield AudioChunk(recording_sha, c.chunk_id, c.span, np.array(x[c.start : c.end]))
