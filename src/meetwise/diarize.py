"""Stage 6: diarization as speaker hints.

A diarizer labels who-spoke-when with anonymous labels (`SPEAKER_00`, ...). Those labels are hints only: they
never name anyone and never make anyone an action-item owner (`summary` enforces the owner rule). Naming a
cluster is the user's job, through a naming prompt, which is later work.

Like VAD, diarization reads the enhanced signal when an `enhance` stage ran (stage 4), else raw-norm, and its
spans index raw-norm. The diarizer is injected, so this module needs no model runtime.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from meetwise.decode import RAW_NORM, load_samples
from meetwise.pipeline import StageContext
from meetwise.types import SAMPLE_RATE, SampleRange, SpeakerHint, TranscriptChunk
from meetwise.vad import ENHANCED

# Smaller clusters are never asked about; they show as "Speaker ?".
ASK_MIN_SPEECH = 8 * SAMPLE_RATE


@runtime_checkable
class Diarizer(Protocol):
    name: str

    def diarize(self, samples: NDArray[np.float32]) -> Sequence[SpeakerHint]:
        """Speaker turns of 16 kHz mono samples, in any order; turns of different labels may overlap."""
        ...


@dataclass(frozen=True, slots=True)
class Diarization:
    diarizer: str
    hints: tuple[SpeakerHint, ...]

    def speech(self) -> dict[str, int]:
        """Samples of speech per label, largest first."""
        out: dict[str, int] = {}
        for h in self.hints:
            out[h.label] = out.get(h.label, 0) + h.span.end - h.span.start
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def askable(self) -> list[str]:
        return [label for label, n in self.speech().items() if n >= ASK_MIN_SPEECH]

    def dump(self, out: Path) -> None:
        (out / "speakers.json").write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, stage_dir: Path) -> Diarization:
        d = json.loads((stage_dir / "speakers.json").read_text())
        return cls(
            d["diarizer"], tuple(SpeakerHint(h["label"], SampleRange(**h["span"])) for h in d["hints"])
        )


def clean(hints: Sequence[SpeakerHint], n_samples: int) -> tuple[SpeakerHint, ...]:
    """Clipped to the recording, turns that start past its end dropped, sorted by start."""
    out = []
    for h in hints:
        if h.span.start < n_samples:
            out.append(SpeakerHint(h.label, SampleRange(h.span.start, min(n_samples, h.span.end))))
    return tuple(sorted(out, key=lambda h: (h.span.start, h.span.end, h.label)))


def chunk_speaker(chunk: TranscriptChunk, hints: Sequence[SpeakerHint]) -> tuple[str | None, float]:
    """The label covering most of a chunk, and the share of the chunk it covers. A hint, not a name."""
    cover: dict[str, int] = {}
    for h in hints:
        n = min(chunk.span.end, h.span.end) - max(chunk.span.start, h.span.start)
        if n > 0:
            cover[h.label] = cover.get(h.label, 0) + n
    if not cover:
        return None, 0.0
    label = max(cover, key=lambda k: cover[k])
    return label, cover[label] / max(chunk.span.end - chunk.span.start, 1)


@dataclass
class Diarize:
    diarizer: Diarizer
    name: str = "diarize"

    @property
    def version(self) -> str:
        return f"1+{self.diarizer.name}"

    def run(self, ctx: StageContext, out: Path) -> None:
        enh = ctx.outputs.get("enhance")
        x = load_samples(enh, ENHANCED) if enh else load_samples(ctx.outputs["decode"], RAW_NORM)
        hints = clean(self.diarizer.diarize(np.asarray(x)), len(x))
        Diarization(self.diarizer.name, hints).dump(out)
