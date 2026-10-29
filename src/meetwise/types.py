"""Domain types. Every position is an integer sample offset into raw-norm (16 kHz mono)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE = 16_000
_SHA256 = re.compile(r"[0-9a-f]{64}")


def mmss(sample: int) -> str:
    """Sample offset as the `[mm:ss]` form summaries cite."""
    s = sample // SAMPLE_RATE
    return f"{s // 60:02d}:{s % 60:02d}"


class QualityGate(StrEnum):
    GREEN = "green"
    AMBER = "amber"
    RED = "red"


@dataclass(frozen=True, slots=True)
class SampleRange:
    start: int
    end: int

    def __post_init__(self) -> None:
        if not 0 <= self.start < self.end:
            raise ValueError(f"bad sample range {self.start}..{self.end}")


@dataclass(frozen=True, slots=True)
class MeetingRecording:
    """One uploaded or dropped file. Its identity is the sha256 of its bytes, never its name."""

    sha256: str
    size_bytes: int
    suffix: str = ""  # as received, e.g. ".mkv"; a hint only, decode trusts ffprobe
    container: str | None = None  # ffprobe format name; decode records it in decode.json

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError("sha256 must be 64 lowercase hex chars")
        if self.size_bytes <= 0:
            raise ValueError("empty recording")


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """One VAD chunk handed to an SttProvider: raw-norm samples, never the enhanced signal."""

    recording_sha: str
    chunk_id: int
    span: SampleRange
    samples: NDArray[np.float32] = field(repr=False)

    def __post_init__(self) -> None:
        if self.samples.dtype != np.float32 or self.samples.ndim != 1:
            raise ValueError("samples must be 1-D float32")
        if len(self.samples) != self.span.end - self.span.start:
            raise ValueError("samples do not match span")


@dataclass(frozen=True, slots=True)
class Word:
    text: str
    span: SampleRange
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class ChunkTranscript:
    """What a provider returns for one chunk. Confidence signals are None when the vendor has none."""

    provider: str
    language: str
    text: str
    words: tuple[Word, ...] = ()
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    language_prob: float | None = None  # None until the router measures it


@dataclass(frozen=True, slots=True)
class TranscriptChunk:
    chunk_id: int
    span: SampleRange
    language: str
    language_confidence: float | None
    provider: str
    text: str
    guard_flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SpeakerHint:
    """A diarization label. A hint only: it never assigns an owner by itself."""

    label: str
    span: SampleRange


@dataclass(frozen=True, slots=True)
class Transcript:
    recording_sha: str
    gate: QualityGate
    chunks: tuple[TranscriptChunk, ...]
    speaker_hints: tuple[SpeakerHint, ...] = ()

    def __post_init__(self) -> None:
        for a, b in zip(self.chunks, self.chunks[1:], strict=False):
            if b.span.start < a.span.end:
                raise ValueError(f"chunks {a.chunk_id} and {b.chunk_id} overlap or are out of order")

    def chunk_at(self, sample: int) -> TranscriptChunk | None:
        return next((c for c in self.chunks if c.span.start <= sample < c.span.end), None)


@dataclass(frozen=True, slots=True)
class SummaryItem:
    text: str
    citations: tuple[int, ...]  # sample offsets; each must resolve to a chunk
    owner: str | None = None  # action items only; from transcript content, else None ("unassigned")

    def __post_init__(self) -> None:
        if not self.citations:
            raise ValueError("every summary item needs at least one citation")


@dataclass(frozen=True, slots=True)
class MeetingSummary:
    output_language: str
    decisions: tuple[SummaryItem, ...] = ()
    action_items: tuple[SummaryItem, ...] = ()
    key_points: tuple[SummaryItem, ...] = ()

    def unresolved_citations(self, transcript: Transcript) -> list[int]:
        items = (*self.decisions, *self.action_items, *self.key_points)
        return [c for i in items for c in i.citations if transcript.chunk_at(c) is None]
