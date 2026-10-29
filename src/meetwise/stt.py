"""The thin ASR interface: no streaming, no diarization, no vendor options."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

from meetwise.types import AudioChunk, ChunkTranscript


@runtime_checkable
class SttProvider(Protocol):
    name: str  # "typhoon-asr", "whisper-large-v3-turbo", "gemini-vertex", ...
    languages: frozenset[str]  # BCP-47 primary tags it accepts
    locality: Literal["local", "cloud"]

    def transcribe(
        self, chunk: AudioChunk, *, language: str | None, prompt_terms: Sequence[str]
    ) -> ChunkTranscript: ...
