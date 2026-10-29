"""Whisper models through faster-whisper (CTranslate2), on CUDA or CPU.

Three providers share this code, one per CTranslate2 model in the pack:
- `FasterWhisper()`: whisper-large-v3 turbo, the en provider and the router's fallback.
- `FasterWhisper(THONBURIAN)`: Thonburian Whisper (medium fine-tuned on Thai), the th provider.
- `FasterWhisper(KB_WHISPER)`: KB-Whisper large (large-v3 fine-tuned on Swedish), the sv provider.

The model directory holds its own tokenizer, so nothing is fetched from the hub. Weights run as int8 (with
float16 activations on CUDA), which keeps large-v3 near 1.5 GB of GPU memory. Decoding is greedy at
temperature 0 with no conditioning on the previous chunk, so a rerun gives the same text and one bad chunk
cannot steer the next.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from meetwise.model_pack import model_dir
from meetwise.runners import free_gpu
from meetwise.types import SAMPLE_RATE, AudioChunk, ChunkTranscript, SampleRange, Word

TURBO = "whisper-large-v3-turbo-ct2"
THONBURIAN = "thonburian-whisper-th-medium-ct2"
KB_WHISPER = "kb-whisper-large-sv-ct2"

# Turbo is multilingual: every language the router knows. The fine-tunes are one language each.
LANGUAGES = {
    TURBO: frozenset({"th", "en", "sv", "hi", "pa"}),
    THONBURIAN: frozenset({"th"}),
    KB_WHISPER: frozenset({"sv"}),
}


def cuda_available() -> bool:
    import ctranslate2

    return ctranslate2.get_cuda_device_count() > 0


class FasterWhisper:
    locality = "local"

    def __init__(
        self, model: str = TURBO, root: Path | None = None, device: str | None = None, threads: int = 0
    ) -> None:
        self.name = model
        self.languages = LANGUAGES[model]
        self._path = model_dir(model, root)
        self._device = device
        self._threads = threads
        self._model: Any = None

    def _loaded(self) -> Any:
        if self._model is None:
            os.environ["HF_HUB_OFFLINE"] = "1"
            from faster_whisper import WhisperModel  # heavy; only when a chunk is transcribed

            device = self._device or ("cuda" if cuda_available() else "cpu")
            self._model = WhisperModel(
                str(self._path),
                device=device,
                compute_type="int8_float16" if device == "cuda" else "int8",
                cpu_threads=self._threads,
                local_files_only=True,
            )
        return self._model

    def release(self) -> None:
        self._model = None
        free_gpu()

    def transcribe(
        self, chunk: AudioChunk, *, language: str | None, prompt_terms: Sequence[str]
    ) -> ChunkTranscript:
        segments, info = self._loaded().transcribe(
            chunk.samples,
            language=language,
            initial_prompt=", ".join(prompt_terms) or None,
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            word_timestamps=True,
            vad_filter=False,
        )
        return to_transcript(self.name, list(segments), info.language or language, chunk)


def to_transcript(
    provider: str, segments: Sequence[Any], language: str | None, chunk: AudioChunk
) -> ChunkTranscript:
    """faster-whisper segments as a ChunkTranscript: words on the chunk's samples, duration-weighted means."""
    segs = [s for s in segments if s.text.strip()]
    dur = [max(s.end - s.start, 1e-3) for s in segs]

    def mean(key: str) -> float | None:
        return sum(getattr(s, key) * d for s, d in zip(segs, dur, strict=True)) / sum(dur) if segs else None

    words = []
    base, last = chunk.span.start, chunk.span.end
    for s in segs:
        for w in s.words or ():
            a = min(base + int(w.start * SAMPLE_RATE), last - 1)
            b = min(max(base + int(w.end * SAMPLE_RATE), a + 1), last)
            words.append(Word(w.word.strip(), SampleRange(a, b), round(float(w.probability), 3)))
    return ChunkTranscript(
        provider=provider,
        language=language or "und",
        text=" ".join(s.text.strip() for s in segs),
        words=tuple(words),
        avg_logprob=mean("avg_logprob"),
        no_speech_prob=mean("no_speech_prob"),
        compression_ratio=mean("compression_ratio"),
    )
