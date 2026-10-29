"""Stages 7-8: transcribe each VAD chunk with an SttProvider, then apply the guards.

The quality gate decides first. RED skips ASR: the output records the gate and its reasons, so the reply can
say what is wrong instead of returning invented text. AMBER transcribes and carries the gate and reasons into
the transcript as a warning. Providers read raw-norm samples only (`audio_chunks`).

The provider is injected, so this module needs no model runtime. Its name is part of the stage version, so
changing provider re-runs ASR and everything after it.

With a forced language, one provider handles every chunk. Otherwise, when a `langid` stage ran, each
chunk goes where the router decided: a routed language to its provider, a language with no route to the
fallback with that language forced, and an uncertain chunk to the fallback with the language unset, flagged
`language_uncertain`. Without a `langid` stage the fallback auto-detects every chunk.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from meetwise.guards import GuardThresholds, guard
from meetwise.langid import UNCERTAIN, LanguageIds
from meetwise.pipeline import StageContext
from meetwise.probe import QualityReport
from meetwise.stt import SttProvider
from meetwise.types import SAMPLE_RATE, AudioChunk, QualityGate, SampleRange, TranscriptChunk, Word
from meetwise.vad import audio_chunks, load_chunks


@dataclass(frozen=True, slots=True)
class AsrResult:
    provider: str
    gate: QualityGate
    gate_reasons: tuple[str, ...]
    skipped: bool
    chunks: tuple[TranscriptChunk, ...] = ()
    words: tuple[tuple[Word, ...], ...] = ()  # per chunk, same order; empty when the provider gives none

    @property
    def text(self) -> str:
        return " ".join(c.text for c in self.chunks if c.text)

    def dump(self, out: Path) -> None:
        doc = asdict(self)
        doc["gate"] = self.gate.value
        (out / "asr.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))

    @classmethod
    def load(cls, stage_dir: Path) -> AsrResult:
        d = json.loads((stage_dir / "asr.json").read_text())
        chunks = tuple(
            TranscriptChunk(**{**c, "span": SampleRange(**c["span"]), "guard_flags": tuple(c["guard_flags"])})
            for c in d["chunks"]
        )
        words = tuple(
            tuple(Word(w["text"], SampleRange(**w["span"]), w["confidence"]) for w in ws) for ws in d["words"]
        )
        return cls(
            d["provider"], QualityGate(d["gate"]), tuple(d["gate_reasons"]), d["skipped"], chunks, words
        )


@dataclass(frozen=True, slots=True)
class Pick:
    """Where one chunk goes: the provider, the language to force (None: the provider detects), extra flags."""

    provider: SttProvider
    language: str | None
    confidence: float | None = None  # the router's, when it decided the language
    flags: tuple[str, ...] = ()


def transcribe(
    provider: SttProvider,
    chunks: Iterable[tuple[AudioChunk, float]],
    report: QualityReport,
    *,
    language: str | None = None,
    prompt_terms: Sequence[str] = (),
    t: GuardThresholds | None = None,
    pick: Callable[[int], Pick] | None = None,
) -> AsrResult:
    """`chunks` pairs each AudioChunk with its VAD speech seconds. `pick` routes a chunk by its id; without it
    every chunk goes to `provider` with `language`."""
    t = t or GuardThresholds()
    if report.gate == QualityGate.RED:
        return AsrResult(provider.name, report.gate, report.reasons, skipped=True)
    out, words = [], []
    for chunk, speech_s in chunks:
        p = pick(chunk.chunk_id) if pick else Pick(provider, language)
        ct = p.provider.transcribe(chunk, language=p.language, prompt_terms=prompt_terms)
        text, flags = guard(ct, speech_s, t)
        confidence = p.confidence if p.confidence is not None else ct.language_prob
        out.append(
            TranscriptChunk(
                chunk.chunk_id, chunk.span, ct.language, confidence, ct.provider, text, p.flags + flags
            )
        )
        words.append(ct.words if text else ())
    return AsrResult(provider.name, report.gate, report.reasons, False, tuple(out), tuple(words))


@dataclass
class Asr:
    provider: SttProvider
    language: str | None = None
    prompt_terms: tuple[str, ...] = ()
    thresholds: GuardThresholds = field(default_factory=GuardThresholds)
    routes: Mapping[str, SttProvider] = field(default_factory=dict)  # language -> provider, from the router
    name: str = "asr"

    @property
    def version(self) -> str:
        v = f"1+{self.provider.name}+{self.language or 'auto'}"
        if self.routes:
            v += "+" + ",".join(f"{k}={p.name}" for k, p in sorted(self.routes.items()))
        return v

    def _pick(self, ids: LanguageIds) -> Callable[[int], Pick]:
        decisions = ids.by_chunk()

        def pick(chunk_id: int) -> Pick:
            d = decisions.get(chunk_id)
            if d is None or d.language is None:
                return Pick(self.provider, None, None, (UNCERTAIN,))
            return Pick(self.routes.get(d.language, self.provider), d.language, d.prob)

        return pick

    def run(self, ctx: StageContext, out: Path) -> None:
        report = QualityReport.load(ctx.outputs["probe"])
        langid = None if self.language else ctx.outputs.get("langid")
        routed = self._pick(LanguageIds.load(langid)) if langid else None
        vad_dir = ctx.outputs["vad"]
        speech = {c.chunk_id: c.speech_samples / SAMPLE_RATE for c in load_chunks(vad_dir)}
        pairs = (
            (c, speech[c.chunk_id])
            for c in audio_chunks(ctx.recording.sha256, ctx.outputs["decode"], vad_dir)
        )
        transcribe(
            self.provider,
            pairs,
            report,
            language=self.language,
            prompt_terms=self.prompt_terms,
            t=self.thresholds,
            pick=routed,
        ).dump(out)
