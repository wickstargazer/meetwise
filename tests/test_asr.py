"""Stages 7-8 with a fake provider: gate handling, guards, stage output. No model runtime."""

import hashlib
import wave

import numpy as np

from meetwise.asr import Asr, AsrResult, transcribe
from meetwise.decode import Decode
from meetwise.pipeline import run_pipeline
from meetwise.probe import Probe, QualityReport
from meetwise.types import AudioChunk, ChunkTranscript, QualityGate, SampleRange, Word
from meetwise.vad import Vad

SHA = hashlib.sha256(b"synthetic").hexdigest()


class Fake:
    name = "fake"
    languages = frozenset({"en"})
    locality = "local"

    def __init__(self, texts=None):
        self.calls = []
        self.texts = texts or {}

    def transcribe(self, chunk, *, language, prompt_terms):
        self.calls.append((chunk.chunk_id, language, tuple(prompt_terms)))
        text = self.texts.get(chunk.chunk_id, f"chunk {chunk.chunk_id} words")
        w = Word("chunk", SampleRange(chunk.span.start, chunk.span.start + 1), 0.9)
        return ChunkTranscript(self.name, language or "en", text, (w,), avg_logprob=-0.2, no_speech_prob=0.01)


def report(gate, reasons=()):
    return QualityReport(gate, tuple(reasons), 0.7, 60.0, 30.0, 0.0, 0.0, ())


def chunks(n):
    return [
        (AudioChunk(SHA, i, SampleRange(i * 16000, (i + 1) * 16000), np.zeros(16000, np.float32)), 0.8)
        for i in range(n)
    ]


def test_red_skips_asr_and_keeps_reasons():
    p = Fake()
    r = transcribe(p, chunks(3), report(QualityGate.RED, ["too_noisy", "echo"]))
    assert r.skipped and p.calls == [] and r.chunks == ()
    assert r.gate == QualityGate.RED and r.gate_reasons == ("too_noisy", "echo")


def test_amber_transcribes_with_warning():
    r = transcribe(Fake(), chunks(2), report(QualityGate.AMBER, ["noisy"]))
    assert not r.skipped and r.gate == QualityGate.AMBER and r.gate_reasons == ("noisy",)
    assert r.text == "chunk 0 words chunk 1 words"


def test_guards_apply_per_chunk_and_drop_words_of_dropped_text():
    p = Fake({1: "Thanks for watching!"})
    r = transcribe(p, chunks(3), report(QualityGate.GREEN), language="sv", prompt_terms=["Northwind"])
    assert [c.text for c in r.chunks] == ["chunk 0 words", "", "chunk 2 words"]
    assert r.chunks[1].guard_flags == ("phantom",)
    assert [len(w) for w in r.words] == [1, 0, 1]
    assert p.calls[0] == (0, "sv", ("Northwind",))


def test_result_round_trips(tmp_path):
    r = transcribe(Fake({0: "ภาษีซื้อ ok"}), chunks(2), report(QualityGate.AMBER, ["echo"]))
    r.dump(tmp_path)
    assert AsrResult.load(tmp_path) == r
    assert "ภาษีซื้อ" in (tmp_path / "asr.json").read_text()


def test_version_names_the_provider():
    assert Asr(Fake()).version == "1+fake+auto"
    assert Asr(Fake(), language="th").version != Asr(Fake()).version


def test_stage_end_to_end_on_tts(tmp_path, speech):
    from meetwise.checkpoint import ArtifactStore

    src = tmp_path / "in.wav"
    with wave.open(str(src), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.clip(speech, -1, 1) * 32767).astype("<i2").tobytes())
    store = ArtifactStore(tmp_path / "store")
    rec = store.ingest(src, max_bytes=50 << 20)
    p = Fake()
    out = run_pipeline(rec, [Decode(), Probe(), Vad(), Asr(p)], store)
    r = AsrResult.load(out["asr"])
    assert r.gate == QualityGate.GREEN and not r.skipped and len(r.chunks) == len(p.calls) >= 1
    assert all(c.span.end <= len(speech) for c in r.chunks)
    run_pipeline(rec, [Decode(), Probe(), Vad(), Asr(p)], store)
    assert len(p.calls) == len(r.chunks)  # checkpoint: not re-run
