"""Stage 7a and the router with a fake language-ID model and fake providers. No model runtime."""

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest

from meetwise.asr import Asr, AsrResult
from meetwise.decode import RAW_NORM
from meetwise.langid import (
    UNCERTAIN,
    LangId,
    LanguageId,
    LanguageIds,
    Router,
    decide,
    windows,
)
from meetwise.pipeline import StageContext
from meetwise.probe import QualityReport
from meetwise.types import SAMPLE_RATE as SR
from meetwise.types import ChunkTranscript, QualityGate

R = Router.load()


def test_default_table_covers_the_five_languages():
    assert set(R.languages) == {"th", "en", "sv", "hi", "pa"}
    assert set(R.routes) <= set(R.languages) and R.fallback


def test_confident_language_is_decided_and_renormalised_over_the_set():
    d = decide(0, {"th": 0.6, "lo": 0.3, "en": 0.1}, 5.0, R)
    assert d.language == "th" and d.reason is None
    assert d.top[0] == ("th", pytest.approx(6 / 7)) and d.out_of_set == pytest.approx(0.3)


def test_uncertain_cases():
    assert decide(0, {"th": 0.5, "en": 0.5}, 5.0, R).reason == "unclear"
    assert decide(0, {"en": 1.0}, 0.5, R).reason == "too_short"
    assert decide(0, {"ja": 0.9, "en": 0.1}, 5.0, R).reason == "out_of_set"
    assert decide(0, {"ja": 1.0}, 5.0, R).reason == "out_of_set"
    mixed = decide(0, {"th": 0.99, "en": 0.01}, 9.0, R, [{"th": 1.0}, {"th": 0.5, "en": 0.5}, {"en": 1.0}])
    assert mixed.language is None and mixed.reason == "mixed" and mixed.windows == ("th", None, "en")


def test_windows_are_back_to_back_and_a_short_tail_joins_the_last():
    assert windows(np.zeros(5 * SR, np.float32), R) == []
    ws = windows(np.zeros(int(10.5 * SR), np.float32), R)
    assert [len(w) for w in ws] == [3 * SR, 3 * SR, int(4.5 * SR)]


class FakeLid:
    name = "fake-lid"

    def __init__(self, by_level):
        self.by_level = by_level

    def scores(self, samples):
        return self.by_level[round(float(samples[0]), 1)]


class FakeStt:
    languages = frozenset({"th", "en", "sv"})
    locality = "local"

    def __init__(self, name):
        self.name = name
        self.calls = []

    def transcribe(self, chunk, *, language, prompt_terms):
        self.calls.append((chunk.chunk_id, language))
        return ChunkTranscript(self.name, language or "en", f"chunk {chunk.chunk_id} words", (), -0.2, 0.01)


def stage_dirs(tmp_path):
    """Four 2-s chunks whose constant level tells the fake model which language it is."""
    dec, vad, probe = (tmp_path / n for n in ("decode", "vad", "probe"))
    for d in (dec, vad, probe):
        d.mkdir()
    np.concatenate([np.full(2 * SR, v, np.float32) for v in (0.1, 0.2, 0.3, 0.4)]).tofile(dec / RAW_NORM)
    chunks = [
        {"chunk_id": i, "start": i * 2 * SR, "end": (i + 1) * 2 * SR, "speech_samples": 2 * SR}
        for i in range(4)
    ]
    (vad / "vad.json").write_text(json.dumps({"chunks": chunks}))
    report = QualityReport(QualityGate.GREEN, (), 0.7, 60.0, 30.0, 0.0, 0.0, ())
    (probe / "probe.json").write_text(json.dumps(dataclasses.asdict(report)))
    return {"decode": dec, "vad": vad, "probe": probe}


def test_langid_stage_then_asr_routes_each_chunk(tmp_path):
    outputs = stage_dirs(tmp_path)
    lid = FakeLid(
        {0.1: {"th": 0.95, "lo": 0.05}, 0.2: {"en": 1.0}, 0.3: {"sv": 1.0}, 0.4: {"th": 0.5, "en": 0.5}}
    )
    assert isinstance(lid, LanguageId)
    rec = SimpleNamespace(sha256="0" * 64)
    (tmp_path / "langid").mkdir()
    LangId(lid, R).run(StageContext(rec, None, outputs), tmp_path / "langid")
    ids = LanguageIds.load(tmp_path / "langid")
    assert [d.language for d in ids.decisions] == ["th", "en", "sv", None]

    th, fallback = FakeStt("thai-model"), FakeStt("multilingual")
    asr = Asr(fallback, routes={"th": th})
    (tmp_path / "asr").mkdir()
    asr.run(StageContext(rec, None, {**outputs, "langid": tmp_path / "langid"}), tmp_path / "asr")
    r = AsrResult.load(tmp_path / "asr")
    assert th.calls == [(0, "th")]
    assert fallback.calls == [(1, "en"), (2, "sv"), (3, None)]  # sv has no route yet: forced on the fallback
    assert [c.provider for c in r.chunks] == ["thai-model", "multilingual", "multilingual", "multilingual"]
    assert r.chunks[3].guard_flags == (UNCERTAIN,) and r.chunks[0].guard_flags == ()
    assert r.chunks[0].language_confidence == 0.95 / 0.95


def test_forced_language_ignores_langid(tmp_path):
    outputs = stage_dirs(tmp_path)
    (tmp_path / "langid").mkdir()
    LanguageIds("fake", ()).dump(tmp_path / "langid")
    p = FakeStt("one")
    (tmp_path / "asr").mkdir()
    rec = SimpleNamespace(sha256="0" * 64)
    Asr(p, language="th").run(
        StageContext(rec, None, {**outputs, "langid": tmp_path / "langid"}), tmp_path / "asr"
    )
    assert p.calls == [(i, "th") for i in range(4)]


def test_versions_change_with_the_table_and_the_routes():
    lid = FakeLid({})
    assert LangId(lid, R).version != LangId(lid, dataclasses.replace(R, min_prob=0.7)).version
    assert Asr(FakeStt("a")).version == "1+a+auto"
    assert Asr(FakeStt("a"), routes={"th": FakeStt("b")}).version == "1+a+auto+th=b"
