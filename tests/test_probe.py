"""Stage 3 on synthetic TTS speech with noise and clipping added in code. No real recordings."""

import shutil
import wave

import numpy as np
import pytest

from meetwise.checkpoint import ArtifactStore
from meetwise.decode import Decode
from meetwise.pipeline import run_pipeline
from meetwise.probe import GateThresholds, Probe, QualityReport, grade, rt60_estimate, snr_db
from meetwise.types import SAMPLE_RATE as SR
from meetwise.types import QualityGate
from meetwise.vad import FRAME

T = GateThresholds()
TTS = shutil.which("say") or shutil.which("espeak-ng")
needs_media = pytest.mark.skipif(
    not (TTS and shutil.which("ffmpeg")), reason="needs ffmpeg and say/espeak-ng"
)


def good(**kw):
    base = dict(speech_ratio=0.7, speech_s=60.0, snr_db=30.0, clip_ratio=0.0, level_gain_db=5.0, t=T)
    return grade(**{**base, **kw})


@pytest.mark.parametrize(
    ("change", "gate", "reason"),
    [
        ({}, QualityGate.GREEN, None),
        ({"speech_ratio": 0.02}, QualityGate.RED, "mostly_silence"),
        ({"speech_s": 1.0}, QualityGate.RED, "mostly_silence"),
        ({"speech_ratio": 0.1}, QualityGate.AMBER, "little_speech"),
        ({"clip_ratio": 0.05}, QualityGate.RED, "clipped"),
        ({"clip_ratio": 0.005}, QualityGate.AMBER, "some_clipping"),
        ({"snr_db": -3.0}, QualityGate.RED, "too_noisy"),
        ({"snr_db": 6.0}, QualityGate.AMBER, "noisy"),
        ({"level_gain_db": 40.0}, QualityGate.AMBER, "very_quiet"),
        ({"dnsmos_ovrl": 1.9}, QualityGate.AMBER, "poor_quality"),
        ({"dnsmos_ovrl": None}, QualityGate.GREEN, None),
        ({"echo": 0.4}, QualityGate.AMBER, "echo"),
        ({"echo": 0.1}, QualityGate.GREEN, None),
        ({"rt60_s": 1.1}, QualityGate.AMBER, "reverberant"),
        ({"echo": None, "rt60_s": None}, QualityGate.GREEN, None),
    ],
)
def test_grade_rules(change, gate, reason):
    g, reasons = good(**change)
    assert g == gate
    assert (reason in reasons) if reason else reasons == ()


def test_red_keeps_amber_reasons_too():
    g, reasons = good(clip_ratio=0.05, snr_db=6.0)
    assert g == QualityGate.RED and reasons == ("clipped", "noisy")


def test_snr_tracks_known_noise_level():
    rng = np.random.default_rng(0)
    n = 200
    probs = np.array([0.9] * (n // 2) + [0.05] * (n // 2), dtype=np.float32)
    speech = rng.standard_normal(n // 2 * FRAME) * 0.1
    noise = rng.standard_normal(n * FRAME) * 0.01  # 20 dB below the speech
    x = (np.concatenate([speech, np.zeros(n // 2 * FRAME)]) + noise).astype(np.float32)
    assert snr_db(x, probs) == pytest.approx(20.0, abs=0.5)
    assert snr_db(np.concatenate([speech, np.zeros(n // 2 * FRAME)]).astype(np.float32), probs) == 60.0


# -- the stage, end to end --


def write_wav(path, x):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def probe(tmp_path, x) -> QualityReport:
    src = tmp_path / "in.wav"
    write_wav(src, x)
    store = ArtifactStore(tmp_path / "store")
    rec = store.ingest(src, max_bytes=50 << 20)
    return QualityReport.load(run_pipeline(rec, [Decode(), Probe()], store)["probe"])


def with_noise(x, snr):
    rng = np.random.default_rng(1)
    p = float(np.mean(x[np.abs(x) > 0.01] ** 2))
    return x + rng.standard_normal(len(x)).astype(np.float32) * np.float32(np.sqrt(p / 10 ** (snr / 10)))


def with_echo(x, gain, delay_s):
    k = int(delay_s * SR)
    y = x.copy()
    y[k:] += np.float32(gain) * x[:-k]
    return y


def with_room(x, rt60):
    """Convolve with a synthetic room: direct path plus exponentially decaying noise (-60 dB at rt60)."""
    rng = np.random.default_rng(2)
    t = np.arange(int(rt60 * SR * 1.2)) / SR
    h = rng.standard_normal(len(t)) * np.exp(-6.91 * t / rt60)
    h[0] = 1.0
    y = np.convolve(x, h / np.sqrt(np.sum(h**2)))[: len(x)]
    return (y * np.sqrt(np.mean(x**2) / np.mean(y**2))).astype(np.float32)


@needs_media
def test_clean_speech_is_green(tmp_path, speech):
    r = probe(tmp_path, speech)
    assert r.gate == QualityGate.GREEN, r.reasons
    assert r.speech_ratio > 0.5 and r.snr_db > 30 and r.dnsmos_ovrl >= T.amber_dnsmos_ovrl
    assert r.echo_score < 0.1 and r.rt60_s < 0.5


@needs_media
def test_far_end_echo_is_amber_with_its_delay(tmp_path, speech):
    r = probe(tmp_path, with_echo(speech, 0.5, 0.18))
    assert "echo" in r.reasons and r.gate == QualityGate.AMBER
    assert r.echo_score == pytest.approx(0.5, abs=0.12) and r.echo_delay_ms == pytest.approx(180, abs=2)


@needs_media
def test_reverberant_room_is_amber_and_ordered(tmp_path, speech):
    small = rt60_estimate(with_room(speech, 0.3))
    large = rt60_estimate(with_room(speech, 1.2))
    assert small < 0.5 < T.amber_rt60_s < large
    r = probe(tmp_path, with_room(speech, 1.2))
    assert "reverberant" in r.reasons and "echo" not in r.reasons


@needs_media
def test_10db_noise_is_amber(tmp_path, speech):
    r = probe(tmp_path, with_noise(speech, 10))
    assert r.gate == QualityGate.AMBER and "noisy" in r.reasons
    assert 6 < r.snr_db < 12


@needs_media
def test_drowned_speech_is_red(tmp_path, speech):
    r = probe(tmp_path, with_noise(speech, -5))
    assert r.gate == QualityGate.RED and "too_noisy" in r.reasons


@needs_media
def test_clipped_speech_is_red(tmp_path, speech):
    r = probe(tmp_path, speech * 8)
    assert r.gate == QualityGate.RED and "clipped" in r.reasons


@needs_media
def test_silence_is_red_without_dnsmos(tmp_path):
    r = probe(tmp_path, np.zeros(5 * SR, dtype=np.float32))
    assert r.gate == QualityGate.RED and "mostly_silence" in r.reasons
    assert r.dnsmos_ovrl is None and r.echo_score is None and r.rt60_s is None
