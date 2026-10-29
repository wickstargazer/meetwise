"""Stage 3: quality probe and QualityGate.

Measures, on raw-norm and on each original channel:

    speech_ratio   share of frames silero calls speech (own pass on raw-norm; stage 5 may see enhanced audio)
    speech_s       seconds of speech
    snr_db         speech-frame vs non-speech-frame energy; 60 means a digitally silent background
    clip_ratio     share of samples at or past full scale in the original channels (before any gain)
    level_gain_db  the gain decode needed to reach -20 dBFS; a very quiet recording is mostly noise floor

    dnsmos_*       DNSMOS P.835 SIG/BAK/OVRL (vendored model, models/dnsmos/SOURCE), mean over 9.01 s windows
    echo_score     estimated gain of a discrete echo (far end played back into the room), from the power
                   cepstrum peak at 30-700 ms; echo_delay_ms is where it sits
    rt60_s         blind reverberation time from the fastest energy decays after speech; coarse, reads about
                   0.8x the true RT60 on synthetic rooms

and turns them into GREEN / AMBER / RED with reason codes. RED means ASR is skipped: that is where
hallucination starts. Echo and reverb only ever raise AMBER. A metric that could not be measured is `None`
and the gate ignores it.

Every threshold is provisional until it is calibrated on an evaluation set.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from meetwise.decode import RAW_NORM, DecodeInfo, load_samples
from meetwise.pipeline import StageContext
from meetwise.types import SAMPLE_RATE, QualityGate
from meetwise.vad import FRAME, SileroVad

_FULL_SCALE = 0.98  # resampling to 16 kHz rounds clipped plateaus, so "at full scale" is a band
_SNR_CAP = 60.0
DNSMOS_MODEL = Path(__file__).parent / "models" / "dnsmos" / "sig_bak_ovr.onnx"
DNSMOS_SHA256 = "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd"
_DNSMOS_WIN = int(9.01 * SAMPLE_RATE)
_DNSMOS_MAX_WINDOWS = 60
_CEPS_WIN = 1 << 15  # 2.05 s: longer than the longest echo delay looked for
_ECHO_LAGS = (int(0.03 * SAMPLE_RATE), int(0.7 * SAMPLE_RATE))  # below 30 ms is pitch and early reflections
_ECHO_MIN_Z = 8.0  # cepstral peak over the robust spread of its band; a smaller peak is not an echo
_ENV_HOP, _ENV_WIN = int(0.005 * SAMPLE_RATE), int(0.02 * SAMPLE_RATE)
# Non-personalised raw-to-MOS polynomials from microsoft/DNS-Challenge dnsmos_local.py (highest power first).
_P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
_P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
_P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])


@dataclass(frozen=True, slots=True)
class GateThresholds:
    red_speech_ratio: float = 0.05
    red_speech_s: float = 2.0
    red_clip_ratio: float = 0.01
    red_snr_db: float = 0.0
    amber_speech_ratio: float = 0.2
    amber_clip_ratio: float = 0.001
    amber_snr_db: float = 10.0
    amber_gain_db: float = 30.0
    amber_dnsmos_ovrl: float = 2.2
    amber_echo: float = 0.2
    amber_rt60_s: float = 0.8


@dataclass(frozen=True, slots=True)
class ChannelMetrics:
    channel: int
    clip_ratio: float
    rms_dbfs: float


@dataclass(frozen=True, slots=True)
class QualityReport:
    gate: QualityGate
    reasons: tuple[str, ...]
    speech_ratio: float
    speech_s: float
    snr_db: float
    clip_ratio: float
    level_gain_db: float
    channels: tuple[ChannelMetrics, ...]
    dnsmos_sig: float | None = None
    dnsmos_bak: float | None = None
    dnsmos_ovrl: float | None = None
    echo_score: float | None = None
    echo_delay_ms: float | None = None
    rt60_s: float | None = None

    @classmethod
    def load(cls, stage_dir: Path) -> QualityReport:
        d = json.loads((stage_dir / "probe.json").read_text())
        d["gate"] = QualityGate(d["gate"])
        d["reasons"] = tuple(d["reasons"])
        d["channels"] = tuple(ChannelMetrics(**c) for c in d["channels"])
        return cls(**d)


def grade(
    *,
    speech_ratio: float,
    speech_s: float,
    snr_db: float,
    clip_ratio: float,
    level_gain_db: float,
    t: GateThresholds,
    dnsmos_ovrl: float | None = None,
    echo: float | None = None,
    rt60_s: float | None = None,
) -> tuple[QualityGate, tuple[str, ...]]:
    red, amber = [], []
    if speech_ratio < t.red_speech_ratio or speech_s < t.red_speech_s:
        red.append("mostly_silence")
    elif speech_ratio < t.amber_speech_ratio:
        amber.append("little_speech")
    if clip_ratio > t.red_clip_ratio:
        red.append("clipped")
    elif clip_ratio > t.amber_clip_ratio:
        amber.append("some_clipping")
    if speech_s > 0:  # SNR means nothing without speech
        if snr_db < t.red_snr_db:
            red.append("too_noisy")
        elif snr_db < t.amber_snr_db:
            amber.append("noisy")
    if level_gain_db > t.amber_gain_db:
        amber.append("very_quiet")
    if dnsmos_ovrl is not None and speech_s > 0 and dnsmos_ovrl < t.amber_dnsmos_ovrl:
        amber.append("poor_quality")
    if echo is not None and echo >= t.amber_echo:
        amber.append("echo")
    if rt60_s is not None and rt60_s >= t.amber_rt60_s:
        amber.append("reverberant")
    if red:
        return QualityGate.RED, tuple(red + amber)
    return (QualityGate.AMBER if amber else QualityGate.GREEN), tuple(amber)


def snr_db(x: NDArray[np.float32], probs: NDArray[np.float32]) -> float:
    """Mean speech-frame power over median non-speech power, background removed from the speech side."""
    n = min(len(probs), len(x) // FRAME)
    if n == 0:
        return 0.0
    power = np.square(np.asarray(x[: n * FRAME], dtype=np.float64)).reshape(n, FRAME).mean(axis=1)
    p = probs[:n]
    speech, quiet = power[p >= 0.5], power[p < 0.2]
    if len(speech) == 0:
        return 0.0
    noise = float(np.median(quiet)) if len(quiet) >= 10 else float(np.percentile(power, 10))
    signal = max(float(speech.mean()) - noise, 0.0)
    if noise <= 1e-12:
        return _SNR_CAP
    if signal <= 0.0:
        return -_SNR_CAP
    return max(-_SNR_CAP, min(_SNR_CAP, 10 * math.log10(signal / noise)))


def dnsmos(x: NDArray[np.float32]) -> tuple[float, ...]:
    """(SIG, BAK, OVRL) as in dnsmos_local.py: short audio is tiled to one window; windows hop 1 s, or
    wider so that at most 60 are scored."""
    import onnxruntime as ort

    blob = DNSMOS_MODEL.read_bytes()
    if hashlib.sha256(blob).hexdigest() != DNSMOS_SHA256:
        raise RuntimeError("sig_bak_ovr.onnx does not match the pinned sha256")
    # One window per run, default intra-op threads: batching 8 windows was 4x slower per window on M-series
    # CPUs, and one thread 2.5x slower again (120 s of audio: 122 s -> about 8 s).
    sess = ort.InferenceSession(blob, providers=["CPUExecutionProvider"])

    a = np.asarray(x, dtype=np.float32)
    if len(a) == 0:
        a = np.zeros(_DNSMOS_WIN, dtype=np.float32)
    while len(a) < _DNSMOS_WIN:
        a = np.concatenate([a, a])
    n = int(np.floor(len(a) / SAMPLE_RATE) - 9.01) + 1
    hop = SAMPLE_RATE * max(1, -(-n // _DNSMOS_MAX_WINDOWS))
    starts = [i for i in range(0, len(a) - _DNSMOS_WIN + 1, hop)][:_DNSMOS_MAX_WINDOWS]
    raw = np.concatenate([sess.run(None, {"input_1": a[None, s : s + _DNSMOS_WIN]})[0] for s in starts])
    # The model returns nonsense on digital silence; MOS is defined on 1..5 only.
    return tuple(
        float(np.clip(np.mean(p(raw[:, i])), 1.0, 5.0)) for i, p in enumerate((_P_SIG, _P_BAK, _P_OVR))
    )


def echo_estimate(x: NDArray[np.float32], probs: NDArray[np.float32]) -> tuple[float, float] | None:
    """(gain, delay_ms) of the strongest discrete echo, or None without a speech window or a clear peak.

    x + g*x(t-d) adds about g/2 to the real cepstrum at quefrency d, so the peak doubled estimates g.
    Averaged over 2.05 s windows that are mostly speech; dense reverberation has no single peak."""
    a = np.asarray(x, dtype=np.float64)
    if len(a) < _CEPS_WIN:
        a = np.pad(a, (0, _CEPS_WIN - len(a)))
    taper = np.hanning(_CEPS_WIN)
    ceps = []
    for s in range(0, len(a) - _CEPS_WIN + 1, _CEPS_WIN // 2):
        p = probs[s // FRAME : (s + _CEPS_WIN) // FRAME]
        if len(p) == 0 or float(p.mean()) < 0.3:
            continue
        mag = np.abs(np.fft.rfft(a[s : s + _CEPS_WIN] * taper)) + 1e-9
        ceps.append(np.fft.irfft(np.log(mag)))
    if not ceps:
        return None
    band = np.mean(ceps, axis=0)[_ECHO_LAGS[0] : _ECHO_LAGS[1]]
    k = int(np.argmax(band))
    med = float(np.median(band))
    spread = float(np.median(np.abs(band - med))) * 1.4826
    if spread <= 0 or (band[k] - med) / spread < _ECHO_MIN_Z:
        return 0.0, 0.0
    return float(min(1.0, 2 * band[k])), (_ECHO_LAGS[0] + k) * 1000 / SAMPLE_RATE


def rt60_estimate(x: NDArray[np.float32]) -> float | None:
    """Blind RT60 in seconds from free decays in a 5 ms energy envelope.

    A decay is a run of at least 80 ms that never rises by more than 0.5 dB, starting 20 dB or more above the
    noise floor; its first 20 ms are skipped and it is fitted until 10 dB above the floor. Speech cannot die
    away faster than the room lets it, so the fast decays measure the room: the 30th percentile of the
    slopes is used. None with fewer than 5 decays."""
    a = np.asarray(x, dtype=np.float64)
    n = (len(a) - _ENV_WIN) // _ENV_HOP
    if n < 20:
        return None
    cs = np.concatenate([[0.0], np.cumsum(a * a)])
    starts = np.arange(n) * _ENV_HOP
    db = 10 * np.log10((cs[starts + _ENV_WIN] - cs[starts]) / _ENV_WIN + 1e-12)
    floor = float(np.percentile(db, 5))
    skip, min_len = 4, 16
    t = np.arange(len(db)) * _ENV_HOP / SAMPLE_RATE
    slopes, i = [], 0
    while i < len(db) - 1:
        j = i
        while j + 1 < len(db) and db[j + 1] <= db[j] + 0.5:
            j += 1
        if j - i >= min_len + skip and db[i] - floor > 20:
            seg = db[i + skip : j + 1]
            above = seg >= floor + 10
            stop = len(seg) if above.all() else int(np.argmin(above))
            if stop >= min_len:
                slope = float(np.polyfit(t[:stop], seg[:stop], 1)[0])
                if slope < 0:
                    slopes.append(slope)
        i = j + 1
    if len(slopes) < 5:
        return None
    return -60.0 / float(np.percentile(slopes, 30))


@dataclass
class Probe:
    name: str = "probe"
    version: str = "2"
    thresholds: GateThresholds = field(default_factory=GateThresholds)

    def run(self, ctx: StageContext, out: Path) -> None:
        dec = ctx.outputs["decode"]
        info = DecodeInfo.load(dec)
        x = np.asarray(load_samples(dec, RAW_NORM))
        probs = SileroVad().probs(x)
        speech_frames = int(np.count_nonzero(probs >= 0.5))
        speech_ratio = speech_frames / len(probs) if len(probs) else 0.0
        speech_s = speech_frames * FRAME / SAMPLE_RATE

        channels = []
        for k in range(info.num_channels):
            c = np.asarray(load_samples(dec, f"ch{k}.f32"))
            rms = float(np.sqrt(np.mean(np.square(c, dtype=np.float64)))) if len(c) else 0.0
            channels.append(
                ChannelMetrics(
                    channel=k,
                    clip_ratio=round(float(np.count_nonzero(np.abs(c) >= _FULL_SCALE)) / max(len(c), 1), 6),
                    rms_dbfs=round(20 * math.log10(rms), 2) if rms > 0 else -120.0,
                )
            )
        clip = max((c.clip_ratio for c in channels), default=0.0)
        snr = round(snr_db(x, probs), 2)
        gain_db = round(20 * math.log10(info.gain), 2) if info.gain > 0 else 0.0

        # DNSMOS scores speech; with none it has nothing to score.
        sig, bak, ovrl = (round(v, 3) for v in dnsmos(x)) if speech_s > 0 else (None, None, None)
        echo = echo_estimate(x, probs) if speech_s > 0 else None
        rt60 = rt60_estimate(x) if speech_s > 0 else None

        gate, reasons = grade(
            speech_ratio=speech_ratio,
            speech_s=speech_s,
            snr_db=snr,
            clip_ratio=clip,
            level_gain_db=gain_db,
            t=self.thresholds,
            dnsmos_ovrl=ovrl,
            echo=echo[0] if echo else None,
            rt60_s=rt60,
        )
        report = QualityReport(
            gate=gate,
            reasons=reasons,
            speech_ratio=round(speech_ratio, 4),
            speech_s=round(speech_s, 2),
            snr_db=snr,
            clip_ratio=clip,
            level_gain_db=gain_db,
            channels=tuple(channels),
            dnsmos_sig=sig,
            dnsmos_bak=bak,
            dnsmos_ovrl=ovrl,
            echo_score=round(echo[0], 3) if echo else None,
            echo_delay_ms=round(echo[1], 1) if echo else None,
            rt60_s=round(rt60, 2) if rt60 is not None else None,
        )
        (out / "probe.json").write_text(json.dumps(asdict(report), indent=1))
