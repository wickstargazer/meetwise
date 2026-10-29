"""Stage 2: decode once.

`ffprobe` reads the container, then `ffmpeg` turns every audio track into 16 kHz float32 at its own channel
count. The stage writes, into its checkpoint dir:

    raw-norm.f32       mono mix of every channel, one linear gain applied, nothing else (the ASR path)
    ch<k>.f32          each original channel at 16 kHz, unnormalised, aligned to raw-norm
    decode.json        container, duration, tracks, gain, pairwise channel correlation, split_channels

All .f32 files are little-endian float32 with the same length, so a sample offset means the same instant in
every file. Track start times from ffprobe are applied as leading silence. Video tracks are never decoded.

Only the fields below are kept from ffprobe: container tags (title, author, comments) can name people.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from meetwise.pipeline import StageContext
from meetwise.types import SAMPLE_RATE

RAW_NORM = "raw-norm.f32"
_DTYPE = np.dtype("<f4")
_TARGET_RMS = 10 ** (-20 / 20)  # -20 dBFS
_PEAK_CEIL = 0.99
_SILENT_RMS = 1e-4  # -80 dBFS: a channel this quiet carries nothing
# Below this, two channels hold different sources (e.g. local mic left, meeting audio right). Provisional;
# the bake-off clips will set it.
SPLIT_CORRELATION = 0.5


class DecodeError(RuntimeError):
    pass


class RecordingTooLong(DecodeError):
    pass


@dataclass(frozen=True, slots=True)
class AudioTrack:
    stream_index: int
    codec: str
    channels: int
    sample_rate: int
    offset_samples: int  # leading silence from the stream's start time
    first_channel: int  # this track's channels are ch<first_channel>..


@dataclass(frozen=True, slots=True)
class ChannelPair:
    a: int
    b: int
    correlation: float


@dataclass(frozen=True, slots=True)
class DecodeInfo:
    container: str
    duration_s: float
    has_video: bool
    num_samples: int
    gain: float
    tracks: tuple[AudioTrack, ...]
    silent_channels: tuple[int, ...]
    correlations: tuple[ChannelPair, ...]
    split_channels: bool  # probe and VAD should score each channel too

    @property
    def num_channels(self) -> int:
        return sum(t.channels for t in self.tracks)

    @classmethod
    def load(cls, stage_dir: Path) -> DecodeInfo:
        d = json.loads((stage_dir / "decode.json").read_text())
        d["tracks"] = tuple(AudioTrack(**t) for t in d["tracks"])
        d["silent_channels"] = tuple(d["silent_channels"])
        d["correlations"] = tuple(ChannelPair(**p) for p in d["correlations"])
        return cls(**d)


def load_samples(stage_dir: Path, name: str = RAW_NORM) -> NDArray[np.float32]:
    """Memory-mapped samples of one decode output; `name` is RAW_NORM or `ch<k>.f32`."""
    return np.memmap(stage_dir / name, dtype=_DTYPE, mode="r")


@dataclass
class Decode:
    name: str = "decode"
    version: str = "1"
    max_duration_s: float | None = None  # None: no limit
    ffmpeg: str | None = None
    ffprobe: str | None = None

    def run(self, ctx: StageContext, out: Path) -> None:
        src = ctx.original
        probe = _probe(self._tool("ffprobe", self.ffprobe), src)
        fmt = probe.get("format", {})
        streams = probe.get("streams", [])
        audio = [s for s in streams if s.get("codec_type") == "audio" and int(s.get("channels") or 0) > 0]
        if not audio:
            raise DecodeError("no audio stream")
        has_video = any(
            s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
            for s in streams
        )
        duration = _float(fmt.get("duration")) or max(_float(s.get("duration")) for s in audio)
        if self.max_duration_s is not None and duration > self.max_duration_s:
            raise RecordingTooLong(f"{duration:.0f} s > limit {self.max_duration_s:.0f} s")

        ffmpeg = self._tool("ffmpeg", self.ffmpeg)
        t0 = min(_float(s.get("start_time")) for s in audio)
        tracks: list[AudioTrack] = []
        decoded: list[NDArray[np.float32]] = []  # (channels, n) per track
        next_channel = 0
        for s in audio:
            ch = int(s["channels"])
            pcm = _decode_track(ffmpeg, src, int(s["index"]), out / f".track{s['index']}.f32")
            n = len(pcm) // ch
            offset = round((_float(s.get("start_time")) - t0) * SAMPLE_RATE)
            tracks.append(
                AudioTrack(
                    stream_index=int(s["index"]),
                    codec=str(s.get("codec_name", "")),
                    channels=ch,
                    sample_rate=int(s.get("sample_rate") or 0),
                    offset_samples=offset,
                    first_channel=next_channel,
                )
            )
            next_channel += ch
            decoded.append(pcm[: n * ch].reshape(n, ch).T)

        length = max(t.offset_samples + d.shape[1] for t, d in zip(tracks, decoded, strict=True))
        chans = np.zeros((sum(t.channels for t in tracks), length), dtype=np.float32)
        for t, d in zip(tracks, decoded, strict=True):
            chans[
                t.first_channel : t.first_channel + t.channels,
                t.offset_samples : t.offset_samples + d.shape[1],
            ] = d
        del decoded

        rms = np.sqrt(np.mean(np.square(chans, dtype=np.float64), axis=1))
        silent = tuple(int(i) for i in np.flatnonzero(rms < _SILENT_RMS))
        pairs = tuple(
            ChannelPair(a, b, round(_pearson(chans[a], chans[b]), 4))
            for a, b in combinations(range(len(chans)), 2)
            if a not in silent and b not in silent
        )

        mono = chans.mean(axis=0, dtype=np.float64)
        gain = _gain(mono)
        mono *= gain
        mono.astype(_DTYPE).tofile(out / RAW_NORM)
        del mono
        for k, c in enumerate(chans):
            c.astype(_DTYPE, copy=False).tofile(out / f"ch{k}.f32")

        info = DecodeInfo(
            container=str(fmt.get("format_name", "")),
            duration_s=round(duration, 3),
            has_video=has_video,
            num_samples=length,
            gain=round(gain, 6),
            tracks=tuple(tracks),
            silent_channels=silent,
            correlations=pairs,
            split_channels=any(p.correlation < SPLIT_CORRELATION for p in pairs),
        )
        (out / "decode.json").write_text(json.dumps(asdict(info), indent=1))

    @staticmethod
    def _tool(name: str, given: str | None) -> str:
        path = given or shutil.which(name)
        if not path:
            raise DecodeError(f"{name} not found on PATH")
        return path


def _probe(ffprobe: str, src: Path) -> dict:
    r = subprocess.run(
        [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", "--", str(src)],
        capture_output=True,
        check=False,
    )
    if r.returncode != 0:
        raise DecodeError(f"ffprobe exit {r.returncode}")
    return json.loads(r.stdout)


def _decode_track(ffmpeg: str, src: Path, index: int, tmp: Path) -> NDArray[np.float32]:
    """One audio stream to 16 kHz f32le at its own channel count, via a temp file in the stage dir."""
    r = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-map",
            f"0:{index}",
            "-vn",
            "-sn",
            "-dn",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_f32le",
            "-f",
            "f32le",
            "-y",
            str(tmp),
        ],
        capture_output=True,
        check=False,
    )
    if r.returncode != 0:
        # stderr is not kept: it can echo container metadata.
        raise DecodeError(f"ffmpeg exit {r.returncode} on stream {index}")
    pcm = np.fromfile(tmp, dtype=_DTYPE)
    tmp.unlink()
    return pcm


def _gain(x: NDArray[np.float64]) -> float:
    """One linear gain to about -20 dBFS RMS, never past the peak ceiling. Shape is untouched."""
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    rms = float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0
    if peak == 0.0 or rms == 0.0:
        return 1.0
    return min(_TARGET_RMS / rms, _PEAK_CEIL / peak)


def _pearson(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    a64, b64 = a.astype(np.float64) - a.mean(), b.astype(np.float64) - b.mean()
    den = math.sqrt(float(a64 @ a64) * float(b64 @ b64))
    return float(a64 @ b64) / den if den else 0.0


def _float(v: object) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
