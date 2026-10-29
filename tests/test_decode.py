"""Stage 2 on synthetic media made by ffmpeg's lavfi sources. No real recordings."""

import shutil
import subprocess

import numpy as np
import pytest

from meetwise.checkpoint import ArtifactStore
from meetwise.decode import RAW_NORM, Decode, DecodeInfo, RecordingTooLong, load_samples
from meetwise.pipeline import StageFailed, run_pipeline
from meetwise.types import SAMPLE_RATE

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")

CAP = 50 << 20


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def decode(tmp_path, src, **kw):
    store = ArtifactStore(tmp_path / "store")
    rec = store.ingest(src, max_bytes=CAP)
    out = run_pipeline(rec, [Decode(**kw)], store)["decode"]
    return DecodeInfo.load(out), out


def test_tone_onset_lands_on_exact_sample(tmp_path):
    # 48 kHz source: silence, then a 440 Hz tone from exactly 1.000 s.
    src = tmp_path / "onset.wav"
    ffmpeg("-f", "lavfi", "-i", "aevalsrc='if(gte(t,1),0.5*sin(2*PI*440*t),0)':s=48000:d=3", str(src))
    info, out = decode(tmp_path, src)
    x = load_samples(out)
    assert info.num_samples == len(x) == 3 * SAMPLE_RATE
    onset = int(np.argmax(np.abs(x) > 0.05))
    assert abs(onset - SAMPLE_RATE) <= 8  # within 0.5 ms of 1.000 s, resampler ringing included
    assert info.tracks[0].sample_rate == 48000 and info.container == "wav"


def test_identical_stereo_is_not_split(tmp_path):
    src = tmp_path / "dual-mono.m4a"
    ffmpeg("-f", "lavfi", "-i", "sine=f=300:d=2", "-ac", "2", "-c:a", "aac", str(src))
    info, out = decode(tmp_path, src)
    assert info.num_channels == 2 and not info.split_channels
    assert info.correlations[0].correlation > 0.99
    rms = float(np.sqrt(np.mean(np.square(load_samples(out), dtype=np.float64))))
    assert 0.08 < rms < 0.11  # gained to about -20 dBFS


def test_mic_left_meeting_right_is_split(tmp_path):
    # Different sources per channel, as a recorder that keeps the local mic separate would write.
    src = tmp_path / "split.wav"
    ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "sine=f=220:d=2",
        "-f",
        "lavfi",
        "-i",
        "anoisesrc=d=2:c=pink:a=0.3",
        "-filter_complex",
        "[0][1]amerge=inputs=2",
        str(src),
    )
    info, out = decode(tmp_path, src)
    assert info.split_channels
    left, right = load_samples(out, "ch0.f32"), load_samples(out, "ch1.f32")
    assert len(left) == len(right) == info.num_samples


def test_silent_channel_is_reported_and_not_correlated(tmp_path):
    src = tmp_path / "one-side.wav"
    ffmpeg("-f", "lavfi", "-i", "sine=f=500:d=1", "-af", "pan=stereo|c0=c0|c1=0*c0", str(src))
    info, _ = decode(tmp_path, src)
    assert info.silent_channels == (1,)
    assert info.correlations == () and not info.split_channels


def test_video_is_accepted_and_only_audio_decoded(tmp_path):
    src = tmp_path / "meeting.mp4"
    ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "testsrc2=s=320x240:r=10:d=2",
        "-f",
        "lavfi",
        "-i",
        "sine=f=400:d=2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(src),
    )
    info, out = decode(tmp_path, src)
    assert info.has_video and "mp4" in info.container
    assert sorted(p.name for p in out.iterdir()) == ["_DONE", "ch0.f32", "decode.json", RAW_NORM]


def test_two_tracks_are_mixed_with_start_offset(tmp_path):
    a, b, src = tmp_path / "a.wav", tmp_path / "b.wav", tmp_path / "two.mkv"
    ffmpeg("-f", "lavfi", "-i", "sine=f=300:d=2", str(a))
    ffmpeg("-f", "lavfi", "-i", "sine=f=700:d=1", str(b))
    ffmpeg(
        "-i",
        str(a),
        "-itsoffset",
        "0.5",
        "-i",
        str(b),
        "-map",
        "0:a",
        "-map",
        "1:a",
        "-c:a",
        "flac",
        str(src),
    )
    info, out = decode(tmp_path, src)
    assert [t.first_channel for t in info.tracks] == [0, 1]
    assert abs(info.tracks[1].offset_samples - SAMPLE_RATE // 2) <= 16
    late = load_samples(out, "ch1.f32")
    assert not np.any(late[: info.tracks[1].offset_samples]) and np.any(late[SAMPLE_RATE:])


def test_too_long_is_rejected_before_decode(tmp_path):
    src = tmp_path / "long.wav"
    ffmpeg("-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono", "-t", "3", str(src))
    with pytest.raises(StageFailed) as e:
        decode(tmp_path, src, max_duration_s=2)
    assert isinstance(e.value.__cause__, RecordingTooLong)


def test_no_audio_fails(tmp_path):
    src = tmp_path / "silent-film.mp4"
    ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=s=160x120:r=5:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)
    )
    with pytest.raises(StageFailed):
        decode(tmp_path, src)
