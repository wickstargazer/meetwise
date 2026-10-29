"""Shared fixture: synthetic TTS speech (say on macOS, espeak-ng elsewhere). No real recordings."""

import shutil
import subprocess
import wave

import numpy as np
import pytest

from meetwise.types import SAMPLE_RATE as SR

TTS = shutil.which("say") or shutil.which("espeak-ng")
TEXT = (
    "Good morning everyone. First item is the invoice backlog, which grew last month. "
    "Second item is hiring for the support team. We will decide on both next week."
)


@pytest.fixture(scope="module")
def speech(tmp_path_factory):
    if not (TTS and shutil.which("ffmpeg")):
        pytest.skip("needs ffmpeg and say/espeak-ng")
    d = tmp_path_factory.mktemp("tts")
    raw, wav = d / "raw.wav", d / "s.wav"
    if TTS.endswith("say"):
        subprocess.run([TTS, "-o", str(raw), "--data-format=LEI16@16000", TEXT], check=True)
    else:
        subprocess.run([TTS, "-w", str(raw), TEXT], check=True)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(raw),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(wav),
        ],
        check=True,
    )
    with wave.open(str(wav)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
    pad = np.zeros(SR // 2, dtype=np.float32)
    return np.concatenate([pad, x, pad])
