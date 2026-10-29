"""Stage 5. The planner is tested on made-up probabilities; the model on synthetic TTS speech."""

import json
import random
import shutil
import subprocess
from itertools import pairwise

import numpy as np
import pytest

from meetwise.checkpoint import ArtifactStore
from meetwise.decode import Decode
from meetwise.pipeline import run_pipeline
from meetwise.types import SAMPLE_RATE as SR
from meetwise.types import SampleRange
from meetwise.vad import FRAME, ChunkParams, SileroVad, Vad, audio_chunks, plan_chunks, speech_segments

P = ChunkParams()


def probs_for(segments, total, low=0.02, high=0.95):
    pr = np.full(-(-total // FRAME), low, dtype=np.float32)
    for s, e in segments:
        pr[s // FRAME : e // FRAME] = high
    return pr


def secs(*pairs):
    return [(round(a * SR) // FRAME * FRAME, round(b * SR) // FRAME * FRAME) for a, b in pairs]


def check_invariants(chunks, segs, total):
    assert all(c.end - c.start <= P.max_len for c in chunks)
    assert all(a.end <= b.start for a, b in pairwise(chunks))
    assert all(0 <= c.start < c.end <= total for c in chunks)
    assert sum(c.speech_samples for c in chunks) == sum(s.end - s.start for s in segs)
    for c in chunks:  # every chunk edge is in silence unless it is a flagged forced cut
        for s in segs:
            for edge in (c.start, c.end):
                assert c.forced_cut or not s.start < edge < s.end


def test_hysteresis_and_short_runs():
    pr = np.array([0.1, 0.6, 0.45, 0.4, 0.2, 0.1, 0.1, 0.9] + [0.1] * 8, dtype=np.float32)
    segs = speech_segments(pr, len(pr) * FRAME, ChunkParams(min_speech=FRAME, min_silence=FRAME))
    # 0.45 and 0.4 stay speech (above 0.35); the single 0.9 frame is kept at min_speech = one frame.
    assert segs == [SampleRange(1 * FRAME, 4 * FRAME), SampleRange(7 * FRAME, 8 * FRAME)]
    assert speech_segments(pr, len(pr) * FRAME, P) == []  # default: both runs under 250 ms


def test_packs_to_target_and_cuts_only_in_silence():
    total = 50 * SR
    spans = secs((1, 9), (10, 18), (19, 27), (28, 36), (37, 45))
    pr = probs_for(spans, total)
    segs = speech_segments(pr, total, P)
    chunks = plan_chunks(segs, pr, total, P)
    assert len(chunks) == 2 and not any(c.forced_cut for c in chunks)
    assert 20 * SR <= chunks[0].end - chunks[0].start <= 30 * SR
    check_invariants(chunks, segs, total)


def test_gap_under_300ms_is_not_a_cut_point():
    total = 40 * SR
    spans = secs((1, 16), (16.2, 31))  # 0.2 s gap: one 30 s island, too long, so it must be forced
    pr = probs_for(spans, total)
    pr[int(16.1 * SR) // FRAME] = 0.01
    segs = speech_segments(pr, total, P)
    chunks = plan_chunks(segs, pr, total, P)
    assert len(chunks) == 2 and chunks[0].forced_cut and chunks[1].forced_cut
    check_invariants(chunks, segs, total)


def test_monologue_is_force_cut_at_quietest_frame():
    total = 75 * SR
    pr = probs_for(secs((0.5, 74.5)), total)
    dip = int(25 * SR) // FRAME
    pr[dip] = 0.4  # still speech, but the least speech-like frame in the window
    segs = speech_segments(pr, total, P)
    chunks = plan_chunks(segs, pr, total, P)
    assert chunks[0].end == dip * FRAME and all(c.forced_cut for c in chunks)
    check_invariants(chunks, segs, total)


def test_silence_gives_no_chunks():
    total = 10 * SR
    pr = probs_for([], total)
    assert plan_chunks(speech_segments(pr, total, P), pr, total, P) == []


@pytest.mark.parametrize("seed", range(30))
def test_random_layouts_keep_invariants(seed):
    rng = random.Random(seed)
    t, spans = rng.uniform(0, 2), []
    while t < 280:
        length = rng.choice([rng.uniform(0.3, 6), rng.uniform(5, 45)])
        spans.append((t, t + length))
        t += length + rng.choice([0.15, 0.25, 0.35, 0.8, 3.0])
    total = int((t + 2) * SR)
    pr = probs_for(secs(*spans), total)
    pr += np.float32(rng.random()) * 0.01  # break argmin ties
    segs = speech_segments(pr, total, P)
    check_invariants(plan_chunks(segs, pr, total, P), segs, total)


# -- the model itself, on synthetic speech --

TTS = shutil.which("say") or shutil.which("espeak-ng")
needs_media = pytest.mark.skipif(
    not (TTS and shutil.which("ffmpeg")), reason="needs ffmpeg and say/espeak-ng"
)


def tts(text, path):
    if TTS.endswith("say"):
        subprocess.run([TTS, "-o", str(path), "--data-format=LEI16@16000", text], check=True)
    else:
        subprocess.run([TTS, "-w", str(path), text], check=True)


def test_model_is_quiet_on_silence_and_noise():
    vad = SileroVad()
    rng = np.random.default_rng(0)
    assert vad.probs(np.zeros(3 * SR, dtype=np.float32)).max() < 0.2
    assert np.median(vad.probs((rng.standard_normal(3 * SR) * 0.01).astype(np.float32))) < 0.2


@needs_media
def test_speech_silence_speech_end_to_end(tmp_path):
    a, b, src = tmp_path / "a.wav", tmp_path / "b.wav", tmp_path / "talk.wav"
    tts("The quarterly report is due on Friday, please send the numbers before noon.", a)
    tts("Okay, I will check the invoices and reply tomorrow morning.", b)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(a),
            "-i",
            str(b),
            "-filter_complex",
            "[0]aresample=16000,apad=pad_dur=3[x];[1]aresample=16000[y];[x][y]concat=n=2:v=0:a=1",
            str(src),
        ],
        check=True,
    )
    store = ArtifactStore(tmp_path / "store")
    rec = store.ingest(src, max_bytes=50 << 20)
    out = run_pipeline(rec, [Decode(), Vad()], store)

    doc = json.loads((out["vad"] / "vad.json").read_text())
    speech = doc["speech"]
    assert len(speech) >= 2
    # Some silence of 2.5 s or more separates the two sentences.
    assert max(b0 - a1 for (_, a1), (b0, _) in pairwise(speech)) >= 2.5 * SR
    chunks = list(audio_chunks(rec.sha256, out["decode"], out["vad"]))
    assert len(chunks) == 1  # under the 20 s target, so both sentences share one chunk
    assert chunks[0].samples.dtype == np.float32
