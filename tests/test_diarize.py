"""Stage 6 with a fake diarizer: hints are cleaned, stored, attached per chunk. No model runtime."""

from types import SimpleNamespace

import numpy as np

from meetwise.decode import RAW_NORM
from meetwise.diarize import Diarization, Diarize, Diarizer, chunk_speaker, clean
from meetwise.pipeline import StageContext
from meetwise.types import SAMPLE_RATE as SR
from meetwise.types import SampleRange, SpeakerHint, TranscriptChunk
from meetwise.vad import ENHANCED


def hint(label, a_s, b_s):
    return SpeakerHint(label, SampleRange(int(a_s * SR), int(b_s * SR)))


class Fake:
    name = "fake-diarizer"

    def __init__(self, hints):
        self.hints = hints
        self.seen = None

    def diarize(self, samples):
        self.seen = samples
        return self.hints


def test_clean_clips_to_the_recording_drops_turns_past_it_and_sorts():
    hints = [hint("B", 5, 12), hint("A", 0, 3), hint("C", 11, 20)]
    assert clean(hints, 10 * SR) == (hint("A", 0, 3), hint("B", 5, 10))


def test_speech_per_label_and_only_large_clusters_are_askable():
    d = Diarization("fake", (hint("A", 0, 6), hint("B", 6, 20), hint("A", 20, 23)))
    assert d.speech() == {"B": 14 * SR, "A": 9 * SR}
    assert d.askable() == ["B", "A"]
    assert Diarization("fake", (hint("A", 0, 7.9),)).askable() == []


def test_chunk_speaker_is_the_majority_label_with_its_share():
    c = TranscriptChunk(0, SampleRange(0, 10 * SR), "en", None, "fake", "x")
    assert chunk_speaker(c, (hint("A", 0, 3), hint("B", 3, 10), hint("A", 9, 12))) == ("B", 0.7)
    assert chunk_speaker(c, (hint("A", 20, 30),)) == (None, 0.0)


def test_stage_reads_enhanced_when_present_and_round_trips(tmp_path):
    dec, enh, out = (tmp_path / n for n in ("decode", "enhance", "out"))
    for d in (dec, enh, out):
        d.mkdir()
    np.zeros(30 * SR, np.float32).tofile(dec / RAW_NORM)
    np.ones(30 * SR, np.float32).tofile(enh / ENHANCED)
    fake = Fake([hint("SPEAKER_01", 12, 40), hint("SPEAKER_00", 1, 11)])
    assert isinstance(fake, Diarizer)
    Diarize(fake).run(
        StageContext(SimpleNamespace(sha256="0" * 64), None, {"decode": dec, "enhance": enh}), out
    )
    assert fake.seen.mean() == 1.0
    assert Diarization.load(out) == Diarization(
        "fake-diarizer", (hint("SPEAKER_00", 1, 11), hint("SPEAKER_01", 12, 30))
    )
