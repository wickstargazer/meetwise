import hashlib

import numpy as np
import pytest

from meetwise import (
    SAMPLE_RATE,
    AudioChunk,
    ChunkTranscript,
    MeetingRecording,
    MeetingSummary,
    QualityGate,
    SampleRange,
    SttProvider,
    SummaryItem,
    Transcript,
    TranscriptChunk,
    mmss,
)

SHA = hashlib.sha256(b"synthetic").hexdigest()


def chunk(i: int, start: int, end: int) -> TranscriptChunk:
    return TranscriptChunk(i, SampleRange(start, end), "th", 0.9, "fake", f"chunk {i}")


def test_mmss():
    assert mmss(0) == "00:00"
    assert mmss(SAMPLE_RATE * 125 + 5) == "02:05"


def test_recording_identity_is_sha256():
    MeetingRecording(SHA, 10, "ogg")
    with pytest.raises(ValueError):
        MeetingRecording("meeting.ogg", 10, "ogg")
    with pytest.raises(ValueError):
        MeetingRecording(SHA, 0, "ogg")


@pytest.mark.parametrize(("start", "end"), [(-1, 5), (5, 5), (6, 5)])
def test_bad_sample_range(start, end):
    with pytest.raises(ValueError):
        SampleRange(start, end)


def test_audio_chunk_must_match_span_and_dtype():
    AudioChunk(SHA, 0, SampleRange(0, 4), np.zeros(4, np.float32))
    with pytest.raises(ValueError):
        AudioChunk(SHA, 0, SampleRange(0, 4), np.zeros(3, np.float32))
    with pytest.raises(ValueError):
        AudioChunk(SHA, 0, SampleRange(0, 4), np.zeros(4, np.float64))


def test_transcript_rejects_overlapping_chunks():
    Transcript(SHA, QualityGate.GREEN, (chunk(0, 0, 100), chunk(1, 100, 200)))
    with pytest.raises(ValueError):
        Transcript(SHA, QualityGate.GREEN, (chunk(0, 0, 100), chunk(1, 50, 200)))


def test_summary_items_need_citations_that_resolve():
    with pytest.raises(ValueError):
        SummaryItem("no citation", ())
    t = Transcript(SHA, QualityGate.AMBER, (chunk(0, 0, 100), chunk(1, 200, 300)))
    s = MeetingSummary(
        "en",
        decisions=(SummaryItem("ok", (50,)),),
        action_items=(SummaryItem("in the gap", (150,), owner=None),),
    )
    assert s.unresolved_citations(t) == [150]


class FakeProvider:
    name = "fake"
    languages = frozenset({"th", "en"})
    locality = "local"

    def transcribe(self, chunk, *, language, prompt_terms):
        return ChunkTranscript(self.name, language or "und", f"{len(chunk.samples)} samples")


def test_fake_provider_satisfies_protocol():
    p = FakeProvider()
    assert isinstance(p, SttProvider)
    out = p.transcribe(
        AudioChunk(SHA, 0, SampleRange(0, 8), np.zeros(8, np.float32)), language=None, prompt_terms=()
    )
    assert out.avg_logprob is None
    assert out.text == "8 samples"
