"""meetwise: an offline meeting-audio pipeline. It imports no network client."""

from meetwise.stt import SttProvider
from meetwise.types import (
    SAMPLE_RATE,
    AudioChunk,
    ChunkTranscript,
    MeetingRecording,
    MeetingSummary,
    QualityGate,
    SampleRange,
    SpeakerHint,
    SummaryItem,
    Transcript,
    TranscriptChunk,
    Word,
    mmss,
)

__all__ = [
    "SAMPLE_RATE",
    "AudioChunk",
    "ChunkTranscript",
    "MeetingRecording",
    "MeetingSummary",
    "QualityGate",
    "SampleRange",
    "SpeakerHint",
    "SttProvider",
    "SummaryItem",
    "Transcript",
    "TranscriptChunk",
    "Word",
    "mmss",
]
