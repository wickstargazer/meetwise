"""transcript.md and summary.md from synthetic stage results."""

from meetwise.report import speaker_of, summary_md, transcript_md
from meetwise.types import SAMPLE_RATE as SR
from meetwise.types import SampleRange, SpeakerHint, TranscriptChunk


def chunk(i, a, b, text, lang="en"):
    return TranscriptChunk(i, SampleRange(a * SR, b * SR), lang, 0.9, "p", text)


HINTS = (
    SpeakerHint("SPEAKER_01", SampleRange(0, 4 * SR)),
    SpeakerHint("SPEAKER_00", SampleRange(4 * SR, 10 * SR)),
)


def test_speaker_is_the_label_with_most_overlap_or_none():
    assert speaker_of(chunk(0, 3, 7, "x"), HINTS) == "SPEAKER_00"
    assert speaker_of(chunk(0, 11, 12, "x"), HINTS) is None


def test_transcript_numbers_speakers_by_first_appearance_and_skips_empty_chunks():
    md = transcript_md(
        [chunk(0, 0, 3, "Hello."), chunk(1, 5, 9, "Hej."), chunk(2, 9, 10, ""), chunk(3, 70, 71, "Bye.")],
        HINTS,
    )
    assert md.splitlines()[2:] == [
        "[00:00] Speaker 1 (en): Hello.",
        "[00:05] Speaker 2 (en): Hej.",
        "[01:10] (en): Bye.",
    ]


def test_summary_lists_items_with_citations_owners_and_a_non_green_gate():
    doc = {
        "gate": "amber",
        "reasons": ["low SNR"],
        "summary": {
            "output_language": "en",
            "decisions": [{"text": "Hire two people.", "citations": [65 * SR], "owner": None}],
            "action_items": [{"text": "Post the job.", "citations": [5 * SR, 70 * SR], "owner": None}],
            "key_points": [],
        },
    }
    md = summary_md(doc)
    assert "Quality: AMBER. low SNR" in md
    assert "- Hire two people. [01:05]" in md
    assert "- Post the job. (owner: unassigned) [00:05] [01:10]" in md
    assert md.split("## Key points")[1].strip() == "- none"


def test_green_summary_has_no_quality_line():
    doc = {"gate": "green", "summary": {"output_language": "en"}}
    assert "Quality" not in summary_md(doc)
