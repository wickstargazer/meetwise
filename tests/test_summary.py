"""Stages 9-10 with a scripted fake LLM: edit checks, citations, reprompt, AMBER. No model runtime."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from meetwise.asr import AsrResult
from meetwise.glossary import Correct, check, correct
from meetwise.llm import parse_json_object
from meetwise.pipeline import StageContext
from meetwise.summary import Summarise, render, resolve, summarise
from meetwise.types import SAMPLE_RATE as SR
from meetwise.types import QualityGate, SampleRange, SpeakerHint, Transcript, TranscriptChunk

SHA = hashlib.sha256(b"synthetic").hexdigest()
GLOSSARY = {"en": {"Northwind", "VAT return"}, "th": {"ภ.พ.30", "หัก ณ ที่จ่าย"}}


class Scripted:
    name = "scripted"
    locality = "local"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        r = self.replies.pop(0)
        return r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)


def chunk(i, text, lang="en", start_s=None, dur_s=10):
    a = (start_s if start_s is not None else i * 12) * SR
    return TranscriptChunk(i, SampleRange(a, a + int(dur_s * SR)), lang, None, "fake", text)


CHUNKS = (
    chunk(0, "We moved the books to north wind last week."),
    chunk(1, "The vat return is due Friday, Somchai will file it."),
    chunk(2, "ต้องยื่น ภพ 30 ภายในวันที่ 15", "th"),
)
TRANSCRIPT = Transcript(SHA, QualityGate.GREEN, CHUNKS)


def test_parse_json_object_tolerates_fences_and_reasoning():
    assert parse_json_object('<think>hm {"x":1}</think>\n```json\n{"edits": []}\n```') == {"edits": []}
    with pytest.raises(ValueError):
        parse_json_object("no json here")


def test_edits_apply_only_when_text_matches_and_target_is_a_term():
    c = check(
        [
            {"chunk_id": 0, "from": "north wind", "to": "Northwind"},
            {"chunk_id": 1, "from": "vat return", "to": "VAT return"},
            {"chunk_id": 2, "from": "ภพ 30", "to": "ภ.พ.30"},
            {"chunk_id": 1, "from": "Friday", "to": "Monday"},  # not a glossary term
            {"chunk_id": 0, "from": "north wnd", "to": "Northwind"},  # not in the text
            {"chunk_id": 9, "from": "x", "to": "Northwind"},  # no such chunk
            {"chunk_id": 2, "from": "ภพ 30", "to": "Northwind"},  # en term in a th chunk
            {"chunk_id": "0", "from": "north wind", "to": "Northwind"},  # malformed
        ],
        CHUNKS,
        GLOSSARY,
    )
    assert [x.text for x in c.chunks] == [
        "We moved the books to Northwind last week.",
        "The VAT return is due Friday, Somchai will file it.",
        "ต้องยื่น ภ.พ.30 ภายในวันที่ 15",
    ]
    assert [r.reason for r in c.rejected] == [
        "target is not a glossary term",
        "text not found",
        "no such chunk",
        "target is not a glossary term",
        "malformed",
    ]
    assert [x.span for x in c.chunks] == [x.span for x in CHUNKS]


def test_ambiguous_text_needs_a_start_and_overlaps_are_rejected():
    ch = (chunk(0, "north wind and north wind"),)
    c = check([{"chunk_id": 0, "from": "north wind", "to": "Northwind"}], ch, GLOSSARY)
    assert c.rejected[0].reason == "text is ambiguous" and c.chunks == ch
    c = check(
        [
            {"chunk_id": 0, "from": "north wind", "to": "Northwind", "start": 15},
            {"chunk_id": 0, "from": "north", "to": "Northwind", "start": 15},
            {"chunk_id": 0, "from": "north wind", "to": "Northwind", "start": 3},
        ],
        ch,
        GLOSSARY,
    )
    assert c.chunks[0].text == "north wind and Northwind"
    assert [r.reason for r in c.rejected] == ["overlaps an accepted edit", "text not found at start"]


def test_correct_survives_an_unreadable_reply_and_skips_the_llm_without_a_glossary():
    llm = Scripted("sorry, I cannot help")
    assert correct(llm, CHUNKS, GLOSSARY).chunks == CHUNKS
    llm = Scripted()
    assert correct(llm, CHUNKS, {"sv": {"Fortnox"}}).chunks == CHUNKS and llm.calls == []


def test_citations_resolve_to_a_chunk_covering_that_second():
    assert resolve("00:12", TRANSCRIPT) == 12 * SR
    assert resolve("[00:15]", TRANSCRIPT) == 15 * SR
    assert resolve("00:11", TRANSCRIPT) is None  # gap between chunks 0 and 1
    assert resolve("99:00", TRANSCRIPT) is None and resolve("soon", TRANSCRIPT) is None
    gap = Transcript(SHA, QualityGate.GREEN, (chunk(0, "x", start_s=0, dur_s=1.5),))
    assert resolve("00:01", gap) == SR
    assert render(CHUNKS[:2]).splitlines()[1].startswith("[00:12] The vat")


def test_summary_green_when_every_citation_resolves():
    llm = Scripted(
        {
            "decisions": [{"text": "Books stay in Northwind.", "cite": ["00:00"]}],
            "action_items": [
                {"text": "File the VAT return by Friday.", "owner": "Somchai", "cite": ["00:12"]}
            ],
            "key_points": [{"text": "ภ.พ.30 is due on the 15th.", "cite": ["00:24", "00:24"]}],
        }
    )
    r = summarise(llm, TRANSCRIPT, "en")
    assert r.gate == QualityGate.GREEN and r.attempts == 1 and r.reasons == ()
    s = r.summary
    assert s.action_items[0].owner == "Somchai" and s.key_points[0].citations == (24 * SR,)
    assert s.unresolved_citations(TRANSCRIPT) == []
    assert "Write the summary in English." in llm.calls[0][1] and "[00:24] ต้องยื่น" in llm.calls[0][1]


def test_bad_citation_is_reprompted_once_then_fixed():
    llm = Scripted(
        {"decisions": [{"text": "Move to Northwind.", "cite": ["05:00"]}]},
        {"decisions": [{"text": "Move to Northwind.", "cite": ["00:03"]}]},
    )
    r = summarise(llm, TRANSCRIPT, "th")
    assert r.gate == QualityGate.GREEN and r.attempts == 2
    assert "05:00" in llm.calls[1][1] and "Write the summary in Thai." in llm.calls[1][1]


def test_still_bad_after_reprompt_keeps_only_cited_items_and_is_amber():
    bad = {
        "decisions": [{"text": "Move to Northwind.", "cite": ["00:03"]}],
        "action_items": [{"text": "Hire two people.", "owner": "", "cite": []}],
    }
    r = summarise(Scripted(bad, bad), TRANSCRIPT, "en")
    assert r.gate == QualityGate.AMBER and r.reasons == ("uncited_items_dropped",) and r.attempts == 2
    assert len(r.summary.decisions) == 1 and r.summary.action_items == ()
    r = summarise(Scripted("nope", "still nope"), TRANSCRIPT, "en")
    assert r.gate == QualityGate.AMBER and r.reasons == ("summary_unparsed",)


def test_nothing_to_summarise_does_not_call_the_llm():
    llm = Scripted()
    r = summarise(llm, Transcript(SHA, QualityGate.RED, ()), "en")
    assert r.gate == QualityGate.AMBER and r.reasons == ("nothing_to_summarise",) and llm.calls == []


def test_stage_versions_follow_llm_glossary_and_language():
    a, b = Correct(Scripted(), GLOSSARY), Correct(Scripted(), {**GLOSSARY, "sv": {"Fortnox"}})
    assert a.version != b.version and len(a.version) < 40
    assert Summarise(Scripted(), "th").version != Summarise(Scripted(), "en").version


def test_stages_chain_asr_to_correction_to_summary(tmp_path):
    dirs = {n: tmp_path / n for n in ("asr", "correct", "summary")}
    for d in dirs.values():
        d.mkdir()
    AsrResult("fake", QualityGate.AMBER, ("noisy",), False, CHUNKS).dump(dirs["asr"])
    ctx = StageContext(SimpleNamespace(sha256=SHA), None, {"asr": dirs["asr"]})
    Correct(Scripted({"edits": [{"chunk_id": 0, "from": "north wind", "to": "Northwind"}]}), GLOSSARY).run(
        ctx, dirs["correct"]
    )
    llm = Scripted({"decisions": [{"text": "Books stay in Northwind.", "cite": ["00:00"]}]})
    ctx = StageContext(SimpleNamespace(sha256=SHA), None, {"asr": dirs["asr"], "correct": dirs["correct"]})
    Summarise(llm, "en").run(ctx, dirs["summary"])
    assert "[00:00] We moved the books to Northwind last week." in llm.calls[0][1]
    doc = json.loads((dirs["summary"] / "summary.json").read_text())
    assert doc["gate"] == "green" and doc["summary"]["decisions"][0]["citations"] == [0]


def test_owner_comes_from_transcript_content_never_from_a_speaker_label():
    hints = (
        SpeakerHint("SPEAKER_00", SampleRange(0, 10 * SR)),
        SpeakerHint("SPEAKER_01", SampleRange(12 * SR, 22 * SR)),
    )
    t = Transcript(SHA, QualityGate.GREEN, CHUNKS, hints)
    items = [
        {"text": "File the VAT return.", "owner": "somchai", "cite": ["00:12"]},  # named in the text
        {"text": "Move the books.", "owner": "SPEAKER_00", "cite": ["00:00"]},  # a diarization label
        {"text": "Pay the bill.", "owner": "Anan", "cite": ["00:24"]},  # nobody said Anan
    ]
    llm = Scripted({"action_items": items})
    s = summarise(llm, t, "en").summary
    assert [i.owner for i in s.action_items] == ["somchai", None, None]
    assert "SPEAKER_" not in llm.calls[0][1]  # labels are never shown to the LLM
