"""Stage 10: the cited MeetingSummary.

The transcript goes to the LLM as `[mm:ss] text` lines. Every decision, action item and key point must cite
at least one `mm:ss` that falls inside a chunk. A citation is stored as a sample offset inside that chunk. A
reply that does not parse, or has an item with no citation or one that resolves to no chunk, is sent back
once with the problems listed. If the second reply still has problems, the items whose citations all resolve
are kept and the summary is marked AMBER with its reasons; nothing uncited reaches the user.

The output language is the user's choice, independent of the spoken languages. Owners of action items come
from what was said: an owner is kept only if the transcript text contains it, else None ("unassigned").
Diarization labels ride along as speaker hints but are never shown to the LLM and never become owners.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from meetwise.asr import AsrResult
from meetwise.diarize import Diarization
from meetwise.glossary import Correction
from meetwise.llm import LlmProvider, parse_json_object
from meetwise.pipeline import StageContext
from meetwise.types import (
    SAMPLE_RATE,
    MeetingSummary,
    QualityGate,
    SummaryItem,
    Transcript,
    TranscriptChunk,
    mmss,
)

KINDS = ("decisions", "action_items", "key_points")
LANGUAGE_NAMES = {"th": "Thai", "en": "English", "sv": "Swedish", "hi": "Hindi", "pa": "Punjabi"}
_MMSS = re.compile(r"^\[?(\d{1,3}):([0-5]\d)\]?$")

SYSTEM = (
    "You summarise a meeting transcript. Use only what the transcript says; never invent facts, names or "
    "dates. Each line of the transcript starts with its time as [mm:ss]. Every item you write must cite at "
    "least one of those times, copied exactly, for the line or lines it comes from. An action item's owner "
    'is the person the transcript names as doing it, else null. Reply with JSON only: {"decisions": '
    '[{"text": "...", "cite": ["mm:ss"]}], "action_items": [{"text": "...", "owner": null, "cite": '
    '["mm:ss"]}], "key_points": [{"text": "...", "cite": ["mm:ss"]}]}. Use empty lists when there is nothing.'
)


@dataclass(frozen=True, slots=True)
class SummaryResult:
    summary: MeetingSummary
    gate: QualityGate
    reasons: tuple[str, ...] = ()
    attempts: int = 1
    problems: tuple[str, ...] = field(default=(), compare=False)  # of the last reply, for the log

    def dump(self, out: Path) -> None:
        doc = asdict(self)
        doc["gate"] = self.gate.value
        (out / "summary.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))


def render(chunks: Sequence[TranscriptChunk]) -> str:
    return "\n".join(f"[{mmss(c.span.start)}] {c.text}" for c in chunks if c.text)


def resolve(cite: str, transcript: Transcript) -> int | None:
    """A `mm:ss` citation as a sample offset inside the chunk it points at, or None.

    `mm:ss` is truncated to the second, so it resolves to a chunk that covers any part of that second.
    """
    m = _MMSS.match(cite.strip()) if isinstance(cite, str) else None
    if not m:
        return None
    lo = (int(m[1]) * 60 + int(m[2])) * SAMPLE_RATE
    hi = lo + SAMPLE_RATE
    for c in transcript.chunks:
        if c.span.start < hi and lo < c.span.end:
            return max(lo, c.span.start)
    return None


def parse(reply: str, transcript: Transcript) -> tuple[dict[str, list[SummaryItem]], list[str]]:
    """Items whose citations all resolve, and the problems with the rest. Raises ValueError if unreadable."""
    doc = parse_json_object(reply)
    items: dict[str, list[SummaryItem]] = {k: [] for k in KINDS}
    problems = []
    for kind in KINDS:
        entries = doc.get(kind, [])
        if not isinstance(entries, list):
            problems.append(f"{kind} is not a list")
            continue
        for n, e in enumerate(entries):
            text = e.get("text") if isinstance(e, dict) else None
            if not isinstance(text, str) or not text.strip():
                problems.append(f"{kind}[{n}] has no text")
                continue
            cites = e.get("cite") or []
            cites = [cites] if isinstance(cites, str) else cites
            if not cites:
                problems.append(f"{kind}[{n}] has no citation")
                continue
            offsets = [resolve(c, transcript) for c in cites]
            if None in offsets:
                bad = [c for c, o in zip(cites, offsets, strict=True) if o is None]
                problems.append(f"{kind}[{n}] cites {bad}, which match no transcript line")
                continue
            owner = _owner(e.get("owner"), transcript) if kind == "action_items" else None
            items[kind].append(SummaryItem(text.strip(), tuple(dict.fromkeys(offsets)), owner))
    return items, problems


def _owner(owner: object, transcript: Transcript) -> str | None:
    """The owner rule: a name the transcript text contains, else None ("unassigned").

    Speaker labels are never shown to the LLM, and one it returns anyway is dropped: a diarization label alone
    never makes an owner.
    """
    if not isinstance(owner, str) or not (owner := owner.strip()):
        return None
    if owner in {h.label for h in transcript.speaker_hints}:
        return None
    folded = owner.casefold()
    return owner if any(folded in c.text.casefold() for c in transcript.chunks) else None


def summarise(llm: LlmProvider, transcript: Transcript, output_language: str) -> SummaryResult:
    empty = MeetingSummary(output_language)
    if not any(c.text for c in transcript.chunks):
        return SummaryResult(empty, QualityGate.AMBER, ("nothing_to_summarise",), attempts=0)
    lang = LANGUAGE_NAMES.get(output_language, output_language)
    user = f"Write the summary in {lang}.\n\nTranscript:\n{render(transcript.chunks)}"
    items: dict[str, list[SummaryItem]] = {}
    problems: list[str] = []
    for attempt in (1, 2):
        reply = llm.complete(SYSTEM, user)
        try:
            items, problems = parse(reply, transcript)
        except ValueError as e:
            items, problems = {}, [str(e)]
        if not problems:
            return SummaryResult(_summary(output_language, items), QualityGate.GREEN, attempts=attempt)
        if attempt == 1:
            user += (
                "\n\nYour previous reply had these problems:\n- "
                + "\n- ".join(problems)
                + "\nReply again with JSON only, citing only times that appear in the transcript."
            )
    reason = "summary_unparsed" if not items else "uncited_items_dropped"
    return SummaryResult(_summary(output_language, items), QualityGate.AMBER, (reason,), 2, tuple(problems))


def _summary(output_language: str, items: dict[str, list[SummaryItem]]) -> MeetingSummary:
    return MeetingSummary(output_language, *(tuple(items.get(k, ())) for k in KINDS))


@dataclass
class Summarise:
    llm: LlmProvider
    output_language: str = "en"
    name: str = "summary"

    @property
    def version(self) -> str:
        return f"2+{self.llm.name}+{self.output_language}"

    def run(self, ctx: StageContext, out: Path) -> None:
        asr = AsrResult.load(ctx.outputs["asr"])
        chunks = Correction.load_chunks(ctx.outputs["correct"]) if "correct" in ctx.outputs else asr.chunks
        hints = Diarization.load(ctx.outputs["diarize"]).hints if "diarize" in ctx.outputs else ()
        transcript = Transcript(ctx.recording.sha256, asr.gate, chunks, hints)
        summarise(self.llm, transcript, self.output_language).dump(out)
