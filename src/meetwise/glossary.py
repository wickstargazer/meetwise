"""Stage 9: glossary correction. The LLM proposes edits; code decides which to apply.

The LLM returns `{"edits": [{"chunk_id", "from", "to", "start"?}]}`. It never rewrites a transcript. An edit
is applied only when all of these hold, and is otherwise rejected with a reason:
- the chunk exists and the chunk was transcribed in a language that has a glossary;
- `from` occurs in the chunk text at `start`, or, with no `start`, occurs exactly once;
- `to` is a term of that language's glossary, and differs from `from`;
- it does not overlap another accepted edit in the same chunk.

Inflections are not guessed: a form the glossary should accept (an English plural, say) is listed as its own
term. A glossary is per language: product names, Thai accounting terms, and the people's names the user gives.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from meetwise.asr import AsrResult
from meetwise.llm import LlmProvider, parse_json_object
from meetwise.pipeline import StageContext
from meetwise.types import SampleRange, TranscriptChunk

Glossary = Mapping[str, Collection[str]]  # language -> terms

SYSTEM = (
    "You correct speech-recognition errors against a glossary. You never rewrite, translate or summarise. "
    "Propose an edit only where a word or phrase in a chunk is a misrecognition of a glossary term. "
    'Reply with JSON only: {"edits": [{"chunk_id": <int>, "from": "<exact text in the chunk>", '
    '"to": "<glossary term>"}]}. Copy "from" exactly as it appears. '
    'Reply {"edits": []} if nothing needs fixing.'
)


@dataclass(frozen=True, slots=True)
class GlossaryEdit:
    chunk_id: int
    start: int  # character offsets into the chunk text
    end: int
    old: str
    new: str


@dataclass(frozen=True, slots=True)
class Rejected:
    proposal: dict
    reason: str


@dataclass(frozen=True, slots=True)
class Correction:
    chunks: tuple[TranscriptChunk, ...]
    applied: tuple[GlossaryEdit, ...]
    rejected: tuple[Rejected, ...]

    def dump(self, out: Path) -> None:
        (out / "correction.json").write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1))

    @staticmethod
    def load_chunks(stage_dir: Path) -> tuple[TranscriptChunk, ...]:
        d = json.loads((stage_dir / "correction.json").read_text())
        return tuple(
            TranscriptChunk(**{**c, "span": SampleRange(**c["span"]), "guard_flags": tuple(c["guard_flags"])})
            for c in d["chunks"]
        )


def prompt(chunks: Sequence[TranscriptChunk], glossary: Glossary) -> str:
    langs = sorted({c.language for c in chunks if glossary.get(c.language)})
    terms = {lang: sorted(glossary[lang]) for lang in langs}
    lines = [
        json.dumps({"chunk_id": c.chunk_id, "language": c.language, "text": c.text}, ensure_ascii=False)
        for c in chunks
        if c.text and c.language in terms
    ]
    return (
        "Glossary by language:\n" + json.dumps(terms, ensure_ascii=False) + "\n\nChunks:\n" + "\n".join(lines)
    )


def check(proposals: Sequence[dict], chunks: Sequence[TranscriptChunk], glossary: Glossary) -> Correction:
    by_id = {c.chunk_id: c for c in chunks}
    accepted: dict[int, list[GlossaryEdit]] = {}
    rejected = []
    for p in proposals:
        reason, edit = _validate(p, by_id, glossary)
        if edit and any(edit.start < e.end and e.start < edit.end for e in accepted.get(edit.chunk_id, ())):
            reason, edit = "overlaps an accepted edit", None
        if edit:
            accepted.setdefault(edit.chunk_id, []).append(edit)
        else:
            rejected.append(Rejected(p, reason))
    out = []
    for c in chunks:
        text = c.text
        for e in sorted(accepted.get(c.chunk_id, ()), key=lambda e: e.start, reverse=True):
            text = text[: e.start] + e.new + text[e.end :]
        out.append(replace(c, text=text) if text != c.text else c)
    applied = tuple(e for c in chunks for e in sorted(accepted.get(c.chunk_id, ()), key=lambda e: e.start))
    return Correction(tuple(out), applied, tuple(rejected))


def _validate(
    p: dict, by_id: Mapping[int, TranscriptChunk], glossary: Glossary
) -> tuple[str, GlossaryEdit | None]:
    cid, old, new, start = p.get("chunk_id"), p.get("from"), p.get("to"), p.get("start")
    if not (isinstance(cid, int) and isinstance(old, str) and isinstance(new, str) and old):
        return "malformed", None
    chunk = by_id.get(cid)
    if chunk is None:
        return "no such chunk", None
    if new not in glossary.get(chunk.language, ()):
        return "target is not a glossary term", None
    if new == old:
        return "no change", None
    if isinstance(start, int) and not isinstance(start, bool):
        if chunk.text[start : start + len(old)] != old:
            return "text not found at start", None
    else:
        if chunk.text.count(old) != 1:
            return "text not found" if old not in chunk.text else "text is ambiguous", None
        start = chunk.text.index(old)
    return "", GlossaryEdit(cid, start, start + len(old), old, new)


def correct(llm: LlmProvider, chunks: Sequence[TranscriptChunk], glossary: Glossary) -> Correction:
    if not any(c.text and glossary.get(c.language) for c in chunks):
        return Correction(tuple(chunks), (), ())
    try:
        proposals = parse_json_object(llm.complete(SYSTEM, prompt(chunks, glossary))).get("edits", [])
    except ValueError:
        proposals = []  # an unreadable reply changes nothing; the transcript stands as transcribed
    if not isinstance(proposals, list):
        proposals = []
    return check([p if isinstance(p, dict) else {"raw": p} for p in proposals], chunks, glossary)


@dataclass
class Correct:
    llm: LlmProvider
    glossary: Glossary
    name: str = "correct"

    @property
    def version(self) -> str:
        terms = json.dumps({k: sorted(v) for k, v in sorted(self.glossary.items())}, ensure_ascii=False)
        return f"1+{self.llm.name}+{hashlib.sha256(terms.encode()).hexdigest()[:12]}"

    def run(self, ctx: StageContext, out: Path) -> None:
        correct(self.llm, AsrResult.load(ctx.outputs["asr"]).chunks, self.glossary).dump(out)
