"""Stage 7a: language ID per VAD chunk, and the language router.

A language-ID model scores each chunk over all the languages it knows. Only the languages the router is
set up for (`router.json`) count: the scores are renormalised over that set, so a Thai chunk heard as Lao
still goes to the Thai provider, and the mass outside the set is recorded for audit.

A chunk is scored whole and in fixed windows. One score over a whole chunk misses a switch of language inside
it (a Thai sentence then an English one scores as confident Thai), so the windows are what find a mix: two
windows confidently in different languages make the chunk mixed. A chunk is decided when it is not mixed,
and its top language is likely enough and clearly ahead of the runner-up. Otherwise it is uncertain (mixed,
unclear, too short, or out of set), goes to the fallback provider with the language unset, and is flagged.

The table is config, not code: `router.json` maps a language to a provider name, and a language with no row
goes to the fallback with its language forced. Like ASR, language ID reads raw-norm samples only.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from meetwise.pipeline import StageContext
from meetwise.types import SAMPLE_RATE
from meetwise.vad import audio_chunks

ROUTER = Path(__file__).parent / "models" / "router.json"
UNCERTAIN = "language_uncertain"  # guard flag on a chunk the router could not decide


@runtime_checkable
class LanguageId(Protocol):
    name: str

    def scores(self, samples: NDArray[np.float32]) -> Mapping[str, float]:
        """Probabilities by BCP-47 primary tag, over every language the model knows."""
        ...


@dataclass(frozen=True, slots=True)
class Router:
    languages: tuple[str, ...]
    routes: Mapping[str, str]  # language -> provider name
    fallback: str
    min_prob: float  # top language, renormalised over `languages`
    min_margin: float  # top minus runner-up
    min_seconds: float  # shorter chunks are uncertain
    window_seconds: float  # windows scored on their own to find a language switch

    @classmethod
    def load(cls, path: Path = ROUTER) -> Router:
        d = json.loads(path.read_text())
        return cls(
            tuple(d["languages"]),
            dict(d["routes"]),
            d["fallback"],
            d["min_prob"],
            d["min_margin"],
            d["min_seconds"],
            d["window_seconds"],
        )

    @property
    def version(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


@dataclass(frozen=True, slots=True)
class LanguageDecision:
    chunk_id: int
    language: str | None  # None: uncertain
    prob: float  # of the top in-set language, renormalised
    top: tuple[tuple[str, float], ...]  # in-set, renormalised, best first
    out_of_set: float  # raw mass on languages the router is not set up for
    reason: str | None = None  # why it is uncertain
    windows: tuple[str | None, ...] = ()  # each window's confident language, None where unclear


def _in_set(scores: Mapping[str, float], r: Router) -> tuple[tuple[tuple[str, float], ...], float, float]:
    """(in-set languages renormalised, best first), in-set mass, out-of-set mass."""
    inset = {lang: max(0.0, scores.get(lang, 0.0)) for lang in r.languages}
    mass = sum(inset.values())
    out_of_set = max(0.0, sum(max(0.0, p) for p in scores.values()) - mass)
    if mass <= 0:
        return (), 0.0, out_of_set
    return tuple(sorted(((k, v / mass) for k, v in inset.items()), key=lambda kv: -kv[1])), mass, out_of_set


def _confident(top: tuple[tuple[str, float], ...], r: Router) -> str | None:
    if not top:
        return None
    second = top[1][1] if len(top) > 1 else 0.0
    return top[0][0] if top[0][1] >= r.min_prob and top[0][1] - second >= r.min_margin else None


def decide(
    chunk_id: int,
    scores: Mapping[str, float],
    seconds: float,
    r: Router,
    windows: Sequence[Mapping[str, float]] = (),
) -> LanguageDecision:
    """`scores` is over the whole chunk, `windows` over its windows in order."""
    top, mass, out_of_set = _in_set(scores, r)
    seen = tuple(_confident(_in_set(w, r)[0], r) for w in windows)
    if not top:
        return LanguageDecision(chunk_id, None, 0.0, (), out_of_set, "out_of_set", seen)
    reason = None
    if seconds < r.min_seconds:
        reason = "too_short"
    elif mass < out_of_set:
        reason = "out_of_set"
    elif len({lang for lang in seen if lang}) > 1:
        reason = "mixed"
    elif _confident(top, r) is None:
        reason = "unclear"
    return LanguageDecision(
        chunk_id, None if reason else top[0][0], top[0][1], top[:3], out_of_set, reason, seen
    )


@dataclass(frozen=True, slots=True)
class LanguageIds:
    model: str
    decisions: tuple[LanguageDecision, ...]

    def by_chunk(self) -> dict[int, LanguageDecision]:
        return {d.chunk_id: d for d in self.decisions}

    def dump(self, out: Path) -> None:
        (out / "langid.json").write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, stage_dir: Path) -> LanguageIds:
        d = json.loads((stage_dir / "langid.json").read_text())
        return cls(
            d["model"],
            tuple(
                LanguageDecision(
                    **{**x, "top": tuple((k, v) for k, v in x["top"]), "windows": tuple(x["windows"])}
                )
                for x in d["decisions"]
            ),
        )


def windows(samples: NDArray[np.float32], r: Router) -> list[NDArray[np.float32]]:
    """Back-to-back windows; a short tail joins the last one. One window or none means nothing to compare."""
    n = round(r.window_seconds * SAMPLE_RATE)
    k = len(samples) // n
    if k < 2:
        return []
    return [samples[i * n : (i + 1) * n if i < k - 1 else len(samples)] for i in range(k)]


@dataclass
class LangId:
    model: LanguageId
    router: Router
    name: str = "langid"

    @property
    def version(self) -> str:
        return f"1+{self.model.name}+{self.router.version}"

    def run(self, ctx: StageContext, out: Path) -> None:
        decisions = []
        for c in audio_chunks(ctx.recording.sha256, ctx.outputs["decode"], ctx.outputs["vad"]):
            ws = [self.model.scores(w) for w in windows(c.samples, self.router)]
            seconds = len(c.samples) / SAMPLE_RATE
            decisions.append(decide(c.chunk_id, self.model.scores(c.samples), seconds, self.router, ws))
        LanguageIds(self.model.name, tuple(decisions)).dump(out)
