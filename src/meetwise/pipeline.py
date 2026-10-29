"""Stage runner: runs stages in order, skips finished checkpoints, resumes after a crash.

A stage's checkpoint key chains its name and version with every upstream key, so bumping a stage's version
re-runs that stage and everything after it, and nothing before it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from meetwise.checkpoint import ArtifactStore
from meetwise.types import MeetingRecording


@dataclass(frozen=True, slots=True)
class StageContext:
    recording: MeetingRecording
    store: ArtifactStore
    outputs: dict[str, Path]  # finished upstream stages by name

    @property
    def original(self) -> Path:
        return self.store.original(self.recording)


class Stage(Protocol):
    name: str
    version: str  # bump when the stage's output would change

    def run(self, ctx: StageContext, out: Path) -> None: ...


@dataclass(frozen=True, slots=True)
class StageEvent:
    stage: str
    status: Literal["skipped", "started", "done", "failed"]


class StageFailed(RuntimeError):
    def __init__(self, stage: str, recording_sha: str) -> None:
        super().__init__(f"stage {stage} failed for {recording_sha[:12]}")
        self.stage = stage


def stage_keys(stages: Sequence[Stage]) -> list[str]:
    keys, chain = [], ""
    for i, s in enumerate(stages):
        chain = hashlib.sha256(f"{chain}|{s.name}@{s.version}".encode()).hexdigest()[:12]
        keys.append(f"{i:02d}-{s.name}-{chain}")
    return keys


def run_pipeline(
    rec: MeetingRecording,
    stages: Sequence[Stage],
    store: ArtifactStore,
    on_event: Callable[[StageEvent], None] = lambda _e: None,
) -> dict[str, Path]:
    names = [s.name for s in stages]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate stage names: {names}")
    outputs: dict[str, Path] = {}
    for stage, key in zip(stages, stage_keys(stages), strict=True):
        if store.is_done(rec.sha256, key):
            on_event(StageEvent(stage.name, "skipped"))
        else:
            on_event(StageEvent(stage.name, "started"))
            try:
                with store.stage(rec.sha256, key) as work:
                    stage.run(StageContext(rec, store, dict(outputs)), work)
            except Exception as e:
                on_event(StageEvent(stage.name, "failed"))
                raise StageFailed(stage.name, rec.sha256) from e
            on_event(StageEvent(stage.name, "done"))
        outputs[stage.name] = store.stage_dir(rec.sha256, key)
    return outputs
