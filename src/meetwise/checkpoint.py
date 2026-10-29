"""Content-addressed artifact store and stage checkpoints.

Layout under the store root (any local directory):

    <sha256>/recording.json          MeetingRecording metadata
    <sha256>/original<suffix>        the received bytes
    <sha256>/stages/<key>/           one finished stage's outputs, plus _DONE
    <sha256>/stages/.tmp-<key>-*/    a stage still running, or one that crashed (removed on retry)

A stage directory appears only by an atomic rename after the stage succeeds, so a crash never leaves a
half-written stage that looks finished.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from meetwise.types import MeetingRecording

_DONE = "_DONE"
_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(_CHUNK):
            h.update(block)
    return h.hexdigest()


class RecordingTooLarge(ValueError):
    pass


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def recording_dir(self, sha: str) -> Path:
        return self.root / sha

    def ingest(self, src: Path, *, max_bytes: int, move: bool = False) -> MeetingRecording:
        """Stage 1: identify `src` by sha256 and keep one copy. Same bytes twice, same recording."""
        size = src.stat().st_size
        if size > max_bytes:
            raise RecordingTooLarge(f"{size} bytes > cap {max_bytes}")
        sha = sha256_file(src)
        rec = MeetingRecording(sha, size, src.suffix.lower())
        d = self.recording_dir(sha)
        if (d / "recording.json").exists():
            if move:
                src.unlink()
            return self.load(sha)
        d.mkdir(parents=True, exist_ok=True)
        dst = d / f"original{rec.suffix}"
        tmp = d / f".tmp-original{rec.suffix}"
        if move:
            shutil.move(src, tmp)
        else:
            shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        _write_json_atomic(d / "recording.json", asdict(rec))
        return rec

    def load(self, sha: str) -> MeetingRecording:
        return MeetingRecording(**json.loads((self.recording_dir(sha) / "recording.json").read_text()))

    def original(self, rec: MeetingRecording) -> Path:
        return self.recording_dir(rec.sha256) / f"original{rec.suffix}"

    # -- stage checkpoints --

    def stage_dir(self, sha: str, key: str) -> Path:
        return self.recording_dir(sha) / "stages" / key

    def is_done(self, sha: str, key: str) -> bool:
        return (self.stage_dir(sha, key) / _DONE).exists()

    @contextmanager
    def stage(self, sha: str, key: str) -> Iterator[Path]:
        """Yield an empty working dir; publish it as the stage's checkpoint only if the block succeeds."""
        final = self.stage_dir(sha, key)
        final.parent.mkdir(parents=True, exist_ok=True)
        for stale in final.parent.glob(f".tmp-{key}-*"):
            shutil.rmtree(stale, ignore_errors=True)
        if final.exists():  # left by a crash between rename and _DONE; never trusted
            shutil.rmtree(final)
        work = Path(tempfile.mkdtemp(prefix=f".tmp-{key}-", dir=final.parent))
        try:
            yield work
            (work / _DONE).touch()
            os.replace(work, final)
        except BaseException:
            shutil.rmtree(work, ignore_errors=True)
            raise


def _write_json_atomic(path: Path, data: object) -> None:
    tmp = path.with_name(f".tmp-{path.name}")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    os.replace(tmp, path)
