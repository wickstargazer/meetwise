"""The downloaded model pack: what the manifest pins and where the files sit on this machine.

models/manifest.json lists every downloaded model with its upstream commit, licence and per-file sha256.
This module only reads and checks local files. Downloading is scripts/fetch_models.py, outside the
package, so the pipeline itself has no network client.

Layout: <root>/<model name>/<file path>. The root is $MEETWISE_MODELS, else
~/Library/Application Support/meetwise/models on macOS and $XDG_DATA_HOME/meetwise/models (default
~/.local/share/meetwise/models) elsewhere.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

MANIFEST = Path(__file__).parent / "models" / "manifest.json"


class ModelMissing(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ModelFile:
    path: str
    size: int
    sha256: str
    url: str


@dataclass(frozen=True, slots=True)
class ModelEntry:
    name: str
    role: str
    languages: tuple[str, ...]
    stage: str
    repo: str
    source: str
    revision: str
    license: str
    attribution: str
    files: tuple[ModelFile, ...]

    @property
    def size(self) -> int:
        return sum(f.size for f in self.files)


def models_root() -> Path:
    env = os.environ.get("MEETWISE_MODELS")
    if env:
        return Path(env)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "meetwise" / "models"
    data = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(data) / "meetwise" / "models"


def load_manifest(path: Path = MANIFEST) -> dict[str, ModelEntry]:
    doc = json.loads(path.read_text())
    entries = {}
    for m in doc["models"]:
        files = tuple(ModelFile(**f) for f in m["files"])
        entries[m["name"]] = ModelEntry(**{**m, "languages": tuple(m["languages"]), "files": files})
    return entries


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def check(entry: ModelEntry, root: Path, *, full: bool = False) -> list[str]:
    """Problems with one model on disk; empty means present. Sizes always, sha256 only when `full`."""
    problems = []
    for f in entry.files:
        p = root / entry.name / f.path
        if not p.is_file():
            problems.append(f"{f.path}: missing")
        elif p.stat().st_size != f.size:
            problems.append(f"{f.path}: size {p.stat().st_size} != {f.size}")
        elif full and file_sha256(p) != f.sha256:
            problems.append(f"{f.path}: sha256 mismatch")
    return problems


def model_dir(name: str, root: Path | None = None) -> Path:
    """Directory of an installed model, or ModelMissing saying how to get it."""
    entry = load_manifest()[name]
    root = root or models_root()
    if problems := check(entry, root):
        raise ModelMissing(
            f"{name} is not installed in {root}: {'; '.join(problems)}. "
            f"Run `uv run python scripts/fetch_models.py {name}`."
        )
    return root / name
