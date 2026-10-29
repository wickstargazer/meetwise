"""Download the pinned models and verify every file's sha256.

    uv run python scripts/fetch_models.py                 # all models in the manifest
    uv run python scripts/fetch_models.py NAME [NAME...]  # some
    uv run python scripts/fetch_models.py --verify        # only re-hash what is installed

Lives outside the package on purpose: meetwise imports no network client. Files go to the
directory `meetwise.model_pack.models_root()` returns ($MEETWISE_MODELS, else a per-user data dir). A
file is written as .part and renamed only after its sha256 matches. HF_TOKEN is sent to huggingface.co
when set (gated models). No audio or user data is involved.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request
from pathlib import Path

from meetwise.model_pack import check, load_manifest, models_root


def fetch(url: str, dest: Path, sha256: str, size: int) -> None:
    req = urllib.request.Request(url)
    token = os.environ.get("HF_TOKEN")
    if token and url.startswith("https://huggingface.co/"):
        req.add_header("Authorization", f"Bearer {token}")
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    h, done = hashlib.sha256(), 0
    with urllib.request.urlopen(req) as r, part.open("wb") as out:
        while block := r.read(1 << 20):
            out.write(block)
            h.update(block)
            done += len(block)
            if size > 50 << 20 and done % (100 << 20) < (1 << 20):
                print(f"    {done >> 20} / {size >> 20} MB", flush=True)
    if h.hexdigest() != sha256:
        part.unlink()
        sys.exit(f"sha256 mismatch for {url}; nothing installed")
    part.rename(dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--verify", action="store_true", help="re-hash installed files, download nothing")
    a = ap.parse_args()
    root = a.root or models_root()
    manifest = load_manifest()
    names = a.names or list(manifest)
    if unknown := [n for n in names if n not in manifest]:
        sys.exit(f"not in the manifest: {unknown}")

    need = sum(f.size for n in names for f in manifest[n].files if check_one(manifest[n], f, root))
    free = shutil.disk_usage(root if root.exists() else Path.home()).free
    if not a.verify and need > free - (5 << 30):
        sys.exit(f"needs {need >> 20} MB but only {free >> 20} MB free; keep 5 GB spare")

    bad = 0
    for n in names:
        entry = manifest[n]
        print(f"{n} ({entry.license}, {entry.size >> 20} MB) -> {root / n}")
        if a.verify:
            problems = check(entry, root, full=True)
            bad += bool(problems)
            print("  ok" if not problems else "  " + "; ".join(problems))
            continue
        for f in entry.files:
            if check_one(entry, f, root):
                print(f"  {f.path}")
                fetch(f.url, root / n / f.path, f.sha256, f.size)
        print("  verified")
    sys.exit(1 if bad else 0)


def check_one(entry, f, root: Path) -> bool:
    """True when this file still has to be downloaded (missing, wrong size or wrong sha256)."""
    p = root / entry.name / f.path
    if not p.is_file() or p.stat().st_size != f.size:
        return True
    return hashlib.sha256(p.read_bytes()).hexdigest() != f.sha256 if f.size < 50 << 20 else False


if __name__ == "__main__":
    main()
