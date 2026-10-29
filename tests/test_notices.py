"""Every vendored model has its upstream LICENSE, a SOURCE with the pinned sha256, and a notices row."""

import hashlib
import re
from pathlib import Path

import meetwise

MODELS = Path(meetwise.__file__).parent / "models"
NOTICES = Path(__file__).parents[1] / "THIRD_PARTY_NOTICES.md"


def test_each_model_dir_is_licensed_pinned_and_listed():
    notices = NOTICES.read_text()
    dirs = sorted(d for d in MODELS.iterdir() if d.is_dir())
    assert dirs
    for d in dirs:
        assert (d / "LICENSE").is_file(), d.name
        source = (d / "SOURCE").read_text()
        for onnx in d.glob("*.onnx"):
            assert hashlib.sha256(onnx.read_bytes()).hexdigest() in source, onnx.name
            assert re.search(rf"\| `{d.name}` \| `{onnx.name}` \|", notices), onnx.name


def test_each_downloaded_model_is_listed():
    from meetwise.model_pack import load_manifest

    notices = NOTICES.read_text()
    for e in load_manifest().values():
        assert re.search(rf"\| `{e.name}` \| {re.escape(e.repo)} \| {re.escape(e.license)} \|", notices), (
            e.name
        )
