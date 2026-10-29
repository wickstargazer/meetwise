"""The manifest pins every downloaded file; local checks catch missing, truncated and altered files."""

import hashlib
import json
import re

import pytest

from meetwise import model_pack
from meetwise.model_pack import ModelMissing, check, load_manifest, model_dir


def test_manifest_pins_every_file():
    entries = load_manifest()
    assert entries
    for e in entries.values():
        assert re.fullmatch(r"[0-9a-f]{40}", e.revision), e.name
        assert e.license and e.attribution, e.name
        for f in e.files:
            assert re.fullmatch(r"[0-9a-f]{64}", f.sha256), f.path
            assert e.revision in f.url and f.url.startswith("https://"), f.path
            assert f.size > 0 and not f.path.startswith("/") and ".." not in f.path


@pytest.fixture
def fake(tmp_path, monkeypatch):
    blob = b"weights" * 100
    manifest = tmp_path / "manifest.json"
    entry = dict(
        name="tiny",
        role="asr",
        languages=["th"],
        stage="7",
        repo="x/tiny",
        source="test",
        revision="0" * 40,
        license="MIT",
        attribution="test",
        files=[
            dict(
                path="w.bin",
                size=len(blob),
                sha256=hashlib.sha256(blob).hexdigest(),
                url="https://x/" + "0" * 40,
            )
        ],
    )
    manifest.write_text(json.dumps({"schema": 1, "models": [entry]}))
    monkeypatch.setattr(model_pack, "MANIFEST", manifest)
    monkeypatch.setattr(model_pack.load_manifest, "__defaults__", (manifest,))
    return tmp_path / "root", blob


def test_check_and_model_dir(fake):
    root, blob = fake
    entry = load_manifest()["tiny"]
    assert check(entry, root) == ["w.bin: missing"]
    with pytest.raises(ModelMissing, match=r"fetch_models\.py tiny"):
        model_dir("tiny", root)

    (root / "tiny").mkdir(parents=True)
    (root / "tiny" / "w.bin").write_bytes(blob[:-1])
    assert "size" in check(entry, root)[0]

    (root / "tiny" / "w.bin").write_bytes(blob[:-1] + b"X")  # right size, altered content
    assert check(entry, root) == []
    assert check(entry, root, full=True) == ["w.bin: sha256 mismatch"]

    (root / "tiny" / "w.bin").write_bytes(blob)
    assert check(entry, root, full=True) == [] and model_dir("tiny", root) == root / "tiny"


def test_models_root_follows_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MEETWISE_MODELS", str(tmp_path))
    assert model_pack.models_root() == tmp_path
