from pathlib import Path

import pytest

from meetwise.checkpoint import ArtifactStore, RecordingTooLarge, sha256_file
from meetwise.pipeline import StageContext, StageFailed, run_pipeline

CAP = 1 << 20


class Step:
    def __init__(self, name: str, version: str = "1", fail_times: int = 0) -> None:
        self.name, self.version, self.fail_times, self.runs = name, version, fail_times, 0

    def run(self, ctx: StageContext, out: Path) -> None:
        self.runs += 1
        prev = [d.joinpath("out.txt").read_text() for d in list(ctx.outputs.values())[-1:]]
        (out / "partial").write_text("x")
        if self.runs <= self.fail_times:
            raise OSError("simulated crash")
        (out / "out.txt").write_text("+".join([*prev, self.name]))


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / "store")


@pytest.fixture
def rec(store, tmp_path):
    src = tmp_path / "meeting.MKV"
    src.write_bytes(b"synthetic recording bytes")
    return store.ingest(src, max_bytes=CAP)


def test_ingest_is_content_addressed(store, tmp_path, rec):
    again = tmp_path / "renamed.mkv"
    again.write_bytes(b"synthetic recording bytes")
    assert store.ingest(again, max_bytes=CAP, move=True) == rec
    assert not again.exists()
    assert rec.suffix == ".mkv"
    assert sha256_file(store.original(rec)) == rec.sha256
    assert store.load(rec.sha256) == rec


def test_ingest_enforces_cap(store, tmp_path):
    big = tmp_path / "big.ogg"
    big.write_bytes(b"x" * 11)
    with pytest.raises(RecordingTooLarge):
        store.ingest(big, max_bytes=10)
    assert big.exists()


def test_resume_after_crash_skips_finished_stages(store, rec):
    a, b, c = Step("decode"), Step("probe", fail_times=1), Step("vad")
    with pytest.raises(StageFailed) as e:
        run_pipeline(rec, [a, b, c], store)
    assert e.value.stage == "probe"
    assert not list((store.recording_dir(rec.sha256) / "stages").glob(".tmp-*"))

    events = []
    out = run_pipeline(rec, [a, b, c], store, on_event=lambda ev: events.append((ev.stage, ev.status)))
    assert (a.runs, b.runs, c.runs) == (1, 2, 1)
    assert events[0] == ("decode", "skipped")
    assert (out["vad"] / "out.txt").read_text() == "decode+probe+vad"


def test_version_bump_reruns_that_stage_and_downstream_only(store, rec):
    a, b, c = Step("decode"), Step("probe"), Step("vad")
    run_pipeline(rec, [a, b, c], store)
    b2, c2 = Step("probe", version="2"), Step("vad")
    run_pipeline(rec, [a, b2, c2], store)
    assert (a.runs, b2.runs, c2.runs) == (1, 1, 1)


def test_unfinished_stage_dir_is_never_trusted(store, rec):
    a = Step("decode")
    run_pipeline(rec, [a], store)
    done = next((store.recording_dir(rec.sha256) / "stages").iterdir())
    (done / "_DONE").unlink()  # as if the process died between rename and marker
    run_pipeline(rec, [a], store)
    assert a.runs == 2


def test_duplicate_stage_names_rejected(store, rec):
    with pytest.raises(ValueError):
        run_pipeline(rec, [Step("x"), Step("x")], store)
