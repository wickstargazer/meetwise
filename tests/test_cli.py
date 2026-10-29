"""`meetwise run` end to end on synthetic speech: real whisper and pyannote, a stand-in LLM.

Skips unless the runners extra and the turbo and pyannote models are installed. The LLM is replaced because
the GGUF is 5 GB; its prompts and parsing are covered by test_summary and test_runners.
"""

import json
import wave

import numpy as np
import pytest

from meetwise.model_pack import ModelMissing
from meetwise.types import SAMPLE_RATE as SR

pytest.importorskip("faster_whisper")

from meetwise import cli
from meetwise.runners import llm_llama

REPLY = {
    "edits": [],
    "decisions": [{"text": "Decide on both items next week.", "cite": ["00:00"]}],
    "action_items": [],
    "key_points": [],
}


class FakeLlm:
    locality = "local"
    name = "fake"
    released = 0

    def __init__(self, **_kw):
        pass

    def complete(self, system, user):
        return json.dumps(REPLY)

    def release(self):
        FakeLlm.released += 1


def test_run_writes_transcript_and_summary(speech, tmp_path, monkeypatch):
    rec = tmp_path / "meeting.wav"
    with wave.open(str(rec), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR)
        w.writeframes((np.clip(speech, -1, 1) * 32767).astype("<i2").tobytes())
    monkeypatch.setattr(llm_llama, "LlamaCpp", FakeLlm)
    args = ["run", str(rec), "--work", str(tmp_path / "work"), "--language", "en", "--device", "cpu"]
    try:
        cli.main(args)
    except ModelMissing as e:
        pytest.skip(str(e).split(":")[0])
    out = tmp_path / "meeting.wav.meetwise"
    transcript = (out / "transcript.md").read_text()
    assert "invoice" in transcript.lower() and "(en):" in transcript
    assert "- Decide on both items next week. [00:00]" in (out / "summary.md").read_text()
    assert FakeLlm.released == 1

    cli.main(args)  # every stage cached: same output, no model loaded again
    assert (out / "transcript.md").read_text() == transcript
    assert FakeLlm.released == 1
