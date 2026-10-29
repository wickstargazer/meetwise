"""The Linux runners: output mapping and LLM calls with fakes, and real models when they are installed.

The model tests skip unless the model is in the pack (scripts/fetch_models.py). Synthetic speech only.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from meetwise.llm import LlmProvider
from meetwise.model_pack import ModelMissing
from meetwise.stt import SttProvider
from meetwise.types import SAMPLE_RATE as SR
from meetwise.types import AudioChunk, SampleRange

pytest.importorskip("faster_whisper")

from meetwise.runners.llm_llama import LlamaCpp
from meetwise.runners.stt_ct2 import TURBO, FasterWhisper, to_transcript


def make_chunk(samples, start=10 * SR):
    return AudioChunk("sha", 3, SampleRange(start, start + len(samples)), samples)


def seg(start, end, text, words=(), logprob=-0.2):
    return SimpleNamespace(
        start=start,
        end=end,
        text=text,
        words=[SimpleNamespace(start=a, end=b, word=w, probability=p) for a, b, w, p in words],
        avg_logprob=logprob,
        no_speech_prob=0.01,
        compression_ratio=1.2,
    )


def test_segments_map_to_the_chunk_sample_axis_with_weighted_means():
    c = make_chunk(np.zeros(4 * SR, np.float32))
    t = to_transcript(
        "p",
        [
            seg(0.0, 1.0, " Hello", [(0.0, 0.5, " Hello", 0.91234)], -0.1),
            seg(1.0, 4.0, " there", (), -0.5),
            seg(2, 3, " "),
        ],
        "en",
        c,
    )
    assert t.text == "Hello there"
    assert t.words[0].text == "Hello" and t.words[0].confidence == 0.912
    assert t.words[0].span == SampleRange(10 * SR, 10 * SR + SR // 2)
    assert t.avg_logprob == pytest.approx((-0.1 * 1 + -0.5 * 3) / 4)


def test_word_spans_stay_inside_the_chunk():
    c = make_chunk(np.zeros(SR, np.float32))
    t = to_transcript("p", [seg(0.0, 1.0, " x", [(0.999, 1.5, " x", 0.5)])], None, c)
    w = t.words[0].span
    assert c.span.start <= w.start < w.end <= c.span.end
    assert t.language == "und"


def test_empty_result_has_no_confidence():
    t = to_transcript("p", [], "sv", make_chunk(np.zeros(SR, np.float32)))
    assert (t.text, t.avg_logprob, t.words) == ("", None, ())


class FakeLlama:
    def __init__(self):
        self.calls, self.closed = [], False

    def create_chat_completion(self, **kw):
        self.calls.append(kw)
        return {"choices": [{"message": {"content": "<think>\n\n</think>\n{}"}}]}

    def close(self):
        self.closed = True


def test_llama_is_greedy_without_thinking_and_releases():
    llm = object.__new__(LlamaCpp)
    llm.name, llm.max_tokens, fake = "q", 64, FakeLlama()
    llm._llm = fake
    assert isinstance(llm, LlmProvider)
    assert llm.complete("sys", "user") == "<think>\n\n</think>\n{}"
    kw = fake.calls[0]
    assert kw["temperature"] == 0.0 and kw["top_k"] == 1 and kw["max_tokens"] == 64
    assert kw["messages"][1]["content"].endswith("/no_think")
    llm.release()
    assert fake.closed and llm._llm is None


def test_a_missing_model_says_how_to_fetch_it(tmp_path):
    with pytest.raises(ModelMissing, match=r"fetch_models\.py"):
        FasterWhisper(TURBO, root=tmp_path)


def installed(factory):
    try:
        return factory()
    except ModelMissing as e:
        pytest.skip(str(e).split(":")[0])


def test_turbo_transcribes_english_on_cpu(speech):
    stt = installed(lambda: FasterWhisper(TURBO, device="cpu"))
    assert isinstance(stt, SttProvider)
    t = stt.transcribe(make_chunk(speech, 0), language="en", prompt_terms=())
    assert "invoice" in t.text.lower() and "support" in t.text.lower()
    assert t.avg_logprob is not None and t.avg_logprob > -0.5
    assert all(0 <= w.span.start < w.span.end <= len(speech) for w in t.words)
    stt.release()
    assert stt._model is None


def test_pyannote_and_voxlingua_on_cpu(speech):
    from meetwise.runners.diarize_pyannote import PyannoteDiarizer
    from meetwise.runners.langid_voxlingua import VoxLingua

    hints = installed(lambda: PyannoteDiarizer(device="cpu")).diarize(speech)
    assert hints and all(h.span.end <= len(speech) for h in hints)
    scores = installed(VoxLingua).scores(speech[: 3 * SR])
    assert max(scores, key=scores.get) == "en"
