"""Stage 8 guards on made-up provider outputs."""

import pytest

from meetwise.guards import GuardThresholds, compression_ratio, guard, max_ngram_repeats, strip_phantoms
from meetwise.types import ChunkTranscript

T = GuardThresholds()


def ct(text, lp=-0.3, ns=0.05, cr=None):
    return ChunkTranscript("fake", "en", text, avg_logprob=lp, no_speech_prob=ns, compression_ratio=cr)


@pytest.mark.parametrize(
    ("out", "speech_s", "text", "flags"),
    [
        (ct("We ship the invoice fix on Monday."), 5.0, "We ship the invoice fix on Monday.", ()),
        (ct(""), 5.0, "", ()),
        (ct("Thank you.", lp=-1.4, ns=0.9), 5.0, "", ("no_speech",)),
        (ct("We decided.", lp=-0.2, ns=0.9), 5.0, "We decided.", ()),  # sure of the words: kept
        (ct("We decided."), 0.1, "", ("text_in_silence",)),
        (ct("Thanks for watching!"), 5.0, "", ("phantom",)),
        (ct("ซับไทยโดย ทีมงาน"), 5.0, "", ("phantom",)),
        (ct("OK, see you Monday. Thanks for watching."), 5.0, "OK, see you Monday.", ("phantom_removed",)),
        (ct("I think so " * 12), 5.0, ("I think so " * 12).strip(), ("repetitive",)),
        (ct("Maybe the budget.", lp=-1.3), 5.0, "Maybe the budget.", ("low_confidence",)),
        (ct("Short answer.", cr=2.9), 5.0, "Short answer.", ("repetitive",)),
    ],
)
def test_guard_rules(out, speech_s, text, flags):
    assert guard(out, speech_s, T) == (text, flags)


def test_unspaced_thai_loop_is_caught():
    loop = "ภาษีมูลค่าเพิ่ม" * 8
    assert max_ngram_repeats(loop) > T.max_ngram_repeats
    assert guard(ct(loop), 5.0, T)[1] == ("repetitive",)
    assert max_ngram_repeats("ยื่นภาษีมูลค่าเพิ่มภายในวันที่สิบห้า") <= T.max_ngram_repeats


def test_helpers():
    assert compression_ratio("") == 0.0
    assert compression_ratio("na " * 50) > 2.4 > compression_ratio("The invoice backlog grew last month.")
    assert strip_phantoms("Tack för att du tittade!") == ""
    assert strip_phantoms("Ship it. Please subscribe!") == "Ship it."
