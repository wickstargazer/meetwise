"""Stage 8: anti-hallucination guards.

Each chunk's provider output is checked with plain rules. Text that fails a hard rule is dropped. Text that
fails a soft rule is kept and flagged. Nothing is rewritten, except that known phantom phrases are cut out
(deleting text, never inventing it).

Hard (dropped):  no_speech       provider says silence and is unsure of its text (Whisper's own rule)
                 text_in_silence the chunk has less VAD speech than any real utterance needs
                 phantom         the text is only phantom phrases ("Thanks for watching", subtitle credits)
Soft (flagged):  repetitive      compression ratio or a repeated n-gram shows a decoding loop
                 low_confidence  average log probability below the threshold
                 phantom_removed a phantom phrase was cut from otherwise real text
"""

from __future__ import annotations

import re
import zlib
from collections import Counter
from dataclasses import dataclass

from meetwise.types import ChunkTranscript


@dataclass(frozen=True, slots=True)
class GuardThresholds:
    max_compression: float = 2.4  # Whisper's own fallback threshold
    min_avg_logprob: float = -1.0
    max_no_speech: float = 0.6
    min_speech_s: float = 0.3
    max_ngram_repeats: int = 4  # the same 3-gram (words, or 6 characters for unspaced scripts)


# Phrases Whisper-family models emit on silence or music: video sign-offs and subtitle credits from their
# training data. Sign-offs are cut as whole phrases with trailing punctuation; a credit ("subtitles by ...")
# is cut to the end of its line, since the name after it is part of the phantom. Case-insensitive.
PHANTOMS = (
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "ขอบคุณที่รับชม",
    "tack för att du tittade",
    "tack för att ni tittade",
)
CREDITS = (
    "subtitles by",
    "ซับไทยโดย",
    "คำบรรยายโดย",
    "undertexter från",
    "svensktextning.nu",
)
_PHANTOM_RE = re.compile(
    r"\s*(?:" + "|".join(re.escape(p) for p in PHANTOMS) + r")[^\s.!?。]*[\s.!?。…]*"
    r"|\s*(?:" + "|".join(re.escape(c) for c in CREDITS) + r")[^\n]*",
    re.IGNORECASE,
)


def compression_ratio(text: str) -> float:
    b = text.encode()
    return len(b) / len(zlib.compress(b)) if b else 0.0


def max_ngram_repeats(text: str, n: int = 3) -> int:
    words = text.split()
    # Thai has no spaces between words: fall back to character 6-grams.
    grams = (
        [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        if len(words) >= 2 * n
        else [text[i : i + 2 * n] for i in range(len(text) - 2 * n + 1)]
    )
    return max(Counter(grams).values(), default=0)


def strip_phantoms(text: str) -> str:
    return re.sub(r"\s{2,}", " ", _PHANTOM_RE.sub(" ", text)).strip()


def guard(ct: ChunkTranscript, speech_s: float, t: GuardThresholds) -> tuple[str, tuple[str, ...]]:
    """(text to keep, flags). An empty text means the chunk's output was dropped."""
    text = ct.text.strip()
    if not text:
        return "", ()
    lp, ns = ct.avg_logprob, ct.no_speech_prob
    if ns is not None and ns > t.max_no_speech and (lp is None or lp < t.min_avg_logprob):
        return "", ("no_speech",)
    if speech_s < t.min_speech_s:
        return "", ("text_in_silence",)
    flags = []
    kept = strip_phantoms(text)
    if kept != text:
        if not kept:
            return "", ("phantom",)
        flags.append("phantom_removed")
    cr = ct.compression_ratio if ct.compression_ratio is not None else compression_ratio(kept)
    if cr > t.max_compression or max_ngram_repeats(kept) > t.max_ngram_repeats:
        flags.append("repetitive")
    if lp is not None and lp < t.min_avg_logprob:
        flags.append("low_confidence")
    return kept, tuple(flags)
