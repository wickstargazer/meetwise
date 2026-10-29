"""Model runners for Linux (and CPU anywhere): the providers behind the core's thin interfaces.

Install with the `runners` extra. Each runner loads its model from the pinned model pack on first use and
reads from disk only. `release()` frees the model, and with it the GPU memory, so a small GPU can run the
stages one after another: whisper, pyannote and the LLM never have to fit at the same time.
"""

from __future__ import annotations

import gc
import sys


def free_gpu() -> None:
    """Collect dropped models and return torch's cached GPU memory, if torch is in use."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
