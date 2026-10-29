"""Qwen3-8B (GGUF, Q4_K_M) through llama.cpp in-process, the LlmProvider for stages 9-10.

The weights come from the pinned model pack and are read from disk only; llama.cpp has no network client.
`gpu_layers` sets how many of the 36 layers go to the GPU: all (-1) fits in 8 GB with the default context,
and a smaller number splits the model with the CPU on a 4-6 GB GPU. A CPU-only build of llama-cpp-python
ignores it. Qwen3's thinking mode is switched off with its `/no_think` tag, and decoding is greedy with a
fixed seed, so a rerun gives the same reply.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from meetwise.model_pack import model_dir
from meetwise.runners import free_gpu

MODEL = "qwen3-8b-q4km-gguf"
FILE = "Qwen3-8B-Q4_K_M.gguf"


class LlamaCpp:
    locality = "local"

    def __init__(
        self,
        model: str = MODEL,
        root: Path | None = None,
        *,
        gpu_layers: int = -1,
        context: int = 8192,
        max_tokens: int = 2048,
        threads: int | None = None,
    ) -> None:
        self.name = model
        self.max_tokens = max_tokens
        self._file = model_dir(model, root) / FILE
        self._options = {"n_gpu_layers": gpu_layers, "n_ctx": context, "n_threads": threads}
        self._llm: Any = None

    def _loaded(self) -> Any:
        if self._llm is None:
            from llama_cpp import Llama  # heavy; only when a prompt is run

            self._llm = Llama(model_path=str(self._file), seed=0, verbose=False, **self._options)
        return self._llm

    def release(self) -> None:
        if self._llm is not None:
            self._llm.close()
        self._llm = None
        free_gpu()

    def complete(self, system: str, user: str) -> str:
        reply = self._loaded().create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"{user} /no_think"},
            ],
            temperature=0.0,
            top_k=1,
            max_tokens=self.max_tokens,
        )
        return reply["choices"][0]["message"]["content"] or ""
