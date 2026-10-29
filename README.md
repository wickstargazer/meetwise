# meetwise

An offline meeting-audio pipeline as a Python library. Give it a recording (video or audio); get a
transcript with speaker hints and a summary in which every decision and action cites a [mm:ss] time.
After the models are downloaded, nothing leaves the machine.

```
recording ─► decode (ffmpeg, 16 kHz mono) ─► quality probe (GREEN / AMBER / RED)
          ─► VAD chunks (silero, cut only in silence) ─► speaker hints (pyannote 3.1)
          ─► language ID per chunk (VoxLingua107) ─► ASR per language, with anti-hallucination guards
          ─► glossary correction (LLM proposes, code accepts exact terms only)
          ─► cited summary (decisions, actions, key points; owners only when the name is said)
```

Every stage result is cached under the recording's sha256, so a rerun only redoes what changed.

## Status

| Part | State |
|---|---|
| Pipeline core: stages, router, guards, summary checks, cache | done, tested |
| Model runners: faster-whisper (CUDA / CPU), pyannote, VoxLingua, llama.cpp | done; tested on CPU, CUDA untested |
| Model set pinned by sha256: Whisper turbo, Thai, Swedish (CTranslate2); Qwen3-8B (GGUF) | done |
| `meetwise run FILE` with Markdown output | done |
| Watch-a-folder mode, Docker images | next |

Languages routed to a dedicated model: Thai (Thonburian Whisper), Swedish (KB-Whisper), English (Whisper
turbo). Any other language goes to Whisper turbo.

## Run it

Needs Python 3.12, [uv](https://docs.astral.sh/uv/) and ffmpeg. An NVIDIA GPU is optional.

```
git clone https://github.com/wickstargazer/meetwise && cd meetwise
uv sync --extra runners
# For the LLM on an NVIDIA GPU, rebuild llama-cpp-python with CUDA:
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install --reinstall --no-cache llama-cpp-python

# Models, about 12 GB. pyannote is gated: accept the terms of pyannote/speaker-diarization-3.1 and
# pyannote/segmentation-3.0 on Hugging Face, then pass a read token.
HF_TOKEN=hf_... uv run python scripts/fetch_models.py

uv run meetwise run meeting.mp4
```

This writes `meeting.mp4.meetwise/transcript.md`, `summary.md` and `summary.json`. Useful options:
`--language sv` skips language detection, `--summary-language th` picks the summary language, `--glossary
terms.json` (`{"en": ["Northwind"]}`) fixes names the ASR gets wrong, and `--device cpu` runs without a GPU.

GPU memory: each model is freed when its last stage ends, so the stages take turns. Whisper in int8 needs
about 1.5 GB per model; Qwen3-8B Q4 needs about 5 GB with all layers on the GPU. On a 4 GB card, use
`--gpu-layers 20` (the rest runs on the CPU) or `--gpu-layers 0`.

## License

Apache-2.0 (`LICENSE`). Models and vendored files keep their own licences: see `THIRD_PARTY_NOTICES.md`.
