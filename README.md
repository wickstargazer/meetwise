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
| Pipeline core (this package): stages, router, guards, summary checks, cache | done, tested |
| Linux model runners: faster-whisper (CUDA / CPU), pyannote, VoxLingua, llama.cpp | next |
| Linux model set, pinned by sha256 (Whisper turbo, Thai, Swedish in CTranslate2; Qwen3-8B GGUF) | next |
| `meetwise run FILE` CLI and Markdown output | next |

Languages routed to a dedicated model: Thai (Thonburian Whisper), Swedish (KB-Whisper), English (Whisper
turbo). Any other language goes to Whisper turbo.

## Develop

```
uv sync
uv run pytest
uv run ruff check . && uv run lint-imports
```

Tests use synthetic text-to-speech audio only (`say` on macOS, `espeak-ng` on Linux) and need `ffmpeg`.
## License

Apache-2.0 (`LICENSE`). Models and vendored files keep their own licences: see `THIRD_PARTY_NOTICES.md`.
