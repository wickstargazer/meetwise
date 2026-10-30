# meetwise runbook

How to install, run and look after meetwise on a Linux machine. Everything runs locally; the only network
use is the one-time model download.

## 1. The pipeline

```
 meeting.mp4 / .m4a / .wav
        |
        v
 +-------------------------------------------------+
 | 1 decode     ffmpeg -> 16 kHz mono              |
 | 2 probe      quality gate GREEN / AMBER / RED   |
 | 3 VAD        silero, cuts only in silences      |
 | 4 speakers   pyannote 3.1 (hints, never owners) |
 | 5 language   VoxLingua107, one decision a chunk |
 | 6 ASR        faster-whisper, routed by language |
 |              + anti-hallucination guards        |
 | 7 glossary   fixes the names you list           |
 | 8 summary    Qwen3-8B via llama.cpp,            |
 |              every item cites [mm:ss]           |
 +-------------------------------------------------+
        |
        v
 meeting.mp4.meetwise/
   transcript.md   [00:12] Speaker 1 (en): ...
   summary.md      Decisions / Action items / Key points
   summary.json
```

Every stage result is cached under the recording's sha256 in `$XDG_CACHE_HOME/meetwise` (default
`~/.cache/meetwise`). Running the same file again skips finished stages.

## 2. Speech to text (stage 6)

```
 chunk ──► 5 language ID ──► router ──► 6 ASR model
                                          |
            th ──────────────────────────►│ Thonburian Whisper medium (Mahidol University)
            sv ──────────────────────────►│ KB-Whisper large (National Library of Sweden)
            en, other, or unsure ────────►│ Whisper large-v3 turbo (OpenAI)
                                          v
                              text + word times + confidence
                                          |
                                     guards: repeats, silence,
                                     low confidence -> flagged
```

| | |
|---|---|
| Engine | faster-whisper (CTranslate2) |
| Hardware | NVIDIA CUDA, or any CPU |
| Precision | int8 (float16 activations on GPU) |
| Decoding | greedy, temperature 0, no carry-over between chunks: a rerun gives the same text |
| Code | `src/meetwise/runners/stt_ct2.py`, behind `SttProvider` in `src/meetwise/stt.py` |

The router table is `src/meetwise/models/router.json`. Use `--language xx` to skip detection and send every
chunk to that language's model.

## 3. Install

Needs Python 3.12, [uv](https://docs.astral.sh/uv/), ffmpeg, and about 15 GB of disk.

```
sudo apt install ffmpeg
git clone https://github.com/wickstargazer/meetwise && cd meetwise
uv sync --extra runners

# NVIDIA only: rebuild llama-cpp-python with CUDA so the summary model uses the GPU
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install --reinstall --no-cache llama-cpp-python
```

## 4. Models (once)

pyannote is gated. With your own Hugging Face account, accept the terms of
`pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`, then create a read token.

```
HF_TOKEN=hf_... uv run python scripts/fetch_models.py      # about 12 GB
uv run python scripts/fetch_models.py --verify             # re-hash what is installed
```

Every file is pinned to an upstream commit and sha256 (`src/meetwise/models/manifest.json`); a file that
does not match is not installed. Models go to `$MEETWISE_MODELS`, else `~/.local/share/meetwise/models`.

## 5. Run

```
uv run meetwise run meeting.mp4
uv run meetwise run a.m4a b.mp4                       # several; each gets its own .meetwise/ folder
uv run meetwise run meeting.mp4 --out ~/notes/standup
```

| Option | Use |
|---|---|
| `--language sv` | one known language; skips detection |
| `--summary-language th` | language of the summary (default en) |
| `--glossary terms.json` | `{"en": ["Northwind", "Kubernetes"]}`: names the ASR gets wrong |
| `--device cpu` | no GPU |
| `--gpu-layers N` | summary model layers on the GPU (-1 all, 0 none) |
| `--context N` | summary model context in tokens (default 8192) |
| `--work DIR` | where stage results are cached |

## 6. GPU memory

Each model is freed when its last stage ends, so the stages take turns on the GPU.

```
 stage      model                   GPU memory
 4          pyannote                ~1 GB
 5          VoxLingua               CPU
 6          whisper models (int8)   ~1.5 GB each, up to 3 loaded
 7-8        Qwen3-8B Q4_K_M         ~5 GB + context
```

| GPU | Setting |
|---|---|
| 8 GB or more | defaults |
| 4-6 GB | `--gpu-layers 20` (rest on CPU), or `--language xx` so only one whisper model loads |
| none | `--device cpu`: works, several times slower than real time |

Check with `nvidia-smi`.

## 7. Reading the output

- The quality line in `summary.md` appears only when the gate is not GREEN. AMBER means some items or
  chunks were dropped or flagged; the reasons are listed.
- "Speaker 1, 2, ..." are hints from diarization, numbered by first appearance. They are not names, and an
  action item's owner is only set when the transcript says the name.
- Every `[mm:ss]` points at the moment in the recording the item came from. Check it there.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `... is not installed ... Run fetch_models.py NAME` | run that command |
| `sha256 mismatch` on download | rerun; nothing broken was installed |
| 401 / 403 fetching pyannote | accept its terms on Hugging Face; check `HF_TOKEN` |
| CUDA out of memory | lower `--gpu-layers`, use `--language`, or `--device cpu` |
| summary on CPU although there is a GPU | llama-cpp-python was not rebuilt with CUDA (section 3) |
| wrong language on short chunks | `--language xx` |
| names misspelt | add them to a `--glossary` file |
| want a clean rerun | delete the recording's folder in `~/.cache/meetwise` |

## 9. Privacy

- After the model download nothing leaves the machine: no cloud, no accounts, no telemetry. The package
  imports no network client (checked by `lint-imports`).
- Recordings and results stay in the cache and output folders. Delete them when done:
  `rm -rf ~/.cache/meetwise` and the `.meetwise/` folders.
- Do not commit recordings, transcripts or summaries to this repo. Tests use synthetic speech only.

## 10. Updating

```
git pull && uv sync --extra runners
uv run python scripts/fetch_models.py      # fetches only models that are new or changed
```

## 11. Contributing

- Branch, then open a PR to `main`. `main` needs a PR, one approval and the `test` check.
- Before pushing: `uv run pytest`, `uv run ruff format --check . && uv run ruff check . && uv run lint-imports`.
- Runner tests that need a model skip until it is downloaded.
