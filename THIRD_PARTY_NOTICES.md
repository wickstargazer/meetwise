# Third-party notices

meetwise ships or downloads these third-party files. Each vendored model directory holds the upstream
`LICENSE` and a `SOURCE` file with origin, version and sha256; `tests/test_notices.py` keeps this list, the
directories and the pins in step.

## Vendored models (`src/meetwise/models/`)

| Directory | File | Upstream | License | Attribution |
|---|---|---|---|---|
| `silero_vad` | `silero_vad.onnx` | silero-vad 6.2.3 (PyPI wheel) | MIT | Silero Team |
| `dnsmos` | `sig_bak_ovr.onnx` | microsoft/DNS-Challenge, commit 82f1b17 | CC-BY-4.0 | Reddy et al., "DNSMOS P.835", Microsoft |

CC-BY-4.0 requires the attribution above to travel with any copy of the model.

## Python runtime dependencies

| Package | License |
|---|---|
| numpy | BSD-3-Clause |
| onnxruntime | MIT |

## Downloaded models (`src/meetwise/models/manifest.json`, fetched by `scripts/fetch_models.py`)

| Model | Upstream | License | Attribution |
|---|---|---|---|
| `thonburian-whisper-th-medium` | biodatlab/whisper-th-medium-combined | Apache-2.0 | Thonburian Whisper, biodatlab (Mahidol University) |
| `kb-whisper-large-sv` | KBLab/kb-whisper-large | Apache-2.0 | KB-Whisper, KBLab (National Library of Sweden) |
| `pyannote-diarization-3.1` | pyannote/speaker-diarization-3.1 | MIT | pyannote.audio speaker-diarization-3.1, Hervé Bredin / pyannoteAI |
| `pyannote-segmentation-3.0` | pyannote/segmentation-3.0 | MIT | pyannote.audio segmentation-3.0, Hervé Bredin / pyannoteAI |
| `wespeaker-voxceleb-resnet34-lm` | pyannote/wespeaker-voxceleb-resnet34-LM | CC-BY-4.0 | WeSpeaker ResNet34-LM trained on VoxCeleb, WeNet community; pyannote.audio wrapper by Hervé Bredin |
| `voxlingua107-ecapa` | speechbrain/lang-id-voxlingua107-ecapa | Apache-2.0 | VoxLingua107 ECAPA-TDNN language ID, SpeechBrain (Valk & Alumäe, 2021) |
| `deepfilternet3-onnx` | Rikorose/DeepFilterNet | MIT OR Apache-2.0 | DeepFilterNet3, Hendrik Schröter |

pyannote models are gated on Hugging Face: each user accepts their terms with their own account and passes
a read token to `fetch_models.py` as `HF_TOKEN`.
