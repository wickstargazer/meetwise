"""pyannote 3.1 speaker diarization, the Diarizer for stage 6, on CUDA or CPU.

All three parts come from the pinned model pack and are read from disk only: the hub is set offline
before pyannote is imported, and no token is needed after the download. The pipeline is built from the two
loaded models, with the tuned parameters of the pinned speaker-diarization-3.1 config, rather than from the
config's hub ids. Runs on CUDA when available, else MPS, else CPU.

The checkpoints are loaded with torch's safe (weights-only) unpickler, allowing only the four classes they
store besides tensors; arbitrary pickles are never executed. The files are also sha256-pinned by the pack.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from meetwise.model_pack import model_dir
from meetwise.runners import free_gpu
from meetwise.types import SAMPLE_RATE, SampleRange, SpeakerHint

PIPELINE = "pyannote-diarization-3.1"
SEGMENTATION = "pyannote-segmentation-3.0"
EMBEDDING = "wespeaker-voxceleb-resnet34-lm"


class PyannoteDiarizer:
    name = PIPELINE

    def __init__(self, root: Path | None = None, device: str | None = None) -> None:
        self._dirs = {m: model_dir(m, root) for m in (PIPELINE, SEGMENTATION, EMBEDDING)}
        self._device = device
        self._pipeline: Any = None

    def _loaded(self) -> Any:
        if self._pipeline is None:
            os.environ["HF_HUB_OFFLINE"] = "1"
            import torch  # heavy; only when a recording is diarized
            import yaml
            from pyannote.audio import Model
            from pyannote.audio.core.task import Problem, Resolution, Specifications
            from pyannote.audio.pipelines import SpeakerDiarization
            from torch.torch_version import TorchVersion

            cfg = yaml.safe_load((self._dirs[PIPELINE] / "config.yaml").read_text())
            p = cfg["pipeline"]["params"]
            with torch.serialization.safe_globals([TorchVersion, Specifications, Problem, Resolution]):
                seg = Model.from_pretrained(self._dirs[SEGMENTATION] / "pytorch_model.bin")
                emb = Model.from_pretrained(self._dirs[EMBEDDING] / "pytorch_model.bin")
            pipeline = SpeakerDiarization(
                segmentation=seg,
                embedding=emb,
                clustering=p["clustering"],
                embedding_exclude_overlap=p["embedding_exclude_overlap"],
                embedding_batch_size=p["embedding_batch_size"],
                segmentation_batch_size=p["segmentation_batch_size"],
            )
            pipeline.instantiate(cfg["params"])
            device = self._device or (
                "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            )
            self._pipeline = pipeline.to(torch.device(device))
        return self._pipeline

    def release(self) -> None:
        self._pipeline = None
        free_gpu()

    def diarize(self, samples: NDArray[np.float32]) -> Sequence[SpeakerHint]:
        import torch

        waveform = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).unsqueeze(0)
        annotation = self._loaded()({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        hints = []
        for turn, _, label in annotation.itertracks(yield_label=True):
            a, b = round(turn.start * SAMPLE_RATE), round(turn.end * SAMPLE_RATE)
            if b > a >= 0:
                hints.append(SpeakerHint(str(label), SampleRange(a, b)))
        return hints
