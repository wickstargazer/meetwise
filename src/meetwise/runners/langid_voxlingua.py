"""VoxLingua107 language ID (SpeechBrain ECAPA-TDNN), the LanguageId for stage 7a.

The weights come from the pinned model pack and are read from disk only. The model is built in code
with the shapes of the pinned `hyperparams.yaml`, rather than through SpeechBrain's YAML loader, which would
construct whatever classes the file names. Both checkpoints are plain state dicts and load with torch's
weights-only unpickler. Runs on CPU: the model is small, and CPU keeps the scores the same on every run.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from meetwise.model_pack import model_dir
from meetwise.runners import free_gpu

MODEL = "voxlingua107-ecapa"
_LABEL = re.compile(r"^'([a-z]{2,3}): [^']*' => (\d+)$")


def labels(path: Path) -> list[str]:
    """Index -> language tag, from the SpeechBrain label encoder file."""
    found = {int(m[2]): m[1] for line in path.read_text().splitlines() if (m := _LABEL.match(line))}
    return [found[i] for i in range(len(found))]


class VoxLingua:
    name = MODEL

    def __init__(self, root: Path | None = None) -> None:
        self._path = model_dir(MODEL, root)
        self._model: Any = None

    def _loaded(self) -> Any:
        if self._model is None:
            import torch  # heavy; only when a recording is identified
            from speechbrain.lobes.features import Fbank
            from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
            from speechbrain.lobes.models.Xvector import Classifier
            from speechbrain.processing.features import InputNormalization

            names = labels(self._path / "label_encoder.txt")
            embedding = ECAPA_TDNN(
                60,
                channels=[1024, 1024, 1024, 1024, 3072],
                kernel_sizes=[5, 3, 3, 3, 1],
                dilations=[1, 2, 3, 4, 1],
                attention_channels=128,
                lin_neurons=256,
            )
            classifier = Classifier(
                input_shape=[None, None, 256],
                activation=torch.nn.LeakyReLU,
                lin_blocks=1,
                lin_neurons=512,
                out_neurons=len(names),
            )
            for module, file in ((embedding, "embedding_model.ckpt"), (classifier, "classifier.ckpt")):
                state = torch.load(self._path / file, map_location="cpu", weights_only=True)
                module.load_state_dict(state)
                module.eval()
            features = Fbank(n_mels=60, left_frames=0, right_frames=0, deltas=False)
            norm = InputNormalization(norm_type="sentence", std_norm=False)
            self._model = (torch, features, norm, embedding, classifier, names)
        return self._model

    def release(self) -> None:
        self._model = None
        free_gpu()

    def scores(self, samples: NDArray[np.float32]) -> Mapping[str, float]:
        torch, features, norm, embedding, classifier, names = self._loaded()
        with torch.inference_mode():
            wav = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).unsqueeze(0)
            lens = torch.ones(1)
            emb = embedding(norm(features(wav), lens), lens)
            probs = classifier(emb).squeeze().exp()  # the classifier ends in a log-softmax
        return {lang: float(p) for lang, p in zip(names, probs.tolist(), strict=True)}
