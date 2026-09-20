"""The ball-kind network: architecture, loading, and one batched call.

Kept separate from training so that importing this costs nothing but torch,
and so the shapes the detector depends on - the label order, the input size,
the crop's radius multiple - are defined in exactly one place and read by both
sides. A model file that disagrees with the code that feeds it is silent and
produces plausible nonsense, so those three constants live here and the
checkpoint carries a copy of them to be checked against.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# The order is the network's output order and is part of the file format: a
# checkpoint written under a different order would still load and would still
# produce confident answers, all of them wrong. `load_classifier` refuses a
# checkpoint whose stored labels differ.
LABELS = ("solid", "stripe", "cue", "eight", "none")

# Crops are square, this many pixels on a side, and cut at 1.7 r about the
# centre - the same multiple `classify.baseten.crop_disc` uses, so what the
# model sees at run time is what it was trained on.
INPUT_PX = 48
CROP_R_MULTIPLE = 1.7

# Width of the first convolution block; the rest are 2x and 4x it. 16 was
# measured better than 32 on this dataset (82.4% against 81.4%, and steadier
# across folds), which is the usual shape of a small-data problem: the larger
# model has more ways to memorise 48 layouts.
WIDTH = 16

_CACHE: dict[Path, object] = {}


def model_path() -> Path:
    """Where the trained network lives."""
    return Path(__file__).resolve().parent / "ball_kinds.pt"


def crop_size() -> int:
    return INPUT_PX


def BallNet(width: int = WIDTH, classes: int = len(LABELS)):
    """Four convolutions, pooled to a vector, one linear layer.

    Deliberately small. The whole dataset is a few hundred crops from one
    table, and the thing being learned - where the white sits on a sphere -
    is not deep. Global average pooling rather than a flattened layer keeps
    the parameter count in the tens of thousands and makes the answer depend
    on what is in the crop rather than where in the crop it sits.
    """
    import torch.nn as nn

    c1, c2, c3 = width, width * 2, width * 4
    return nn.Sequential(
        nn.Conv2d(3, c1, 3, 1, 1), nn.BatchNorm2d(c1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(c1, c2, 3, 1, 1), nn.BatchNorm2d(c2), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(c2, c3, 3, 1, 1), nn.BatchNorm2d(c3), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(c3, c3, 3, 1, 1), nn.BatchNorm2d(c3), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.4),
        nn.Linear(c3, classes),
    )


def load_classifier(path: Path | None = None):
    """The trained network, or None if there is not one to load.

    None is a normal answer, not an error: the detector is expected to run
    without a model, and does, on the measurements alone. Torch missing is
    the same case - the repo's own tests run in environments that have no
    torch, and a hard import here would take the whole module down with it.
    """
    path = path or model_path()
    if path in _CACHE:
        return _CACHE[path]
    if not path.is_file():
        return None
    try:
        import torch
    except ImportError:
        return None
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        stored = tuple(blob.get("labels", ()))
        if stored != LABELS:
            raise ValueError(
                f"checkpoint labels {stored} do not match {LABELS}; it was "
                f"trained against a different label order and its answers "
                f"would be silently wrong")
        net = BallNet(blob.get("width", WIDTH))
        net.load_state_dict(blob["state"])
        net.eval()
    except Exception:
        # A corrupt or mismatched checkpoint must not stop the tool working.
        return None
    _CACHE[path] = net
    return net


def predict(net, crops: list[np.ndarray]) -> list[tuple[str, float]]:
    """A label and its probability for each crop, in one batched call.

    Batched because the per-call overhead dominates at this size: sixteen
    crops cost about 6 ms together and far more one at a time, and the whole
    point of doing this locally is that it disappears inside the detection
    that was already happening.
    """
    import torch

    if not crops:
        return []
    batch = np.stack([
        cv2.resize(c, (INPUT_PX, INPUT_PX),
                   interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        for c in crops])
    with torch.no_grad():
        logits = net(torch.from_numpy(batch).permute(0, 3, 1, 2))
        probs = torch.softmax(logits, dim=1).numpy()
    best = probs.argmax(axis=1)
    return [(LABELS[i], float(probs[row, i]))
            for row, i in enumerate(best)]
