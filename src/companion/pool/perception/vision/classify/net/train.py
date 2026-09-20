"""Train the ball-kind network on the crops collected by `scripts/capture_crops.py`.

    python -m models.train            # train, report, and write the model
    python -m models.train --folds 5  # cross-validate instead, write nothing

Two things about the method are worth stating, because both change the number
that comes out by about ten points and only one of them is the honest choice.

**The split is by frame, never by crop.** Sixteen crops from one layout are
not sixteen independent examples: they share the light, the white balance and
the camera's own noise, and several of them are the *same ball* seen moments
apart. Split those across train and test and the model is scored on balls it
has already been shown, which on this dataset reads about ten points higher
than the truth. Every split here keeps a layout whole.

**The classes are weighted, not resampled.** A pool set has one 8 and one cue
against fourteen object balls, so the 8 is scarce for a reason that will not
go away with more capture. Weighting the loss says "getting the 8 wrong costs
more" without inventing copies of the few examples there are.

Augmentation is rotation and reflection in the plane, plus small changes of
exposure. A ball is a sphere photographed from above: turning the crop is a
different pose of the same ball and is exactly the variation the rim test
cannot cope with, so it is the augmentation that carries the most weight here.
Nothing crops or shifts, because the detector always centres what it hands
over and teaching the model to expect otherwise would waste capacity.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .kinds import INPUT_PX, LABELS, WIDTH, BallNet, model_path

CROPS_DIR = Path(__file__).resolve().parent.parent / "crops"

EPOCHS = 80
BATCH = 32
LR = 2e-3
WEIGHT_DECAY = 1e-4


def _load(crops_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every labelled crop, with the frame it came from as its group."""
    images, labels, groups = [], [], []
    for index, label in enumerate(LABELS):
        for path in sorted((crops_dir / label).glob("*.png")):
            image = cv2.imread(str(path))
            if image is None:
                continue
            images.append(cv2.resize(image, (INPUT_PX, INPUT_PX),
                                     interpolation=cv2.INTER_AREA))
            labels.append(index)
            # "frame-20260919-230251-07.png" -> "frame-20260919-230251"
            groups.append(path.stem.rsplit("-", 1)[0])
    if not images:
        raise SystemExit(f"no crops under {crops_dir} - run capture_crops.py")
    return (np.array(images, np.float32) / 255.0,
            np.array(labels), np.array(groups))


def _augment(batch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A different pose of the same ball, under slightly different light."""
    out = np.empty_like(batch)
    for i, image in enumerate(batch):
        image = np.rot90(image, rng.integers(4))
        if rng.random() < 0.5:
            image = image[:, ::-1]
        image = (image * rng.uniform(0.8, 1.2)
                 + rng.normal(0, 0.03, image.shape))
        out[i] = np.clip(image, 0.0, 1.0)
    return out


def _class_weights(labels: np.ndarray):
    import torch

    counts = np.array([max(1, int((labels == i).sum()))
                       for i in range(len(LABELS))], np.float32)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(LABELS)
    return torch.tensor(weights, dtype=torch.float32)


def _fit(x: np.ndarray, y: np.ndarray, seed: int, epochs: int = EPOCHS):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = BallNet()
    opt = torch.optim.Adam(net.parameters(), LR, weight_decay=WEIGHT_DECAY)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.CrossEntropyLoss(weight=_class_weights(y))
    for _ in range(epochs):
        net.train()
        order = rng.permutation(len(x))
        for start in range(0, len(order), BATCH):
            picked = order[start:start + BATCH]
            xb = torch.from_numpy(
                _augment(x[picked], rng)).permute(0, 3, 1, 2)
            yb = torch.from_numpy(y[picked]).long()
            opt.zero_grad()
            loss_fn(net(xb), yb).backward()
            opt.step()
        schedule.step()
    net.eval()
    return net


def _predict(net, x: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        return net(torch.from_numpy(x).permute(0, 3, 1, 2)).argmax(1).numpy()


def _split(groups: np.ndarray, seed: int, hold_out: float = 0.25):
    """Hold out whole layouts, never crops. See the module docstring."""
    frames = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(frames)
    held = set(frames[:max(1, int(hold_out * len(frames)))])
    test = np.array([g in held for g in groups])
    return ~test, test


def _report(truth: np.ndarray, predicted: np.ndarray) -> None:
    print("\nper class:")
    for i, label in enumerate(LABELS):
        n = int((truth == i).sum())
        if n:
            hit = int(((truth == i) & (predicted == i)).sum())
            print(f"  {label:7} {hit:4d}/{n:<4d} {hit / n * 100:5.1f}%")
    print("\nconfusion (rows = truth):")
    print("           " + " ".join(f"{l[:6]:>6}" for l in LABELS))
    for i, label in enumerate(LABELS):
        row = " ".join(f"{int(((truth == i) & (predicted == j)).sum()):6d}"
                       for j in range(len(LABELS)))
        print(f"  {label:8} {row}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", type=Path, default=CROPS_DIR)
    parser.add_argument("--folds", type=int, default=0,
                        help="cross-validate over this many held-out splits "
                             "and write nothing")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        print("error: torch is needed to train (pip install torch)",
              file=sys.stderr)
        return 2

    x, y, groups = _load(args.crops)
    print(f"{len(x)} crops over {len(set(groups))} layouts")
    for i, label in enumerate(LABELS):
        print(f"  {label:7} {int((y == i).sum())}")

    if args.folds:
        started = time.perf_counter()
        scores, truth, predicted = [], [], []
        for seed in range(args.folds):
            train, test = _split(groups, seed)
            net = _fit(x[train], y[train], seed, args.epochs)
            got = _predict(net, x[test])
            scores.append(float((got == y[test]).mean()))
            truth.append(y[test])
            predicted.append(got)
            print(f"  fold {seed}: {scores[-1] * 100:5.1f}% "
                  f"({test.sum()} crops from "
                  f"{len(set(groups[test]))} unseen layouts)")
        print(f"\nheld-out accuracy {np.mean(scores) * 100:.1f}% "
              f"+/- {np.std(scores) * 100:.1f}  "
              f"({time.perf_counter() - started:.0f}s)")
        _report(np.concatenate(truth), np.concatenate(predicted))
        print("\ncross-validation only; nothing written")
        return 0

    # One honest score from a held-out quarter, then a final fit on
    # everything: the score is what the model is worth, the shipped weights
    # are the best that can be made from all the data there is.
    train, test = _split(groups, 0)
    checked = _fit(x[train], y[train], 0, args.epochs)
    got = _predict(checked, x[test])
    print(f"\nheld-out accuracy {float((got == y[test]).mean()) * 100:.1f}% "
          f"on {test.sum()} crops from "
          f"{len(set(groups[test]))} unseen layouts")
    _report(y[test], got)

    print("\nrefitting on every layout for the shipped weights...")
    final = _fit(x, y, 0, args.epochs)
    out = args.out or model_path()
    import torch

    torch.save({"state": final.state_dict(), "labels": LABELS,
                "width": WIDTH, "input_px": INPUT_PX,
                "crops": int(len(x)), "layouts": int(len(set(groups)))}, out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
