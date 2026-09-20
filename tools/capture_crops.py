"""Collect and label ball crops, for training a classifier on this table.

Why this exists
---------------
`detect.balls` tells a stripe from a solid by how much white sits in a thin
annulus at the ball's rim, and that measurement has a ceiling which is not a
matter of tuning: a stripe resting with its coloured band square to the camera
shows no white at all, so from one overhead view it *is* a solid. The same
frame also has to separate a black 8 from dark red cloth, and on a real table
the dimmest ball and the brightest cushion have been measured 0.5 apart on a
scale where a clear ball scores 100.

Both are appearance problems, and a small CNN over the ball crop is what the
literature reaches for once thresholds run out (Gao et al. report 98.5% on
stripe/solid this way, against the rim test's 86%). A CNN needs examples, and
this is how they get collected.

What it collects
----------------
One labelled square per ball, cut with `crop_disc` at the same 1.7 r that the
detector's own crops use, so what the model is trained on is exactly what it
will be shown later.

The crops come from the finder rather than from a click, which matters in two
directions. Every crop is one the detector would really produce, including its
centring error. And the finder's mistakes are collectable too: a piece of
cushion that scored like a ball is labelled `none`, which is how the model
learns the thing a contrast threshold cannot express.

The labelling has to be a person
--------------------------------
The kinds are not recoverable from the pixels by any rule this repo has - that
is the whole reason for collecting them. Labelling with the current detector
would teach the model the detector's existing errors, and labelling with the
hosted vision model was measured at 13/16, worse than the OpenCV it would be
teaching. So a human presses a key per ball.

Pose variety is the point
-------------------------
A hundred crops of one rack teach nothing a single frame does not already say.
Rearrange the balls between captures, and deliberately include the cases that
break the rim test: stripes band-on, stripes pole-up, the 8 in shadow, balls
touching. The model can only learn that a band-on stripe is still a stripe if
it is shown band-on stripes that are labelled stripe.

Usage
-----
    python capture_crops.py            # camera, live
    python capture_crops.py FRAME.png  # label a saved frame instead

Keys, per crop: s solid, t stripe, c cue, e eight, x not-a-ball,
u undo the last label, SPACE skip this crop, q save and quit.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# This script lives one level down but drives the pipeline at the repo root,
# so the root goes on the path before the project imports below.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from companion.pool.perception.vision import detector as detect_balls  # noqa: E402
from tools import htn26_cli as cli  # noqa: E402
from companion.pool.perception.vision.calibration import calibrate_from_frames  # noqa: E402
from companion.pool.perception.vision.camera import Camera, CameraConfig, find_camera_index  # noqa: E402
from companion.pool.perception.vision.classify.baseten import crop_disc  # noqa: E402
from companion.pool.perception.vision.surface import set_active_surface  # noqa: E402

# Where the dataset accumulates. One directory per label keeps it readable by
# eye and loadable by anything, and makes a mislabelled run easy to inspect
# and delete without a manifest to keep in step.
DATA_DIR = ROOT / "data/local/crops"

# The labels. `none` is not a ball at all - cushion, pocket liner, shadow -
# and is what lets the trained model replace a contrast threshold rather than
# just sit behind one.
LABELS = {
    ord("s"): "solid",
    ord("t"): "stripe",
    ord("c"): "cue",
    ord("e"): "eight",
    ord("x"): "none",
}

# Saved at this size. Large enough that a number patch survives and a band is
# unambiguous, small enough that a few hundred of them train in seconds on a
# CPU. The crop is square and the ball is centred, so no aspect to preserve.
CROP_PX = 64

# How much of the screen one crop gets while labelling. The crops are ~80 px
# of camera pixels; shown at native size they are too small to judge a band
# on, which is the one judgement this tool exists to collect.
PREVIEW_PX = 320


@dataclass
class Pending:
    """One crop waiting for a person to say what it is."""

    crop: np.ndarray
    source: str
    index: int


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _existing_counts() -> dict[str, int]:
    return {label: len(list((DATA_DIR / label).glob("*.png")))
            if (DATA_DIR / label).is_dir() else 0
            for label in sorted(set(LABELS.values()))}


def _find_crops(frame: np.ndarray) -> list[np.ndarray]:
    """Every candidate the detector would consider on this frame.

    Deliberately taken *before* `_is_a_ball`, so the things the detector gets
    wrong are collectable. Those are the examples that carry information the
    current code does not have.
    """
    grid, _ = calibrate_from_frames([frame])
    radius_mm = detect_balls.ball_radius_mm()
    view = detect_balls.rectify(
        frame, grid.homography,
        px_per_mm=detect_balls.BALL_RADIUS_PX / radius_mm,
        margin_mm=3.0 * radius_mm)
    r = detect_balls.BALL_RADIUS_PX
    field = detect_balls._colour_field(view.image, r)
    region = detect_balls._search_region(view, r)
    return [crop_disc(view.image, cx, cy, r)
            for cx, cy in detect_balls._find_centres(field.delta, region, r)]


def _label_panel(crop: np.ndarray, done: int, total: int,
                 counts: dict[str, int]) -> np.ndarray:
    """The crop, big, with the keys and the running totals under it."""
    big = cv2.resize(crop, (PREVIEW_PX, PREVIEW_PX),
                     interpolation=cv2.INTER_NEAREST)
    panel = np.full((PREVIEW_PX + 96, max(PREVIEW_PX, 430), 3), 32, np.uint8)
    panel[:PREVIEW_PX, :PREVIEW_PX] = big
    lines = [
        f"crop {done + 1} of {total}",
        "s solid   t stripe   c cue   e eight   x not-a-ball",
        "u undo    SPACE skip    q save and quit",
        "  ".join(f"{k}:{v}" for k, v in sorted(counts.items())),
    ]
    for i, text in enumerate(lines):
        cv2.putText(panel, text, (8, PREVIEW_PX + 20 + 22 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1,
                    cv2.LINE_AA)
    return panel


def _save(crop: np.ndarray, label: str, source: str, index: int) -> Path:
    out = DATA_DIR / label
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{source}-{index:02d}.png"
    cv2.imwrite(str(path), cv2.resize(crop, (CROP_PX, CROP_PX),
                                      interpolation=cv2.INTER_AREA))
    return path


def _label_batch(pending: list[Pending]) -> int:
    """Walk a frame's crops, one keypress each. Returns how many were saved."""
    window = "label - s/t/c/e solid stripe cue eight, x none, u undo, q quit"
    saved: list[Path] = []
    i = 0
    while i < len(pending):
        item = pending[i]
        cv2.imshow(window, _label_panel(item.crop, i, len(pending),
                                        _existing_counts()))
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("u"):
            if saved:
                saved.pop().unlink(missing_ok=True)
                i -= 1
            continue
        if key == ord(" "):
            i += 1
            continue
        label = LABELS.get(key)
        if label is None:
            continue
        saved.append(_save(item.crop, label, item.source, item.index))
        i += 1
    cv2.destroyWindow(window)
    return len(saved)


def _from_camera() -> int:
    """Live: SPACE grabs the table, then you label what it found."""
    index, reason = find_camera_index()
    print(f"camera: {reason}")
    total = 0
    with Camera(CameraConfig(index=index)) as camera:
        window = "table - SPACE capture and label, q quit"
        while True:
            frame = camera.read()
            shown = frame.copy()
            cv2.putText(shown, "SPACE capture   q quit", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                        cv2.LINE_AA)
            saved_so_far = sum(_existing_counts().values())
            cv2.putText(shown, f"saved so far: {saved_so_far}",
                        (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(window, shown)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key != ord(" "):
                continue

            set_active_surface("table")
            if cli.measure_sheet([frame]) is None:
                print("  could not measure the table in that frame; try again")
                continue
            try:
                crops = _find_crops(frame)
            except Exception as exc:
                print(f"  could not find balls in that frame ({exc})")
                continue
            if not crops:
                print("  no candidates in that frame")
                continue
            source = _timestamp()
            cv2.imwrite(str(DATA_DIR / f"frame-{source}.png"), frame)
            print(f"  {len(crops)} candidates - labelling")
            total += _label_batch([Pending(c, source, i)
                                   for i, c in enumerate(crops)])
            print(f"  running total: {sum(_existing_counts().values())} crops")
    cv2.destroyAllWindows()
    return total


def _from_file(path: Path) -> int:
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"error: cannot read {path}", file=sys.stderr)
        return 0
    set_active_surface("table")
    if cli.measure_sheet([frame]) is None:
        print(f"error: could not measure the table in {path}", file=sys.stderr)
        return 0
    crops = _find_crops(frame)
    print(f"{path.name}: {len(crops)} candidates")
    saved = _label_batch([Pending(c, path.stem, i)
                          for i, c in enumerate(crops)])
    cv2.destroyAllWindows()
    return saved


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    before = sum(_existing_counts().values())
    if len(sys.argv) > 1:
        _from_file(Path(sys.argv[1]))
    else:
        _from_camera()
    counts = _existing_counts()
    total = sum(counts.values())
    print(f"\ncrops: {total} ({total - before} new)")
    for label, n in sorted(counts.items()):
        print(f"  {label:7} {n}")
    (DATA_DIR / "counts.json").write_text(json.dumps(counts, indent=2),
                                          encoding="utf-8")

    # Say what is still missing rather than leaving it to be discovered at
    # training time. A class the model never sees is a class it cannot call,
    # and the rare ones here are exactly the ones worth having.
    thin = [label for label, n in counts.items()
            if label in ("cue", "eight") and n < 20]
    thin += [label for label, n in counts.items()
             if label in ("stripe", "solid", "none") and n < 60]
    if thin:
        print(f"\nstill thin: {', '.join(sorted(set(thin)))} - capture more "
              f"layouts that show them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
