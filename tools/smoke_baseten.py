"""Cheapest possible Baseten check: one 64 px JPEG of the known 8 ball.

OpenCV finds the 8 on the fixture we already have (no extra photo, no camera).
The API call is still one Flash request, `detail=low`, one tiny JPEG.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

# This script lives one level down but drives the pipeline at the repo root,
# so the root goes on the path before the project imports below.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tools import htn26_cli as cli  # noqa: E402
from companion.pool.perception.vision.classify.baseten import (CROP_PX, MODEL, classify_one_crop,  # noqa: E402
                              crop_disc)
from companion.pool.perception.vision.calibration import calibrate_from_frames  # noqa: E402
from companion.pool.perception.vision.detector import BALL_RADIUS_PX, detect_balls  # noqa: E402
from companion.pool.perception.vision.envutil import load_dotenv  # noqa: E402
from companion.pool.perception.vision.surface import set_active_surface  # noqa: E402

# Hand label from tests/test_pipeline.py — the 8 on real_table_balls.png.
EIGHT_XY = (0.7025, 0.3056)


def main() -> int:
    load_dotenv()
    # Finding centres is free. Do not let detect_balls spend a 16-crop call
    # before the one-image smoke has even run.
    os.environ["BASETEN_CLASSIFY"] = "0"
    path = ROOT / "data/local/fixtures/real_table_balls.png"
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"error: missing {path}", file=sys.stderr)
        return 2

    set_active_surface("table")
    if cli.measure_sheet([frame]) is None:
        print("error: could not measure the fixture table", file=sys.stderr)
        return 2
    grid, _ = calibrate_from_frames([frame])
    balls, view = detect_balls(frame, grid.homography)
    if not balls:
        print("error: OpenCV found no balls", file=sys.stderr)
        return 2

    eight = min(balls, key=lambda b: float(np.hypot(
        b.xy_mm[0] - EIGHT_XY[0], b.xy_mm[1] - EIGHT_XY[1])))
    cx, cy = view.mm_to_px(*eight.xy_mm)
    crop = crop_disc(view.image, cx, cy, BALL_RADIUS_PX)

    print(f"smoke: {MODEL}  {CROP_PX}px jpeg  detail=low  1 image")
    print(f"  crop from {eight.id} at ({eight.xy_mm[0]:.3f},{eight.xy_mm[1]:.3f})")
    try:
        call = classify_one_crop(crop)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"got kind={call.kind} number={call.number}")
    if call.kind != "eight":
        print("error: expected eight", file=sys.stderr)
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
