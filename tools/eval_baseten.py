"""Run the wired detector on a saved frame and score kinds against hand labels."""

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
from companion.pool.perception.vision.detector import detect_balls  # noqa: E402
from companion.pool.perception.vision.envutil import load_dotenv  # noqa: E402
from companion.pool.perception.vision.surface import set_active_surface  # noqa: E402
from companion.pool.perception.vision.table_spec import active_spec  # noqa: E402
from tests.perception.test_vision_pipeline import BALL_POSITION_TOLERANCE, BALL_TRUTH
from companion.pool.perception.vision.calibration import calibrate_from_frames  # noqa: E402


def main() -> int:
    load_dotenv()
    os.environ["BASETEN_CLASSIFY"] = "1"
    if not os.environ.get("BASETEN_API_KEY"):
        print("error: BASETEN_API_KEY missing from os.environ", file=sys.stderr)
        return 2

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "data/local/fixtures/real_table_balls.png")
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"error: cannot read {path}", file=sys.stderr)
        return 2

    set_active_surface("table")
    assert cli.measure_sheet([frame]) is not None
    grid, _ = calibrate_from_frames([frame])
    balls, _ = detect_balls(frame, grid.homography)
    spec = active_spec()
    print(f"{path.name}: {len(balls)} balls on {spec.describe()}")

    used, right, rows = set(), 0, []
    for x, y, kind in BALL_TRUTH:
        best, distance = None, 1e9
        for i, ball in enumerate(balls):
            d = float(np.hypot(ball.xy_mm[0] - x, ball.xy_mm[1] - y))
            if d < distance:
                best, distance = i, d
        if best is None or best in used or distance > BALL_POSITION_TOLERANCE:
            rows.append((kind, None, distance))
            continue
        used.add(best)
        ball = balls[best]
        ok = ball.kind == kind
        right += ok
        tag = "ok" if ok else "WRONG"
        number = f" #{ball.number}" if ball.number else ""
        print(f"  {tag:5} truth={kind:<6} got={ball.kind:<6}{number}  "
              f"{ball.id}  d={distance:.4f}")
        rows.append((kind, ball, distance))

    extra = [b for i, b in enumerate(balls) if i not in used]
    print(f"kinds {right}/{len(BALL_TRUTH)}   extra {len(extra)}")
    return 0 if right >= 14 and not extra else 1


if __name__ == "__main__":
    raise SystemExit(main())
