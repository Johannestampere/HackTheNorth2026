"""Scan the real table with the camera and plot the shot it suggests.

The whole workflow, from hardware to decision, in one command:

    1. camera      open a webcam and pool a burst of frames
    2. grid        measure the cloth and lock the x/y grid          (vision/)
    3. balls       find them and classify each one                  (vision/classify)
    4. plot        draw the measured grid with the balls on it
    5. physics     simulate candidate shots and choose one          (pooltool)
    6. plot        draw the decision: paths, ghost ball, aim vector

Steps 4 and 6 are two figures, written side by side, because they answer
different questions: the first is "did the camera read the table correctly",
the second is "what should I do about it". When a shot looks wrong, the first
figure is where you find out whether the fault was perception or planning.

Unlike `shot_demo.py`, which replays a saved capture, this reads the camera.
It also expects a *partial* rack - balls leave a table as the game is played -
and says plainly which readings a short count makes unreliable rather than
presenting them at the same confidence as a full one.

`--frame` substitutes an image on disk for the camera and changes nothing
else, so a saved frame reproduces a decision exactly - on a machine with no
camera, or to re-examine a scan that went wrong. The grid is then locked from
that single frame rather than from a burst, which is a weaker measurement of
the table's shape: the burst exists so many views can outvote the badly
conditioned ones, and one still has nothing to pool.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2                                                   # noqa: E402
import matplotlib                                            # noqa: E402
import numpy as np                                           # noqa: E402

from companion.pool.contracts import (BallType, PerceptionReady,  # noqa: E402
                                      PlanningReady)
from companion.pool.contracts.serialization import (           # noqa: E402
    load_game_context, load_geometry, save_table_state)
from companion.pool.perception.service import PerceptionService  # noqa: E402
from companion.pool.perception.vision import surface, table_spec  # noqa: E402
from companion.pool.perception.vision.calibration import (      # noqa: E402
    CalibrationError, calibrate_from_camera, calibrate_from_frames)
from companion.pool.perception.vision.camera import (           # noqa: E402
    Camera, CameraConfig, CameraError, find_camera_index)
from companion.pool.perception.vision.ball_scale import (       # noqa: E402
    measure_ball_scale)
from companion.pool.perception.vision.detector import (         # noqa: E402
    BALL_RADIUS_FRAC, describe_inconsistency, detect_balls)

from shot_plot import plot_state, plot_shot                    # noqa: E402

# A pool set. Below this the comparative calls - which ball is the cue, which
# is the 8, the seven-and-seven pairing - have less to work with, and the
# detector says so itself; this is only used to phrase the warning.
SET_SIZE = 16


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan the table with the camera, then plan and plot a shot.")
    parser.add_argument("--camera", type=int, default=None,
                        help="Camera index; default picks an external one")
    parser.add_argument("--frames", type=int, default=30,
                        help="Frames to pool for the grid (default 30)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--settle", type=float, default=1.5,
                        help="Seconds to let exposure settle before capturing")
    parser.add_argument("--geometry", type=Path,
                        default=ROOT / "fixtures/table_geometry.json")
    parser.add_argument("--game", type=Path,
                        default=ROOT / "fixtures/game_contexts/solids.json")
    parser.add_argument("--outdir", type=Path, default=ROOT / "artifacts/live")
    parser.add_argument("--show", action="store_true",
                        help="Open windows as well as writing the files")
    parser.add_argument("--save-frame", action="store_true",
                        help="Keep the captured frame, to re-run offline")
    parser.add_argument("--frame", type=Path, default=None,
                        help="Read this image instead of opening the camera")
    parser.add_argument("--no-plan", action="store_true",
                        help="Scan and plot the table only; skip the physics")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    geometry = load_geometry(args.geometry)

    # The pipeline measures shape from the view and takes scale from the
    # caller, exactly as PerceptionService does. Installing the same spec here
    # keeps the live path and the stage path in one frame.
    table_spec.set_active_spec(table_spec.TableSpec(
        width_mm=geometry.length, height_mm=geometry.width, units="u"))
    surface.set_active_surface("table")

    # ---- 1. camera ----------------------------------------------------
    frame, grid = _scan(args)
    if frame is None:
        return 1

    if args.save_frame:
        path = args.outdir / "frame.png"
        cv2.imwrite(str(path), frame)
        print(f"  saved {path}")

    # ---- 2/3. balls, classified ---------------------------------------
    print("balls: detecting and classifying ...")
    found, _ = detect_balls(frame, grid.homography)
    if not found:
        print("  no balls found on the measured cloth", file=sys.stderr)
        print("  the grid locked, so this is lighting or ball contrast, not "
              "the table", file=sys.stderr)
        return 1

    kinds = {}
    for ball in found:
        kinds[ball.kind] = kinds.get(ball.kind, 0) + 1
    print(f"  {len(found)} balls: "
          + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())))

    # ---- 2b. the ball radius, measured rather than assumed --------------
    # The fixture's radius is equipment calibration for one table, and the
    # planner's blocking test is where a wrong one becomes a wrong shot: it
    # compares a measured ball separation against `2 * ball_radius`, so a
    # radius half the truth reports a blocked lane as clear. Measure it from
    # this frame and let the fixture be the seed only.
    #
    # The measurement goes to the *geometry* and nowhere else. It deliberately
    # does not re-run detection, even though the detector's kernels are all
    # multiples of the radius and are therefore the wrong size. On this table
    # re-detecting at the true 0.0277 kept all 9 balls and their positions but
    # destroyed the stripe/solid call - 3 stripes became 0, and the network
    # hallucinated a second 8 ball. Both classifiers were calibrated around
    # the 0.0150 prior: the rim annulus reaches a real ball's shadowed edge
    # rather than its bright interior, and the CNN's training crops were cut
    # from this pipeline at that same prior, so a correctly-sized crop is a
    # domain it has never seen.
    #
    # What makes the split legitimate rather than a fudge is that a ball's
    # *position* comes out in table units and does not depend on the prior at
    # all. So detection keeps the scale its classifiers were tuned for, the
    # planner gets the scale the table actually has, and neither is asked to
    # use a number that is wrong for it. Retraining the classifier at a
    # measured radius is the real fix and is not this change.
    scale = measure_ball_scale(frame, grid.homography,
                               prior_frac=BALL_RADIUS_FRAC)
    if scale is None:
        print(f"  ball radius: not measurable on this frame; keeping the "
              f"assumed {BALL_RADIUS_FRAC:.4f} of the long side",
              file=sys.stderr)
    else:
        print(f"  {scale.describe()}")
        if not scale.converged:
            print("  NOTE: the radius estimate had not settled; treat the "
                  "blocking calls as provisional", file=sys.stderr)
        if scale.spread > 0.20:
            print(f"  NOTE: balls disagree about their own size by "
                  f"{scale.spread:.0%}; check the lock and the lighting",
                  file=sys.stderr)
        geometry = replace(geometry, ball_radius=scale.radius_frac)

    # The count check belongs to the detector's own prior, which is what
    # found these balls; the measured radius did not.
    warning = describe_inconsistency(found, BALL_RADIUS_FRAC)
    if warning:
        print(f"\n  NOTE: {warning}\n", file=sys.stderr)
    elif len(found) < SET_SIZE:
        # Below the detector's own warning floor it stays quiet, because balls
        # legitimately leave the table. The comparative calls are still doing
        # less work than on a full rack, and that is worth one line.
        print(f"  (partial rack: {len(found)} of {SET_SIZE}. The cue and the 8 "
              f"are chosen by comparison, so confirm them in the grid plot.)")

    # Reuse the stage for the contract conversion, so the live path and
    # `companion.app perceive` cannot disagree about units or axes.
    state = _as_table_state(frame, geometry, args.outdir)
    if state is None:
        return 1
    save_table_state(args.outdir / "table_state.json", state)
    print(f"  wrote {args.outdir / 'table_state.json'}")

    # ---- 4. the measured grid, with the balls on it --------------------
    if not (args.show):
        matplotlib.use("Agg")
    grid_png = args.outdir / "1-table-scan.png"
    plot_state(state, grid_png, show=args.show,
               title=f"measured grid · {len(state.balls)} balls")
    print(f"  wrote {grid_png}")

    if args.no_plan:
        return 0

    # ---- 5. physics and the decision -----------------------------------
    blocker = _why_planning_cannot_run(state)
    if blocker:
        print(f"\nplanning: skipped - {blocker}", file=sys.stderr)
        print("The scan above is still valid; only the shot choice needs "
              "these.", file=sys.stderr)
        return 1

    print("planning: simulating candidate shots ...")
    try:
        from companion.pool.planning.service import (PlannerConfig,
                                                     PooltoolPlanner)
    except ImportError:
        print("  pooltool is not installed; see docs/perception.md",
              file=sys.stderr)
        return 1

    started = time.monotonic()
    result = PooltoolPlanner(PlannerConfig()).plan(
        state, load_game_context(args.game))
    if not isinstance(result, PlanningReady):
        print(f"  {type(result).__name__}: "
              f"{getattr(result, 'reason', '')}", file=sys.stderr)
        # An earlier run's decision plot must not survive a run that reached
        # no decision. The two figures are meant to be read side by side, so
        # a stale one next to a fresh scan reads as this scan's answer - and
        # "no legal pot" is exactly when someone goes looking at the picture.
        stale = args.outdir / "2-shot-decision.png"
        if stale.exists():
            stale.unlink()
            print(f"  removed {stale}, which was from an earlier scan",
                  file=sys.stderr)
        return 1
    plan = result.plan
    print(f"  pot {plan.target_ball_id} into {plan.target_pocket_id} "
          f"({time.monotonic() - started:.1f}s)")
    print(f"  aim ({plan.cue_aim.direction.x:+.3f}, "
          f"{plan.cue_aim.direction.y:+.3f}) at "
          f"{plan.cue_stick_speed:.2f} table lengths/s")

    # ---- 6. the decision, plotted --------------------------------------
    shot_png = args.outdir / "2-shot-decision.png"
    plot_shot(state, plan, shot_png, show=args.show)
    print(f"  wrote {shot_png}")
    return 0


def _scan(args) -> tuple[np.ndarray | None, object]:
    """Open the camera, settle it, pool frames and lock the grid.

    With `--frame` the image is read from disk instead. The grid is locked
    from that one frame, which is the whole difference: the burst exists to
    pool the perspective across many views, and a single still cannot do
    that, so the shape it recovers is weaker than a live lock's. Everything
    downstream is identical, which is the point - it lets a saved frame
    reproduce a decision exactly, on hardware with no camera attached.
    """
    if args.frame is not None:
        frame = cv2.imread(str(args.frame))
        if frame is None or not frame.size:
            print(f"frame: {args.frame} is not a readable image",
                  file=sys.stderr)
            return None, None
        print(f"frame: {args.frame} ({frame.shape[1]}x{frame.shape[0]})")
        try:
            grid, last = calibrate_from_frames([frame])
        except CalibrationError as error:
            print(f"grid: {error}", file=sys.stderr)
            return None, None
        print(f"grid: locked from 1 frame, fit error "
              f"{grid.homography.reprojection_error_mm:.4f} u")
        for note in grid.notes:
            print(f"  note: {note}")
        return last.image, grid

    try:
        index = args.camera
        if index is None:
            index, name = find_camera_index(args.width, args.height)
            print(f"camera: using index {index} ({name})")
        else:
            print(f"camera: using index {index}")
    except CameraError as error:
        print(f"camera: {error}", file=sys.stderr)
        return None, None

    config = CameraConfig(index=index, width=args.width, height=args.height)
    try:
        with Camera(config) as camera:
            for note in getattr(camera, "notes", []):
                print(f"  note: {note}")
            if args.settle > 0:
                # Auto-exposure is disabled where the driver allows it, but a
                # camera still needs a moment before its first frames are
                # representative. Calibrating on those costs accuracy.
                print(f"  settling {args.settle:.1f}s ...")
                deadline = time.monotonic() + args.settle
                while time.monotonic() < deadline:
                    camera.read()
            print(f"  pooling {args.frames} frames for the grid ...")
            grid, last = calibrate_from_camera(camera, frames=args.frames)
    except CameraError as error:
        print(f"camera: {error}", file=sys.stderr)
        return None, None
    except CalibrationError as error:
        print(f"grid: {error}", file=sys.stderr)
        print("  aim so all four cushions are in frame, and light the cloth "
              "enough for its colour to show.", file=sys.stderr)
        return None, None

    print(f"grid: locked from {grid.frames_used}/{grid.frames_attempted} "
          f"frames, fit error {grid.homography.reprojection_error_mm:.4f} u")
    for note in grid.notes:
        print(f"  note: {note}")
    return last.image, grid


def _as_table_state(frame: np.ndarray, geometry, outdir: Path):
    """The frame through PerceptionService, so one conversion serves both.

    The service detects at its own tuned prior, not at `geometry.ball_radius`.
    Those are two different jobs for one number - finding and classifying a
    ball, versus knowing how much room it takes up - and only the second is
    measured here. See the note at the call site.
    """
    import json
    path = outdir / "_scan_capture.json"
    image_path = outdir / "_scan_frame.png"
    cv2.imwrite(str(image_path), frame)
    path.write_text(json.dumps({
        "schema_version": 3,
        "kind": "capture_batch",
        "data": {
            "batch_id": f"live-{int(time.time())}",
            "captures": [{
                "capture_id": "live-1",
                "captured_at_s": float(int(time.time())),
                "rgb_path": image_path.name,
                "calibration_id": "live-uncalibrated",
                "view_id": "live",
            }],
        },
    }, indent=2), encoding="utf-8")

    from companion.sensors.models import load_capture_batch
    result = PerceptionService().estimate(load_capture_batch(path), geometry)
    if not isinstance(result, PerceptionReady):
        print(f"  {type(result).__name__}: {result.reason}", file=sys.stderr)
        return None
    return result.state


def _why_planning_cannot_run(state) -> str | None:
    """The planner's own preconditions, checked before the long search.

    It returns `InsufficientInformation` for each of these anyway, but after
    loading pooltool and building a table; saying it here is immediate and
    names the missing thing in terms of what to put back on the cloth.
    """
    kinds = [ball.type for ball in state.balls]
    if BallType.CUE not in kinds:
        return ("no cue ball was found. A shot needs one: put it on the "
                "cloth, or check the grid plot in case a ball was "
                "misclassified.")
    if BallType.EIGHT not in kinds:
        return ("no 8 ball was found. The planner needs it to judge legality "
                "even when it is not the target.")
    if BallType.UNKNOWN in kinds:
        return (f"{kinds.count(BallType.UNKNOWN)} ball(s) unclassified. The "
                f"planner will not choose a target while a ball's type is "
                f"unknown, since it cannot tell whether potting it is legal.")
    return None


if __name__ == "__main__":
    raise SystemExit(main())
