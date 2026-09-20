"""CLI entry point: calibrate the table grid and export it."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from companion.pool.perception.vision.calibration import (DEFAULT_FRAMES, CalibrationError,
                                FrameResult, Grid,
                                calibrate_from_camera,
                                calibrate_from_frames)
from companion.pool.perception.vision.camera import Camera, CameraConfig, CameraError, find_camera_index
from companion.pool.perception.vision.aspect import (estimate_aspect_ratio, estimate_focal_length,
                             estimate_focal_length_multi, focal_conditioning)
from companion.pool.perception.vision.surface import (SURFACES, active_surface, describe_failure,
                            detect_reference_points, find_quad,
                            set_active_surface)
from companion.pool.perception.vision.table_spec import (DEFAULT_REFERENCE_SIDE, DEFAULT_UNITS,
                                 active_spec,
                                 TableSpec, set_active_spec, sheet_h_mm,
                                 sheet_w_mm)
from companion.pool.perception.vision.detector import (BALL_RADIUS_FRAC, Ball, BallMemory, ball_radius_mm,
                          describe_inconsistency, detect_balls)
from companion.pool.perception.vision.envutil import load_dotenv
from companion.pool.perception.vision.marks import MARK_RADIUS_MM, Mark, detect_marks
from companion.pool.perception.vision.overlay import (ball_counts, ball_summary, draw_overlay,
                            draw_rectified, draw_xy_graph, size_label)

DEFAULT_OUTDIR = Path("out")

# The reference dimensions in force for the live preview, set from the CLI:
# (reference_mm, reference_side, height_mm or None).
_LIVE_REFERENCE: tuple[float | None, str, float | None] = (
    None, DEFAULT_REFERENCE_SIDE, None)

# A ball's radius as a fraction of the surface's long side, from the CLI.
_BALL_RADIUS_FRAC = BALL_RADIUS_FRAC

# Offline `--vlm` only. Live B is always OpenCV; live V is the model.
_USE_VLM = False

# Diagnostics from the most recent measurement, for reporting.
_LAST_MEASUREMENT: dict = {}

# Below this, the view cannot determine the surface's shape and the
# focal-length estimate must not be used to derive an aspect ratio.
MIN_SHAPE_CONFIDENCE = 0.35


def export(grid: Grid, result: FrameResult, outdir: Path) -> None:
    """Write grid.json, overlay.png and rectified.png."""
    outdir.mkdir(parents=True, exist_ok=True)
    grid_path = outdir / "grid.json"
    grid_path.write_text(json.dumps(grid.as_dict(), indent=2), encoding="utf-8")
    cv2.imwrite(str(outdir / "overlay.png"), draw_overlay(result, grid))
    cv2.imwrite(str(outdir / "rectified.png"),
                draw_rectified(result.view, grid))
    cv2.imwrite(str(outdir / "xy_graph.png"), draw_xy_graph(grid))

    found = sum(h.found for h in grid.holes)
    unit = active_spec().units
    # Normalised coordinates run 0..1, so they need decimals where millimetres
    # do not. One formatter, chosen from the units actually in force.
    fmt = (lambda v: f"{v:8.1f}") if unit == "mm" else (lambda v: f"{v:8.4f}")

    print(f"wrote {grid_path}")
    print(f"wrote {outdir / 'overlay.png'}")
    print(f"wrote {outdir / 'rectified.png'}")
    print(f"wrote {outdir / 'xy_graph.png'}")
    print(f"\npockets: {found}/6   "
          f"corner fit: {grid.homography.reprojection_error_mm:.4f} {unit}   "
          f"frames used: {grid.frames_used}/{grid.frames_attempted}")
    print(f"\ncorners (table {unit} -> image px):")
    for corner in grid.as_dict()["corners"]:
        print(f"  {corner['id']:>2}  ({fmt(corner['xy_mm'][0])},"
              f"{fmt(corner['xy_mm'][1])}) {unit}   "
              f"px ({corner['px'][0]:7.1f},{corner['px'][1]:7.1f})")
    print("\npockets:")
    for hole in grid.holes:
        if hole.found:
            print(f"  {hole.id:>2}  ({fmt(hole.xy_mm[0])},{fmt(hole.xy_mm[1])})"
                  f" {unit}   err {hole.error_mm:.4f}")
        else:
            print(f"  {hole.id:>2}  MISSING   nominal "
                  f"({fmt(hole.nominal_xy_mm[0])},{fmt(hole.nominal_xy_mm[1])})")
    for note in grid.notes:
        print(f"note: {note}")


def measure_sheet(frames: list[np.ndarray],
                  reference: float | None = None,
                  reference_side: str = "long",
                  height_mm: float | None = None) -> TableSpec | None:
    """Measure the surface's shape and install it as the active spec.

    What a single camera can measure is the *shape* - the ratio of the sides.
    What it cannot measure, at all, is the absolute size: a small surface
    close to the lens and a large one far away project to pixel-identical
    images, so no algorithm distinguishes them. The tool therefore reports
    normalised units (long side = 1.0) unless the caller supplies a real
    measurement, in which case that one number sets the scale and the
    recovered shape supplies the rest.
    """
    quads = [q for q in (find_quad(f) for f in frames) if q is not None]
    if not quads:
        return None
    shape = frames[0].shape

    # The focal length is one fixed number the whole burst measures, so pool
    # it across frames and weight by conditioning. A single near-straight-on
    # view gives a wild answer (1 px of corner noise moves it by tens of px),
    # which would then corrupt every ratio derived from it.
    focal, confidence = estimate_focal_length_multi(quads, shape)

    # A badly conditioned focal length must not be used. The recovered aspect
    # ratio is extremely sensitive to it - measured on a real near-top-down
    # frame of the pool table, f=500 gives 0.54 and f=3044 gives 0.60, and
    # the focal estimate there was worthless (one vanishing point 19,000 px
    # from the image centre). Passing it on anyway is how a 2:1 table came
    # out as 140 x 97 mm.
    #
    # With no trustworthy focal length the honest fallback is the apparent
    # ratio, which at least assumes nothing: it is exactly right for a
    # top-down view and degrades smoothly as the view tilts. The caller is
    # told, via `confidence`, that this happened.
    trusted = focal if confidence >= MIN_SHAPE_CONFIDENCE else None

    ratios = []
    for quad in quads:
        ratio = estimate_aspect_ratio(
            quad, shape, focal_length=trusted,
            allow_estimated_focal=trusted is not None)
        if ratio is not None and np.isfinite(ratio):
            ratios.append(ratio)
    if not ratios:
        return None

    ratio = float(np.median(ratios))  # height / width

    if reference is None:
        # No measurement offered, so claim no units. Normalise on the longer
        # side: width x height with the larger of the two at 1.0.
        if ratio >= 1.0:
            spec = TableSpec(1.0 / ratio, 1.0, DEFAULT_UNITS)
        else:
            spec = TableSpec(1.0, ratio, DEFAULT_UNITS)
    elif height_mm is not None:
        # Both dimensions given: nothing is estimated at all.
        spec = (TableSpec(reference, height_mm, "mm")
                if reference_side != "height"
                else TableSpec(height_mm, reference, "mm"))
    elif reference_side == "height":
        spec = TableSpec(reference / ratio, reference, "mm")
    elif reference_side == "long":
        # Scale the longer side, whichever it turns out to be.
        if ratio >= 1.0:
            spec = TableSpec(reference / ratio, reference, "mm")
        else:
            spec = TableSpec(reference, reference * ratio, "mm")
    else:
        spec = TableSpec(reference, reference * ratio, "mm")
    set_active_spec(spec)
    _LAST_MEASUREMENT.update(focal=focal, confidence=confidence,
                             focal_used=trusted is not None,
                             measured_both=height_mm is not None,
                             frames=len(quads), ratio=ratio,
                             conditioning=max(
                                 focal_conditioning(
                                     q, (shape[1] / 2.0, shape[0] / 2.0))
                                 for q in quads))
    return spec


def _report_measurement(spec: TableSpec | None, frames: list[np.ndarray],
                        reference: float | None, reference_side: str) -> None:
    if spec is None:
        print("note: could not measure the sheet; using the previous spec",
              file=sys.stderr)
        return
    if reference is None:
        print(f"measured shape: {spec.describe()}")
        print(f"  aspect (short/long) {spec.aspect:.4f}  - no absolute size "
              "is claimed: a single camera cannot measure one")
    else:
        print(f"measured surface: {spec.describe()}   "
              f"({reference_side} side pinned to {reference:.1f} mm)")

    thing = "table" if active_surface() == "table" else "sheet"
    if _LAST_MEASUREMENT.get("measured_both"):
        print("  both dimensions measured by hand - nothing was estimated")
        return

    focal = _LAST_MEASUREMENT.get("focal")
    confidence = float(_LAST_MEASUREMENT.get("confidence") or 0.0)
    if focal:
        used = "used" if _LAST_MEASUREMENT.get("focal_used") else "NOT used"
        print(f"  focal length {focal:.0f} px from "
              f"{_LAST_MEASUREMENT.get('frames', 0)} views "
              f"(confidence {confidence:.0%}, {used})")

    if confidence < MIN_SHAPE_CONFIDENCE:
        print(f"note: the view is close to straight-on, so the {thing}'s "
              "shape comes from its outline rather than from perspective. "
              "That is the right answer for a top-down view and degrades "
              "gently as the view tilts, but it is an approximation in "
              "between.")
        print("      For the sharpest shape estimate, view the surface at a "
              "clear angle so both pairs of edges visibly converge.")


def run_image(path: Path, outdir: Path, require_all_holes: bool,
              reference: float | None = None,
              reference_side: str = DEFAULT_REFERENCE_SIDE,
              height_mm: float | None = None) -> int:
    image = cv2.imread(str(path))
    if image is None:
        print(f"error: cannot read image {path}", file=sys.stderr)
        return 2
    spec = measure_sheet([image], reference, reference_side, height_mm)
    _report_measurement(spec, [image], reference, reference_side)
    try:
        grid, result = calibrate_from_frames([image],
                                             require_all_holes=require_all_holes)
    except CalibrationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    export(grid, result, outdir)
    # The same detection the live keys run, so a saved frame can be checked
    # without the camera.
    if active_surface() == "table":
        _detect_balls_and_export(image, grid, result, outdir,
                                 use_vlm=_USE_VLM)
    else:
        _detect_and_export(image, grid, result, outdir)
    return 0


def run_camera(config: CameraConfig, frames: int, outdir: Path,
               require_all_holes: bool, reference: float | None,
               reference_side: str, height_mm: float | None = None) -> int:
    try:
        with Camera(config) as camera:
            for note in camera.notes:
                print(f"note: {note}")
            print(f"capturing {frames} frames...")
            captured = [camera.read() for _ in range(frames)]
            spec = measure_sheet(captured, reference, reference_side,
                                 height_mm)
            _report_measurement(spec, captured, reference,
                                reference_side)
            grid, result = calibrate_from_frames(
                captured, require_all_holes=require_all_holes)
    except (CameraError, CalibrationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    export(grid, result, outdir)
    # The balls come from the last frame alone, not the burst: the geometry
    # is worth pooling because the table does not move, and the balls are
    # not, because they might have.
    if active_surface() == "table":
        _detect_balls_and_export(captured[-1], grid, result, outdir,
                                 use_vlm=_USE_VLM)
    return 0


LOCK_SECONDS = 5.0


def run_preview(config: CameraConfig, frames: int, outdir: Path,
                require_all_holes: bool, reference: float | None,
                reference_side: str, height_mm: float | None = None) -> int:
    """Live window: lock the surface for 5 s, then B finds the balls.

    The surface's geometry is measured once, at the start, and then held.
    The balls are looked for on demand against that held geometry, so
    playing a shot and pressing B again re-reads only the balls - the grid
    they are measured against does not shift underneath them.
    """
    window = "live - B to find the balls, q to quit"
    graph_window = "x,y grid"
    try:
        with Camera(config) as camera:
            for note in camera.notes:
                print(f"note: {note}")
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, config.width, config.height)
            cv2.namedWindow(graph_window, cv2.WINDOW_NORMAL)
            cv2.moveWindow(graph_window, config.width + 30, 0)
            print("\nPreview running.")
            print(f"  Holding still for {LOCK_SECONDS:.0f} s to lock the "
                  "corners and pockets...")

            locked = _lock_geometry(camera, window, graph_window, frames,
                                    require_all_holes, reference,
                                    reference_side, height_mm)
            if locked is None:
                return 1
            grid, result = locked

            print("\n  Locked.")
            print("  Press B to find the balls (OpenCV, instant).")
            print("  Press V to re-classify those balls with Baseten.")
            print("  Press D to detect dark scraps instead.")
            print("  Press R to re-lock the table geometry.")
            print("  Press q to quit.\n")

            marks: list[Mark] = []
            balls: list[Ball] = []
            # Kept across presses, not across locks: it remembers where balls
            # were resting, and a re-lock can move the origin those positions
            # are measured against. Reset with the geometry below.
            memory = BallMemory()
            while True:
                frame = camera.read()
                canvas = _locked_frame(frame, grid, result, marks, balls)
                cv2.imshow(window, canvas)
                cv2.imshow(graph_window,
                           draw_xy_graph(grid, marks=marks or None,
                                         balls=balls or None))
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    print("quit")
                    return 0
                if key in (ord("b"), ord("B")):
                    balls = _detect_balls_and_export(frame, grid, result,
                                                     outdir, use_vlm=False,
                                                     memory=memory)
                if key in (ord("v"), ord("V")):
                    balls = _detect_balls_and_export(frame, grid, result,
                                                     outdir, use_vlm=True)
                if key in (ord("d"), ord("D")):
                    marks = _detect_and_export(frame, grid, result, outdir)
                if key in (ord("r"), ord("R")):
                    print("\nre-locking...")
                    relocked = _lock_geometry(camera, window, graph_window,
                                              frames, require_all_holes,
                                              reference, reference_side,
                                              height_mm)
                    if relocked is not None:
                        grid, result = relocked
                        marks, balls = [], []
                        memory = BallMemory()
                        print("  Locked. Press B to find the balls.\n")
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    print("window closed")
                    return 0
    except (CameraError, CalibrationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()


def _lock_geometry(camera, window: str, graph_window: str, frames: int,
                   require_all_holes: bool, reference: float | None,
                   reference_side: str, height_mm: float | None = None
                   ) -> tuple[Grid, FrameResult] | None:
    """Watch the sheet for LOCK_SECONDS, then measure it from that burst.

    Every frame of the countdown is kept, not just the last few: the shape
    comes out of the perspective across the whole burst, and pooling many
    views is what lets the well-conditioned ones outvote the rest (see
    `geometry.aspect.estimate_focal_length_multi`).
    """
    deadline = time.monotonic() + LOCK_SECONDS
    captured: list[np.ndarray] = []
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            break
        frame = camera.read()
        captured.append(frame)
        preview, live_grid = _preview_frame(frame)
        _banner(preview, f"LOCKING IN {remaining:.1f}s - hold still",
                (0, 170, 220))
        cv2.imshow(window, preview)
        cv2.imshow(graph_window, _graph_frame(live_grid))
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            print("quit during lock")
            return None

    if not captured:
        print("error: no frames captured during the lock", file=sys.stderr)
        return None

    print(f"  measuring from {len(captured)} frames...")
    spec = measure_sheet(captured, reference, reference_side, height_mm)
    _report_measurement(spec, captured, reference, reference_side)
    try:
        return calibrate_from_frames(captured,
                                     require_all_holes=require_all_holes)
    except CalibrationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return None


def _detect_and_export(frame: np.ndarray, grid: Grid, result: FrameResult,
                       outdir: Path) -> list[Mark]:
    """Find the marks on this frame, report them, and write the PNG."""
    marks, _ = detect_marks(frame, grid.homography)
    unit = active_spec().units
    fmt = (lambda v: f"{v:7.1f}") if unit == "mm" else (lambda v: f"{v:7.4f}")
    print(f"\ndetected {len(marks)} marks:")
    for mark in marks:
        print(f"  {mark.id:>3}  ({fmt(mark.xy_mm[0])},{fmt(mark.xy_mm[1])})"
              f" {unit}   area {mark.area_mm2:.4g} {unit}2")
    if not marks:
        print("  none found - are the scraps fully on the white paper?")
    export_marks(frame, grid, result, marks, outdir)
    return marks


def _detect_balls_and_export(frame: np.ndarray, grid: Grid,
                             result: FrameResult, outdir: Path,
                             *, use_vlm: bool = False,
                             memory: BallMemory | None = None) -> list[Ball]:
    """Find the balls on this frame, report them, and write the outputs."""
    if use_vlm:
        from companion.pool.perception.vision.classify.baseten import api_key
        if not api_key():
            print("note: BASETEN_API_KEY is not set; using OpenCV")
        else:
            print("\nclassifying with Baseten (this is the slow path)...")
    balls, _ = detect_balls(frame, grid.homography,
                            radius_frac=_BALL_RADIUS_FRAC,
                            use_vlm=use_vlm, memory=memory)
    unit = active_spec().units
    fmt = (lambda v: f"{v:7.1f}") if unit == "mm" else (lambda v: f"{v:7.4f}")

    print(f"\ndetected {len(balls)} balls   ({ball_summary(balls)})")
    for ball in balls:
        flag = "  uncertain" if ball.confidence < 0.6 else ""
        number = f" {ball.number}" if ball.number else ""
        print(f"  {ball.id:>3}  {ball.kind:>6}{number:<3}  "
              f"({fmt(ball.xy_mm[0])},{fmt(ball.xy_mm[1])}) {unit}"
              f"   conf {ball.confidence:.2f}{flag}")
    if not balls:
        print("  none found - is the table lit well enough to see them?")

    # A wrong radius prior does not look like a failure from inside any one
    # measurement, so say so here rather than leaving the counts to be read as
    # a result. See `detect.balls.describe_inconsistency`.
    problem = describe_inconsistency(balls, _BALL_RADIUS_FRAC)
    if problem:
        for line in _wrap(problem, 72):
            print(f"  ! {line}")
    export_balls(frame, grid, result, balls, outdir)
    return balls


def export_balls(frame: np.ndarray, grid: Grid, result: FrameResult,
                 balls: list[Ball], outdir: Path) -> None:
    """Write the ball graph, overlay and JSON alongside the grid."""
    outdir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(outdir / "balls_graph.png"),
                draw_xy_graph(grid, balls=balls))

    shot = FrameResult(homography=grid.homography, holes=grid.holes,
                       view=result.view, image=frame)
    cv2.imwrite(str(outdir / "balls_overlay.png"),
                draw_overlay(shot, grid, balls=balls))

    payload = {
        "units": active_spec().units,
        "origin": f"{active_surface()} top-left corner",
        "frame": {"width": round(sheet_w_mm(), 4),
                  "height": round(sheet_h_mm(), 4)},
        "radius": round(ball_radius_mm(_BALL_RADIUS_FRAC), 4),
        "radius_frac_of_long_side": _BALL_RADIUS_FRAC,
        "counts": ball_counts(balls),
        # Recorded, not just printed: a saved reading has to carry the reason
        # it should not be trusted, or the JSON outlives the warning.
        "warning": describe_inconsistency(balls, _BALL_RADIUS_FRAC),
        "balls": [b.as_dict() for b in balls],
    }
    (outdir / "balls.json").write_text(json.dumps(payload, indent=2),
                                       encoding="utf-8")
    print(f"wrote {outdir / 'balls_graph.png'}")
    print(f"wrote {outdir / 'balls_overlay.png'}")
    print(f"wrote {outdir / 'balls.json'}")


def export_marks(frame: np.ndarray, grid: Grid, result: FrameResult,
                 marks: list[Mark], outdir: Path) -> None:
    """Write the marks PNG and JSON alongside the grid."""
    outdir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(outdir / "marks_graph.png"),
                draw_xy_graph(grid, marks=marks))

    shot = FrameResult(homography=grid.homography, holes=grid.holes,
                       view=result.view, image=frame)
    cv2.imwrite(str(outdir / "marks_overlay.png"),
                draw_overlay(shot, grid, marks=marks))

    payload = {
        "units": "mm",
        "origin": "paper top-left corner",
        "frame": {"width_mm": round(sheet_w_mm(), 2),
                  "height_mm": round(sheet_h_mm(), 2)},
        "radius_mm": MARK_RADIUS_MM,
        "marks": [m.as_dict() for m in marks],
    }
    (outdir / "marks.json").write_text(json.dumps(payload, indent=2),
                                       encoding="utf-8")
    print(f"wrote {outdir / 'marks_graph.png'}")
    print(f"wrote {outdir / 'marks_overlay.png'}")
    print(f"wrote {outdir / 'marks.json'}")


def _locked_frame(frame: np.ndarray, grid: Grid, result: FrameResult,
                  marks: list[Mark], balls: list[Ball]) -> np.ndarray:
    """The live view with the locked grid and whatever was last detected."""
    shot = FrameResult(homography=grid.homography, holes=grid.holes,
                       view=result.view, image=frame)
    canvas = draw_overlay(shot, grid, marks=marks, balls=balls)
    # Not "{:.0f} x {:.0f} mm": with no --reference-mm the spec is normalised,
    # and that rounded a 1.000 x 0.608 surface to a square "1 x 1 mm".
    _banner(canvas, f"LOCKED {size_label()} - press B to find the balls",
            (0, 170, 0))
    status = ball_summary(balls) if balls else f"marks: {len(marks)}"
    _hint(canvas, f"{status}    B = balls    V = Baseten    D = marks    "
                  "R = re-lock    q = quit", y=62)
    # The one thing that cannot be read off the picture: that the counts
    # themselves are impossible, so nothing else on screen means what it says.
    if describe_inconsistency(balls, _BALL_RADIUS_FRAC):
        _hint(canvas, f"{len(balls)} balls - a set has 16: --ball-radius-frac "
                      f"({_BALL_RADIUS_FRAC:.4f}) is too small for this table",
              y=88, colour=(60, 60, 255))
    return canvas


def _preview_frame(frame: np.ndarray) -> tuple[np.ndarray, Grid | None]:
    """Live feedback, plus the grid it implies (None when nothing is found).

    Returns the grid as well as the picture so the companion x,y window can
    show the same measurement without running detection a second time.
    """
    from companion.pool.perception.vision.calibration import process_frame

    measure_sheet([frame], *_LIVE_REFERENCE)
    result = process_frame(frame)
    if result is None:
        canvas = frame.copy()
        _banner(canvas, f"{active_surface()} not found", (0, 0, 255))
        # The specific reason, wrapped: "too little contrast" and "too far
        # away" call for opposite fixes, so naming which one matters.
        reason = describe_failure(frame)
        for i, line in enumerate(_wrap(reason, 58)[:3]):
            cv2.putText(canvas, line, (12, 62 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3,
                        cv2.LINE_AA)
            cv2.putText(canvas, line, (12, 62 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
        _hint(canvas, "q = quit", y=140)
        return canvas, None

    grid = Grid(corners={}, holes=result.holes, homography=result.homography,
                frames_used=1, frames_attempted=1, timestamp="")
    canvas = draw_overlay(result, grid)
    found = sum(h.found for h in result.holes)
    confidence = float(_LAST_MEASUREMENT.get("confidence") or 0.0)

    if found < 6:
        _banner(canvas, f"{found}/6 pockets in view - move the camera back",
                (0, 140, 220))
    elif confidence < MIN_SHAPE_CONFIDENCE:
        # The shape, not the detection, is what is uncertain. Name the fix
        # precisely: the usual cause is tilt about one axis only, where one
        # pair of edges stays parallel in the image and the geometry cannot
        # be solved however steep that single tilt is.
        _banner(canvas, "MOVE THE CAMERA TO ONE SIDE - both edge pairs must "
                        "converge", (0, 140, 220))
    else:
        _banner(canvas, "ready - hold still to lock", (0, 170, 0))

    bar = f"shape confidence {confidence:.0%}"
    focal = _LAST_MEASUREMENT.get("focal")
    if focal:
        bar += f"   focal {focal:.0f}px"
    _hint(canvas, bar, y=62)
    _hint(canvas, "q = quit", y=88)
    return canvas, grid


def _graph_frame(grid: Grid | None) -> np.ndarray:
    """The x,y window: the measured rectangle, or a placeholder until there is one."""
    if grid is not None:
        return draw_xy_graph(grid)
    canvas = np.full((460, 620, 3), 250, np.uint8)
    cv2.putText(canvas, "waiting for the paper...", (150, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1, cv2.LINE_AA)
    return canvas


def _wrap(text: str, width: int) -> list[str]:
    """Greedy word wrap, so a long diagnosis still fits the window."""
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _banner(canvas: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    """Status line at the top, outlined so it reads over any background."""
    org = (12, 32)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4,
                cv2.LINE_AA)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2,
                cv2.LINE_AA)


def _hint(canvas: np.ndarray, text: str, y: int = 60,
          colour: tuple[int, int, int] = (255, 255, 255)) -> None:
    org = (12, y)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4,
                cv2.LINE_AA)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1,
                cv2.LINE_AA)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Measure a pool table's playing surface and the balls on "
                    "it. With no arguments this opens a live preview: aim the "
                    "camera and hold still while the geometry locks, then "
                    "press B to find and classify the balls.")
    ap.add_argument("--image", type=Path,
                    help="calibrate from a saved image instead of the camera")
    ap.add_argument("--no-preview", action="store_true",
                    help="skip the live window: capture and export immediately")
    ap.add_argument("--camera", type=int, default=None,
                    help="camera index (default: auto-select, preferring an "
                         "external USB webcam)")
    ap.add_argument("--list-cameras", action="store_true",
                    help="list the cameras found, then exit")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                    help="frames to median over when locking")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--require-all-holes", action="store_true",
                    help="fail instead of reporting a hole as missing")
    ap.add_argument("--reference-mm", type=float, default=None, dest="reference",
                    help="OPTIONAL. One real dimension of the surface, in mm, "
                         "measured with a ruler. Without it the tool reports "
                         "normalised units (long side = 1.0) and claims no "
                         "absolute size, which is all a single camera can "
                         "honestly support.")
    ap.add_argument("--height-mm", type=float, default=None,
                    help="the OTHER dimension, also measured with a ruler. "
                         "Give this when the camera looks nearly straight "
                         "down: the shape then cannot be recovered from "
                         "perspective, and two measurements are exact where "
                         "one plus a guess is not.")
    ap.add_argument("--surface", choices=SURFACES, default="table",
                    help="what to look for: 'table' finds the cloth inside "
                         "the cushions by colour, 'paper' finds a bright "
                         "sheet on a darker desk (default: table)")
    ap.add_argument("--reference-side", choices=("long", "width", "height"),
                    default=DEFAULT_REFERENCE_SIDE,
                    help="which side --reference-mm refers to (default: the "
                         "long one)")
    ap.add_argument("--ball-radius-frac", type=float,
                    default=BALL_RADIUS_FRAC,
                    help="a ball's radius as a fraction of the playing "
                         f"surface's long side (default {BALL_RADIUS_FRAC}). "
                         "Regulation tables use a 57.15 mm ball, so this is "
                         "0.0144 on a 7 ft playfield, 0.0128 on an 8 ft and "
                         "0.0113 on a 9 ft.")
    ap.add_argument("--vlm", action="store_true",
                    help="classify balls with Baseten after OpenCV finds "
                         "them. Slow (seconds). Live preview: press V, not B.")
    args = ap.parse_args(argv)
    load_dotenv()

    if args.list_cameras:
        return list_cameras()

    set_active_surface(args.surface)

    global _LIVE_REFERENCE, _BALL_RADIUS_FRAC, _USE_VLM
    _LIVE_REFERENCE = (args.reference, args.reference_side,
                       args.height_mm)
    if args.ball_radius_frac <= 0.0:
        print("error: --ball-radius-frac must be positive", file=sys.stderr)
        return 2
    _BALL_RADIUS_FRAC = args.ball_radius_frac
    _USE_VLM = bool(args.vlm)
    if _USE_VLM:
        os.environ["BASETEN_CLASSIFY"] = "1"

    if args.image:
        return run_image(args.image, args.outdir, args.require_all_holes,
                         args.reference, args.reference_side, args.height_mm)

    config = CameraConfig(index=args.camera, width=args.width,
                          height=args.height)
    if args.no_preview:
        return run_camera(config, args.frames, args.outdir,
                          args.require_all_holes, args.reference,
                          args.reference_side, args.height_mm)
    return run_preview(config, args.frames, args.outdir,
                       args.require_all_holes, args.reference,
                       args.reference_side, args.height_mm)


def list_cameras() -> int:
    """Print the camera devices found, so a wrong pick is easy to diagnose."""
    from companion.pool.perception.vision.camera import list_device_names

    names = list_device_names()
    if not names:
        print("no camera device names available on this platform")
    for i, name in enumerate(names):
        print(f"  index {i}: {name}")
    try:
        index, name = find_camera_index()
    except CameraError as exc:
        print(f"\nauto-select failed: {exc}", file=sys.stderr)
        return 1
    print(f"\nauto-select picks index {index}: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
