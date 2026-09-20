"""Drawing helpers for the exported overlay and rectified images."""

from __future__ import annotations

import cv2
import numpy as np

from companion.pool.perception.vision.calibration import FrameResult, Grid
from companion.pool.perception.vision.detector import KINDS, Ball
from companion.pool.perception.vision.holes import RectifiedView
from companion.pool.perception.vision.marks import MARK_RADIUS_MM, Mark
from companion.pool.perception.vision.table_spec import (CORNER_IDS, active_spec, corners_mm,
                                 sheet_h_mm, sheet_w_mm)

GREEN = (0, 200, 0)
CYAN = (255, 200, 0)
RED = (0, 0, 255)
MAGENTA = (200, 0, 200)
MARK_COLOUR = (60, 60, 60)        # the black scraps, on the white graph
MARK_COLOUR_BGR = (0, 210, 255)   # and on the photo, where dark vanishes
FONT = cv2.FONT_HERSHEY_SIMPLEX

# The four kinds, drawn as the game draws them rather than as a legend of
# arbitrary colours: a solid is filled, a stripe is the same colour with a
# white band across it, the cue is white and the 8 is black. The ball's own
# measured colour does the identifying, so two balls of the same colour are
# still told apart by the band.
CUE_FILL = (250, 250, 250)
EIGHT_FILL = (25, 25, 25)
BALL_EDGE = (20, 20, 20)
BALL_GRAPH_WIDTH = 1100

# On the photo the ball's own colour is already there, so a ring drawn in it
# disappears. These are for the overlay only: bright, and clear of the green
# outline, the magenta pockets and the azure grid, so the kind can be read
# off the picture without matching labels to balls.
KIND_MARK = {
    "cue": (255, 255, 255),
    "eight": (170, 170, 170),
    "stripe": (0, 255, 255),
    "solid": (0, 140, 255),
}

GRID_STEP_MM = 25.0  # spacing of the drawn xy grid, on a paper-sized surface


def _tick(value: float) -> str:
    """Axis tick text: whole numbers for mm, decimals when normalised."""
    return f"{value:.0f}" if active_spec().units == "mm" else f"{value:.2f}"


def size_label() -> str:
    """The surface's size with its units, however it was measured."""
    spec = active_spec()
    if spec.units == "mm":
        return f"{spec.width_mm:.0f} x {spec.height_mm:.0f} mm"
    return f"{spec.width_mm:.3f} x {spec.height_mm:.3f} (long side = 1)"


def grid_step_mm() -> float:
    """Grid spacing that keeps the drawn grid readable at any surface size.

    25 mm suits a sheet of paper but draws 40 lines across a pool table,
    which obscures the thing it is meant to annotate. Picking the step from
    the surface keeps roughly 8-12 divisions across the long side whatever
    is being measured.
    """
    longest = max(sheet_w_mm(), sheet_h_mm())
    for step in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5,
                 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 250.0, 500.0):
        if longest / step <= 12.0:
            return step
    return longest / 8.0


def draw_grid_lines(canvas: np.ndarray, grid: Grid, *,
                    step_mm: float | None = None) -> None:
    """Draw the table's xy grid, projected into the image.

    Each line is drawn as a polyline through several intermediate points
    rather than as a single segment: under perspective a straight table line
    is still straight in the image, but sampling keeps it correct even if a
    lens-distortion model is added to the homography later.
    """
    step_mm = grid_step_mm() if step_mm is None else step_mm

    def project(points: list[tuple[float, float]]) -> np.ndarray:
        return grid.homography.to_image(
            np.array(points, np.float64)).astype(np.int32).reshape(-1, 1, 2)

    xs = list(np.arange(0.0, sheet_w_mm() + 1e-6, step_mm))
    ys = list(np.arange(0.0, sheet_h_mm() + 1e-6, step_mm))

    for x in xs:
        line = project([(x, y) for y in np.linspace(0.0, sheet_h_mm(), 12)])
        cv2.polylines(canvas, [line], False, CYAN, 1, cv2.LINE_AA)
    for y in ys:
        line = project([(x, y) for x in np.linspace(0.0, sheet_w_mm(), 12)])
        cv2.polylines(canvas, [line], False, CYAN, 1, cv2.LINE_AA)

    # Label the axes in mm along the top and left edges.
    for x in xs:
        px = grid.homography.to_image(np.array([(x, 0.0)], np.float64))[0]
        cv2.putText(canvas, _tick(x), (int(px[0]) - 10, int(px[1]) - 6),
                    FONT, 0.4, CYAN, 1, cv2.LINE_AA)
    for y in ys:
        px = grid.homography.to_image(np.array([(0.0, y)], np.float64))[0]
        cv2.putText(canvas, _tick(y), (int(px[0]) - 34, int(px[1]) + 4),
                    FONT, 0.4, CYAN, 1, cv2.LINE_AA)


def draw_overlay(result: FrameResult, grid: Grid, *,
                 show_grid: bool = True,
                 marks: list[Mark] | None = None,
                 balls: list[Ball] | None = None) -> np.ndarray:
    """The camera frame with the paper outline, xy grid and pockets drawn."""
    canvas = result.image.copy()

    if show_grid:
        draw_grid_lines(canvas, grid)

    # Paper outline and its 4 corners.
    corner_px = grid.homography.to_image(
        np.array([corners_mm()[c] for c in CORNER_IDS], np.float64))
    cv2.polylines(canvas, [corner_px.astype(np.int32).reshape(-1, 1, 2)],
                  True, GREEN, 2, cv2.LINE_AA)
    for cid, (x, y) in zip(CORNER_IDS, corner_px):
        cv2.circle(canvas, (int(x), int(y)), 6, GREEN, -1, cv2.LINE_AA)
        label = (f"{cid} ({_tick(corners_mm()[cid][0])},"
                 f"{_tick(corners_mm()[cid][1])})")
        cv2.putText(canvas, label, (int(x) + 8, int(y) - 8), FONT, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (int(x) + 8, int(y) - 8), FONT, 0.5, GREEN,
                    1, cv2.LINE_AA)

    # Pockets.
    for hole in grid.holes:
        if hole.found and hole.px is not None:
            x, y = int(hole.px[0]), int(hole.px[1])
            cv2.circle(canvas, (x, y), 9, MAGENTA, 2, cv2.LINE_AA)
            cv2.drawMarker(canvas, (x, y), MAGENTA, cv2.MARKER_CROSS, 12, 1)
            cv2.putText(canvas, hole.id, (x + 12, y + 4), FONT, 0.5, (0, 0, 0),
                        3, cv2.LINE_AA)
            cv2.putText(canvas, hole.id, (x + 12, y + 4), FONT, 0.5, MAGENTA,
                        1, cv2.LINE_AA)
        else:
            px = grid.homography.to_image(
                np.array([hole.nominal_xy_mm], np.float64))[0]
            cv2.drawMarker(canvas, (int(px[0]), int(px[1])), RED,
                           cv2.MARKER_TILTED_CROSS, 14, 2)

    # Marks, drawn at their true radius by projecting a point r mm away
    # through the same homography: under perspective a disc on the table is
    # an ellipse in the image, and a fixed pixel radius would misreport how
    # big the scrap actually is.
    for mark in marks or []:
        if mark.px is None:
            continue
        centre = (int(round(mark.px[0])), int(round(mark.px[1])))
        rim = grid.homography.to_image(
            np.array([(mark.xy_mm[0] + mark.radius_mm, mark.xy_mm[1])],
                     np.float64))[0]
        radius = max(2, int(round(float(np.hypot(rim[0] - mark.px[0],
                                                 rim[1] - mark.px[1])))))
        cv2.circle(canvas, centre, radius, MARK_COLOUR_BGR, 2, cv2.LINE_AA)
        cv2.drawMarker(canvas, centre, MARK_COLOUR_BGR, cv2.MARKER_CROSS, 10, 1)
        label = f"{mark.id} ({mark.xy_mm[0]:.0f},{mark.xy_mm[1]:.0f})"
        cv2.putText(canvas, label, (centre[0] + radius + 4, centre[1] + 4),
                    FONT, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (centre[0] + radius + 4, centre[1] + 4),
                    FONT, 0.45, MARK_COLOUR_BGR, 1, cv2.LINE_AA)

    # Balls, ringed on the photo rather than filled: the point here is to
    # check the detector landed on the right thing, which means seeing the
    # ball underneath. The kind is named instead of being drawn.
    for ball in balls or []:
        if ball.px is None:
            continue
        centre = (int(round(ball.px[0])), int(round(ball.px[1])))
        rim = grid.homography.to_image(
            np.array([(ball.xy_mm[0] + ball.radius_mm, ball.xy_mm[1])],
                     np.float64))[0]
        radius = max(3, int(round(float(np.hypot(rim[0] - ball.px[0],
                                                 rim[1] - ball.px[1])))))
        colour = KIND_MARK.get(ball.kind, (0, 255, 255))
        cv2.circle(canvas, centre, radius, colour, 2, cv2.LINE_AA)
        label = f"{ball.id} {ball.kind}"
        if ball.number:
            label += f" {ball.number}"
        if ball.confidence < 0.6:
            label += "?"
        cv2.putText(canvas, label, (centre[0] + radius + 3, centre[1] + 4),
                    FONT, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (centre[0] + radius + 3, centre[1] + 4),
                    FONT, 0.45, colour, 1, cv2.LINE_AA)

    footer = (f"{size_label()}  |  "
              f"fit {grid.homography.reprojection_error_mm:.2f} mm  |  "
              f"frames {grid.frames_used}/{grid.frames_attempted}  |  "
              f"pockets {sum(h.found for h in grid.holes)}/6")
    if balls is not None:
        footer += f"  |  {ball_summary(balls)}"
    cv2.putText(canvas, footer, (10, canvas.shape[0] - 12), FONT, 0.55,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, footer, (10, canvas.shape[0] - 12), FONT, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def ball_fill(ball: Ball) -> tuple[int, int, int]:
    """The colour to draw this ball in."""
    if ball.kind == "cue":
        return CUE_FILL
    if ball.kind == "eight":
        return EIGHT_FILL
    return tuple(int(c) for c in ball.colour_bgr)


def draw_ball(canvas: np.ndarray, centre: tuple[int, int], radius: int,
              ball: Ball) -> None:
    """One ball, drawn the way the game distinguishes them.

    The white band on a stripe is clipped to the disc through a mask rather
    than drawn as a chord, so it reads as a band round a sphere at any size
    and cannot spill over the ball's edge.
    """
    radius = max(2, int(radius))
    fill = ball_fill(ball)
    cv2.circle(canvas, centre, radius, fill, -1, cv2.LINE_AA)

    if ball.kind == "stripe":
        x, y = centre
        x0, y0 = max(0, x - radius), max(0, y - radius)
        x1 = min(canvas.shape[1], x + radius + 1)
        y1 = min(canvas.shape[0], y + radius + 1)
        if x1 > x0 and y1 > y0:
            roi = canvas[y0:y1, x0:x1]
            disc = np.zeros(roi.shape[:2], np.uint8)
            cv2.circle(disc, (x - x0, y - y0), radius, 255, -1, cv2.LINE_AA)
            band = np.zeros_like(disc)
            half = max(1, int(round(0.42 * radius)))
            cv2.rectangle(band, (0, y - y0 - half), (x1 - x0, y - y0 + half),
                          255, -1)
            roi[(disc > 0) & (band > 0)] = CUE_FILL

    cv2.circle(canvas, centre, radius, BALL_EDGE, 1, cv2.LINE_AA)
    if ball.confidence < 0.6:
        # The call was close to its threshold; say so on the picture rather
        # than only in the JSON.
        cv2.circle(canvas, centre, radius + 3, (0, 140, 230), 1, cv2.LINE_AA)


def ball_counts(balls: list[Ball]) -> dict[str, int]:
    """How many of each kind, with every kind present even at zero."""
    return {kind: sum(b.kind == kind for b in balls) for kind in KINDS}


def ball_summary(balls: list[Ball]) -> str:
    """Counts by kind, for a footer."""
    counts = ball_counts(balls)
    return (f"cue {counts['cue']}  8 {counts['eight']}  "
            f"stripes {counts['stripe']}  solids {counts['solid']}")


def draw_xy_graph(grid: Grid, *, width: int | None = None,
                  step_mm: float | None = None,
                  marks: list[Mark] | None = None,
                  balls: list[Ball] | None = None) -> np.ndarray:
    """A clean x,y graph of the table: no photo, just the measured geometry.

    The companion to the camera overlay. The overlay answers "is it tracking
    the right thing"; this answers "what shape did we actually measure",
    which is hard to judge against a perspective view of a desk.
    """
    step_mm = grid_step_mm() if step_mm is None else step_mm
    # A ball is about 1.5% of the table's long side, so the default canvas
    # draws it 7 px across - too small to read a stripe's band, or to see
    # that two balls are touching. Widen when there are balls to show.
    if width is None:
        width = BALL_GRAPH_WIDTH if balls else 620
    margin = 62
    scale = (width - 2 * margin) / sheet_w_mm()
    height = int(round(sheet_h_mm() * scale)) + 2 * margin
    canvas = np.full((height, width, 3), 250, np.uint8)

    def to_px(x_mm: float, y_mm: float) -> tuple[int, int]:
        return (int(round(margin + x_mm * scale)),
                int(round(margin + y_mm * scale)))

    # Grid lines.
    for x in np.arange(0.0, sheet_w_mm() + 1e-6, step_mm):
        cv2.line(canvas, to_px(x, 0.0), to_px(x, sheet_h_mm()), (222, 222, 222),
                 1, cv2.LINE_AA)
    for y in np.arange(0.0, sheet_h_mm() + 1e-6, step_mm):
        cv2.line(canvas, to_px(0.0, y), to_px(sheet_w_mm(), y), (222, 222, 222),
                 1, cv2.LINE_AA)

    # The sheet itself: an exact rectangle, because that is what the table is
    # in table coordinates - any perspective has been divided out by now.
    cv2.rectangle(canvas, to_px(0.0, 0.0), to_px(sheet_w_mm(), sheet_h_mm()),
                  (40, 40, 40), 2, cv2.LINE_AA)

    # Axes: x across the top, y down the left, labelled in mm.
    for x in np.arange(0.0, sheet_w_mm() + 1e-6, step_mm * 2):
        px, py = to_px(x, 0.0)
        cv2.line(canvas, (px, py - 5), (px, py), (90, 90, 90), 1, cv2.LINE_AA)
        cv2.putText(canvas, _tick(x), (px - 10, py - 10), FONT, 0.38,
                    (90, 90, 90), 1, cv2.LINE_AA)
    for y in np.arange(0.0, sheet_h_mm() + 1e-6, step_mm * 2):
        px, py = to_px(0.0, y)
        cv2.line(canvas, (px - 5, py), (px, py), (90, 90, 90), 1, cv2.LINE_AA)
        cv2.putText(canvas, _tick(y), (px - 40, py + 4), FONT, 0.38,
                    (90, 90, 90), 1, cv2.LINE_AA)
    unit = active_spec().units
    cv2.putText(canvas, f"x ({unit})", (width // 2 - 22, 24), FONT, 0.45,
                (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"y ({unit})", (8, margin - 24), FONT, 0.45, (60, 60, 60),
                1, cv2.LINE_AA)

    # Corners, labelled with their coordinates.
    for cid in CORNER_IDS:
        x_mm, y_mm = corners_mm()[cid]
        px, py = to_px(x_mm, y_mm)
        cv2.circle(canvas, (px, py), 5, (40, 140, 40), -1, cv2.LINE_AA)
        dx = 10 if x_mm == 0.0 else -74
        dy = 18 if y_mm == 0.0 else -10
        cv2.putText(canvas, f"{cid} ({_tick(x_mm)},{_tick(y_mm)})",
                    (px + dx, py + dy),
                    FONT, 0.42, (40, 140, 40), 1, cv2.LINE_AA)

    # Pockets.
    for hole in grid.holes:
        px, py = to_px(*hole.nominal_xy_mm)
        colour = MAGENTA if hole.found else RED
        cv2.circle(canvas, (px, py), 9, colour, 2, cv2.LINE_AA)
        cv2.drawMarker(canvas, (px, py), colour, cv2.MARKER_CROSS, 10, 1)
        label = hole.id if hole.found else f"{hole.id} missing"
        cv2.putText(canvas, label, (px + 12, py + 4), FONT, 0.45, colour, 1,
                    cv2.LINE_AA)

    # The marks, as discs of their true radius: drawing them to scale is
    # what makes the graph a measurement rather than an illustration, so the
    # radius is converted through the same mm->px scale as everything else.
    for mark in marks or []:
        px, py = to_px(*mark.xy_mm)
        radius = max(2, int(round(mark.radius_mm * scale)))
        cv2.circle(canvas, (px, py), radius, MARK_COLOUR, -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), radius, (20, 20, 20), 1, cv2.LINE_AA)
        label = f"{mark.id} ({mark.xy_mm[0]:.0f},{mark.xy_mm[1]:.0f})"
        cv2.putText(canvas, label, (px + radius + 5, py + 4), FONT, 0.42,
                    (20, 20, 20), 1, cv2.LINE_AA)

    # The balls, at the one fixed radius every ball shares. Drawing them to
    # scale is what makes this a plan of the table rather than a scatter
    # plot: whether two balls are touching, or whether one blocks the line
    # to another, is only readable if the discs are the right size.
    for ball in balls or []:
        px, py = to_px(*ball.xy_mm)
        draw_ball(canvas, (px, py), int(round(ball.radius_mm * scale)), ball)
    for ball in balls or []:
        px, py = to_px(*ball.xy_mm)
        radius = max(2, int(round(ball.radius_mm * scale)))
        cv2.putText(canvas, ball.id, (px + radius + 3, py - radius + 2),
                    FONT, 0.36, (90, 90, 90), 1, cv2.LINE_AA)

    footer = (f"{size_label()}   "
              f"grid {step_mm:.3g}{active_spec().units}   "
              f"pockets {sum(h.found for h in grid.holes)}/6")
    if marks is not None:
        footer += f"   marks {len(marks)} (r={MARK_RADIUS_MM:.0f} mm)"
    if balls is not None:
        footer += f"   {ball_summary(balls)}"
    cv2.putText(canvas, footer, (margin, height - 18), FONT, 0.45,
                (60, 60, 60), 1, cv2.LINE_AA)
    return canvas


def draw_rectified(view: RectifiedView, grid: Grid) -> np.ndarray:
    """The top-down warped view with the grid and pockets drawn."""
    canvas = (view.image.copy() if view.image.ndim == 3
              else cv2.cvtColor(view.image, cv2.COLOR_GRAY2BGR))

    step = grid_step_mm()
    for x in np.arange(0.0, sheet_w_mm() + 1e-6, step):
        p0 = view.mm_to_px(x, 0.0)
        p1 = view.mm_to_px(x, sheet_h_mm())
        cv2.line(canvas, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])),
                 CYAN, 1, cv2.LINE_AA)
    for y in np.arange(0.0, sheet_h_mm() + 1e-6, step):
        p0 = view.mm_to_px(0.0, y)
        p1 = view.mm_to_px(sheet_w_mm(), y)
        cv2.line(canvas, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])),
                 CYAN, 1, cv2.LINE_AA)

    outline = np.array([view.mm_to_px(*corners_mm()[c]) for c in CORNER_IDS],
                       np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [outline], True, GREEN, 2, cv2.LINE_AA)

    for hole in grid.holes:
        x, y = view.mm_to_px(*hole.nominal_xy_mm)
        colour = MAGENTA if hole.found else RED
        cv2.circle(canvas, (int(x), int(y)), 8, colour, 2, cv2.LINE_AA)
        cv2.putText(canvas, hole.id, (int(x) + 10, int(y) + 4), FONT, 0.45,
                    colour, 1, cv2.LINE_AA)
    return canvas
