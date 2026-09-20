"""Camera captures to a physical table state, via the OpenCV pipeline.

The detection itself lives in `vision/`, ported from the standalone `htn26`
project. It works in whatever units its active `TableSpec` carries, with the
origin at the top-left playing corner, x across the spec's width and y down
its height - and it puts the *long* side on width, exactly as this contract
puts the long side on x. So installing the caller's geometry as the spec
makes the two frames the same frame, and no axis swap or rescale is needed
on the way out.

That agreement is load-bearing rather than incidental: `ball_radius_mm`
derives the search radius from the spec's long side, so handing the pipeline
a transposed spec does not merely relabel the axes, it changes what size
disc it looks for and the detection itself degrades. `_check_spec` below
fails loudly rather than letting that happen quietly.
"""

from pathlib import Path

import cv2
import numpy as np

from companion.pool.contracts import (
    Ball, BallType, CoverageStatus, NeedsMoreViews, PerceptionReady,
    PerceptionResult, Point2, TableGeometry, TableState, UnusableCapture)
from companion.pool.perception.vision import surface, table_spec
from companion.pool.perception.vision.calibration import (
    CalibrationError, calibrate_from_frames)
from companion.pool.perception.vision.detector import detect_balls
from companion.sensors.models import CaptureBatch

# The detector names its kinds with the same words the contract uses, but as
# plain strings. Mapping them explicitly means an unrecognised kind becomes
# UNKNOWN rather than an AttributeError at the boundary.
_KINDS = {
    "cue": BallType.CUE,
    "eight": BallType.EIGHT,
    "stripe": BallType.STRIPE,
    "solid": BallType.SOLID,
}


class PerceptionService:
    """Implementation owned by teammate 1. See docs/team-guide.md."""

    def estimate(self, captures: CaptureBatch, geometry: TableGeometry) -> PerceptionResult:
        bad = _check_spec(geometry)
        if bad is not None:
            return UnusableCapture(reason=bad)

        frames = []
        for capture in captures.captures:
            frame = _read(capture.rgb_path)
            if frame is None:
                return UnusableCapture(
                    reason=f"Capture {capture.capture_id} could not be read as an image")
            frames.append(frame)

        # The pipeline measures shape from the view; the caller's geometry
        # supplies the scale. `length` is the long side and belongs on the
        # spec's width, which is the axis the pipeline calls x - the same
        # axis the contract calls x. See the module docstring.
        table_spec.set_active_spec(table_spec.TableSpec(
            width_mm=geometry.length, height_mm=geometry.width, units="u"))
        surface.set_active_surface("table")

        try:
            grid, _ = calibrate_from_frames(frames)
        except CalibrationError as error:
            # The cloth was not found, or not found consistently enough to
            # measure. That is a property of the captures, not of the table.
            return NeedsMoreViews(reason=str(error))

        found, _ = detect_balls(frames[-1], grid.homography)
        balls = tuple(_to_ball(b, geometry) for b in found)
        balls = tuple(b for b in balls if b is not None)

        started = min(c.captured_at_s for c in captures.captures)
        ended = max(c.captured_at_s for c in captures.captures)
        state = TableState(
            observation_id=captures.batch_id,
            capture_started_at_s=started,
            capture_ended_at_s=ended,
            geometry=geometry,
            balls=balls,
            coverage=CoverageStatus.COMPLETE,
        )
        return PerceptionReady(state=state)


def _check_spec(geometry: TableGeometry) -> str | None:
    """Why this geometry cannot drive the pipeline, if it cannot.

    The contract already guarantees length 1.0 and width in (0, 1], so this
    only has to catch the degenerate end of that range: a table so nearly
    square that "long side" stops being meaningful, or a ball radius that
    leaves no room to search. Both would otherwise surface as a detector
    that quietly finds the wrong number of balls.
    """
    if geometry.width > geometry.length:
        return ("Table width exceeds its length; the contract puts the long "
                "side on x, so this geometry is transposed")
    if geometry.ball_radius >= 0.25 * geometry.width:
        return ("Ball radius is too large a fraction of the table for the "
                "detector's search to be meaningful")
    return None


def _read(path: str) -> np.ndarray | None:
    """One capture as BGR pixels, or None if it is not a readable image."""
    image = cv2.imread(str(Path(path)))
    return image if image is not None and image.size else None


def _to_ball(ball, geometry: TableGeometry) -> Ball | None:
    """One detected ball in contract terms, or None if it fell outside.

    The spec installed in `estimate` already put the pipeline in the
    contract's own frame and units, so the position passes straight through.

    A centre a hair outside the cloth is clamped rather than dropped: the
    contract rejects the whole state for one out-of-bounds ball, and a ball
    resting against a cushion can measure a fraction of a radius past the
    edge. Anything further out than that is a detection error and is dropped.
    """
    x, y = ball.xy_mm
    slack = geometry.ball_radius
    if not (-slack <= x <= geometry.length + slack
            and -slack <= y <= geometry.width + slack):
        return None
    x = min(max(x, 0.0), geometry.length)
    y = min(max(y, 0.0), geometry.width)
    return Ball(
        id=ball.id,
        position=Point2(x, y),
        type=_KINDS.get(ball.kind, BallType.UNKNOWN),
        type_confidence=ball.confidence,
    )
