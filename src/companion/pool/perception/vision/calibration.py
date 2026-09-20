"""Run the pipeline over a burst of frames and lock a median grid."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import cv2
import numpy as np

from companion.pool.perception.vision.holes import (
    Hole, RectifiedView, detect_holes)
from companion.pool.perception.vision.surface import (
    active_surface, describe_failure, detect_reference_points)
from companion.pool.perception.vision.homography import (
    Homography, fit_homography)
from companion.pool.perception.vision.table_spec import (
    CORNER_IDS, HOLE_IDS, active_spec, corners_mm, holes_mm, sheet_h_mm,
    sheet_w_mm)

DEFAULT_FRAMES = 30


class CalibrationError(RuntimeError):
    """Calibration could not produce a grid."""


@dataclass
class FrameResult:
    homography: Homography
    holes: list[Hole]
    view: RectifiedView
    image: np.ndarray


@dataclass
class Grid:
    """The locked result: table geometry in mm, plus how it was obtained."""

    corners: dict[str, tuple[float, float]]
    holes: list[Hole]
    homography: Homography
    frames_used: int
    frames_attempted: int
    timestamp: str
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        corner_px = self.homography.to_image(
            np.array([corners_mm()[c] for c in CORNER_IDS], np.float64))
        return {
            "units": active_spec().units,
            "origin": f"{active_surface()} top-left corner",
            "axes": f"x across the {sheet_w_mm():.3g} side, y down the "
                    f"{sheet_h_mm():.3g} side",
            "frame": {"width": round(sheet_w_mm(), 4),
                      "height": round(sheet_h_mm(), 4),
                      "aspect": round(active_spec().aspect, 4)},
            "corners": [
                {
                    "id": cid,
                    "xy_mm": [round(v, 2) for v in corners_mm()[cid]],
                    "px": [round(float(corner_px[i][0]), 2),
                           round(float(corner_px[i][1]), 2)],
                }
                for i, cid in enumerate(CORNER_IDS)
            ],
            "holes": [h.as_dict() for h in self.holes],
            "homography": [[float(v) for v in row] for row in self.homography.H],
            "reprojection_error_mm": round(
                self.homography.reprojection_error_mm, 4),
            "frames_used": self.frames_used,
            "frames_attempted": self.frames_attempted,
            "timestamp": self.timestamp,
            "notes": self.notes,
        }


def process_frame(image: np.ndarray) -> FrameResult | None:
    """Run detection -> homography -> holes on one frame, or None if unusable."""
    reference = detect_reference_points(image)
    if reference is None:
        return None
    image_pts, table_pts = reference
    homography = fit_homography(image_pts, table_pts)
    holes, view = detect_holes(image, homography)
    return FrameResult(homography=homography, holes=holes, view=view,
                       image=image)


def _median_holes(results: list[FrameResult]) -> list[Hole]:
    """Per-hole median over the frames in which that hole was found.

    A hole found in no frame stays missing rather than falling back to its
    nominal position - a silently substituted nominal would look like a
    perfect detection downstream.
    """
    holes: list[Hole] = []
    for hole_id in HOLE_IDS:
        nominal = holes_mm()[hole_id]
        seen = [h for r in results for h in r.holes
                if h.id == hole_id and h.found and h.xy_mm is not None]
        if not seen:
            holes.append(Hole(id=hole_id, found=False, nominal_xy_mm=nominal))
            continue
        xs = np.array([h.xy_mm[0] for h in seen])
        ys = np.array([h.xy_mm[1] for h in seen])
        x_mm, y_mm = float(np.median(xs)), float(np.median(ys))
        pxs = np.array([h.px for h in seen if h.px is not None])
        px = (float(np.median(pxs[:, 0])), float(np.median(pxs[:, 1])))
        holes.append(Hole(
            id=hole_id,
            found=True,
            nominal_xy_mm=nominal,
            xy_mm=(x_mm, y_mm),
            px=px,
            error_mm=float(np.hypot(x_mm - nominal[0], y_mm - nominal[1])),
        ))
    return holes


def _median_homography(results: list[FrameResult]) -> Homography:
    """Element-wise median of the per-frame homographies, normalised.

    Each H is scaled to H[2,2] = 1 first so the medians are taken over
    comparable numbers rather than arbitrary projective scales.
    """
    stack = np.stack([r.homography.H / r.homography.H[2, 2] for r in results])
    H = np.median(stack, axis=0)
    H /= H[2, 2]
    errors = [r.homography.reprojection_error_mm for r in results]
    return Homography(
        H=H,
        H_inv=np.linalg.inv(H),
        reprojection_error_mm=float(np.median(errors)),
        inliers=int(np.median([r.homography.inliers for r in results])),
        total_points=results[0].homography.total_points,
    )


def calibrate_from_frames(frames: list[np.ndarray], *,
                          require_all_holes: bool = False) -> tuple[Grid, FrameResult]:
    """Calibrate from already-captured frames.

    Returns the locked grid plus the last usable frame result, which the
    caller uses to draw the overlay and rectified images.
    """
    if not frames:
        raise CalibrationError("no frames to calibrate from")

    results: list[FrameResult] = []
    for frame in frames:
        result = process_frame(frame)
        if result is not None:
            results.append(result)

    if not results:
        raise CalibrationError(
            f"could not find the {active_surface()} in any of "
            f"{len(frames)} frames: "
            f"{describe_failure(frames[-1])}")

    notes: list[str] = []
    if len(results) < len(frames):
        notes.append(f"{len(frames) - len(results)} of {len(frames)} "
                     f"frames had no usable {active_surface()} quad and "
                     "were skipped")

    homography = _median_homography(results)
    holes = _median_holes(results)
    missing_holes = [h.id for h in holes if not h.found]
    if missing_holes:
        note = f"holes not found: {', '.join(missing_holes)}"
        if require_all_holes:
            raise CalibrationError(note)
        notes.append(note)

    grid = Grid(
        corners={cid: corners_mm()[cid] for cid in CORNER_IDS},
        holes=holes,
        homography=homography,
        frames_used=len(results),
        frames_attempted=len(frames),
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        notes=notes,
    )
    return grid, results[-1]


def calibrate_from_camera(camera, *, frames: int = DEFAULT_FRAMES,
                          require_all_holes: bool = False
                          ) -> tuple[Grid, FrameResult]:
    """Grab consecutive frames from an open Camera and calibrate."""
    captured = [camera.read() for _ in range(frames)]
    return calibrate_from_frames(captured, require_all_holes=require_all_holes)


def calibrate_from_image(image: np.ndarray, *, frames: int = 1,
                         require_all_holes: bool = False
                         ) -> tuple[Grid, FrameResult]:
    """Calibrate from a single saved image (offline testing path)."""
    grid, result = calibrate_from_frames([image],
                                         require_all_holes=require_all_holes)
    return grid, result
