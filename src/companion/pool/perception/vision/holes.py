"""Pocket positions and the rectified top-down view.

Nothing is printed on the sheet, so pockets are not searched for: they are
defined by the table geometry and reported through the measured homography.
`found` therefore reflects whether the corner fit supports that position, not
whether a dark blob was seen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from companion.pool.perception.vision.homography import Homography
from companion.pool.perception.vision.table_spec import (
    HOLE_IDS, holes_mm, sheet_h_mm, sheet_w_mm)

# The rectified view is sized from the surface rather than in absolute
# millimetres. A fixed px-per-mm only ever suited a sheet of paper: at a real
# table's size it asks for a 17-megapixel warp, and in normalised units (long
# side = 1.0) it collapses the whole table into 3 px inside a margin ten times
# wider than the table itself. Scaling to a target resolution keeps the view
# usable whatever units are in force, as `render.overlay.draw_xy_graph` already does
# with its fixed canvas width.
RECT_LONG_SIDE_PX = 900.0  # pixels across the surface's long side
RECT_MARGIN_FRAC = 0.05    # border around it, as a fraction of that side


@dataclass
class Hole:
    id: str
    found: bool
    nominal_xy_mm: tuple[float, float]
    xy_mm: tuple[float, float] | None = None
    px: tuple[float, float] | None = None
    error_mm: float | None = None

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "found": self.found,
            "xy_mm": None if self.xy_mm is None
                     else [round(v, 3) for v in self.xy_mm],
            "px": None if self.px is None else [round(v, 2) for v in self.px],
            "nominal_xy_mm": list(self.nominal_xy_mm),
            "error_mm": None if self.error_mm is None
                        else round(self.error_mm, 3),
        }


@dataclass
class RectifiedView:
    """Top-down view of the table plane at a fixed scale.

    The warp is deferred until the pixels are actually asked for. A
    calibration burst builds one of these per frame but wants only the scale
    from all but one of them - `detect.holes` places the pockets from the
    homography alone - so warping every frame was work thrown away.
    """

    px_per_mm: float
    margin_mm: float
    width: int
    height: int
    _source: np.ndarray = field(repr=False)
    _warp: np.ndarray = field(repr=False)
    _image: np.ndarray | None = field(default=None, repr=False)

    @property
    def image(self) -> np.ndarray:
        if self._image is None:
            self._image = cv2.warpPerspective(
                self._source, self._warp, (self.width, self.height),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                borderValue=(30, 30, 30))
        return self._image

    def mm_to_px(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        return ((x_mm + self.margin_mm) * self.px_per_mm,
                (y_mm + self.margin_mm) * self.px_per_mm)

    def px_to_mm(self, x_px: float, y_px: float) -> tuple[float, float]:
        return (x_px / self.px_per_mm - self.margin_mm,
                y_px / self.px_per_mm - self.margin_mm)


def rect_scale() -> tuple[float, float]:
    """(px_per_mm, margin_mm) for the surface currently in force.

    Both are derived from the surface's long side, so the warped view comes
    out the same size on screen whether the spec is in millimetres or in
    normalised units.
    """
    longest = max(sheet_w_mm(), sheet_h_mm())
    if longest <= 0.0:
        raise ValueError("the active spec has no size to rectify against")
    return RECT_LONG_SIDE_PX / longest, RECT_MARGIN_FRAC * longest


def rectify(image: np.ndarray, homography: Homography, *,
            px_per_mm: float | None = None,
            margin_mm: float | None = None) -> RectifiedView:
    """Warp the camera frame into a top-down view of the table plane."""
    scale, margin = rect_scale()
    px_per_mm = scale if px_per_mm is None else px_per_mm
    margin_mm = margin if margin_mm is None else margin_mm
    w = int(round((sheet_w_mm() + 2 * margin_mm) * px_per_mm))
    h = int(round((sheet_h_mm() + 2 * margin_mm) * px_per_mm))
    S = np.array([[px_per_mm, 0.0, margin_mm * px_per_mm],
                  [0.0, px_per_mm, margin_mm * px_per_mm],
                  [0.0, 0.0, 1.0]], np.float64)
    return RectifiedView(px_per_mm=px_per_mm, margin_mm=margin_mm,
                         width=w, height=h,
                         _source=image, _warp=S @ homography.H)


def detect_holes(image: np.ndarray, homography: Homography, *,
                 px_per_mm: float | None = None
                 ) -> tuple[list[Hole], RectifiedView]:
    """Place the 6 pockets from the table geometry and project them to pixels.

    Each pocket's `error_mm` is the round trip through the fitted homography
    (table mm -> image px -> table mm), so it reports how much the corner fit
    actually distorts that position rather than a hardcoded zero.
    """
    view = rectify(image, homography, px_per_mm=px_per_mm)
    height, width = image.shape[:2]

    holes: list[Hole] = []
    for hole_id in HOLE_IDS:
        nominal = holes_mm()[hole_id]
        px = homography.to_image(np.array([nominal], np.float64))[0]
        back = homography.to_table(np.array([px], np.float64))[0]
        error = float(np.hypot(back[0] - nominal[0], back[1] - nominal[1]))

        # A pocket projecting outside the frame means the sheet is only
        # partly visible; report it missing rather than inventing a position.
        margin = 2.0
        inside = (-margin <= px[0] <= width + margin
                  and -margin <= px[1] <= height + margin)
        if not inside:
            holes.append(Hole(id=hole_id, found=False, nominal_xy_mm=nominal))
            continue

        holes.append(Hole(
            id=hole_id,
            found=True,
            nominal_xy_mm=nominal,
            xy_mm=(float(nominal[0]), float(nominal[1])),
            px=(float(px[0]), float(px[1])),
            error_mm=error,
        ))
    return holes, view
