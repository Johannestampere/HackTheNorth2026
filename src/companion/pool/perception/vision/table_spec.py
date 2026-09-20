"""The table geometry the pipeline measures against.

Nothing is printed on the sheet, so its own size sets the millimetre scale.
That size is *measured*, not assumed: the shape comes from the image (see
`geometry/aspect.py`, which recovers a rectangle's true proportions from perspective)
and the absolute scale from one real dimension the user supplies with a ruler.

Table coordinate system
-----------------------
Origin at the sheet's top-left corner, x across the width, y down the height,
units mm. "Top-left" is resolved geometrically (see `order_quad`), not by
which corner happens to appear first in a contour.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CORNER_IDS = ("TL", "TR", "BR", "BL")
HOLE_IDS = ("TL", "TM", "TR", "BL", "BM", "BR")


@dataclass(frozen=True)
class TableSpec:
    """A rectangular playing surface, in whatever units `units` names.

    The field names keep the `_mm` suffix because every call site uses them,
    but the numbers are only millimetres when `units == "mm"` - that is, when
    the user supplied a real measurement. Otherwise they are normalised
    units with the long side at 1.0, and `units` says so.
    """

    width_mm: float
    height_mm: float
    units: str = "u"

    @property
    def corners_mm(self) -> dict[str, tuple[float, float]]:
        return {
            "TL": (0.0, 0.0),
            "TR": (self.width_mm, 0.0),
            "BR": (self.width_mm, self.height_mm),
            "BL": (0.0, self.height_mm),
        }

    @property
    def corner_quad_mm(self) -> np.ndarray:
        return np.array([self.corners_mm[c] for c in CORNER_IDS], np.float64)

    @property
    def holes_mm(self) -> dict[str, tuple[float, float]]:
        """Pockets: the 4 corners plus the midpoints of the two long sides.

        Which sides are "long" follows the measured shape rather than an
        assumed orientation, so the middle pockets land where a pool table's
        do whichever way the sheet is lying.
        """
        w, h = self.width_mm, self.height_mm
        if h >= w:  # portrait: middle pockets on the left and right edges
            return {
                "TL": (0.0, 0.0),
                "TM": (0.0, h / 2.0),
                "TR": (w, 0.0),
                "BL": (0.0, h),
                "BM": (w, h / 2.0),
                "BR": (w, h),
            }
        return {  # landscape: middle pockets on the top and bottom edges
            "TL": (0.0, 0.0),
            "TM": (w / 2.0, 0.0),
            "TR": (w, 0.0),
            "BL": (0.0, h),
            "BM": (w / 2.0, h),
            "BR": (w, h),
        }

    def describe(self) -> str:
        if self.units == "mm":
            return f"{self.width_mm:.1f} x {self.height_mm:.1f} mm"
        # Normalised: more decimals, because the numbers are around 1.
        return (f"{self.width_mm:.3f} x {self.height_mm:.3f} "
                f"(normalised, long side = 1)")

    @property
    def aspect(self) -> float:
        """Short side over long side - the one thing a camera can measure."""
        lo, hi = sorted((self.width_mm, self.height_mm))
        return lo / hi if hi else 1.0


# The spec in force for this run. `main.py` replaces it once the sheet has
# been measured; the default is only a placeholder so the modules import
# cleanly, and every code path that matters takes the spec explicitly.
# Absolute size is not observable from one camera. A 140 mm table at 900 mm
# and a 1400 mm table at 9000 mm project to pixel-identical images, so any
# millimetre figure the tool printed without being told one would be an
# invention. The default is therefore normalised: the long side is 1.0 and
# every coordinate is a fraction of it. Pass a real measurement to get real
# units - see `--reference`.
DEFAULT_REFERENCE = 1.0
DEFAULT_REFERENCE_SIDE = "long"
DEFAULT_UNITS = "u"          # "u" for unit; mm only when the user supplies it

# Kept for callers that still name the old constants.
DEFAULT_REFERENCE_MM = DEFAULT_REFERENCE

_active = TableSpec(1.0, 0.5, "u")


def active_spec() -> TableSpec:
    return _active


def set_active_spec(spec: TableSpec) -> None:
    """Install the measured sheet size for the rest of the run."""
    global _active
    _active = spec


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL regardless of how they were found.

    Sorts by angle about the centroid, which is robust to the perspective
    skew of a tilted view, then rotates so the corner nearest the image
    origin comes first.
    """
    pts = np.asarray(pts, np.float64).reshape(4, 2)
    centre = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    clockwise = pts[np.argsort(angles)]
    start = int(np.argmin(clockwise.sum(axis=1)))  # smallest x+y is top-left
    return np.roll(clockwise, -start, axis=0)


# --- Backwards-compatible module-level views of the active spec -------------
# These keep the call sites readable. They are functions, not constants,
# because the size is not known until the sheet has been measured.

def sheet_w_mm() -> float:
    return _active.width_mm


def sheet_h_mm() -> float:
    return _active.height_mm


def corners_mm() -> dict[str, tuple[float, float]]:
    return _active.corners_mm


def corner_quad_mm() -> np.ndarray:
    return _active.corner_quad_mm


def holes_mm() -> dict[str, tuple[float, float]]:
    return _active.holes_mm


def aspect_ratio() -> float:
    """Long side / short side, for rejecting wrong-shaped quads."""
    return max(_active.width_mm, _active.height_mm) / min(_active.width_mm,
                                                          _active.height_mm)
