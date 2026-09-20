"""Find dark marks lying on the measured sheet and report them in table mm.

The sheet's geometry is already known by the time this runs (see
`geometry/calibrate.py`), so the search is deliberately confined to the sheet's
interior and phrased in millimetres rather than pixels. That confinement is
what makes the detector simple: anything outside the paper - the desk, a
cable, the dark board the sheet lies on - is excluded by construction rather
than by trying to describe it.

The marks are scraps of black paper, so the signal is straightforward: they
are much darker than the page they sit on. The threshold is taken from the
page's own brightness rather than fixed, because "dark" only means anything
relative to the paper in this particular frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from companion.pool.perception.vision.holes import (
    RECT_LONG_SIDE_PX, RectifiedView, rectify)
from companion.pool.perception.vision.homography import Homography
from companion.pool.perception.vision.table_spec import sheet_h_mm, sheet_w_mm

# Marks are reported as discs of this fixed radius, as specified: the scraps
# are irregular, and a common radius keeps the plotted grid readable.
MARK_RADIUS_MM = 5.0

# Size limits in millimetres, not pixels, so they hold at any camera distance.
MIN_MARK_AREA_MM2 = 12.0    # ~4 mm across; smaller is dirt or a pen dot
MAX_MARK_AREA_MM2 = 2000.0  # ~45 mm across; larger is a shadow, not a scrap

# How far inside the sheet's edge a mark must lie. The rectified view carries
# the contact shadow and a few millimetres of the darker board just outside
# the paper, and both are darker than the page - exactly what the threshold
# looks for. Ignoring a border band removes them without any special case.
EDGE_MARGIN_MM = 4.0

MIN_FILL = 0.35       # blob area over its bounding rect; rules out thin smears
MAX_ASPECT = 6.0      # a long thin dark streak is a shadow or a cable

# The morphology kernel is a fraction of the surface's long side, not an
# absolute distance. The rectified view is scaled to the surface (see
# `detect_holes.rect_scale`), so "1.5 mm in pixels" means nothing once the
# spec is normalised: at long side = 1.0 it asks for a 1351 px kernel and
# takes a minute and a half. 0.5% of the long side is the same 5 px this was
# tuned to on a 297 mm sheet, and stays 5 px at every other scale.
MARK_MORPH_FRAC = 0.005


@dataclass
class Mark:
    """A dark scrap on the sheet, in table millimetres."""

    id: str
    xy_mm: tuple[float, float]
    radius_mm: float
    area_mm2: float
    px: tuple[float, float] | None = None

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "xy_mm": [round(v, 2) for v in self.xy_mm],
            "radius_mm": round(self.radius_mm, 2),
            "area_mm2": round(self.area_mm2, 1),
            "px": None if self.px is None else [round(v, 2) for v in self.px],
        }


def detect_marks(image: np.ndarray, homography: Homography, *,
                 radius_mm: float = MARK_RADIUS_MM,
                 max_marks: int | None = None
                 ) -> tuple[list[Mark], RectifiedView]:
    """Dark marks on the sheet, ordered by area (largest first).

    Detection runs in the rectified top-down view rather than the camera
    frame. Warping first means a square-millimetre is the same number of
    pixels everywhere, so the size limits above are meaningful at any camera
    angle; done in the raw frame, a scrap at the far end of a tilted sheet
    would measure several times smaller than the same scrap near the camera.
    """
    view = rectify(image, homography)
    gray = (view.image if view.image.ndim == 2
            else cv2.cvtColor(view.image, cv2.COLOR_BGR2GRAY))

    page = _page_mask(view)
    if cv2.countNonZero(page) < 100:
        return [], view

    mask = _dark_mask(gray, page)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    px_per_mm2 = view.px_per_mm ** 2
    found: list[Mark] = []
    for contour in contours:
        area_mm2 = float(cv2.contourArea(contour)) / px_per_mm2
        if not MIN_MARK_AREA_MM2 <= area_mm2 <= MAX_MARK_AREA_MM2:
            continue
        if not _is_blob(contour):
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        x_mm, y_mm = view.px_to_mm(cx, cy)
        found.append(Mark(id="", xy_mm=(x_mm, y_mm), radius_mm=radius_mm,
                          area_mm2=area_mm2))

    found.sort(key=lambda m: m.area_mm2, reverse=True)
    if max_marks is not None:
        found = found[:max_marks]

    # Number them in reading order - down the page, then across - so the same
    # physical scrap keeps the same label between runs.
    found.sort(key=lambda m: (m.xy_mm[1], m.xy_mm[0]))
    marks: list[Mark] = []
    for i, mark in enumerate(found, start=1):
        px = homography.to_image(np.array([mark.xy_mm], np.float64))[0]
        marks.append(Mark(id=f"M{i}", xy_mm=mark.xy_mm,
                          radius_mm=mark.radius_mm, area_mm2=mark.area_mm2,
                          px=(float(px[0]), float(px[1]))))
    return marks, view


def _page_mask(view: RectifiedView) -> np.ndarray:
    """The sheet's interior in the rectified view, inset from its edges."""
    mask = np.zeros(view.image.shape[:2], np.uint8)
    x0, y0 = view.mm_to_px(EDGE_MARGIN_MM, EDGE_MARGIN_MM)
    x1, y1 = view.mm_to_px(sheet_w_mm() - EDGE_MARGIN_MM,
                           sheet_h_mm() - EDGE_MARGIN_MM)
    cv2.rectangle(mask, (int(round(x0)), int(round(y0))),
                  (int(round(x1)), int(round(y1))), 255, -1)
    return mask


def _dark_mask(gray: np.ndarray, page: np.ndarray) -> np.ndarray:
    """Pixels markedly darker than the page around them.

    The cut is taken from the page's own brightness distribution. A fixed
    grey level cannot work: the same scrap on the same paper reads 40 under a
    desk lamp and 150 by a window, while the *gap* between black paper and
    white paper stays large in both. The median stands in for the page
    because the scraps occupy only a small part of it, and the 10th
    percentile bounds the cut so a sheet with no marks at all does not have
    its own shading thresholded into phantom blobs.
    """
    values = gray[page > 0]
    page_level = float(np.median(values))
    floor = float(np.percentile(values, 10))

    # Two-thirds of the way from the page down to black, but never above the
    # page's own darkest tenth - shading and the printed grid live there.
    cut = min(page_level * 0.55, floor - 5.0)
    if cut <= 0.0:
        return np.zeros_like(gray)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, cut, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.bitwise_and(mask, page)

    # Close pinholes inside a scrap, then open to drop speckle. Sized as a
    # fraction of the view so both hold at any surface size or unit system.
    k = max(3, int(round(MARK_MORPH_FRAC * RECT_LONG_SIDE_PX)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)



def _is_blob(contour: np.ndarray) -> bool:
    """True if the contour is compact enough to be a scrap of paper.

    Shadows and cable edges survive thresholding as long thin regions; a
    scrap of paper, however irregularly torn, fills a good share of its own
    bounding box and is not many times longer than it is wide.
    """
    _, (w, h), _ = cv2.minAreaRect(contour)
    if w < 1e-6 or h < 1e-6:
        return False
    if max(w, h) / min(w, h) > MAX_ASPECT:
        return False
    return float(cv2.contourArea(contour)) / (w * h) >= MIN_FILL
