"""Find the playing surface of a real pool table: the cloth inside the rails.

This is the table-sized counterpart of `detect/reference.py`, which finds a
sheet of paper. The two answer the same question - where are the four corners
of the playing surface - but from opposite evidence, so they are separate
modules rather than one function with a flag.

Paper is found by brightness: a white sheet on a darker desk. A pool table
inverts that (the cloth here reads gray 19-32 against a gray-55 floor) and
brightness alone would pick the floor. What the cloth has instead is colour.

Which colour, though, is not knowable in advance, and neither is how far it
stands out. An earlier version fixed both as constants - a list of plausible
cloth hues, and a saturation floor of 70 taken from a table measured at 224
against a floor below 45 - and it worked until the camera was raised, at
which point a cardboard box beyond the table entered frame at saturation 115
and passed a gate built to exclude a floor at 45. The constants were not
wrong so much as local: they described one table at one distance under one
light, and every one of those numbers moves with the setup.

So nothing about the colour is assumed here. The cloth is identified as the
largest saturated expanse in the frame, its hue read off a saturation-
weighted histogram, and the gate then built around that measured hue with a
threshold taken from the cloth's own pixels. What stays fixed is only the
geometry of the situation - a frame set up to see the table is mostly table -
which holds for any cloth colour, any camera height and any room.

The corners wanted are the cloth's, not the cabinet's: play happens inside
the cushions, so that is where table coordinates must start.
"""

from __future__ import annotations

import cv2
import numpy as np

from companion.pool.perception.vision.table_spec import (
    corner_quad_mm, order_quad)

ReferencePoints = tuple[np.ndarray, np.ndarray]

# The cloth's colour is measured from the frame rather than matched against a
# list of expected ranges. Fixed hue bands cannot separate cloth from the room:
# a burgundy cloth sits at hue 178 and cardboard at hue 8, and any band wide
# enough to hold "red" cloth holds both. Worse, the discriminator the fixed
# gate leaned on - saturation - is a property of the scene, not of cloth: the
# same table that measured saturation 224 against a floor below 45 close up
# measured a cardboard box at 115 once the camera was raised, i.e. above the
# threshold. Both numbers move with distance, lighting and white balance.
#
# What does not move is that the cloth is the single largest expanse of one
# saturated colour in a frame framed to see the table. So the dominant
# saturated hue is found first, and the gate is then built around that hue -
# which costs one histogram and makes the detector independent of cloth
# colour, camera height and room clutter alike.
HUE_TOLERANCE = 12    # degrees either side of the measured cloth hue
SAT_PERCENTILE = 10   # keep the bottom decile of the cloth's own saturation
MIN_SATURATION = 50   # floor under the measured threshold, for pale cloth
MIN_VALUE = 30        # exclude near-black shadow, which has unstable hue
MAX_VALUE = 250       # exclude blown-out specular highlights
MIN_HUE_PIXELS = 500  # too little colour in frame to name a cloth hue

MIN_AREA_FRACTION = 0.10   # the table dominates a frame set up to see it
MAX_AREA_FRACTION = 0.95
MIN_FILL = 0.70            # cloth fills its own bounding quad
MIN_CORNER_COSINE = 0.55   # perspective skews the corners but not absurdly


def cloth_hue(hsv: np.ndarray) -> int | None:
    """The dominant saturated hue in the frame - the cloth's - or None.

    Votes are weighted by saturation so that a large dull region (a floor, a
    cardboard box) cannot outvote the cloth merely by being wide: what is
    being sought is the most colourful expanse, not the biggest one. The
    histogram is smoothed circularly because hue wraps at 180, and burgundy
    cloth straddles that seam - the measured peak here sits at 178, with its
    shoulder over on 0-2.
    """
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    candidate = (sat >= MIN_SATURATION) & (val >= MIN_VALUE) & (val <= MAX_VALUE)
    if int(candidate.sum()) < MIN_HUE_PIXELS:
        return None

    votes = np.zeros(180, np.float64)
    np.add.at(votes, hue[candidate].astype(np.intp),
              sat[candidate].astype(np.float64))
    window = np.ones(9) / 9.0
    wrapped = np.concatenate([votes[-8:], votes, votes[:8]])
    smoothed = np.convolve(wrapped, window, "same")[8:-8]
    return int(np.argmax(smoothed))


def cloth_mask(image: np.ndarray) -> np.ndarray:
    """Pixels matching the frame's own cloth colour, as a binary mask.

    Only the colour gate lives here. The morphological close that used to
    follow it has moved to `_largest_filled_component`, because closing
    before the component is chosen can only merge regions and never separate
    them: with the camera raised, a 31 px close reached across the rail and
    welded the cloth to a cardboard box below the table, and the resulting
    "largest component" spanned both. Choosing the component on the raw
    colour mask - where cloth and box are cleanly separate - and only then
    closing across the balls gets the same hole-filling without the bridge.
    """
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    peak = cloth_hue(hsv)
    if peak is None:
        return np.zeros(image.shape[:2], np.uint8)

    distance = np.abs(hue.astype(np.int16) - peak)
    distance = np.minimum(distance, 180 - distance)  # hue wraps at 180
    near = distance <= HUE_TOLERANCE

    # Take the saturation threshold from the cloth's own pixels rather than a
    # constant: a decile keeps the shaded end of the cloth while still
    # sitting far above anything duller in the room. Measured on the raised
    # frame this lands at 198, where the fixed 70 admitted the box at 115.
    cloth_sat = sat[near & (sat >= MIN_SATURATION)]
    threshold = (MIN_SATURATION if cloth_sat.size < MIN_HUE_PIXELS
                 else max(float(MIN_SATURATION),
                          float(np.percentile(cloth_sat, SAT_PERCENTILE))))

    mask = near & (sat >= threshold) & (val >= MIN_VALUE) & (val <= MAX_VALUE)
    return (mask.astype(np.uint8)) * 255


def _largest_filled_component(mask: np.ndarray) -> np.ndarray | None:
    """The biggest blob in the mask, closed up and with its holes filled.

    The component is chosen on the raw colour mask, before any closing, so
    that only pixels actually the colour of the cloth can decide which blob
    is the table. The close then runs on that blob alone, where it can reach
    across the balls and the triangle - the holes it is there to bridge -
    without being able to reach anything outside the cloth at all.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    blob = np.where(labels == largest, 255, 0).astype(np.uint8)

    # A kernel sized to the blob, not the frame: what has to be bridged is a
    # ball, and a ball is a fixed fraction of the table however far away the
    # camera is. Sizing this off the frame instead made the reach grow
    # relative to the table as the table shrank in view.
    width = int(stats[largest, cv2.CC_STAT_WIDTH])
    height = int(stats[largest, cv2.CC_STAT_HEIGHT])
    span = max(9, (max(width, height) // 12) | 1)
    blob = cv2.morphologyEx(
        blob, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (span, span)))
    blob = cv2.morphologyEx(
        blob, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    # Fill interior holes (balls, pocket mouths, the triangle) by flood
    # filling the background from a corner and OR-ing in the complement.
    h, w = blob.shape
    flood = blob.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    return blob | cv2.bitwise_not(flood)


def playing_surface_mask(image: np.ndarray) -> np.ndarray | None:
    """The cloth region as one solid mask, or None if there is no cloth.

    The filled form, so that the balls and the triangle - which are holes in
    the raw colour mask - count as part of the surface. `detect.balls` uses
    it to bound its search: the fitted corners can sit slightly outside the
    true cloth, and the lit edge of a cushion just beyond them scores like a
    row of touching balls.
    """
    if image.ndim != 3:
        return None
    return _largest_filled_component(cloth_mask(image))


def find_cloth_quad(image: np.ndarray) -> np.ndarray | None:
    """The playing surface's 4 corners, ordered TL, TR, BR, BL, or None."""
    if image.ndim != 3:
        return None  # colour is the whole signal; a grey frame cannot work
    filled = _largest_filled_component(cloth_mask(image))
    if filled is None:
        return None

    area = float(cv2.countNonZero(filled))
    if not MIN_AREA_FRACTION <= area / filled.size <= MAX_AREA_FRACTION:
        return None

    # Every boundary pixel, not the run-length summary. `CHAIN_APPROX_SIMPLE`
    # collapses a straight run to its two endpoints, which is exactly what the
    # cushion lines are: on one real frame it left the two short sides with 6
    # and 14 points, too few to fit, while the long sides kept 57-71 only
    # because the pockets break them up. `refine_cloth_quad` fits those lines
    # from this contour, so it needs the run itself.
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)

    quad = _quad_from_contour(contour)
    if quad is None:
        return None
    if not _is_plausible(quad, contour):
        return None
    return refine_cloth_quad(order_quad(quad), image, contour)


def _quad_from_contour(contour: np.ndarray) -> np.ndarray | None:
    """Reduce the cloth outline to four corners.

    Tries polygon approximation first and falls back to the minimum-area
    rectangle. The fallback matters at the pockets: each corner pocket cuts
    the cloth away diagonally, so the true outline is closer to an octagon
    than a quadrilateral and approxPolyDP often returns 5-8 vertices. The
    enclosing rectangle recovers the corners the cushions would meet at,
    which is what table coordinates are defined against.
    """
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None
    for eps in (0.02, 0.03, 0.04):
        approx = cv2.approxPolyDP(contour, eps * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float64)
    return cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float64)


def _is_plausible(quad: np.ndarray, contour: np.ndarray) -> bool:
    """Reject shapes a playing surface could not project to."""
    ordered = order_quad(quad)
    for i in range(4):
        a = ordered[(i - 1) % 4] - ordered[i]
        b = ordered[(i + 1) % 4] - ordered[i]
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na < 1e-6 or nb < 1e-6:
            return False
        if abs(float(np.dot(a, b)) / (na * nb)) > MIN_CORNER_COSINE:
            return False
    quad_area = abs(cv2.contourArea(ordered.astype(np.float32)))
    if quad_area <= 0:
        return False
    # The cloth must fill the quad it is claimed to occupy; a low fill means
    # the mask leaked into the floor and dragged a corner out with it.
    return abs(cv2.contourArea(contour)) / quad_area >= MIN_FILL


def refine_cloth_quad(quad: np.ndarray, image: np.ndarray,
                     contour: np.ndarray | None = None) -> np.ndarray:
    """Sharpen the corners by re-fitting the four cushion lines.

    The corners are the least reliable part of the outline - every one of
    them is cut away by a pocket - while the cushion edges between them are
    long, straight and unambiguous. So each side is fitted from the middle
    of its own stretch, well clear of the pockets, and the corners recovered
    by intersecting neighbouring lines. This is the same argument as for the
    paper's drawn-on pockets, and it is what puts the corner where the
    cushions would meet rather than where the pocket cut begins.

    The fit is over the region's *boundary* where one is supplied, not over
    a band of its interior. A band is centred on the quad it is given, so
    whatever bias that quad already carries decides which pixels vote, and
    the fit inherits it: measured across the recorded frames, the incoming
    quad sat 16-47 px outside the cloth on every side and the band fit moved
    it about 4 px, leaving the drawn edge out on the rail. The boundary does
    not move when the quad does.
    """
    points = None
    if contour is not None:
        boundary = np.asarray(contour).reshape(-1, 2).astype(np.float64)
        if len(boundary) >= 200:
            points = boundary
    if points is None:
        return _refine_from_interior(quad, image)
    return _refine_from_points(quad, points, outer_percentile=45.0,
                               band_frac=0.10, min_points=25)


def _refine_from_interior(quad: np.ndarray, image: np.ndarray) -> np.ndarray:
    """The original interior-band fit, for callers with no contour to hand."""
    mask = _largest_filled_component(cloth_mask(image))
    if mask is None:
        return quad
    # Subsample the region's pixels rather than using every one. The fit is
    # a least-squares line over a band of cloth, so a fraction of the pixels
    # locates it just as well; using all ~400k of them made a single frame
    # take seconds, which a 30-frame burst cannot afford.
    #
    # Interior pixels, not just the outline: the rail-face cloth sits just
    # outside the true boundary, so an outline-only fit is pulled onto it,
    # while the band of interior pixels outvotes it. Measured on a real
    # frame, outline-only gave opposite-side ratios 0.855/0.829 against
    # 0.905/0.965 for the band.
    ys, xs = np.nonzero(mask)
    if len(xs) < 200:
        return quad
    step = max(1, len(xs) // 60000)
    points = np.column_stack((xs[::step], ys[::step])).astype(np.float64)

    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        edge = b - a
        length = float(np.linalg.norm(edge))
        if length < 20.0:
            return quad
        direction = edge / length
        normal = np.array([-direction[1], direction[0]])
        offset = (points - a) @ normal
        along = (points - a) @ direction
        # Cloth pixels hugging this side, sampled from its middle 60% so the
        # corner pockets at either end cannot pull the fit.
        near = ((np.abs(offset) < 0.04 * length)
                & (along > 0.20 * length) & (along < 0.80 * length))
        if int(near.sum()) < 40:
            return quad
        lines.append(_fit_line(points[near]))

    corners = []
    for i in range(4):
        point = _intersect_lines(lines[i - 1], lines[i])
        if point is None or not np.all(np.isfinite(point)):
            return quad
        corners.append(point)
    refined = order_quad(np.array(corners, np.float64))

    # A refinement that moves a corner a long way is not describing this
    # shape any more; keep the original rather than trusting it.
    diagonal = float(np.linalg.norm(quad[2] - quad[0]))
    if float(np.abs(refined - order_quad(quad)).max()) > 0.15 * diagonal:
        return quad
    return refined


def _refine_from_points(quad: np.ndarray, points: np.ndarray, *,
                        outer_percentile: float, band_frac: float,
                        min_points: int) -> np.ndarray:
    """Re-cut the corners by fitting each cushion to the boundary points.

    For each side, the boundary points lying near it and along its middle
    stretch are collected, and the *outermost* fraction of those is what the
    line is fitted to. Taking the outer fraction is what separates the
    cushion from the pockets: a pocket cuts the boundary inward, so its
    points sit well inside the cushion's own run and drop out of the
    percentile rather than tilting the line, and no assumption about where
    the pockets are is needed to exclude them.
    """
    quad = order_quad(np.asarray(quad, np.float64))
    centre = quad.mean(axis=0)

    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        edge = b - a
        length = float(np.linalg.norm(edge))
        if length < 20.0:
            return quad
        direction = edge / length
        normal = np.array([-direction[1], direction[0]])
        if float((centre - a) @ normal) < 0.0:
            normal = -normal  # points into the table

        offset = (points - a) @ normal
        along = (points - a) @ direction
        near = ((np.abs(offset) < band_frac * length)
                & (along > 0.20 * length) & (along < 0.80 * length))
        if int(near.sum()) < min_points:
            return quad
        candidates = points[near]
        outer = candidates[
            offset[near] <= np.percentile(offset[near], outer_percentile)]
        if len(outer) < 8:
            return quad
        lines.append(_fit_line(outer))

    corners = []
    for i in range(4):
        point = _intersect_lines(lines[i - 1], lines[i])
        if point is None or not np.all(np.isfinite(point)):
            return quad
        corners.append(point)
    refined = order_quad(np.array(corners, np.float64))

    diagonal = float(np.linalg.norm(quad[2] - quad[0]))
    if float(np.abs(refined - quad).max()) > 0.15 * diagonal:
        return quad
    return refined


def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Total-least-squares line through points, as (point, unit direction)."""
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid)
    return centroid, vt[0]


def _intersect_lines(l0: tuple[np.ndarray, np.ndarray],
                     l1: tuple[np.ndarray, np.ndarray]) -> np.ndarray | None:
    (p0, d0), (p1, d1) = l0, l1
    denom = float(d0[0] * d1[1] - d0[1] * d1[0])
    if abs(denom) < 1e-9:
        return None
    diff = p1 - p0
    return p0 + d0 * (float(diff[0] * d1[1] - diff[1] * d1[0]) / denom)


def detect_reference_points(image: np.ndarray) -> ReferencePoints | None:
    """Matched (image_pts, table_pts) for the cloth's 4 corners, or None.

    Same contract as `detect.reference.detect_reference_points`, so the rest
    of the pipeline does not know or care which surface it is looking at.
    """
    quad = find_cloth_quad(image)
    if quad is None:
        return None
    return quad, corner_quad_mm()


def describe_failure(image: np.ndarray) -> str:
    """Why the cloth was not found, in terms that suggest a fix."""
    if image.ndim != 3:
        return "the frame is greyscale; cloth detection needs colour"
    hsv = cv2.cvtColor(cv2.GaussianBlur(image, (7, 7), 0), cv2.COLOR_BGR2HSV)
    peak = cloth_hue(hsv)
    if peak is None:
        return (f"no saturated colour in the frame to read a cloth hue from "
                f"(peak saturation {int(hsv[..., 1].max())}, need "
                f"{MIN_SATURATION}). Is the table in view and lit well enough "
                "for its colour to show?")
    mask = cloth_mask(image)
    fraction = cv2.countNonZero(mask) / mask.size
    if fraction < 0.02:
        return (f"the dominant colour in frame is hue {peak}, but almost "
                "nothing matches it as a solid region. Is the table in view, "
                "and is something more colourful than the cloth filling the "
                "frame?")
    filled = _largest_filled_component(mask)
    if filled is None:
        return "cloth-coloured pixels found, but not as one connected region"
    area = cv2.countNonZero(filled) / filled.size
    if area < MIN_AREA_FRACTION:
        return (f"the largest cloth region covers only {area:.1%} of the "
                f"frame (need {MIN_AREA_FRACTION:.0%}) - move the camera "
                "closer, or fit more of the table in view")
    if area > MAX_AREA_FRACTION:
        return ("cloth fills almost the whole frame - move the camera back "
                "so all four cushions are visible")
    return ("found the cloth, but its outline is not table-shaped - check "
            "all four cushions are in view and nothing large is covering "
            "the surface")
