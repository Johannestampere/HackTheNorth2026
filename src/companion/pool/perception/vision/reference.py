"""Find the reference points that tie the image to table coordinates.

Everything downstream consumes only `detect_reference_points`, so this is the
one place that changes when the reference changes. It currently finds a plain
sheet of paper lying on a darker surface; a felt-quad detector for a real
table would replace the same function.

The sheet must be clearly brighter than whatever it lies on. That is a real
constraint, not an implementation detail: where the paper and the surface are
equally bright, the boundary carries no signal at all and no amount of
processing recovers it. `describe_failure` reports that case specifically.
"""

from __future__ import annotations

import cv2
import numpy as np

from companion.pool.perception.vision.table_spec import (
    corner_quad_mm, order_quad)

ReferencePoints = tuple[np.ndarray, np.ndarray]

# A quad must cover this fraction of the frame to be the sheet, which rules out
# scraps, glare patches and distant clutter without enumerating them.
MIN_AREA_FRACTION = 0.04
MAX_AREA_FRACTION = 0.92
# Shape limits that do NOT assume the sheet's size, since measuring it is the
# whole job. Perspective can stretch a square towards 3:1 and skew a right
# angle well past 45 degrees, so these only exclude the genuinely absurd.
MAX_SIDE_RATIO = 4.0      # longest/shortest side; slivers are not sheets
MIN_OPPOSITE_RATIO = 0.55  # opposite sides stay comparable under perspective
MIN_CORNER_COSINE = 0.50  # reject only badly non-rectangular quads
MIN_EDGE_CONTRAST = 10.0  # paper must be this much brighter than its surround
# Fraction of the outline that must lie on a real image edge. Kept modest
# because a taped-down sheet genuinely interrupts its own outline: tape tabs,
# a case lip and the contact shadow each break a stretch of edge. Measured at
# ~0.39 on a real photo of a sheet in a tablet case, so 0.30 leaves headroom
# while still rejecting quads that float over flat background (those score
# near zero, not near 0.3).
MIN_EDGE_SUPPORT = 0.30
EDGE_STRENGTH = 12.0      # gradient magnitude (0-255) counting as an edge
MAX_BORDER_FRACTION = 0.25  # at most this much of the outline may be frame edge


# --- Masks ----------------------------------------------------------------

def _masks(gray: np.ndarray) -> list[np.ndarray]:
    """Binary masks in which the sheet may appear as its own bright region.

    Several thresholdings are pooled because no single one covers every case:
    Otsu is decisive when the sheet sits on a clearly darker surface, adaptive
    survives a lighting gradient across the sheet, and the high percentiles
    still isolate the paper when another bright object drags Otsu's split
    point away from it.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    out: list[np.ndarray] = []

    _, otsu = cv2.threshold(blurred, 0, 255,
                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    out.append(otsu)
    out.append(cv2.adaptiveThreshold(blurred, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 51, -10))
    for percentile in (70.0, 80.0, 88.0):
        level = float(np.percentile(blurred, percentile))
        _, fixed = cv2.threshold(blurred, level, 255, cv2.THRESH_BINARY)
        out.append(fixed)
    return out


def _candidate_quads(gray: np.ndarray) -> list[np.ndarray]:
    """Quadrilaterals that could be the sheet."""
    quads: list[np.ndarray] = []
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    for mask in _masks(gray):
        # Close gaps where tape or glare breaks the outline, then open to
        # sever thin bridges joining the sheet to another bright region.
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        for variant in (cleaned,
                        cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel)):
            contours, _ = cv2.findContours(variant, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                quad = _approx_quad(contour)
                if quad is not None:
                    quads.append(quad)
                # A ragged outline can defeat approxPolyDP while minAreaRect
                # still recovers the shape, but only trust that when the
                # contour actually fills its own rectangle.
                rect = cv2.boxPoints(cv2.minAreaRect(contour))
                if _fills_rect(contour, rect):
                    quads.append(rect.astype(np.float64))

                # Extreme points of the convex hull. Shadow can eat a whole
                # stretch of one side out of the mask and tape can bite the
                # corners, leaving a shape that is neither a clean 4-gon nor
                # a filled rectangle - but whose outermost points are still
                # the sheet's corners.
                extremes = _extreme_corners(contour)
                if extremes is not None:
                    quads.append(extremes)
    return quads


def _extreme_corners(contour: np.ndarray) -> np.ndarray | None:
    """Corners recovered by fitting the contour's four sides and intersecting.

    The corners themselves are the least reliable part of a real sheet: the
    pockets drawn at each corner cut it off with a curve, and shadow can eat
    a corner out of the mask entirely. The *sides* survive all of that, so
    each side is fitted from the contour points that lie along it - sampled
    from the middle of the side, away from the corners - and neighbouring
    lines are intersected. That reconstructs the true corner even when no
    contour point is anywhere near it.
    """
    if len(contour) < 8:
        return None
    rect = order_quad(cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float64))
    points = contour.reshape(-1, 2).astype(np.float64)

    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(4):
        a, b = rect[i], rect[(i + 1) % 4]
        edge = b - a
        length = float(np.linalg.norm(edge))
        if length < 1e-6:
            return None
        direction = edge / length
        normal = np.array([-direction[1], direction[0]])
        offset = np.abs((points - a) @ normal)
        along = (points - a) @ direction
        # Close to this side, and in its middle stretch: the ends are where
        # the corner pockets and neighbouring sides confuse the fit.
        near = (offset < 0.10 * length) & (along > 0.20 * length) \
            & (along < 0.80 * length)
        if int(near.sum()) < 6:
            return None
        lines.append(_fit_line(points[near]))

    corners = []
    for i in range(4):
        point = _intersect_lines(lines[i - 1], lines[i])
        if point is None or not np.all(np.isfinite(point)):
            return None
        corners.append(point)
    corners = np.array(corners, np.float64)

    # The reconstruction must stay near the rectangle it came from; a wild
    # intersection means the side fits were not describing this shape.
    if float(np.abs(corners - rect).max()) > 0.35 * float(
            np.linalg.norm(rect[2] - rect[0])):
        return None
    return order_quad(corners)


def _approx_quad(contour: np.ndarray) -> np.ndarray | None:
    """Reduce a contour to a convex 4-gon, or None if it is not one."""
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None
    # Sweep epsilon: a single value either over- or under-simplifies depending
    # on how ragged the outline came out of thresholding.
    for eps in (0.02, 0.03, 0.04, 0.05):
        approx = cv2.approxPolyDP(contour, eps * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float64)
    return None


def _fills_rect(contour: np.ndarray, rect: np.ndarray,
                min_fill: float = 0.85) -> bool:
    """True if the contour fills most of its own bounding rectangle.

    Separates a real sheet from two bright objects joined by a bridge: the
    merged shape leaves large empty corners, a sheet fills its box.
    """
    rect_area = abs(cv2.contourArea(rect.astype(np.float32)))
    if rect_area <= 0:
        return False
    return abs(cv2.contourArea(contour)) / rect_area >= min_fill


# --- Scoring ---------------------------------------------------------------

def _score_quad(quad: np.ndarray, gray: np.ndarray,
                frame_area: float) -> float | None:
    """How sheet-like a quad is (higher is better), or None if disqualified."""
    ordered = order_quad(quad)
    area = abs(cv2.contourArea(ordered.astype(np.float32)))
    if not MIN_AREA_FRACTION <= area / frame_area <= MAX_AREA_FRACTION:
        return None

    # Corner angles: a sheet stays roughly rectangular under perspective.
    for i in range(4):
        a = ordered[(i - 1) % 4] - ordered[i]
        b = ordered[(i + 1) % 4] - ordered[i]
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na < 1e-6 or nb < 1e-6:
            return None
        if abs(float(np.dot(a, b)) / (na * nb)) > MIN_CORNER_COSINE:
            return None

    sides = [float(np.linalg.norm(ordered[(i + 1) % 4] - ordered[i]))
             for i in range(4)]
    if min(sides) < 1e-6:
        return None
    ratio = max(sides) / min(sides)
    # Deliberately NOT compared against the active spec's aspect ratio. The
    # whole point is to measure an unknown rectangle, so requiring it to
    # already match an assumed size is circular - it rejects exactly the
    # sheets we are trying to measure. Only absurd slivers are excluded, and
    # perspective alone can stretch a square to roughly 3:1.
    if ratio > MAX_SIDE_RATIO:
        return None

    # Opposite sides of a rectangle stay comparable under perspective: one
    # can be foreshortened relative to the other, but not arbitrarily. This
    # is the shape test that survives not knowing the true size.
    if (min(sides[0], sides[2]) / max(sides[0], sides[2]) < MIN_OPPOSITE_RATIO
            or min(sides[1], sides[3]) / max(sides[1], sides[3])
            < MIN_OPPOSITE_RATIO):
        return None

    contrast = _edge_contrast(ordered, gray)
    if contrast is None or contrast < MIN_EDGE_CONTRAST:
        return None

    # Edge support is the decisive term, not area. A quad that merely
    # *encloses* the sheet has more area than the true outline, so scoring on
    # area alone systematically prefers the wrong answer. A correct edge lies
    # along a real intensity step; an enclosing edge floats over background.
    support = _edge_support(ordered, gray)
    if support < MIN_EDGE_SUPPORT:
        return None

    # Reject quads that lean on the image border. A bright desk cropped by
    # the frame edge can look convincingly sheet-shaped, but the frame edge
    # is not an object boundary - a real sheet is bounded on all four sides
    # by its own edges, inside the picture.
    if _border_fraction(ordered, gray.shape) > MAX_BORDER_FRACTION:
        return None

    # Interior uniformity is what separates a sheet from a bright patch of
    # desk that happens to be quad-shaped. Paper is blank and evenly lit;
    # a desk region large enough to look sheet-like almost always swallows
    # something - a box edge, printed text, a cable - and its interior
    # variance gives it away. Contrast says "brighter than its surround";
    # uniformity says "and blank inside", and a sheet needs both.
    uniformity = _interior_uniformity(ordered, gray)

    # Area is deliberately absent: an enclosing quad always has more of it.
    return support * (contrast / 255.0) * uniformity


def _interior_uniformity(quad: np.ndarray, gray: np.ndarray) -> float:
    """How blank the quad's interior is, from 0 (busy) to 1 (featureless).

    A sheet of paper is flat in brightness; a quad-shaped patch of desk large
    enough to be mistaken for one almost always swallows something darker - a
    box edge, printed text, a cable - and its spread gives it away. Measured
    on a real frame: the sheet came out at std 10.5 while a competing desk
    region, brighter overall, sat at std 17.5.

    The spread is taken between percentiles rather than as a plain standard
    deviation so that a single specular highlight or one shadowed corner does
    not condemn an otherwise blank sheet.
    """
    # Shrink towards the centroid rather than eroding by a fixed number of
    # pixels: the quad's own size then sets how far in we look, and the
    # contact shadow and slightly-misplaced edges that always ring a real
    # sheet stay outside the sample. A fixed erosion leaves them in for a
    # large quad, which penalises exactly the sheets that were found well.
    centre = quad.mean(axis=0)
    inner = (centre + (quad - centre) * 0.75).astype(np.int32)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(mask, inner, 255)
    if cv2.countNonZero(mask) < 50:
        return 0.0
    values = gray[mask > 0]
    spread = float(np.percentile(values, 95) - np.percentile(values, 5))
    # ~60 grey levels of spread is where a region stops looking like paper.
    return float(max(0.0, 1.0 - spread / 60.0))


def _border_fraction(quad: np.ndarray, shape: tuple[int, ...],
                     margin: float = 12.0) -> float:
    """Fraction of the quad's perimeter running along the image border."""
    h, w = shape[:2]
    on_border = 0.0
    total = 0.0
    for i in range(4):
        p0, p1 = quad[i], quad[(i + 1) % 4]
        length = float(np.linalg.norm(p1 - p0))
        total += length
        midpoints = [p0 + (p1 - p0) * t for t in (0.25, 0.5, 0.75)]
        hits = sum(1 for m in midpoints
                   if m[0] < margin or m[0] > w - margin
                   or m[1] < margin or m[1] > h - margin)
        on_border += length * hits / len(midpoints)
    return on_border / total if total else 0.0


def _edge_contrast(quad: np.ndarray, gray: np.ndarray) -> float | None:
    """Mean brightness just inside the quad minus just outside it."""
    inside = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(inside, quad.astype(np.int32), 255)
    core = cv2.erode(inside, np.ones((9, 9), np.uint8))
    ring = cv2.subtract(cv2.dilate(inside, np.ones((25, 25), np.uint8)),
                        cv2.dilate(inside, np.ones((9, 9), np.uint8)))
    if cv2.countNonZero(core) == 0 or cv2.countNonZero(ring) == 0:
        return None
    return float(cv2.mean(gray, core)[0] - cv2.mean(gray, ring)[0])


def _edge_support(quad: np.ndarray, gray: np.ndarray,
                  samples_per_edge: int = 48) -> float:
    """Fraction of the quad's perimeter lying on a real image edge."""
    grad = _gradient_magnitude(gray)
    h, w = grad.shape
    hits = total = 0
    for i in range(4):
        p0, p1 = quad[i], quad[(i + 1) % 4]
        edge = p1 - p0
        length = float(np.linalg.norm(edge))
        if length < 1e-6:
            continue
        # Search a band either side of the nominal edge: a candidate from a
        # dilated mask sits a few pixels off the true boundary, and sampling
        # only exact pixels would score it near zero.
        normal = np.array([-edge[1], edge[0]]) / length
        for t in np.linspace(0.05, 0.95, samples_per_edge):
            base = p0 + edge * t
            total += 1
            for offset in (-4.0, -2.0, 0.0, 2.0, 4.0):
                x, y = base + normal * offset
                xi, yi = int(round(x)), int(round(y))
                if 0 <= xi < w and 0 <= yi < h and grad[yi, xi] >= EDGE_STRENGTH:
                    hits += 1
                    break
    return hits / total if total else 0.0


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Blurred Sobel magnitude, normalised to 0-255.

    Scaled by a high percentile rather than the maximum. Normalising by the
    maximum makes the whole map hostage to the single strongest edge in the
    scene - a dark chair or a window frame next to the sheet is enough to
    scale the paper's own edges below any fixed threshold, which silently
    disables both the support test and corner refinement.
    """
    # Edge-preserving smoothing first. Desk speckle, paper grain and sensor
    # noise all produce gradients comparable to a real paper edge, so a plain
    # Gaussian leaves the true boundary indistinguishable from the texture
    # around it; a bilateral filter flattens the texture and keeps the step.
    smooth = cv2.bilateralFilter(gray, 9, 60, 9)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=5)
    mag = cv2.dilate(cv2.magnitude(gx, gy), np.ones((5, 5), np.uint8))
    # Scaled by a high percentile rather than the maximum: normalising by the
    # maximum makes the map hostage to the single strongest edge in the scene
    # (a dark chair beside the sheet is enough), scaling the paper's own edges
    # below any fixed threshold.
    scale = float(np.percentile(mag, 99.5))
    if scale <= 1e-6:
        return np.zeros_like(mag)
    return np.clip(mag * (255.0 / scale), 0.0, 255.0)


# --- Public API ------------------------------------------------------------

def find_paper_quad(image: np.ndarray) -> np.ndarray | None:
    """The sheet's 4 corners in the image, ordered TL, TR, BR, BL.

    Corners are refined to sub-pixel accuracy; the homography is only as good
    as these four points.
    """
    gray = (image if image.ndim == 2
            else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    frame_area = float(gray.size)

    best: tuple[float, np.ndarray] | None = None
    for quad in _candidate_quads(gray):
        score = _score_quad(quad, gray, frame_area)
        if score is not None and (best is None or score > best[0]):
            best = (score, order_quad(quad))
    if best is None:
        return None

    return refine_quad(best[1], gray)


def refine_quad(quad: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Sharpen a quad by re-fitting its four sides to the image gradient.

    Taking the mask's own corners is not good enough when tape crosses them:
    the blob's corner is then the tape's, several millimetres inside the
    paper's. Each side is instead re-fitted from gradient peaks found along
    its middle - where no tape lies - and the true corners recovered by
    intersecting those lines, which also gives sub-pixel precision for free.
    """
    grad = _gradient_magnitude(gray)
    h, w = grad.shape
    lines: list[tuple[np.ndarray, np.ndarray]] = []

    for i in range(4):
        p0, p1 = quad[i], quad[(i + 1) % 4]
        edge = p1 - p0
        length = float(np.linalg.norm(edge))
        if length < 10.0:
            return quad
        normal = np.array([-edge[1], edge[0]]) / length

        points: list[np.ndarray] = []
        # Sample the middle 70% only: the ends are where tape and rounded
        # corners corrupt the edge.
        for t in np.linspace(0.15, 0.85, 40):
            base = p0 + edge * t
            best_offset, best_value = None, 0.0
            for offset in np.linspace(-6.0, 6.0, 25):
                x, y = base + normal * offset
                xi, yi = int(round(x)), int(round(y))
                if 0 <= xi < w and 0 <= yi < h and grad[yi, xi] > best_value:
                    best_value, best_offset = grad[yi, xi], offset
            if best_offset is not None and best_value >= EDGE_STRENGTH:
                points.append(base + normal * best_offset)

        if len(points) < 8:
            return quad  # too little support; keep the original quad
        lines.append(_fit_line(np.array(points)))

    corners = []
    for i in range(4):
        point = _intersect_lines(lines[i - 1], lines[i])
        if point is None:
            return quad
        corners.append(point)
    refined = order_quad(np.array(corners, np.float64))

    # Guard against a wild refinement: if any corner moved implausibly far,
    # the line fits were not describing this quad.
    if float(np.abs(refined - order_quad(quad)).max()) > 0.25 * min(h, w):
        return quad
    return refined


def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Total-least-squares line through points, as (point, unit direction)."""
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid)
    return centroid, vt[0]


def _intersect_lines(l0: tuple[np.ndarray, np.ndarray],
                     l1: tuple[np.ndarray, np.ndarray]) -> np.ndarray | None:
    """Intersection of two (point, direction) lines, or None if parallel."""
    (p0, d0), (p1, d1) = l0, l1
    denom = _cross2(d0, d1)
    if abs(denom) < 1e-9:
        return None
    return p0 + d0 * (_cross2(p1 - p0, d1) / denom)


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    """2D cross product; np.cross no longer accepts 2-vectors."""
    return float(a[0] * b[1] - a[1] * b[0])


def _orient_to_sheet(quad: np.ndarray) -> np.ndarray:
    """Return the quad unchanged; `order_quad` has already fixed the axes.

    Kept as a named step because the mapping from image corners to table
    coordinates is exactly the kind of thing that looks like it needs a
    correction and does not. `order_quad` resolves the corners geometrically
    as TL, TR, BR, BL, so the first side is always the sheet's top edge and
    the origin corner is always its top-left: the sheet's width maps to the
    first side by construction.

    A 90 degree roll would only be right if the paper were lying on its side
    relative to the camera, and nothing in one frame reliably distinguishes
    that from a steep viewing angle - a tilted portrait sheet is
    foreshortened until it measures wider than it is tall. Rotating the
    paper is a setup choice, so it belongs in the measured TableSpec, not in
    a per-frame guess.
    """
    return quad


def detect_reference_points(image: np.ndarray) -> ReferencePoints | None:
    """Matched (image_pts, table_pts) for the sheet's 4 corners, or None.

    Returns None when no convincing sheet is visible, so callers fail loudly
    rather than measuring against something that is not the paper.
    """
    quad = find_paper_quad(image)
    if quad is None:
        return None
    return _orient_to_sheet(quad), corner_quad_mm()


# --- Diagnostics -----------------------------------------------------------

def paper_surface_contrast(image: np.ndarray) -> float:
    """Brightness gap between the brightest large region and its surroundings.

    Reported when detection fails, because the usual cause is physical: the
    sheet and the surface under it are too close in brightness for any edge
    to exist at their boundary.
    """
    gray = (image if image.ndim == 2
            else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    bright = float(np.percentile(blurred, 90))
    dark = float(np.percentile(blurred, 40))
    return bright - dark


def describe_failure(image: np.ndarray) -> str:
    """Why detection failed, in terms the user can act on."""
    gray = (image if image.ndim == 2
            else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    frame_area = float(gray.size)
    quads = _candidate_quads(gray)

    if not quads:
        return ("no four-sided shape found - is the whole sheet in view, on a "
                "clearly darker surface?")

    # Contrast is checked before shape, because it is the more fundamental
    # cause: with no brightness step at the paper's boundary there is nothing
    # for any shape test to work on, and blaming the camera angle would send
    # the user chasing the wrong fix.
    big_enough = [order_quad(q) for q in quads
                  if abs(cv2.contourArea(order_quad(q).astype(np.float32)))
                  / frame_area >= MIN_AREA_FRACTION]
    contrasts = [c for c in (_edge_contrast(q, gray)
                             for q in (big_enough or [order_quad(q)
                                                      for q in quads]))
                 if c is not None]
    best_contrast = max(contrasts) if contrasts else 0.0
    if best_contrast < MIN_EDGE_CONTRAST:
        return (f"the best paper-to-surface contrast found is only "
                f"{best_contrast:.0f} grey levels (need "
                f"{MIN_EDGE_CONTRAST:.0f}). Whatever the paper is lying on is "
                "nearly as bright as the paper, so its edges carry no signal. "
                "Put the whole sheet on something clearly darker.")

    if not big_enough:
        biggest = max(abs(cv2.contourArea(order_quad(q).astype(np.float32)))
                      / frame_area for q in quads)
        return (f"the largest four-sided shape covers only {biggest:.1%} of the "
                "frame - move the camera closer, or lower it towards the desk")

    return ("found four-sided shapes, but none is sheet-shaped - check the "
            "whole sheet is in view, unobstructed, and lying flat")
