"""Recover a rectangle's true aspect ratio from its perspective projection.

The sheet's dimensions are not known in advance and must not be assumed: a
tilted rectangle's apparent proportions in the image tell you almost nothing
(a 210 x 297 sheet seen at a steep angle can measure wider than it is tall).

What does determine the ratio is the perspective itself. For a rectangle, the
two pairs of opposite sides meet at two vanishing points, and those points
constrain both the camera's focal length and the rectangle's real aspect
ratio. This is the classical single-view metrology result (Zhang, and
Liebowitz & Zisserman); the derivation below follows the standard treatment
for a rectangle of unknown size viewed by a camera with square pixels and a
centred principal point.

Only the ratio is recoverable from one view - a small sheet close to the
camera and a large one further away project identically - so the absolute
scale comes from one measured dimension supplied by the caller.
"""

from __future__ import annotations

import numpy as np


def _to_homogeneous(points: np.ndarray) -> np.ndarray:
    return np.hstack([points, np.ones((len(points), 1))])


def vanishing_points(quad: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """The two vanishing points of a quad's opposite side pairs, or None."""
    corners = _to_homogeneous(np.asarray(quad, np.float64).reshape(4, 2))
    tl, tr, br, bl = corners
    v1 = np.cross(np.cross(tl, tr), np.cross(bl, br))
    v2 = np.cross(np.cross(tl, bl), np.cross(tr, br))
    if abs(v1[2]) < 1e-9 or abs(v2[2]) < 1e-9:
        return None  # sides parallel in the image: no perspective to exploit
    return v1 / v1[2], v2 / v2[2]


def focal_conditioning(quad: np.ndarray,
                       principal_point: tuple[float, float]) -> float:
    """How trustworthy this view's focal estimate is, from 0 (useless) to 1.

    The focal length comes from where the two vanishing points sit, and a
    vanishing point far from the image centre is one whose position is barely
    constrained: as a view approaches fronto-parallel its vanishing points run
    towards infinity and a one-pixel corner error swings the focal estimate by
    hundreds of pixels. Measured on a real near-straight-on frame, one
    vanishing point sat 21,000 px out and 1 px of corner noise moved the
    estimate by ~40 px.

    Scoring each view lets a burst of frames lean on the well-conditioned ones
    and ignore the rest, instead of averaging good estimates with nonsense.
    """
    points = vanishing_points(quad)
    if points is None:
        return 0.0
    cx, cy = principal_point
    scale = max(abs(cx), abs(cy)) * 2.0  # roughly the image diagonal
    worst = 0.0
    for v in points:
        distance = float(np.hypot(v[0] - cx, v[1] - cy))
        worst = max(worst, distance / scale)
    if worst <= 0.0:
        return 0.0
    # A vanishing point a few image-widths out is still usable; ten widths
    # out is not. Falls off smoothly so there is no cliff edge.
    return float(1.0 / (1.0 + (worst / 3.0) ** 2))


def estimate_focal_length(quad: np.ndarray,
                          principal_point: tuple[float, float]
                          ) -> float | None:
    """Focal length in pixels implied by a rectangle's projection, or None.

    Returns None when the view is too close to fronto-parallel: the sides are
    then nearly parallel in the image, the vanishing points run off towards
    infinity, and the focal length is not observable from this view.

    A single estimate is only as good as its conditioning - see
    `focal_conditioning`, and prefer `estimate_focal_length_multi` whenever
    several frames are available.
    """
    points = vanishing_points(quad)
    if points is None:
        return None
    v1, v2 = points
    cx, cy = principal_point

    # With square pixels and a centred principal point, orthogonality of the
    # two world directions gives
    #     (v1 - c) . (v2 - c) + f^2 = 0
    # so f^2 is the negated dot product of the centred vanishing points.
    d1 = np.array([v1[0] - cx, v1[1] - cy])
    d2 = np.array([v2[0] - cx, v2[1] - cy])
    f_squared = -float(np.dot(d1, d2))
    if f_squared <= 1e-6:
        return None  # degenerate geometry; no real focal length
    return float(np.sqrt(f_squared))


def estimate_focal_length_multi(quads: list[np.ndarray],
                                image_shape: tuple[int, ...]
                                ) -> tuple[float | None, float]:
    """Focal length pooled over several views, with a confidence in 0..1.

    The focal length is a fixed property of the camera, so many views of the
    sheet are many measurements of one number - but they are not equally
    informative. Each view is weighted by its conditioning and the result is a
    weighted median, which ignores the wild estimates that near-fronto-parallel
    frames produce rather than letting them drag an average.

    Returns (focal, confidence). Confidence near zero means no view had usable
    perspective and the caller should not trust the value.
    """
    height, width = image_shape[:2]
    principal_point = (width / 2.0, height / 2.0)

    samples: list[tuple[float, float]] = []
    for quad in quads:
        weight = focal_conditioning(quad, principal_point)
        if weight <= 0.02:
            continue
        focal = estimate_focal_length(quad, principal_point)
        if focal is None or not np.isfinite(focal):
            continue
        # A plausible focal length for any real camera, in pixels: rule out
        # numerically absurd solutions before they reach the median.
        if not 0.2 * width <= focal <= 6.0 * width:
            continue
        samples.append((focal, weight))

    if not samples:
        return None, 0.0
    focal = _weighted_median([f for f, _ in samples], [w for _, w in samples])

    # Confidence is about whether this view can determine the shape at all,
    # not about how many frames were supplied. Counting frames made a single
    # live frame permanently "low confidence" however good the view was,
    # which left the preview stuck telling the user to tilt further.
    #
    # The divisor sets where "well conditioned" starts, and at 0.25 it sat too
    # low to mean anything: `focal_conditioning` calls a vanishing point ten
    # image-widths out unusable and scores it 0.09, but dividing by 0.25 made
    # that a confidence of 0.36, clearing the caller's 0.35 gate. A
    # near-top-down view of the table - conditioning 0.10, vanishing points
    # 17,000 and 4,800 px outside a 1280 px frame - was trusted on that basis
    # to measure the table's shape and returned a 1:0.62 table that is really
    # 1:0.5.
    #
    # 0.5 is calibrated against both ends instead. Synthetic views tilted
    # enough to recover the focal length to within 10% score up to 0.49 raw,
    # so they still come out near 1.0; the near-top-down table frames score
    # 0.10 raw and land at 0.20, well under the gate. In vanishing-point
    # terms the gate now asks for one within ~6.5 image-widths, which is the
    # "few widths usable, ten not" that `focal_conditioning` documents.
    CONFIDENT_CONDITIONING = 0.5
    best_weight = max(w for _, w in samples)
    confidence = float(min(1.0, best_weight / CONFIDENT_CONDITIONING))

    # Agreement between views can corroborate a pose that is already
    # informative, but it cannot rescue one that is not, so it only ever
    # lowers the score. Letting it raise the score (`max`) meant repeating a
    # measurement manufactured certainty from nothing: the same near-top-down
    # frame of the real table scored 0.17 alone and 1.00 when passed five
    # times, which took the shape estimate off a 0.53 outline ratio and onto
    # a 0.60 derived from vanishing points 20,000 px out. Frames from a
    # camera sitting still on a stand are one view sampled repeatedly, not
    # independent evidence, so they always agree however bad the pose is.
    if len(samples) >= 3:
        values = np.array([f for f, _ in samples], np.float64)
        spread = float(np.median(np.abs(values - focal)) / max(focal, 1e-6))
        agreement = float(max(0.0, 1.0 - spread / 0.25))
        confidence *= agreement
    return focal, confidence


def _weighted_median(values: list[float], weights: list[float]) -> float:
    order = np.argsort(values)
    v = np.asarray(values, np.float64)[order]
    w = np.asarray(weights, np.float64)[order]
    cumulative = np.cumsum(w)
    if cumulative[-1] <= 0:
        return float(np.median(v))
    return float(v[int(np.searchsorted(cumulative, cumulative[-1] / 2.0))])


def estimate_aspect_ratio(quad: np.ndarray, image_shape: tuple[int, ...],
                          focal_length: float | None = None,
                          *, allow_estimated_focal: bool = True
                          ) -> float | None:
    """Height / width of the real rectangle, from its projection alone.

    `quad` is ordered TL, TR, BR, BL. Returns None if the geometry does not
    support an estimate, leaving the caller to fall back on the apparent
    ratio.

    Pass `allow_estimated_focal=False` to forbid deriving a focal length here
    when none is supplied. A caller that has already judged its focal
    estimate untrustworthy needs that: without it, passing `focal_length=None`
    silently re-derives the very same bad number from the very same quad, and
    the caller's judgement is discarded. That was a real bug - a 2:1 table
    measured 140 x 97 mm because a rejected focal length came back in
    through this path.
    """
    quad = np.asarray(quad, np.float64).reshape(4, 2)
    height, width = image_shape[:2]
    principal_point = (width / 2.0, height / 2.0)

    if focal_length is None:
        if not allow_estimated_focal:
            return _apparent_ratio(quad)
        focal_length = estimate_focal_length(quad, principal_point)
    if focal_length is None or not np.isfinite(focal_length):
        return _apparent_ratio(quad)

    # The vanishing points give the rectangle's two side directions in camera
    # coordinates directly: a vanishing point back-projects to the direction
    # its family of parallel lines runs along. The ratio of the sides then
    # follows from how the corners project onto those two directions.
    corners = _to_homogeneous(quad)
    tl, tr, br, bl = corners
    v1 = np.cross(np.cross(tl, tr), np.cross(bl, br))
    v2 = np.cross(np.cross(tl, bl), np.cross(tr, br))

    K = np.array([[focal_length, 0.0, principal_point[0]],
                  [0.0, focal_length, principal_point[1]],
                  [0.0, 0.0, 1.0]])
    K_inv = np.linalg.inv(K)

    # World directions of the two side families, as unit vectors.
    d1 = K_inv @ v1
    d2 = K_inv @ v2
    if np.linalg.norm(d1) < 1e-12 or np.linalg.norm(d2) < 1e-12:
        return _apparent_ratio(quad)
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)

    # The plane normal is perpendicular to both side directions.
    normal = np.cross(d1, d2)
    if np.linalg.norm(normal) < 1e-12:
        return _apparent_ratio(quad)
    normal = normal / np.linalg.norm(normal)

    # Back-project each corner onto that plane (depth up to a common scale).
    rays = (K_inv @ corners.T).T
    denominators = rays @ normal
    if np.any(np.abs(denominators) < 1e-9):
        return _apparent_ratio(quad)
    points = rays / denominators[:, None]

    top = np.linalg.norm(points[1] - points[0])
    bottom = np.linalg.norm(points[2] - points[3])
    left = np.linalg.norm(points[3] - points[0])
    right = np.linalg.norm(points[2] - points[1])
    mean_width = (top + bottom) / 2.0
    mean_height = (left + right) / 2.0
    if mean_width < 1e-9:
        return _apparent_ratio(quad)

    ratio = float(mean_height / mean_width)
    if not np.isfinite(ratio) or not 0.05 < ratio < 20.0:
        return _apparent_ratio(quad)
    return ratio


def _apparent_ratio(quad: np.ndarray) -> float:
    """Height / width as it appears in the image, ignoring perspective."""
    top = float(np.linalg.norm(quad[1] - quad[0]))
    bottom = float(np.linalg.norm(quad[2] - quad[3]))
    left = float(np.linalg.norm(quad[3] - quad[0]))
    right = float(np.linalg.norm(quad[2] - quad[1]))
    mean_width = (top + bottom) / 2.0
    if mean_width < 1e-9:
        return 1.0
    return ((left + right) / 2.0) / mean_width


def sheet_size_mm(quad: np.ndarray, image_shape: tuple[int, ...],
                  reference_mm: float, reference_side: str = "width",
                  focal_length: float | None = None) -> tuple[float, float]:
    """The sheet's (width, height) in mm.

    The projection fixes the shape; one real measurement fixes the size. Give
    whichever side you measured with a ruler as `reference_mm`, naming it
    "width" (across x) or "height" (down y).
    """
    ratio = estimate_aspect_ratio(quad, image_shape, focal_length)
    if ratio is None:
        ratio = _apparent_ratio(np.asarray(quad, np.float64).reshape(4, 2))
    if reference_side == "width":
        return float(reference_mm), float(reference_mm * ratio)
    return float(reference_mm / ratio), float(reference_mm)
