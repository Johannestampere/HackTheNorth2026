"""Measure the ball radius from the frame instead of assuming it.

Why this module exists
----------------------
`BALL_RADIUS_FRAC` in `detector` is a *prior*: a ball's radius over the
playing surface's long side, measured once on the table this was developed
against. Its own docstring is explicit that a table whose balls are a
different size relative to the cloth makes it simply wrong, and that every
kernel in that module inherits the error.

That is not a hypothetical. On a tabletop table the true fraction is 0.0286
against a prior of 0.0150 - the balls are nearly twice the assumed size
relative to the cloth, because a mini table shrinks the cloth but keeps
something close to a normal ball. Detection survived it (the count came out
right), but the *planner* did not: `candidates.path_clear` compares a
measured ball separation against a prior-derived `2 * ball_radius`, so a lane
two stripes were sitting in read as clear by 1.3 ball-widths and the tool
called a shot that cannot be played.

The circularity, and the way out
--------------------------------
The radius cannot be read off the rectified view, because `detect_balls`
scales that view so a ball comes out exactly `BALL_RADIUS_PX` across - the
prior sets the scale, so re-measuring there just returns the prior. This is
a real fixed point, not an oversight, and any estimator that runs on
rectified pixels will confirm whatever it was told.

So measurement happens in the *raw* frame, where nothing has been scaled by
the prior yet, and the homography converts the result into surface units.
The homography comes from the table's corners and knows nothing about balls,
which is what makes it an independent ruler.

One pass is not enough, though, because the prior also decides which blobs
were called balls at all: too small a disc splits one ball into two
detections. So this iterates - measure, adopt, re-detect - until the fraction
stops moving. In practice that is two or three passes from a 2x error.

What this cannot do
-------------------
Absolute size stays unobservable from one camera, exactly as `table_spec`
says: this reports a *ratio* to the long side, which is the same kind of
number `BALL_RADIUS_FRAC` already was, and is what every consumer wants.

Accuracy is a few percent low on a straight-on view and falls further as the
view tilts - measured on synthetic tables across a 3.5x range of ball sizes,
-1% to -3% at nadir and about -10% at a steep angle, and low at every size
rather than scattered. Two causes, both one-sided: a ball is a sphere resting
*on* the plane the homography describes, so its silhouette is not a figure in
that plane at all, and the blur at its edge loses a fraction of a pixel to the
threshold. The bias is small next to the 2x error this exists to catch, but it runs
in the *unsafe* direction and should not be talked up as a safety margin: the
planner needs `2 * ball_radius` of clearance, so a radius read low lowers the
bar and errs toward calling a tight lane clear - the same direction as the
bug this replaces, two orders of magnitude smaller. A caller that wants a
hard guarantee should inflate the result rather than trust it raw; nothing
here does that on its behalf, because the right margin depends on what the
caller does with a false block.

`RADIUS_FRAC_MIN` and `RADIUS_FRAC_MAX` bracket the answer to what a pool set
can physically be, and a measurement outside that says the frame, not the
equipment, is the problem.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from companion.pool.perception.vision.homography import Homography
from companion.pool.perception.vision.table_spec import (
    sheet_h_mm, sheet_w_mm)

# A pool set's ball, over the playing surface's long side. A regulation
# 57.15 mm ball gives 0.0113 on a 9 ft table; the smallest toy tables reach
# about 0.045. Outside this the "ball" is table hardware, a reflection, or a
# frame the homography did not really fit.
RADIUS_FRAC_MIN = 0.008
RADIUS_FRAC_MAX = 0.060

# Stop when a pass moves the estimate by less than this, relatively.
CONVERGENCE_TOL = 0.02
MAX_PASSES = 4

# A radius estimate is taken from the ball's edge, so it needs an edge to
# find: candidates whose fitted circle is a poor fit for their own contour
# are dropped rather than averaged in.
MIN_CIRCULARITY = 0.72

# Where the edge search starts, as a multiple of the current estimate, and
# how it widens when the silhouette still touches the window's border. The
# start is deliberately generous and the growth deliberately open-ended,
# because the prior that sizes it is the quantity under test: a window that
# only fits the assumed ball is exactly the window that cannot see a bigger
# one. Growth stops at a quarter of the frame, well past any real ball.
SEARCH_R = 3.0
WINDOW_GROWTH = 1.6
WINDOW_GROWTH_STEPS = 6


@dataclass(frozen=True)
class BallScale:
    """A measured ball radius, with the evidence for it.

    `radius_frac` is directly comparable to `detector.BALL_RADIUS_FRAC` and
    to `TableGeometry.ball_radius`: all three are a radius over the playing
    surface's long side.
    """

    radius_frac: float
    samples: int
    spread: float          # robust relative spread across balls, 0 is perfect
    passes: int
    converged: bool
    prior_frac: float

    @property
    def ratio_to_prior(self) -> float:
        return self.radius_frac / self.prior_frac if self.prior_frac else 0.0

    def describe(self) -> str:
        return (f"ball radius {self.radius_frac:.4f} of the long side "
                f"({self.ratio_to_prior:.2f}x the assumed {self.prior_frac:.4f}), "
                f"from {self.samples} balls, spread {self.spread:.1%}")

    def as_dict(self) -> dict:
        return {
            "radius_frac": round(self.radius_frac, 5),
            "samples": self.samples,
            "spread": round(self.spread, 4),
            "passes": self.passes,
            "converged": self.converged,
            "prior_frac": round(self.prior_frac, 5),
            "ratio_to_prior": round(self.ratio_to_prior, 3),
        }


def _cloth_colour(patch: np.ndarray) -> np.ndarray:
    """The cloth's colour in this window, as its border's median.

    The border is used rather than the whole window because the ball is in
    the middle by construction: whatever surrounds it is what it sits on,
    and a median over that ignores a second ball clipping the corner.
    """
    edges = np.concatenate([
        patch[0, :].reshape(-1, patch.shape[2]),
        patch[-1, :].reshape(-1, patch.shape[2]),
        patch[:, 0].reshape(-1, patch.shape[2]),
        patch[:, -1].reshape(-1, patch.shape[2]),
    ])
    return np.median(edges.astype(np.float32), axis=0)


def _edge_at_window(lab: np.ndarray, cx: float, cy: float,
                    window_r: float) -> tuple[float, bool] | None:
    """One ball's silhouette radius in a window of the given size.

    A ball is the only thing in a window this size that is not cloth, so the
    split is between the two and neither colour has to be named in advance:
    the cloth's own colour comes from the window's border and Otsu finds the
    cut. Returns the radius and whether the blob reached the window's border
    - a blob that did was clipped, and its radius is a lower bound rather
    than a measurement.
    """
    h, w = lab.shape[:2]
    x0, y0 = int(max(0, cx - window_r)), int(max(0, cy - window_r))
    x1, y1 = int(min(w, cx + window_r)), int(min(h, cy + window_r))
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    local = (cx - x0, cy - y0)
    patch = cv2.GaussianBlur(lab[y0:y1, x0:x1], (5, 5), 0)
    if not (0 <= int(local[1]) < patch.shape[0]
            and 0 <= int(local[0]) < patch.shape[1]):
        return None
    # Distance from the cloth's own colour, not brightness. A red ball on red
    # cloth and a teal ball under a coloured projection are the same
    # luminance as what they sit on, so a grey Otsu splits the *lighting*
    # instead of the ball: measured on one such frame it either found two
    # thirds of a ball or ran away into the projected image, growing without
    # bound as the window widened. Chroma against the local cloth is the
    # distinction the detector already reasons in, and it survives both.
    cloth = _cloth_colour(patch)
    delta = np.linalg.norm(patch.astype(np.float32) - cloth, axis=2)
    scaled = cv2.normalize(delta, None, 0, 255, cv2.NORM_MINMAX)
    _, mask = cv2.threshold(scaled.astype(np.uint8), 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if mask[int(local[1]), int(local[0])] == 0:
        mask = cv2.bitwise_not(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    hit = [c for c in contours if cv2.pointPolygonTest(c, local, False) >= 0]
    if not hit:
        return None
    contour = max(hit, key=cv2.contourArea)

    area = cv2.contourArea(contour)
    if area < 9.0:
        return None
    (_, _), fitted = cv2.minEnclosingCircle(contour)
    if fitted <= 0:
        return None
    # A ball is round. Anything whose area disagrees with its enclosing
    # circle is a merged pair, a ball cut by the window, or a shadow.
    if area / (np.pi * fitted * fitted) < MIN_CIRCULARITY:
        return None

    bx, by, bw, bh = cv2.boundingRect(contour)
    clipped = (bx <= 1 or by <= 1
               or bx + bw >= mask.shape[1] - 1 or by + bh >= mask.shape[0] - 1)
    # Area-equivalent rather than the enclosing circle: specular highlights
    # and number patches push the hull out, and the area does not care.
    return float(np.sqrt(area / np.pi)), clipped


def _edge_radius_px(lab: np.ndarray, cx: float, cy: float,
                    start_r: float) -> float | None:
    """One ball's radius in raw-frame pixels, independent of the prior.

    The window cannot be sized from the estimate being measured. Too tight
    and the ball fills it, leaving Otsu no cloth to split against and
    returning whatever fraction of the ball happened to threshold - which is
    the same "too small a disc splits a ball" failure the detector's own
    docstring describes, reappearing one level up. On a 2x-wrong prior that
    is not an edge case; it is the expected case.

    So the window grows until the silhouette stops touching its border and
    the radius stops changing. A correct window is one with cloth around the
    whole ball, and that condition is checkable without knowing the answer.
    """
    radius = None
    window = max(start_r, 4.0)
    for _ in range(WINDOW_GROWTH_STEPS):
        found = _edge_at_window(lab, cx, cy, window)
        if found is not None:
            candidate, clipped = found
            if not clipped:
                # Settled: a further widening no longer changes the answer.
                if radius is not None and abs(candidate - radius) <= 0.05 * radius:
                    return candidate
                radius = candidate
        window *= WINDOW_GROWTH
        if window > 0.25 * min(lab.shape[:2]):
            break
    return radius


def _frac_from_pixels(radius_px: float, cx: float, cy: float,
                      homography: Homography, long_side: float) -> float | None:
    """Convert a raw-frame pixel radius into a fraction of the long side.

    The scale is taken *at the ball*, by stepping one measured radius along
    each image axis and asking the homography how far that was on the table.
    Doing it per ball rather than once for the frame is what keeps this
    honest under perspective: the far end of a tilted table has fewer pixels
    per unit than the near end, and a single global scale would bias every
    ball by its position.
    """
    probe = np.array([(cx, cy), (cx + radius_px, cy), (cx, cy + radius_px)],
                     np.float64)
    try:
        table = homography.to_table(probe)
    except Exception:
        return None
    if table is None or len(table) != 3:
        return None
    centre, along_x, along_y = (np.asarray(p, np.float64) for p in table)
    rx = float(np.hypot(*(along_x - centre)))
    ry = float(np.hypot(*(along_y - centre)))
    if not np.isfinite(rx) or not np.isfinite(ry) or rx <= 0 or ry <= 0:
        return None
    # The two axes disagree under perspective; their mean is the local
    # isotropic scale, which is what a sphere's radius is.
    return float(0.5 * (rx + ry) / long_side)


def _robust(values: list[float]) -> tuple[float, float]:
    """Median and a relative spread that a couple of bad balls cannot move."""
    arr = np.asarray(values, np.float64)
    median = float(np.median(arr))
    if median <= 0:
        return 0.0, 1.0
    mad = float(np.median(np.abs(arr - median)))
    return median, mad / median


def measure_from_detections(image: np.ndarray, homography: Homography,
                            centres_px: list[tuple[float, float]],
                            prior_frac: float) -> tuple[float, int, float] | None:
    """One pass: the median ball radius over already-detected centres.

    `centres_px` are in the raw frame, which is where `Ball.px` already puts
    them. Returns (radius_frac, samples, spread), or None if too few balls
    yielded a usable edge to be worth believing.
    """
    if not centres_px:
        return None
    long_side = max(sheet_w_mm(), sheet_h_mm())
    if long_side <= 0:
        return None
    bgr = (image if image.ndim == 3
           else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    # The window follows the *current* estimate, so a badly wrong prior still
    # gets a window wide enough to contain the real ball.
    approx_px = _prior_radius_px(homography, prior_frac, long_side, lab.shape)
    if approx_px is None or approx_px < 2.0:
        return None

    fracs = []
    for cx, cy in centres_px:
        radius_px = _edge_radius_px(lab, cx, cy, SEARCH_R * approx_px)
        if radius_px is None:
            continue
        frac = _frac_from_pixels(radius_px, cx, cy, homography, long_side)
        if frac is None or not (RADIUS_FRAC_MIN <= frac <= RADIUS_FRAC_MAX):
            continue
        fracs.append(frac)

    # Three is the fewest that lets a median outvote a single bad edge.
    if len(fracs) < 3:
        return None
    median, spread = _robust(fracs)
    if not (RADIUS_FRAC_MIN <= median <= RADIUS_FRAC_MAX):
        return None
    return median, len(fracs), spread


def _prior_radius_px(homography: Homography, prior_frac: float,
                     long_side: float, shape: tuple[int, ...]) -> float | None:
    """Roughly how many raw-frame pixels the current estimate implies.

    Only used to size the search window, so the centre of the table is a
    good enough place to ask.
    """
    cx_mm, cy_mm = sheet_w_mm() / 2.0, sheet_h_mm() / 2.0
    step = prior_frac * long_side
    try:
        pts = homography.to_image(np.array(
            [(cx_mm, cy_mm), (cx_mm + step, cy_mm)], np.float64))
    except Exception:
        return None
    if pts is None or len(pts) != 2:
        return None
    radius_px = float(np.hypot(*(np.asarray(pts[1]) - np.asarray(pts[0]))))
    if not np.isfinite(radius_px) or radius_px <= 0:
        return None
    return min(radius_px, 0.25 * min(shape[:2]))


def measure_ball_scale(image: np.ndarray, homography: Homography, *,
                       prior_frac: float,
                       detect=None) -> BallScale | None:
    """Measure the ball radius, re-detecting until the estimate settles.

    `detect` is called as `detect(image, homography, radius_frac)` and must
    return objects with a raw-frame `.px`; it defaults to `detector.detect_balls`.
    Injecting it keeps this module free of a circular import and lets a test
    drive the loop without the full detector.

    Returns None when the frame does not support a measurement - too few
    balls, or edges that will not agree. A caller that gets None should keep
    its prior and say so, rather than substitute a number nothing measured.
    """
    if detect is None:
        from companion.pool.perception.vision.detector import detect_balls

        def detect(img, h, frac):
            balls, _ = detect_balls(img, h, radius_frac=frac)
            return balls

    current = prior_frac
    best: tuple[float, int, float] | None = None
    passes = 0
    converged = False

    for passes in range(1, MAX_PASSES + 1):
        balls = detect(image, homography, current)
        centres = [b.px for b in balls if getattr(b, "px", None) is not None]
        measured = measure_from_detections(image, homography, centres, current)
        if measured is None:
            break
        best = measured
        moved = abs(measured[0] - current) / current if current else 1.0
        current = measured[0]
        if moved < CONVERGENCE_TOL:
            converged = True
            break

    if best is None:
        return None
    return BallScale(radius_frac=best[0], samples=best[1], spread=best[2],
                     passes=passes, converged=converged, prior_frac=prior_frac)
