"""The ball radius is measured from the frame, not inherited from the prior.

The property that matters is independence: whatever radius the pipeline is
told to assume, the answer must come out the same, because it is read off
pixels the assumption never touched. A test that only checks accuracy would
pass just as happily on an estimator that returns its own input.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from companion.pool.perception.vision import table_spec
from companion.pool.perception.vision.ball_scale import (
    RADIUS_FRAC_MAX, RADIUS_FRAC_MIN, BallScale, measure_ball_scale,
    measure_from_detections)
from companion.pool.perception.vision.homography import Homography

LONG_PX = 1200
SHORT_PX = 600


def _homography() -> Homography:
    """A straight-on view: 1200 px across a long side of 1.0 unit."""
    src = np.array([[0, 0], [LONG_PX, 0], [LONG_PX, SHORT_PX], [0, SHORT_PX]],
                   np.float32)
    dst = np.array([[0, 0], [1.0, 0], [1.0, 0.5], [0, 0.5]], np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return Homography(H=H, H_inv=np.linalg.inv(H), reprojection_error_mm=0.0,
                      inliers=4, total_points=4)


def _table(radius_frac: float, n: int = 9,
           cloth=(40, 40, 160)) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """A synthetic frame with `n` balls of a known size on coloured cloth."""
    radius_px = radius_frac * LONG_PX
    image = np.full((SHORT_PX, LONG_PX, 3), cloth, np.uint8)
    rng = np.random.default_rng(7)
    centres = []
    margin = 3.0 * radius_px
    for i in range(n):
        # Spread them out so no two blobs touch at any search window size.
        cx = margin + (LONG_PX - 2 * margin) * (i + 0.5) / n
        cy = SHORT_PX / 2 + (60 if i % 2 else -60)
        colour = tuple(int(v) for v in rng.integers(180, 256, 3))
        cv2.circle(image, (int(cx), int(cy)), int(round(radius_px)), colour, -1)
        centres.append((cx, cy))
    return image, centres


class _Detected:
    def __init__(self, px):
        self.px = px


def _detector(centres):
    return lambda image, homography, frac: [_Detected(c) for c in centres]


@pytest.fixture(autouse=True)
def _normalised_spec():
    previous = table_spec.active_spec()
    table_spec.set_active_spec(table_spec.TableSpec(1.0, 0.5, "u"))
    yield
    table_spec.set_active_spec(previous)


@pytest.mark.parametrize("truth", [0.0113, 0.0150, 0.0286, 0.040])
def test_measures_the_true_radius_whatever_the_size(truth):
    """A bigger ball on the same cloth must read as a bigger ball."""
    image, centres = _table(truth)
    got = measure_from_detections(image, _homography(), centres, 0.0150)
    assert got is not None
    assert got[0] == pytest.approx(truth, rel=0.08)


@pytest.mark.parametrize("prior", [0.0090, 0.0150, 0.0286, 0.055])
def test_the_answer_does_not_depend_on_the_prior(prior):
    """The whole point: the prior seeds the search and nothing else.

    This is the regression that matters. The estimator cannot be allowed to
    run on the rectified view, where the scale is set *from* the prior and
    any measurement returns it straight back.
    """
    truth = 0.0286
    image, centres = _table(truth)
    scale = measure_ball_scale(image, _homography(), prior_frac=prior,
                               detect=_detector(centres))
    assert scale is not None
    assert scale.radius_frac == pytest.approx(truth, rel=0.08)
    assert scale.converged


def test_a_two_times_wrong_prior_is_corrected():
    """The live failure: a mini table read with a regulation-table prior."""
    truth = 0.0286
    image, centres = _table(truth)
    scale = measure_ball_scale(image, _homography(), prior_frac=0.0142875,
                               detect=_detector(centres))
    assert scale is not None
    assert scale.ratio_to_prior == pytest.approx(2.0, rel=0.10)


def test_a_too_small_window_does_not_halve_the_radius():
    """A window that only fits the assumed ball must still see a bigger one.

    Sizing the search window from the quantity under test is the same
    "too small a disc splits a ball" failure the detector documents; here it
    returned exactly half the true radius before the window was allowed to
    grow.
    """
    truth = 0.040
    image, centres = _table(truth)
    got = measure_from_detections(image, _homography(), centres, 0.008)
    assert got is not None
    assert got[0] == pytest.approx(truth, rel=0.10)


def test_balls_the_colour_of_the_cloth_are_still_found():
    """Chroma, not brightness: a red ball on red cloth is the hard case.

    A grey threshold splits the lighting rather than the ball, which on a
    real frame either clipped the ball or ran away into a projection.
    """
    image, centres = _table(0.0286, cloth=(40, 40, 170))
    # Balls a shade off the cloth in hue but matched in luminance.
    for cx, cy in centres:
        cv2.circle(image, (int(cx), int(cy)),
                   int(round(0.0286 * LONG_PX)), (150, 60, 175), -1)
    got = measure_from_detections(image, _homography(), centres, 0.0150)
    assert got is not None
    assert got[0] == pytest.approx(0.0286, rel=0.12)


def test_too_few_balls_reports_nothing_rather_than_guessing():
    """Below three samples a median cannot outvote one bad edge."""
    image, centres = _table(0.0286, n=2)
    assert measure_from_detections(image, _homography(), centres, 0.0150) is None


def test_no_detections_is_not_an_error():
    image, _ = _table(0.0286)
    assert measure_from_detections(image, _homography(), [], 0.0150) is None
    assert measure_ball_scale(image, _homography(), prior_frac=0.0150,
                              detect=lambda *a: []) is None


def test_an_impossible_radius_is_refused():
    """Outside what a pool set can be, the frame is wrong, not the equipment."""
    image, centres = _table(0.0286)
    # Claim a long side so short that the balls span most of the table.
    table_spec.set_active_spec(table_spec.TableSpec(0.02, 0.01, "u"))
    assert measure_from_detections(image, _homography(), centres, 0.0150) is None


@pytest.mark.parametrize("truth", [0.0113, 0.0286, 0.040])
def test_the_bias_is_small_and_known_to_run_low(truth):
    """Pin the size and the sign of the error.

    The estimate reads a few percent low, because a ball is a sphere sitting
    on the plane the homography describes rather than a figure in it, and the
    blur at its edge loses a fraction of a pixel. That runs toward calling a
    tight lane clear - the same direction as the bug this replaces - so it is
    worth failing a test if it ever grows.
    """
    image, centres = _table(truth)
    got = measure_from_detections(image, _homography(), centres, 0.0150)
    assert got is not None
    error = (got[0] - truth) / truth
    assert -0.08 < error <= 0.02, f"bias {error:+.1%} outside the known band"


def test_a_measured_radius_is_not_fed_back_into_detection():
    """The measurement is for geometry only; detection keeps its own prior.

    Re-detecting at the measured radius looks obviously right and is not: on
    the live frame it kept every ball and every position but destroyed the
    stripe/solid call, 3 stripes to 0, because both classifiers were
    calibrated around the old prior - the rim annulus reaches a real ball's
    shadowed edge instead of its bright interior, and the network's training
    crops were cut at that prior. Positions are in table units and do not
    depend on the prior, which is what lets the two uses be separated.

    This pins the split by reading the tool's own source, because the
    alternative is a live camera.
    """
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2]
              / "tools" / "live_scan.py").read_text(encoding="utf-8")
    body = source.split("def _scan(")[0]
    assert "ball_radius=scale.radius_frac" in body, (
        "the measured radius must still reach the planner's geometry")
    assert "radius_frac=scale.radius_frac" not in body, (
        "the measured radius must not be fed back into detection; see "
        "the note in live_scan.py")


def test_the_estimate_is_reported_with_its_evidence():
    image, centres = _table(0.0286)
    scale = measure_ball_scale(image, _homography(), prior_frac=0.0150,
                               detect=_detector(centres))
    assert isinstance(scale, BallScale)
    assert scale.samples >= 3
    assert 0.0 <= scale.spread < 0.2
    assert RADIUS_FRAC_MIN <= scale.radius_frac <= RADIUS_FRAC_MAX
    assert "of the long side" in scale.describe()
    assert scale.as_dict()["ratio_to_prior"] == pytest.approx(
        scale.ratio_to_prior, rel=1e-3)
