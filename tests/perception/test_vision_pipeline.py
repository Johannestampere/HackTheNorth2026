"""End-to-end verification with no camera required.

Renders a plain sheet on a darker surface, warps it through a known
perspective to stand in for a tilted webcam, degrades it with an uneven
lighting ramp, blur and noise, then runs the real pipeline and checks the
recovered geometry.
"""

from __future__ import annotations

import itertools
import os
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
# Recorded camera frames are device data, kept out of the repo by .gitignore
# (see fixtures/README.md). The synthetic tests below need none of them; the
# real-table ones skip when the images are not on this machine.
FIXTURES = ROOT / "data/local/fixtures"
needs_fixtures = pytest.mark.skipif(
    not FIXTURES.is_dir(), reason=f"no recorded captures at {FIXTURES}")

from tools import htn26_cli as cli
from companion.pool.perception.vision.calibration import CalibrationError, calibrate_from_frames
from companion.pool.perception.vision.reference import (detect_reference_points,
                              find_paper_quad)
from companion.pool.perception.vision.surface import (active_surface, find_quad,
                            set_active_surface)
from .synth import render_sheet, scene, simulate_camera_view
from companion.pool.perception.vision.table_spec import (CORNER_IDS, HOLE_IDS, TableSpec,
                                 active_spec, holes_mm, order_quad, set_active_spec,
                                 sheet_h_mm, sheet_w_mm)

CORNER_TOLERANCE_PX = 8.0
POSITION_TOLERANCE_MM = 4.0


@pytest.fixture(autouse=True)
def paper_surface():
    """These tests render paper, so they need the paper detector.

    The CLI defaults to the pool table, which is what the tool is pointed at
    now. Pinning it here rather than relying on that default keeps the two
    independent: changing what the CLI looks for by default must not silently
    change what the synthetic tests exercise.
    """
    set_active_surface("paper")
    yield
    set_active_surface("paper")


@pytest.fixture(scope="module")
def frame_and_truth() -> tuple[np.ndarray, np.ndarray]:
    return scene()


def test_spec_drives_the_geometry() -> None:
    """The layout follows the measured size; nothing is hardcoded to A4."""
    set_active_spec(TableSpec(140.0, 169.0))
    try:
        assert (sheet_w_mm(), sheet_h_mm()) == (140.0, 169.0)
        assert len(holes_mm()) == 6
        assert tuple(holes_mm()) == HOLE_IDS
        # Corner pockets sit on the measured corners, not on A4's.
        assert holes_mm()["BR"] == (140.0, 169.0)
    finally:
        set_active_spec(TableSpec(210.0, 297.0))


def test_finds_the_sheet(frame_and_truth) -> None:
    frame, truth = frame_and_truth
    quad = find_paper_quad(frame)
    assert quad is not None, "the sheet should be found on a darker surface"
    errors = np.linalg.norm(quad - truth, axis=1)
    assert errors.max() < CORNER_TOLERANCE_PX, (
        f"corner errors {np.round(errors, 2).tolist()} px")


@pytest.mark.parametrize("tilt", [0.6, 1.0, 1.4])
def test_tolerates_a_range_of_tilts(tilt: float) -> None:
    frame, truth = scene(tilt=tilt)
    quad = find_paper_quad(frame)
    assert quad is not None, f"not found at tilt {tilt}"
    assert np.linalg.norm(quad - truth, axis=1).max() < CORNER_TOLERANCE_PX


def test_corners_are_ordered_tl_tr_br_bl(frame_and_truth) -> None:
    """Ordering must be geometric, not whatever the contour happened to give."""
    frame, _ = frame_and_truth
    quad = find_paper_quad(frame)
    tl, tr, br, bl = quad
    assert tl[0] < tr[0] and bl[0] < br[0], "left corners must be left of right"
    assert tl[1] < bl[1] and tr[1] < br[1], "top corners must be above bottom"


def test_round_trip_through_the_homography(frame_and_truth) -> None:
    """Table mm -> image px -> table mm must return where it started."""
    frame, _ = frame_and_truth
    grid, _ = calibrate_from_frames([frame])

    for name, xy in holes_mm().items():
        px = grid.homography.to_image(np.array([xy], np.float64))
        back = grid.homography.to_table(px)[0]
        assert np.hypot(*(back - np.array(xy))) < 0.5, f"{name} round trip"


def test_measured_sheet_matches_its_real_size(frame_and_truth) -> None:
    """The corners, pushed through the homography, span a real A4 sheet."""
    frame, truth = frame_and_truth
    grid, _ = calibrate_from_frames([frame])

    table = grid.homography.to_table(truth)
    width = (np.linalg.norm(table[1] - table[0])
             + np.linalg.norm(table[2] - table[3])) / 2.0
    height = (np.linalg.norm(table[3] - table[0])
              + np.linalg.norm(table[2] - table[1])) / 2.0
    assert abs(width - sheet_w_mm()) < POSITION_TOLERANCE_MM, f"width {width:.1f}"
    assert abs(height - sheet_h_mm()) < POSITION_TOLERANCE_MM, f"height {height:.1f}"


def test_all_six_pockets_reported(frame_and_truth) -> None:
    frame, _ = frame_and_truth
    grid, _ = calibrate_from_frames([frame])
    assert [h.id for h in grid.holes] == list(HOLE_IDS)
    assert all(h.found for h in grid.holes), (
        f"missing {[h.id for h in grid.holes if not h.found]}")


def test_median_over_many_frames() -> None:
    """A 30-frame burst with independent noise still resolves the sheet."""
    sheet, corners = render_sheet()
    frames = [simulate_camera_view(sheet, corners, seed=i, noise=5.0)[0]
              for i in range(30)]
    grid, _ = calibrate_from_frames(frames)
    assert grid.frames_used == 30
    assert all(h.found for h in grid.holes)


def test_no_sheet_fails_loudly() -> None:
    """An empty desk must raise, not invent a grid."""
    blank = np.full((720, 1280, 3), 90, np.uint8)
    assert detect_reference_points(blank) is None
    with pytest.raises(CalibrationError, match="could not find the paper"):
        calibrate_from_frames([blank])


def test_low_contrast_is_diagnosed_specifically() -> None:
    """A sheet on an equally bright surface gets the real explanation.

    This is the failure the real setup hit: paper and desk at the same
    brightness leave no edge at the boundary, and the message has to say so
    rather than blaming the camera angle.
    """
    from companion.pool.perception.vision.reference import describe_failure, detect_reference_points

    # A sheet drawn straight onto a full-frame surface of nearly the same
    # brightness. Rendered directly rather than through the warp, so the
    # frame has no synthetic border whose edge would supply the very contrast
    # this test is asserting is absent.
    rng = np.random.default_rng(3)
    frame = np.full((720, 1280, 3), 148, np.uint8)
    frame = np.clip(frame + rng.normal(0.0, 3.0, frame.shape),
                    0, 255).astype(np.uint8)
    cv2.rectangle(frame, (420, 130), (860, 620), (150, 150, 150), -1)

    assert detect_reference_points(frame) is None, (
        "a sheet with no contrast against its surface must not be 'found'")
    message = describe_failure(frame).lower()
    assert "darker" in message or "contrast" in message, message


def test_cli_image_path_writes_outputs(frame_and_truth, tmp_path: Path) -> None:
    frame, _ = frame_and_truth
    image_path = tmp_path / "frame.png"
    cv2.imwrite(str(image_path), frame)
    outdir = tmp_path / "out"

    # --surface paper explicitly: these frames are rendered paper, and the
    # CLI now defaults to the pool table.
    assert cli.main(["--image", str(image_path), "--outdir", str(outdir),
                     "--surface", "paper"]) == 0

    grid = json.loads((outdir / "grid.json").read_text(encoding="utf-8"))
    # No reference was supplied, so the grid must NOT claim millimetres.
    assert grid["units"] == "u"
    assert grid["frame"]["width"] == round(sheet_w_mm(), 4)
    assert grid["frame"]["height"] == round(sheet_h_mm(), 4)
    assert 0.0 < grid["frame"]["aspect"] <= 1.0
    assert [c["id"] for c in grid["corners"]] == list(CORNER_IDS)
    assert [h["id"] for h in grid["holes"]] == list(HOLE_IDS)
    assert np.array(grid["homography"]).shape == (3, 3)

    assert cv2.imread(str(outdir / "overlay.png")) is not None

    # Not merely decodable: a fixed mm-based scale wrote a 63x62 px smudge
    # here whenever the run was in normalised units, and `is not None` was
    # happy with it.
    from companion.pool.perception.vision.holes import RECT_LONG_SIDE_PX

    rectified = cv2.imread(str(outdir / "rectified.png"))
    assert rectified is not None
    assert max(rectified.shape[:2]) >= 0.5 * RECT_LONG_SIDE_PX, (
        f"rectified.png is {rectified.shape[1]}x{rectified.shape[0]}")


def _cli_env() -> dict:
    """The CLI runs out of tools/ and imports the installed-in-place package."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_cli_subprocess_runs(frame_and_truth, tmp_path: Path) -> None:
    frame, _ = frame_and_truth
    image_path = tmp_path / "frame.png"
    cv2.imwrite(str(image_path), frame)
    proc = subprocess.run(
        [sys.executable, "tools/htn26_cli.py", "--image", str(image_path),
         "--outdir", str(tmp_path / "cli_out"), "--surface", "paper"],
        cwd=ROOT, capture_output=True, text=True, env=_cli_env())
    assert proc.returncode == 0, proc.stderr
    # Without a reference the tool reports normalised units and says so.
    assert "normalised" in proc.stdout.lower()
    assert "no absolute size is claimed" in proc.stdout.lower()


def test_cli_reports_mm_only_when_given_a_reference(frame_and_truth,
                                                    tmp_path: Path) -> None:
    """Millimetres appear exactly when the user supplies a real measurement.

    The pairing that matters: absolute size is not observable from one
    camera, so the tool must not print mm on its own - but it must use them
    faithfully when told.
    """
    frame, _ = frame_and_truth
    image_path = tmp_path / "frame.png"
    cv2.imwrite(str(image_path), frame)
    proc = subprocess.run(
        [sys.executable, "tools/htn26_cli.py", "--image", str(image_path),
         "--outdir", str(tmp_path / "mm_out"), "--surface", "paper",
         "--reference-mm", "297"],
        cwd=ROOT, capture_output=True, text=True, env=_cli_env())
    assert proc.returncode == 0, proc.stderr
    assert "mm" in proc.stdout.lower()
    assert "normalised" not in proc.stdout.lower()

    grid = json.loads(
        (tmp_path / "mm_out" / "grid.json").read_text(encoding="utf-8"))
    assert grid["units"] == "mm"
    # 297 was pinned to the long side, so it must come back unchanged.
    assert max(grid["frame"]["width"], grid["frame"]["height"]) == 297.0


@pytest.mark.parametrize("width_mm,height_mm", [
    (140.0, 220.0), (210.0, 297.0), (150.0, 150.0), (297.0, 210.0),
])
def test_aspect_ratio_recovered_from_perspective(width_mm: float,
                                                 height_mm: float) -> None:
    """The sheet's true proportions come out of the projection itself.

    This is what makes the millimetre scale honest for unknown paper: the
    apparent ratio in the image is wrong under perspective (a tilted portrait
    sheet can measure wider than it is tall), so the ratio has to be
    recovered from the vanishing points rather than read off.
    """
    from companion.pool.perception.vision.aspect import estimate_aspect_ratio

    obj = np.array([[0, 0, 0], [width_mm, 0, 0],
                    [width_mm, height_mm, 0], [0, height_mm, 0]], np.float64)
    obj -= obj.mean(axis=0)
    focal, size = 1400.0, (1080, 1920)
    K = np.array([[focal, 0, size[1] / 2], [0, focal, size[0] / 2], [0, 0, 1]],
                 np.float64)
    for rx, ry in ((0.5, 0.2), (0.8, 0.35), (0.65, -0.3)):
        projected, _ = cv2.projectPoints(obj, np.array([rx, ry, 0.0]),
                                         np.array([0, 0, 800.0]), K, None)
        quad = projected.reshape(4, 2)
        recovered = estimate_aspect_ratio(quad, size)
        assert recovered == pytest.approx(height_mm / width_mm, rel=0.02), (
            f"{width_mm}x{height_mm} at rx={rx}: got {recovered:.3f}")


def test_measurement_pins_scale_to_one_real_dimension() -> None:
    """One ruler measurement fixes the size; the image fixes the shape."""
    from companion.pool.perception.vision.aspect import sheet_size_mm

    obj = np.array([[0, 0, 0], [140.0, 0, 0], [140.0, 220.0, 0],
                    [0, 220.0, 0]], np.float64)
    obj -= obj.mean(axis=0)
    focal, size = 1400.0, (1080, 1920)
    K = np.array([[focal, 0, size[1] / 2], [0, focal, size[0] / 2], [0, 0, 1]],
                 np.float64)
    projected, _ = cv2.projectPoints(obj, np.array([0.6, 0.25, 0.0]),
                                     np.array([0, 0, 800.0]), K, None)
    quad = projected.reshape(4, 2)

    w, h = sheet_size_mm(quad, size, 140.0, "width")
    assert (w, h) == pytest.approx((140.0, 220.0), rel=0.02)
    w, h = sheet_size_mm(quad, size, 220.0, "height")
    assert (w, h) == pytest.approx((140.0, 220.0), rel=0.02)


def _project_sheet(width_mm: float, height_mm: float, rx: float, ry: float,
                   focal: float = 1600.0, noise: float = 0.0,
                   seed: int = 0) -> tuple[np.ndarray, tuple[int, int]]:
    """Project a known sheet through a known camera, for ground-truth tests."""
    size = (1080, 1920)
    obj = np.array([[0, 0, 0], [width_mm, 0, 0], [width_mm, height_mm, 0],
                    [0, height_mm, 0]], np.float64)
    obj -= obj.mean(axis=0)
    K = np.array([[focal, 0, size[1] / 2], [0, focal, size[0] / 2], [0, 0, 1]],
                 np.float64)
    projected, _ = cv2.projectPoints(obj, np.array([rx, ry, 0.0]),
                                     np.array([0, 0, 700.0]), K, None)
    quad = projected.reshape(4, 2)
    if noise:
        quad = quad + np.random.default_rng(seed).normal(0.0, noise, quad.shape)
    return quad, size


def test_focal_pooled_over_frames_beats_a_single_view() -> None:
    """Many views of one camera measure one focal length far better than one.

    Single-frame focal estimation is hostage to conditioning: on a
    near-straight-on view a 1 px corner error moves the estimate by tens of
    pixels, and the aspect ratio inherits that error. Pooling weighted by
    conditioning is what makes an uncalibrated camera usable.
    """
    from companion.pool.perception.vision.aspect import (estimate_aspect_ratio,
                                 estimate_focal_length_multi)

    width_mm, height_mm, focal = 140.0, 216.0, 1600.0
    rng = np.random.default_rng(5)
    quads, size = [], (1080, 1920)
    for i in range(30):
        rx = 0.15 + 0.30 * rng.random()
        ry = 0.05 + 0.25 * rng.random()
        quad, size = _project_sheet(width_mm, height_mm, rx, ry, focal,
                                    noise=0.8, seed=i)
        quads.append(quad)

    estimated, confidence = estimate_focal_length_multi(quads, size)
    assert confidence > 0.5, f"confidence {confidence}"
    assert estimated == pytest.approx(focal, rel=0.10), estimated

    ratios = [estimate_aspect_ratio(q, size, focal_length=estimated)
              for q in quads]
    recovered = float(np.median(ratios))
    measured_height = width_mm * recovered
    assert measured_height == pytest.approx(height_mm, abs=3.0), (
        f"measured {measured_height:.1f} mm, true {height_mm}")


def test_straight_on_view_reports_low_confidence() -> None:
    """A fronto-parallel view must admit it cannot determine the shape.

    This is the honest-failure case: the sides are nearly parallel in the
    image, so the perspective carries almost no information. Reporting low
    confidence is what stops a confident wrong answer.
    """
    from companion.pool.perception.vision.aspect import estimate_focal_length_multi, focal_conditioning

    quads, size = [], (1080, 1920)
    for i in range(10):
        quad, size = _project_sheet(140.0, 216.0, 0.02, 0.01, noise=0.8,
                                    seed=i)
        quads.append(quad)

    principal = (size[1] / 2.0, size[0] / 2.0)
    assert max(focal_conditioning(q, principal) for q in quads) < 0.35
    _, confidence = estimate_focal_length_multi(quads, size)
    assert confidence < 0.35, f"confidence {confidence} should be low"


def test_measures_an_arbitrary_non_standard_sheet() -> None:
    """Any flat rectangle, not just standard paper sizes."""
    from companion.pool.perception.vision.aspect import (estimate_aspect_ratio,
                                 estimate_focal_length_multi)

    for width_mm, height_mm in ((140.0, 216.0), (95.0, 310.0), (183.0, 183.0)):
        quads, size = [], (1080, 1920)
        for i, (rx, ry) in enumerate(((0.45, 0.18), (0.35, 0.25),
                                      (0.5, 0.1), (0.4, 0.3))):
            quad, size = _project_sheet(width_mm, height_mm, rx, ry,
                                        noise=0.5, seed=i)
            quads.append(quad)
        focal, _ = estimate_focal_length_multi(quads, size)
        ratio = float(np.median([estimate_aspect_ratio(q, size,
                                                       focal_length=focal)
                                 for q in quads]))
        assert width_mm * ratio == pytest.approx(height_mm, abs=3.0), (
            f"{width_mm}x{height_mm}: measured height {width_mm * ratio:.1f}")


def test_blank_frame_is_recognised(frame_and_truth) -> None:
    """A hardware-disabled camera returns flat frames; catch that explicitly."""
    from companion.pool.perception.vision.camera import _is_blank

    frame, _ = frame_and_truth
    grey = np.full((720, 1280, 3), 134, np.uint8)
    assert _is_blank(grey)
    assert not _is_blank(frame)


# --- Marks on the sheet ----------------------------------------------------

# The scrap tests pin the spec themselves. `measure_sheet` installs whatever
# it measures as the active spec, so a test that merely reads `active_spec()`
# inherits whatever the previously-run test happened to leave there - which
# is how these first failed only in a full run.
MARK_SPEC = TableSpec(210.0, 297.0)


@pytest.fixture
def mark_spec():
    """Hold the spec fixed for one test, and restore it afterwards."""
    previous = active_spec()
    set_active_spec(MARK_SPEC)
    try:
        yield MARK_SPEC
    finally:
        set_active_spec(previous)


def _scene_with_marks(positions_mm, **kwargs):
    """A tilted view of the sheet carrying black scraps at known positions."""
    from .synth import (add_clutter, add_marks, render_sheet,
                             simulate_camera_view)

    sheet, corners = render_sheet()
    sheet = add_marks(sheet, corners, positions_mm)
    return simulate_camera_view(add_clutter(sheet, corners), corners, **kwargs)


def _marks_for(positions_mm, **kwargs):
    from companion.pool.perception.vision.calibration import process_frame
    from companion.pool.perception.vision.marks import detect_marks

    frame, _ = _scene_with_marks(positions_mm, **kwargs)
    result = process_frame(frame)
    assert result is not None, "the sheet itself was not found"
    marks, _ = detect_marks(frame, result.homography)
    return marks


def test_finds_three_scraps_where_they_were_placed(mark_spec):
    """The core claim: three scraps, located to within a few millimetres."""
    spec = mark_spec
    placed = [(0.30 * spec.width_mm, 0.25 * spec.height_mm),
              (0.70 * spec.width_mm, 0.45 * spec.height_mm),
              (0.40 * spec.width_mm, 0.75 * spec.height_mm)]
    marks = _marks_for(placed)

    assert len(marks) == 3, f"expected 3 scraps, got {len(marks)}"
    for mark in marks:
        nearest = min(placed,
                      key=lambda q: np.hypot(mark.xy_mm[0] - q[0],
                                             mark.xy_mm[1] - q[1]))
        error = float(np.hypot(mark.xy_mm[0] - nearest[0],
                               mark.xy_mm[1] - nearest[1]))
        assert error < 6.0, f"{mark.id} is {error:.1f} mm from where it was put"


def test_reports_the_fixed_radius_regardless_of_scrap_size(mark_spec):
    """Radius is a reporting convention, not a measurement of the scrap."""
    from companion.pool.perception.vision.marks import MARK_RADIUS_MM

    spec = mark_spec
    marks = _marks_for([(0.5 * spec.width_mm, 0.5 * spec.height_mm)])
    assert len(marks) == 1
    assert marks[0].radius_mm == MARK_RADIUS_MM


def test_a_blank_sheet_yields_no_marks(mark_spec):
    """No scraps means none reported - shading must not become phantom blobs."""
    assert _marks_for([]) == []


def test_marks_are_numbered_down_the_page(mark_spec):
    """Labels follow position, so the same scrap keeps its name between runs."""
    spec = mark_spec
    marks = _marks_for([(0.5 * spec.width_mm, 0.20 * spec.height_mm),
                        (0.5 * spec.width_mm, 0.55 * spec.height_mm),
                        (0.5 * spec.width_mm, 0.85 * spec.height_mm)])
    assert [m.id for m in marks] == ["M1", "M2", "M3"]
    ys = [m.xy_mm[1] for m in marks]
    assert ys == sorted(ys)


def test_the_dark_backing_outside_the_sheet_is_not_a_mark(mark_spec):
    """The board the paper lies on is darker than the paper but is not a scrap.

    This is the failure the edge margin exists to prevent: the rectified view
    carries a band of backing and contact shadow just outside the paper, and
    both pass a "darker than the page" test.
    """
    marks = _marks_for([])
    assert marks == [], f"backing leaked in as {[m.as_dict() for m in marks]}"


def test_marks_land_inside_the_sheet(mark_spec):
    """Every reported position must be on the paper, in table coordinates."""
    spec = mark_spec
    marks = _marks_for([(0.25 * spec.width_mm, 0.30 * spec.height_mm),
                        (0.75 * spec.width_mm, 0.70 * spec.height_mm)])
    assert marks
    for mark in marks:
        x, y = mark.xy_mm
        assert 0.0 <= x <= spec.width_mm, f"{mark.id} x={x:.1f} is off the sheet"
        assert 0.0 <= y <= spec.height_mm, f"{mark.id} y={y:.1f} is off the sheet"


def test_graph_draws_marks_without_disturbing_the_grid(mark_spec):
    """The graph with marks is the same picture, plus the scraps."""
    from companion.pool.perception.vision.calibration import process_frame
    from companion.pool.perception.vision.marks import detect_marks
    from companion.pool.perception.vision.overlay import draw_xy_graph

    spec = mark_spec
    placed = [(0.5 * spec.width_mm, 0.5 * spec.height_mm)]
    frame, _ = _scene_with_marks(placed)
    result = process_frame(frame)
    assert result is not None
    grid, _ = calibrate_from_frames([frame])
    marks, _ = detect_marks(frame, grid.homography)

    plain = draw_xy_graph(grid)
    with_marks = draw_xy_graph(grid, marks=marks)
    assert with_marks.shape == plain.shape
    assert marks, "no scrap to draw"
    assert np.any(with_marks != plain), "the marks were not drawn"


# --- The rectified view's scale --------------------------------------------
#
# The warp is sized from the surface's long side, not from a fixed number of
# pixels per millimetre. The fixed scale was tuned for a sheet of paper and
# broke at both ends: a normalised spec (long side = 1.0, which is what the
# CLI reports unless given a reference) collapsed the surface to 3 px inside
# a margin ten times wider than the surface itself, while a real table in mm
# asked for a 5400x3306 warp on every frame and every mark detection.

RECT_SPECS = [
    TableSpec(1.0, 0.608, "u"),       # normalised - the CLI's default
    TableSpec(1780.0, 1082.0, "mm"),  # a real pool table
    TableSpec(210.0, 297.0, "mm"),    # A4, the scale this was tuned for
    TableSpec(0.05, 0.03, "mm"),      # far below the tuned scale
]


def _homography_onto(spec: TableSpec):
    """Maps an inset rectangle of a 1280x720 frame onto the spec's corners."""
    from companion.pool.perception.vision.homography import fit_homography

    image_quad = np.array([[140.0, 100.0], [1140.0, 100.0],
                           [1140.0, 620.0], [140.0, 620.0]], np.float64)
    return fit_homography(image_quad, spec.corner_quad_mm)


@pytest.mark.parametrize("spec", RECT_SPECS, ids=lambda s: s.describe())
def test_rectified_view_is_sized_from_the_surface(spec: TableSpec) -> None:
    """The warped view is usable at any unit scale, not just paper mm."""
    from companion.pool.perception.vision.holes import (RECT_LONG_SIDE_PX, RECT_MARGIN_FRAC, rectify)

    previous = active_spec()
    set_active_spec(spec)
    try:
        frame = np.full((720, 1280, 3), 200, np.uint8)
        view = rectify(frame, _homography_onto(spec))

        long_px = RECT_LONG_SIDE_PX * (1.0 + 2.0 * RECT_MARGIN_FRAC)
        assert max(view.image.shape[:2]) == pytest.approx(long_px, rel=0.01)

        # The surface must fill the view rather than vanish into the margin:
        # this is the assertion the 3-px-wide table failed.
        x0, y0 = view.mm_to_px(0.0, 0.0)
        x1, y1 = view.mm_to_px(spec.width_mm, spec.height_mm)
        longest = max(spec.width_mm, spec.height_mm)
        assert x1 - x0 == pytest.approx(
            RECT_LONG_SIDE_PX * spec.width_mm / longest, rel=0.01)
        assert y1 - y0 == pytest.approx(
            RECT_LONG_SIDE_PX * spec.height_mm / longest, rel=0.01)
    finally:
        set_active_spec(previous)


def test_units_do_not_change_the_rectified_view() -> None:
    """The same shape rectifies identically in normalised units and in mm.

    Absolute size is not observable from one camera, so the unit system the
    result happens to be reported in must not change the picture the
    pipeline works in - mark detection runs inside this view.
    """
    from companion.pool.perception.vision.holes import rectify

    previous = active_spec()
    frame = np.full((720, 1280, 3), 200, np.uint8)
    views = []
    for spec in (TableSpec(1.0, 0.608, "u"),
                 TableSpec(1780.0, 1082.24, "mm")):
        set_active_spec(spec)
        try:
            views.append(rectify(frame, _homography_onto(spec)))
        finally:
            set_active_spec(previous)

    normalised, millimetres = views
    assert normalised.image.shape == millimetres.image.shape
    # And a point at the same fraction across the surface lands on the same
    # pixel, so mm_to_px is consistent between the two.
    assert normalised.mm_to_px(0.5, 0.304) == pytest.approx(
        millimetres.mm_to_px(890.0, 541.12), rel=1e-6)


# --- The real pool table ---------------------------------------------------
#
# These run against a frame from the actual camera and table, not a render.
# The cloth detector is driven by colour and by how a real table is built
# (cloth wrapping over the cushion onto the rail face), neither of which a
# synthetic scene would reproduce honestly.

REAL_TABLE = FIXTURES / "real_table.png"

# The same table seen from a raised camera: further away, more top-down, and
# with a cardboard box beyond the foot rail in frame. Every generalisation
# failure the detector had showed up here and nowhere else.
REAL_TABLE_ELEVATED = (FIXTURES
                       / "real_table_elevated.png")


@pytest.fixture
def table_frame():
    frame = cv2.imread(str(REAL_TABLE))
    assert frame is not None, f"missing fixture {REAL_TABLE}"
    return frame


@pytest.fixture
def table_surface():
    """Switch to the cloth detector for a test, then switch back."""
    set_active_surface("table")
    yield
    set_active_surface("paper")


def test_finds_the_cloth_on_the_real_table(table_frame, table_surface):
    """The four corners land on the playing surface, not the floor."""
    from companion.pool.perception.vision.cloth import find_cloth_quad

    quad = find_cloth_quad(table_frame)
    assert quad is not None, "the cloth was not found at all"
    h, w = table_frame.shape[:2]
    for x, y in quad:
        assert 0 <= x < w and 0 <= y < h, f"corner ({x:.0f},{y:.0f}) off-frame"

    area = abs(cv2.contourArea(quad.astype(np.float32)))
    assert 0.15 < area / (w * h) < 0.75, f"implausible area {area/(w*h):.1%}"


def test_cloth_quad_is_rectangular_under_perspective(table_frame,
                                                     table_surface):
    """Opposite sides stay comparable - the test that catches a leaked corner.

    A corner dragged out onto the rail face or the floor shows up here long
    before it shows up in the measured size, because it lengthens one side
    while leaving its opposite alone.
    """
    from companion.pool.perception.vision.cloth import find_cloth_quad

    quad = find_cloth_quad(table_frame)
    assert quad is not None
    sides = [float(np.linalg.norm(quad[(i + 1) % 4] - quad[i]))
             for i in range(4)]
    assert min(sides[0], sides[2]) / max(sides[0], sides[2]) > 0.80
    assert min(sides[1], sides[3]) / max(sides[1], sides[3]) > 0.80


def test_balls_do_not_eat_into_the_playing_surface(table_frame,
                                                   table_surface):
    """The rack sits inside the quad; clutter must not shrink the surface.

    The balls and the triangle are dark holes in the cloth mask. If they were
    treated as boundary rather than filled, the measured surface would stop
    at the rack - so this asserts the rack's own centre is well inside.
    """
    from companion.pool.perception.vision.cloth import find_cloth_quad

    quad = find_cloth_quad(table_frame)
    assert quad is not None
    # The rack sits near the middle of this frame.
    assert cv2.pointPolygonTest(quad.astype(np.float32), (585.0, 330.0),
                                False) > 0, "the rack fell outside the quad"
    centre = tuple(float(v) for v in quad.mean(axis=0))
    assert cv2.pointPolygonTest(quad.astype(np.float32), centre, False) > 0


def test_the_table_view_can_determine_its_shape(table_frame, table_surface):
    """The pose must be solvable, which is what the paper setup never was.

    Conditioning near zero means the vanishing points are at infinity and no
    focal length can be recovered - the failure that dogged the paper. A real
    table seen from the side clears it comfortably, and this pins that.
    """
    from companion.pool.perception.vision.aspect import estimate_focal_length, focal_conditioning
    from companion.pool.perception.vision.cloth import find_cloth_quad

    quad = find_cloth_quad(table_frame)
    assert quad is not None
    h, w = table_frame.shape[:2]
    pp = (w / 2.0, h / 2.0)
    assert focal_conditioning(quad, pp) > 0.02
    focal = estimate_focal_length(quad, pp)
    assert focal is not None, "the focal length was not solvable"
    assert 0.2 * w <= focal <= 6.0 * w, f"implausible focal {focal:.0f} px"


def test_the_whole_pipeline_runs_on_the_table(table_frame, table_surface):
    """Detection through to a grid, on the real frame."""
    grid, result = calibrate_from_frames([table_frame])
    assert grid.frames_used == 1
    assert len(grid.holes) == len(HOLE_IDS)
    assert grid.homography.reprojection_error_mm < 1.0


def test_a_paper_scene_is_not_mistaken_for_cloth(frame_and_truth,
                                                 table_surface):
    """The cloth detector must decline a grey paper scene, not guess.

    Cheap confidence would be to return something for any input; the point of
    a colour gate is that it can say no.
    """
    from companion.pool.perception.vision.cloth import describe_failure, find_cloth_quad

    frame, _ = frame_and_truth
    assert find_cloth_quad(frame) is None
    assert "cloth" in describe_failure(frame).lower()


def test_failure_message_names_a_fix(table_surface):
    """A blank frame is diagnosed in terms the user can act on."""
    from companion.pool.perception.vision.cloth import describe_failure

    blank = np.full((720, 1280, 3), 90, np.uint8)
    message = describe_failure(blank).lower()
    assert "cloth" in message or "saturation" in message


# --- Trusting the focal length only when it is worth trusting --------------

LIT_TABLE = FIXTURES / "real_table_lit.png"


@pytest.fixture
def lit_table_frame():
    frame = cv2.imread(str(LIT_TABLE))
    assert frame is not None, f"missing fixture {LIT_TABLE}"
    return frame


def test_two_measured_dimensions_are_used_verbatim(lit_table_frame,
                                                   table_surface):
    """Given both dimensions, nothing is estimated and nothing is altered.

    This is the way out of an ambiguous pose: a near-top-down view cannot
    recover a rectangle's proportions from perspective, so the user measures
    the second side instead of the tool guessing it.
    """
    spec = cli.measure_sheet([lit_table_frame], 140.0, "width", 74.0)
    assert spec is not None
    assert (spec.width_mm, spec.height_mm) == (140.0, 74.0)
    assert cli._LAST_MEASUREMENT.get("measured_both") is True


def test_height_side_reference_is_not_transposed(lit_table_frame,
                                                 table_surface):
    """--reference-side height names which measurement is which."""
    spec = cli.measure_sheet([lit_table_frame], 74.0, "height", 140.0)
    assert spec is not None
    assert (spec.width_mm, spec.height_mm) == (140.0, 74.0)


def test_an_untrustworthy_focal_length_is_not_used(lit_table_frame,
                                                   table_surface):
    """A badly conditioned focal must not reach the aspect-ratio estimate.

    The recovered ratio is very sensitive to the focal length - on this frame
    f=500 gives 0.54 and f=3044 gives 0.60 - and a near-top-down view cannot
    determine the focal length at all. Using it anyway is what turned a 2:1
    table into 140 x 97 mm, so the confidence gate is load-bearing, not
    decoration.
    """
    spec = cli.measure_sheet([lit_table_frame], 140.0, "width")
    assert spec is not None
    confidence = float(cli._LAST_MEASUREMENT.get("confidence") or 0.0)
    assert confidence < cli.MIN_SHAPE_CONFIDENCE, "expected a weak pose here"
    assert cli._LAST_MEASUREMENT.get("focal_used") is False

    # The fallback (apparent ratio) must be closer to the truth than the
    # estimate the bad focal length would have produced.
    from companion.pool.perception.vision.aspect import estimate_aspect_ratio
    focal = cli._LAST_MEASUREMENT["focal"]
    quad = find_quad(lit_table_frame)
    with_bad_focal = estimate_aspect_ratio(quad, lit_table_frame.shape, focal)
    used = cli._LAST_MEASUREMENT["ratio"]
    true_ratio = 74.0 / 140.0
    assert abs(used - true_ratio) < abs(with_bad_focal - true_ratio)


def test_a_well_conditioned_focal_is_still_used() -> None:
    """The gate must not reject good poses: a clear angle still solves.

    Guarding against overcorrection - a confidence gate that refused
    everything would "fix" the bug by never measuring anything.

    This is deliberately not one of the real table frames. Both of those sit
    near top-down at conditioning below 0.05, so neither is an example of a
    pose that *should* pass; this test used to run on one of them and only
    passed because five copies of one frame agreed with each other.
    """
    from companion.pool.perception.vision.aspect import estimate_focal_length_multi, focal_conditioning

    focal = 1600.0
    quads, size = [], (1080, 1920)
    for i, (rx, ry) in enumerate(((0.45, 0.30), (0.40, 0.26),
                                  (0.50, 0.34), (0.42, 0.28))):
        quad, size = _project_sheet(140.0, 216.0, rx, ry, focal,
                                    noise=0.5, seed=i)
        quads.append(quad)

    principal = (size[1] / 2.0, size[0] / 2.0)
    assert max(focal_conditioning(q, principal) for q in quads) > 0.35, (
        "this fixture is meant to be a well-conditioned pose")

    estimated, confidence = estimate_focal_length_multi(quads, size)
    assert confidence >= cli.MIN_SHAPE_CONFIDENCE, f"confidence {confidence}"
    assert estimated == pytest.approx(focal, rel=0.10), estimated


def test_repeating_one_view_does_not_raise_confidence(table_frame,
                                                      table_surface) -> None:
    """Seeing the same thing twice is not a second opinion.

    A camera sitting still on a stand delivers one view sampled over and
    over, so the frames always agree however bad the pose is. Treating that
    agreement as evidence let a duplicated near-top-down frame of the real
    table score 1.00 where the single frame scored 0.17, which put a focal
    length from vanishing points 20,000 px out back into the shape estimate.
    """
    from companion.pool.perception.vision.aspect import estimate_focal_length_multi, focal_conditioning
    from companion.pool.perception.vision.cloth import find_cloth_quad

    quad = find_cloth_quad(table_frame)
    assert quad is not None
    shape = table_frame.shape
    principal = (shape[1] / 2.0, shape[0] / 2.0)
    assert focal_conditioning(quad, principal) < 0.1, (
        "this frame is meant to be a weak, near-top-down pose")

    _, one = estimate_focal_length_multi([quad], shape)
    _, many = estimate_focal_length_multi([quad] * 8, shape)
    assert many <= one + 1e-9, (
        f"duplicating one view raised confidence from {one:.3f} to {many:.3f}")
    assert many < cli.MIN_SHAPE_CONFIDENCE


def test_a_weak_pose_falls_back_to_the_outline_ratio(table_frame,
                                                     table_surface) -> None:
    """A near-top-down burst must measure the shape from the outline.

    The apparent ratio is exactly right for a top-down view, so it is the
    correct answer here - not a degraded one. Trusting the perspective
    instead moved this table from 0.52 to 0.60.
    """
    from companion.pool.perception.vision.aspect import _apparent_ratio
    from companion.pool.perception.vision.cloth import find_cloth_quad

    spec = cli.measure_sheet([table_frame] * 5)
    assert spec is not None
    assert cli._LAST_MEASUREMENT.get("focal_used") is False
    outline = _apparent_ratio(find_cloth_quad(table_frame))
    assert cli._LAST_MEASUREMENT["ratio"] == pytest.approx(outline, rel=0.02)


# --- What a single camera can and cannot measure ---------------------------

def _project_rect(w, h, dist, f_px=1200.0, rx=0.25, ry=0.15):
    """Project a w x h rectangle at `dist`, returning its 4 image corners."""
    obj = np.array([[-w/2, -h/2, 0], [w/2, -h/2, 0],
                    [w/2, h/2, 0], [-w/2, h/2, 0]], np.float64)
    R = (cv2.Rodrigues(np.array([0.0, ry, 0.0]))[0]
         @ cv2.Rodrigues(np.array([rx, 0.0, 0.0]))[0])
    K = np.array([[f_px, 0, 640.0], [0, f_px, 360.0], [0, 0, 1.0]])
    pts, _ = cv2.projectPoints(obj, cv2.Rodrigues(R)[0],
                               np.array([0.0, 0.0, dist]), K, None)
    return pts.reshape(4, 2)


def test_absolute_scale_is_not_observable_from_one_camera():
    """A small surface near the lens and a large one far away look identical.

    This is why the tool reports normalised units by default. It is not a
    limitation of the implementation but of monocular geometry, and stating
    it as a test stops anyone "fixing" it later by inventing a number.
    """
    small = _project_rect(140.0, 74.0, 900.0)
    large = _project_rect(1400.0, 740.0, 9000.0)
    assert np.abs(small - large).max() < 1e-6


def test_shape_is_observable_even_though_scale_is_not():
    """The ratio of the sides survives, and that is what the tool reports."""
    from companion.pool.perception.vision.aspect import estimate_aspect_ratio

    for w, h in ((140.0, 74.0), (210.0, 297.0), (200.0, 100.0)):
        quad = _project_rect(w, h, 900.0)
        recovered = estimate_aspect_ratio(quad, (720, 1280))
        assert abs(recovered - h / w) < 0.01, f"{w}x{h}: got {recovered}"


def test_no_reference_means_no_millimetres_anywhere(frame_and_truth):
    """With nothing supplied, the spec is normalised and says so."""
    frame, _ = frame_and_truth
    spec = cli.measure_sheet([frame])
    assert spec is not None
    assert spec.units == "u"
    assert max(spec.width_mm, spec.height_mm) == pytest.approx(1.0)
    assert "mm" not in spec.describe()


def test_one_reference_sets_the_scale_exactly(frame_and_truth):
    """The supplied number is used verbatim; the shape supplies the rest."""
    frame, _ = frame_and_truth
    spec = cli.measure_sheet([frame], 297.0, "long")
    assert spec is not None
    assert spec.units == "mm"
    assert max(spec.width_mm, spec.height_mm) == pytest.approx(297.0)


def test_normalised_and_scaled_specs_have_the_same_shape(frame_and_truth):
    """Supplying a reference must rescale, never reshape.

    Guards the branch structure in measure_sheet: it would be easy for the
    'long' branch to transpose the sides and go unnoticed, because the
    numbers would still look plausible.
    """
    frame, _ = frame_and_truth
    plain = cli.measure_sheet([frame])
    scaled = cli.measure_sheet([frame], 297.0, "long")
    assert plain is not None and scaled is not None
    assert plain.aspect == pytest.approx(scaled.aspect, abs=1e-6)

# --- The balls on the table ------------------------------------------------
#
# Against a real frame of a real table with a full set spread out on it, hand
# labelled by reading the numbers off 6x crops. A synthetic scene would not
# be honest here: what makes the stripe/solid call hard is the size of the
# number patch on real balls and the gloss highlight of a real ceiling light,
# and a renderer would only reproduce whatever the author already believed.

BALLS_TABLE = FIXTURES / "real_table_balls.png"
BALLS_TABLE_SD = FIXTURES / "real_table_balls_sd.png"

# The same table, balls spread the way a game leaves them, under dimmer light
# than the frame above. This is the one that caught `MIN_BALL_RESPONSE` being
# fitted: at 26 its three dimmest balls scored 24.0, 23.4 and 23.3 and were
# dropped, so a full set reported as 13 - and every set-of-sixteen rule that
# the kinds depend on silently stopped applying.
BALLS_TABLE_SPREAD = FIXTURES / "real_table_spread.png"

# (x, y, kind) as fractions of the long side, origin at the top-left corner.
# The full set: seven solids, seven stripes, the cue and the 8.
#
# The positions are in cloth coordinates, so they moved when the corner fit was
# corrected to sit on the cloth rather than out on the rail: every ball shifted
# by 0.008-0.021, uniformly, because the origin did. The ball each row refers to
# is unchanged - the numbers in the comments still identify them - and the kinds
# are read off the frame by eye, not copied from the detector, which is why two
# of them (the 6 and the 14) disagree with what it currently reports.
BALL_TRUTH = [
    (0.8581, 0.0650, "solid"),    # 3, red
    (0.6856, 0.0931, "solid"),    # 6, green
    (0.1094, 0.1256, "solid"),    # 4, purple
    (0.8931, 0.1706, "stripe"),   # 14, green
    (0.3719, 0.2225, "stripe"),   # 13, orange
    (0.5731, 0.2588, "stripe"),   # 10, blue
    (0.2419, 0.2588, "cue"),
    (0.3488, 0.2681, "solid"),    # 2, blue
    (0.4944, 0.2913, "stripe"),   # 12, purple
    (0.6994, 0.3200, "eight"),
    (0.0850, 0.3350, "stripe"),   # 11, red
    (0.4250, 0.3488, "stripe"),   # 9, yellow
    (0.2381, 0.3863, "solid"),    # 1, yellow
    (0.8931, 0.4125, "stripe"),   # 15, maroon
    (0.5881, 0.4213, "solid"),    # 7, maroon
    (0.3669, 0.4513, "solid"),    # 5, orange
]

# A ball must land within a third of its own radius of where it really is.
BALL_POSITION_TOLERANCE = 0.005


def _detect_on(path: Path):
    """Measure the table in this frame, then find the balls on it."""
    from companion.pool.perception.vision.detector import detect_balls

    frame = cv2.imread(str(path))
    assert frame is not None, f"missing fixture {path}"
    previous = active_surface()
    set_active_surface("table")
    try:
        assert cli.measure_sheet([frame]) is not None
        grid, _ = calibrate_from_frames([frame])
        balls, _ = detect_balls(frame, grid.homography)
        return frame, grid, balls, active_spec()
    finally:
        set_active_surface(previous)


@pytest.fixture(scope="module")
def table_balls():
    """Detected once: the pipeline is deterministic and costs a few seconds."""
    return _detect_on(BALLS_TABLE)


@pytest.fixture(scope="module")
def _balls_geometry():
    """The balls frame and the grid locked from it, measured once."""
    frame = cv2.imread(str(BALLS_TABLE))
    assert frame is not None, f"missing fixture {BALLS_TABLE}"
    previous = active_surface()
    set_active_surface("table")
    try:
        assert cli.measure_sheet([frame]) is not None
        grid, _ = calibrate_from_frames([frame])
        return frame, grid, active_spec()
    finally:
        set_active_surface(previous)


@pytest.fixture
def balls_scene(_balls_geometry):
    """That frame and grid, with the table surface and its spec installed.

    The tests below run detection several times over at different radii, which
    is where the seconds go; the geometry they run against is identical every
    time, so it is locked once and shared.
    """
    frame, grid, spec = _balls_geometry
    previous_surface, previous_spec = active_surface(), active_spec()
    set_active_surface("table")
    set_active_spec(spec)
    try:
        yield frame, grid
    finally:
        set_active_surface(previous_surface)
        set_active_spec(previous_spec)


def _pair_with_truth(balls):
    """Nearest-neighbour match of detections to the hand labels."""
    matched, unmatched, used = [], [], set()
    for x, y, kind in BALL_TRUTH:
        best, distance = None, 1e9
        for i, ball in enumerate(balls):
            d = float(np.hypot(ball.xy_mm[0] - x, ball.xy_mm[1] - y))
            if d < distance:
                best, distance = i, d
        if best is None or best in used or distance > BALL_POSITION_TOLERANCE:
            unmatched.append((x, y, kind))
            continue
        used.add(best)
        matched.append((kind, balls[best], distance))
    spurious = [b for i, b in enumerate(balls) if i not in used]
    return matched, unmatched, spurious


def test_every_ball_is_found_and_nothing_else_is(table_balls):
    """All sixteen, at the right places, with no extras."""
    _, _, balls, _ = table_balls
    matched, unmatched, spurious = _pair_with_truth(balls)
    assert unmatched == [], f"not found: {unmatched}"
    assert spurious == [], (
        f"reported balls that are not there: "
        f"{[(b.id, b.kind, b.xy_mm) for b in spurious]}")
    assert len(matched) == len(BALL_TRUTH)


def test_the_cue_ball_is_picked_out(table_balls):
    """Exactly one cue ball, and it is the white one.

    The comparative test earns its keep here: a stripe lying with a white
    pole up measured 94% white against the cue ball's 98%, so an absolute
    threshold would have to sit inside that gap to work.
    """
    _, _, balls, _ = table_balls
    cues = [b for b in balls if b.kind == "cue"]
    assert len(cues) == 1, f"expected one cue ball, got {len(cues)}"
    truth = next(t for t in BALL_TRUTH if t[2] == "cue")
    assert np.hypot(cues[0].xy_mm[0] - truth[0],
                    cues[0].xy_mm[1] - truth[1]) < BALL_POSITION_TOLERANCE


def test_the_eight_ball_is_picked_out(table_balls):
    """The 8 is found, and dark stripes are not mistaken for it."""
    _, _, balls, _ = table_balls
    eights = [b for b in balls if b.kind == "eight"]
    assert len(eights) == 1, f"expected one 8 ball, got {len(eights)}"
    truth = next(t for t in BALL_TRUTH if t[2] == "eight")
    assert np.hypot(eights[0].xy_mm[0] - truth[0],
                    eights[0].xy_mm[1] - truth[1]) < BALL_POSITION_TOLERANCE


def test_stripes_and_solids_are_told_apart(table_balls):
    """At least twelve of the fourteen, which is what this view allows.

    Not all fourteen, and the number is pinned rather than aspired to: a
    stripe resting with its coloured band square to the camera shows no
    white at all and measures exactly like a solid. That is a property of
    one overhead view, not of the threshold, so the test records the real
    figure instead of a target nobody can reach.

    On this frame the remaining two are the green pair, and they are the
    same ball twice over: the green stripe is the band-on one, so the pair
    is split the wrong way round. Its twin is wrong for the same reason it
    is, which is why the count does not improve by pairing them.
    """
    _, _, balls, _ = table_balls
    matched, _, _ = _pair_with_truth(balls)
    pairs = [(t, b) for t, b, _ in matched if t in ("stripe", "solid")]
    right = [(t, b) for t, b in pairs if b.kind == t]
    assert len(pairs) == 14
    assert len(right) >= 12, (
        "stripe/solid got worse: "
        f"{[(b.id, t, b.kind) for t, b in pairs if b.kind != t]}")


def test_stripe_solid_does_not_depend_on_where_the_cut_sits(table_balls):
    """The colour pairing carries the call, not `STRIPE_RIM_WHITE`.

    This is the reason the pairing is there, and it is a robustness claim
    rather than an accuracy one: the score is 12/14 either way on this
    frame. What changes is that the threshold reaches 12 only inside a
    window measured on this one table - it gives 11/14 at 0.30 and at 0.60 -
    while pairing holds 12 across the whole range, because within a pair the
    cut only has to order two balls of the same colour rather than sit in a
    gap that every table and every light has to share.

    Pinned as a sweep rather than as a single number: a cut that has stopped
    mattering is exactly what this change bought, and a later edit that
    quietly makes the threshold load-bearing again would otherwise pass.
    """
    from companion.pool.perception.vision import detector as detect_balls

    frame, grid, balls, _ = table_balls
    matched, _, spurious = _pair_with_truth(balls)
    assert spurious == [] and len(matched) == len(BALL_TRUTH)

    # The candidates, measured once: only the kinds are being swept, and
    # re-running the search at each cut would measure the same pixels four
    # times over to no purpose.
    radius_mm = detect_balls.ball_radius_mm()
    view = detect_balls.rectify(
        frame, grid.homography,
        px_per_mm=detect_balls.BALL_RADIUS_PX / radius_mm,
        margin_mm=3.0 * radius_mm)
    r = detect_balls.BALL_RADIUS_PX
    field = detect_balls._colour_field(view.image, r)
    region = detect_balls._search_region(view, r)
    # Only candidates that landed on a real ball are scored. The finder may
    # legitimately hand back a candidate that `_classify` never sees as a
    # ball; what this test is about is whether the cut moves the *kinds*.
    found, truths = [], []
    for cx, cy in detect_balls._find_centres(field.delta, region, r):
        stats = detect_balls._measure(field, cx, cy, r)
        if not detect_balls._is_a_ball(stats):
            continue
        x_mm, y_mm = view.px_to_mm(cx, cy)
        nearest = min(BALL_TRUTH,
                      key=lambda t: np.hypot(t[0] - x_mm, t[1] - y_mm))
        if np.hypot(nearest[0] - x_mm, nearest[1] - y_mm) > (
                BALL_POSITION_TOLERANCE):
            continue
        found.append((cx, cy, stats))
        truths.append(nearest[2])
    assert sum(t in ("stripe", "solid") for t in truths) == 14

    original = detect_balls.STRIPE_RIM_WHITE
    scores = {}
    try:
        for cut in (0.30, 0.40, 0.50, 0.60):
            detect_balls.STRIPE_RIM_WHITE = cut
            kinds = detect_balls._classify(found)
            scores[cut] = sum(
                kind == truth
                for (kind, _), truth in zip(kinds, truths)
                if truth in ("stripe", "solid"))
    finally:
        detect_balls.STRIPE_RIM_WHITE = original

    assert min(scores.values()) >= 12, (
        f"stripe/solid became sensitive to the cut again: {scores}")


def test_a_full_set_is_reported_as_seven_of_each(table_balls):
    """Seven stripes and seven solids, because that is what a set contains.

    A consequence of pairing rather than a separate rule: each pair hands out
    one of each, so a complete set cannot come back eight-and-six the way
    fourteen independent threshold calls could. Worth pinning because it is
    the part a player would notice - a table showing eight stripes is wrong
    on its face, whatever the individual confidences say.
    """
    _, _, balls, _ = table_balls
    kinds = Counter(b.kind for b in balls)
    assert kinds == Counter({"solid": 7, "stripe": 7, "cue": 1, "eight": 1})


def test_repeating_a_still_frame_does_not_invent_certainty(balls_scene):
    """Memory must not turn a stale answer into a better one.

    The tempting version of "use more frames" is a burst on a still table,
    and it cannot work: the stripe/solid error is deterministic, so the same
    pose returns the same wrong answer however many times it is measured.
    Pinned because averaging a bias is the obvious thing to reach for and it
    would look like it was working - the confidences would rise while the
    kinds stayed wrong.
    """
    from companion.pool.perception.vision.detector import BallMemory, detect_balls

    frame, grid = balls_scene
    memory = BallMemory()
    scores = []
    for _ in range(3):
        balls, _ = detect_balls(frame, grid.homography, use_vlm=False,
                                memory=memory)
        matched, _, _ = _pair_with_truth(balls)
        pairs = [(t, b) for t, b, _ in matched if t in ("stripe", "solid")]
        scores.append(sum(b.kind == t for t, b in pairs))
    assert scores == [12, 12, 12], (
        f"a still table changed its mind across identical frames: {scores}")


def test_a_better_view_of_a_resting_ball_is_remembered(balls_scene):
    """A ball that has re-posed is settled by the frame that saw it best.

    This is the one thing no single overhead frame can do, and the only
    reason to keep frames at all. The band-on green stripe shows no white
    here and reads as a solid; shown a frame where the same ball at the same
    spot has a pole up, the detector keeps that reading and applies it.

    The twin follows, which is why this recovers two balls rather than one:
    the pairing had split the green pair the wrong way round, so correcting
    either end corrects both.
    """
    from companion.pool.perception.vision.detector import BallMemory, detect_balls

    frame, grid = balls_scene
    stripe_xy = next((x, y) for x, y, k in BALL_TRUTH
                     if (round(x, 4), round(y, 4)) == (0.8931, 0.1706))

    # The same table with that one ball showing a white pole. Everything else
    # is untouched, so anything that changes is attributable to this ball.
    posed = frame.copy()
    px = grid.homography.to_image(np.array([stripe_xy], np.float64))[0]
    cv2.circle(posed, (int(px[0]), int(px[1])), 14, (238, 238, 238), -1)

    memory = BallMemory()
    detect_balls(posed, grid.homography, use_vlm=False, memory=memory)
    balls, _ = detect_balls(frame, grid.homography, use_vlm=False,
                            memory=memory)

    matched, _, _ = _pair_with_truth(balls)
    pairs = [(t, b) for t, b, _ in matched if t in ("stripe", "solid")]
    assert sum(b.kind == t for t, b in pairs) == 14, (
        "the remembered pose did not settle the green pair: "
        f"{[(b.id, t, b.kind) for t, b in pairs if b.kind != t]}")
    assert Counter(b.kind for b in balls) == Counter(
        {"solid": 7, "stripe": 7, "cue": 1, "eight": 1})


def test_a_struck_ball_does_not_keep_its_old_reading(balls_scene):
    """Memory is keyed to a resting position, and a shot clears it.

    A remembered kind describes a pose. Once the ball has been hit that pose
    is gone, so carrying the reading forward would be worse than having no
    memory at all - it would state an old fact confidently about a new
    situation.
    """
    from companion.pool.perception.vision.detector import BallMemory, SAME_BALL_R, ball_radius_mm

    memory = BallMemory()
    radius_mm = ball_radius_mm()
    memory.remember((0.5, 0.2), radius_mm, "stripe", 0.9)

    # Still there: remembered.
    assert memory.recall((0.5, 0.2), radius_mm) == ("stripe", 0.9)

    # Struck, and now resting elsewhere: the old reading is dropped, and the
    # new position starts with no history.
    moved = (0.5 + 4.0 * SAME_BALL_R * radius_mm, 0.2)
    memory.forget_moved([moved], radius_mm)
    assert memory.recall((0.5, 0.2), radius_mm) is None
    assert memory.recall(moved, radius_mm) is None


@needs_fixtures
def test_a_dimmer_frame_of_the_same_table_still_finds_the_whole_set():
    """Sixteen balls under light the response cut was not measured on.

    `MIN_BALL_RESPONSE` is the one cut that decides whether a ball exists at
    all, and everything downstream assumes a whole set: the colour pairing
    needs exactly fourteen object balls, and the cue and the 8 are picked
    comparatively from the balls present. At 26 - fitted on the brighter
    frame - this table's three dimmest balls scored 24.0, 23.4 and 23.3 and
    vanished, and the failure did not look like a threshold problem from
    outside. It looked like bad classification: thirteen balls, no pairing,
    and the 8 awarded to whatever dark ball happened to be found.

    So this pins the count, not the kinds. A missing ball is upstream of
    every other decision in the module.
    """
    from companion.pool.perception.vision.detector import detect_balls

    frame = cv2.imread(str(BALLS_TABLE_SPREAD))
    assert frame is not None, f"missing fixture {BALLS_TABLE_SPREAD}"
    previous = active_surface()
    set_active_surface("table")
    try:
        assert cli.measure_sheet([frame]) is not None
        grid, _ = calibrate_from_frames([frame])
        balls, _ = detect_balls(frame, grid.homography, use_vlm=False)
    finally:
        set_active_surface(previous)

    assert len(balls) == 16, (
        f"expected the full set, got {len(balls)}: "
        f"{sorted(Counter(b.kind for b in balls).items())}")
    kinds = Counter(b.kind for b in balls)
    assert kinds["cue"] == 1 and kinds["eight"] == 1


def test_the_detector_runs_without_a_trained_model(balls_scene):
    """No model is a supported state, not a broken one.

    The network is an extra witness that a fresh checkout does not have: the
    weights are built from `crops/`, which is local and gitignored. Every
    decision the module makes must therefore still be reachable from the
    measurements alone, and the failure modes - no file, a checkpoint from a
    different label order, no torch - all have to land on the same path
    rather than on a traceback in the middle of a live session.
    """
    from companion.pool.perception.vision import detector as detect_balls

    frame, grid = balls_scene
    original = detect_balls._learned_opinion
    detect_balls._learned_opinion = lambda *a, **k: None
    try:
        balls, _ = detect_balls.detect_balls(frame, grid.homography,
                                             use_vlm=False)
    finally:
        detect_balls._learned_opinion = original

    assert len(balls) == len(BALL_TRUTH)
    assert Counter(b.kind for b in balls) == Counter(
        {"solid": 7, "stripe": 7, "cue": 1, "eight": 1})


def test_a_bad_checkpoint_is_refused_rather_than_believed(tmp_path: Path):
    """A model trained against a different label order must not load.

    It would load cleanly and answer confidently, and every answer would be
    permuted - the quietest possible failure. The label tuple is part of the
    file format for that reason, and a mismatch is treated as no model at
    all, which is a state the detector already handles.
    """
    torch = pytest.importorskip("torch")
    from classify.net.kinds import LABELS, load_classifier

    assert load_classifier(tmp_path / "absent.pt") is None

    wrong = tmp_path / "wrong.pt"
    torch.save({"state": {}, "labels": tuple(reversed(LABELS)), "width": 16},
               wrong)
    assert load_classifier(wrong) is None


def test_the_set_rules_still_outrank_the_network(balls_scene):
    """The model advises on stripe/solid; the game decides the rest.

    A network that has seen a few hundred crops of one table can be
    confidently wrong, and the things it could be wrong about here are facts:
    a set holds one cue, one 8, and seven of each kind. Those are settled by
    comparison across the whole set in `_classify`, after the network has
    spoken, and this pins that ordering - a model asserting two 8s must not
    be able to put two on the table.
    """
    from companion.pool.perception.vision import detector as detect_balls

    frame, grid = balls_scene
    # Every crop claimed to be the 8, with total confidence.
    detect_balls_original = detect_balls._learned_opinion
    detect_balls._learned_opinion = (
        lambda rectified, found, r: [("eight", 1.0)] * len(found))
    try:
        balls, _ = detect_balls.detect_balls(frame, grid.homography,
                                             use_vlm=False)
    finally:
        detect_balls._learned_opinion = detect_balls_original

    kinds = Counter(b.kind for b in balls)
    assert kinds["eight"] <= 1, f"the network put {kinds['eight']} 8 balls up"
    assert kinds["cue"] == 1


def test_a_doubtful_call_says_so(table_balls):
    """Confidence tracks the margin, and never claims certainty.

    A stripe that shows no white is indistinguishable from a solid however
    clean the measurement is, so a confident-looking 1.0 here would be a
    claim about the ball rather than about the pixels.
    """
    _, _, balls, _ = table_balls
    for ball in balls:
        assert 0.5 <= ball.confidence <= 1.0
        if ball.kind in ("stripe", "solid"):
            assert ball.confidence <= 0.95


def test_the_pocket_liner_is_not_reported_as_a_ball(table_balls):
    """Bright plastic across a pocket mouth scores like a ball. It is not one.

    On this table the middle-pocket liner sits inside the cloth outline, is
    swallowed by the fill that stops the balls eating into the surface, and
    scored 36 where the real balls scored 30 to 107 - so neither the search
    region nor the response separates it. What does is that it is a mid-grey,
    and a pool set contains no mid-grey ball.
    """
    from companion.pool.perception.vision.detector import BALL_RADIUS_FRAC

    _, _, balls, spec = table_balls
    pocket = holes_mm()["TM"]
    reach = 4.0 * BALL_RADIUS_FRAC * max(spec.width_mm, spec.height_mm)
    intruders = [b for b in balls
                 if np.hypot(b.xy_mm[0] - pocket[0],
                             b.xy_mm[1] - pocket[1]) < reach]
    assert intruders == [], (
        f"pocket hardware reported as balls: "
        f"{[(b.id, b.kind, b.xy_mm) for b in intruders]}")


def test_every_ball_lies_on_the_playing_surface(table_balls):
    """A ball's centre sits a full radius inside the cloth, or it is not a ball.

    This used to allow a centre a whole radius *outside* the surface, which is
    not a position a ball can occupy: resting against a cushion still leaves
    its centre one radius in. The slack was being spent on the cushion, where
    the red cloth wraps over the rail - a table whose radius prior was too
    small reported two solids out there at 0.95 confidence.
    """
    from companion.pool.perception.vision.detector import BALL_RADIUS_FRAC

    _, _, balls, spec = table_balls
    radius = BALL_RADIUS_FRAC * max(spec.width_mm, spec.height_mm)
    for ball in balls:
        x, y = ball.xy_mm
        assert radius <= x <= spec.width_mm - radius, f"{ball.id} x={x}"
        assert radius <= y <= spec.height_mm - radius, f"{ball.id} y={y}"


def test_no_two_balls_are_closer_than_two_balls_can_be(table_balls):
    """Two centres nearer than 2r are one ball found twice.

    Balls are solid, so their centres cannot come closer than 2r however the
    table is racked - which makes this the one check on the detector that needs
    no threshold. Suppression used to run at 1.5r, and through a window half as
    wide as that again, so a single ball could return one detection on its
    number patch and another on its coloured body.
    """
    from companion.pool.perception.vision.detector import BALL_RADIUS_FRAC, MIN_SEPARATION_R

    _, _, balls, spec = table_balls
    radius = BALL_RADIUS_FRAC * max(spec.width_mm, spec.height_mm)
    assert MIN_SEPARATION_R <= 2.0, (
        "suppressing further apart than 2r would discard touching balls")
    for a, b in itertools.combinations(balls, 2):
        gap = float(np.hypot(a.xy_mm[0] - b.xy_mm[0], a.xy_mm[1] - b.xy_mm[1]))
        assert gap >= MIN_SEPARATION_R * radius, (
            f"{a.id} and {b.id} are {gap / radius:.2f} radii apart")


def test_a_wrong_radius_prior_is_reported_rather_than_hidden(balls_scene,
                                                             table_balls):
    """Too small a radius must be called out, not returned as a reading.

    This is the failure that prompted all of the above, and the reason it was
    worth a check of its own: every individual measurement stays
    self-consistent, so nothing inside a candidate can tell. What gives it away
    is the count, because a pool set has sixteen balls and that is a fact about
    the game rather than a tuned number.
    """
    from companion.pool.perception.vision.detector import (BALL_RADIUS_FRAC, describe_inconsistency,
                              detect_balls)

    frame, grid = balls_scene
    _, _, right, _ = table_balls
    assert describe_inconsistency(right, BALL_RADIUS_FRAC) is None, (
        "the correct radius must not be flagged")

    halved = BALL_RADIUS_FRAC / 2.0
    wrong, _ = detect_balls(frame, grid.homography, radius_frac=halved)
    message = describe_inconsistency(wrong, halved)
    assert message is not None, (
        f"{len(wrong)} balls at half the radius went unreported")
    assert "ball-radius-frac" in message, message


def test_a_short_count_on_a_full_table_is_reported(table_balls):
    """Missing balls are said out loud; a game in progress is not.

    The silent version of this is what made a real failure unreadable: a
    frame returned thirteen balls, said nothing about it, and the visible
    symptom was that the cue and the 8 had been awarded to the wrong balls -
    which is what happens when a comparative test is run over an incomplete
    set. The count was the only thing that knew, and it was not asked.

    The floor matters as much as the message. Balls leave a pool table as it
    is played, so warning on every reading after the first pot would be noise.
    """
    from companion.pool.perception.vision.detector import (BALL_RADIUS_FRAC, SET_SIZE, SHORT_COUNT_FLOOR,
                              describe_inconsistency)

    _, _, balls, _ = table_balls
    assert len(balls) == SET_SIZE
    assert describe_inconsistency(balls, BALL_RADIUS_FRAC) is None

    # A nearly-full table missing a few: worth saying.
    for missing in (1, 2, SET_SIZE - SHORT_COUNT_FLOOR):
        short = balls[:SET_SIZE - missing]
        message = describe_inconsistency(short, BALL_RADIUS_FRAC)
        assert message is not None, f"{len(short)} balls went unreported"
        assert str(len(short)) in message

    # A game well under way: not a detection fault, so no warning.
    playing = balls[:SHORT_COUNT_FLOOR - 1]
    assert describe_inconsistency(playing, BALL_RADIUS_FRAC) is None


def test_at_most_one_eight_ball_whatever_the_radius(balls_scene):
    """There is one 8 ball, so the detector may never report two.

    The 8's test is a pair of absolute cuts on how dark and how colourless a
    body is, and more than one ball can clear them - a dark stripe lying
    band-on is dark and colourless too, and on this frame a purple stripe
    measures darker than the 8 itself. With no comparison between the
    candidates a radius prior slightly off reported five.
    """
    from companion.pool.perception.vision.detector import detect_balls

    frame, grid = balls_scene
    for frac in (0.0110, 0.0200, 0.0300):
        balls, _ = detect_balls(frame, grid.homography, radius_frac=frac)
        eights = [b for b in balls if b.kind == "eight"]
        assert len(eights) <= 1, (
            f"radius {frac}: {len(eights)} 8 balls "
            f"{[b.xy_mm for b in eights]}")


def test_a_dark_ball_reads_as_dark_without_a_highlight():
    """How dark a ball is cannot be measured against the ball's own brightest.

    That normalisation is self-referential and inverts: an evenly dark ball
    divides its dark body by an equally dark 90th percentile and comes out
    *light*, so the 8 was only ever found when a gloss highlight happened to
    land on it. It went missing on the first frame where one did not - and
    since the reading was internally consistent, it went missing quietly.

    Built by hand rather than from a photo because the claim is about the
    arithmetic, not about what a real 8 ball looks like: the same dark body has
    to read the same whether or not something bright sits next to it.
    """
    from companion.pool.perception.vision.detector import (EIGHT_BODY_DARKNESS, RIM_OUTER_R, _ColourField,
                              _measure)

    size, r, white_reference = 140, 24.0, 173.0

    def field_with(highlight: bool) -> _ColourField:
        lightness = np.full((size, size), 50.0, np.float32)   # the cloth
        chroma = np.full((size, size), 40.0, np.float32)      # red cloth
        ys, xs = np.mgrid[:size, :size]
        disc = np.hypot(xs - size / 2, ys - size / 2) <= RIM_OUTER_R * r
        lightness[disc] = 20.0                               # a dark ball
        chroma[disc] = 2.0                                   # and colourless
        if highlight:
            spot = np.hypot(xs - size / 2, ys - size / 2 + 10) <= 0.25 * r
            lightness[spot] = 250.0
        cloth = np.full((size, size), 50.0, np.float32)
        return _ColourField(
            lightness=lightness, chroma=chroma, cloth_lightness=cloth,
            delta=np.where(disc, 60.0, 0.0).astype(np.float32),
            white=np.zeros((size, size), bool), white_lightness=white_reference,
            bgr=np.zeros((size, size, 3), np.uint8))

    with_spot = _measure(field_with(True), size / 2, size / 2, r)
    without = _measure(field_with(False), size / 2, size / 2, r)

    assert without["body_darkness"] <= EIGHT_BODY_DARKNESS, (
        f"an evenly dark ball read as {without['body_darkness']:.2f}, which is "
        f"lighter than the 8's cut of {EIGHT_BODY_DARKNESS}")
    assert with_spot["body_darkness"] == pytest.approx(
        without["body_darkness"], abs=0.02), (
        "a gloss highlight changed how dark the body measured")


def test_one_fixed_radius_for_every_ball(table_balls):
    """Every ball is reported at the same radius, set by the table's size."""
    from companion.pool.perception.vision.detector import BALL_RADIUS_FRAC

    _, _, balls, spec = table_balls
    expected = BALL_RADIUS_FRAC * max(spec.width_mm, spec.height_mm)
    assert balls
    for ball in balls:
        assert ball.radius_mm == pytest.approx(expected)


def test_the_radius_follows_the_surface_not_the_units():
    """r is a fraction of the long side, so it scales with what was measured.

    The tool reports normalised units unless given a real measurement, and
    the plotted ball has to stay the right size in both.
    """
    from companion.pool.perception.vision.detector import BALL_RADIUS_FRAC, ball_radius_mm

    previous = active_spec()
    try:
        set_active_spec(TableSpec(1.0, 0.5, "u"))
        assert ball_radius_mm() == pytest.approx(BALL_RADIUS_FRAC)
        set_active_spec(TableSpec(2540.0, 1270.0, "mm"))
        assert ball_radius_mm() == pytest.approx(2540.0 * BALL_RADIUS_FRAC)
        # And the caller can override it for a table with a different ratio.
        assert ball_radius_mm(0.0113) == pytest.approx(2540.0 * 0.0113)
    finally:
        set_active_spec(previous)


@needs_fixtures
def test_a_lower_resolution_frame_gives_the_same_answer():
    """The same scene at 720p, which is what the CLI captures by default.

    Guards against the detector quietly depending on the extra pixels: the
    rectified view is scaled to a fixed ball radius, so the thresholds are
    the same count of pixels either way.
    """
    _, _, hd, _ = _detect_on(BALLS_TABLE)
    _, _, sd, _ = _detect_on(BALLS_TABLE_SD)
    assert len(sd) == len(hd)
    matched, unmatched, spurious = _pair_with_truth(sd)
    assert unmatched == [] and spurious == []
    right = sum(1 for t, b, _ in matched if b.kind == t)
    assert right >= 14, [(b.id, t, b.kind) for t, b, _ in matched
                         if b.kind != t]


def test_graph_draws_the_balls_without_disturbing_the_grid(table_balls):
    """The graph with balls is the same picture, plus the balls."""
    from companion.pool.perception.vision.overlay import draw_xy_graph

    _, grid, balls, spec = table_balls
    previous = active_spec()
    set_active_spec(spec)
    try:
        plain = draw_xy_graph(grid, width=900)
        with_balls = draw_xy_graph(grid, width=900, balls=balls)
        assert with_balls.shape == plain.shape
        assert balls, "no ball to draw"
        assert np.any(with_balls != plain), "the balls were not drawn"
    finally:
        set_active_spec(previous)


@needs_fixtures
def test_cli_image_path_writes_the_ball_outputs(tmp_path: Path):
    """The saved-frame path produces the same artefacts as the live B key."""
    previous = active_surface()
    set_active_surface("table")
    try:
        code = cli.main(["--image", str(BALLS_TABLE),
                         "--outdir", str(tmp_path), "--surface", "table"])
    finally:
        set_active_surface(previous)
    assert code == 0

    payload = json.loads((tmp_path / "balls.json").read_text(encoding="utf-8"))
    assert payload["counts"]["cue"] == 1
    assert payload["counts"]["eight"] == 1
    assert len(payload["balls"]) == len(BALL_TRUTH)
    assert payload["radius"] == pytest.approx(
        payload["radius_frac_of_long_side"] * 1.0)
    for ball in payload["balls"]:
        assert ball["kind"] in ("cue", "eight", "stripe", "solid")
        assert len(ball["xy_mm"]) == 2
    assert (tmp_path / "balls_graph.png").exists()
    assert (tmp_path / "balls_overlay.png").exists()


@pytest.fixture
def elevated_frame():
    frame = cv2.imread(str(REAL_TABLE_ELEVATED))
    assert frame is not None, f"missing fixture {REAL_TABLE_ELEVATED}"
    return frame


def _cloth_fraction_inside(quad, frame):
    """How much of the quad is actually the colour of the cloth.

    The measure the side-ratio tests cannot make. A quad that has run off the
    cloth onto something else still has matching opposite sides if it grew
    symmetrically, and that is exactly how the raised-camera failure looked:
    ratios 0.93/0.97, and a cardboard box inside the quad.
    """
    from companion.pool.perception.vision.cloth import cloth_hue

    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (7, 7), 0), cv2.COLOR_BGR2HSV)
    peak = cloth_hue(hsv)
    assert peak is not None, "no cloth hue could be read from the frame"
    distance = np.abs(hsv[..., 0].astype(np.int16) - peak)
    distance = np.minimum(distance, 180 - distance)
    is_cloth = (distance <= 12) & (hsv[..., 1] >= 50)

    inside = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillConvexPoly(inside, quad.astype(np.int32), 255)
    total = int(np.count_nonzero(inside))
    assert total > 0
    return float(np.count_nonzero(is_cloth & (inside > 0))) / total


def test_the_quad_holds_cloth_and_not_the_room(elevated_frame, table_surface):
    """Raised camera, table further off, a cardboard box beyond the rail.

    This is the regression test for the failure that a fixed colour gate
    could not survive. The box sits at saturation 115 - under the old
    absolute floor of 70 it read as cloth, and a close sized to the frame
    then welded it to the table, so the "largest cloth region" spanned both
    and the bottom corners landed on the frame edge.

    It is asserted as cloth *inside* the quad rather than as side ratios
    because the broken quad passed every ratio test there was: it grew
    symmetrically, so its opposite sides stayed matched while it swallowed
    the box. Filled area is what tells the two apart - the bad quad was 63%
    cloth, a good one is well over 90%.
    """
    from companion.pool.perception.vision.cloth import find_cloth_quad

    quad = find_cloth_quad(elevated_frame)
    assert quad is not None, "the cloth was not found on the raised view"
    fraction = _cloth_fraction_inside(quad, elevated_frame)
    assert fraction > 0.90, (
        f"only {fraction:.1%} of the quad is cloth - it has run off the "
        "table onto something else")


def test_the_cloth_is_found_at_either_camera_height(table_frame,
                                                    elevated_frame,
                                                    table_surface):
    """One detector, two camera heights, no per-setup tuning.

    Both frames are the same table, so whatever the pipeline measures from
    the corners has to agree between them. The aspect ratio is the thing to
    compare: it is what the corners are for, and it is unitless, so it does
    not care that the table covers 42% of one frame and 25% of the other.
    """
    from companion.pool.perception.vision.cloth import find_cloth_quad

    ratios = []
    for frame in (table_frame, elevated_frame):
        quad = find_cloth_quad(frame)
        assert quad is not None
        assert _cloth_fraction_inside(quad, frame) > 0.90
        sides = [float(np.linalg.norm(quad[(i + 1) % 4] - quad[i]))
                 for i in range(4)]
        long_side = max(sides[0], sides[2])
        short_side = max(sides[1], sides[3])
        ratios.append(short_side / long_side)

    assert ratios[0] == pytest.approx(ratios[1], abs=0.08), (
        f"the same table measured {ratios[0]:.3f} from one height and "
        f"{ratios[1]:.3f} from another")


def test_a_cloth_coloured_distractor_does_not_move_the_corners(table_frame,
                                                               table_surface):
    """Something the colour of the cloth, off the table, is not the table.

    The old gate admitted anything in a broad hue band above a fixed
    saturation, so a red object near a burgundy table joined the mask and
    the close could bridge to it. Choosing the component before closing is
    what makes this a non-event: the patch is its own blob and loses.
    """
    from companion.pool.perception.vision.cloth import find_cloth_quad

    before = find_cloth_quad(table_frame)
    assert before is not None

    cluttered = table_frame.copy()
    h, w = cluttered.shape[:2]
    patch = cluttered[int(0.45 * h):int(0.52 * h), int(0.40 * w):int(0.60 * w)]
    cluttered[h - patch.shape[0]:, :patch.shape[1]] = patch

    after = find_cloth_quad(cluttered)
    assert after is not None, "the cloth was lost once a distractor appeared"
    moved = float(np.abs(after - before).max())
    diagonal = float(np.linalg.norm(before[2] - before[0]))
    assert moved < 0.05 * diagonal, (
        f"a distractor off the table moved a corner by {moved:.0f} px")
