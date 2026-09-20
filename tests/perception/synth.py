"""Render synthetic camera views of a plain sheet, for camera-free testing.

Models the real setup: a white A4 sheet on a darker surface, seen from a
tilted laptop webcam, with the corner tape, clutter and lighting gradient that
the detector has to cope with.
"""

from __future__ import annotations

import cv2
import numpy as np

from companion.pool.perception.vision.table_spec import active_spec

CAMERA_W, CAMERA_H = 1280, 720
PX_PER_MM = 4.0

# What a rendered sheet measures when nothing else says.
DEFAULT_SHEET_W_MM = 210.0
DEFAULT_SHEET_H_MM = 297.0


def render_sheet(*, tape: bool = True, paper_grey: int = 235,
                 desk_grey: int = 70, width_mm: float | None = None,
                 height_mm: float | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """A flat top-down sheet on its backing, plus its 4 corners in that image.

    Defaults to the active spec's size, but any size can be rendered - the
    pipeline is meant to measure whatever paper it is shown.
    """
    # A concrete default size in mm. The active spec is NOT used as a
    # fallback any more: it is normalised (long side = 1.0) unless the user
    # supplied a real measurement, and rendering a 1 x 0.5 "mm" sheet gives a
    # 4 x 2 pixel image. A renderer needs real dimensions; what the pipeline
    # reports afterwards is a separate question.
    spec = active_spec()
    use_spec = spec.units == "mm"
    sheet_w = width_mm if width_mm is not None else (
        spec.width_mm if use_spec else DEFAULT_SHEET_W_MM)
    sheet_h = height_mm if height_mm is not None else (
        spec.height_mm if use_spec else DEFAULT_SHEET_H_MM)
    margin_mm = 60.0
    w = int(round((sheet_w + 2 * margin_mm) * PX_PER_MM))
    h = int(round((sheet_h + 2 * margin_mm) * PX_PER_MM))

    img = np.full((h, w, 3), desk_grey, np.uint8)
    # Desk texture, so the backing is not unnaturally uniform.
    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 6.0, (h, w, 1))
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    x0 = int(round(margin_mm * PX_PER_MM))
    y0 = int(round(margin_mm * PX_PER_MM))
    x1 = x0 + int(round(sheet_w * PX_PER_MM))
    y1 = y0 + int(round(sheet_h * PX_PER_MM))
    cv2.rectangle(img, (x0, y0), (x1, y1), (paper_grey,) * 3, -1)

    # Soft contact shadow just outside the sheet, as in a real photo.
    shadow = np.zeros((h, w), np.uint8)
    cv2.rectangle(shadow, (x0 - 6, y0 - 6), (x1 + 6, y1 + 6), 255, 12)
    shadow = cv2.GaussianBlur(shadow, (31, 31), 0)
    img = np.clip(img.astype(np.float32)
                  - shadow[..., None].astype(np.float32) * 0.12,
                  0, 255).astype(np.uint8)

    if tape:
        # Tape triangles across each corner, like the photo.
        side = int(round(18 * PX_PER_MM))
        for cx, cy, sx, sy in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                               (x1, y1, -1, -1), (x0, y1, 1, -1)):
            tri = np.array([[cx, cy + sy * side], [cx + sx * side, cy],
                            [cx, cy]], np.int32)
            cv2.fillPoly(img, [tri], (168, 168, 168))

    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)
    return img, corners


def add_marks(img: np.ndarray, corners: np.ndarray,
              positions_mm: list[tuple[float, float]], *,
              size_mm: float = 18.0, grey: int = 28) -> np.ndarray:
    """Black paper scraps on the sheet at the given table-mm positions.

    Drawn as rotated, slightly irregular quads rather than neat squares: the
    real scraps are torn, and a detector tuned on perfect rectangles would
    not be tested by them.
    """
    out = img.copy()
    x0, y0 = corners[0]
    rng = np.random.default_rng(11)
    for x_mm, y_mm in positions_mm:
        cx = x0 + x_mm * PX_PER_MM
        cy = y0 + y_mm * PX_PER_MM
        half = size_mm * PX_PER_MM / 2.0
        angle = rng.uniform(0.0, np.pi / 2.0)
        pts = []
        for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            jitter = rng.uniform(0.82, 1.0)
            px = dx * half * jitter
            py = dy * half * jitter
            pts.append([cx + px * np.cos(angle) - py * np.sin(angle),
                        cy + px * np.sin(angle) + py * np.cos(angle)])
        cv2.fillPoly(out, [np.array(pts, np.int32)], (grey,) * 3)
    return out


def add_clutter(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Distractors placed in the margin, never overlapping the sheet.

    Clutter that crossed the sheet would not be a fair test: it would be
    testing occlusion, which the detector does not claim to handle, rather
    than its ability to pick the sheet out of a busy desk.
    """
    out = img.copy()
    h, w = out.shape[:2]
    left = int(corners[:, 0].min())
    top = int(corners[:, 1].min())

    # A second bright page in the top-left margin, clear of the sheet.
    x1 = max(4, min(left - 20, int(w * 0.18)))
    y1 = max(4, min(top - 20, int(h * 0.14)))
    if x1 > 12 and y1 > 12:
        cv2.rectangle(out, (4, 4), (x1, y1), (240, 240, 240), -1)
    # A pencil, also in the left margin.
    if left > 60:
        cv2.line(out, (12, int(h * 0.45)), (min(left - 12, 60), int(h * 0.60)),
                 (70, 60, 120), 9)
    return out


def simulate_camera_view(sheet: np.ndarray, corners: np.ndarray, *,
                         tilt: float = 1.0, blur: int = 3, noise: float = 4.0,
                         seed: int = 0, gradient: float = 0.18
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Warp a top-down render into a tilted webcam view.

    Returns the frame and the sheet's 4 corners in frame pixels, so a test can
    check detected corners against ground truth.
    """
    h, w = sheet.shape[:2]
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)

    inset_top = 0.20 * tilt
    inset_bottom = 0.05 * tilt
    top_y = 0.10 * tilt
    bottom_y = 1.0 - 0.04 * tilt
    dst = np.array([
        [CAMERA_W * inset_top, CAMERA_H * top_y],
        [CAMERA_W * (1.0 - inset_top), CAMERA_H * top_y],
        [CAMERA_W * (1.0 - inset_bottom), CAMERA_H * bottom_y],
        [CAMERA_W * inset_bottom, CAMERA_H * bottom_y],
    ], np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    frame = cv2.warpPerspective(sheet, M, (CAMERA_W, CAMERA_H),
                                flags=cv2.INTER_AREA,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(120, 120, 120))

    if gradient:
        # Uneven lighting across the frame, which is what breaks a single
        # global threshold.
        ramp = np.linspace(1.0 + gradient, 1.0 - gradient, CAMERA_W,
                           dtype=np.float32)
        frame = np.clip(frame.astype(np.float32) * ramp[None, :, None],
                        0, 255).astype(np.uint8)
    if blur:
        frame = cv2.GaussianBlur(frame, (blur | 1, blur | 1), 0)
    if noise:
        rng = np.random.default_rng(seed)
        frame = np.clip(frame.astype(np.float32)
                        + rng.normal(0.0, noise, frame.shape),
                        0, 255).astype(np.uint8)

    warped_corners = cv2.perspectiveTransform(
        corners.reshape(-1, 1, 2), M).reshape(-1, 2)
    return frame, warped_corners


def scene(**kwargs) -> tuple[np.ndarray, np.ndarray]:
    """The default test scene: tape, clutter, tilt, gradient, blur, noise."""
    sheet, corners = render_sheet()
    return simulate_camera_view(add_clutter(sheet, corners), corners, **kwargs)
