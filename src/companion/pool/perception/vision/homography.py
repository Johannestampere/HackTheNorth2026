"""Image <-> table-plane homography."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


class HomographyError(RuntimeError):
    """The image-to-table homography could not be fitted."""


@dataclass
class Homography:
    """Maps image pixels to table millimetres on the paper plane.

    `camera_matrix` and `dist_coeffs` are accepted but unused for now: with no
    lens calibration the mild wide-angle distortion is absorbed into the fit.
    Passing them later lets `undistort_points` run before the fit without any
    change to callers.
    """

    H: np.ndarray            # 3x3, image -> table mm
    H_inv: np.ndarray        # 3x3, table mm -> image
    reprojection_error_mm: float
    inliers: int
    total_points: int
    camera_matrix: np.ndarray | None = None
    dist_coeffs: np.ndarray | None = None

    def to_table(self, pts: np.ndarray) -> np.ndarray:
        """Image px (N,2) -> table mm (N,2)."""
        return _apply(self.H, pts)

    def to_image(self, pts: np.ndarray) -> np.ndarray:
        """Table mm (N,2) -> image px (N,2)."""
        return _apply(self.H_inv, pts)


def _apply(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, M).reshape(-1, 2)


def fit_homography(
    image_pts: np.ndarray,
    table_pts: np.ndarray,
    *,
    camera_matrix: np.ndarray | None = None,
    dist_coeffs: np.ndarray | None = None,
    ransac_reproj_mm: float = 2.0,
) -> Homography:
    """Fit image -> table mm from matched points, with the error in mm."""
    image_pts = np.asarray(image_pts, np.float64).reshape(-1, 2)
    table_pts = np.asarray(table_pts, np.float64).reshape(-1, 2)
    if len(image_pts) != len(table_pts):
        raise HomographyError("image and table point counts differ")
    if len(image_pts) < 4:
        raise HomographyError(
            f"need at least 4 matched points, got {len(image_pts)}")

    if camera_matrix is not None:
        image_pts = cv2.undistortPoints(
            image_pts.reshape(-1, 1, 2), camera_matrix, dist_coeffs,
            P=camera_matrix).reshape(-1, 2)

    H, mask = cv2.findHomography(image_pts, table_pts, cv2.RANSAC,
                                 ransac_reproj_mm)
    if H is None:
        raise HomographyError(
            "findHomography failed - reference points are degenerate "
            "(collinear or coincident)")

    projected = _apply(H, image_pts)
    errors = np.linalg.norm(projected - table_pts, axis=1)
    inliers = int(mask.sum()) if mask is not None else len(image_pts)
    return Homography(
        H=H,
        H_inv=np.linalg.inv(H),
        reprojection_error_mm=float(errors.mean()),
        inliers=inliers,
        total_points=len(image_pts),
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )
