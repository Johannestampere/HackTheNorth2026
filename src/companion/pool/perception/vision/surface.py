"""Which surface the pipeline is looking at, and the detector for it.

The pipeline measures a flat rectangle and reports millimetres on it. What
that rectangle is made of only matters at one point: finding its four corners
in the image. This module is that point.

Two detectors exist because the evidence is opposite. Paper is bright against
a darker desk and is found by brightness; a pool table is dark cloth whose
distinguishing feature is colour, and brightness alone would pick the floor.
Neither generalises to the other, so rather than one function with a mode
flag, there are two modules and this switch.
"""

from __future__ import annotations

import numpy as np

from companion.pool.perception.vision import cloth as detect_cloth
from companion.pool.perception.vision import reference as detect_reference

SURFACES = ("paper", "table")
DEFAULT_SURFACE = "table"

_active = DEFAULT_SURFACE


def active_surface() -> str:
    return _active


def set_active_surface(name: str) -> None:
    """Choose the detector for the rest of the run."""
    if name not in SURFACES:
        raise ValueError(f"unknown surface {name!r}; expected one of "
                         f"{', '.join(SURFACES)}")
    global _active
    _active = name


def _module():
    return detect_cloth if _active == "table" else detect_reference


def detect_reference_points(image: np.ndarray):
    """The active surface's 4 corners as (image_pts, table_pts), or None."""
    return _module().detect_reference_points(image)


def find_quad(image: np.ndarray) -> np.ndarray | None:
    """The active surface's 4 corners in the image, ordered TL, TR, BR, BL."""
    if _active == "table":
        return detect_cloth.find_cloth_quad(image)
    return detect_reference.find_paper_quad(image)


def describe_failure(image: np.ndarray) -> str:
    """Why the active detector found nothing, in terms the user can act on."""
    return _module().describe_failure(image)
