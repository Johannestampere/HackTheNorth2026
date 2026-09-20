"""Offline checks for the Baseten classifier. No network, no key required."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
import sys


from companion.pool.perception.vision.classify.baseten import (CROP_PX, _enforce_unique, encode_crop,
                              kind_from_number, merge_with_opencv,
                              vlm_wanted)
from companion.pool.perception.vision.envutil import load_dotenv


def test_kind_follows_the_number_not_the_models_guess() -> None:
    assert kind_from_number("stripe", 3) == "solid"
    assert kind_from_number("solid", 12) == "stripe"
    assert kind_from_number("solid", 8) == "eight"
    assert kind_from_number("cue", None) == "cue"
    assert kind_from_number("stripe", None) == "stripe"


def test_vlm_stays_off_under_pytest_without_an_explicit_flag() -> None:
    os.environ.pop("BASETEN_CLASSIFY", None)
    os.environ.pop("BASETEN_API_KEY", None)
    assert vlm_wanted() is False
    os.environ["BASETEN_API_KEY"] = "not-a-real-key"
    try:
        assert vlm_wanted() is False, "a key alone must not enable the slow path"
        os.environ["BASETEN_CLASSIFY"] = "1"
        assert vlm_wanted() is True
    finally:
        os.environ.pop("BASETEN_CLASSIFY", None)
        os.environ.pop("BASETEN_API_KEY", None)


def test_dotenv_does_not_override_a_real_environment(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text('"BASETEN_API_KEY"="from-file"\n', encoding="utf-8")
    os.environ["BASETEN_API_KEY"] = "from-shell"
    try:
        load_dotenv(dotenv)
        assert os.environ["BASETEN_API_KEY"] == "from-shell"
    finally:
        os.environ.pop("BASETEN_API_KEY", None)


def test_dotenv_strips_quoted_keys(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text('"BASETEN_API_KEY"="from-file"\n', encoding="utf-8")
    os.environ.pop("BASETEN_API_KEY", None)
    try:
        load_dotenv(dotenv)
        assert os.environ["BASETEN_API_KEY"] == "from-file"
    finally:
        os.environ.pop("BASETEN_API_KEY", None)


def test_two_eights_and_two_cues_collapse_to_one_each() -> None:
    found = [
        (0, 0, {"body_darkness": 0.40, "white": 0.30}),
        (0, 0, {"body_darkness": 0.10, "white": 0.20}),
        (0, 0, {"body_darkness": 0.80, "white": 0.97}),
        (0, 0, {"body_darkness": 0.70, "white": 0.85}),
    ]
    rows = [
        ("eight", 0.9, 8),
        ("eight", 0.9, 8),
        ("cue", 0.9, None),
        ("cue", 0.9, None),
    ]
    out = _enforce_unique(rows, found)
    assert [r[0] for r in out] == ["solid", "eight", "cue", "stripe"]


def test_merge_keeps_a_unique_number_and_otherwise_the_rim_test() -> None:
    found = [(0, 0, {"body_darkness": 0.2, "white": 0.1})] * 4
    vlm = [
        ("solid", 0.9, 3),
        ("stripe", 0.9, 9),
        ("solid", 0.9, 6),
        ("solid", 0.9, 3),
    ]
    opencv = [
        ("solid", 0.9),
        ("solid", 0.9),
        ("solid", 0.9),
        ("solid", 0.9),
    ]
    out = merge_with_opencv(vlm, opencv, found)
    assert [r[0] for r in out] == ["solid", "solid", "solid", "solid"]
    assert out[1][2] is None          # unique 9 disagreed with solid
    assert out[2] == ("solid", 0.9, 6)  # unique 6 agreed
    assert out[0][2] is None          # duplicated 3 is not a reading


def test_crop_jpeg_is_tiny() -> None:
    image = np.full((80, 80, 3), 180, np.uint8)
    url = encode_crop(image)
    assert url.startswith("data:image/jpeg;base64,")
    payload = url.split(",", 1)[1]
    # 64 px, quality 40: a few kilobytes, not a camera frame.
    assert len(payload) < 8000
    assert CROP_PX == 64
