"""Classify found balls with a hosted vision model. OpenCV still finds them.

One request, tiny JPEGs, Flash, `detail=low`. That is the cheap path: tokens
scale with pixels and with round-trips, not with how clever the prompt is.

Does not load `.env` on import. Callers that want the file (`main.py`, the
smoke test) put it in `os.environ` first. Pytest therefore stays offline.
"""

from __future__ import annotations

import base64
import os
import time
from collections import Counter
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field

# Flash is cheapest per call but thinks by default and burns the token cap
# once more than one crop is in the request. One-ball smoke stays on Flash.
# The table uses GLM-5.2-Fast: vision, thinking off unless asked, 8 images/call.
SMOKE_MODEL = "zai-org/GLM-5.3-Flash"
TABLE_MODEL = "zai-org/GLM-5.2-Fast"
MODEL = SMOKE_MODEL
CROP_PX = 64
JPEG_QUALITY = 40
TABLE_CROP_PX = 128
TABLE_JPEG_QUALITY = 70
MAX_TOKENS_ONE = 160
MAX_TOKENS_BATCH = 400
BATCH = 8

KINDS = ("cue", "eight", "stripe", "solid")


class BallCall(BaseModel):
    index: int = Field(default=0, ge=0)
    kind: Literal["cue", "eight", "stripe", "solid"]
    number: int | None = Field(default=None, ge=1, le=15)


class TableCall(BaseModel):
    balls: list[BallCall]


def api_key() -> str | None:
    key = os.environ.get("BASETEN_API_KEY", "").strip()
    return key or None


def vlm_wanted() -> bool:
    """Whether this process should spend a Baseten call.

    Off unless `BASETEN_CLASSIFY=1`. A key in `.env` is not enough — that is
    what made live `B` wait tens of seconds on the network. The live key is
    OpenCV; `V` or `--vlm` opts into the model.
    """
    flag = os.environ.get("BASETEN_CLASSIFY", "").strip().lower()
    if flag not in ("1", "true", "on", "yes"):
        return False
    return bool(api_key())


def kind_from_number(kind: str, number: int | None) -> str:
    """The number on a pool ball is the kind. Colour is not.

    1-7 solid, 8 the eight, 9-15 stripe. The model's `kind` is only used when
    no number is visible (the cue, or a ball face-down).
    """
    if number == 8:
        return "eight"
    if number is not None and 1 <= number <= 7:
        return "solid"
    if number is not None and 9 <= number <= 15:
        return "stripe"
    return kind if kind in KINDS else "solid"


def encode_crop(bgr: np.ndarray, *, size: int = CROP_PX,
                quality: int = JPEG_QUALITY) -> str:
    """A small JPEG data URL. `detail=low` plus 64 px keeps the tile count down."""
    small = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY),
                                           quality])
    if not ok:
        raise RuntimeError("could not encode a ball crop as JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def crop_disc(image: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray:
    """A square around one centre, padded with cloth-dark if it hits the edge."""
    half = max(8, int(round(1.7 * r)))
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    x1, y1 = x0 + 2 * half, y0 + 2 * half
    h, w = image.shape[:2]
    crop = np.full((2 * half, 2 * half, 3), (20, 20, 40), np.uint8)
    src_x0, src_y0 = max(0, x0), max(0, y0)
    src_x1, src_y1 = min(w, x1), min(h, y1)
    dst_x0, dst_y0 = src_x0 - x0, src_y0 - y0
    patch = image[src_y0:src_y1, src_x0:src_x1]
    if patch.size:
        crop[dst_y0:dst_y0 + patch.shape[0],
             dst_x0:dst_x0 + patch.shape[1]] = patch
    return crop


def _client():
    from openai import OpenAI

    key = api_key()
    if not key:
        raise RuntimeError("BASETEN_API_KEY is not set in os.environ")
    return OpenAI(api_key=key, base_url="https://inference.baseten.co/v1")


def _parse(client, messages, schema, max_tokens: int, *,
           model: str, extra_body: dict | None = None):
    kwargs = dict(
        model=model,
        messages=messages,
        response_format=schema,
        temperature=0,
        max_tokens=max_tokens,
    )
    if extra_body:
        kwargs["extra_body"] = extra_body
    last_error: Exception | None = None
    for delay in (0.0, 1.5, 3.0):
        if delay:
            time.sleep(delay)
        try:
            return client.beta.chat.completions.parse(**kwargs)
        except Exception as exc:
            last_error = exc
            if "429" not in str(exc):
                raise
    assert last_error is not None
    raise last_error


def classify_one_crop(bgr: np.ndarray) -> BallCall:
    """One ball, one cheap request. Used by the smoke test."""
    client = _client()
    response = _parse(client, [{
        "role": "user",
        "content": [
            {"type": "text", "text":
             "One overhead pool ball. kind=cue|eight|stripe|solid. "
             "number 1-15 if readable, else null. 8=eight, 1-7=solid, "
             "9-15=stripe, plain white=cue."},
            {"type": "image_url",
             "image_url": {"url": encode_crop(bgr), "detail": "low"}},
        ],
    }], BallCall, MAX_TOKENS_ONE,
                 model=SMOKE_MODEL,
                 extra_body={"reasoning_effort": "none"})
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("Baseten returned no structured ball call")
    parsed.kind = kind_from_number(parsed.kind, parsed.number)
    return parsed


def classify_found(rectified: np.ndarray,
                   found: list[tuple[float, float, dict]],
                   r: float
                   ) -> list[tuple[str, float, int | None]] | None:
    """Kinds for every candidate, one Flash request. None = caller keeps OpenCV."""
    if not found or not api_key():
        return None
    try:
        return _classify_batched(rectified, found, r)
    except Exception as exc:
        # A 429 is not fixed by issuing sixteen more requests.
        if "429" in str(exc):
            raise
        return _classify_singles(rectified, found, r)


def _enforce_unique(rows: list[tuple[str, float, int | None]],
                    found: list[tuple[float, float, dict]]
                    ) -> list[tuple[str, float, int | None]]:
    """A set has one cue and one 8. The model is allowed to be wrong twice.

    When it is, the OpenCV measurements already on `found` break the tie:
    darkest colourless body is the 8, whitest disc is the cue. Everyone else
    keeps a stripe/solid from their number, or from the model's leftover kind.
    """
    rows = list(rows)
    eights = [i for i, (kind, _, _) in enumerate(rows) if kind == "eight"]
    if len(eights) > 1:
        best = min(eights, key=lambda i: found[i][2]["body_darkness"])
        for i in eights:
            if i == best:
                continue
            number = rows[i][2] if rows[i][2] != 8 else None
            kind = kind_from_number("solid", number)
            rows[i] = (kind, 0.70, number)
    cues = [i for i, (kind, _, _) in enumerate(rows) if kind == "cue"]
    if len(cues) > 1:
        best = max(cues, key=lambda i: found[i][2]["white"])
        for i in cues:
            if i == best:
                continue
            number = rows[i][2]
            rows[i] = (kind_from_number("stripe", number), 0.70, number)
    return rows


def merge_with_opencv(
        vlm: list[tuple[str, float, int | None]],
        opencv: list[tuple[str, float]],
        found: list[tuple[float, float, dict]],
        ) -> list[tuple[str, float, int | None]]:
    """Trust a unique readable number; otherwise keep the rim test.

    The VLM is better at cue/8 when the number is visible, and worse at
    inventing a number on a dark crop. A number that appears twice is not
    a reading, so those balls fall back to OpenCV. Stripe/solid with no
    number does too — that is the case the rim test was written for.
    """
    vlm = _enforce_unique(vlm, found)
    counts = Counter(number for _, _, number in vlm if number is not None)
    merged = []
    for (kind, confidence, number), (cv_kind, cv_conf) in zip(vlm, opencv):
        if kind in ("cue", "eight"):
            merged.append((kind, confidence, number))
            continue
        if number is not None and counts[number] == 1:
            numbered = kind_from_number(kind, number)
            # A unique number that agrees with the rim test is a label, not a
            # vote. A unique number that *disagrees* is usually a misread 9
            # on a solid (seen on the fixture). Do not let that regress the
            # 12/14 OpenCV already has.
            if numbered == cv_kind:
                merged.append((numbered, confidence, number))
                continue
        merged.append((cv_kind, cv_conf, None))
    return merged


def _row(kind: str, number: int | None) -> tuple[str, float, int | None]:
    kind = kind_from_number(kind, number)
    confidence = 0.92 if number is not None else 0.80
    if kind in ("cue", "eight"):
        confidence = max(confidence, 0.88)
    return kind, confidence, number


def _classify_singles(rectified: np.ndarray,
                      found: list[tuple[float, float, dict]],
                      r: float
                      ) -> list[tuple[str, float, int | None]] | None:
    """One Flash call per ball. Slower, but Flash can actually finish a crop."""
    out = []
    for cx, cy, _ in found:
        crop = crop_disc(rectified, cx, cy, r)
        crop = cv2.resize(crop, (TABLE_CROP_PX, TABLE_CROP_PX),
                          interpolation=cv2.INTER_AREA)
        call = classify_one_crop(crop)
        out.append(_row(call.kind, call.number))
    return _enforce_unique(out, found)


def _classify_batched(rectified: np.ndarray,
                      found: list[tuple[float, float, dict]],
                      r: float
                      ) -> list[tuple[str, float, int | None]]:
    client = _client()
    out: list[tuple[str, float, int | None]] = []
    # Eight crops per call: under the cheaper models' image cap, and short
    # enough that low reasoning cannot eat the whole token budget.
    for start in range(0, len(found), BATCH):
        chunk = found[start:start + BATCH]
        content: list[dict] = [{
            "type": "text",
            "text": (
                f"{len(chunk)} overhead pool-ball crops, index {start}.."
                f"{start + len(chunk) - 1}. One record per crop, same index. "
                "kind=cue|eight|stripe|solid. number 1-15 or null. "
                "8=eight, 1-7=solid, 9-15=stripe, unmarked white=cue."
            ),
        }]
        for cx, cy, _ in chunk:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": encode_crop(crop_disc(rectified, cx, cy, r),
                                   size=TABLE_CROP_PX,
                                   quality=TABLE_JPEG_QUALITY),
                    "detail": "low",
                },
            })
        response = _parse(client, [{"role": "user", "content": content}],
                          TableCall, MAX_TOKENS_BATCH, model=TABLE_MODEL)
        parsed = response.choices[0].message.parsed
        if parsed is None or len(parsed.balls) < len(chunk):
            raise RuntimeError("Baseten batch returned an incomplete table call")
        by_index = {row.index: row for row in parsed.balls}
        for offset in range(len(chunk)):
            row = by_index.get(start + offset)
            if row is None:
                raise RuntimeError(f"Baseten batch missed index {start + offset}")
            out.append(_row(row.kind, row.number))
    return _enforce_unique(out, found)
