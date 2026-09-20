# Perception: measuring the table from an image

`PerceptionService.estimate()` is implemented by an OpenCV pipeline in
`pool/perception/vision/`, ported from a standalone project. It finds the
cloth, measures it, locates the balls on it and classifies each one, then
returns a `TableState` in table-length units.

Classical CV only — no detection model and no depth model.

## What it does, in order

1. **Find the cloth.** The playing surface is found by *saturation*, not
   brightness: on a real table the cloth is darker than the floor around it
   (measured: cloth at grey 19–32, floor at 55), so brightness alone picks
   the floor. Saturation separates them by 224–231 against under 45.
2. **Measure it.** The shape comes from the image — `vision/aspect.py`
   recovers a rectangle's true proportions from its perspective projection —
   and the scale comes from the caller's `TableGeometry`.
3. **Lock a grid.** `vision/calibration.py` runs the above over every frame
   in the batch and takes the median, so one bad frame cannot move the grid.
4. **Find the balls.** Against the rectified top-down view, where a ball is
   the same number of pixels across wherever it lies.
5. **Classify each one.** Cue, eight, stripe or solid.

## Units and axes

The pipeline works in whatever units its active `TableSpec` carries, origin
at the top-left playing corner. Its `width_mm` is the x extent, and
`estimate()` installs the caller's `length` there — so the pipeline's x is
the contract's x, both running along the long side. **No axis swap happens
anywhere**, and none should be added.

This agreement is load-bearing, not incidental. `ball_radius_mm` derives the
search radius from the spec's long side, so a transposed spec does not merely
relabel axes — it makes the detector look for a wrong-sized disc, and the ball
count silently degrades. `_check_spec` rejects that case before measuring.

## Known limitation: this expects an overhead view

The ported pipeline was built and measured against a **roughly overhead**
camera. [calibration.md](calibration.md) specifies an oblique side-mounted
camera with model-estimated depth, which this does not implement.

What that means concretely:

- The cloth homography maps the *plane*. A ball center sits a radius above
  that plane, so an oblique view introduces parallax error that grows with
  obliquity. Overhead, that error is small; from the side it is not.
- There is no depth estimation here at all. Steps 4 and 6 of the teammate-1
  milestone in [team-guide.md](team-guide.md) are **not** done.
- Multi-view fusion is not implemented. A batch with several captures
  calibrates the grid from all frames but detects balls in the last one;
  detections are not merged or deduplicated across views.
- `coverage` is always `COMPLETE` when a grid was found. Genuine coverage
  tracking, and returning `NeedsMoreViews` for unresolved occlusion, is not
  implemented — `NeedsMoreViews` currently means only "the cloth could not be
  measured from these frames."

So this is a working perception stage for an overhead capture, and a
starting point — not the finished oblique-view implementation.

## Classification, and what is trusted

Two paths answer "stripe or solid", and neither is allowed to overrule
arithmetic:

- **Rim measurement** (default): how much white sits in a thin annulus at the
  ball's rim. Held out over unseen layouts this scored **62.8%**.
- **A small CNN** (`vision/classify/net/`, used when torch is installed):
  **86.7%** on the same crops, ~6 ms for all sixteen.

The gap is not a better threshold waiting to be found. A stripe lying with
its coloured band square to the camera shows no white at all and measures
exactly like a solid. Only "does this look like a stripe" survives rotation.

What stays arithmetic in the detector, whatever the classifier says: one cue
and one eight per set, seven stripes and seven solids. A network that is
merely usually right must not overrule a fact about the game.

A hosted vision model (`vision/classify/baseten.py`) is kept for comparison
and is **not** on the default path: measured on this table it took 57–78 s
for a full table against 0.38 s for all of OpenCV, and scored 13/16 where
OpenCV scored 14/16.

## Measured behaviour

On one recorded overhead frame of a full rack spread on the cloth
(`tests/perception/test_perception.py`): **16 balls, 7 solids, 7 stripes,
1 cue, 1 eight**, every position inside the playing surface.

This is a regression guard on one image, not an accuracy target. Per the
team guide, a target belongs after the hardware is measured. Localization
error against measured truth at near/far table regions has **not** been
evaluated, because that needs the real mount.

## Running it

```sh
pip install -e ".[perception]"

PYTHONPATH=src python3.11 -m companion.app perceive \
  --captures data/local/captures/real_table.json \
  --geometry fixtures/table_geometry.json \
  --output artifacts/table_state.json
```

Recorded captures are device data and stay out of the repo, under the
ignored `data/local/` that [fixtures/README.md](../fixtures/README.md) calls
for. The real-table tests skip when no capture is present, rather than
failing on a machine that does not have the image.

The shipped `fixtures/captures/synthetic.json` exercises file loading only —
it is a drawn schematic, not a photograph of cloth, and is not a useful
accuracy benchmark.

## Dependencies

`opencv-contrib-python` and `numpy`, in the `perception` extra. Torch is
optional: without it the detector falls back to the rim measurement.
