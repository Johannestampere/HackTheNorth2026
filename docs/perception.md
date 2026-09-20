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

## Development tools

`tools/` holds the standalone utilities the pipeline was built with. They are
not part of the stage pipeline - `companion.app` remains the entry point -
but they are how the detector gets debugged and retrained:

| Tool | What it is for |
| --- | --- |
| `htn26_cli.py` | The original CLI: live preview, or a saved image, writing the annotated overlay, the rectified view and an x/y graph. The fastest way to see *why* a frame read the way it did. |
| `capture_crops.py` | Collect and hand-label ball crops from the camera, into `data/local/crops/`. |
| `eval_baseten.py` | Score classification on a saved frame against hand labels. |
| `smoke_baseten.py` | One-crop check that the hosted model is reachable. |

```sh
PYTHONPATH=src python3.11 tools/htn26_cli.py   --image data/local/fixtures/real_table_balls.png   --outdir artifacts/debug --surface table
```

The drawing they rely on lives in `vision/overlay.py`, beside the detector
whose output it renders.

## Scanning the real table

`tools/live_scan.py` runs the whole workflow against the camera:

```sh
PYTHONPATH=src python3.11 tools/live_scan.py
PYTHONPATH=src python3.11 tools/live_scan.py --show        # also open windows
PYTHONPATH=src python3.11 tools/live_scan.py --no-plan     # scan and plot only
PYTHONPATH=src python3.11 tools/live_scan.py --save-frame  # keep the frame
```

It opens a webcam, pools frames into a locked grid, detects and classifies
the balls, and writes two figures to `artifacts/live/`:

| File | What it answers |
| --- | --- |
| `1-table-scan.png` | Did the camera read the table correctly? The measured x/y grid with each ball drawn where it was found and labelled with its ID. |
| `2-shot-decision.png` | What should I do about it? The simulated paths, the ghost ball and the aim vector. |

Two figures rather than one because they fail differently. When a suggested
shot looks wrong, the first says whether perception or planning was at fault.

`--camera N` picks a device (default prefers an external one), `--frames N`
sets how many frames are pooled for the grid, and `--settle` gives the
exposure time to settle before capture - calibrating on a camera's first
frames costs accuracy.

### A partial rack

Balls leave a table as the game is played, and this expects that. Two things
behave differently from a full rack, and neither is silent:

- **The cue and the eight are chosen by comparison** against the balls that
  were found, and the stripe/solid pairing needs all fourteen object balls.
  Below twelve balls the detector says so itself; below sixteen the scan
  prints a one-line reminder to confirm both in the grid plot.
- **The planner needs a cue ball and an eight ball on the cloth**, and
  refuses any state with an unclassified ball, since it cannot judge whether
  potting an unknown ball is legal. `live_scan.py` checks these before
  loading the simulator, so a missing ball is named immediately rather than
  after a search.

## End to end: photo in, shot vector out

`tools/shot_demo.py` runs both stages on one capture and plots the result -
the table, the balls as perception classified them, the simulated paths, and
the aim vector anchored on the cue ball:

```sh
PYTHONPATH=src python3.11 tools/shot_demo.py
PYTHONPATH=src python3.11 tools/shot_demo.py --show        # also open a window
```

It writes `artifacts/shot.png`. It is a check, not a stage: it calls the same
`PerceptionService` and `PooltoolPlanner` the pipeline does, so a disagreement
between them shows up as a wrong-looking picture rather than as numbers to
verify by hand. The plot is drawn from the returned `ShotPlan` alone - if the
arrow does not start on the cue ball and run through the ghost ball, the
contract is being filled in wrong.

### Installing pooltool

`pip install -e ".[planning]"` fails on its own: pooltool 0.6.0 pins a
panda3d development build that is not on PyPI. Add panda3d's own index:

```sh
pip install -e ".[planning]" --extra-index-url https://archive.panda3d.org/simple
```

Note that this resolves numpy down (2.4 to 2.3 here); the vision pipeline is
unaffected, but it is why the planning extra is not installed by default.

## Tests

`tests/perception/` holds 99 tests: the 90 ported with the pipeline plus 9
covering the contract boundary.

The synthetic ones need no camera and no recorded data - a sheet is rendered,
warped through a known perspective and degraded with a lighting ramp, blur
and noise, then measured. Those run anywhere.

The real-table ones read recorded frames from the ignored
`data/local/fixtures/`, and **skip** when it is absent rather than failing on
a machine that does not have the images. A skipped test is honest about what
was not checked.

## The original project's notes

The standalone project's README is kept verbatim at
[perception-pipeline-reference.md](perception-pipeline-reference.md). It
carries the detail this guide summarises: why the cloth is found by
saturation, what each accuracy figure was measured against, the failure
modes, and the troubleshooting notes.

## Dependencies

`opencv-contrib-python` and `numpy`, in the `perception` extra. Torch is
optional: without it the detector falls back to the rim measurement.
