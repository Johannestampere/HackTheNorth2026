> **Ported reference.** This is the original standalone project's README,
> kept for the detail it carries: the measurement rationale, the accuracy
> figures and how each part was validated. Paths and commands in it refer to
> the old layout - see [perception.md](perception.md) for where each module
> lives now and how to run it here.

# Pool table grid calibration

Point a webcam at a **pool table** and get an x,y grid of the playing
surface, its corners and its six pockets — and the balls on it, each placed
on that grid and sorted into cue, 8, stripe or solid.

Classical CV only (OpenCV + numpy), no ML. Physics and shot prediction are
out of scope.

A **paper mode** is also included (`--surface paper`), which finds a plain
sheet on a dark desk. It was the original stand-in for the table and is kept
because it still works and the synthetic tests exercise it.

## Install

```
pip install "opencv-contrib-python>=4.7" numpy pytest
```

Python 3.10+.

## Setup — this is the part that matters

**All four cushions must be in view, and the cloth must read as colour.**
The playing surface is found by its colour, not its brightness: on a real
table the cloth is *darker* than the floor around it (measured: cloth at grey
19–32, floor at 55), so brightness alone picks the floor. What separates the
cloth is saturation — 224–231 against under 45 for the rails, floor and
shadow, a gap no lighting change is going to close.

- Get the **whole playing surface** in frame, all four cushions included.
- Light it well enough for the colour to show. Colour survives dim light
  better than contrast does, but not darkness.
- **View the table from one side, not square-on.** This is the one that is
  easy to get wrong — see [Why tilt matters](#why-it-pools-frames-and-why-tilt-matters).
- Balls, a rack, chalk and a triangle on the cloth are all fine *for
  measuring the surface*. They are holes in the cloth region and get filled
  in; they cannot shrink it. For finding the **balls** themselves, spread
  them out — a tight rack is a known limitation, see
  [Accuracy → Balls](#balls).

The tool says which of these is wrong when detection fails, rather than just
"not found".

### What it measures against

Table coordinates start at the **cloth inside the cushions**, not the cabinet:
origin at the top-left playing corner, x across the width, y down the length.
That is where play actually happens, so that is what the grid describes.

Only the *shape* comes out of one view — a small table close to the camera and
a large one further away project identically — so the absolute scale comes
from one real dimension you measure with a tape:

```
python main.py --reference-mm 1000            # the width, in mm
python main.py --reference-mm 1780 --reference-side height
```

## Run

```
python main.py
```

This opens **two windows**:

- **live** — the camera feed with the grid fitted to the surface's corners,
  so you can see it is tracking the right thing.
- **x,y grid** — a clean graph of what was measured: the surface as an exact
  rectangle at its measured size, axes, corners and pockets, no photo behind
  it. Once you press `B` the balls appear on it too.

### The run has two phases

**1. Lock (the first 5 seconds, automatic).** Aim the webcam down at the
table and hold still. Every frame of the countdown is kept, and the
surface's corners, size and pockets are measured from the whole burst at
once. The banner counts down: *LOCKING IN 3.4s - hold still*.

**2. Detect (from then on).** Press **`B`**. The balls are located and
classified against the geometry locked in phase 1, and written out.

| Key | Does |
|---|---|
| **`B`** | find and classify the balls; save `balls_graph.png`, `balls_overlay.png`, `balls.json` |
| **`D`** | detect dark scraps instead (paper mode; see *Marks*) |
| **`R`** | re-lock the surface geometry (if you moved the table or the camera) |
| **`q`** (or Esc) | quit |

Press **`B`** as often as you like — play a shot, press it again. The grid
underneath does not move, because the surface is only measured during a
lock. That separation is the point: the balls are measured *against* a fixed
frame of reference rather than against a grid that re-fits itself every time
something on the table changes.

The banner tells you when you are ready:

| Banner | Meaning |
|---|---|
| orange — *LOCKING IN Ns - hold still* | phase 1, measuring the surface |
| green — *LOCKED 1.000 x 0.515 - press B* | phase 2, ready for the balls |
| red — *table not found* | with the specific reason underneath |

Other entry points:

```
python main.py --image photo.jpg    # offline, from a saved photo
python main.py --no-preview         # skip the window: capture and export
python main.py --list-cameras       # show detected cameras and exit
```

| Flag | Meaning |
|---|---|
| `--surface paper\|table` | what to look for (default: `table`) |
| `--camera N` | force a camera index (default: auto-select, see below) |
| `--width` / `--height` | capture resolution (default 1280x720) |
| `--frames N` | frames to median over when locking (default 30) |
| `--outdir DIR` | where to write outputs (default `out/`) |
| `--require-all-holes` | fail instead of reporting a pocket as missing |
| `--reference-mm` | one real surface dimension, measured with a ruler. Without it the tool reports normalised units (long side = 1.0) |
| `--height-mm` | the other dimension, when the view is too straight-on to recover the shape |
| `--reference-side` | which side `--reference-mm` is: `long`, `width` (across x) or `height` (down y) |
| `--ball-radius-frac` | a ball's radius as a fraction of the long side (default 0.0150) |

`--ball-radius-frac` is the one number that sets both the ball search and the
plotted disc. The default is measured on the table this was developed
against; regulation tables use a 57.15 mm ball, so the ratio falls as the
table grows — **0.0144** on a 7 ft playfield, **0.0128** on an 8 ft,
**0.0113** on a 9 ft. A smaller table with the same balls needs a *larger*
number, which is the case to watch for: the detector is genuinely sensitive to
this and checks its own answer against the fact that a pool set has 16 balls.
See [Accuracy → Balls](#the-radius-prior-is-the-one-number-you-may-have-to-set).

### Camera selection

The camera is chosen automatically, preferring an external USB webcam over a
laptop's built-in one. `tools/htn26_cli.py` prints its choice:

```
note: using camera 1: Logitech Webcam C925e
```

A camera switched off in hardware still opens and still returns frames — flat
grey ones. That is detected and reported explicitly rather than showing a blank
preview. `--list-cameras` shows what was found; `--camera N` overrides.

## Geometry — the sheet is measured, not assumed

No paper size is hardcoded, and the tool is meant to measure **any flat
rectangle** it is shown, standard or not. The size comes from two things:

1. **Shape, from the perspective.** A rectangle's two pairs of opposite sides
   meet at two vanishing points, and those constrain both the camera's focal
   length and the rectangle's true aspect ratio — the classical single-view
   metrology result (Zhang; Liebowitz & Zisserman). This matters because the
   *apparent* ratio in the image is simply wrong: a tilted sheet is
   foreshortened, and on real photos the apparent ratio is off by 25% or more.

2. **Scale, from one ruler measurement.** A small sheet close to the camera
   and a large one further away project identically, so no single view can
   give absolute size. That one number has to come from you:

```
python main.py --reference-mm 140                  # 140 mm across (default)
python main.py --reference-mm 216 --reference-side height
```

The measured size is printed and drives everything downstream:

```
measured sheet: 140.0 x 216.3 mm   (width pinned to 140.0 mm)
  focal length 1593 px from 30 views (confidence 100%)
```

Origin at the sheet's top-left corner, **x** across the width, **y** down the
height, units mm. Six pockets: the four corners plus the midpoints of the two
long sides — and which sides are "long" follows the measured shape, so the
middle pockets land correctly whichever way the sheet lies.

### Why it pools frames, and why tilt matters

The focal length is a fixed property of the camera, so a 30-frame burst is 30
measurements of one number — but they are not equally informative. As a view
approaches fronto-parallel the vanishing points run towards infinity and the
estimate becomes hopelessly ill-conditioned: measured on a real near-
straight-on frame, one vanishing point sat 21,000 px from the image centre and
a **1-pixel** corner error moved the focal estimate by ~40 px, which in turn
moved the measured height by tens of millimetres.

So each view is scored by its conditioning and the focal length is a
*weighted median* across the burst, which ignores the wild estimates instead
of averaging them in. Against ground truth (140 x 216 mm, 30 frames, 0.8 px
corner noise):

| view | focal (true 1600) | measured height (true 216) |
|---|---|---|
| moderate tilt | 1593 px | 215.3 mm |
| mixed handheld | 1600 px | 215.6 mm |
| near straight-on | 1460 px, flagged low confidence | 215.8 mm |

**Rotate the paper, don't just tilt the camera.** This is the part that is
easy to get wrong, because the obvious fix is the wrong one.

The shape is recovered from *two* vanishing points, one per pair of opposite
edges. Tilting the camera down at the desk makes the top and bottom edges
converge — that is one vanishing point, and it is usually fine. But if the
paper sits square-on in the frame, the left and right edges stay parallel in
the image, their vanishing point is at infinity, and there is nothing to
solve with. **Tilting further in the same direction never fixes this.**
Measured on a real frame: the left/right vanishing point sat a healthy
1,229 px from the image centre while the top/bottom one sat at 41,506 px, and
a 1-pixel corner error swung the focal estimate from 294 px to 454 px.

The fix: **move the camera off to one side** so the table sits skewed in
the frame rather than square-on. You want *both* pairs of edges visibly
converging. (With paper you can equivalently just turn the sheet on the
desk.) The real table in the test fixture clears this easily — the camera
sees it from one side, and 30 consecutive frames all solved.

| second-axis rotation | conditioning | focal recovered (true 1400) |
|---|---|---|
| 0° (paper square-on) | 0.000 | fails — unsolvable |
| ~17° | 0.396 | 1400 |
| ~30° | 0.668 | 1400 |

When the pose cannot determine the shape, the tool says so rather than
guessing:

```
note: this camera pose cannot determine the sheet's shape. One pair of the
      sheet's edges is still parallel in the image ...
      FIX: turn the paper about 30 degrees on the desk
```

and the live preview shows a running `shape confidence` so you can watch it
go green as you rotate.

## Outputs

Written to `out/`:

- **`grid.json`** — the locked grid (below).
- **`overlay.png`** — the camera frame with the paper outline, a 25 mm x,y grid
  projected onto it, mm axis labels, and the pockets marked.
- **`rectified.png`** — the top-down warped view with the same grid.
- **`xy_graph.png`** — the clean x,y graph: the sheet as an exact rectangle
  with mm axes, corners and pockets, no photo.

Written each time you press **`B`**:

- **`balls_graph.png`** — the x,y graph with every ball drawn at its measured
  position as a disc of one fixed radius: white for the cue, black for the 8,
  the ball's own measured colour for a solid, and that colour with a white
  band across it for a stripe. A ball whose classification was a close call
  gets an amber ring. **This is the picture you are after.**
- **`balls_overlay.png`** — the camera frame with each ball ringed and
  labelled with its kind, to confirm it found the right things.
- **`balls.json`** — the balls as `{id, kind, xy_mm, radius_mm, confidence,
  colour_bgr, px}`, plus counts by kind and the radius used, and a `warning`
  field that is `null` unless the reading is impossible — more balls than a set
  has — in which case it says so and why. A saved reading has to carry the
  reason not to trust it, or the JSON outlives the warning.

`kind` is one of `cue`, `eight`, `stripe`, `solid`. `confidence` runs 0.5 to
1.0 and reports how far the deciding measurement sat from its threshold; see
*Accuracy → Balls* for what it can and cannot tell you.

Drawing the balls at their true relative radius is what makes the graph a
plan of the table rather than a scatter plot — whether two balls are
touching, or whether one blocks the line to another, is only readable if the
discs are the right size.

Written each time you press **`D`** (paper mode):

- **`marks_graph.png`** — the same x,y graph with the black scraps drawn as
  discs of a fixed 5 mm radius at their measured positions. This is the
  picture you are after.
- **`marks_overlay.png`** — the camera frame with the scraps circled, to
  confirm it found the right things.
- **`marks.json`** — the scraps as `{id, xy_mm, radius_mm, area_mm2, px}`.

The radius is a fixed 5 mm by convention, not a measurement: the scraps are
torn and irregular, so a common radius keeps the plot readable. The actual
size that was seen is reported separately as `area_mm2`.

### Reading `grid.json`

```jsonc
{
  "units": "mm",
  "origin": "paper top-left corner",
  "axes": "x across the 210 mm side, y down the 297 mm side",
  "frame": { "width_mm": 210.0, "height_mm": 297.0 },

  // The 4 sheet corners: table mm (exact by definition) and where they
  // landed in the camera image.
  "corners": [ { "id": "TL", "xy_mm": [0, 0], "px": [660.0, 347.0] }, ... ],

  // The 6 pockets, in TL TM TR BL BM BR order.
  "holes": [
    {
      "id": "TL",
      "found": true,           // false => projects outside the frame
      "xy_mm": [0.0, 0.0],     // table position
      "px": [660.0, 347.0],    // where it sits in the camera image
      "nominal_xy_mm": [0, 0],
      "error_mm": 0.0          // homography round-trip error
    }, ...
  ],

  "homography": [[...], [...], [...]],  // 3x3, image px -> table mm
  "reprojection_error_mm": 0.0,
  "frames_used": 30,
  "frames_attempted": 30,
  "timestamp": "2026-09-19T...+00:00",
  "notes": []
}
```

To use the grid, apply `homography` to any image point to get table mm:

```python
import cv2, json, numpy as np
H = np.array(json.load(open("out/grid.json"))["homography"])
xy_mm = cv2.perspectiveTransform(np.array([[[x_px, y_px]]], np.float64), H)
```

A note on `error_mm` and `reprojection_error_mm`: with four corner
correspondences the homography fit is exact, so both read ~0. They are **not**
a measure of how accurately the sheet was located — that is set by how well the
corners were found. The honest check is the sheet's measured size (see below).

## Accuracy

### Real pool table

The camera looks almost straight down at the table, which finds the surface
easily and measures its *shape* only from the outline. Measured on the three
real frames in `tests/`:

| frame | conditioning | confidence | focal length | aspect used |
|---|---|---|---|---|
| `real_table.png` | 0.044 | 0.17 | 2295 px, **rejected** | 0.521 |
| `real_table_lit.png` | 0.038 | 0.15 | 3044 px, **rejected** | 0.536 |
| `real_table_balls.png` | 0.016 | 0.00 | not solvable | 0.515 |

Conditioning near zero means the two vanishing points are effectively at
infinity: both pairs of cushions stay parallel in the image, so perspective
carries no information about the table's proportions. The focal lengths above
are what the geometry returns anyway, and they disagree with each other by
33% on two frames of the same table — which is why they sit below the 0.35
gate and are not used. The aspect ratio comes from the outline instead, which
is exactly right for a top-down view.

Do not read the three aspect figures as a repeatability measurement: 0.515 to
0.536 is a 4% spread caused by the corner error below, not by noise.

**This is a worse-conditioned pose than it looks.** Tilting the camera down
from directly overhead does not help on its own — that only tilts about one
axis, and one pair of cushions stays parallel. Both pairs have to visibly
converge, which means moving the camera to a corner rather than a side. The
preview says so when it sees it: *MOVE THE CAMERA TO ONE SIDE — both edge
pairs must converge*.

**Known limitation: the bottom-right corner.** On the test frame that corner
lands roughly 20 px outside the cushion, out on the rail. The cause is
physical, not a threshold: the red cloth wraps over the cushion and continues
down the outer face of the rail, so it is *the same fabric at the same hue and
saturation* on both sides of the boundary — sampled at H 178 / S 220 / V 67
outside versus H 178 / S 229 / V 64 inside. No colour gate can separate them.
The dark rail between the two is only 6–10 px wide, while the gaps the balls
cut are 17–136 px, so no morphological bridge can span the balls without also
spanning the rail.

The other three corners and all six pocket markers land correctly, and the
error is confined to one corner, so the grid is usable — but it is a real
error and it is not tuned away. Fixing it properly needs a cue other than
colour (the cushion crease, or depth).

### Paper mode

On synthetic scenes with known ground truth (tilt, uneven lighting, blur,
noise, corner tape, desk clutter): corners land within **3-4 px**, and the
sheet measures within **1.6-3.5 mm** of its true 210 x 297 mm across a range of
tilts. The residual is the contact shadow at the paper's edge biasing the
boundary slightly.

Accuracy degrades when a corner is occluded — tape across a corner is the usual
cause, and it shows up as a measured size noticeably off 210 x 297.

### Balls

Measured against `tests/fixtures/real_table_balls.png` — a full set of 16 spread out
on the cloth, hand-labelled by reading the numbers off 6x crops.

| | |
|---|---|
| balls found | **16 / 16**, no false positives |
| position error | **0.0003** of the long side (0.02 of a ball radius) |
| same scene at 720p | **16 / 16**, error 0.002 of the long side |
| cue ball | correct |
| 8 ball | correct |
| stripe vs solid | **12 / 14** |

**Stripe vs solid is the hard part, and 12/14 is close to the ceiling for one
overhead view.** A pool set pairs every solid with a stripe of the same
colour, so colour cannot identify them; the only signal is how the white is
arranged. Reading the white over the whole ball gives 71%, because a ball's
number sits in a white patch large enough to look like a stripe's white pole
when it faces the camera — two *solids* measured 72% and 73% white. Reading
it only in a thin annulus at the ball's edge gives 86%, because a stripe's
white poles run out to the silhouette while a number patch stops short.

The two it gets wrong are both stripes, and one of them is not recoverable: a
stripe resting with its coloured band square to the camera shows no white at
all and measures exactly like a solid. Both are reported with a confidence,
and stripe/solid confidence is capped below 1 for that reason.

**Known limitation: a tight rack.** The detector scores a ball against the
cloth around it, so a ball with six neighbours touching it has nothing to
stand out against. On `real_table.png`, where the balls are racked, responses
fall from 86 down into the noise and most of the rack is missed. Balls in
open play — which is when you want the positions — are found reliably.

**Known limitation: pocket jaws.** Nothing is reported within 2.5 ball radii
of a pocket, because a pocket mouth is a ball-sized dark hole that would
otherwise be read as an 8 ball. A ball hanging on the lip is missed.

#### The radius prior is the one number you may have to set

`--ball-radius-frac` is not a tuning knob — it is the detector's only
assumption about the equipment, and everything else is a fixed multiple of it.
Get it wrong and every stage inherits the error at once. Measured on the test
frame by halving it:

| radius used | balls reported | found | 8 ball | stripe vs solid |
|---|---|---|---|---|
| 0.0150 (correct) | 16 | 16/16 | 1 | 12/14 |
| 0.0130 | 17 | 11/16 | 0 | 7/11 |
| 0.0075 (half) | 17 | 1/16 | 0 | 0/1 |

Too small a radius fails in a specific and initially convincing way. The search
disc fits *inside* a ball, so a single ball returns two detections — one
centred on its white number patch, one on its coloured body — and each centre
is pulled off the ball's middle. The annulus that tells a stripe from a solid
then samples the ball's interior instead of its rim, which is the entire basis
of that call. The cushions' own shadows become candidates. And all of it is
reported at high confidence, because each measurement really is internally
consistent; nothing about one candidate reveals the problem.

So the tool checks the one thing that cannot be argued with: **a pool set has
16 balls.** More than 16 is not a close call about one ball, it is proof that
the scale is wrong, and the tool says so — in the live window, on the console
and in `balls.json`'s `warning` field — instead of reporting the count as a
result:

```
! 17 balls reported, but a pool set has 16, so the ball radius in use
! (0.0075 of the playing surface's long side) is too small for this table ...
```

Too *large* a radius fails the other way, by missing balls rather than
inventing them, and the count cannot catch that. See the troubleshooting table.

The radius is not measured from the frame, and that is a real limitation rather
than an oversight. Four estimators were tried and none was trustworthy enough
to put in front of the matched filter: scale-selection on the response peaks
(a half-radius disc centred on a number patch outscores a full-radius disc
centred on the ball, 107 to 99), blob area and circular Hough (both latch onto
clusters of touching balls), and a distance transform of the non-cloth mask
(the best of them, but still 0.81x to 1.26x the truth depending on where the
mask is thresholded). Refining the radius from the detections' own silhouettes
does converge, but only from within about 30% — and the case that matters here
was off by a factor of two.

### Marks on the surface

Mark positions are accurate to **0.4 – 1.4 mm** when measured through a known
homography. Any error in the surface measurement scales the mark coordinates
with it, so the marks are never better than the grid they sit on.

## Repository layout

```
main.py            CLI entry point: preview, calibration, export
geometry/          the plane being measured in
  table_spec.py      dimensions the pipeline measures against
  homography.py      image <-> table-plane map
  aspect.py          true proportions from a perspective view
  calibrate.py       pools a burst of frames into one median grid
detect/            finding things on the measured surface
  surface.py         picks the cloth or paper detector for the run
  cloth.py           the playing surface of a real table
  reference.py       the paper stand-in
  holes.py           pocket placement and the rectified view
  balls.py           the balls, and their kinds
  marks.py           dark marks lying on the sheet
render/overlay.py  drawing the exported images and the preview
hw/                the machine: camera.py, envutil.py
classify/          a second opinion on each ball's kind
  net/               the local CNN (kinds.py, train.py, ball_kinds.pt)
  baseten.py         the hosted model, kept for comparison
scripts/           run by hand, not imported by the pipeline
  capture_crops.py   collect and label crops for training
  eval_baseten.py    score kinds on a saved frame
  smoke_baseten.py   one-crop API check
tests/             suite plus fixtures/ (the real camera frames)
```

Imports are package-qualified (`from detect.balls import ...`) and the
project runs in place — `pyproject.toml` carries pytest config only, there is
nothing to install.

## How it works

| Step | Module | What it does |
|---|---|---|
| 1 | `vision/camera.py` | Selects the webcam (preferring external USB), fixes resolution, disables autofocus/auto-exposure (best effort), rejects a hardware-disabled camera. |
| 2 | `vision/reference.py` | Finds the sheet: pooled thresholdings → candidate quads → scored on shape, contrast and edge support → corners re-fitted from the image gradient. |
| 3 | `vision/homography.py` | Fits the image→table homography, returns the inverse and the error in mm. |
| 4 | `vision/holes.py` | Places the 6 pockets from the geometry and projects them into the image. |
| 5 | `vision/calibration.py` | Runs 2-4 over ~30 frames, takes the median, locks the grid. |
| 6 | `vision/detector.py` | Finds the balls against the locked grid and sorts them into cue / 8 / stripe / solid. |
| 7 | `tools/htn26_cli.py` | Preview, CLI and export. |

### Finding the balls

Three ideas carry `vision/detector.py`:

- **The radius is known — *given* the table.** A ball's size is fixed relative
  to the playing surface, and the surface has already been measured, so the
  detector never has to ask how big a ball is. The rectified view is scaled to
  put a ball at exactly 24 px radius whatever the table's size or units, which
  means every kernel in the module is a fixed pixel count that was tuned once.
  The cost of that leverage is that the ratio is an assumption about the
  equipment, not a measurement, and a wrong one breaks every stage at once —
  see [Accuracy → Balls](#the-radius-prior-is-the-one-number-you-may-have-to-set).
- **The cloth is locally uniform and a ball is not.** No global colour
  threshold separates them — the cloth vignettes to near-black in the corners
  and the balls run from white to black. A median over a window four
  ball-widths across *is* the cloth's colour at that spot, whatever the
  lighting is doing there, and a ball is whatever departs from it.
- **Centre minus surround, not a silhouette.** Thresholding the difference
  fragments every ball, because its shaded side matches the cloth. Scoring a
  disc of the right radius against the ring around it does not, and the same
  subtraction rejects the cushion edge — a ridge scores as high in its
  surround as in its centre, while a ball sits on bare cloth.

One rejection test is worth naming: a candidate that is colourless while
being neither pale enough for the cue nor dark enough for the 8 is thrown
away, because a pool set contains no mid-grey ball. That is what removes the
bright plastic liner across a pocket mouth, which sits inside the cloth
outline and scored 36 where the real balls scored 30 to 107 — neither its
position nor its contrast rules it out.

Four details that took real photos to get right:

- **Scoring on edge support, not area.** A quad that merely *encloses* the
  sheet has more area than the true outline, so scoring on area systematically
  prefers the wrong answer. A correct edge lies on a real intensity step.
- **Rejecting quads that lean on the image border.** A bright desk cropped by
  the frame edge looks convincingly sheet-shaped, but the frame edge is not an
  object boundary.
- **Normalising the gradient by a percentile, not the maximum.** A single
  strong edge elsewhere in the scene (a dark chair) otherwise scales the
  paper's own edges below any fixed threshold, silently disabling both the
  support test and corner refinement.
- **Bilateral filtering before the gradient.** Desk speckle and paper grain
  produce gradients comparable to a real paper edge; a plain Gaussian leaves
  the boundary indistinguishable from the texture around it.

## Swapping in another surface

Everything downstream consumes exactly one function:

```python
find_quad(image) -> np.ndarray | None      # four corners, TL TR BR BL
```

`vision/surface.py` dispatches it to the cloth detector or the paper one. Adding a
third means writing that function and registering it; the homography fit,
pocket placement, median calibration, ball detection, export and CLI need no
changes, because everything after the quad works in table coordinates.

The ball detector is the one part that assumes a pool table specifically —
it needs a locally uniform surface to model as background, and it bounds its
search with the cloth mask when the table detector is active.

Lens distortion is not calibrated; the mild wide-angle distortion of a laptop
webcam is absorbed into the homography fit. `fit_homography` already accepts
`camera_matrix` and `dist_coeffs` and undistorts before fitting when given
them, so adding a calibration later is a one-line change at the call site.

## Tests

```
python -m pytest -v
```

`pyproject.toml` points pytest at `tests/` and puts the repo root on the
path, so the bare command works from anywhere in the project.

90 tests, in three groups, about 110 seconds.

**Synthetic (no camera needed).** A sheet is rendered on a darker surface,
warped through a known perspective, then degraded with an uneven lighting
ramp, blur and noise. These check corner accuracy, measured size, corner
ordering, the homography round trip, the median burst, mark detection, and
both failure modes (no sheet; a sheet with no contrast against its surface).

**Real table.** Run against actual camera frames (`tests/fixtures/real_table.png`,
`real_table_lit.png`). The cloth detector depends on real colour and on how
a real table is built — cloth wrapping over the cushion onto the rail face —
and a synthetic scene would not reproduce either honestly. They check that
the corners land on the playing surface, that opposite sides stay comparable
(which is what catches a corner leaking onto the rail), that the rack does
not shrink the measured surface, that the pose can actually determine the
shape, and that a paper scene is *declined* rather than guessed at.

They also pin the focal-length gate: that repeating one view cannot raise
confidence (a stationary camera's frames all agree, which once let agreement
override conditioning and inflated a 0.17 to 1.00), and that a weak pose
falls back to the outline ratio.

**Balls.** Run against `tests/fixtures/real_table_balls.png` and its 720p twin — a
full set of 16 spread on the cloth, hand-labelled by reading the numbers off
6x crops. A synthetic scene would be dishonest here: what makes stripe
versus solid hard is the size of the number patch on real balls and the
gloss highlight of a real ceiling light, and a renderer would only reproduce
whatever the author already believed. They pin the counts, the positions,
the cue and the 8, and that stripe/solid stays at 12 of 14 — a recorded
figure rather than a target, since one of the two failures is not
recoverable from a single overhead view.

Four of them pin things that have no threshold in them, and are the more
useful for it: that no two centres come closer than two balls can (2r, which
is geometry, not tuning); that every centre sits a full radius inside the
cloth, since that is where a ball resting on a cushion has its centre; that
at most one 8 ball is ever reported, across a sweep of radii; and that too
small a radius prior is *reported* rather than returned as a reading. A fifth
builds a dark ball by hand — the only synthetic ball in the suite, and
deliberately so — to check that how dark a body measures does not depend on
whether a gloss highlight happens to be sitting on it.

## Troubleshooting

**Table mode**

| Symptom | Fix |
|---|---|
| *almost no cloth-coloured pixels found* | The cloth is not reading as colour. Light the table better, or check you are not in `--surface paper` by mistake. |
| *the largest cloth region covers only N% of the frame* | Move the camera closer, or fit more of the table in view. |
| *cloth fills almost the whole frame* | Move back so all four cushions are visible. |
| *its outline is not table-shaped* | A cushion is out of frame, or something large is covering the cloth. |
| A corner sits out on the rail | Known limitation — see [Accuracy](#accuracy). The cloth wraps over the cushion onto the rail face, and where the black rail between them is thin, the two merge. |

**Balls** (`B`)

| Symptom | Fix |
|---|---|
| Most of a rack is missed | Known limitation — see [Accuracy → Balls](#balls). A ball ringed by touching neighbours has no cloth to stand out against. Break, or spread them out. |
| **More balls reported than a set has**, often two rings on one ball | `--ball-radius-frac` is too small for this table, and the tool says so. Nothing in that reading is usable — not the positions, not stripe/solid, not the 8. Measure one ball's radius and the long side of the cloth, and pass the ratio. Proportionally large balls on a small table are the usual cause: the default suits the table this was developed against, and a smaller table with the same balls needs a bigger number. |
| Balls missed all over a full-size table | The radius prior is wrong the other way — too large. Pass `--ball-radius-frac 0.0113` (9 ft), `0.0128` (8 ft) or `0.0144` (7 ft). The count cannot detect this case, so it is not warned about. |
| A ball near a pocket is not reported | By design: nothing within 2.5 ball radii of a pocket, so a pocket mouth is not read as an 8 ball. |
| Stripes reported as solids | Expected some of the time — a stripe with its coloured band facing the camera shows no white at all. Check `confidence` in `balls.json`; a low value means the call was close. |
| Every ball reads as a solid | The white test is relative to the cloth's lightness. A very pale cloth narrows the gap; light the table more evenly. |

**Paper mode** (`--surface paper`)

| Symptom | Fix |
|---|---|
| Measured rectangle looks inset from the paper | Tape across the corners. The detector measures to the visible paper, so corner tape shrinks the rectangle. Tape the middle of each edge. |
| *the best paper-to-surface contrast is only N grey levels* | The surface under the paper is as bright as the paper. Move the whole sheet onto something clearly darker. |
| *the largest four-sided shape covers only N% of the frame* | Move the camera closer, or lower it towards the desk. |
| *none is sheet-shaped* | Part of the sheet is out of frame, occluded, or not lying flat. |
| Measured size is noticeably off 210 x 297 | A corner is hidden — usually tape across it. Tape the middle of the edges instead. |
| Preview is blank / grey | Camera off in hardware: privacy shutter, camera key (often F8/F9), or Windows Settings > Privacy & security > Camera. |
| Wrong camera opens | `--list-cameras`, then `--camera N`. |
