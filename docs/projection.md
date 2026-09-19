# Planner to projector integration

Branch: `feat/pool-projection-integration`.

Stage 2 now feeds a working stage 3 renderer: `ShotPlan` + `ProjectionTarget` → RGB `ProjectionFrame`. The explicit HTTP adapter forwards that exact frame to the Raspberry Pi display. No camera/LLM/hardware is required for a local preview.

## Local preview

From the repository root, using the planning environment:

```sh
PYTHONPATH=src .venv/bin/python -m companion.app project \
  --state fixtures/table_states/direct_shot.json \
  --game fixtures/game_contexts/solids.json \
  --target fixtures/projection_targets/synthetic.json \
  --output artifacts/projector-preview.png
```

This computes a real shot, validates it against the observation, renders at the calibrated resolution and saves the exact RGB pixels as a PNG. Open that PNG to inspect it. The included calibration is synthetic; it proves software integration, not real projector alignment.

To render an existing plan without Pooltool, use standard-library Python:

```sh
PYTHONPATH=src python3.11 -m companion.app render \
  --plan fixtures/shot_plans/direct_shot.json \
  --target fixtures/projection_targets/synthetic.json \
  --output artifacts/projector-preview.png
```

Both commands also accept `.ppm`. White is cue alignment, cyan is cue travel before contact/ghost ball, blue is cue travel after contact, yellow is object-ball travel, and green marks the called pocket. Black is the background. Geometry is transformed before rasterization; strokes are three output pixels. Circles are approximated by 64 transformed segments. Paths clip to the playing rectangle and output bounds. Aim-only plans still draw a cue line.

## Four corners and calibration

Four corners belong to the projection calibration, not to the shot decision. `ProjectionTarget.corners_px` provides them in **projector pixels**, in this correspondence order:

1. Agreed table top-left `(0,0)`.
2. Table top-right `(1,0)`.
3. Table bottom-right `(1,width)`.
4. Table bottom-left `(0,width)`.

The property derives them from the existing `table_to_pixel` homography, so there is no duplicated/stale corner field and no schema change. These are physical table identities: do not independently sort corners by their apparent position in the projector image. Camera pixel corners cannot be used here without camera-to-projector calibration.

To construct calibration, save a JSON array of those corresponding projector-pixel positions, for example:

```json
[[80, 60], [720, 90], [670, 350], [120, 320]]
```

Then run:

```sh
PYTHONPATH=src python3.11 -m companion.app calibrate \
  --geometry fixtures/table_geometry.json --corners calibration/local/corners.json \
  --width-px 800 --height-px 400 --calibration-id table-setup-001 \
  --pose-id fixed-pose-001 --output calibration/local/target.json
```

The command solves the eight homography parameters from the four point correspondences. Duplicate, collinear, concave and crossed corner arrangements are rejected. Rendering also rejects a mapping with a projective horizon crossing the table. Use real output dimensions and measured correspondences for a physical projector; the example coordinates are illustrative only. Lens distortion is not modeled.

## Send to the Pi

Run the display on the Pi at its actual HDMI resolution. From a remote planner, explicitly listen on the Pi network interface:

```sh
python3 raspi/vec2projector.py --http 8080 --http-host 0.0.0.0
```

Use a dedicated display process without concurrent clock/vector/demo senders, grid, or HUD overlays. The receiver can combine image/vector layers; the raw frame endpoint does not erase another sender's persistent vectors. Restart a previously used demo display before calibrated output.

Add `--display-url http://PI_ADDRESS:8080` to either `project` or `render`. The preview is saved first, then the same RGB bytes are posted to `/raw?w=...&h=...`. Without that explicit option no network or hardware operation occurs. HTTP failures propagate rather than printing success. `HttpProjector.blank()` sends a black frame of the configured size.

The current Pi receiver accepts images at different resolutions and fits them to the screen. **The calibration resolution must match the actual HDMI output** to preserve alignment. The client enforces its configured resolution, but cannot verify the HDMI mode or robot pose remotely. It also cannot confirm physical display completion: HTTP 200 means the frame was accepted. Use the matching pose and calibration; automatic rotation/pose verification and startup calibration remain future integration work.

## Files

- `pool/projection/calibration.py`: homography fitting and coordinate mapping.
- `pool/projection/service.py`: clipped RGB guidance rendering.
- `pool/projection/preview.py`: lossless standard-library PNG export.
- `hardware/http_projector.py`: explicit Pi transport and blanking.
- `app.py`: `calibrate`, `render`, and combined `project` commands.
- `tests/projection/test_rendering.py`: calibration, clipping, rendering, preview and HTTP payload checks.

Local validation uses synthetic geometry and mocked HTTP transport; Raspberry Pi output has not been exercised for this integration branch. Perception and robot rotation remain separate work.
