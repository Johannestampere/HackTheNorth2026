# Synthetic development fixtures

These files illustrate the contracts and let each stage develop independently. They are **not a real capture dataset, measured calibration, simulated optimal shot, or hardware validation**.

| File | Purpose |
| --- | --- |
| `table_geometry.json` | Artificial 2 m × 1 m table, 0.028575 m ball radius, six nominal pocket mouths |
| `captures/synthetic.json` | One file-backed capture manifest with explicitly synthetic calibration/view IDs |
| `captures/synthetic.ppm` | Generated top-down color schematic of the state; not an oblique camera image |
| `table_states/direct_shot.json` | Known cue, solid, stripe, and eight-ball positions |
| `shot_plans/aim_only.json` | V1 cue aim and one alignment segment |
| `shot_plans/direct_shot.json` | Same aim plus a geometric ghost ball, pre-contact cue path, and target-to-pocket path |
| `projection_targets/synthetic.json` | Artificial perspective homography and 800×400 output for rasterizer development |

The cue at `(0.6,0.3)` aims toward a solid at `(1.2,0.6)`, which is aligned with the nominal corner pocket at `(2,1)`. The sample ghost ball uses `B - 2r*normalize(P-B)`. No speed, post-impact trajectory, success probability, or physics validation is asserted.

The schematic uses table x to image right and table y to image down only for convenient illustration; do not learn that convention as a camera calibration. It has no usable camera calibration artifact. Perception needs separately recorded real RGB images to evaluate detection and model-estimated depth.

The fixture validator checks parsing, references, and shared geometry/identity. It does not run any stage algorithm. To add real fixtures, include recorded acquisition settings and calibration provenance, measured truth where possible, and notes about permitted uses. Keep large/private device data under ignored `data/local/` and machine calibration under `calibration/local/`.
