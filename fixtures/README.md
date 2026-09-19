# Synthetic development fixtures

These files illustrate the contracts and let each stage develop independently. They are **not a real capture dataset, measured calibration, simulated optimal shot, or hardware validation**.

| File | Purpose |
| --- | --- |
| `table_geometry.json` | Normalized 1 × 0.5 table, radius 0.0142875, six nominal pocket mouths |
| `captures/synthetic.json` | One file-backed capture manifest with explicitly synthetic calibration/view IDs |
| `captures/synthetic.ppm` | Generated top-down color schematic of the state; not an oblique camera image |
| `table_states/direct_shot.json` | Known cue, solid, stripe, and eight-ball positions |
| `game_contexts/solids.json` | Current shooter is solids, after the break, cue ball placed |
| `table_states/eight_ball_finish.json` | Cleared solids group; eight ball is now the eligible target |
| `shot_plans/aim_only.json` | Required strike/called pot with no optional guides |
| `shot_plans/direct_shot.json` | Same aim plus a geometric ghost ball, pre-contact cue path, and target-to-pocket path |
| `shot_plans/eight_ball_finish.json` | Illustrative called eight-ball pot with identified ball paths |
| `projection_targets/synthetic.json` | Artificial perspective homography and 800×400 output for rasterizer development |

The cue at `(0.3,0.15)` aims toward a solid at `(0.6,0.3)`, aligned with the bottom-right pocket at `(1,0.5)`. All distances use the long-side unit, with top-left origin, x right and y down. The sample ghost ball uses `B - 2r*normalize(P-B)`. The required cue-stick speed of 0.75 table lengths/second is an illustrative value, not an optimized recommendation. No post-impact trajectory, success probability, or physics validation is asserted. All JSON files use schema 3. The eight-ball finish fixture moves the eight to the target position and removes the solid; the RGB schematic depicts only the original direct-shot layout.

The schematic shows the agreed top-down orientation: x right and y down. It is not camera calibration; actual oblique images need a transform. Its pixels are unchanged because all distances and radii were uniformly normalized. The projector homography was converted to preserve the previous pixel locations. It has no usable camera calibration artifact. Perception needs separately recorded real RGB images to evaluate detection and model-estimated depth.

The fixture validator checks parsing, references, and shared geometry/identity. It does not run any stage algorithm. To add real fixtures, include recorded acquisition settings and calibration provenance, measured truth where possible, and notes about permitted uses. Keep large/private device data under ignored `data/local/` and machine calibration under `calibration/local/`.
