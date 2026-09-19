# Shared contracts: schema version 1

Source under `src/companion/pool/contracts/` is authoritative for field names and enum values. This document defines their meaning. Changes to stage boundaries require team agreement and updated fixtures.

## Coordinates and identity

- All table positions/distances use **meters**; optional cue-ball speed uses meters/second.
- Origin is a physically marked corner of the playing surface, not the outside of the rails. +x follows its long edge; +y follows its short edge. +z points above the cloth. Mark and photograph the axes before calibrating.
- The cloth is `z=0`; the playing rectangle is `[0, length_m] × [0, width_m]`.
- A ball's position is the physical sphere center's `(x,y)`, vertically projected onto the cloth. The actual center is at `z=ball_radius_m`. It is not an image bounding-box center.
- `UnitVector2` is dimensionless and must have length 1 within `1e-6`. It is not a velocity.
- Cue direction points **forward into the intended shot** from the cue-ball center.
- Pixel origin is upper-left, pixel x increases right, and pixel y increases down. Table y must not be inferred from image y.
- `table_id` identifies the geometry AND its fixed coordinate frame. Changing physical origin/axes requires a new ID.
- Times are Unix seconds on a common clock. A fused observation records the first and last acquisition times.

The `2` in `Point2`, `UnitVector2`, and `Segment2` means **two-dimensional (x and y)**, not a version number. `Point2(x,y)` is a position, `UnitVector2(x,y)` is a unit direction, and `Segment2(start,end)` is a finite line between two positions. These are immutable dataclasses. Coordinates must be finite; segments must have nonzero length. Boundaries use tuples rather than mutable lists.

## Camera input

`CameraCapture` is file-backed for the initial recorded-input workflow:

| Field | Meaning |
| --- | --- |
| `capture_id` | Unique capture within the batch |
| `captured_at_s` | Exposure/acquisition time, not processing time |
| `rgb_path` | Color image; decode to RGB, not OpenCV's default BGR |
| `calibration_id` | Reference to RGB camera intrinsics, distortion, and calibration version |
| `view_id` | Reference to calibrated camera pose relative to the table |

`CaptureBatch(batch_id, captures)` contains at least one capture. The loader resolves image paths against the JSON manifest's directory. Camera matrices are not embedded in the manifest: teammate 1 owns calibration artifacts and their resolver. An in-memory camera adapter may be added later without changing downstream pool contracts.

The input is RGB only. Perception runs a depth-estimation model internally; its depth map is not a shared input or output contract. Document the model's output units/scale and how estimates are converted to table meters using known geometry. A batch's ordering does not establish that balls stayed stationary.

## Perception → planning

`TableGeometry` contains `table_id`, `length_m`, `width_m`, `ball_radius_m`, and six `Pocket` objects. Each pocket has unique `id`, nominal mouth-center `position`, and `mouth_width_m`. Precise cushion-jaw/pocket acceptance geometry may need a future extension. The sample dimensions are not measurements of the real table.

`TableState` contains:

| Field | Meaning |
| --- | --- |
| `observation_id` | New ID for this fused observation |
| `capture_started_at_s`, `capture_ended_at_s` | Time interval represented by the state |
| `geometry` | Supplied table geometry, unchanged by perception |
| `balls` | Tuple of observed `Ball` objects |
| `coverage` | `complete`, `partial`, or `unknown` |

`Ball` fields are `id`, `position: Point2`, `type: BallType`, optional `type_confidence`, and optional `position_uncertainty_m`.

Types are **`cue`, `solid`, `stripe`, `eight`, `unknown`**. IDs distinguish balls of the same type and stay stable during batch fusion; persistence across separate shots is not required in v1. Pocketed balls are absent, not placed at `(0,0)`. A partial state may lack a cue ball. Multiple cue or eight balls, duplicate IDs, and positions outside the playing rectangle are rejected.

Type confidence is in `[0,1]`. Position uncertainty is a conservative radial error estimate in meters whose estimation method perception must document. Missing values mean unavailable, not perfect certainty. Neither field is a shot-success probability. Overlapping balls and other physically questionable estimates must be handled by stage logic, not silently corrected by serialization.

Coverage is complete only when no unresolved unobserved/occluded region may conceal another ball. Seeing the table outline is insufficient. Expected outcomes:

- `PerceptionReady(state)`: requires complete coverage.
- `NeedsMoreViews(reason, missing_regions=(), partial_state=None)`: missing regions are polygons of table-coordinate points. Empty means the region cannot yet be localized.
- `UnusableCapture(reason)`: for motion, severe blur, missing calibration, or other unusable captures. The application decides how to recapture.

## Game context

The scaffold supports two practice modes:

- `GameContext(mode=DEMO)`: consider solid/stripe targets; the eight ball is an obstacle.
- `GameContext(mode=GROUP_PRACTICE, player_group=SOLIDS|STRIPES)`: consider only the selected group; the eight ball remains an obstacle.

These modes are **not a regulation eight-ball rules engine**. Unknown balls remain obstacles but are not assumed to be legal targets. Full rules, winning-shot legality, fouls, ball-in-hand, and turn tracking require an agreed extension. Positions alone cannot establish those facts.

## Planning → projection

| `ShotPlan` field | Required? | Meaning |
| --- | --- | --- |
| `observation_id` | Yes | Source state ID |
| `table_id` | Yes | Source geometry's coordinate-frame ID |
| `cue_aim` | Yes | `CueAim(cue_ball_id, origin, direction)` |
| `guides` | Defaults to empty | Tuple of `GuideSegment(role, segment)` |
| `ghost_ball` | Optional | Cue-ball center at first object-ball contact |
| `target_ball_id` | Optional | An observed object ball |
| `target_pocket_id` | Optional | A pocket in the table geometry |
| `suggested_cue_ball_speed_mps` | Optional | Positive initial cue-ball speed, not cue-stick speed or arbitrary power |

`CueAim.origin` copies the cue-ball position exactly from the source state. Do not round/mutate it. A bare vector without an origin is insufficient to draw guidance.

Guide roles:

| Value | Meaning |
| --- | --- |
| `cue_alignment` | Cue-stick alignment behind the cue ball, ordered toward it |
| `cue_ball_before_contact` | Cue-ball center path before first contact |
| `cue_ball_after_contact` | Predicted cue-ball center path after contact |
| `object_ball_path` | Predicted object-ball center path |

V1 returns cue aim and preferably one alignment segment. The renderer must also accept no guides, using a documented default alignment length clipped to the table/projectable area. V2 adds paths and optional ghost-ball/target metadata using the same schema. The aim remains authoritative. Multiple ordered segments of one role can represent cushion bounces for one path. Multiple simultaneously moving object-ball paths require a future per-path ball identifier.

The planner owns physical endpoints and predictions. The renderer owns colors, stroke widths, annotation placement, and clipping. It must not invent predicted trajectories when absent. Ghost-ball radius and playing-area boundaries come from the projection target's table geometry.

Outcomes: `PlanningReady(plan)`, `InsufficientInformation(reason)`, or `NoFeasibleShot(reason)`. The last means none found under the implemented model/search, not proof that no possible shot exists. Do not use a zero vector to encode failure.

## Projection target and output

`ProjectionTarget` has `geometry: TableGeometry`, `calibration_id`, `pose_id`, `width_px`, `height_px`, and `table_to_pixel`, a finite invertible 3×3 homography. Its `table_id` property delegates to `geometry.table_id`.

```text
[a,b,c] = H × [x_m,y_m,1]
u_px = a/c
v_px = b/c
```

The renderer must handle near-zero denominators, raster boundaries, and the table footprint. A homography does not identify physical occluders; handling those needs additional observation/masking. Calibration must establish acceptable optical distortion or compensate inside the renderer. Automatic keystone/scaling must remain consistent with calibration.

`ProjectionFrame` contains observation/table/calibration/pose IDs, dimensions, and packed immutable `rgb: bytes`: **RGB8, row-major, no alpha/padding**, exactly `width_px * height_px * 3` bytes. In NumPy terms, use an unsigned-byte `(height,width,3)` array exported with `tobytes()`. Use black outside guidance. `save_ppm()` writes a P6 artifact without additional dependencies.

Before display, the application must verify actual settled robot pose and calibration against the frame. Skeleton metadata checks do not establish physical alignment, observation freshness, or coverage.

## Serialization and failures

All JSON files have the same envelope:

```json
{
  "schema_version": 1,
  "kind": "shot_plan",
  "data": {
    "observation_id": "example-observation",
    "table_id": "example-table",
    "cue_aim": {
      "cue_ball_id": "cue",
      "origin": {"x": 0.6, "y": 0.3},
      "direction": {"x": 1.0, "y": 0.0}
    }
  }
}
```

Kinds: `capture_batch`, `table_geometry`, `table_state`, `shot_plan`, `projection_target`. Enum values are lowercase strings; tuples are arrays in JSON. Optional fields can be omitted where defaults exist; `guides` can be omitted but cannot be `null`. Use shared codecs, not custom JSON/pickle per stage. Unsupported versions, unknown enum values/fields, and invalid geometry are rejected.

Expected result variants are in-process dataclasses. CLI success writes a state/plan/frame and exits 0. An expected non-ready outcome prints its type/reason and exits 1. Invalid input or an unimplemented stage exits 2. Consumers must check exit status and must not reuse an old output file after failure.

Breaking schema changes require agreement, a version update, and fixture/test updates. Stage-internal experiments do not need to change shared types.
