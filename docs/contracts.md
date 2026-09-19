# Shared contracts: schema version 3

Source under `src/companion/pool/contracts/` is authoritative for field names and enum values. This document defines their meaning. Changes to stage boundaries require team agreement and updated fixtures.

## Coordinates and identity

- One **table-length unit** is the long side of the playing surface. `length` is always **1.0**.
- `width = short_side / long_side`. Both axes use the same scale; do **not** independently map the short side to 1.
- Origin is the **top-left playing-surface corner in one agreed top-down view**. +x runs right along the long side; +y runs down along the short side. This fixed corner does not change when the camera rotates.
- For a 2:1 table: top-left `(0,0)`, top-right `(1,0)`, bottom-left `(0,0.5)`, bottom-right `(1,0.5)`.
- Ball radius, pocket mouth widths, position uncertainty, ghost-ball positions, and all line endpoints use the same table-length unit.
- The cloth is `z=0`; a ball center is at `z=ball_radius` if a 3D calculation is needed internally. Its exported `(x,y)` is its physical center vertically projected onto the cloth, not an image bounding-box center.
- `UnitVector2` is dimensionless and must have length 1 within `1e-6`. Uniform normalization does not change aim angles. Cue direction points forward into the shot.
- `cue_stick_speed` is in **table lengths per second**, separate from direction. It is not m/s or a power percentage.
- Pixel origin is upper-left, x right and y down, but pixel positions still require a perspective transform into table coordinates. Do not divide slanted image x/y by separate bounding-box dimensions.
- `table_id` identifies the geometry and fixed coordinate frame. Changing origin/axes or unit interpretation requires a new ID and matching calibration.
- Times remain Unix seconds on a common clock. A fused observation records first and last acquisition times.

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

The input is RGB only. Perception runs a depth-estimation model internally; its depth map is not a shared input or output contract. Document the model's output units/scale and how estimates are converted to table-length units using the table proportions. A batch's ordering does not establish that balls stayed stationary.

## Perception → planning

`TableGeometry` contains `table_id`, `length`, `width`, `ball_radius`, and six `Pocket` objects. Each pocket has unique `id`, nominal mouth-center `position`, and `mouth_width`. Precise cushion-jaw/pocket acceptance geometry may need a future extension. The sample dimensions are not measurements of the real table.

`TableState` contains:

| Field | Meaning |
| --- | --- |
| `observation_id` | New ID for this fused observation |
| `capture_started_at_s`, `capture_ended_at_s` | Time interval represented by the state |
| `geometry` | Supplied table geometry, unchanged by perception |
| `balls` | Tuple of observed `Ball` objects |
| `coverage` | `complete`, `partial`, or `unknown` |

`Ball` fields are `id`, `position: Point2`, `type: BallType`, optional `type_confidence`, and optional `position_uncertainty`.

Types are **`cue`, `solid`, `stripe`, `eight`, `unknown`**. IDs distinguish balls of the same type and stay stable during batch fusion; persistence across separate shots is not required in v1. Pocketed balls are absent, not placed at `(0,0)`. A partial state may lack a cue ball. Multiple cue or eight balls, duplicate IDs, and positions outside the playing rectangle are rejected.

Type confidence is in `[0,1]`. Position uncertainty is a conservative radial error estimate in table-length units whose estimation method perception must document. Missing values mean unavailable, not perfect certainty. Neither field is a shot-success probability. Overlapping balls and other physically questionable estimates must be handled by stage logic, not silently corrected by serialization.

Coverage is complete only when no unresolved unobserved/occluded region may conceal another ball. Seeing the table outline is insufficient. Expected outcomes:

- `PerceptionReady(state)`: requires complete coverage.
- `NeedsMoreViews(reason, missing_regions=(), partial_state=None)`: missing regions are polygons of table-coordinate points. Empty means the region cannot yet be localized.
- `UnusableCapture(reason)`: for motion, severe blur, missing calibration, or other unusable captures. The application decides how to recapture.

## Game context: operator → planning

The objective is to select the current shot for the shooter's eventual rack success, including a legal eight-ball finish. The initial rules baseline is [WPA 8-ball, effective September 15, 2025](https://wpapool.com/wp-content/uploads/2025/09/2025.09.15-WPA-Rules.pdf). The contract supports ordinary pot attempts; it does not implement a complete referee or simulator.

`GameContext` is supplied by the application/operator, **not inferred by perception**:

| Field | Meaning |
| --- | --- |
| `player_group` | Required: `solids`, `stripes`, or explicit `null` for open/unknown assignment |
| `is_break` | Boolean, default false; true means the opening break |
| `ball_in_hand` | Boolean, default false; true means cue-ball placement is still part of the decision |

The MVP requires assigned groups, after the break, and a placed cue ball. Break, open/unknown assignment, and ball-in-hand inputs return `InsufficientInformation` instead of being silently treated as ordinary play. After a human places the cue ball, recapture it and supply `ball_in_hand=false` for the aim-only placement scope.

While own-group balls remain, the called target must belong to that group. Once the group is cleared, the called target is the eight. Derive remaining balls from the complete observation; do not maintain a duplicate count or `can_hit_eight` flag. Unknown types prevent confirming that a group is cleared. Unknown balls are still physical obstacles.

```python
from companion.pool.contracts import GameContext, PlayerGroup

game = GameContext(player_group=PlayerGroup.SOLIDS)
```

Planning must evaluate first contact, scratches/fouls, the intended called pocket, and terminal rack outcomes from simulated events. A legal target ID alone does not establish a legal shot or a win. During lookahead, a legal pot normally retains the shooter's turn; do not blindly alternate turns at each depth. Search scores and sampled probabilities stay internal until their semantics and calibration are established.

## Planning → projection

| `ShotPlan` field | Required? | Meaning |
| --- | --- | --- |
| `observation_id` | Yes | Source state ID |
| `table_id` | Yes | Source geometry's coordinate-frame ID |
| `cue_aim` | Yes | `CueAim(cue_ball_id, origin, direction)` |
| `guides` | Defaults to empty | Tuple of `GuideSegment(role, ball_id, segment)` |
| `ghost_ball` | Optional | Cue-ball center at first object-ball contact |
| `target_ball_id` | Yes | Observed object ball for the intended called pot, including a legal eight-ball attempt |
| `target_pocket_id` | Yes | Intended called pocket from table geometry |
| `cue_stick_speed` | Yes | Positive cue-tip speed immediately before impact, in table lengths/second; not initial cue-ball speed |

**Strike assumption:** level, center-ball hit, with no intentional tip offset. The direction is unit length and contains no power information. Even an aim-only display needs an explicit planned speed: later motion depends on it. The simulator adapter must not substitute initial cue-ball speed for cue-stick speed. A UI may show the physical speed or a separately calibrated strength label; the renderer must not invent a percentage. Spin/elevation controls are outside the current contract.

`CueAim.origin` copies the cue-ball position exactly from the source state. Do not round/mutate it. A bare vector without an origin is insufficient to draw guidance.

Guide roles:

| Value | Meaning |
| --- | --- |
| `cue_alignment` | Cue-stick alignment behind the cue ball, ordered toward it |
| `cue_ball_before_contact` | Cue-ball center path before first contact |
| `cue_ball_after_contact` | Predicted cue-ball center path after contact |
| `object_ball_path` | Predicted object-ball center path |

The first visual version draws cue aim; later versions add paths using the same schema. `guides=()` means the renderer draws alignment from `cue_aim` with a documented default length clipped to the table/projectable area. It does not mean a zero-length shot or stationary balls. Called target and strike speed remain required even when no trajectories are displayed.

Each guide has an observed `ball_id`. Cue alignment and cue-ball path roles must reference the cue ball; `object_ball_path` must reference an object ball. Multiple segments for a ball are in travel order. The list is not a globally synchronized timeline across balls. Paths are nominal ball-center predictions conditional on the planned strike and physics model; they are not guaranteed outcomes or uncertainty bands.

Safety shots, bank/kick-specific metadata, spin controls, confidence percentages, and training data are not added to this shared contract. Introduce fields only when a consumer needs them. In particular, a future safety representation must be agreed before relaxing the required called-pot fields.

The planner owns physical endpoints and predictions. The renderer owns colors, stroke widths, annotation placement, and clipping. It must not invent predicted trajectories when absent. Ghost-ball radius and playing-area boundaries come from the projection target's table geometry.

Outcomes: `PlanningReady(plan)`, `InsufficientInformation(reason)`, or `NoFeasibleShot(reason)`. The last means none found under the implemented model/search, not proof that no possible shot exists. Do not use a zero vector to encode failure.

## Projection target and output

`ProjectionTarget` has `geometry: TableGeometry`, `calibration_id`, `pose_id`, `width_px`, `height_px`, and `table_to_pixel`, a finite invertible 3×3 homography. Its `table_id` property delegates to `geometry.table_id`.

```text
[a,b,c] = H × [x,y,1]
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
  "schema_version": 3,
  "kind": "shot_plan",
  "data": {
    "observation_id": "example-observation",
    "table_id": "example-table",
    "cue_aim": {
      "cue_ball_id": "cue",
      "origin": {"x": 0.3, "y": 0.15},
      "direction": {"x": 1.0, "y": 0.0}
    },
    "cue_stick_speed": 0.75,
    "target_ball_id": "solid-1",
    "target_pocket_id": "xmax-ymax"
  }
}
```

Kinds: `capture_batch`, `table_geometry`, `table_state`, `game_context`, `shot_plan`, `projection_target`. Enum values are lowercase strings; tuples are arrays in JSON. Optional fields can be omitted where defaults exist; `guides` can be omitted but cannot be `null`. Use shared codecs, not custom JSON/pickle per stage. Unsupported versions, unknown enum values/fields, and invalid geometry are rejected.

Expected result variants are in-process dataclasses. CLI success writes a state/plan/frame and exits 0. An expected non-ready outcome prints its type/reason and exits 1. Invalid input or an unimplemented stage exits 2. Consumers must check exit status and must not reuse an old output file after failure.

Breaking schema changes require agreement, a version update, and fixture/test updates. Stage-internal experiments do not need to change shared types.

## Physical simulation scale

Perception and projection need no meter measurement for this coordinate contract. A metric physics engine does need a scale: planning must obtain a measured long-side length or explicitly configure an assumed one. Keep that conversion inside the simulator adapter, not in camera output.

For physical long-side length `L` meters:

```text
position_meters = position_table_units * L
radius_meters = ball_radius * L
cue_tip_speed_mps = cue_stick_speed * L
```

Convert simulated positions/speeds back by dividing by `L`, and adapt the y-axis if the engine uses a different orientation. Friction/acceleration and other dimensional physics must be consistent with that engine's units; changing coordinates alone does not calibrate the physics. Do not treat a normalized length of 1 as a measured one-meter table. Aim geometry works without physical scale; speed and trajectory predictions remain conditional on the chosen model/scale.

## Migrating to schema 3

- All JSON envelopes now use version **3**; versions 1 and 2 are rejected rather than silently reinterpreted.
- Geometry fields are `length`, `width`, `ball_radius`; pockets use `mouth_width`; balls use optional `position_uncertainty`. The `_m` suffixes have been removed.
- Divide all previous metric positions, lengths, radii, and uncertainties by the physical long-side length. Set `length=1.0`; keep `width=short_side/long_side`. Do not rescale unit directions, confidence, or timestamps. Establish the new top-left origin/right-down axes explicitly if the old frame differs.
- `cue_stick_speed_mps` becomes `cue_stick_speed` in table lengths/second: divide the old cue-stick speed by the same physical long-side length. Schema 1's optional cue-ball speed is a different velocity and must not simply be renamed.
- Update projector calibration. If only the length scale changes and the origin/axes are already identical, `H_normalized = H_meters @ diag(L, L, 1)`. If origin/axes change, include that transform or recalibrate. The sample fixture preserves the same pixel locations.
- Keep the schema-2 game changes: explicit `GameContext(player_group=...)`, required `--game`, called ball/pocket, and ball IDs on guides. No default practice mode remains.
- Fixture geometry is now `1.0 × 0.5`, radius `0.0142875`, and illustrative cue-stick speed `0.75` table lengths/second. These are synthetic values, not measurements or optimized shots.

Visual milestones (aim only now, trajectories later) are independent of the JSON schema version. Both use schema 3.
