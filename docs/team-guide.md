# Team implementation guide

Read [contracts.md](contracts.md) first. Each owner can work from checked-in fixtures. No teammate needs an upstream model or downstream hardware implementation to start.

## Ownership

| Owner | Primary files | Shared responsibility |
| --- | --- | --- |
| Perception | `pool/perception/`, camera adapters, capture fixtures | Camera calibration and acquisition metadata |
| Planning | `pool/planning/`, table-state scenarios | Shot semantics and rules context |
| Projection | `pool/projection/`, projector adapters, target fixtures | Projector calibration; pose mapping with hardware owner |
| Agreed integration owner | `pool/pipeline.py`, `app.py`, router, shared contracts | Connecting stages; dependencies; shared fixture/schema review |

The three stage services are the implementation entry points. Keep their public methods stable. Add focused internal classes/functions as needed; inherit from neither a shared service base nor one another. `Protocol` allows existing functions/services to be wrapped without changing their implementation family.

An integration owner is a responsibility one teammate takes on, not a required fourth teammate. Decide who owns camera acquisition versus base control before physical integration.

## Teammate 1: perception

Entry point: `src/companion/pool/perception/service.py`, `PerceptionService.estimate()`.

Input: one or more RGB captures with calibration/view IDs, plus measured `TableGeometry`. Output: `PerceptionReady(TableState)` or a typed incomplete/unusable outcome.

Suggested internal split as code grows:

```text
detector.py       Image → masks/centers/type probabilities
localization.py   Calibrated observations → ball centers in table meters
fusion.py         Merge observations/deduplicate; coverage and uncertainty
calibration.py    RGB camera intrinsics/extrinsics and table pose
```

First milestone:

1. Record stationary real table views and calibration metadata.
2. Identify the playing plane and its marked coordinate axes.
3. Detect balls in RGB. Classify cue/solid/stripe/eight; retain unknown when evidence is insufficient.
4. Run a depth-estimation model on RGB and recover physical centers with ball-height correction. Keep the estimated depth inside perception and establish its scale against known table/ball geometry before returning meters.
5. Compare estimates against measured positions and emit a valid JSON state.

Then add multi-view capture fusion. The merge occurs in table coordinates; panorama stitching is not required. Deduplicate repeated observations rather than concatenating balls. Never infer a missing stripe's identity confidently from one occluded side. Return `NeedsMoreViews` when coverage is unresolved. Any ball movement during a scan invalidates that stationary-state batch.

Acceptance evidence: real oblique captures plus measured truth; localization error in mm at near/far table regions; classification results including unknowns; duplicate handling across overlapping views; incomplete and moving-scene outcomes. The team must choose an accuracy target after measuring the hardware; no claimed threshold is baked into the skeleton.

Run independently:

```sh
PYTHONPATH=src python3.11 -m companion.app perceive \
  --captures fixtures/captures/synthetic.json \
  --geometry fixtures/table_geometry.json \
  --output artifacts/table_state.json
```

The synthetic capture only exercises file loading; replace it with real recorded captures and real calibration for perception development. It is not a useful accuracy benchmark.

## Teammate 2: shot planning

Entry point: `src/companion/pool/planning/service.py`, `PlanningService.plan()`.

Input: physical table state and explicit practice/game context. Output: `PlanningReady(ShotPlan)` or an information/feasibility outcome. No image, camera, projector, or motor dependency is allowed.

Suggested internal split:

```text
candidates.py     Target ball/pocket pairs, cue aim, speed candidates
physics.py        Deterministic trajectory simulation behind a local interface
scoring.py        Target success, scratch risk, difficulty, position
```

V1: output one anchored cue direction plus a cue-alignment segment. V2: add cue/object-ball paths, ghost-ball position, target pocket, and optionally cue-ball speed. Keep the same `ShotPlan` boundary.

Recommended first algorithm: direct no-spin shots. For object-ball position B, pocket position P, and radius r:

```text
d = normalize(P - B)
ghost_ball = B - 2r*d
cue_direction = normalize(ghost_ball - cue_position)
```

Check path clearance using ball radius, finite segments, and endpoint/contact geometry; a zero-width line intersection check is insufficient. Reject invalid/degenerate candidate geometry. Simulate a bounded set of directions/speeds, then rank according to a documented objective. Handle scratch risk and other balls as obstacles, including unknown types. Tune rolling friction, cushion response, and speed against real observations before presenting predicted paths as accurate.

Physics engines, ML, or RL can fit behind this interface. Analytic geometry plus a simple simulation is the suggested initial approach; no choice is mandated. “Best” means best among the evaluated candidates under the model. A confidence number would require uncertainty trials or evaluation, so the scaffold does not invent one.

Acceptance evidence: direct unobstructed shot, blocked cue path, blocked target path, near-rail/degenerate shot, missing cue ball, unknown types, no feasible candidate, and selected-group targeting. Include repeatable outputs and simulated/real outcomes once a simulator exists.

Run independently:

```sh
PYTHONPATH=src python3.11 -m companion.app plan \
  --state fixtures/table_states/direct_shot.json \
  --output artifacts/shot_plan.json
```

Add `--group solids` or `--group stripes` for group-practice mode. Without it, the documented demo mode applies. Full eight-ball rules are future work.

## Teammate 3: projection

Entry point: `src/companion/pool/projection/service.py`, `ProjectionService.render()`.

Input: physical shot geometry plus calibrated projector target (including table geometry). Output: an RGB frame at native projector resolution. Do not run shot planning or move hardware inside this method.

Suggested internal split:

```text
calibration.py    Resolve/estimate table-to-pixel mapping for each projector pose
renderer.py       Draw guidance, apply perspective, clip to table/raster
```

First milestone:

1. Load the synthetic plan/target and render a black-background pixel image.
2. Support aim-only plans, styled guide roles, and an optional ghost-ball circle.
3. Calibrate a fixed real projector pose using measured points/projected dots.
4. Verify additional held-out projected points on the cloth.
5. Project a known physical line at the intended mount distance and height.

Then add repeatable rotating poses and target selection. A base angle by itself is not a calibration. Moving the mount/table/projector or changing resolution/keystone settings invalidates the relevant mapping. A renderer should not extrapolate blindly through a homography horizon. Clip guidance to the playing surface and projectable footprint.

Acceptance evidence: synthetic pixel output; known table points mapping to expected pixels; held-out physical dot errors; finite clipping near boundaries; aim-only versus v2 plans; RGB channel/dimension correctness; repeatable alignment after rotation. Do not count a scaled desktop preview as physical validation.

Run independently:

```sh
PYTHONPATH=src python3.11 -m companion.app render \
  --plan fixtures/shot_plans/direct_shot.json \
  --target fixtures/projection_targets/synthetic.json \
  --output artifacts/projection.ppm
```

The target homography is artificial. Replace it with measured calibration before hardware projection. PPM is a simple inspection format; the hardware adapter can send raw pixels directly.

## Integration and collaboration

Suggested feature branches: `feat/perception`, `feat/planning`, `feat/projection`. No branches are created by the scaffold. Agree changes to `contracts/` before depending on them. Keep private captures, weights, and machine-specific calibration under ignored `data/local/` or `calibration/local/`; share small intentional test assets through fixtures.

Avoid adding heavy dependencies to the shared core. When selecting an implementation, add a scoped optional dependency group (for example perception/planning/projection) with agreed versions in `pyproject.toml`. No speculative dependencies or empty extras are installed now.

Before handoff:

1. Keep the existing fixture check and contract/integration tests passing.
2. Add evidence-driven tests for your algorithm's failure modes, not just its happy path.
3. Produce an artifact from the independent stage command.
4. Document calibration assumptions, units, limits, and installation needs.
5. Integrate one stationary view and one fixed projection pose before scanning/LLM orchestration.

The current tests use explicit test doubles for stage orchestration. Passing them proves contract compatibility, not functioning perception, physics, or projection.
