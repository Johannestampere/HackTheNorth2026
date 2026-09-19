# Pool planning: first working milestone

The planner now runs real Pooltool 0.6.0 simulations. Its existing schema-3 input/output contract is unchanged. Perception and projection do not import Pooltool.

## Install and run

Use Python 3.11–3.13 (Pooltool 0.6 does not support 3.14). From the repository root:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[planning]'
PYTHONPATH=src .venv/bin/python -m companion.app plan \
  --state fixtures/table_states/direct_shot.json \
  --game fixtures/game_contexts/solids.json \
  --table-length-m 2.0 --output artifacts/shot_plan.json
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Pooltool's dependencies include Panda3D even though this code opens no 3D window. Its first import creates configuration files under `~/.config/pooltool`. The adapter explicitly constructs the release's default physics resolver so personal resolver settings do not affect results. Initial imports/JIT compilation can take longer than subsequent runs.

## Watch a game

```sh
PYTHONPATH=src .venv/bin/python -m companion.pool.planning.demo
# A short two-shot endgame, using the existing fixture:
PYTHONPATH=src .venv/bin/python -m companion.pool.planning.demo \
  --state fixtures/table_states/direct_shot.json --output artifacts/pool-endgame.html
```

Open `artifacts/pool-game.html` in a browser. It is a standalone offline Canvas replay with play/pause, restart, shot selection, timeline scrubbing, playback speed, and predicted paths. The adjacent JSON stores trajectories and shot summaries. Generated files are ignored by Git; teammates generate them with the same command.

The default seed is 7; change it with `--seed`. This is a generated 16-ball spread layout, not a simulated break. Groups are assigned; solids start. Each simulated successful called pot keeps the turn. The actual settled positions feed the next planning call. A legal called eight-ball pot ends the rack.

**Demo limitation:** if the planner finds no direct pot, the demo logs a player switch without executing a shot. That is a visualization convention, not a legal pass or implemented safety. If neither player has a supported shot, the demo stops without declaring a winner. It also stops at `--max-shots` (default 32). This first version assumes exact nominal execution; it does not demonstrate human aiming accuracy or win probability.

## File ownership and flow

- `service.py`: configuration, input checks, bounded search, scoring, and `ShotPlan` construction. `last_selection` exposes private diagnostics only for tests/demo.
- `candidates.py`: private candidate record, legal target group, and finite swept-ball path checks.
- `physics.py`: all Pooltool imports, coordinate conversion, table construction, ghost-ball candidates, simulation, outcome classification, and trajectories.
- `demo.py` / `replay.html`: synthetic self-play runner and offline viewer.

`PooltoolPlanner.plan(state, game)` performs:

1. Reject unsupported game contexts, incomplete observations, missing cue/eight, unknown types, overlapping balls, and balls too close to rails for this initializer.
2. Construct stationary balls on a standard six-pocket table. Preserve observation ball IDs; map pocket IDs by location.
3. Generate legal ball/pocket candidates using Pooltool's ghost-ball and potting-angle helpers. Check swept cue/object paths, ghost-ball bounds, pocket jaws, and cuts no greater than 75 degrees.
4. Search up to 16 geometric candidates, five cue speeds (0.35, 0.5, 0.7, 0.95, 1.25 table lengths/s), and offsets of 0, -0.75, +0.75 degrees. Maximum 240 nominal simulations per call.
5. Inspect collision and pocket events. Discard scratches, illegal first contact, early/wrong-pocket eight-ball losses, and failed called pots. A shot must pocket a ball or hit a rail after first contact.
6. Shortlist 12 successful trials by win, geometric ease, smaller aim adjustment, and lower speed. Compare their settled positions: win first, then availability and ease of the next direct pot, then current shot difficulty and speed. This is a bounded heuristic, not a game-tree search or learned probability.
7. Return a forward unit cue vector, cue-tip speed, called ball/pocket, actual simulated cue center at first contact, and event-to-event path guides. If none succeeds, return `NoFeasibleShot`; this does not mean no legal shot exists.

The adapter inspects Pooltool events directly for the supported rule subset, rather than initializing its full rules engine with a reconstructed game history. Breaks, ball-in-hand placement, jumped balls, physical cue fouls, and safeties remain unsupported. Expected input limitations return typed outcomes; unexpected simulator errors are not silently converted into a successful recommendation.

## Units and calibration

Shared geometry remains long side = 1, width = short/long, origin top-left. The adapter uses:

```text
engine_x_m = shared_y * physical_length_m
engine_y_m = shared_x * physical_length_m
engine_cue_speed_mps = shared_cue_speed * physical_length_m
shared_direction = (sin(engine_phi), cos(engine_phi))
```

Default physical length is an explicit **2.0 m assumption**, overrideable through `PlannerConfig` or `--table-length-m`. It is not inferred from RGB. Both ball radius and pocket mouth widths use this same scale. The MVP requires four corner pockets and two long-rail midpoint pockets; corner widths must match each other, as must side widths. Unsupported geometry is rejected, not averaged away.

Pocket jaw shape, pocket depth, mass, friction and restitution use uncalibrated Pooltool defaults; cushion height is set relative to ball radius. Cue offsets and elevation are explicitly zero. Even correct geometry does not make predicted travel distances accurate on a real table until these physical parameters and player speed are calibrated.

## Next milestones

1. Perturb measured positions, aim, and speed to rank robust shots rather than nominal successes; use provided perception uncertainty.
2. Calibrate rolling distance and cushion/pocket behavior on the physical table.
3. Add deeper own/opponent lookahead, banks and a contract for safeties. A broader search is needed to keep games progressing without demo skips.
4. Consider learned value estimates only after collecting simulator/real outcome data.

Sources: [Pooltool release](https://pypi.org/project/pooltool-billiards/0.6.0/), [potting helper implementation](https://github.com/ekiefl/pooltool/blob/v0.6.0/pooltool/ai/pot/core.py), [table specification](https://pooltool.readthedocs.io/en/latest/resources/table_specs.html).
