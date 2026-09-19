# AI Projector Companion

A robot with an RGB camera, a projector, and a 360° rotating base that brings an AI interface into the physical world. It observes a scene, selects a useful tool, and projects guidance onto the relevant surface. Perception estimates depth from the RGB image using a model; there is no depth camera. Audio is planned later.

The first tool is **pool**. The robot sits beside the table, some distance away and roughly 1–2 m high; it is **not mounted overhead**. A general multimodal LLM will inspect the initial image and choose `pool` or `generalization`. Pool may request multiple views, identify balls in table coordinates, plan a shot, and prepare a perspective-correct projector image.

## Current status

The **pool planner now runs Pooltool simulations** to choose direct center-ball pots, evaluate basic shot legality, rank successes using sampled execution/position errors and bounded two-shot lookahead, and output cue aim, speed, and predicted paths. A standalone 2D replay shows synthetic game runs. Calibrated perspective rendering, PNG previews, and an explicit Pi HTTP sender are implemented on this branch; see [projection integration](docs/projection.md). Perception, LLM routing, and rotation control remain stubs. The merged [Raspberry Pi display](raspi/README.md) accepts RGB frames or screen-space vectors over HTTP and drives HDMI. See the [planning guide](docs/planning.md) for setup, scoring, and limitations.

The fixtures are synthetic examples, not camera measurements or a validated shot recommendation. No model weights, camera SDK, LLM credentials, or hardware are required to start.

## Start here

Python **3.11+** is required. The scaffold itself uses only the standard library. From the repository root:

```sh
PYTHONPATH=src python3.11 -m companion.app check-fixtures
PYTHONPATH=src python3.11 -m unittest discover -s tests -v
```

Use `python` instead of `python3.11` if your active interpreter is already 3.11+. Optional editable installation:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
companion check-fixtures
```

Installation may download the setuptools build backend; the `PYTHONPATH` commands do not install anything.

## Team split

| Owner | Directory | Input | Output |
| --- | --- | --- | --- |
| Teammate 1: perception | `src/companion/pool/perception/` | RGB capture batch + known table geometry | `TableState`: ball positions/types, coverage |
| Teammate 2: planning | `src/companion/pool/planning/` | `TableState` + `GameContext` | `ShotPlan`: cue aim, cue-stick speed, called pot, optional guides |
| Teammate 3: projection | `src/companion/pool/projection/` | `ShotPlan` + calibrated `ProjectionTarget` | `ProjectionFrame`: packed RGB pixels |

**Contract version: 3.** Planning takes explicit 8-ball game context. Shot plans require a called ball/pocket and cue-stick speed in table lengths/second, separate from the unit aim direction. See the [migration notes](docs/contracts.md#migrating-to-schema-3) before using older fixtures.

All positions crossing pool stage boundaries are **table-length units: long side = 1, short side = short/long ratio**. Origin is top-left in the agreed top-down view, x right and y down. Camera pixels, projector pixels, motor angles, and table coordinates are distinct quantities.

- [Architecture and scope](docs/architecture.md): the whole system, OOP boundaries, and runtime sequence.
- [Shared contracts](docs/contracts.md): fields, units, enums, serialization, and errors.
- [Team implementation guide](docs/team-guide.md): independent commands, ownership, milestones, and acceptance checks.
- [Calibration and physical setup](docs/calibration.md): side-view geometry, scanning, projection, and open hardware questions.
- [Decisions and remaining choices](docs/decisions.md): what is fixed and what the team still needs to decide.
- [Fixture guide](fixtures/README.md): what each sample proves and what it does not.

## Independent stage commands

```sh
PYTHONPATH=src python3.11 -m companion.app perceive \
  --captures fixtures/captures/synthetic.json \
  --geometry fixtures/table_geometry.json \
  --output artifacts/table_state.json

PYTHONPATH=src python3.11 -m companion.app plan \
  --state fixtures/table_states/direct_shot.json \
  --game fixtures/game_contexts/solids.json \
  --output artifacts/shot_plan.json

PYTHONPATH=src python3.11 -m companion.app render \
  --plan fixtures/shot_plans/direct_shot.json \
  --target fixtures/projection_targets/synthetic.json \
  --output artifacts/projection.ppm
```

The planning command works after installing `.[planning]`; perception still exits with **stage not implemented**. Projection works with standard-library Python. Planning and projection use checked-in inputs, so they do not depend on an upstream implementation. Generated output belongs under ignored `artifacts/`.

## Visualized planner demo

```sh
.venv/bin/python -m pip install -e '.[planning]'
PYTHONPATH=src .venv/bin/python -m companion.pool.planning.demo
```

Open `artifacts/pool-game.html` to watch the generated game. It uses actual simulated trajectories, with play/pause and shot scrubbing. Unsupported direct-pot positions cause a clearly logged demo player switch; safeties remain future work. Add `--execution-seed 23` to demonstrate assumed aim/speed errors and misses. See [details and short endgame command](docs/planning.md).

## Raspberry Pi display

See [raspi/README.md](raspi/README.md) for the HDMI display process, sender, and installation instructions. This is the output transport: it does not compute the table-to-projector homography. Send rendered RGB frames at the calibrated output resolution; its default vector coordinates (-1..1, center origin, y up) differ from our table coordinates.
