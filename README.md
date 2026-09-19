# AI Projector Companion

A robot with an RGB camera, a projector, and a 360° rotating base that brings an AI interface into the physical world. It observes a scene, selects a useful tool, and projects guidance onto the relevant surface. Perception estimates depth from the RGB image using a model; there is no depth camera. Audio is planned later.

The first tool is **pool**. The robot sits beside the table, some distance away and roughly 1–2 m high; it is **not mounted overhead**. A general multimodal LLM will inspect the initial image and choose `pool` or `generalization`. Pool may request multiple views, identify balls in table coordinates, plan a shot, and prepare a perspective-correct projector image.

## Current status

This repository contains **the skeleton and contracts**, not working perception, physics, projection, LLM routing, or hardware control. The three stage services deliberately raise `NotImplementedError`. The runnable parts are contract validation, JSON adapters, local command entry points, and single-batch pipeline wiring.

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
| Teammate 2: planning | `src/companion/pool/planning/` | `TableState` + `GameContext` | `ShotPlan`: cue aim and optional guidance |
| Teammate 3: projection | `src/companion/pool/projection/` | `ShotPlan` + calibrated `ProjectionTarget` | `ProjectionFrame`: packed RGB pixels |

All positions crossing pool stage boundaries are **meters in the same table coordinate frame**. Camera pixels, projector pixels, motor angles, and table coordinates are distinct quantities.

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
  --output artifacts/shot_plan.json

PYTHONPATH=src python3.11 -m companion.app render \
  --plan fixtures/shot_plans/direct_shot.json \
  --target fixtures/projection_targets/synthetic.json \
  --output artifacts/projection.ppm
```

These commands currently exit with an explicit **stage not implemented** message. Each teammate implements their own `service.py` to make their command work. Planning and projection use checked-in inputs, so they do not depend on an upstream implementation. Generated output belongs under ignored `artifacts/`.
