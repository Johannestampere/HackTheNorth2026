"""Local stage runners. These do not use an LLM or control hardware."""

import argparse
from pathlib import Path

from companion.pool.contracts import PerceptionReady, PlanningReady
from companion.pool.contracts.serialization import (
    load_game_context, load_geometry, load_shot_plan, load_table_state,
    save_shot_plan, save_table_state,
)
from companion.pool.perception.service import PerceptionService
from companion.pool.pipeline import validate_plan_for_state
from companion.pool.planning.service import PlanningService
from companion.pool.projection.models import load_projection_target
from companion.pool.projection.service import ProjectionService
from companion.sensors.models import load_capture_batch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check-fixtures", help="Validate the checked-in synthetic contracts")
    check.add_argument("--root", type=Path, default=Path("fixtures"))
    perceive = commands.add_parser("perceive", help="Run teammate 1's implementation")
    perceive.add_argument("--captures", type=Path, required=True)
    perceive.add_argument("--geometry", type=Path, required=True)
    perceive.add_argument("--output", type=Path, required=True)
    plan = commands.add_parser("plan", help="Run teammate 2's implementation")
    plan.add_argument("--state", type=Path, required=True)
    plan.add_argument("--game", type=Path, required=True,
                      help="Versioned game_context JSON for the current shooter")
    plan.add_argument("--output", type=Path, required=True)
    render = commands.add_parser("render", help="Run teammate 3's implementation")
    render.add_argument("--plan", type=Path, required=True)
    render.add_argument("--target", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "check-fixtures":
            state = load_table_state(args.root / "table_states" / "direct_shot.json")
            geometry = load_geometry(args.root / "table_geometry.json")
            if state.geometry != geometry:
                raise ValueError("Sample state and geometry fixture differ")
            game = load_game_context(args.root / "game_contexts" / "solids.json")
            for name in ("aim_only", "direct_shot"):
                validate_plan_for_state(
                    load_shot_plan(args.root / "shot_plans" / f"{name}.json"), state, game
                )
            validate_plan_for_state(
                load_shot_plan(args.root / "shot_plans" / "eight_ball_finish.json"),
                load_table_state(args.root / "table_states" / "eight_ball_finish.json"), game,
            )
            target = load_projection_target(args.root / "projection_targets" / "synthetic.json")
            if target.table_id != state.geometry.table_id:
                raise ValueError("Fixture projector target uses a different coordinate frame")
            batch = load_capture_batch(args.root / "captures" / "synthetic.json")
            for capture in batch.captures:
                if not Path(capture.rgb_path).is_file():
                    raise ValueError(f"Missing fixture image: {capture.rgb_path}")
            print("Fixtures valid. All samples are synthetic; no vision, physics, or calibration was verified.")
        elif args.command == "perceive":
            result = PerceptionService().estimate(
                load_capture_batch(args.captures), load_geometry(args.geometry)
            )
            if not isinstance(result, PerceptionReady):
                print(f"{type(result).__name__}: {result.reason}")
                return 1
            save_table_state(args.output, result.state)
        elif args.command == "plan":
            game = load_game_context(args.game)
            state = load_table_state(args.state)
            result = PlanningService().plan(state, game)
            if not isinstance(result, PlanningReady):
                print(f"{type(result).__name__}: {result.reason}")
                return 1
            validate_plan_for_state(result.plan, state, game)
            save_shot_plan(args.output, result.plan)
        elif args.command == "render":
            plan = load_shot_plan(args.plan)
            target = load_projection_target(args.target)
            if plan.table_id != target.table_id:
                raise ValueError("Plan and projection target use different table frames")
            ProjectionService().render(plan, target).save_ppm(args.output)
    except NotImplementedError as error:
        parser.exit(2, f"Stage not implemented: {error}\n")
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.exit(2, f"Invalid input: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
