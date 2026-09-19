import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from companion.pool.contracts import (
    BallType, CoverageStatus, GameContext, PerceptionReady, PlayerGroup,
    Point2, UnitVector2,
)
from companion.pool.contracts.serialization import (
    load_game_context, load_shot_plan, load_table_state,
    save_game_context, save_shot_plan, save_table_state,
)
from companion.pool.projection.models import ProjectionFrame, load_projection_target
from companion.sensors.models import load_capture_batch

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.state = load_table_state(FIXTURES / "table_states/direct_shot.json")
        self.plan = load_shot_plan(FIXTURES / "shot_plans/direct_shot.json")

    def test_json_round_trip_preserves_stage_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            plan_path = Path(directory) / "plan.json"
            save_table_state(state_path, self.state)
            save_shot_plan(plan_path, self.plan)
            self.assertEqual(load_table_state(state_path), self.state)
            self.assertEqual(load_shot_plan(plan_path), self.plan)
            self.assertIsInstance(load_table_state(state_path).balls[0].type, BallType)

    def test_aim_only_plan_requires_strike_and_call_but_not_paths(self):
        data = json.loads((FIXTURES / "shot_plans/aim_only.json").read_text())
        data["data"] = {key: data["data"][key] for key in (
            "observation_id", "table_id", "cue_aim", "cue_stick_speed_mps",
            "target_ball_id", "target_pocket_id",
        )}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "minimal.json"
            path.write_text(json.dumps(data))
            plan = load_shot_plan(path)
            self.assertEqual(plan.guides, ())
            self.assertIsNone(plan.ghost_ball)
            self.assertEqual(plan.cue_aim, self.plan.cue_aim)

    def test_unsupported_schema_and_enum_fail_explicitly(self):
        original = json.loads((FIXTURES / "table_states/direct_shot.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            for variant in ("version", "enum"):
                with self.subTest(variant=variant):
                    data = json.loads(json.dumps(original))
                    if variant == "version":
                        data["schema_version"] = 99
                    else:
                        data["data"]["balls"][0]["type"] = "white-ish"
                    path.write_text(json.dumps(data))
                    with self.assertRaises(ValueError):
                        load_table_state(path)

    def test_nonfinite_positions_and_nonunit_directions_are_rejected(self):
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                Point2(value, 0)
        for x, y in ((0, 0), (2, 0), (float("nan"), 0)):
            with self.assertRaises(ValueError):
                UnitVector2(x, y)

    def test_duplicate_ids_multiple_cues_and_outside_positions_are_rejected(self):
        cue = self.state.balls[0]
        invalid_ball_sets = (
            (cue, cue),
            (cue, replace(cue, id="second-cue")),
            (replace(cue, position=Point2(100, 0)),),
        )
        for balls in invalid_ball_sets:
            with self.subTest(balls=balls), self.assertRaises(ValueError):
                replace(self.state, balls=balls)

    def test_partial_coverage_cannot_be_returned_as_ready(self):
        with self.assertRaises(ValueError):
            PerceptionReady(replace(self.state, coverage=CoverageStatus.PARTIAL))

    def test_game_context_round_trip_keeps_unknown_group_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "game.json"
            for game in (GameContext(PlayerGroup.SOLIDS), GameContext(None, is_break=True),
                         GameContext(PlayerGroup.STRIPES, ball_in_hand=True)):
                with self.subTest(game=game):
                    save_game_context(path, game)
                    self.assertEqual(load_game_context(path), game)
        with self.assertRaises(ValueError):
            GameContext(PlayerGroup.SOLIDS, ball_in_hand="false")

    def test_schema_one_is_rejected_instead_of_reinterpreting_speed(self):
        document = json.loads((FIXTURES / "shot_plans/direct_shot.json").read_text())
        document["schema_version"] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old-plan.json"
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "Unsupported schema"):
                load_shot_plan(path)

    def test_strike_speed_must_be_explicit_positive_and_finite(self):
        for speed in (0, -1, float("inf"), float("nan")):
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                replace(self.plan, cue_stick_speed_mps=speed)
        document = json.loads((FIXTURES / "shot_plans/direct_shot.json").read_text())
        del document["data"]["cue_stick_speed_mps"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing-speed.json"
            path.write_text(json.dumps(document))
            with self.assertRaises(TypeError):
                load_shot_plan(path)

    def test_capture_paths_are_manifest_relative(self):
        batch = load_capture_batch(FIXTURES / "captures/synthetic.json")
        self.assertEqual(Path(batch.captures[0].rgb_path), FIXTURES / "captures/synthetic.ppm")
        self.assertTrue(Path(batch.captures[0].rgb_path).is_file())

    def test_singular_homography_is_rejected(self):
        target = load_projection_target(FIXTURES / "projection_targets/synthetic.json")
        with self.assertRaises(ValueError):
            replace(target, table_to_pixel=((1, 0, 0), (2, 0, 0), (0, 0, 1)))

    def test_frame_size_and_ppm_encoding(self):
        with self.assertRaises(ValueError):
            ProjectionFrame("obs", "table", "calib", "pose", 2, 1, b"\x00")
        frame = ProjectionFrame("obs", "table", "calib", "pose", 1, 1, b"\xff\x00\x00")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "red.ppm"
            frame.save_ppm(path)
            self.assertEqual(path.read_bytes(), b"P6\n1 1\n255\n\xff\x00\x00")


if __name__ == "__main__":
    unittest.main()
