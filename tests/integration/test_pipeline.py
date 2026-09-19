import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import create_autospec

from companion.pool.contracts import (
    BallType, GameContext, InsufficientInformation, NeedsMoreViews, NoFeasibleShot,
    PerceptionReady, PlanningReady, PlayerGroup, Point2, UnusableCapture,
)
from companion.pool.contracts.serialization import load_shot_plan, load_table_state
from companion.pool.perception.interface import TablePerception
from companion.pool.pipeline import PoolPipeline, PreparedProjection, validate_plan_for_state
from companion.pool.planning.interface import ShotPlanner
from companion.pool.planning.service import PlanningService
from companion.pool.projection.interface import ShotRenderer
from companion.pool.projection.models import ProjectionFrame, load_projection_target
from companion.sensors.models import load_capture_batch

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.state = load_table_state(FIXTURES / "table_states/direct_shot.json")
        self.plan = load_shot_plan(FIXTURES / "shot_plans/direct_shot.json")
        self.target = load_projection_target(FIXTURES / "projection_targets/synthetic.json")
        self.captures = load_capture_batch(FIXTURES / "captures/synthetic.json")
        self.game = GameContext(PlayerGroup.SOLIDS)
        self.perception = create_autospec(TablePerception, instance=True)
        self.planner = create_autospec(ShotPlanner, instance=True)
        self.renderer = create_autospec(ShotRenderer, instance=True)
        self.perception.estimate.return_value = PerceptionReady(self.state)
        self.planner.plan.return_value = PlanningReady(self.plan)
        self.frame = ProjectionFrame(
            self.state.observation_id, self.target.table_id, self.target.calibration_id,
            self.target.pose_id, self.target.width_px, self.target.height_px,
            bytes(self.target.width_px * self.target.height_px * 3),
        )
        self.renderer.render.return_value = self.frame
        self.pipeline = PoolPipeline(self.perception, self.planner, self.renderer)

    def prepare(self):
        return self.pipeline.prepare(self.captures, self.state.geometry, self.game, self.target)

    def test_composes_interfaces_without_real_hardware_or_algorithms(self):
        result = self.prepare()
        self.assertEqual(result, PreparedProjection(self.state, self.plan, self.frame))
        self.perception.estimate.assert_called_once_with(self.captures, self.state.geometry)
        self.planner.plan.assert_called_once_with(self.state, self.game)
        self.renderer.render.assert_called_once_with(self.plan, self.target)

    def test_incomplete_capture_does_not_call_downstream_stages(self):
        for outcome in (NeedsMoreViews("far rail occluded"), UnusableCapture("balls moving")):
            with self.subTest(outcome=outcome):
                self.perception.estimate.return_value = outcome
                self.assertIs(self.prepare(), outcome)
                self.planner.plan.assert_not_called()
                self.renderer.render.assert_not_called()

    def test_no_plan_does_not_render(self):
        for outcome in (InsufficientInformation("no cue ball"), NoFeasibleShot("all paths blocked")):
            with self.subTest(outcome=outcome):
                self.planner.plan.return_value = outcome
                self.assertIs(self.prepare(), outcome)
                self.renderer.render.assert_not_called()

    def test_stale_observation_is_rejected_before_rendering(self):
        self.planner.plan.return_value = PlanningReady(replace(self.plan, observation_id="old"))
        with self.assertRaisesRegex(ValueError, "observation/table"):
            self.prepare()
        self.renderer.render.assert_not_called()

    def test_bad_cue_origin_or_unknown_targets_are_rejected(self):
        invalid_plans = (
            replace(self.plan, cue_aim=replace(self.plan.cue_aim, origin=Point2(0, 0))),
            replace(self.plan, target_ball_id="not-observed"),
            replace(self.plan, target_pocket_id="not-a-pocket"),
        )
        for plan in invalid_plans:
            with self.subTest(plan=plan):
                self.planner.plan.return_value = PlanningReady(plan)
                with self.assertRaises(ValueError):
                    self.prepare()
                self.renderer.render.assert_not_called()

    def test_projector_target_must_match_full_table_geometry(self):
        self.target = replace(self.target, geometry=replace(self.target.geometry, length_m=3))
        with self.assertRaises(ValueError):
            self.prepare()
        self.perception.estimate.assert_not_called()

    def test_premature_eight_and_opponent_ball_are_rejected(self):
        for target in ("eight", "stripe-1"):
            with self.subTest(target=target):
                self.planner.plan.return_value = PlanningReady(replace(self.plan, target_ball_id=target))
                with self.assertRaisesRegex(ValueError, "uncleared group"):
                    self.prepare()
                self.renderer.render.assert_not_called()

    def test_cleared_group_allows_eight_ball_finish(self):
        state = load_table_state(FIXTURES / "table_states/eight_ball_finish.json")
        plan = load_shot_plan(FIXTURES / "shot_plans/eight_ball_finish.json")
        validate_plan_for_state(plan, state, self.game)
        unknown = replace(state.balls[1], type=BallType.UNKNOWN)
        uncertain_state = replace(state, balls=(state.balls[0], unknown, state.balls[2]))
        with self.assertRaisesRegex(ValueError, "Unknown ball types"):
            validate_plan_for_state(plan, uncertain_state, self.game)

    def test_guide_ball_identity_must_match_role_and_observation(self):
        for ball_id in ("missing", "solid-1"):
            with self.subTest(ball_id=ball_id):
                guide = replace(self.plan.guides[0], ball_id=ball_id)
                self.planner.plan.return_value = PlanningReady(replace(self.plan, guides=(guide,)))
                with self.assertRaisesRegex(ValueError, "Guide"):
                    self.prepare()
                self.renderer.render.assert_not_called()

    def test_unsupported_context_is_not_silently_treated_as_normal_play(self):
        for game in (GameContext(None), GameContext(PlayerGroup.SOLIDS, is_break=True),
                     GameContext(PlayerGroup.SOLIDS, ball_in_hand=True)):
            with self.subTest(game=game):
                self.assertIsInstance(PlanningService().plan(self.state, game), InsufficientInformation)
                with self.assertRaisesRegex(ValueError, "ordinary placed-ball"):
                    validate_plan_for_state(self.plan, self.state, game)

    def test_frame_pose_and_calibration_must_match_target(self):
        for field in ("pose_id", "calibration_id", "observation_id"):
            with self.subTest(field=field):
                self.renderer.render.return_value = replace(self.frame, **{field: "wrong"})
                with self.assertRaisesRegex(ValueError, "provenance"):
                    self.prepare()


if __name__ == "__main__":
    unittest.main()
