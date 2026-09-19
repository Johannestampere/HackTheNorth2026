"""Physics integration tests use real Pooltool; contract-only installs skip them."""

from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

from companion.pool.contracts import (
    Ball, BallType, GameContext, InsufficientInformation, NoFeasibleShot,
    PlanningReady, PlayerGroup, Point2,
)
from companion.pool.contracts.serialization import load_table_state
from companion.pool.pipeline import validate_plan_for_state
from companion.pool.planning.candidates import Candidate, path_clear
from companion.pool.planning.service import PooltoolPlanner

ROOT = Path(__file__).resolve().parents[2]


class GeometryTests(unittest.TestCase):
    def test_finite_path_and_ball_radius(self):
        state = load_table_state(ROOT/'fixtures/table_states/direct_shot.json')
        self.assertFalse(path_clear(Point2(.2,.3), Point2(.8,.3), state, {'cue'}))
        # solid-1 lies beyond the endpoint and should not block this segment.
        self.assertTrue(path_clear(Point2(.2,.3), Point2(.4,.3), state, {'cue'}))


@unittest.skipUnless(importlib.util.find_spec('pooltool'), 'Install .[planning] for physics tests')
class PlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = load_table_state(ROOT/'fixtures/table_states/direct_shot.json')
        cls.game = GameContext(PlayerGroup.SOLIDS)
        cls.planner = PooltoolPlanner()
        cls.result = cls.planner.plan(cls.state, cls.game)

    def test_direct_pot_uses_real_physics_and_preserves_contract(self):
        self.assertIsInstance(self.result, PlanningReady)
        validate_plan_for_state(self.result.plan, self.state, self.game)
        chosen = self.planner.last_selection
        self.assertTrue(chosen.outcome.made_call)
        self.assertFalse(chosen.outcome.foul)
        self.assertIn('solid-1', chosen.outcome.pocketed)
        self.assertTrue(self.result.plan.guides)
        # Replaying the public vector and speed reproduces the selected pot.
        from math import atan2, degrees
        direction = self.result.plan.cue_aim.direction
        shot = self.planner.adapter.simulate(self.planner.adapter.build(self.state),
                    degrees(atan2(direction.x, direction.y)), self.result.plan.cue_stick_speed)
        self.assertEqual(self.planner.adapter.evaluate(shot, self.state, self.game, chosen.candidate), chosen.outcome)

    def test_follow_up_finishes_eight(self):
        following = self.planner.adapter.next_state(self.planner.last_selection.simulation, self.state, 'next')
        service = PooltoolPlanner()
        result = service.plan(following, self.game)
        self.assertIsInstance(result, PlanningReady)
        self.assertEqual(result.plan.target_ball_id, 'eight')
        self.assertTrue(service.last_selection.outcome.win)

    def test_units_and_axes_round_trip(self):
        engine = self.planner.adapter.build(self.state)
        for observed in self.state.balls:
            self.assertEqual(self.planner.adapter.point(engine.balls[observed.id].xyz), observed.position)
        self.assertEqual(engine.table.l, 2)
        self.assertEqual(engine.table.w, 1)
        self.assertEqual(engine.cue.b, 0)

    def test_missing_unknown_and_overlap_are_rejected(self):
        for balls in (
            tuple(b for b in self.state.balls if b.type is not BallType.CUE),
            tuple(replace(b, type=BallType.UNKNOWN) if b.id == 'solid-1' else b for b in self.state.balls),
            tuple(replace(b, position=Point2(.3,.15)) if b.id == 'solid-1' else b for b in self.state.balls),
        ):
            with self.subTest(balls=balls):
                self.assertIsInstance(PooltoolPlanner().plan(replace(self.state, balls=balls), self.game), InsufficientInformation)

    def test_scratch_early_eight_wrong_contact_and_no_rail(self):
        def event(kind, *ids):
            return SimpleNamespace(event_type=kind, agents=[SimpleNamespace(id=i) for i in ids], time=1)
        candidate = Candidate('solid-1', 'xmax-ymax', 0, Point2(.5,.3), 1)
        def assess(events):
            shot = SimpleNamespace(cue=SimpleNamespace(cue_ball_id='cue'), events=events)
            return self.planner.adapter.evaluate(shot, self.state, self.game, candidate)
        first = event('ball_ball', 'cue', 'solid-1')
        scratch = assess([first, event('ball_pocket', 'cue', 'rt')])
        self.assertTrue(scratch.foul)
        self.assertFalse(scratch.win)
        early = assess([first, event('ball_pocket', 'eight', 'rt')])
        self.assertTrue(early.loss)
        self.assertTrue(assess([event('ball_ball', 'cue', 'stripe-1')]).foul)
        self.assertTrue(assess([first]).foul)
        self.assertFalse(assess([first, event('ball_linear_cushion', 'solid-1', 'rail')]).foul)

    def test_wrong_called_pocket_does_not_count(self):
        chosen = self.planner.last_selection
        wrong = replace(chosen.candidate, pocket_id='x0-y0')
        outcome = self.planner.adapter.evaluate(chosen.simulation, self.state, self.game, wrong)
        self.assertFalse(outcome.made_call)

    def test_surrounded_target_returns_no_feasible_shot(self):
        from math import cos, sin, pi
        solid = next(b for b in self.state.balls if b.type is BallType.SOLID)
        blockers = tuple(Ball(f'blocker-{i}', Point2(
            solid.position.x+.04*cos(i*pi/3), solid.position.y+.04*sin(i*pi/3)),
            BallType.STRIPE) for i in range(6))
        state = replace(self.state, balls=tuple(b for b in self.state.balls
                        if b.type is not BallType.STRIPE) + blockers)
        self.assertIsInstance(PooltoolPlanner().plan(state, self.game), NoFeasibleShot)

    def test_unsupported_pocket_layout_is_not_silently_remapped(self):
        pockets = self.state.geometry.pockets
        geometry = replace(self.state.geometry, pockets=(
            replace(pockets[0], position=Point2(.1, 0)), *pockets[1:]))
        result = PooltoolPlanner().plan(replace(self.state, geometry=geometry), self.game)
        self.assertIsInstance(result, InsufficientInformation)

    def test_replay_samples_are_finite_and_end_at_settled_state(self):
        from math import isfinite
        shot = self.planner.last_selection.simulation
        samples = self.planner.adapter.trajectories(shot)
        for key, points in samples.items():
            self.assertTrue(all(isfinite(v) for row in points for v in row))
            self.assertEqual(points[0][0], 0)
            self.assertAlmostEqual(points[-1][0], shot.events[-1].time)
        self.assertEqual(samples['solid-1'][-1][3], 0)
        self.assertEqual(samples['cue'][-1][3], 1)
