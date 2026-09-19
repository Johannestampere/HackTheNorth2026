"""Behavioral tests for reproducible noise, fair search, and risk preferences."""

from dataclasses import replace
import importlib.util
from math import hypot
from pathlib import Path
from random import Random
from types import SimpleNamespace
import unittest

from companion.pool.contracts import GameContext, PlayerGroup
from companion.pool.contracts.serialization import load_table_state
from companion.pool.planning.candidates import Candidate
from companion.pool.planning.service import PlannerConfig, PooltoolPlanner
from companion.pool.planning.uncertainty import ErrorModel, TrialStats, diverse_shortlist

ROOT = Path(__file__).resolve().parents[2]


class UncertaintyTests(unittest.TestCase):
    def setUp(self):
        self.state = load_table_state(ROOT/'fixtures/table_states/direct_shot.json')

    def test_zero_errors_preserve_positions_and_strike(self):
        model = ErrorModel(0, 0, 0)
        self.assertEqual(model.layout(self.state, Random(1)), self.state)
        self.assertEqual(model.strike(Random(1)), (0, 1))

    def test_reported_uncertainty_overrides_fallback_without_mutation(self):
        state = replace(self.state, balls=tuple(replace(b, position_uncertainty=0) for b in self.state.balls))
        self.assertEqual(ErrorModel(position_radius=.1).layout(state, Random(1)), state)
        sample = ErrorModel(position_radius=.002).layout(self.state, Random(9))
        for original, perturbed in zip(self.state.balls, sample.balls):
            self.assertLessEqual(hypot(original.position.x-perturbed.position.x,
                                       original.position.y-perturbed.position.y), .002)
        self.assertEqual(sample, ErrorModel(position_radius=.002).layout(self.state, Random(9)))

    def test_risky_successes_lose_to_reliable_pots(self):
        reliable = TrialStats(20, 0, 16, 4, 0, 0)
        risky = TrialStats(20, 0, 18, 0, 2, 0)
        self.assertGreater(reliable.utility, risky.utility)
        catastrophic = TrialStats(20, 0, 19, 0, 0, 1)
        self.assertGreater(reliable.utility, catastrophic.utility)
        winning = TrialStats(20, 19, 0, 1, 0, 0)
        self.assertGreater(winning.utility, reliable.utility)

    def test_outcomes_are_exclusive(self):
        def outcome(**kwargs):
            return SimpleNamespace(**dict(dict(loss=False, foul=False, win=False, made_call=False), **kwargs))
        stats = TrialStats.from_outcomes([
            outcome(loss=True, foul=True), outcome(win=True, made_call=True),
            outcome(foul=True, made_call=True), outcome(made_call=True), outcome(),
        ])
        self.assertEqual(stats, TrialStats(5, 1, 1, 1, 1, 1))

    def test_shortlist_preserves_alternative_pots(self):
        point = self.state.balls[0].position
        a, b = Candidate('a','p',0,point,1), Candidate('b','p',0,point,2)
        trials = [((10-i,), a, i) for i in range(5)] + [((0,), b, 0)]
        selected = diverse_shortlist(trials, 3)
        self.assertEqual([x[1].ball_id for x in selected], ['a','b','a'])

    def test_invalid_error_and_trial_settings_fail_early(self):
        for kwargs in ({'aim_std_degrees':-1}, {'speed_fraction':1}, {'position_radius':float('nan')}):
            with self.assertRaises(ValueError):
                ErrorModel(**kwargs)
        with self.assertRaises(ValueError):
            PlannerConfig(uncertainty_trials=0)

    @unittest.skipUnless(importlib.util.find_spec('pooltool'), 'Install .[planning]')
    def test_real_planner_repeats_identical_trials_and_nominal_plan(self):
        config = PlannerConfig(uncertainty_trials=8, shortlist_size=4)
        game = GameContext(PlayerGroup.SOLIDS)
        first, second = PooltoolPlanner(config), PooltoolPlanner(config)
        self.assertEqual(first.plan(self.state, game), second.plan(self.state, game))
        self.assertEqual(first.last_selection.stats, second.last_selection.stats)
        self.assertEqual(first.last_selection.stats.trials, 8)
        self.assertGreater(first.last_selection.simulations, 8)

    @unittest.skipUnless(importlib.util.find_spec('pooltool'), 'Install .[planning]')
    def test_real_trials_detect_aiming_error_that_nominal_simulation_hides(self):
        game = GameContext(PlayerGroup.SOLIDS)
        exact = PooltoolPlanner(PlannerConfig(shortlist_size=2, uncertainty_trials=24,
                                            errors=ErrorModel(0, 0, 0)))
        noisy = PooltoolPlanner(PlannerConfig(shortlist_size=2, uncertainty_trials=24,
                                            errors=ErrorModel(5, 0, 0)))
        exact.plan(self.state, game)
        noisy.plan(self.state, game)
        self.assertEqual(exact.last_selection.stats.success_rate, 1)
        self.assertLess(noisy.last_selection.stats.success_rate, 1)
        self.assertTrue(noisy.last_selection.outcome.made_call)
