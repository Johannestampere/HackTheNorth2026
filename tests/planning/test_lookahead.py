"""Lookahead must affect choices, stop at terminal states, and count its work."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from companion.pool.contracts import GameContext, PlayerGroup
from companion.pool.contracts.serialization import load_table_state
from companion.pool.planning.lookahead import compare_futures
from companion.pool.planning.service import PlannerConfig, PooltoolPlanner
from companion.pool.planning.uncertainty import TrialStats


class LookaheadTests(unittest.TestCase):
    def root(self, name, utility_stats, result):
        candidate = SimpleNamespace(ball_id=name, pocket_id='p')
        return ((utility_stats.utility,), candidate, .5, 0, None, result,
                utility_stats, [(name, None, result)]*3)

    def test_actual_continuation_can_reverse_immediate_ranking(self):
        legal = SimpleNamespace(made_call=True, foul=False, loss=False, win=False)
        roots = [self.root('easy-now', TrialStats(10,0,10,0,0,0), legal),
                 self.root('sets-up-eight', TrialStats(10,0,9,1,0,0), legal)]
        adapter = SimpleNamespace(next_state=lambda shot, state, name: state)
        def future(adapter, state, game, config, budget):
            return (3 if state == 'sets-up-eight' else 1), 10, 'eight'
        with patch('companion.pool.planning.lookahead.continuation', side_effect=future):
            selected, diagnostics, used = compare_futures(adapter, roots,
                SimpleNamespace(observation_id='test'), None, PlannerConfig(), 472)
        self.assertEqual(selected[1].ball_id, 'sets-up-eight')
        self.assertEqual(used, 60)
        self.assertGreater(diagnostics[1]['combined'], diagnostics[0]['combined'])

    def test_misses_fouls_and_wins_do_not_continue(self):
        for flags in ({'made_call':False}, {'foul':True}, {'loss':True}, {'win':True}):
            result = SimpleNamespace(**dict(dict(made_call=True, foul=False, loss=False, win=False), **flags))
            with patch('companion.pool.planning.lookahead.continuation') as future:
                _, diagnostic, used = compare_futures(None,
                    [self.root('target', TrialStats(3,0,3,0,0,0), result)],
                    SimpleNamespace(observation_id='test'), None, PlannerConfig(), 472)
                future.assert_not_called()
                self.assertEqual(used, 0)
                self.assertEqual(diagnostic[0]['future'], 0)

    def test_no_budget_keeps_immediate_ranking(self):
        legal = SimpleNamespace(made_call=True, foul=False, loss=False, win=False)
        roots = [self.root('a', TrialStats(3,0,3,0,0,0), legal)]
        adapter = SimpleNamespace(next_state=lambda *args: None)
        _, diagnostic, used = compare_futures(adapter, roots,
            SimpleNamespace(observation_id='test'), None, PlannerConfig(), 0)
        self.assertEqual(used, 0)
        self.assertEqual(diagnostic[0]['future'], 0)

    @unittest.skipUnless(importlib.util.find_spec('pooltool'), 'Install .[planning]')
    def test_real_physics_simulation_count_matches_budget_accounting(self):
        root = Path(__file__).resolve().parents[2]
        state = load_table_state(root/'fixtures/table_states/direct_shot.json')
        planner = PooltoolPlanner()
        with patch.object(planner.adapter, 'simulate', wraps=planner.adapter.simulate) as simulate:
            planner.plan(state, GameContext(PlayerGroup.SOLIDS))
            chosen = planner.last_selection
            self.assertEqual(chosen.simulations, simulate.call_count)
            self.assertLessEqual(chosen.simulations, 1000)
            self.assertGreater(chosen.lookahead_simulations, 0)
            self.assertTrue(any('eight' in d['next_targets'] for d in chosen.lookahead_diagnostics))

    def test_configuration_cannot_exceed_budget_before_lookahead(self):
        with self.assertRaises(ValueError):
            PlannerConfig(simulation_budget=100)
