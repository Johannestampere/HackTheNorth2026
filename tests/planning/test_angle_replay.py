"""Every angle replay starts from the same observation, including failed strikes."""
import importlib.util
from pathlib import Path
import unittest

from companion.pool.contracts.serialization import load_table_state
from companion.pool.planning.demo import compare_angles
from companion.pool.planning.service import PlannerConfig


@unittest.skipUnless(importlib.util.find_spec('pooltool'), 'Install .[planning]')
class AngleReplayTests(unittest.TestCase):
    def test_complete_grid_resets_layout_and_includes_failures(self):
        state = load_table_state(Path(__file__).resolve().parents[2]/'fixtures/table_states/direct_shot.json')
        config = PlannerConfig(lookahead=False, uncertainty_trials=2, shortlist_size=2)
        data = compare_angles(state, config)
        attempts = data['shots']
        groups = {}
        for attempt in attempts:
            groups.setdefault((attempt['target_ball'], attempt['target_pocket']), set()).add(
                (attempt['speed'], attempt['offset_degrees']))
            for ball in state.balls:
                initial = attempt['trajectories'][ball.id][0]
                self.assertAlmostEqual(initial[1], ball.position.x)
                self.assertAlmostEqual(initial[2], ball.position.y)
        expected = {(speed, angle) for speed in config.speeds for angle in config.angle_offsets}
        self.assertTrue(groups)
        self.assertTrue(all(grid == expected for grid in groups.values()))
        self.assertEqual(len(attempts), len(groups)*len(expected))
        self.assertEqual(sum(a['selected'] for a in attempts), 1)
        self.assertTrue(any(not a['outcome']['made_call'] for a in attempts))
