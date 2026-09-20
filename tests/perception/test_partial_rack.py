"""A table part-way through a game, not a fresh rack.

Balls leave a pool table as it is played, so the ordinary case is a partial
rack - and it is the case most likely to break quietly. The detector's cue
and eight are chosen by comparison against the balls present, the planner
refuses to act without both, and the plot must not imply a type is absent
when it simply was not drawn.

These build partial states from a known-good full one rather than from a
photograph, so a failure is unambiguously in the handling of a short count
and not in what the camera happened to see that day.
"""

import unittest
from dataclasses import replace
from pathlib import Path

from companion.pool.contracts import (BallType, InsufficientInformation,
                                      PerceptionReady, PlanningReady)
from companion.pool.contracts.serialization import (load_game_context,
                                                    load_geometry)
from companion.sensors.models import load_capture_batch

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "fixtures"
REAL = ROOT / "data/local/captures/real_table.json"

try:
    from companion.pool.perception.service import PerceptionService
    from companion.pool.planning.service import PlannerConfig, PooltoolPlanner
    MISSING = None
except ImportError as error:
    MISSING = str(error)


def _subset(state, **counts):
    """A state holding only the requested number of each ball type."""
    kept = []
    for kind, n in counts.items():
        matching = [b for b in state.balls if b.type is kind]
        kept.extend(matching[:n])
    return replace(state, balls=tuple(kept))


@unittest.skipIf(MISSING, f"stage dependencies not installed: {MISSING}")
@unittest.skipUnless(REAL.exists(), f"no recorded capture at {REAL}")
class PartialRackTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.geometry = load_geometry(FIXTURES / "table_geometry.json")
        perceived = PerceptionService().estimate(
            load_capture_batch(REAL), cls.geometry)
        assert isinstance(perceived, PerceptionReady), perceived
        cls.full = perceived.state
        cls.game = load_game_context(FIXTURES / "game_contexts/solids.json")

    def test_a_six_ball_table_still_plans(self):
        """The ordinary mid-game case: a few balls, a cue and the eight."""
        partial = _subset(self.full, **{BallType.CUE: 1, BallType.EIGHT: 1,
                                        BallType.SOLID: 2, BallType.STRIPE: 2})
        self.assertEqual(len(partial.balls), 6)
        result = PooltoolPlanner(PlannerConfig()).plan(partial, self.game)
        self.assertIsInstance(result, PlanningReady)
        self.assertIn(result.plan.target_ball_id,
                      {b.id for b in partial.balls})

    def test_a_table_with_no_cue_ball_is_refused_not_guessed(self):
        """Without a cue ball there is no shot, and inventing one is worse."""
        partial = _subset(self.full, **{BallType.EIGHT: 1, BallType.SOLID: 3})
        result = PooltoolPlanner(PlannerConfig()).plan(partial, self.game)
        self.assertIsInstance(result, InsufficientInformation)

    def test_a_table_with_no_eight_ball_is_refused(self):
        """The eight decides legality even when it is not the target."""
        partial = _subset(self.full, **{BallType.CUE: 1, BallType.SOLID: 3})
        result = PooltoolPlanner(PlannerConfig()).plan(partial, self.game)
        self.assertIsInstance(result, InsufficientInformation)

    def test_the_last_solid_is_a_legal_target(self):
        """Down to one object ball, the shooter's group is still shootable."""
        partial = _subset(self.full, **{BallType.CUE: 1, BallType.EIGHT: 1,
                                        BallType.SOLID: 1})
        result = PooltoolPlanner(PlannerConfig()).plan(partial, self.game)
        if isinstance(result, PlanningReady):
            target = next(b for b in partial.balls
                          if b.id == result.plan.target_ball_id)
            self.assertIs(target.type, BallType.SOLID)
        else:
            # A geometrically impossible layout is a legitimate answer; an
            # unhandled state is not.
            self.assertIsInstance(result, (InsufficientInformation,))


@unittest.skipIf(MISSING, f"stage dependencies not installed: {MISSING}")
@unittest.skipUnless(REAL.exists(), f"no recorded capture at {REAL}")
class PartialRackPlotTests(unittest.TestCase):
    """The plots must describe what is there, not a full set."""

    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, str(ROOT / "tools"))
        geometry = load_geometry(FIXTURES / "table_geometry.json")
        perceived = PerceptionService().estimate(
            load_capture_batch(REAL), geometry)
        cls.state = _subset(perceived.state,
                            **{BallType.CUE: 1, BallType.EIGHT: 1,
                               BallType.SOLID: 2})

    def test_the_legend_names_only_the_types_present(self):
        """A type listed but absent reads as a drawing choice, not a fact."""
        import matplotlib
        matplotlib.use("Agg")
        from shot_plot import _type_swatches
        labels = [h.get_label() for h in _type_swatches(self.state)]
        self.assertTrue(any(l.startswith("cue") for l in labels))
        self.assertTrue(any(l.startswith("solid") for l in labels))
        self.assertFalse(any(l.startswith("stripe") for l in labels))

    def test_the_grid_plot_writes_a_png(self):
        import matplotlib
        matplotlib.use("Agg")
        from shot_plot import plot_state
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "grid.png"
            plot_state(self.state, out)
            self.assertTrue(out.exists())
            self.assertEqual(out.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
