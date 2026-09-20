"""Perception and planning, run in order on one capture.

These check the seam between the two stages rather than either stage's own
correctness: that a `TableState` this perception produces is one this planner
accepts, and that the plan it returns is geometrically consistent with the
state it was given.

That seam is where a units or axis mistake would survive both stages' own
tests - each would still be self-consistent, and only the composition would
be wrong. The arithmetic here is deliberately independent of the planner's:
it re-derives the aim from the ball positions rather than trusting the same
code that produced it.

Both stages need their optional dependencies and a recorded capture, so the
whole module skips when any is missing.
"""

import math
import unittest
from pathlib import Path

from companion.pool.contracts import BallType, PerceptionReady, PlanningReady
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


@unittest.skipIf(MISSING, f"stage dependencies not installed: {MISSING}")
@unittest.skipUnless(REAL.exists(), f"no recorded capture at {REAL}")
class PerceiveThenPlanTests(unittest.TestCase):
    """One photo, through both stages, once for the whole class."""

    @classmethod
    def setUpClass(cls):
        cls.geometry = load_geometry(FIXTURES / "table_geometry.json")
        perceived = PerceptionService().estimate(
            load_capture_batch(REAL), cls.geometry)
        assert isinstance(perceived, PerceptionReady), perceived
        cls.state = perceived.state
        planned = PooltoolPlanner(PlannerConfig()).plan(
            cls.state, load_game_context(FIXTURES / "game_contexts/solids.json"))
        cls.planned = planned

    def test_the_planner_accepts_what_perception_produced(self):
        """The handoff itself: no adapter, no reshaping between the stages."""
        self.assertIsInstance(self.planned, PlanningReady)

    def test_the_plan_names_balls_that_are_actually_on_the_table(self):
        """IDs bind the plan to its source state, so they must resolve."""
        ids = {ball.id for ball in self.state.balls}
        plan = self.planned.plan
        self.assertIn(plan.target_ball_id, ids)
        self.assertIn(plan.cue_aim.cue_ball_id, ids)

    def test_the_cue_aim_is_anchored_on_the_observed_cue_ball(self):
        """`origin` is copied from the ball's position, not recomputed."""
        plan = self.planned.plan
        cue = next(b for b in self.state.balls
                   if b.id == plan.cue_aim.cue_ball_id)
        self.assertIs(cue.type, BallType.CUE)
        self.assertAlmostEqual(plan.cue_aim.origin.x, cue.position.x, places=9)
        self.assertAlmostEqual(plan.cue_aim.origin.y, cue.position.y, places=9)

    def test_the_aim_points_at_the_ghost_ball(self):
        """Re-derive the direction from positions and compare.

        This is the check that would catch a swapped axis or a sign error
        between the stages: the planner's own direction and one computed here
        from the state's coordinates must agree.
        """
        plan = self.planned.plan
        if plan.ghost_ball is None:
            self.skipTest("aim-only plan carries no ghost ball")
        aim = plan.cue_aim
        dx = plan.ghost_ball.x - aim.origin.x
        dy = plan.ghost_ball.y - aim.origin.y
        length = math.hypot(dx, dy)
        self.assertGreater(length, 0.0)
        dot = (aim.direction.x * dx + aim.direction.y * dy) / length
        self.assertAlmostEqual(dot, 1.0, places=6)

    def test_the_target_ball_belongs_to_the_shooter_group(self):
        """The game context says solids, so the eight must not be called."""
        plan = self.planned.plan
        target = next(b for b in self.state.balls
                      if b.id == plan.target_ball_id)
        self.assertIn(target.type, (BallType.SOLID, BallType.UNKNOWN))

    def test_every_guide_stays_on_the_table(self):
        """A path leaving the cloth means a units error somewhere upstream."""
        g = self.geometry
        slack = g.ball_radius
        for guide in self.planned.plan.guides:
            for name, point in (("start", guide.segment.start),
                                ("end", guide.segment.end)):
                with self.subTest(ball=guide.ball_id, role=guide.role.value,
                                  end=name):
                    self.assertGreaterEqual(point.x, -slack)
                    self.assertLessEqual(point.x, g.length + slack)
                    self.assertGreaterEqual(point.y, -slack)
                    self.assertLessEqual(point.y, g.width + slack)

    def test_the_cue_stick_speed_is_a_usable_positive_quantity(self):
        """Table lengths per second - not a power percentage, never zero."""
        speed = self.planned.plan.cue_stick_speed
        self.assertGreater(speed, 0.0)
        self.assertTrue(math.isfinite(speed))


@unittest.skipIf(MISSING, f"stage dependencies not installed: {MISSING}")
@unittest.skipUnless(REAL.exists(), f"no recorded capture at {REAL}")
class ShotPlotTests(unittest.TestCase):
    """The demo plot renders without a display and writes a real image."""

    def test_the_demo_writes_a_png(self):
        import subprocess
        import sys
        with __import__("tempfile").TemporaryDirectory() as tmp:
            out = Path(tmp) / "shot.png"
            proc = subprocess.run(
                [sys.executable, str(ROOT / "tools/shot_demo.py"),
                 "--output", str(out)],
                cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(out.exists())
            # A PNG, and not an empty canvas.
            self.assertEqual(out.read_bytes()[:8],
                             b"\x89PNG\r\n\x1a\n")
            self.assertGreater(out.stat().st_size, 20_000)


if __name__ == "__main__":
    unittest.main()
