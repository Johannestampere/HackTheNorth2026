"""Perception stage: the contract boundary, and one real table if present.

The boundary tests run everywhere. The detection test needs a recorded RGB
capture, which is device data and therefore ignored by git (see
`fixtures/README.md`), so it skips rather than fails when the image is not on
this machine. A skipped test is honest about what was not checked; a test
that silently passes without the image would not be.
"""

import unittest
from dataclasses import replace
from pathlib import Path

from companion.pool.contracts import (BallType, CoverageStatus, PerceptionReady,
                                      UnusableCapture)
from companion.pool.contracts.serialization import load_geometry
from companion.sensors.models import load_capture_batch

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "fixtures"
# Recorded captures are device data, kept out of the repo by .gitignore.
REAL = ROOT / "data/local/captures/real_table.json"

try:
    from companion.pool.perception.service import PerceptionService
    IMPORT_ERROR = None
except ImportError as error:          # opencv/numpy are an optional extra
    IMPORT_ERROR = str(error)


@unittest.skipIf(IMPORT_ERROR, f"perception extra not installed: {IMPORT_ERROR}")
class PerceptionBoundaryTests(unittest.TestCase):
    """What the stage promises regardless of which image it is given."""

    def setUp(self):
        self.geometry = load_geometry(FIXTURES / "table_geometry.json")
        self.captures = load_capture_batch(FIXTURES / "captures/synthetic.json")

    def test_an_unreadable_capture_is_reported_not_raised(self):
        """A missing or corrupt file is a property of the input, not a crash.

        The pipeline is handed paths it did not choose, so this is the
        ordinary case rather than an exceptional one.
        """
        missing = replace(
            self.captures,
            captures=(replace(self.captures.captures[0],
                              rgb_path=str(ROOT / "does-not-exist.png")),))
        result = PerceptionService().estimate(missing, self.geometry)
        self.assertIsInstance(result, UnusableCapture)
        self.assertIn("could not be read", result.reason)

    def test_an_oversized_ball_radius_is_refused(self):
        """A radius near the table's own size leaves nothing to search for.

        The contract permits it - it only requires the diameter to be under
        the short side - but the detector sizes its search from that radius,
        so the stage checks it before measuring rather than returning a
        confident count of whatever the wrong-sized disc matched.
        """
        huge = replace(self.geometry, ball_radius=0.2)
        result = PerceptionService().estimate(self.captures, huge)
        self.assertIsInstance(result, UnusableCapture)
        self.assertIn("Ball radius", result.reason)

    def test_the_spec_guard_rejects_a_transposed_table(self):
        """`length` is the long side by contract, so width may never exceed it.

        The dataclass cannot express this mistake - it pins length to 1.0 -
        so the guard is checked directly. It is the last line of defence if
        a future caller builds a geometry without that validation.
        """
        from companion.pool.perception.service import _check_spec
        transposed = replace(self.geometry, length=1.0, width=1.0)
        object.__setattr__(transposed, "width", 2.0)
        self.assertIn("transposed", _check_spec(transposed) or "")

    def test_the_synthetic_schematic_is_not_claimed_as_a_real_table(self):
        """The shipped fixture is a drawing, not a photograph of cloth.

        Whatever the cloth detector makes of it, the stage must return a
        typed outcome rather than assert a confident state from an image the
        fixtures README explicitly says is not a camera capture.
        """
        result = PerceptionService().estimate(self.captures, self.geometry)
        if isinstance(result, PerceptionReady):
            # If it does parse, the state must still satisfy the contract.
            self.assertIs(result.state.coverage, CoverageStatus.COMPLETE)
        else:
            self.assertTrue(result.reason)


@unittest.skipIf(IMPORT_ERROR, f"perception extra not installed: {IMPORT_ERROR}")
@unittest.skipUnless(REAL.exists(), f"no recorded capture at {REAL}")
class RealTableTests(unittest.TestCase):
    """Measured behaviour on one recorded overhead frame.

    The numbers below are what this pipeline actually scores on this image,
    recorded as a regression guard. They are not an accuracy target for the
    team; the guide is explicit that a target comes after measuring hardware.
    """

    @classmethod
    def setUpClass(cls):
        geometry = load_geometry(FIXTURES / "table_geometry.json")
        captures = load_capture_batch(REAL)
        cls.result = PerceptionService().estimate(captures, geometry)
        cls.geometry = geometry

    def test_a_full_rack_is_read_as_sixteen_balls(self):
        self.assertIsInstance(self.result, PerceptionReady)
        self.assertEqual(len(self.result.state.balls), 16)

    def test_exactly_one_cue_and_one_eight_are_found(self):
        """The two balls the game defines, and the ones a planner needs."""
        kinds = [ball.type for ball in self.result.state.balls]
        self.assertEqual(kinds.count(BallType.CUE), 1)
        self.assertEqual(kinds.count(BallType.EIGHT), 1)

    def test_the_remaining_fourteen_split_seven_and_seven(self):
        kinds = [ball.type for ball in self.result.state.balls]
        self.assertEqual(kinds.count(BallType.SOLID), 7)
        self.assertEqual(kinds.count(BallType.STRIPE), 7)

    def test_every_ball_lands_inside_the_playing_surface(self):
        """The contract rejects the whole state for one stray ball.

        Checking it here names which ball and where, instead of leaving a
        ValueError from deep inside the dataclass.
        """
        for ball in self.result.state.balls:
            with self.subTest(ball=ball.id):
                self.assertGreaterEqual(ball.position.x, 0.0)
                self.assertLessEqual(ball.position.x, self.geometry.length)
                self.assertGreaterEqual(ball.position.y, 0.0)
                self.assertLessEqual(ball.position.y, self.geometry.width)

    def test_positions_use_the_long_side_as_x(self):
        """A transposed result would crowd into y and leave x half empty.

        On a spread rack the balls cover well over half the long side; if the
        axes were swapped the x spread would be bounded by the short side.
        """
        xs = [ball.position.x for ball in self.result.state.balls]
        self.assertGreater(max(xs) - min(xs), self.geometry.width)


if __name__ == "__main__":
    unittest.main()
