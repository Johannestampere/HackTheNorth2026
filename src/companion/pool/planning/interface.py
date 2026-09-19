from typing import Protocol

from companion.pool.contracts import GameContext, PlanningResult, TableState


class ShotPlanner(Protocol):
    def plan(self, state: TableState, game: GameContext) -> PlanningResult:
        """Choose a pot attempt for eventual rack success, not just pot count.

        Read positions/types/uncertainty from state and the current shooter's
        assignment from game. For the MVP, return InsufficientInformation for
        an open/unknown group, break, ball-in-hand, or incomplete ball map.
        Return NoFeasibleShot if the supported search finds no acceptable pot.
        Success must include a unit cue direction, cue-stick speed in m/s, and
        called ball/pocket. Geometry, simulation, and scoring remain internal.
        """
        ...
