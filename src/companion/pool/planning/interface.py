from typing import Protocol

from companion.pool.contracts import GameContext, PlanningResult, TableState


class ShotPlanner(Protocol):
    def plan(self, state: TableState, game: GameContext) -> PlanningResult:
        """Recommend a shot or return an explicit information/feasibility outcome."""
        ...
