from companion.pool.contracts import GameContext, PlanningResult, TableState


class PlanningService:
    """Implementation owned by teammate 2. No physics/ML/RL choice is imposed."""

    def plan(self, state: TableState, game: GameContext) -> PlanningResult:
        raise NotImplementedError("Planning: implement candidate generation, evaluation, and selection")
