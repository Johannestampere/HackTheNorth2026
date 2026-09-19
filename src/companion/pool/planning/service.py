from companion.pool.contracts import (
    CoverageStatus, GameContext, InsufficientInformation, PlanningResult, TableState,
)


class PlanningService:
    """Implementation owned by teammate 2. No physics/ML/RL choice is imposed."""

    def plan(self, state: TableState, game: GameContext) -> PlanningResult:
        """Check MVP input scope; candidate search and simulation are still TODO."""
        if game.player_group is None or game.is_break or game.ball_in_hand:
            return InsufficientInformation(
                "MVP requires assigned groups after the break and a placed cue ball"
            )
        if state.coverage is not CoverageStatus.COMPLETE:
            return InsufficientInformation("MVP requires a complete table observation")
        raise NotImplementedError("Planning: implement candidate generation, evaluation, and selection")
