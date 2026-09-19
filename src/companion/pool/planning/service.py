"""Direct-pot search using Pooltool, with no changes to the shared stage contracts."""

from dataclasses import dataclass
from math import cos, hypot, isfinite, radians, sin
from typing import Any

from companion.pool.contracts import (
    BallType, CoverageStatus, CueAim, GameContext, InsufficientInformation,
    NoFeasibleShot, PlanningReady, PlanningResult, ShotPlan, TableState, UnitVector2,
)
from .candidates import Candidate


@dataclass(frozen=True)
class PlannerConfig:
    """Search budget and explicit physical scale; speeds use table lengths/second.

    The nominal search tests every listed speed and angle offset for the best
    geometric candidates. This first milestone is deterministic, not a learned
    policy or a calibrated win-probability estimator.
    """

    table_length_m: float = 2.0
    speeds: tuple[float, ...] = (0.35, 0.5, 0.7, 0.95, 1.25)
    angle_offsets: tuple[float, ...] = (0.0, -0.75, 0.75)
    max_candidates: int = 16

    def __post_init__(self):
        if not isfinite(self.table_length_m) or self.table_length_m <= 0:
            raise ValueError("Physical table length must be positive")
        if not self.speeds or any(not isfinite(v) or v <= 0 for v in self.speeds):
            raise ValueError("Search requires positive finite speeds")
        if not self.angle_offsets or any(not isfinite(a) for a in self.angle_offsets):
            raise ValueError("Search requires finite angle offsets")
        if self.max_candidates < 1:
            raise ValueError("Search requires at least one candidate")


@dataclass
class SelectedShot:
    """Private diagnostics for the demo; simulator state never crosses stage APIs."""

    plan: ShotPlan
    simulation: Any
    outcome: Any
    candidate: Candidate
    simulations: int


class PooltoolPlanner:
    """Implement ShotPlanner using Pooltool simulation and direct-pot search.

    Return no shot when the supported search finds no nominally legal pot.
    """

    def __init__(self, config: PlannerConfig | None = None):
        self.config = config or PlannerConfig()
        self._adapter = None
        self.last_selection: SelectedShot | None = None

    @property
    def adapter(self):
        """Load optional physics only when needed, so other stages remain lightweight."""
        if self._adapter is None:
            from .physics import PooltoolAdapter
            self._adapter = PooltoolAdapter(self.config.table_length_m)
        return self._adapter

    def plan(self, state: TableState, game: GameContext) -> PlanningResult:
        """Search center-ball direct pots and emit the best supported nominal shot.

        Rank legal wins first, then successful called pots with more follow-up
        direct options; prefer easier geometry and gentler strikes on ties.
        This is one-shot positional scoring, not multi-turn game-tree search.
        """
        self.last_selection = None
        if game.player_group is None or game.is_break or game.ball_in_hand:
            return InsufficientInformation(
                "MVP requires assigned groups after the break and a placed cue ball")
        if state.coverage is not CoverageStatus.COMPLETE:
            return InsufficientInformation("MVP requires a complete table observation")
        cue = next((b for b in state.balls if b.type is BallType.CUE), None)
        if cue is None or not any(b.type is BallType.EIGHT for b in state.balls):
            return InsufficientInformation("A live rack needs an observed cue ball and eight ball")
        if any(b.type is BallType.UNKNOWN for b in state.balls):
            return InsufficientInformation("Classify unknown balls before choosing a legal target")
        radius = state.geometry.ball_radius
        for i, ball in enumerate(state.balls):
            if not (radius <= ball.position.x <= 1-radius and
                    radius <= ball.position.y <= state.geometry.width-radius):
                return InsufficientInformation("MVP cannot initialize balls inside pocket jaws or rails")
            for other in state.balls[i+1:]:
                if hypot(ball.position.x-other.position.x, ball.position.y-other.position.y) < 2*radius-1e-8:
                    return InsufficientInformation("Observed ball centers overlap; refine perception")
        adapter = self.adapter
        try:
            system = adapter.build(state)
        except ValueError as error:
            return InsufficientInformation(str(error))
        candidates = adapter.candidates(state, game, system)[:self.config.max_candidates]
        successful = []
        simulations = 0
        for candidate in candidates:
            for speed in self.config.speeds:
                for offset in self.config.angle_offsets:
                    phi = candidate.phi + offset
                    simulation = adapter.simulate(system, phi, speed)
                    simulations += 1
                    outcome = adapter.evaluate(simulation, state, game, candidate)
                    if outcome.foul or outcome.loss or not outcome.made_call:
                        continue
                    # Cheap nominal score shortlists before positional analysis.
                    rank = (outcome.win, -candidate.difficulty, -abs(offset), -speed)
                    successful.append((rank, candidate, speed, phi, simulation, outcome))
        if not successful:
            return NoFeasibleShot(f"No legal direct pot found in {simulations} simulated strikes")
        # Position matters: compare the best nominal results, with a bounded cost.
        best = None
        for _, candidate, speed, phi, simulation, outcome in sorted(successful, key=lambda x: x[0], reverse=True)[:12]:
            try:
                following = adapter.next_state(simulation, state, state.observation_id + '-predicted')
                options = [] if outcome.win else adapter.candidates(following, game)
            except ValueError:
                options = []
            score = (outcome.win, bool(options), -min((c.difficulty for c in options), default=10),
                     -candidate.difficulty, -speed)
            if best is None or score > best[0]:
                best = (score, candidate, speed, phi, simulation, outcome)
        _, candidate, speed, phi, simulation, outcome = best
        # Swap engine axes back. Both direction components remain unitless.
        direction = UnitVector2(sin(radians(phi)), cos(radians(phi)))
        first_contact = next(e for e in simulation.events
                             if str(e.event_type) == 'ball_ball'
                             and cue.id in {a.id for a in e.agents})
        contact_center = adapter.point(first_contact.get_ball(cue.id, initial=True).state.rvw[0])
        plan = ShotPlan(state.observation_id, state.geometry.table_id,
                        CueAim(cue.id, cue.position, direction), speed,
                        candidate.ball_id, candidate.pocket_id,
                        adapter.guides(simulation), contact_center)
        self.last_selection = SelectedShot(plan, simulation, outcome, candidate, simulations)
        return PlanningReady(plan)
