"""Direct-pot search using Pooltool, with no changes to the shared stage contracts."""

from dataclasses import dataclass, field
from math import cos, hypot, isfinite, radians, sin
from typing import Any
from random import Random

from companion.pool.contracts import (
    BallType, CoverageStatus, CueAim, GameContext, InsufficientInformation,
    NoFeasibleShot, PlanningReady, PlanningResult, ShotPlan, TableState, UnitVector2,
)
from .candidates import Candidate
from .uncertainty import ErrorModel, TrialStats, diverse_shortlist


@dataclass(frozen=True)
class PlannerConfig:
    """Search budget and explicit physical scale; speeds use table lengths/second.

    The nominal search tests every listed speed and angle offset for the best
    geometric candidates. Seeded uncertainty trials rank nominal successes by sampled utility.
    This is not a learned policy or a calibrated win-probability estimator.
    """

    table_length_m: float = 2.0
    speeds: tuple[float, ...] = (0.35, 0.5, 0.7, 0.95, 1.25)
    angle_offsets: tuple[float, ...] = (0.0, -0.75, 0.75)
    max_candidates: int = 16
    shortlist_size: int = 12
    uncertainty_trials: int = 24
    random_seed: int = 17
    simulation_budget: int = 1000
    lookahead: bool = True
    errors: ErrorModel = field(default_factory=ErrorModel)

    def __post_init__(self):
        if not isfinite(self.table_length_m) or self.table_length_m <= 0:
            raise ValueError("Physical table length must be positive")
        if not self.speeds or any(not isfinite(v) or v <= 0 for v in self.speeds):
            raise ValueError("Search requires positive finite speeds")
        if not self.angle_offsets or any(not isfinite(a) for a in self.angle_offsets):
            raise ValueError("Search requires finite angle offsets")
        if any(type(v) is not int or v < 1 for v in (
            self.max_candidates, self.shortlist_size, self.uncertainty_trials, self.simulation_budget
        )):
            raise ValueError("Search requires at least one candidate")
        nominal_limit = self.max_candidates*len(self.speeds)*len(self.angle_offsets)
        if nominal_limit+self.shortlist_size*self.uncertainty_trials > self.simulation_budget:
            raise ValueError("Simulation budget must cover configured nominal and uncertainty search")
        if type(self.lookahead) is not bool:
            raise ValueError("lookahead must be a bool")


@dataclass
class SelectedShot:
    """Private diagnostics for the demo; simulator state never crosses stage APIs."""

    plan: ShotPlan
    simulation: Any
    outcome: Any
    candidate: Candidate
    simulations: int
    stats: TrialStats
    lookahead_diagnostics: list[dict] = field(default_factory=list)
    lookahead_simulations: int = 0


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

        Rank nominally legal pots by sampled win/pot/miss/foul/loss utility.
        Compare the top three strikes using simulated second shots across
        sampled first-shot outcomes, within the configured simulation budget.
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
        # Common random draws make comparisons repeatable and reduce sampling
        # differences between candidates. Never optimize the aim for a noisy layout.
        rng = Random(self.config.random_seed)
        try:
            trials = []
            for _ in range(self.config.uncertainty_trials):
                sampled = self.config.errors.layout(state, rng)
                angle_error, speed_factor = self.config.errors.strike(rng)
                trials.append((sampled, adapter.build(sampled), angle_error, speed_factor))
        except ValueError as error:
            return InsufficientInformation(str(error))
        ranked = []
        for _, candidate, speed, phi, simulation, outcome in diverse_shortlist(
            successful, self.config.shortlist_size
        ):
            outcomes = []
            root_trials = []
            for sampled, trial_system, angle_error, speed_factor in trials:
                trial = adapter.simulate(trial_system, phi+angle_error, speed*speed_factor)
                simulations += 1
                result = adapter.evaluate(trial, sampled, game, candidate)
                outcomes.append(result)
                root_trials.append((sampled, trial, result))
            stats = TrialStats.from_outcomes(outcomes)
            try:
                following = adapter.next_state(simulation, state, state.observation_id + '-predicted')
                options = [] if outcome.win else adapter.candidates(following, game)
            except ValueError:
                options = []
            score = (stats.utility, stats.success_rate, -stats.losses, -stats.fouls,
                     outcome.win, bool(options),
                     -min((c.difficulty for c in options), default=10),
                     -candidate.difficulty, -speed)
            ranked.append((score, candidate, speed, phi, simulation, outcome, stats, root_trials))
        diagnostics, future_simulations = [], 0
        if self.config.lookahead:
            from .lookahead import compare_futures
            best, diagnostics, future_simulations = compare_futures(
                adapter, ranked, state, game, self.config,
                self.config.simulation_budget-simulations)
            simulations += future_simulations
        else:
            best = max(ranked, key=lambda item: item[0])
        _, candidate, speed, phi, simulation, outcome, stats, _ = best
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
        self.last_selection = SelectedShot(plan, simulation, outcome, candidate, simulations, stats,
                                           diagnostics, future_simulations)
        return PlanningReady(plan)
