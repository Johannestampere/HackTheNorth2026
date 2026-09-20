"""Budgeted own-turn search with a narrow beam and sampled execution outcomes."""

from random import Random
from dataclasses import dataclass, field

from .uncertainty import TrialStats, diverse_shortlist


@dataclass
class FutureValue:
    """Private search result; path is one illustrative branch, not a committed plan.

    reached counts simulated shot levels from this node, including alternatives.
    exhausted signals budget truncation somewhere in this subtree, not elapsed time.
    """

    value: float = 0.0
    used: int = 0
    target: str | None = None
    path: list[str] = field(default_factory=list)
    reached: int = 0
    exhausted: bool = False


def continuation(adapter, state, game, config, budget, depth=1, ply=1):
    """Search up to depth more shots, spending no more than budget simulations.

    Each node evaluates at most 30 nominal strikes and 15 uncertainty trials.
    Keep two promising actions and two evenly spaced noisy outcomes per action
    for deeper search. Miss/foul/loss/win branches stop. Re-plan only after a
    legal nonterminal pot. No opponent moves, safeties or ball-in-hand are modeled.
    """
    if depth <= 0:
        return FutureValue()
    if budget < 2:
        return FutureValue(exhausted=True)
    system = adapter.build(state)
    candidates = adapter.candidates(state, game, system)[:2]
    successful = []
    used = 0
    nominal_budget = min(30, budget*2//3)
    nominal_count = len(candidates)*len(config.speeds)*len(config.angle_offsets)
    for candidate in candidates:
        for speed in config.speeds:
            for offset in config.angle_offsets:
                if used >= nominal_budget:
                    break
                phi = candidate.phi+offset
                shot = adapter.simulate(system, phi, speed)
                used += 1
                outcome = adapter.evaluate(shot, state, game, candidate)
                if outcome.made_call and not outcome.foul and not outcome.loss:
                    successful.append(((outcome.win, -candidate.difficulty, -abs(offset), -speed),
                                       candidate, speed, phi))
    truncated = nominal_count > nominal_budget
    shortlist = diverse_shortlist(successful, 3)
    if not shortlist:
        return FutureValue(used=used, reached=int(used>0), exhausted=truncated)
    count = min(5, (budget-used)//len(shortlist))
    if count == 0:
        return FutureValue(used=used, reached=int(used>0), exhausted=True)
    rng = Random(config.random_seed+ply)
    trials = []
    try:
        for _ in range(count):
            sampled = config.errors.layout(state, rng)
            error, factor = config.errors.strike(rng)
            trials.append((sampled, adapter.build(sampled), error, factor))
    except ValueError:
        return FutureValue(used=used, reached=1, exhausted=truncated)
    evaluated = []
    for _, candidate, speed, phi in shortlist:
        outcomes, branches = [], []
        for sampled, system, error, factor in trials:
            shot = adapter.simulate(system, phi+error, speed*factor)
            used += 1
            outcome = adapter.evaluate(shot, sampled, game, candidate)
            outcomes.append(outcome)
            branches.append((sampled, shot, outcome))
        stats = TrialStats.from_outcomes(outcomes)
        evaluated.append((stats.utility, candidate, branches))
    # Every expanded sibling receives an equal quota; unused quota is not
    # silently donated to an earlier branch and cannot exceed the parent budget.
    beam = sorted(evaluated, key=lambda item: item[0], reverse=True)[:2]
    branch_count = min(2, count)
    quota = (budget-used)//(len(beam)*branch_count) if depth>1 else 0
    best = FutureValue(used=used, reached=1, exhausted=truncated or count<5)
    total_used, reached, exhausted = used, 1, best.exhausted
    for immediate, candidate, branches in beam:
        futures = []
        for i in range(branch_count):
            sample, shot, outcome = branches[i*len(branches)//branch_count]
            future = FutureValue()
            if depth>1 and outcome.made_call and not (outcome.foul or outcome.loss or outcome.win):
                try:
                    following = adapter.next_state(shot, sample, state.observation_id+'-future')
                except ValueError:
                    following = None
                if following is not None:
                    future = continuation(adapter, following, game, config, quota, depth-1, ply+1)
            total_used += future.used
            reached = max(reached, 1+future.reached)
            exhausted |= future.exhausted
            futures.append(future)
        value = immediate+0.5*sum(f.value for f in futures)/branch_count
        if best.target is None or value > best.value:
            example = max(futures, key=lambda f: f.value)
            best = FutureValue(max(0.0,value), target=candidate.ball_id,
                               path=[candidate.ball_id]+example.path)
    best.used, best.reached, best.exhausted = total_used, reached, exhausted
    return best


def compare_futures(adapter, ranked, state, game, config, remaining):
    """Compare the top three roots over up to three evenly spaced noise trials.

    Misses, fouls and terminal outcomes receive zero future value, but remain
    in the branch denominator. Their immediate cost is already in root utility.
    The continuation is discounted by 0.5; its estimate is not a win probability.
    Returns (selected root tuple, diagnostics, additional simulation count).
    """
    finalists = sorted(ranked, key=lambda item: item[0], reverse=True)[:3]
    branch_count = min(3, config.uncertainty_trials)
    per_branch = remaining//(len(finalists)*branch_count)
    if config.search_depth == 2:
        per_branch = min(45, per_branch)
    best = None
    used = 0
    diagnostics = []
    for root in finalists:
        score, candidate, speed, phi, shot, outcome, stats, trials = root
        values, targets, paths = [], [], []
        reached, exhausted = 1, False
        for i in range(branch_count):
            sampled, trial, result = trials[i*len(trials)//branch_count]
            value, target, path = 0.0, None, []
            if result.made_call and not (result.foul or result.loss or result.win):
                try:
                    following = adapter.next_state(trial, sampled, state.observation_id+'-lookahead')
                except ValueError:
                    # Out-of-frame resting centers cannot form our next observation.
                    following = None
                if following is not None:
                    future = continuation(adapter, following, game, config, per_branch,
                                          config.search_depth-1)
                    value, target, path = future.value, future.target, future.path
                    used += future.used
                    reached = max(reached, 1+future.reached)
                    exhausted |= future.exhausted
            values.append(value)
            targets.append(target)
            paths.append([candidate.ball_id]+path)
        future = sum(values)/len(values)
        combined = stats.utility + 0.5*future
        diagnostics.append(dict(ball_id=candidate.ball_id, pocket_id=candidate.pocket_id,
                                speed=speed, phi=phi, immediate=stats.utility,
                                future=future, combined=combined, branches=len(values),
                                next_targets=targets, paths=paths, requested_depth=config.search_depth,
                                reached_depth=reached, budget_limited=exhausted))
        rank = (combined, *score)
        if best is None or rank > best[0]:
            best = (rank, root)
    chosen = best[1]
    for diagnostic in diagnostics:
        diagnostic['selected'] = (diagnostic['ball_id'] == chosen[1].ball_id
                                  and diagnostic['pocket_id'] == chosen[1].pocket_id
                                  and diagnostic['speed'] == chosen[2]
                                  and diagnostic['phi'] == chosen[3])
    return chosen, diagnostics, used
