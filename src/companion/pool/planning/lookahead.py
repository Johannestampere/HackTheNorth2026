"""Bounded second-shot search over sampled first-shot outcomes; no recursion."""

from random import Random

from .uncertainty import TrialStats, diverse_shortlist


def continuation(adapter, state, game, config, budget):
    """Return (second-shot value, simulations used, representative called ball).

    Re-plan after observing the settled layout. Search up to two target/pocket
    candidates, then test up to three successful strikes with five shared noise
    samples each. All work counts toward the caller's budget. If nothing works,
    continuation value is zero: safeties and opponent replies are not modeled.
    """
    if budget < 2:
        return 0.0, 0, None
    system = adapter.build(state)
    candidates = adapter.candidates(state, game, system)[:2]
    successful = []
    used = 0
    # Reserve one third for uncertainty; never reward an untested nominal pot.
    nominal_budget = min(30, budget*2//3)
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
    shortlist = diverse_shortlist(successful, 3)
    if not shortlist:
        return 0.0, used, None
    count = min(5, (budget-used)//len(shortlist))
    if count == 0:
        return 0.0, used, None
    rng = Random(config.random_seed+1)
    trials = []
    try:
        for _ in range(count):
            sampled = config.errors.layout(state, rng)
            error, factor = config.errors.strike(rng)
            trials.append((sampled, adapter.build(sampled), error, factor))
    except ValueError:
        # This future layout cannot be evaluated with our uncertainty model.
        # It earns no continuation bonus; the root observation remains usable.
        return 0.0, used, None
    best, ball_id = 0.0, None
    for _, candidate, speed, phi in shortlist:
        outcomes = []
        for sampled, system, error, factor in trials:
            shot = adapter.simulate(system, phi+error, speed*factor)
            used += 1
            outcomes.append(adapter.evaluate(shot, sampled, game, candidate))
        value = TrialStats.from_outcomes(outcomes).utility
        if value > best:
            best, ball_id = value, candidate.ball_id
    return best, used, ball_id


def compare_futures(adapter, ranked, state, game, config, remaining):
    """Compare the top three roots over up to three evenly spaced noise trials.

    Misses, fouls and terminal outcomes receive zero future value, but remain
    in the branch denominator. Their immediate cost is already in root utility.
    The continuation is discounted by 0.5; its estimate is not a win probability.
    Returns (selected root tuple, diagnostics, additional simulation count).
    """
    finalists = sorted(ranked, key=lambda item: item[0], reverse=True)[:3]
    branch_count = min(3, config.uncertainty_trials)
    per_branch = min(45, remaining//(len(finalists)*branch_count))
    best = None
    used = 0
    diagnostics = []
    for root in finalists:
        score, candidate, speed, phi, shot, outcome, stats, trials = root
        values, targets = [], []
        for i in range(branch_count):
            sampled, trial, result = trials[i*len(trials)//branch_count]
            value, target = 0.0, None
            if result.made_call and not (result.foul or result.loss or result.win):
                try:
                    following = adapter.next_state(trial, sampled, state.observation_id+'-lookahead')
                except ValueError:
                    # Out-of-frame resting centers cannot form our next observation.
                    following = None
                if following is not None:
                    value, count, target = continuation(adapter, following, game, config, per_branch)
                    used += count
            values.append(value)
            targets.append(target)
        future = sum(values)/len(values)
        combined = stats.utility + 0.5*future
        diagnostics.append(dict(ball_id=candidate.ball_id, pocket_id=candidate.pocket_id,
                                speed=speed, phi=phi, immediate=stats.utility,
                                future=future, combined=combined, branches=len(values),
                                next_targets=targets))
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
