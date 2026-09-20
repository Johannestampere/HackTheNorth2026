"""Run synthetic post-break eight-ball self-play and write an offline 2D replay."""

import argparse
from dataclasses import asdict, replace
import json
from math import atan2, degrees, sin, cos, radians
from pathlib import Path
import random
from time import perf_counter

from companion.pool.contracts import Ball, BallType, GameContext, InsufficientInformation, PlanningReady, PlayerGroup, Point2
from companion.pool.contracts.serialization import load_table_state
from companion.pool.pipeline import validate_plan_for_state
from .service import PlannerConfig, PooltoolPlanner


def spread_rack(template, seed):
    """Generate a reproducible spread layout, not a simulated break or camera input."""
    rng = random.Random(seed)
    balls = [Ball('cue', Point2(.25, .25), BallType.CUE)]
    kinds = [BallType.SOLID]*7 + [BallType.STRIPE]*7 + [BallType.EIGHT]
    for index, kind in enumerate(kinds, 1):
        for _ in range(10000):
            p = Point2(rng.uniform(.06, .94), rng.uniform(.06, template.geometry.width-.06))
            if all((p.x-b.position.x)**2+(p.y-b.position.y)**2 > (.065)**2 for b in balls):
                ball_id = str(index if index <= 7 else index+1) if kind is not BallType.EIGHT else 'eight'
                balls.append(Ball(ball_id, p, kind))
                break
        else:
            raise ValueError('Could not generate nonoverlapping demo layout')
    return replace(template, observation_id=f'synthetic-spread-{seed}', balls=tuple(balls))


def run(state, config, max_shots, execution_seed=None):
    """Re-observe exact positions; optionally perturb execution aim and speed.

    Planner trials also sample position error, but this demo uses exact observations.

    A player with no supported direct pot is skipped for demonstration purposes.
    This is explicitly a demo convention, not a legal safety shot or WPA pass.
    Two such stops terminate the replay without inventing a winner.
    """
    planner = PooltoolPlanner(config)
    execution_rng = random.Random(execution_seed) if execution_seed is not None else None
    player = PlayerGroup.SOLIDS
    shots, log = [], []
    skipped = 0
    ending = 'Shot budget reached; game unfinished'
    for turn in range(max_shots):
        started = perf_counter()
        game = GameContext(player)
        result = planner.plan(state, game)
        if not isinstance(result, PlanningReady):
            message = f'{player.value}: {result.reason}'
            print(message, flush=True)
            if isinstance(result, InsufficientInformation):
                ending = 'Stopped: ' + message
                break
            log.append(message + ' — demo switches player (no safety implemented)')
            skipped += 1
            if skipped == 2:
                ending = 'Stopped: neither player has a supported direct pot. No winner.'
                break
            player = PlayerGroup.STRIPES if player is PlayerGroup.SOLIDS else PlayerGroup.SOLIDS
            continue
        skipped = 0
        validate_plan_for_state(result.plan, state, game)
        chosen = planner.last_selection
        plan = chosen.plan
        simulation, outcome = chosen.simulation, chosen.outcome
        angle_error, speed_factor = 0.0, 1.0
        if execution_rng is not None:
            angle_error, speed_factor = config.errors.strike(execution_rng)
            phi = degrees(atan2(plan.cue_aim.direction.x, plan.cue_aim.direction.y))
            simulation = planner.adapter.simulate(planner.adapter.build(state),
                phi+angle_error, plan.cue_stick_speed*speed_factor)
            outcome = planner.adapter.evaluate(simulation, state, game, chosen.candidate)
        trajectories = planner.adapter.trajectories(simulation)
        summary = (f'{player.value}: {plan.target_ball_id} → {plan.target_pocket_id}; '
                   f'{plan.cue_stick_speed:.2f} table lengths/s; {outcome.reason}')
        print(f'{len(shots)+1}. {summary} ({chosen.simulations} simulations, {perf_counter()-started:.2f}s)', flush=True)
        shots.append(dict(
            player=player.value, summary=summary,
            balls=[dict(id=b.id, type=b.type.value) for b in state.balls],
            trajectories=trajectories,
            duration=max(points[-1][0] for points in trajectories.values()),
            aim=asdict(plan.cue_aim), ghost=asdict(plan.ghost_ball) if plan.ghost_ball else None,
            guides=[dict(ball_id=g.ball_id, role=g.role.value, start=asdict(g.segment.start),
                         end=asdict(g.segment.end)) for g in plan.guides],
            pocketed=sorted(outcome.pocketed), simulations=chosen.simulations,
            lookahead=chosen.lookahead_diagnostics, lookahead_simulations=chosen.lookahead_simulations,
            trials=asdict(chosen.stats), success_rate=chosen.stats.success_rate,
            utility=chosen.stats.utility, aim_error_degrees=angle_error,
            speed_factor=speed_factor,
        ))
        if outcome.win or outcome.loss:
            winner = player if outcome.win else (PlayerGroup.STRIPES if player is PlayerGroup.SOLIDS else PlayerGroup.SOLIDS)
            ending = f'{winner.value.title()} win the synthetic rack'
            break
        if outcome.foul:
            ending = 'Stopped after foul: opponent needs ball-in-hand placement (not implemented)'
            break
        try:
            state = planner.adapter.next_state(simulation, state, f'demo-shot-{turn+1}')
        except ValueError as error:
            ending = f'Stopped: {error}'
            break
        if not outcome.made_call:
            player = PlayerGroup.STRIPES if player is PlayerGroup.SOLIDS else PlayerGroup.SOLIDS
    return dict(shots=shots, log=log, ending=ending, width=state.geometry.width,
                radius=state.geometry.ball_radius, pockets=[asdict(p) for p in state.geometry.pockets],
                table_length_m=config.table_length_m, errors=asdict(config.errors),
                execution_seed=execution_seed, noisy_execution=execution_rng is not None)


def compare_angles(state, config):
    """Replay every nominal first-shot strike from the same unchanged layout.

    These reproduce the planner's candidate/speed/offset loops, including rejected
    outcomes. They exclude random uncertainty trials and second-shot lookahead.
    Display angles use our frame: 0 degrees right, +90 degrees down the table.
    """
    planner = PooltoolPlanner(config)
    game = GameContext(PlayerGroup.SOLIDS)
    result = planner.plan(state, game)
    if isinstance(result, InsufficientInformation):
        raise ValueError(result.reason)
    selected = planner.last_selection
    adapter = planner.adapter
    system = adapter.build(state)
    candidates = adapter.candidates(state, game, system)[:config.max_candidates]
    cue = next(b for b in state.balls if b.type is BallType.CUE)
    attempts = []
    for candidate in candidates:
        for speed in config.speeds:
            for offset in config.angle_offsets:
                phi = candidate.phi+offset
                shot = adapter.simulate(system, phi, speed)
                outcome = adapter.evaluate(shot, state, game, candidate)
                direction = dict(x=sin(radians(phi)), y=cos(radians(phi)))
                angle = degrees(atan2(direction['y'], direction['x'])) % 360
                chosen = bool(selected and candidate.ball_id == selected.plan.target_ball_id
                    and candidate.pocket_id == selected.plan.target_pocket_id
                    and speed == selected.plan.cue_stick_speed
                    and abs(direction['x']-selected.plan.cue_aim.direction.x) < 1e-9
                    and abs(direction['y']-selected.plan.cue_aim.direction.y) < 1e-9)
                trajectories = adapter.trajectories(shot)
                label = (f'{candidate.ball_id} → {candidate.pocket_id} | '
                         f'{angle:.2f}° | offset {offset:+.2f}° in engine frame | '
                         f'speed {speed:.2f} | {outcome.reason}' + (' | SELECTED' if chosen else ''))
                attempts.append(dict(
                    player=game.player_group.value, summary=label, label=label,
                    balls=[dict(id=b.id, type=b.type.value) for b in state.balls],
                    trajectories=trajectories,
                    duration=max(points[-1][0] for points in trajectories.values()),
                    aim=dict(cue_ball_id=cue.id, origin=asdict(cue.position), direction=direction),
                    ghost=asdict(candidate.ghost),
                    guides=[dict(ball_id=g.ball_id, role=g.role.value,
                                 start=asdict(g.segment.start), end=asdict(g.segment.end))
                            for g in adapter.guides(shot)],
                    target_ball=candidate.ball_id, target_pocket=candidate.pocket_id,
                    angle_degrees=angle, offset_degrees=offset, speed=speed,
                    selected=chosen, outcome=asdict(outcome) | {'pocketed':sorted(outcome.pocketed)},
                ))
    return dict(shots=attempts, comparison=True, log=[],
                ending=f'{len(attempts)} independent attempts from the SAME starting positions',
                width=state.geometry.width, radius=state.geometry.ball_radius,
                pockets=[asdict(p) for p in state.geometry.pockets],
                table_length_m=config.table_length_m, noisy_execution=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compare-angles', action='store_true', help='Replay all nominal first-shot angles/speeds, including misses')
    parser.add_argument('--state', type=Path, help='Use an existing TableState instead of a spread rack')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--execution-seed', type=int, help='Enable reproducible aim/speed errors, independent of planner trials')
    parser.add_argument('--max-shots', type=int, default=32)
    parser.add_argument('--table-length-m', type=float, default=2.0)
    parser.add_argument('--output', type=Path, default=Path('artifacts/pool-game.html'))
    parser.add_argument('--search-depth', type=int, choices=range(1,5), default=4)
    parser.add_argument('--simulation-budget', type=int, default=4000)
    args = parser.parse_args(argv)
    if args.compare_angles and args.execution_seed is not None:
        parser.error('--compare-angles uses nominal attempts; omit --execution-seed')
    if args.max_shots < 1:
        parser.error('--max-shots must be positive')
    fixture = Path(__file__).resolve().parents[4] / 'fixtures/table_states/direct_shot.json'
    state = load_table_state(args.state or fixture)
    if args.state is None and not args.compare_angles:
        state = spread_rack(state, args.seed)
    config = PlannerConfig(table_length_m=args.table_length_m, search_depth=args.search_depth,
                           simulation_budget=args.simulation_budget)
    data = (compare_angles(state, config) if args.compare_angles else
            run(state, config, args.max_shots, args.execution_seed))
    template = Path(__file__).with_name('replay.html').read_text()
    payload = json.dumps(data, separators=(',', ':')).replace('<', '\\u003c')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template.replace('__REPLAY_DATA__', payload))
    args.output.with_suffix('.json').write_text(json.dumps(data, indent=2))
    print(data['ending'])
    print(f'Replay: {args.output.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
