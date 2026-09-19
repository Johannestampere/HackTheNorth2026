"""Run synthetic post-break eight-ball self-play and write an offline 2D replay."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
from time import perf_counter

from companion.pool.contracts import Ball, BallType, GameContext, PlanningReady, PlayerGroup, Point2
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


def run(state, config, max_shots):
    """Re-observe exact simulated positions after each shot; no perception noise.

    A player with no supported direct pot is skipped for demonstration purposes.
    This is explicitly a demo convention, not a legal safety shot or WPA pass.
    Two such stops terminate the replay without inventing a winner.
    """
    planner = PooltoolPlanner(config)
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
        trajectories = planner.adapter.trajectories(chosen.simulation)
        plan = chosen.plan
        summary = (f'{player.value}: {plan.target_ball_id} → {plan.target_pocket_id}; '
                   f'{plan.cue_stick_speed:.2f} table lengths/s; {chosen.outcome.reason}')
        print(f'{len(shots)+1}. {summary} ({chosen.simulations} simulations, {perf_counter()-started:.2f}s)', flush=True)
        shots.append(dict(
            player=player.value, summary=summary,
            balls=[dict(id=b.id, type=b.type.value) for b in state.balls],
            trajectories=trajectories,
            duration=max(points[-1][0] for points in trajectories.values()),
            aim=asdict(plan.cue_aim), ghost=asdict(plan.ghost_ball) if plan.ghost_ball else None,
            guides=[dict(ball_id=g.ball_id, role=g.role.value, start=asdict(g.segment.start),
                         end=asdict(g.segment.end)) for g in plan.guides],
            pocketed=sorted(chosen.outcome.pocketed), simulations=chosen.simulations,
        ))
        if chosen.outcome.win or chosen.outcome.loss:
            winner = player if chosen.outcome.win else (PlayerGroup.STRIPES if player is PlayerGroup.SOLIDS else PlayerGroup.SOLIDS)
            ending = f'{winner.value.title()} win the synthetic rack'
            break
        try:
            state = planner.adapter.next_state(chosen.simulation, state, f'demo-shot-{turn+1}')
        except ValueError as error:
            ending = f'Stopped: {error}'
            break
        if chosen.outcome.foul or not chosen.outcome.made_call:
            player = PlayerGroup.STRIPES if player is PlayerGroup.SOLIDS else PlayerGroup.SOLIDS
    return dict(shots=shots, log=log, ending=ending, width=state.geometry.width,
                radius=state.geometry.ball_radius, pockets=[asdict(p) for p in state.geometry.pockets],
                table_length_m=config.table_length_m)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, help='Use an existing TableState instead of a spread rack')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--max-shots', type=int, default=32)
    parser.add_argument('--table-length-m', type=float, default=2.0)
    parser.add_argument('--output', type=Path, default=Path('artifacts/pool-game.html'))
    args = parser.parse_args(argv)
    if args.max_shots < 1:
        parser.error('--max-shots must be positive')
    fixture = Path(__file__).resolve().parents[4] / 'fixtures/table_states/direct_shot.json'
    state = load_table_state(args.state or fixture)
    if args.state is None:
        state = spread_rack(state, args.seed)
    data = run(state, PlannerConfig(table_length_m=args.table_length_m), args.max_shots)
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
