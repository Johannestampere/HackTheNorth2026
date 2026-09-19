"""Pooltool 0.6 adapter. Shared contracts never expose simulator objects or SI units."""

from dataclasses import dataclass, replace
from math import cos, hypot, isfinite, radians
from typing import Any

from companion.pool.contracts import (
    BallType, GameContext, GuideRole, GuideSegment, Point2, Segment2, TableState,
)
from .candidates import Candidate, path_clear, target_ids


@dataclass(frozen=True)
class Outcome:
    """Private evaluation of an ordinary called shot, derived from simulated events."""

    foul: bool
    loss: bool
    win: bool
    made_call: bool
    pocketed: frozenset[str]
    reason: str


class PooltoolAdapter:
    """Convert normalized top-left coordinates to a standard six-pocket table.

    Physical long-side length defaults to an explicit 2 m assumption. This is
    calibration, not a measurement inferred from an RGB image. Mapping is
    (engine x, engine y) = (our y, our x) * length_m. Pocket depths/jaw shapes
    and material properties use Pooltool defaults; they require real calibration.
    """

    def __init__(self, length_m: float = 2.0):
        if not isfinite(length_m) or length_m <= 0:
            raise ValueError("Physical table length must be positive and finite")
        try:
            import pooltool as pt
        except ImportError as error:
            raise RuntimeError("Install planning dependencies: pip install -e '.[planning]'") from error
        from pooltool.physics import PhysicsEngine
        from pooltool.physics.resolve.resolver import default_resolver
        self.pt = pt
        self.length_m = length_m
        # Do not let a developer's personal resolver YAML change search results.
        self.engine = PhysicsEngine(resolver=default_resolver())

    def point(self, xy: Any) -> Point2:
        """Convert simulator meters back to the shared coordinate frame."""
        return Point2(float(xy[1]) / self.length_m, float(xy[0]) / self.length_m)

    def pocket_map(self, state: TableState) -> dict[str, str]:
        """Map standard mouth locations to engine IDs; reject unsupported geometry."""
        width = state.geometry.width
        slots = {(0, 0): "lb", (0.5, 0): "lc", (1, 0): "lt",
                 (0, width): "rb", (0.5, width): "rc", (1, width): "rt"}
        result = {}
        for pocket in state.geometry.pockets:
            slot = min(slots, key=lambda xy: hypot(pocket.position.x-xy[0], pocket.position.y-xy[1]))
            if hypot(pocket.position.x-slot[0], pocket.position.y-slot[1]) > 1e-5:
                raise ValueError("MVP expects pockets at the four corners and long-rail midpoints")
            if slots[slot] in result.values():
                raise ValueError("Two pockets occupy the same table location")
            result[pocket.id] = slots[slot]
        return result

    def build(self, state: TableState):
        """Create stationary balls and a table, preserving observed ball IDs."""
        from pooltool.objects import PocketTableSpecs
        mapping = self.pocket_map(state)
        corner = [p.mouth_width for p in state.geometry.pockets if mapping[p.id] not in {"lc", "rc"}]
        side = [p.mouth_width for p in state.geometry.pockets if mapping[p.id] in {"lc", "rc"}]
        if max(corner)-min(corner) > 1e-5 or max(side)-min(side) > 1e-5:
            raise ValueError("MVP requires equal corner widths and equal side widths")
        scale = self.length_m
        radius = state.geometry.ball_radius * scale
        table = self.pt.Table.from_table_specs(PocketTableSpecs(
            l=scale, w=state.geometry.width*scale,
            corner_pocket_width=corner[0]*scale, side_pocket_width=side[0]*scale,
            cushion_height=1.28*radius,
        ))
        cue = next(b for b in state.balls if b.type is BallType.CUE)
        balls = {b.id: self.pt.Ball.create(b.id, xy=(b.position.y*scale, b.position.x*scale),
                                         R=radius) for b in state.balls}
        return self.pt.System(table=table, balls=balls,
                              cue=self.pt.Cue(cue_ball_id=cue.id, a=0, b=0, theta=0))

    def candidates(self, state: TableState, game: GameContext, system=None) -> list[Candidate]:
        """Use Pooltool ghost-ball geometry, then check finite paths and cut feasibility."""
        from pooltool.ai.pot.core import calc_potting_angle, calc_shadow_ball_center, get_potting_point, is_jaw_in_way
        system = system if system is not None else self.build(state)
        cue_id = system.cue.cue_ball_id
        cue = next(b for b in state.balls if b.id == cue_id)
        radius, width = state.geometry.ball_radius, state.geometry.width
        result = []
        for ball in state.balls:
            if ball.id not in target_ids(state, game):
                continue
            for pocket_id, engine_id in self.pocket_map(state).items():
                obj, pocket = system.balls[ball.id], system.table.pockets[engine_id]
                ghost = self.point(calc_shadow_ball_center(obj, system.table, pocket))
                pot = self.point(get_potting_point(obj, system.table, pocket))
                dx, dy = ghost.x-cue.position.x, ghost.y-cue.position.y
                ox, oy = pot.x-ball.position.x, pot.y-ball.position.y
                travel, object_travel = hypot(dx, dy), hypot(ox, oy)
                if travel < 1e-8 or object_travel < 1e-8:
                    continue
                alignment = (dx*ox+dy*oy)/(travel*object_travel)
                if alignment < cos(radians(75)):
                    continue
                if not (radius <= ghost.x <= 1-radius and radius <= ghost.y <= width-radius):
                    continue
                if not path_clear(cue.position, ghost, state, {cue_id, ball.id}):
                    continue
                if not path_clear(ball.position, pot, state, {cue_id, ball.id}):
                    continue
                if is_jaw_in_way(obj, system.table, pocket):
                    continue
                phi = calc_potting_angle(system.balls[cue_id], obj, system.table, pocket)
                result.append(Candidate(ball.id, pocket_id, phi, ghost,
                                        travel + object_travel + 2*(1-alignment)))
        return sorted(result, key=lambda c: (c.difficulty, c.ball_id, c.pocket_id))

    def simulate(self, system, phi: float, speed: float):
        """Simulate a fresh copy; speed is cue-tip table lengths/second, no spin."""
        shot = system.copy()
        shot.cue.set_state(phi=phi, V0=speed*self.length_m, a=0, b=0, theta=0)
        # A capped simulation must never be mistaken for a naturally settled shot.
        shot = self.pt.simulate(shot, engine=self.engine, inplace=True, max_events=1000)
        if len(shot.events) >= 1000:
            raise RuntimeError("Simulation exceeded the event budget")
        return shot

    def evaluate(self, shot, state: TableState, game: GameContext, candidate: Candidate) -> Outcome:
        """Evaluate the supported post-break, assigned-group, called-pot rule subset.

        We inspect events directly to preserve arbitrary perception IDs. This is
        not a full referee: break, placement, jumped balls and physical cue fouls
        are outside this level center-ball simulator and the MVP input scope.
        """
        cue_id = shot.cue.cue_ball_id
        first = None
        first_time = float("inf")
        pockets = {}
        rail_after = False
        for event in shot.events:
            ids = tuple(agent.id for agent in event.agents)
            kind = str(event.event_type)
            if kind == "ball_ball" and cue_id in ids and first is None:
                first = next(i for i in ids if i != cue_id)
                first_time = event.time
            elif kind == "ball_pocket":
                pockets[ids[0]] = ids[1]
            elif kind in {"ball_linear_cushion", "ball_circular_cushion"} and event.time >= first_time:
                rail_after = True
        eligible = target_ids(state, game)
        scratch = cue_id in pockets
        foul = scratch or first not in eligible or (not pockets and not rail_after)
        made_call = pockets.get(candidate.ball_id) == self.pocket_map(state)[candidate.pocket_id]
        eight = next((b.id for b in state.balls if b.type is BallType.EIGHT), None)
        eight_down = eight is not None and eight in pockets
        win = eight_down and candidate.ball_id == eight and made_call and not foul and eight in eligible
        loss = eight_down and not win
        reason = ("illegal eight-ball pot" if loss else "scratch" if scratch else
                  "illegal first contact" if first not in eligible else
                  "no pocket or rail after contact" if foul else
                  "rack won" if win else "called pot made" if made_call else "called pot missed")
        return Outcome(foul, loss, win, made_call, frozenset(pockets), reason)

    def next_state(self, shot, state: TableState, observation_id: str) -> TableState:
        """Use settled unpocketed centers as a synthetic observation for tests/replay."""
        from pooltool.constants import pocketed
        balls = []
        for observed in state.balls:
            ball = shot.balls[observed.id]
            if ball.state.s == pocketed:
                continue
            p = self.point(ball.state.rvw[0])
            # Pocket jaws can leave a resting center outside the rectangle. Our
            # contract cannot represent that state, so never silently clip it.
            if not (0 <= p.x <= 1 and 0 <= p.y <= state.geometry.width):
                raise ValueError("Settled ball outside shared playing rectangle; new observation needed")
            balls.append(replace(observed, position=p))
        return replace(state, observation_id=observation_id, balls=tuple(balls))

    def trajectories(self, shot) -> dict[str, list[list[float]]]:
        """Return normalized [time,x,y,visible] samples for the HTML replay."""
        from pooltool.constants import pocketed
        self.pt.continuize(shot, dt=1/30, inplace=True)
        return {key: [[float(s.t), self.point(s.rvw[0]).x, self.point(s.rvw[0]).y,
                       int(s.s != pocketed)] for s in ball.history_cts]
                for key, ball in shot.balls.items()}

    def guides(self, shot) -> tuple[GuideSegment, ...]:
        """Build event-to-event path segments, splitting cue motion at first contact."""
        from pooltool.constants import pocketed
        cue = shot.cue.cue_ball_id
        contact = next((e.time for e in shot.events if str(e.event_type) == "ball_ball"
                        and cue in {a.id for a in e.agents}), float("inf"))
        guides = []
        captures = {e.agents[0].id: e for e in shot.events
                    if str(e.event_type) == "ball_pocket"}
        for key, ball in shot.balls.items():
            for before, after in zip(ball.history, list(ball.history)[1:]):
                if before.s == pocketed:
                    break
                a, b = self.point(before.rvw[0]), self.point(after.rvw[0])
                if hypot(a.x-b.x, a.y-b.y) < 1e-7:
                    continue
                # Pocketed states move to a storage location below the table;
                # use the ball's position immediately before capture instead.
                if after.s == pocketed:
                    b = self.point(captures[key].get_ball(key, initial=True).state.rvw[0])
                    if hypot(a.x-b.x, a.y-b.y) < 1e-7:
                        continue
                role = (GuideRole.OBJECT_BALL_PATH if key != cue else
                        GuideRole.CUE_BALL_BEFORE_CONTACT if before.t < contact else
                        GuideRole.CUE_BALL_AFTER_CONTACT)
                guides.append(GuideSegment(role, key, Segment2(a, b)))
        return tuple(guides)
