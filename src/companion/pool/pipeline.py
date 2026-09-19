"""Pure single-batch orchestration. Acquisition, retries, and display belong to the app."""

from dataclasses import dataclass

from companion.sensors.models import CaptureBatch

from .contracts import (
    BallType, CoverageStatus, GameContext, GuideRole, InsufficientInformation,
    NeedsMoreViews, NoFeasibleShot, PerceptionReady, PlanningReady, PlayerGroup,
    ShotPlan, TableGeometry, TableState, UnusableCapture,
)
from .perception.interface import TablePerception
from .planning.interface import ShotPlanner
from .projection.interface import ShotRenderer
from .projection.models import ProjectionFrame, ProjectionTarget


def validate_plan_for_state(plan: ShotPlan, state: TableState, game: GameContext) -> None:
    """Check stage handoff references and MVP target eligibility, not shot physics.

    A valid reference to the eight ball is not proof of a legal winning outcome:
    planning must still evaluate first contact, fouls, and the called pocket in
    simulated events. These checks only catch inconsistent recommendations.
    """
    if plan.observation_id != state.observation_id or plan.table_id != state.geometry.table_id:
        raise ValueError("Shot plan does not belong to this observation/table frame")
    if state.coverage is not CoverageStatus.COMPLETE:
        raise ValueError("A shot plan requires a complete table observation")
    if game.player_group is None or game.is_break or game.ball_in_hand:
        raise ValueError("A ready MVP plan requires assigned groups and an ordinary placed-ball shot")
    balls = {ball.id: ball for ball in state.balls}
    cue = balls.get(plan.cue_aim.cue_ball_id)
    if cue is None or cue.type is not BallType.CUE or cue.position != plan.cue_aim.origin:
        raise ValueError("Cue aim must reference the observed cue ball at its original position")
    if plan.target_ball_id not in balls or plan.target_ball_id == cue.id:
        raise ValueError("Target ball must identify an observed object ball")
    if plan.target_pocket_id not in {p.id for p in state.geometry.pockets}:
        raise ValueError("Target pocket must belong to this table")

    own_type = BallType.SOLID if game.player_group is PlayerGroup.SOLIDS else BallType.STRIPE
    own_balls_remain = any(ball.type is own_type for ball in state.balls)
    target = balls[plan.target_ball_id]
    if own_balls_remain:
        if target.type is not own_type:
            raise ValueError("Called target must belong to the shooter's uncleared group")
    else:
        # Unknown balls could belong to the shooter; never mistake them for a
        # cleared group and authorize an early eight-ball attempt.
        if any(ball.type is BallType.UNKNOWN for ball in state.balls):
            raise ValueError("Unknown ball types prevent confirming eight-ball eligibility")
        if target.type is not BallType.EIGHT:
            raise ValueError("After clearing the shooter's group, the called target must be the eight")

    for guide in plan.guides:
        if guide.ball_id not in balls:
            raise ValueError("Guide must reference an observed ball")
        is_cue_guide = guide.role is not GuideRole.OBJECT_BALL_PATH
        if is_cue_guide != (guide.ball_id == cue.id):
            raise ValueError("Guide role does not match its cue/object ball ID")


@dataclass(frozen=True)
class PreparedProjection:
    """A completed pipeline pass: observed table, chosen shot, and output pixels."""

    state: TableState
    plan: ShotPlan
    frame: ProjectionFrame


PoolResult = PreparedProjection | NeedsMoreViews | UnusableCapture | InsufficientInformation | NoFeasibleShot


class PoolPipeline:
    def __init__(self, perception: TablePerception, planner: ShotPlanner, renderer: ShotRenderer):
        self.perception = perception
        self.planner = planner
        self.renderer = renderer

    def prepare(
        self,
        captures: CaptureBatch,
        geometry: TableGeometry,
        game: GameContext,
        target: ProjectionTarget,
    ) -> PoolResult:
        """Prepare a frame for a target pose; this does not rotate or project anything."""
        if target.geometry != geometry:
            raise ValueError("Projector calibration belongs to a different table frame")
        perception = self.perception.estimate(captures, geometry)
        if isinstance(perception, (NeedsMoreViews, UnusableCapture)):
            return perception
        if not isinstance(perception, PerceptionReady):
            raise TypeError("Unexpected perception result")
        state = perception.state
        if state.geometry != geometry:
            raise ValueError("Perception changed the supplied table geometry")
        planning = self.planner.plan(state, game)
        if isinstance(planning, (InsufficientInformation, NoFeasibleShot)):
            return planning
        if not isinstance(planning, PlanningReady):
            raise TypeError("Unexpected planning result")
        plan = planning.plan
        validate_plan_for_state(plan, state, game)
        frame = self.renderer.render(plan, target)
        if (frame.observation_id, frame.table_id, frame.calibration_id, frame.pose_id,
            frame.width_px, frame.height_px) != (
            plan.observation_id, target.table_id, target.calibration_id, target.pose_id,
            target.width_px, target.height_px
        ):
            raise ValueError("Rendered frame provenance or resolution does not match the request")
        return PreparedProjection(state=state, plan=plan, frame=frame)
