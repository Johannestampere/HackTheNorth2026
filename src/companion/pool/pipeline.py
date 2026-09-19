"""Pure single-batch orchestration. Acquisition, retries, and display belong to the app."""

from dataclasses import dataclass

from companion.sensors.models import CaptureBatch

from .contracts import (
    BallType, GameContext, InsufficientInformation, NeedsMoreViews, NoFeasibleShot,
    PerceptionReady, PlanningReady, ShotPlan, TableGeometry, TableState, UnusableCapture,
)
from .perception.interface import TablePerception
from .planning.interface import ShotPlanner
from .projection.interface import ShotRenderer
from .projection.models import ProjectionFrame, ProjectionTarget


def validate_plan_for_state(plan: ShotPlan, state: TableState) -> None:
    if plan.observation_id != state.observation_id or plan.table_id != state.geometry.table_id:
        raise ValueError("Shot plan does not belong to this observation/table frame")
    balls = {ball.id: ball for ball in state.balls}
    cue = balls.get(plan.cue_aim.cue_ball_id)
    if cue is None or cue.type is not BallType.CUE or cue.position != plan.cue_aim.origin:
        raise ValueError("Cue aim must reference the observed cue ball at its original position")
    if plan.target_ball_id is not None:
        if plan.target_ball_id not in balls or plan.target_ball_id == cue.id:
            raise ValueError("Target ball must identify an observed object ball")
    if plan.target_pocket_id is not None:
        if plan.target_pocket_id not in {p.id for p in state.geometry.pockets}:
            raise ValueError("Target pocket must belong to this table")


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
        validate_plan_for_state(plan, state)
        frame = self.renderer.render(plan, target)
        if (frame.observation_id, frame.table_id, frame.calibration_id, frame.pose_id,
            frame.width_px, frame.height_px) != (
            plan.observation_id, target.table_id, target.calibration_id, target.pose_id,
            target.width_px, target.height_px
        ):
            raise ValueError("Rendered frame provenance or resolution does not match the request")
        return PreparedProjection(state=state, plan=plan, frame=frame)
