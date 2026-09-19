"""Explicit JSON codecs: no pickle, model objects, or implicit coordinate conversion."""

from pathlib import Path
from typing import Any

from companion.serialization import read_document, write_document

from .geometry import Point2, Segment2, UnitVector2
from .shot import CueAim, GuideRole, GuideSegment, ShotPlan
from .table import Ball, BallType, CoverageStatus, Pocket, TableGeometry, TableState


def geometry_from_dict(data: dict[str, Any]) -> TableGeometry:
    fields = dict(data)
    fields["pockets"] = tuple(
        Pocket(**{**pocket, "position": Point2(**pocket["position"])})
        for pocket in fields["pockets"]
    )
    return TableGeometry(**fields)


def load_geometry(path: Path) -> TableGeometry:
    return geometry_from_dict(read_document(path, "table_geometry"))


def load_table_state(path: Path) -> TableState:
    fields = read_document(path, "table_state")
    fields["geometry"] = geometry_from_dict(fields["geometry"])
    fields["coverage"] = CoverageStatus(fields["coverage"])
    fields["balls"] = tuple(
        Ball(**{**ball, "position": Point2(**ball["position"]), "type": BallType(ball["type"])})
        for ball in fields["balls"]
    )
    return TableState(**fields)


def save_table_state(path: Path, state: TableState) -> None:
    write_document(path, "table_state", state)


def load_shot_plan(path: Path) -> ShotPlan:
    fields = read_document(path, "shot_plan")
    aim = fields["cue_aim"]
    fields["cue_aim"] = CueAim(**{
        **aim, "origin": Point2(**aim["origin"]), "direction": UnitVector2(**aim["direction"])
    })
    fields["guides"] = tuple(
        GuideSegment(**{
            **guide,
            "role": GuideRole(guide["role"]),
            "segment": Segment2(**{
                key: Point2(**point) for key, point in guide["segment"].items()
            }),
        })
        for guide in fields.get("guides", [])
    )
    if fields.get("ghost_ball") is not None:
        fields["ghost_ball"] = Point2(**fields["ghost_ball"])
    return ShotPlan(**fields)


def save_shot_plan(path: Path, plan: ShotPlan) -> None:
    write_document(path, "shot_plan", plan)
