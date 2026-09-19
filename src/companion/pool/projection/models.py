from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from companion.pool.contracts import TableGeometry
from companion.pool.contracts.serialization import geometry_from_dict
from companion.serialization import read_document

Homography = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


@dataclass(frozen=True)
class ProjectionTarget:
    """Table-to-projector calibration for one fixed pose and output resolution.

    ``table_to_pixel`` maps homogeneous (x, y, 1) in table-length units
    to projector pixels after division by the third component. Geometry supplies
    table bounds and ball radius; IDs identify the calibration and physical pose.
    """

    geometry: TableGeometry
    calibration_id: str
    pose_id: str
    width_px: int
    height_px: int
    table_to_pixel: Homography

    @property
    def table_id(self) -> str:
        return self.geometry.table_id

    @property
    def corners_px(self) -> tuple[tuple[float, float], ...]:
        """Table TL, TR, BR, BL in projector pixels, derived from calibration.

        Correspondence order follows the agreed table frame, not image sorting.
        Camera pixels cannot be substituted for these projector-pixel locations.
        """
        from .calibration import map_point
        return tuple(map_point(self.table_to_pixel, x, y) for x,y in
                     ((0,0),(1,0),(1,self.geometry.width),(0,self.geometry.width)))

    def __post_init__(self) -> None:
        if not all((self.table_id, self.calibration_id, self.pose_id)):
            raise ValueError("Projection requires table, calibration, and pose IDs")
        if any(type(v) is not int or v <= 0 for v in (self.width_px, self.height_px)):
            raise ValueError("Projector dimensions must be positive integer pixels")
        h = self.table_to_pixel
        if len(h) != 3 or any(len(row) != 3 for row in h):
            raise ValueError("table_to_pixel must be a 3x3 homography")
        if not all(isfinite(v) for row in h for v in row):
            raise ValueError("Homography entries must be finite")
        scale = max(abs(v) for row in h for v in row)
        if scale == 0:
            raise ValueError("Homography must be invertible")
        a, b, c = (tuple(v / scale for v in row) for row in h)
        determinant = (a[0] * (b[1] * c[2] - b[2] * c[1])
                       - a[1] * (b[0] * c[2] - b[2] * c[0])
                       + a[2] * (b[0] * c[1] - b[1] * c[0]))
        if abs(determinant) < 1e-15:
            raise ValueError("Homography is singular or numerically unusable")


@dataclass(frozen=True)
class ProjectionFrame:
    """The final projector image plus the observation and calibration it belongs to.

    ``rgb`` contains width * height * 3 RGB8 bytes, row-major from the top-left,
    with no alpha or padding. The application verifies the pose before display.
    """

    observation_id: str
    table_id: str
    calibration_id: str
    pose_id: str
    width_px: int
    height_px: int
    rgb: bytes

    def __post_init__(self) -> None:
        if not all((self.observation_id, self.table_id, self.calibration_id, self.pose_id)):
            raise ValueError("A frame must retain observation/table/calibration/pose provenance")
        if any(type(v) is not int or v <= 0 for v in (self.width_px, self.height_px)):
            raise ValueError("Frame dimensions must be positive integer pixels")
        if not isinstance(self.rgb, bytes) or len(self.rgb) != self.width_px * self.height_px * 3:
            raise ValueError("Frame requires exactly width * height * 3 RGB bytes")

    def save_ppm(self, path: Path) -> None:
        """Dependency-free artifact format; display adapters may use other formats."""
        path.parent.mkdir(parents=True, exist_ok=True)
        header = f"P6\n{self.width_px} {self.height_px}\n255\n".encode("ascii")
        path.write_bytes(header + self.rgb)


def load_projection_target(path: Path) -> ProjectionTarget:
    fields = read_document(path, "projection_target")
    fields["geometry"] = geometry_from_dict(fields["geometry"])
    fields["table_to_pixel"] = tuple(tuple(row) for row in fields["table_to_pixel"])
    return ProjectionTarget(**fields)
