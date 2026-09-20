"""Render table-space guidance into calibrated projector RGB pixels, using stdlib."""

from math import cos, sin, tau

from companion.pool.contracts import GuideRole, ShotPlan
from .calibration import map_point
from .models import ProjectionFrame, ProjectionTarget


def clip_segment(a, b, width, height):
    """Clip a finite segment to a rectangle; bound raster work even for offscreen paths."""
    dx, dy = b[0]-a[0], b[1]-a[1]
    lo, hi = 0., 1.
    for p,q in ((-dx,a[0]),(dx,width-a[0]),(-dy,a[1]),(dy,height-a[1])):
        if p == 0:
            if q < 0:
                return None
        elif p < 0:
            lo = max(lo,q/p)
        else:
            hi = min(hi,q/p)
        if lo > hi:
            return None
    return ((a[0]+lo*dx,a[1]+lo*dy),(a[0]+hi*dx,a[1]+hi*dy))


class ProjectionService:
    """Draw cue alignment, nominal trajectories, ghost ball and called pocket.

    Output is black except for guidance. Homography maps geometry before drawing;
    stroke thickness stays three projector pixels. Table/pixel clipping prevents
    off-table path extensions. No Pi/network access occurs while rendering.
    """

    def render(self, plan: ShotPlan, target: ProjectionTarget) -> ProjectionFrame:
        if plan.table_id != target.table_id:
            raise ValueError('Plan and calibration use different table frames')
        pocket = next((p for p in target.geometry.pockets if p.id == plan.target_pocket_id), None)
        if pocket is None:
            raise ValueError('Called pocket is missing from calibrated geometry')
        w, h = target.width_px, target.height_px
        if w*h*3 > 64*1024*1024:
            raise ValueError('Frame exceeds the Pi receiver 64 MiB limit')
        matrix = target.table_to_pixel
        scale = max(abs(v) for row in matrix for v in row)
        matrix = tuple(tuple(v/scale for v in row) for row in matrix)
        width = target.geometry.width
        denominators = [matrix[2][0]*x+matrix[2][1]*y+matrix[2][2]
                        for x,y in ((0,0),(1,0),(1,width),(0,width))]
        if not (all(d > 1e-12 for d in denominators) or all(d < -1e-12 for d in denominators)):
            raise ValueError('Calibration crosses infinity within the table')
        rgb = bytearray(w*h*3)

        def line(a, b, color):
            clipped = clip_segment(a,b,1,width)
            if clipped is None:
                return
            projected = [map_point(matrix,*p) for p in clipped]
            clipped = clip_segment(*projected,w-1,h-1)
            if clipped is None:
                return
            a,b = clipped
            steps = max(1, int(max(abs(b[0]-a[0]),abs(b[1]-a[1])))+1)
            for i in range(steps+1):
                x,y = round(a[0]+(b[0]-a[0])*i/steps), round(a[1]+(b[1]-a[1])*i/steps)
                for oy in (-1,0,1):
                    for ox in (-1,0,1):
                        px,py = x+ox,y+oy
                        if 0 <= px < w and 0 <= py < h:
                            start = (py*w+px)*3
                            rgb[start:start+3] = bytes(color)

        def circle(center, radius, color):
            points = [(center.x+radius*cos(tau*i/64),center.y+radius*sin(tau*i/64)) for i in range(65)]
            for a,b in zip(points,points[1:]):
                line(a,b,color)

        colors = {GuideRole.CUE_ALIGNMENT:(255,255,255),
                  GuideRole.CUE_BALL_BEFORE_CONTACT:(80,230,255),
                  GuideRole.CUE_BALL_AFTER_CONTACT:(80,140,255),
                  GuideRole.OBJECT_BALL_PATH:(255,205,70)}
        for guide in plan.guides:
            a,b = guide.segment.start, guide.segment.end
            line((a.x,a.y),(b.x,b.y),colors[guide.role])
        cue,d = plan.cue_aim.origin,plan.cue_aim.direction
        # Always render an anchored cue line, even for aim-only plans.
        line((cue.x-.12*d.x,cue.y-.12*d.y),(cue.x+.08*d.x,cue.y+.08*d.y),(255,255,255))
        if plan.ghost_ball is not None:
            circle(plan.ghost_ball,target.geometry.ball_radius,(80,230,255))
        circle(pocket.position,pocket.mouth_width/2,(80,255,120))
        return ProjectionFrame(plan.observation_id,target.table_id,target.calibration_id,
                               target.pose_id,w,h,bytes(rgb))
