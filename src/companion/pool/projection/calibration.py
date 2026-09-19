"""Map the fixed table rectangle to four corresponding projector-pixel corners."""

from math import isfinite


def map_point(h, x, y):
    """Apply a homography; return projector (x,y) pixels, never camera pixels."""
    denominator = h[2][0]*x+h[2][1]*y+h[2][2]
    if abs(denominator) < 1e-12:
        raise ValueError('Calibration maps this point to infinity')
    return ((h[0][0]*x+h[0][1]*y+h[0][2])/denominator,
            (h[1][0]*x+h[1][1]*y+h[1][2])/denominator)


def from_corners(width, pixels):
    """Solve table→projector homography from TL, TR, BR, BL correspondences.

    Table points are (0,0), (1,0), (1,width), (0,width). Pixel points refer
    to those same physical corners, not sorting by their screen coordinates.
    A convex quadrilateral is required; either winding supports mirrored setups.
    """
    if len(pixels) != 4 or any(len(p) != 2 for p in pixels):
        raise ValueError('Provide four [x,y] projector corners in TL, TR, BR, BL order')
    if not isfinite(width) or width <= 0 or not all(isfinite(v) for p in pixels for v in p):
        raise ValueError('Corner coordinates and table width must be finite')
    cross = []
    for i in range(4):
        a, b, c = pixels[i], pixels[(i+1)%4], pixels[(i+2)%4]
        cross.append((b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0]))
    if not (all(v > 1e-8 for v in cross) or all(v < -1e-8 for v in cross)):
        raise ValueError('Corners must form a nondegenerate convex quadrilateral in perimeter order')
    rows = []
    for (x,y),(u,v) in zip(((0,0),(1,0),(1,width),(0,width)), pixels):
        rows.extend(([x,y,1,0,0,0,-u*x,-u*y,u], [0,0,0,x,y,1,-v*x,-v*y,v]))
    for col in range(8):
        pivot = max(range(col,8), key=lambda i: abs(rows[i][col]))
        rows[col], rows[pivot] = rows[pivot], rows[col]
        value = rows[col][col]
        if abs(value) < 1e-12:
            raise ValueError('Corner calibration is singular')
        rows[col] = [v/value for v in rows[col]]
        for i in range(8):
            if i != col:
                factor = rows[i][col]
                rows[i] = [a-factor*b for a,b in zip(rows[i], rows[col])]
    values = [row[-1] for row in rows]+[1.0]
    return tuple(tuple(values[i:i+3]) for i in (0,3,6))
