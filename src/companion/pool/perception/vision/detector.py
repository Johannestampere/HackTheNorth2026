"""Find the balls on the measured cloth and sort them into the four kinds.

The table's geometry is already known by the time this runs (see
`geometry/calibrate.py`), which is what makes the search tractable: a ball is a disc
of *known* radius, because a ball's size is fixed relative to the playing
surface. The detector is built around that one fact. It does not look for
"roughly circular blobs" and then filter by size; it asks, at every point,
how much a disc of exactly the right radius stands out from the cloth.

Everything happens in the rectified top-down view, for the same reason
`detect.marks` works there: after warping, a ball is the same number of
pixels wherever it sits, so a single radius is correct across the whole
table rather than only near the camera.

What separates a ball from the cloth
------------------------------------
Not a fixed colour. The cloth here is a deep red at Lab lightness ~45, but
it vignettes to near-black in the corners and the balls themselves range
from white through to black, so no global threshold divides them. What does
hold everywhere is that the cloth is *locally uniform* and a ball is not: a
median over a window several ball-widths across is the cloth's own colour at
that spot, whatever the lighting is doing there, and a ball is whatever
departs from it. That background model costs one `medianBlur` and removes
the lighting problem outright.

Telling the four kinds apart
----------------------------
Colour cannot do it. A pool set pairs every solid with a stripe of the same
colour - 1 with 9, 2 with 10, and so on - precisely so that colour does not
identify them. The only signal is how the white is arranged, and read from
straight above that is harder than it sounds, because a ball's number sits
in a white patch big enough (on this set, most of a hemisphere) to look like
a stripe's white pole when it happens to face the camera.

The feature that does work is *where* the white is, not how much. A stripe
is a white ball with a coloured band round its equator, so its white poles
run all the way out to the silhouette's edge. A solid's white is a patch
centred on a pole, and a patch is smaller than a pole cap: even face-on it
stops short of the rim. Measuring the white inside a thin annulus at the
ball's edge therefore separates them where measuring the whole disc does
not - on the frame this was developed against, 86% against 71%.

That is not 100%, and it cannot be. A stripe resting with its coloured band
square to the camera shows no white at all and is, from one view, simply a
solid; the tool reports low confidence rather than pretending otherwise.

What stops that measurement from resting on a tuned number is the set. A
pool set pairs every solid with a stripe of the same colour, so once the
fourteen object balls are matched into their colour twins the call becomes a
comparison rather than a threshold: within a pair, the stripe is whichever
of the two shows more rim white. The cut then only has to order two balls of
one colour, instead of sitting in a gap that holds across every table and
every light. That is worth doing even though it does not score better - on
the frame this was developed against both give 12/14 - because the threshold
alone gives 12/14 only between 0.40 and 0.575, a window measured here, while
the pairing holds 12/14 from 0.30 to 0.60. It also makes a complete set come
back as seven and seven, which fourteen independent calls do not.

Where a trained model changes this
----------------------------------
The rim measurement is a proxy for what a person reads off the crop directly,
and `classify/net/` holds a small CNN that reads the crop instead. Scored on
layouts it had never seen, the rim test called 62.8% of the collected crops
correctly and the network 86.7%; through this whole module, against hand
labels over 48 layouts, stripe/solid went from 52.2% to 75.1%. So when a
model is present it makes the opening stripe/solid call and the rim test is
the fallback for when it is not.

Nothing after that changes. One cue, one 8, seven of each kind, colour twins:
those are facts about a pool set rather than measurements, and they still
arbitrate. A network is only ever usually right, and the set rules are what
stop "usually" from putting two 8 balls on a table.

What the known radius costs
---------------------------
Every kernel below is a fixed multiple of `BALL_RADIUS_FRAC`, which is the
leverage the first paragraph describes - and also the one place this module can
be wrong about everything at once. That fraction is an assumption about the
equipment, not a measurement: it is right for the table it was measured on and
wrong for a table whose balls are a different size relative to the cloth.

Too small, and the failures compound while each measurement stays internally
consistent. The search disc fits inside a ball, so one ball returns two
detections, one on its number patch and one on its body. The centres come off
the middle. The rim annulus samples the ball's interior rather than its edge,
which is the whole of the stripe/solid test above. The cushions' shadows become
candidates. Nothing about any one candidate reveals it, and on a real frame all
23 of them came back at up to 0.95 confidence.

So the count is checked against the game instead of against a threshold: a
pool set has 16 balls, and more than 16 means the scale is wrong rather than
that one ball was a close call. See `describe_inconsistency`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import cv2
import numpy as np

from companion.pool.perception.vision import cloth as detect_cloth
from companion.pool.perception.vision.holes import RectifiedView, rectify
from companion.pool.perception.vision.homography import Homography
from companion.pool.perception.vision.surface import active_surface
from companion.pool.perception.vision.table_spec import (
    holes_mm, sheet_h_mm, sheet_w_mm)

# A ball's radius as a fraction of the playing surface's long side. This is
# the one number that sets both the search and the plotted disc, and it is a
# ratio rather than a length because absolute scale is not observable from
# one camera (see `table_spec`) - but the *ratio* is fixed by the equipment.
#
# 0.0150 is measured on the table this was developed against. Regulation
# tables use a 57.15 mm ball, so the ratio falls as the table grows: 0.0144
# on a 7 ft playfield, 0.0128 on an 8 ft, 0.0113 on a 9 ft. Pass
# --ball-radius-frac for a full-size table.
BALL_RADIUS_FRAC = 0.0150

# The rectified view is scaled so a ball comes out exactly this many pixels
# in radius, whatever the table's size or units. Every kernel below is then a
# fixed pixel count that was tuned once and stays tuned, and the warp costs
# the same work on any table.
BALL_RADIUS_PX = 24.0

# The background window: several ball-widths, so a ball never dominates the
# median that is meant to describe the cloth underneath it.
BACKGROUND_WINDOW_R = 4.0

# A candidate must beat the cloth by this much: a mean Lab distance over its
# own disc, minus the same over the ring around it. Subtracting the ring is
# what rejects the cushion edge, which looks like a ball across its width but
# continues along its length, so its surround scores as high as its centre
# while a ball's surround is bare cloth.
#
# 26 was measured on one frame and was too high for the same table under
# different light: three balls scored 24.0, 23.4 and 23.3 and were dropped,
# so a full set came back as 13 and the set-of-fourteen tests that the kinds
# depend on never ran. 22 is where both frames agree - the fixture still
# reports its 16 and the darker frame recovers all of its 16 - and it has
# margin on the side that matters, because at 20 the fixture starts finding
# 18. The gap is narrow, which is the honest description of this cut: it
# separates the dimmest real ball from the brightest piece of cloth, and
# those two are genuinely close on a red table.
MIN_BALL_RESPONSE = 22.0
SURROUND_INNER_R = 1.25
SURROUND_OUTER_R = 2.00

# Two balls in contact have centres exactly 2r apart, so two centres closer
# than that are one ball found twice: no arrangement of real balls puts them
# nearer. The cut sits just inside 2r to leave room for the centring error on
# a pair that really is touching, and no further. At 1.5r it admitted a second
# detection on the same ball - one peak on the number patch and one on the
# body - which is what too small a radius prior produces and what made a
# 16-ball table report 23. See `describe_inconsistency`.
MIN_SEPARATION_R = 1.9

# How close to a pocket a ball may be found. A pocket mouth is a ball-sized
# hole in the cloth, dark and colourless, and would otherwise be reported as
# an 8 ball sitting on the edge of the table. The cost is that a ball hanging
# in the jaws is not placed - which is the reading a pool tool wants anyway,
# since it is on its way in.
#
# The disc is only the hole at the cushion line. Mouths open inward from
# there, and growing this radius until it reaches them swallows real balls
# that are merely near a cushion first. `_search_region` therefore also cuts
# a slot along each pocket's inward axis; see `MOUTH_ALONG_R`.
POCKET_CLEARANCE_R = 2.5

# The mouth of a pocket, as a slot rather than a disc. `along` is half the
# opening's width, `depth` is how far the cut reaches into the cloth, both
# in ball radii from the cushion-line hole. A ball on the cloth beside the
# pocket is off-axis; a ball in the middle of the table on the pocket's
# centreline is too deep. Measured: empty mouths and balls in the jaws sat
# at along 0.96-1.79 r and depth 2.8-6.3 r, while every real ball on the
# cloth was either off-axis by more than 3 r or deeper than 8 r.
MOUTH_ALONG_R = 2.4
MOUTH_DEPTH_R = 7.0

# How sure the network must be that a crop is not a ball before its opinion
# is allowed to remove a candidate. High, because the cost is asymmetric: a
# dropped ball is a hole in the reading the game rules cannot recover from,
# while a kept pocket is one spurious ball the counts already warn about. See
# `_drop_non_balls`.
#
# 0.80 is where the recorded frames agree: every `none` the network emitted
# was a pocket mouth (0.816-1.00) and no real ball drew one. 0.85 left the
# side pocket on the full-rack fixture in, at 0.816, when the stored 2:1
# spec stretched the crop just enough to shake the model's confidence.
NOT_A_BALL_CONFIDENCE = 0.80

# How sure the network must be that a crop is the 8 before that crop is
# considered for it alongside the ones the measurements nominate. It only ever
# adds a candidate; which one wins is still decided by `_eight_score`.
LEARNED_EIGHT_CONFIDENCE = 0.90


# A ball resting against a cushion still has its centre a full radius inside
# the cloth, so that is where the search has to stop. Searching the whole
# measured rectangle instead let centres land on the cushion itself, where the
# red cloth wraps over the rail: two of them came back as solids at 0.95
# confidence, measured on the rail's own shadow.
#
# This spends the slack the corner fit needs - the fitted corners can sit a
# little outside the true cloth (see the README's known limitation) - but a
# centre less than a radius from the cushion is not a reading a ball can
# produce, so the geometry is the right thing to believe here.
CUSHION_CLEARANCE_R = 1.0

# White paint, against the cloth it sits on: colourless, and a good part of
# the way from the cloth's lightness up to pure white. The lift is a fraction
# of that headroom rather than a fixed step, so a pale cloth does not read as
# white paint everywhere.
WHITE_CHROMA_MAX = 22.0
WHITE_LIFT_FRAC = 0.30

# Speckle thinner than this is a specular highlight - the ceiling light
# reflected in the gloss - and not paint. Opening by a disc this size removes
# the bright streaks without touching a number patch or a pole cap.
WHITE_OPEN_R = 0.28

# How far from the local cloth colour a pixel must sit to belong to the ball
# rather than to the cloth showing past its edge.
CLOTH_DELTA = 10.0

# The annulus that tells a stripe from a solid, and the white fraction inside
# it that calls a stripe. See the module docstring.
RIM_INNER_R = 0.72
RIM_OUTER_R = 0.98
STRIPE_RIM_WHITE = 0.50

# A pool set pairs every solid with a stripe of the same colour, which is a
# fact about the equipment in the same way that "16 balls" is, and it turns
# the stripe/solid call from a measurement against a threshold into a
# comparison between two balls. Within a pair the stripe is simply whichever
# of the two shows more rim white; the cut only has to order them, not sit in
# a gap that holds across tables and lighting.
#
# That is the whole benefit, and it is robustness rather than accuracy: on the
# frame this was developed against, pairing scores 12/14, the same as the
# threshold. But the threshold reaches 12/14 only for cuts between 0.40 and
# 0.575 - a plateau measured on this one table - while pairing has no cut to
# place. Where the two disagree the pair is believed, because "these two
# oranges, one of them is the 13" is a stronger statement than "this ball is
# 51% white and the line is at 50%".
#
# Pairs are found by matching the balls into sevens on body colour, choosing
# the matching with the lowest total colour distance. Greedy nearest-colour
# does not do: on the test frame the red solid's nearest neighbour is the
# maroon stripe, not its own twin. A whole-set matching is what stops one
# collision cascading, and 7 pairs from 14 balls is small enough to solve
# exactly (see `_best_pairing`).
PAIR_MAX_COLOUR_DISTANCE = 40.0

# Below this rim-white difference the two balls of a pair are not telling us
# anything - both poles are hidden, or both faces are showing - and the pair
# is left to the threshold rather than forced into one of each. A pair split
# on a 0.01 difference would be a coin toss wearing a constraint's authority.
PAIR_MIN_RIM_MARGIN = 0.05

# The 8 ball, judged on the part of it that is not white: nothing else on the
# table is both that dark and that colourless. Checked only after the stripe
# test, because a dark stripe's body is dark and colourless too.
#
# Darkness is the body's lightness against the white paint on the balls, which
# is a reference outside the ball being judged. Against the ball's *own*
# brightest pixels - which is how this was measured before - the test is
# self-referential and inverts: a uniformly dark ball divides a dark body by an
# equally dark 90th percentile and reads as light, so the 8 was only ever found
# when a gloss highlight happened to be sitting on it. It went missing on the
# first frame where one was not. 0.22 is where the recorded full rack's 8
# (0.207) is a candidate without a model, now that pocket mouths are excluded
# by geometry rather than by this cut - at 0.20 the 8 was not nominated and a
# machine with no classifier reported no 8 at all.
EIGHT_BODY_DARKNESS = 0.22   # median body lightness over the table's white
EIGHT_BODY_CHROMA = 12.0

# The cue ball. There is exactly one, so the test is comparative: the whitest
# ball on the table wins, provided it is white enough to be a cue ball at all.
# An absolute cut cannot be trusted here - a stripe lying pole-up measured 94%
# white against the cue ball's 98%.
CUE_MIN_WHITE = 0.80
CUE_MAX_CHROMA = 14.0

# What is not a ball at all. A pool set is white, black and seven saturated
# colours; it contains nothing that is a uniform mid-grey. Table hardware
# does - the bright plastic liner across a pocket mouth is exactly that, and
# it sits inside the cloth outline, survives the cloth mask (the fill that
# stops the balls eating into the surface swallows it too), and scored 36
# where the real balls scored 30 to 107, so neither region nor response
# rules it out. Being grey does.
NOT_A_BALL_CHROMA = 6.0     # colourless, even allowing for a dark ball
NOT_A_BALL_WHITE = 0.22     # and showing too little white to be the cue
NOT_A_BALL_DARKNESS = 0.35  # and too light to be the 8

# The white cut above had been 0.15, which is where the liner itself measures:
# 0.14 on the test frame and 0.16 on the same scene at 720p, so the same object
# was rejected at one resolution and reported as a ball at the other. A cut has
# to sit in a gap rather than on the thing it is cutting. Among candidates
# colourless enough to reach this test at all, the liner shows 0.16 and the
# next one up is the 8 ball at 0.28 - and the 8 is held by the darkness clause
# regardless - so 0.22 has margin on both sides where 0.15 had none.

KINDS = ("cue", "eight", "stripe", "solid")

# A pool set is 16 balls: the cue, the 8, seven stripes and seven solids.
# Nothing that happens on a table makes it 17, which is what lets the count be
# used as a check rather than as a threshold. See `describe_inconsistency`.
SET_SIZE = 16

# Below this many balls, a short count is taken as a game in progress rather
# than as a detection failure, and nothing is said. Above it - a table that is
# nearly full but not quite - a missing ball is the likelier explanation, and
# it is worth saying so, because everything the module decides about kinds
# assumes a whole set. See `describe_inconsistency`.
SHORT_COUNT_FLOOR = 12

# How far a ball may be from a remembered one and still be taken for it. A
# ball that has not been hit does not move at all, and the measured drift of
# a resting ball between two readings off one locked grid is 0.06 r, so this
# is an eight-fold margin on the thing it has to separate. It is deliberately
# far below the 2 r that would reach the next ball along: the question this
# answers is "is this the same ball, still sitting where it was", and a ball
# that has been struck is a different question with a different answer.
SAME_BALL_R = 0.5

# A remembered reading is only allowed to overrule the current frame if it was
# a clearer look at the ball than this frame is getting. Rim white is the
# measurement, so the margin is in the same units as `STRIPE_RIM_WHITE`: a
# pose that shows a quarter more of the ball's paint than the present one is
# seeing something the present one is not.
BETTER_VIEW_MARGIN = 0.25


@dataclass
class Ball:
    """A ball on the table, in table units."""

    id: str
    kind: str
    xy_mm: tuple[float, float]
    radius_mm: float
    confidence: float
    colour_bgr: tuple[int, int, int]
    px: tuple[float, float] | None = None
    number: int | None = None

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "xy_mm": [round(v, 4) for v in self.xy_mm],
            "radius_mm": round(self.radius_mm, 4),
            "confidence": round(self.confidence, 3),
            "colour_bgr": [int(c) for c in self.colour_bgr],
            "px": None if self.px is None else [round(v, 2) for v in self.px],
            "number": self.number,
        }


def ball_radius_mm(radius_frac: float = BALL_RADIUS_FRAC) -> float:
    """A ball's radius in whatever units the active spec is using."""
    return radius_frac * max(sheet_w_mm(), sheet_h_mm())


def describe_inconsistency(balls: list[Ball],
                           radius_frac: float = BALL_RADIUS_FRAC
                           ) -> str | None:
    """Why this reading cannot be a pool table, if it cannot be.

    `radius_frac` is the detector's one free parameter, and it is a *prior*
    about the equipment rather than anything measured from the frame: a ball's
    radius over the playing surface's long side. Point the tool at a table
    whose balls are a different size relative to the cloth - a small table with
    proportionally large balls, say - and the number is simply wrong, and every
    kernel in this module inherits the error, because they are all fixed
    multiples of it.

    What that looks like matters more than that it happens, because it used to
    happen silently. Too small a radius splits one ball into two detections,
    one centred on its number patch and one on its body; it pulls each centre
    off the ball's middle; it samples the stripe/solid annulus in the ball's
    interior instead of at its rim, which is the whole basis of that call; and
    it admits the cushions' shadows. On a 16-ball table it reported 23 balls,
    no 8, and rail shadow as a solid - every one of them at 0.95 confidence,
    because each measurement was internally consistent. Nothing in a
    per-candidate test can see the problem.

    The count can. 16 is a fact about the game, so more than 16 is not a close
    call about one ball but proof that the scale is wrong.
    """
    if SHORT_COUNT_FLOOR <= len(balls) < SET_SIZE:
        # Too few is as informative as too many, and it used to say nothing.
        # A short count is not a tidy partial reading: the cue and the 8 are
        # picked comparatively from the balls present, so a missing ball can
        # hand the 8 to whatever dark ball was found, and the colour pairing
        # needs the full fourteen and silently stops applying without them.
        # The reading that results looks like bad classification while the
        # actual fault is that a ball was never found.
        #
        # Only a nearly-full table is worth saying this about. Balls leave a
        # pool table as the game is played, and a warning on every reading
        # after the first pot is noise that teaches the user to ignore it.
        # A count in the high teens down to `SHORT_COUNT_FLOOR` is the range
        # where "a ball was missed" is likelier than "a ball was potted".
        return (f"only {len(balls)} of a pool set's {SET_SIZE} balls were "
                f"found. The missing ones are not simply absent from the "
                f"report: the cue and the 8 are chosen by comparison against "
                f"the balls that were found, and the stripe/solid pairing "
                f"needs all fourteen object balls, so it does not run at all "
                f"here. Treat every kind on this reading as unsupported, not "
                f"just the ones that are missing. "
                f"LIKELY CAUSE: the balls are too close to the cloth's own "
                f"colour for this light - a dim frame, or a ball in shadow. "
                f"FIX: light the table more evenly, or lower "
                f"MIN_BALL_RESPONSE (currently {MIN_BALL_RESPONSE:.0f}) if "
                f"this table is consistently dimmer than the one it was "
                f"measured on. A ball resting in a pocket's jaws is excluded "
                f"deliberately and is not a fault.")
    if len(balls) <= SET_SIZE:
        return None
    return (f"{len(balls)} balls reported, but a pool set has {SET_SIZE}, so "
            f"the ball radius in use ({radius_frac:.4f} of the playing "
            f"surface's long side) is too small for this table: single balls "
            f"are being found twice, once on the number patch and once on the "
            f"body, and the cushions' shadows are being found as well. "
            f"Positions, stripe/solid and the 8 are all unreliable in this "
            f"state, whatever confidence they carry. "
            f"FIX: pass --ball-radius-frac with a larger value - it is one "
            f"ball's radius divided by the long side of the cloth, so measure "
            f"both with a ruler if this is not a standard table (regulation "
            f"tables are 0.0144 on a 7 ft playfield, 0.0128 on an 8 ft and "
            f"0.0113 on a 9 ft).")


class BallMemory:
    """The clearest look this table has had at each resting ball.

    One overhead frame cannot always tell a stripe from a solid: a stripe
    lying with its coloured band square to the camera shows no white at all
    and measures exactly like a solid. That is a fact about the pose, not
    about the threshold, and the module docstring is careful to say so.

    What breaks it is time, and only time of a particular kind. Repeating the
    measurement on a still table does not: the error is deterministic, so a
    burst of frames returns the same wrong answer every time (measured: seven
    noisy frames of the fixture, the same two balls wrong in all seven).
    Averaging cannot remove a bias. What does is the ball having *rolled* -
    after a shot, a band-on stripe comes to rest showing a pole, and then one
    frame does settle it.

    So this remembers per position rather than per frame: a ball that has not
    moved keeps the best view anyone has had of it, and a ball that has been
    struck is a new ball with no history, because its old reading describes a
    pose it is no longer in. `SAME_BALL_R` draws that line.

    Only a *clearer* look is allowed to overrule the present one - more of the
    ball's paint in view, by `BETTER_VIEW_MARGIN` - so this can add evidence
    the current frame lacks but cannot overwrite evidence it has.
    """

    def __init__(self) -> None:
        self._seen: list[tuple[tuple[float, float], str, float]] = []

    def recall(self, xy_mm: tuple[float, float], radius_mm: float
               ) -> tuple[str, float] | None:
        """The best remembered reading of the ball resting here, if any."""
        for seen_xy, kind, rim_white in self._seen:
            gap = float(np.hypot(xy_mm[0] - seen_xy[0], xy_mm[1] - seen_xy[1]))
            if gap <= SAME_BALL_R * radius_mm:
                return kind, rim_white
        return None

    def remember(self, xy_mm: tuple[float, float], radius_mm: float,
                 kind: str, rim_white: float) -> None:
        """Keep this reading if it is the clearest one seen at this spot."""
        for i, (seen_xy, _, seen_white) in enumerate(self._seen):
            gap = float(np.hypot(xy_mm[0] - seen_xy[0], xy_mm[1] - seen_xy[1]))
            if gap <= SAME_BALL_R * radius_mm:
                if rim_white > seen_white:
                    self._seen[i] = (xy_mm, kind, rim_white)
                return
        self._seen.append((xy_mm, kind, rim_white))

    def forget_moved(self, resting: list[tuple[float, float]],
                     radius_mm: float) -> None:
        """Drop everything that is no longer sitting where it was.

        Called with the positions found in this frame, so a ball that has been
        struck loses its history rather than carrying a stale pose forward.
        """
        kept = []
        for entry in self._seen:
            near = any(np.hypot(entry[0][0] - x, entry[0][1] - y)
                       <= SAME_BALL_R * radius_mm for x, y in resting)
            if near:
                kept.append(entry)
        self._seen = kept


def detect_balls(image: np.ndarray, homography: Homography, *,
                 radius_frac: float = BALL_RADIUS_FRAC,
                 use_vlm: bool | None = None,
                 memory: BallMemory | None = None
                 ) -> tuple[list[Ball], RectifiedView]:
    """The balls on the table, numbered in reading order.

    Returns the balls and the rectified view they were found in, so a caller
    can draw on the same pixels the decision was made from.

    Pass a `BallMemory` to let readings taken before the last shot inform this
    one. Without it every call is independent, which is what the tests and a
    single still image want.
    """
    radius_mm = ball_radius_mm(radius_frac)
    if radius_mm <= 0.0:
        return [], rectify(image, homography)

    view = rectify(image, homography,
                   px_per_mm=BALL_RADIUS_PX / radius_mm,
                   margin_mm=3.0 * radius_mm)
    r = BALL_RADIUS_PX

    field = _colour_field(view.image, r)
    region = _search_region(view, r)
    found: list[tuple[float, float, dict]] = []
    for cx, cy in _find_centres(field.delta, region, r):
        stats = _measure(field, cx, cy, r)
        if _is_a_ball(stats):
            found.append((cx, cy, stats))
    found = _drop_non_balls(view.image, found, r)
    kinds = _classify_with_optional_vlm(view.image, found, r, use_vlm)
    if memory is not None:
        kinds = _recall_better_views(view, found, kinds, radius_mm, memory)

    balls: list[Ball] = []
    ordered = sorted(zip(found, kinds), key=lambda pair: (pair[0][1],
                                                          pair[0][0]))
    for i, ((cx, cy, stats), (kind, confidence, number)) in enumerate(
            ordered, start=1):
        x_mm, y_mm = view.px_to_mm(cx, cy)
        px = homography.to_image(np.array([(x_mm, y_mm)], np.float64))[0]
        balls.append(Ball(
            id=f"B{i}",
            kind=kind,
            xy_mm=(x_mm, y_mm),
            radius_mm=radius_mm,
            confidence=confidence,
            colour_bgr=stats["colour_bgr"],
            px=(float(px[0]), float(px[1])),
            number=number,
        ))
    return balls, view


@dataclass
class _ColourField:
    """The rectified view reduced to the quantities the detector reasons in."""

    lightness: np.ndarray        # Lab L
    chroma: np.ndarray           # distance from the grey axis in Lab
    cloth_lightness: np.ndarray  # L of the local background
    delta: np.ndarray            # Lab distance from the local background
    white: np.ndarray            # white paint, specular speckle removed
    white_lightness: float       # L of that paint: the frame's white reference
    bgr: np.ndarray


def _colour_field(rect: np.ndarray, r: float) -> _ColourField:
    """Model the cloth locally, then describe every pixel relative to it."""
    bgr = rect if rect.ndim == 3 else cv2.cvtColor(rect, cv2.COLOR_GRAY2BGR)
    blurred = cv2.GaussianBlur(bgr, (3, 3), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

    # The cloth's own colour at each point. A ball covers a small part of a
    # window this wide, so the median ignores it and reports the cloth.
    window = int(BACKGROUND_WINDOW_R * r) | 1
    background = cv2.medianBlur(lab, min(255, window))

    lab_f = lab.astype(np.float32)
    bg_f = background.astype(np.float32)
    lightness = lab_f[..., 0]
    cloth_lightness = bg_f[..., 0]
    chroma = np.sqrt((lab_f[..., 1] - 128.0) ** 2 + (lab_f[..., 2] - 128.0) ** 2)
    delta = np.linalg.norm(lab_f - bg_f, axis=2)

    # White paint: colourless, and well up the range from the cloth to pure
    # white. Opening removes the gloss highlights, which are just as bright
    # and just as colourless but far thinner than any painted marking.
    is_white = ((chroma < WHITE_CHROMA_MAX)
                & (lightness > cloth_lightness
                   + WHITE_LIFT_FRAC * (255.0 - cloth_lightness)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (max(3, int(WHITE_OPEN_R * r)) | 1,) * 2)
    white = cv2.morphologyEx(is_white.astype(np.uint8), cv2.MORPH_OPEN,
                             kernel) > 0

    # One white reference for the whole frame, for judging how dark a ball's
    # body is. The paint on the balls is the same white on every one of them,
    # so it reports the light actually falling on the table - which is what a
    # darkness has to be measured against, and what a single ball cannot
    # supply about itself.
    #
    # Where too little paint is showing to measure (balls face-down, or a set
    # with no markings) the brightest pixels in the view stand in. That is
    # weaker, but it is still a reference the ball under test does not set.
    if int(white.sum()) > 40:
        white_lightness = float(np.percentile(lightness[white], 90))
    else:
        white_lightness = float(np.percentile(lightness, 99.5))

    return _ColourField(lightness=lightness, chroma=chroma,
                        cloth_lightness=cloth_lightness, delta=delta,
                        white=white, white_lightness=max(white_lightness, 1.0),
                        bgr=bgr)


def _search_region(view: RectifiedView, r: float) -> np.ndarray:
    """Where a ball's centre is allowed to be: cloth, away from the pockets.

    Bounded four ways, because each catches something the others do not.
    The measured rectangle, inset by one radius, excludes the room and the
    cushions - a ball touching a cushion still has its centre a full radius
    in, so the rest of the rectangle is not somewhere a centre can be. The
    cloth mask excludes the rail, which matters because the fitted corners can
    sit a little outside the true cloth and the lit edge of a rail scores like
    a row of touching balls. The pocket discs exclude the holes at the
    cushion line. The mouth slots exclude the openings that cut inward from
    there, which a disc at the cushion line cannot reach without first
    eating balls that rest against a rail.
    """
    region = np.zeros(view.image.shape[:2], np.float32)
    inset = CUSHION_CLEARANCE_R * r / view.px_per_mm
    cv2.rectangle(region, _ipt(view.mm_to_px(inset, inset)),
                  _ipt(view.mm_to_px(sheet_w_mm() - inset,
                                     sheet_h_mm() - inset)), 1.0, -1)

    if active_surface() == "table":
        cloth = detect_cloth.playing_surface_mask(view.image)
        if cloth is not None and cv2.countNonZero(cloth) > 16 * r * r:
            eroded = cv2.erode(cloth, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (max(3, int(0.5 * r)) | 1,) * 2))
            region *= (eroded > 0).astype(np.float32)

    for xy in holes_mm().values():
        cv2.circle(region, _ipt(view.mm_to_px(*xy)),
                   int(POCKET_CLEARANCE_R * r), 0.0, -1)
    _paint_pocket_mouths(region, view, r)
    return region


def _paint_pocket_mouths(region: np.ndarray, view: RectifiedView,
                         r: float) -> None:
    """Cut each pocket's inward opening out of the search region, in place.

    The hole coordinates sit on the cushion line. The mouths themselves open
    from there into the cloth, which is why a disc at `POCKET_CLEARANCE_R`
    never reaches them without first eating balls that rest against a rail.
    A slot along the inward axis does: it is wide enough for the opening and
    long enough to cover a mouth that sits several radii in, and it misses a
    ball that is merely beside the pocket because that ball is off-axis.
    """
    radius_mm = r / view.px_per_mm
    centre = np.array([sheet_w_mm() / 2.0, sheet_h_mm() / 2.0], np.float64)
    along = MOUTH_ALONG_R * radius_mm
    depth = MOUTH_DEPTH_R * radius_mm
    for xy in holes_mm().values():
        hole = np.array(xy, np.float64)
        inward = centre - hole
        length = float(np.linalg.norm(inward))
        if length < 1e-6:
            continue
        inward = inward / length
        tangent = np.array([-inward[1], inward[0]])
        back = hole - radius_mm * inward
        front = hole + depth * inward
        corners = np.array([
            _ipt(view.mm_to_px(*(back + along * tangent))),
            _ipt(view.mm_to_px(*(back - along * tangent))),
            _ipt(view.mm_to_px(*(front - along * tangent))),
            _ipt(view.mm_to_px(*(front + along * tangent))),
        ], np.int32)
        cv2.fillConvexPoly(region, corners, 0.0)


def _ipt(xy: tuple[float, float]) -> tuple[int, int]:
    return (int(round(xy[0])), int(round(xy[1])))


def _disc(radius: float) -> np.ndarray:
    size = int(radius) * 2 + 1
    y, x = np.mgrid[:size, :size] - radius
    disc = ((x * x + y * y) <= radius * radius).astype(np.float32)
    return disc / disc.sum()


def _annulus(r_in: float, r_out: float) -> np.ndarray:
    size = int(r_out) * 2 + 1
    y, x = np.mgrid[:size, :size] - r_out
    rr = x * x + y * y
    ring = ((rr > r_in * r_in) & (rr <= r_out * r_out)).astype(np.float32)
    return ring / ring.sum()


def _find_centres(delta: np.ndarray, region: np.ndarray,
                  r: float) -> list[tuple[float, float]]:
    """Centre-surround peaks: a ball's worth of contrast on bare cloth."""
    inside = delta * region
    centre = cv2.filter2D(inside, -1, _disc(r))
    surround = cv2.filter2D(inside, -1, _annulus(SURROUND_INNER_R * r,
                                                 SURROUND_OUTER_R * r))
    response = (centre - surround) * region

    # A tight peak test; separation is the greedy pass below. A window sized
    # to `MIN_SEPARATION_R` collapsed two touching balls into one whenever
    # the stronger ball's skirt outranked the weaker ball's centre inside
    # that window - measured on a clustered live frame, the 12 scored 35.8
    # against a cut at 22 and sat 3 r from its neighbour, but never became a
    # local max because the 2-ball's slope 40 px away was still higher.
    # Half a radius is enough to ignore speckle and small enough that a bump
    # on a neighbour's skirt still counts as its own peak.
    peak_span = max(3, int(0.5 * r) | 1)
    local_max = cv2.dilate(response, np.ones((peak_span, peak_span), np.uint8))
    ys, xs = np.nonzero((response >= local_max - 1e-6)
                        & (response > MIN_BALL_RESPONSE))

    picked: list[tuple[float, float]] = []
    for i in np.argsort(-response[ys, xs]):
        x, y = float(xs[i]), float(ys[i])
        if all(np.hypot(x - px, y - py) > MIN_SEPARATION_R * r
               for px, py in picked):
            picked.append((x, y))
    return picked


def _measure(field: _ColourField, cx: float, cy: float, r: float) -> dict:
    """The handful of numbers the four kinds are told apart by.

    Everything is cropped to the ball's own bounding box first. Working on
    the full frame per candidate is the same arithmetic but three thousand
    times as much of it, and it dominated the run: 1.44 s of a 1.94 s
    detection, against 0.02 s here.
    """
    height, width = field.lightness.shape
    pad = int(np.ceil(r)) + 2
    x0, x1 = max(0, int(cx) - pad), min(width, int(cx) + pad + 1)
    y0, y1 = max(0, int(cy) - pad), min(height, int(cy) + pad + 1)

    lightness = field.lightness[y0:y1, x0:x1]
    chroma = field.chroma[y0:y1, x0:x1]
    delta = field.delta[y0:y1, x0:x1]
    white = field.white[y0:y1, x0:x1]
    bgr = field.bgr[y0:y1, x0:x1]

    ys, xs = np.mgrid[y0:y1, x0:x1]
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    disc = d <= RIM_OUTER_R * r
    rim = (d > RIM_INNER_R * r) & disc
    body = disc & ~white          # the ball minus its paint markings

    if int(body.sum()) > 20:
        body_lightness = float(np.median(lightness[body]))
        body_chroma = float(np.median(chroma[body]))
    else:
        # Nothing but paint, which is the cue ball. Read it as the white it is:
        # standing in the ball's own brightest pixels here would hand the 8's
        # test a body_darkness of 1.0 for a reason that has nothing to do with
        # how dark the ball is.
        body_lightness, body_chroma = field.white_lightness, 0.0

    return {
        "white": float(white[disc].mean()),
        "rim_white": float(white[rim].mean()) if rim.any() else 0.0,
        "body_darkness": body_lightness / field.white_lightness,
        "body_chroma": body_chroma,
        "chroma_p90": float(np.percentile(chroma[disc], 90)),
        "colour_bgr": _ball_colour(bgr, delta, white, disc, d, r),
    }


def _ball_colour(bgr: np.ndarray, delta: np.ndarray, white: np.ndarray,
                 disc: np.ndarray, d: np.ndarray,
                 r: float) -> tuple[int, int, int]:
    """The ball's own colour, for drawing it on the plan of the table.

    Taken from the pixels that are neither its white markings nor the cloth
    showing past its edge, and preferring the middle of the ball where there
    is no cloth to leak in at all. A plain chroma threshold does not work:
    the darker balls here sit below any cut that also excludes the cloth, so
    the sample came out either empty or, worse, made of red cloth.
    """
    paint = disc & ~white & (delta > CLOTH_DELTA)
    middle = paint & (d <= 0.75 * r)
    sample = middle if int(middle.sum()) >= 30 else paint
    if int(sample.sum()) < 15:
        return (240, 240, 240)
    return tuple(int(v) for v in np.median(bgr[sample], axis=0))


def _is_a_ball(stats: dict) -> bool:
    """Reject table hardware that scored like a ball.

    The one thing every candidate must pass. A ball is white, black, or
    coloured; a candidate that is colourless while being neither pale enough
    for the cue nor dark enough for the 8 is a mid-grey object, and a pool
    set does not contain one. See `NOT_A_BALL_CHROMA`.
    """
    return not (stats["body_chroma"] < NOT_A_BALL_CHROMA
                and stats["white"] < NOT_A_BALL_WHITE
                and stats["body_darkness"] > NOT_A_BALL_DARKNESS)


def _cue_score(stats: dict) -> float:
    """How much this looks like the cue ball: all white, no colour anywhere."""
    return stats["white"] - 0.02 * stats["chroma_p90"]


def _eight_score(stats: dict) -> float:
    """How much this looks like the 8: dark body, no colour in it. Lower wins.

    The two halves are put on one scale so neither can be traded away - a
    black-but-slightly-coloured dark stripe and a grey-but-colourless piece of
    hardware are both further from the 8 than the 8 is.
    """
    return (stats["body_darkness"] / EIGHT_BODY_DARKNESS
            + stats["body_chroma"] / EIGHT_BODY_CHROMA)


def _learned_opinion(rectified: np.ndarray,
                     found: list[tuple[float, float, dict]],
                     r: float) -> list[tuple[str, float]] | None:
    """What the trained network calls each crop, or None if there is no model.

    None is the ordinary case on a machine that has never trained one, and
    everything downstream is written to work without this. The network is an
    additional witness, not a replacement for the measurements: see
    `_classify` for what is allowed to do with it.
    """
    try:
        from companion.pool.perception.vision.classify.net import (
            load_classifier, predict)
    except ImportError:
        return None
    net = load_classifier()
    if net is None or not found:
        return None
    try:
        from companion.pool.perception.vision.classify.baseten import crop_disc

        crops = [crop_disc(rectified, cx, cy, r) for cx, cy, _ in found]
        return predict(net, crops)
    except Exception as exc:
        print(f"note: ball classifier failed ({exc}); using measurements",
              file=sys.stderr)
        return None


def _drop_non_balls(rectified: np.ndarray,
                    found: list[tuple[float, float, dict]],
                    r: float) -> list[tuple[float, float, dict]]:
    """Discard candidates the network confidently says are not balls.

    `_is_a_ball` rejects hardware on colour alone, which cannot see a pocket:
    a mouth is a ball-sized hole that is dark and colourless, which is also a
    fair description of the 8. The network can see it, because it was trained
    with a `none` class on crops cut from this very pipeline, and it is the
    only witness here that has ever been shown one.

    This matters more than it used to. A mouth only stayed out of the reading
    while the fitted quad sat inside the true cloth and cropped the mouths out
    of the rectified view before anything looked at them; with the quad on the
    cushion line they are inside it, and two of them came back as balls on a
    full rack. Excluding them by geometry instead means growing a clearance
    disc centred on the cushion line until it reaches a mouth that opens
    inward from there, and that disc reaches real balls resting near a cushion
    first - measured, it cost a correctly-read rack before it caught either
    mouth.

    Only a confident `none` is acted on, and only to drop a candidate: the
    network is never allowed to *name* a ball here, which stays the job of
    `_classify` and the arithmetic about what a pool set contains. Measured
    across the recorded frames, every `none` above this threshold was a pocket
    mouth (0.816-1.00) and no real ball drew one. A mouth the network is
    unsure about is still a job for the mouth slots in `_search_region`.
    """
    if not found:
        return found
    learned = _learned_opinion(rectified, found, r)
    if learned is None:
        return found
    keep = [candidate for candidate, call in zip(found, learned)
            if not (call[0] == "none" and call[1] >= NOT_A_BALL_CONFIDENCE)]
    # All of them cannot be pockets. A reading that says so is a model that
    # has lost its footing - a badly scaled crop, a frame it cannot read -
    # and dropping the whole table on its word is worse than keeping it.
    return keep if keep else found


def _classify_with_optional_vlm(rectified: np.ndarray,
                                found: list[tuple[float, float, dict]],
                                r: float,
                                use_vlm: bool | None = None
                                ) -> list[tuple[str, float, int | None]]:
    """OpenCV kinds, optionally merged with one Baseten call.

    Live `B` passes `use_vlm=False`. The model is opt-in because a round trip
    is seconds, not milliseconds.
    """
    opencv = _classify(found, _learned_opinion(rectified, found, r))
    fallback = [(kind, conf, None) for kind, conf in opencv]
    try:
        from companion.pool.perception.vision.classify.baseten import (classify_found, merge_with_opencv,
                                      vlm_wanted)
    except ImportError:
        return fallback
    enabled = vlm_wanted() if use_vlm is None else use_vlm
    if not enabled:
        return fallback
    try:
        override = classify_found(rectified, found, r)
    except Exception as exc:
        print(f"note: Baseten classify failed ({exc}); using OpenCV kinds",
              file=sys.stderr)
        return fallback
    if override is None:
        return fallback
    return merge_with_opencv(override, opencv, found)


def _recall_better_views(view: RectifiedView,
                         found: list[tuple[float, float, dict]],
                         kinds: list[tuple[str, float, int | None]],
                         radius_mm: float,
                         memory: BallMemory
                         ) -> list[tuple[str, float, int | None]]:
    """Let a clearer earlier look at a resting ball stand in for this one.

    Only stripe and solid are revisited. The cue and the 8 are already decided
    comparatively over the whole set, which is a stronger argument than one
    ball's history, and letting memory touch them could put two 8s on a table
    that has one.
    """
    resting = [view.px_to_mm(cx, cy) for cx, cy, _ in found]
    memory.forget_moved(resting, radius_mm)

    out = list(kinds)
    recalled: list[int] = []
    for i, ((cx, cy, stats), (kind, confidence, number)) in enumerate(
            zip(found, kinds)):
        if kind not in ("stripe", "solid"):
            continue
        xy_mm = view.px_to_mm(cx, cy)
        rim_white = stats["rim_white"]
        seen = memory.recall(xy_mm, radius_mm)
        if seen is not None:
            seen_kind, seen_white = seen
            if seen_white > rim_white + BETTER_VIEW_MARGIN:
                # An earlier pose showed materially more of this ball's paint
                # than the present one does. Believe the better look, and say
                # it is a recalled reading by keeping the confidence modest.
                out[i] = (seen_kind, min(confidence, 0.75), number)
                recalled.append(i)
        memory.remember(xy_mm, radius_mm, kind, rim_white)

    # A recalled ball was half of a pair, and its twin was given the opposite
    # kind by a comparison that has just been overruled. Left alone that puts
    # eight stripes on a table that holds seven. The twin follows the ball
    # that was remembered, because the pairing is the thing that said they
    # differ and the remembered view is the better evidence of which is which.
    if recalled:
        _follow_recalled_twins(found, out, recalled)
    return out


def _follow_recalled_twins(found: list[tuple[float, float, dict]],
                           kinds: list[tuple[str, float, int | None]],
                           recalled: list[int]) -> None:
    """Give a recalled ball's colour twin the other kind. In place."""
    objects = [i for i, (kind, _, _) in enumerate(kinds)
               if kind in ("stripe", "solid")]
    if len(objects) != SET_SIZE - 2:
        return
    for i, j in _best_pairing(objects, found):
        one = i if i in recalled else (j if j in recalled else None)
        if one is None:
            continue
        other = j if one == i else i
        if other in recalled:
            continue            # both remembered; nothing to propagate
        opposite = "solid" if kinds[one][0] == "stripe" else "stripe"
        kinds[other] = (opposite, min(kinds[other][1], 0.75), kinds[other][2])


def _lab_ab(bgr: tuple[int, int, int]) -> tuple[float, float]:
    """The colour's position on the two Lab chroma axes.

    Lightness is deliberately dropped. A stripe is mostly white, so it is far
    lighter than its own solid twin; hue is the part of the colour the two
    share, and it is the only part a pairing can be built on.
    """
    pixel = np.uint8([[list(bgr)]])
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float64)
    return float(lab[1]) - 128.0, float(lab[2]) - 128.0


def _hue_distance(one: tuple[float, float], two: tuple[float, float]) -> float:
    """How far apart two ball colours are, ignoring how light they are."""
    return float(np.hypot(one[0] - two[0], one[1] - two[1]))


def _best_pairing(indices: list[int],
                  found: list[tuple[float, float, dict]]
                  ) -> list[tuple[int, int]]:
    """Match these balls into colour twins, cheapest total distance wins.

    Exhaustive over perfect matchings, which is the right algorithm at this
    size: 14 balls is 135135 pairings, a few milliseconds, and it is exact
    where a greedy pass is not. The recursion fixes the lowest unmatched ball
    and tries every partner for it, so each matching is generated once.
    """
    ab = {i: _lab_ab(found[i][2]["colour_bgr"]) for i in indices}

    def cost(i: int, j: int) -> float:
        return _hue_distance(ab[i], ab[j])

    best: list[tuple[float, list[tuple[int, int]]]] = [(float("inf"), [])]

    def walk(rest: list[int], sofar: list[tuple[int, int]], total: float
             ) -> None:
        if total >= best[0][0]:
            return                      # no completion can beat the incumbent
        if not rest:
            best[0] = (total, list(sofar))
            return
        head, tail = rest[0], rest[1:]
        for k, other in enumerate(tail):
            sofar.append((head, other))
            walk(tail[:k] + tail[k + 1:], sofar, total + cost(head, other))
            sofar.pop()

    walk(sorted(indices), [], 0.0)
    return best[0][1]


def _pair_kinds(objects: list[int], found: list[tuple[float, float, dict]],
                kinds: list[tuple[str, float]]) -> None:
    """Re-read stripe/solid as a comparison inside each colour twin.

    Modifies `kinds` in place. Applied only to a full set of fourteen object
    balls: with any other number the pairing is guesswork, since a missing
    ball leaves a real twin to be matched against a stranger.
    """
    if len(objects) != SET_SIZE - 2:
        return
    for i, j in _best_pairing(objects, found):
        one, two = found[i][2], found[j][2]
        apart = _hue_distance(_lab_ab(one["colour_bgr"]),
                              _lab_ab(two["colour_bgr"]))
        if apart > PAIR_MAX_COLOUR_DISTANCE:
            continue                    # not a twin; leave both to the cut
        margin = abs(one["rim_white"] - two["rim_white"])
        if margin < PAIR_MIN_RIM_MARGIN:
            continue                    # neither is showing a pole
        brighter = one["rim_white"] > two["rim_white"]
        stripe, solid = (i, j) if brighter else (j, i)
        confidence = float(np.clip(0.5 + 0.65 * margin / STRIPE_RIM_WHITE,
                                   0.5, 0.95))
        kinds[stripe] = ("stripe", confidence)
        kinds[solid] = ("solid", confidence)


def _classify(found: list[tuple[float, float, dict]],
              learned: list[tuple[str, float]] | None = None
              ) -> list[tuple[str, float]]:
    """Assign a kind and a confidence to every candidate.

    Done over the whole set rather than one ball at a time, because the cue
    ball's test is comparative: it is the whitest ball present, and no
    absolute cut separates it from a stripe lying pole-up.

    `learned` is the network's opinion of each crop when a trained model is
    present. It decides the *opening* stripe/solid call, because that is the
    judgement the measurements are worst at - on held-out layouts the rim
    test scored 62.8% against the network's 86.7%, and the balls it gets
    wrong are the ones a threshold cannot reach rather than the ones near
    the line.

    Everything after that is unchanged and still arbitrates, because those
    rules are facts about the game and the network is only usually right. A
    set has one cue and one 8, and seven of each kind; a model confident
    that two balls are the 8 is a model that is wrong about one of them, and
    the comparative tests below are what catch it.
    """
    kinds: list[tuple[str, float]] = []
    for i, (_, _, stats) in enumerate(found):
        call = learned[i] if learned is not None else None
        if call is not None and call[0] in ("stripe", "solid"):
            kinds.append((call[0], _learned_confidence(call[1])))
        else:
            # No model, or the model called this crop something the opening
            # pass does not decide (cue, 8, not-a-ball). Those are all settled
            # comparatively below, so the measurement is the right starting
            # point and the network's view of them is not discarded so much
            # as deferred.
            kind = ("stripe" if stats["rim_white"] >= STRIPE_RIM_WHITE
                    else "solid")
            kinds.append((kind, _confidence(kind, stats)))

    # The 8 is one particular ball, not a category, so like the cue its test
    # has to end with one of them. Several candidates clear the absolute gates
    # - a dark stripe lying band-on has a dark colourless body too, and on the
    # test frame a purple stripe measures darker than the 8 itself - and with
    # no comparison between them a radius prior slightly too large reported
    # five 8 balls on one frame. The darkest colourless body wins; the others
    # keep the kind their colour already gave them.
    eights = [i for i, (_, _, s) in enumerate(found)
              if s["rim_white"] < STRIPE_RIM_WHITE
              and s["body_darkness"] <= EIGHT_BODY_DARKNESS
              and s["body_chroma"] <= EIGHT_BODY_CHROMA]
    # The network gets a say in who is even considered, for the same reason it
    # decides stripe from solid: the darkness gate is still a close call under
    # a different light, and a confident `eight` from the network is better
    # evidence about which crop looks like the 8 than a threshold a dark
    # stripe can also clear. Mouths used to win this comparison by being
    # blacker than any ball; they are excluded upstream in `_search_region`.
    if learned is not None:
        eights += [i for i, call in enumerate(learned)
                   if call[0] == "eight" and call[1] >= LEARNED_EIGHT_CONFIDENCE
                   and i not in eights
                   and found[i][2]["rim_white"] < STRIPE_RIM_WHITE]
    if eights:
        best = min(eights, key=lambda i: _eight_score(found[i][2]))
        kinds[best] = ("eight", _confidence("eight", found[best][2]))

    # Exactly one cue ball, and it is the whitest thing on the table.
    cue = [i for i, (_, _, s) in enumerate(found)
           if s["white"] >= CUE_MIN_WHITE and s["chroma_p90"] <= CUE_MAX_CHROMA]
    if cue:
        best = max(cue, key=lambda i: _cue_score(found[i][2]))
        runner_up = max((_cue_score(found[i][2]) for i in cue if i != best),
                        default=0.0)
        margin = _cue_score(found[best][2]) - runner_up
        kinds[best] = ("cue", float(np.clip(0.5 + 2.0 * margin, 0.5, 1.0)))

    # Everything the cue and the 8 did not claim is a solid or a stripe, and
    # those come in colour twins. Done last because it needs to know which
    # balls are spoken for: the cue and the 8 have no twin, and pairing them
    # into the sevens would drag a real pair apart to find them a partner.
    _pair_kinds([i for i, (kind, _) in enumerate(kinds)
                 if kind in ("stripe", "solid")], found, kinds)
    return kinds


def _learned_confidence(probability: float) -> float:
    """The network's probability, on the same scale the rest of this reports.

    Squeezed into the same 0.5-0.95 band as the measured confidences so the
    two are comparable to a caller and to `_pair_kinds`, and capped below 1
    for the same reason every stripe/solid call is: a model that has seen a
    few hundred crops of one table can be confidently wrong, and softmax
    output is not calibrated evidence.
    """
    return float(np.clip(0.5 + 0.45 * (probability - 0.5) / 0.5, 0.5, 0.95))


def _confidence(kind: str, stats: dict) -> float:
    """How far the deciding measurement sat from the line it had to cross.

    Reported rather than hidden because the stripe/solid call is a genuine
    measurement with a margin, and a ball sitting near the line should not
    be presented as settled.

    Capped below 1 for stripe and solid however clear the measurement is.
    The reading can be unambiguous and still wrong: a stripe lying with its
    coloured band square to the camera shows no white at all, which measures
    exactly like a solid and cannot be told from one, so certainty here would
    be a claim about the ball rather than about the pixels.
    """
    if kind == "eight":
        slack = min(EIGHT_BODY_DARKNESS - stats["body_darkness"],
                    (EIGHT_BODY_CHROMA - stats["body_chroma"])
                    / EIGHT_BODY_CHROMA)
        return float(np.clip(0.5 + 2.5 * slack, 0.5, 1.0))
    gap = abs(stats["rim_white"] - STRIPE_RIM_WHITE) / STRIPE_RIM_WHITE
    return float(np.clip(0.5 + 0.65 * gap, 0.5, 0.95))
