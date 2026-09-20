"""End to end on one photo: find the balls, plan a shot, draw the vector.

This is the whole chain in one command, for checking that the three stages
agree with each other on a real image rather than on fixtures:

    photo -> PerceptionService  (OpenCV + the CNN)      -> TableState
          -> PooltoolPlanner    (simulated shot search) -> ShotPlan
          -> matplotlib                                 -> a picture

Nothing here is part of the pipeline. `companion.app` remains the entry
point; this only calls the same two services in order and plots what comes
back, so a disagreement between stages shows up as a wrong-looking picture
instead of as numbers that need checking by hand.

The plot is deliberately drawn from the *plan*, not from the planner's
internals: if the cue arrow does not start on the cue ball and point at the
ghost ball, the contract is being filled in wrong.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib                                          # noqa: E402
import matplotlib.pyplot as plt                            # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch                      # noqa: E402

from companion.pool.contracts import (BallType, GuideRole,  # noqa: E402
                                      PerceptionReady, PlanningReady)
from companion.pool.contracts.serialization import (        # noqa: E402
    load_game_context, load_geometry)
from companion.pool.perception.service import PerceptionService   # noqa: E402
from companion.pool.planning.service import (PlannerConfig,       # noqa: E402
                                             PooltoolPlanner)
from companion.sensors.models import load_capture_batch     # noqa: E402

# Ball fill by type, drawn the way the real ball looks: a solid is one
# colour through, a stripe is white with a coloured band across it. Getting
# this backwards would make a classification error invisible, which is most
# of what this picture is for.
FILL = {
    BallType.CUE: "#f8f8f4",
    BallType.EIGHT: "#161616",
    BallType.SOLID: "#d8453a",
    BallType.STRIPE: "#f4f4f0",
    BallType.UNKNOWN: "#8d8d8d",
}
BAND = "#d8453a"       # the stripe's colour band

# One colour per guide role, so the roles stay distinguishable in print and
# for the most common colour-vision deficiencies.
GUIDE = {
    GuideRole.CUE_BALL_BEFORE_CONTACT: ("#f0d048", 2.0, (0, (5, 3))),
    GuideRole.CUE_BALL_AFTER_CONTACT: ("#7bb6ff", 1.6, (0, (2, 3))),
    GuideRole.OBJECT_BALL_PATH: ("#5ad48b", 2.2, "solid"),
    GuideRole.CUE_ALIGNMENT: ("#ffffff", 1.4, (0, (1, 2))),
}

CLOTH = "#1a4a32"          # dark cloth green


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run perception and planning on one capture and plot the shot.")
    parser.add_argument("--captures", type=Path,
                        default=ROOT / "data/local/captures/real_table.json")
    parser.add_argument("--geometry", type=Path,
                        default=ROOT / "fixtures/table_geometry.json")
    parser.add_argument("--game", type=Path,
                        default=ROOT / "fixtures/game_contexts/solids.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/shot.png")
    parser.add_argument("--show", action="store_true",
                        help="Open a window as well as writing the file")
    args = parser.parse_args()

    if not args.captures.exists():
        print(f"error: no capture manifest at {args.captures}", file=sys.stderr)
        print("       recorded frames are device data; see docs/perception.md",
              file=sys.stderr)
        return 2

    print(f"perception: reading {args.captures.name} ...")
    geometry = load_geometry(args.geometry)
    result = PerceptionService().estimate(
        load_capture_batch(args.captures), geometry)
    if not isinstance(result, PerceptionReady):
        print(f"  {type(result).__name__}: {result.reason}", file=sys.stderr)
        return 1
    state = result.state
    counts = {t: sum(b.type is t for b in state.balls) for t in BallType}
    print(f"  {len(state.balls)} balls: "
          + ", ".join(f"{n} {t.value}" for t, n in counts.items() if n))

    print("planning: searching shots with pooltool ...")
    planner = PooltoolPlanner(PlannerConfig())
    plan_result = planner.plan(state, load_game_context(args.game))
    if not isinstance(plan_result, PlanningReady):
        print(f"  {type(plan_result).__name__}: "
              f"{getattr(plan_result, 'reason', '')}", file=sys.stderr)
        return 1
    plan = plan_result.plan
    aim = plan.cue_aim
    print(f"  pot {plan.target_ball_id} into {plan.target_pocket_id}")
    print(f"  aim ({aim.direction.x:+.3f}, {aim.direction.y:+.3f}) "
          f"at {plan.cue_stick_speed:.2f} table lengths/s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _plot(state, plan, args.output, show=args.show)
    print(f"wrote {args.output}")
    return 0


def _plot(state, plan, path: Path, *, show: bool) -> None:
    """Draw the table, the balls, the predicted paths and the aim vector."""
    if not show:
        matplotlib.use("Agg")
    g = state.geometry
    fig, ax = plt.subplots(figsize=(11, 11 * g.width / g.length + 1.4))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor(CLOTH)

    # Cushions and pockets, so the shot is read against the real boundary.
    ax.add_patch(plt.Rectangle((0, 0), g.length, g.width, fill=False,
                               edgecolor="#6b4a2a", linewidth=6, zorder=1))
    for pocket in g.pockets:
        ax.add_patch(Circle((pocket.position.x, pocket.position.y),
                            pocket.mouth_width / 2, color="#0a0a0a", zorder=2))

    # Predicted paths first, so the balls sit on top of them.
    seen_roles = set()
    for guide in plan.guides:
        colour, width, dashes = GUIDE[guide.role]
        s = guide.segment
        ax.plot([s.start.x, s.end.x], [s.start.y, s.end.y],
                color=colour, linewidth=width, linestyle=dashes, zorder=3,
                solid_capstyle="round",
                label=(guide.role.value.replace("_", " ")
                       if guide.role not in seen_roles else None))
        seen_roles.add(guide.role)

    if plan.ghost_ball is not None:
        ax.add_patch(Circle((plan.ghost_ball.x, plan.ghost_ball.y),
                            g.ball_radius, fill=False, edgecolor="#f0d048",
                            linewidth=1.6, linestyle=(0, (3, 2)), zorder=4,
                            label="ghost ball"))

    for ball in state.balls:
        x, y = ball.position.x, ball.position.y
        ax.add_patch(Circle((x, y), g.ball_radius, facecolor=FILL[ball.type],
                            edgecolor="#101010", linewidth=1.0, zorder=5))
        if ball.type is BallType.STRIPE:
            # A horizontal band across the middle, clipped to the ball, so a
            # stripe reads as a stripe rather than as a differently-filled dot.
            ball_face = Circle((x, y), g.ball_radius, transform=ax.transData)
            band = plt.Rectangle((x - g.ball_radius, y - g.ball_radius * 0.46),
                                 2 * g.ball_radius, g.ball_radius * 0.92,
                                 facecolor=BAND, edgecolor="none", zorder=6)
            band.set_clip_path(ball_face)
            ax.add_patch(band)
        if ball.id in (plan.target_ball_id, plan.cue_aim.cue_ball_id):
            ax.add_patch(Circle((x, y), g.ball_radius * 1.7, fill=False,
                                edgecolor="#f0d048", linewidth=1.4, zorder=7))

    # The aim vector itself: anchored at the cue ball, pointing forward.
    # Both endpoints are given in data coordinates, so the arrow head lands
    # where the aim direction actually points even though the y axis is
    # flipped. `ax.arrow` sizes its head in display space instead, which the
    # flip reverses - it draws this shot backwards.
    aim = plan.cue_aim
    reach = 5 * g.ball_radius
    ax.add_patch(FancyArrowPatch(
        (aim.origin.x, aim.origin.y),
        (aim.origin.x + aim.direction.x * reach,
         aim.origin.y + aim.direction.y * reach),
        arrowstyle="-|>", mutation_scale=20, shrinkA=0, shrinkB=0,
        facecolor="#f0d048", edgecolor="#f0d048", linewidth=2.4, zorder=8))

    ax.set_xlim(-0.03, g.length + 0.03)
    ax.set_ylim(g.width + 0.03, -0.03)          # y down, per the contract
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        f"pot {plan.target_ball_id} into {plan.target_pocket_id}   ·   "
        f"{plan.cue_stick_speed:.2f} table lengths/s",
        color="#f2f2f2", fontsize=13, pad=14)
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02),
                       ncol=3, frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color("#cfcfcf")

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
