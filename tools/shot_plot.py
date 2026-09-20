"""Drawing a measured table, and a shot on top of it.

Two figures share almost all of their geometry - the cloth, the pockets, the
balls - and differ only in what is laid over it, so they are one module. A
second copy would drift, and the point of the pair is that the same table is
recognisably the same in both.

Everything here works in table-length units straight from the contracts: long
side 1.0 on x, origin top-left, y down. The y axis is inverted so the picture
matches that convention rather than matplotlib's default.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

from companion.pool.contracts import BallType, GuideRole

# Ball fill by type, drawn the way the real ball looks: a solid is one colour
# through, a stripe is white with a coloured band across it. Drawing these the
# same way would hide exactly the error these plots exist to reveal.
FILL = {
    BallType.CUE: "#f8f8f4",
    BallType.EIGHT: "#161616",
    BallType.SOLID: "#d8453a",
    BallType.STRIPE: "#f4f4f0",
    BallType.UNKNOWN: "#8d8d8d",
}
BAND = "#d8453a"
CLOTH = "#1a4a32"
RAIL = "#6b4a2a"
PAPER = "#1a1a1a"
INK = "#f2f2f2"
ACCENT = "#f0d048"

GUIDE = {
    GuideRole.CUE_BALL_BEFORE_CONTACT: (ACCENT, 2.0, (0, (5, 3))),
    GuideRole.CUE_BALL_AFTER_CONTACT: ("#7bb6ff", 1.6, (0, (2, 3))),
    GuideRole.OBJECT_BALL_PATH: ("#5ad48b", 2.2, "solid"),
    GuideRole.CUE_ALIGNMENT: ("#ffffff", 1.4, (0, (1, 2))),
}


def plot_state(state, path: Path, *, show: bool = False,
               title: str | None = None, grid_lines: bool = True) -> None:
    """The measured table with its balls: what the camera read.

    Drawn before any planning, so that when a suggested shot looks wrong this
    is where you check whether perception or the planner was at fault.
    """
    fig, ax = _table(state.geometry, grid_lines=grid_lines)
    for ball in state.balls:
        _ball(ax, ball, state.geometry.ball_radius)
        ax.annotate(ball.id,
                    (ball.position.x, ball.position.y - state.geometry.ball_radius * 1.9),
                    color="#d8d8d8", fontsize=7, ha="center", va="center")
    _finish(fig, ax, state.geometry, title or "measured table",
            handles=_type_swatches(state))
    _write(fig, path, show)


def plot_shot(state, plan, path: Path, *, show: bool = False) -> None:
    """The same table with the chosen shot laid over it."""
    fig, ax = _table(state.geometry, grid_lines=False)
    g = state.geometry

    seen = set()
    for guide in plan.guides:
        colour, width, dashes = GUIDE[guide.role]
        s = guide.segment
        ax.plot([s.start.x, s.end.x], [s.start.y, s.end.y], color=colour,
                linewidth=width, linestyle=dashes, zorder=3,
                solid_capstyle="round",
                label=(guide.role.value.replace("_", " ")
                       if guide.role not in seen else None))
        seen.add(guide.role)

    if plan.ghost_ball is not None:
        ax.add_patch(Circle((plan.ghost_ball.x, plan.ghost_ball.y),
                            g.ball_radius, fill=False, edgecolor=ACCENT,
                            linewidth=1.6, linestyle=(0, (3, 2)), zorder=4,
                            label="ghost ball"))

    highlight = {plan.target_ball_id, plan.cue_aim.cue_ball_id}
    for ball in state.balls:
        _ball(ax, ball, g.ball_radius)
        if ball.id in highlight:
            ax.add_patch(Circle((ball.position.x, ball.position.y),
                                g.ball_radius * 1.7, fill=False,
                                edgecolor=ACCENT, linewidth=1.4, zorder=7))

    # Both endpoints in data coordinates, so the head lands where the aim
    # points despite the inverted y axis. `ax.arrow` sizes its head in display
    # space, which the flip reverses - it draws the shot backwards.
    aim = plan.cue_aim
    reach = 5 * g.ball_radius
    ax.add_patch(FancyArrowPatch(
        (aim.origin.x, aim.origin.y),
        (aim.origin.x + aim.direction.x * reach,
         aim.origin.y + aim.direction.y * reach),
        arrowstyle="-|>", mutation_scale=20, shrinkA=0, shrinkB=0,
        facecolor=ACCENT, edgecolor=ACCENT, linewidth=2.4, zorder=8))

    _finish(fig, ax, g,
            f"pot {plan.target_ball_id} into {plan.target_pocket_id}   ·   "
            f"{plan.cue_stick_speed:.2f} table lengths/s")
    _write(fig, path, show)


def _table(g, *, grid_lines: bool):
    """The cloth, the cushions and the pockets, in table-length units."""
    fig, ax = plt.subplots(figsize=(11, 11 * g.width / g.length + 1.6))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(CLOTH)
    ax.add_patch(plt.Rectangle((0, 0), g.length, g.width, fill=False,
                               edgecolor=RAIL, linewidth=6, zorder=1))
    if grid_lines:
        # The measured frame itself, at tenths of the long side. This is the
        # "x/y grid" the geometry produced, drawn so positions can be read off
        # the picture rather than only out of the JSON.
        for i in range(1, 10):
            x = i * g.length / 10
            ax.plot([x, x], [0, g.width], color="#ffffff", alpha=0.10,
                    linewidth=0.8, zorder=1)
        step = g.length / 10
        y = step
        while y < g.width:
            ax.plot([0, g.length], [y, y], color="#ffffff", alpha=0.10,
                    linewidth=0.8, zorder=1)
            y += step
        for i in range(0, 11, 2):
            ax.annotate(f"{i / 10:.1f}", (i * g.length / 10, -0.012),
                        color="#9a9a9a", fontsize=7, ha="center", va="bottom")
    for pocket in g.pockets:
        ax.add_patch(Circle((pocket.position.x, pocket.position.y),
                            pocket.mouth_width / 2, color="#0a0a0a", zorder=2))
    return fig, ax


def _ball(ax, ball, radius: float) -> None:
    x, y = ball.position.x, ball.position.y
    ax.add_patch(Circle((x, y), radius, facecolor=FILL[ball.type],
                        edgecolor="#101010", linewidth=1.0, zorder=5))
    if ball.type is BallType.STRIPE:
        face = Circle((x, y), radius, transform=ax.transData)
        band = plt.Rectangle((x - radius, y - radius * 0.46),
                             2 * radius, radius * 0.92,
                             facecolor=BAND, edgecolor="none", zorder=6)
        band.set_clip_path(face)
        ax.add_patch(band)


def _type_swatches(state) -> list:
    """One legend handle per ball type actually on the table.

    Listing types that are not present would make a missing cue ball look
    like a drawing choice rather than a fact about the scan.
    """
    order = (BallType.CUE, BallType.SOLID, BallType.STRIPE,
             BallType.EIGHT, BallType.UNKNOWN)
    present = [k for k in order if any(b.type is k for b in state.balls)]
    return [plt.Line2D([], [], marker="o", linestyle="none",
                       markerfacecolor=FILL[k], markeredgecolor="#101010",
                       markersize=9,
                       label=f"{k.value} "
                             f"({sum(b.type is k for b in state.balls)})")
            for k in present]


def _finish(fig, ax, g, title: str, handles: list | None = None) -> None:
    ax.set_xlim(-0.03, g.length + 0.03)
    ax.set_ylim(g.width + 0.04, -0.05)          # y down, per the contract
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, color=INK, fontsize=13, pad=14)

    if handles is None:
        handles, _ = ax.get_legend_handles_labels()
    if handles:
        leg = ax.legend(handles=handles, loc="upper center",
                        bbox_to_anchor=(0.5, -0.01), ncol=4, frameon=False,
                        fontsize=9)
        for text in leg.get_texts():
            text.set_color("#cfcfcf")


def _write(fig, path: Path, show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close(fig)
