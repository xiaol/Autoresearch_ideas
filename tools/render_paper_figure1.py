from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ROOT = os.path.dirname(os.path.dirname(__file__))
OUT_DIR = os.path.join(ROOT, "papers", "associative_state_universal_transformers", "figures")
OUT_PNG = os.path.join(OUT_DIR, "unimatrix_paper_overview.png")
OUT_PDF = os.path.join(OUT_DIR, "unimatrix_paper_overview.pdf")


COLORS = {
    "paper": "#fbfaf7",
    "neutral": "#f4f1ea",
    "matrix": "#dbe7f4",
    "accent": "#e6efe3",
    "warm": "#f6e5c9",
    "guard": "#ebe7f3",
}


plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "mathtext.fontset": "dejavuserif",
    }
)


def add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    facecolor: str,
    *,
    edgecolor: str = "#111111",
    lw: float = 1.4,
    dashed: bool = False,
    fontsize: int = 10,
    round_size: float = 0.02,
    zorder: int = 2,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={round_size}",
        linewidth=lw,
        edgecolor=edgecolor,
        facecolor=facecolor,
        linestyle="--" if dashed else "-",
        zorder=zorder,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, zorder=zorder + 1)


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    lw: float = 1.4,
    style: str = "-|>",
    dashed: bool = False,
    rad: float = 0.0,
    color: str = "#111111",
    zorder: int = 3,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=12,
        linewidth=lw,
        linestyle="--" if dashed else "-",
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder,
    )
    ax.add_patch(arrow)


def add_label(ax, x: float, y: float, text: str, *, fontsize: int = 9, ha: str = "center", va: str = "center") -> None:
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color="#222222")


def render() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.6, 4.9), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(COLORS["paper"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Shared-depth boundary.
    add_box(
        ax,
        0.25,
        0.14,
        0.67,
        0.70,
        "",
        facecolor="none",
        dashed=True,
        lw=1.2,
        round_size=0.015,
        zorder=1,
    )
    add_label(ax, 0.27, 0.82, "Shared depth step $\\ell$ (repeat $K$ times)", fontsize=11, ha="left")

    # Inputs and controllers.
    add_box(ax, 0.04, 0.47, 0.12, 0.10, "Input token\n$x_t$", COLORS["neutral"])
    add_box(ax, 0.18, 0.47, 0.15, 0.10, "Token projections\n$q_t, k_t, v_t, d_t$", COLORS["neutral"], fontsize=9)
    add_box(ax, 0.69, 0.71, 0.15, 0.09, "DeepEmbed\nmodulator $m_t$", COLORS["accent"], fontsize=9, dashed=True)
    add_box(ax, 0.18, 0.25, 0.15, 0.09, "UT step embed.\n$e_\\ell$", COLORS["accent"], fontsize=9, dashed=True)

    # Main recurrent core.
    add_box(
        ax,
        0.38,
        0.39,
        0.27,
        0.23,
        "Matrix-state update\n"
        "$S_t = \\rho_t \\odot S_{t-1} + (1-\\rho_t) \\odot \\sum_i \\pi_{t,i} U_t^i$\n"
        "$U_t^i \\in \\{k_t v_t^\\top,\\ \\mathrm{Diag}(d_t),\\ \\mathrm{Sym}(k_t v_t^\\top)\\}$",
        COLORS["matrix"],
        fontsize=8,
    )
    add_label(ax, 0.515, 0.65, "RuleMix + retention", fontsize=9)
    add_box(ax, 0.42, 0.22, 0.20, 0.08, "Spectral guard / clamp", COLORS["guard"], fontsize=9, dashed=True)
    add_box(ax, 0.70, 0.50, 0.14, 0.09, "State readout\n$y_t = W_o\\,\\mathrm{vec}(S_t q_t)$", COLORS["neutral"], fontsize=9)
    add_box(ax, 0.70, 0.37, 0.14, 0.09, "Shared FFN\n$h_t = \\mathrm{FFN}(y_t) \\odot (1+m_t)$", COLORS["warm"], fontsize=9)

    # Residual side memory and output.
    add_box(ax, 0.70, 0.20, 0.14, 0.08, "Residual memory\n$r_t$", COLORS["accent"], fontsize=9, dashed=True)
    add_box(ax, 0.80, 0.62, 0.12, 0.08, "Output\n$z_t$", COLORS["neutral"], fontsize=9)
    plus = Circle((0.86, 0.43), 0.023, edgecolor="#111111", facecolor="white", linewidth=1.4, zorder=4)
    ax.add_patch(plus)
    add_label(ax, 0.86, 0.43, "+", fontsize=13)

    # Core flow.
    add_arrow(ax, (0.16, 0.52), (0.18, 0.52))
    add_arrow(ax, (0.33, 0.52), (0.38, 0.52))
    add_arrow(ax, (0.65, 0.545), (0.70, 0.545))
    add_arrow(ax, (0.77, 0.50), (0.77, 0.46))
    add_arrow(ax, (0.86, 0.453), (0.86, 0.62), rad=0.0)
    add_arrow(ax, (0.84, 0.415), (0.837, 0.43))

    # Modulation and control paths.
    add_arrow(ax, (0.765, 0.71), (0.765, 0.46), dashed=True)
    add_arrow(ax, (0.33, 0.295), (0.43, 0.39), dashed=True)
    add_arrow(ax, (0.33, 0.295), (0.70, 0.405), dashed=True, rad=-0.05)
    add_arrow(ax, (0.52, 0.39), (0.52, 0.30), dashed=True)

    # Residual and memory paths.
    add_arrow(ax, (0.10, 0.47), (0.10, 0.12))
    add_arrow(ax, (0.10, 0.12), (0.88, 0.12))
    add_arrow(ax, (0.88, 0.12), (0.88, 0.407))
    add_arrow(ax, (0.77, 0.28), (0.845, 0.42), dashed=True)

    # Shared-depth recurrence cue.
    add_arrow(ax, (0.92, 0.76), (0.92, 0.22), rad=-0.50, lw=1.2)
    add_label(ax, 0.955, 0.50, "repeat\nshared\nweights", fontsize=9, ha="center")

    fig.savefig(OUT_PNG, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


if __name__ == "__main__":
    render()
