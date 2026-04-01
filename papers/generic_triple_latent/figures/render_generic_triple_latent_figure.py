from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "papers" / "associative_state_universal_transformers" / "figures"
PDF_PATH = OUT_DIR / "generic_triple_latent_overview.pdf"
PNG_PATH = OUT_DIR / "generic_triple_latent_overview.png"


def add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    *,
    subtitle: str = "",
    fc: str,
    ec: str = "#22313F",
    title_size: int = 11,
    subtitle_size: int = 9,
    radius: float = 0.03,
    pad: float = 0.0048,
    lw: float = 1.5,
    zorder: float = 3.0,
) -> None:
    shadow = FancyBboxPatch(
        (x + 0.0032, y - 0.0032),
        w,
        h,
        boxstyle=f"round,pad={pad},rounding_size={radius}",
        linewidth=0,
        facecolor="#c9d4df",
        alpha=0.15,
        zorder=zorder - 1.0,
    )
    ax.add_patch(shadow)

    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad={pad},rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=zorder,
    )
    ax.add_patch(box)

    ax.text(
        x + w / 2.0,
        y + h * (0.61 if subtitle else 0.50),
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=ec,
        zorder=zorder + 1.0,
    )
    if subtitle:
        ax.text(
            x + w / 2.0,
            y + h * 0.30,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color=ec,
            zorder=zorder + 1.0,
        )


def add_stack_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    *,
    subtitle: str = "",
    fc: str,
    ec: str = "#22313F",
    title_size: int = 11,
    subtitle_size: int = 9,
    radius: float = 0.03,
    pad: float = 0.0048,
    lw: float = 1.5,
    zorder: float = 3.0,
) -> None:
    for layer, alpha in ((2, 0.14), (1, 0.24)):
        back = FancyBboxPatch(
            (x + layer * 0.007, y + layer * 0.007),
            w,
            h,
            boxstyle=f"round,pad={pad},rounding_size={radius}",
            linewidth=1.0,
            edgecolor=ec,
            facecolor=fc,
            alpha=alpha,
            zorder=zorder - 0.5 * layer,
        )
        ax.add_patch(back)

    add_box(
        ax,
        x,
        y,
        w,
        h,
        title,
        subtitle=subtitle,
        fc=fc,
        ec=ec,
        title_size=title_size,
        subtitle_size=subtitle_size,
        radius=radius,
        pad=pad,
        lw=lw,
        zorder=zorder,
    )


def add_diamond(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    *,
    subtitle: str = "",
    fc: str,
    ec: str = "#22313F",
    title_size: int = 10,
    subtitle_size: int = 8,
    lw: float = 1.5,
    zorder: float = 3.0,
) -> None:
    shadow_pts = [
        (x + w / 2.0 + 0.0032, y + h + 0.0005),
        (x + w + 0.0032, y + h / 2.0 - 0.0027),
        (x + w / 2.0 + 0.0032, y - 0.0032),
        (x + 0.0032, y + h / 2.0 - 0.0032),
    ]
    ax.add_patch(
        Polygon(
            shadow_pts,
            closed=True,
            linewidth=0,
            facecolor="#c9d4df",
            alpha=0.15,
            zorder=zorder - 1.0,
        )
    )

    pts = [
        (x + w / 2.0, y + h),
        (x + w, y + h / 2.0),
        (x + w / 2.0, y),
        (x, y + h / 2.0),
    ]
    ax.add_patch(
        Polygon(
            pts,
            closed=True,
            linewidth=lw,
            edgecolor=ec,
            facecolor=fc,
            zorder=zorder,
            joinstyle="round",
        )
    )

    ax.text(
        x + w / 2.0,
        y + h * (0.58 if subtitle else 0.50),
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=ec,
        zorder=zorder + 1.0,
    )
    if subtitle:
        ax.text(
            x + w / 2.0,
            y + h * 0.34,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color=ec,
            zorder=zorder + 1.0,
        )


def add_panel(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    *,
    fc: str,
    ec: str = "#22313F",
    zorder: float = 1.0,
) -> None:
    panel = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.035",
        linewidth=1.6,
        edgecolor=ec,
        facecolor=fc,
        zorder=zorder,
    )
    ax.add_patch(panel)
    ax.text(
        x + 0.02,
        y + h - 0.045,
        title,
        ha="left",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=ec,
        zorder=zorder + 1.0,
    )


def add_chip(ax, x: float, y: float, text: str, *, fc: str, ec: str = "#22313F") -> None:
    chip = FancyBboxPatch(
        (x, y),
        0.12,
        0.038,
        boxstyle="round,pad=0.0048,rounding_size=0.02",
        linewidth=1.1,
        edgecolor=ec,
        facecolor=fc,
        zorder=4.0,
    )
    ax.add_patch(chip)
    ax.text(x + 0.06, y + 0.019, text, ha="center", va="center", fontsize=8.8, color=ec, zorder=5.0)


def add_arrow(
    ax,
    start,
    end,
    *,
    color: str = "#22313F",
    lw: float = 1.7,
    style: str = "-|>",
    mutation: float = 14.0,
    connectionstyle: str = "arc3",
    shrink_a: float = 4.0,
    shrink_b: float = 4.0,
    zorder: float = 2.0,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=shrink_a,
        shrinkB=shrink_b,
        zorder=zorder,
    )
    ax.add_patch(arrow)


def add_label(ax, x: float, y: float, text: str, *, size: int = 9, color: str = "#22313F", ha: str = "left") -> None:
    ax.text(x, y, text, ha=ha, va="center", fontsize=size, color=color, zorder=5.0)


def draw() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    fig, ax = plt.subplots(figsize=(14.8, 8.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    bg = "#f8f4ee"
    ink = "#22313F"
    navy = "#dfe9f7"
    warm = "#ffe7cf"
    mint = "#e3f5ea"
    sea = "#dff4f1"
    gold = "#fff1bf"
    coral = "#f9d9cf"
    sand = "#f6efe4"
    panel = "#fffaf4"

    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    ax.text(0.05, 0.962, "Triple-Latent Compression + Gated Retrieval", fontsize=20, fontweight="bold", color=ink, ha="left", va="top")
    ax.text(
        0.05,
        0.918,
        "A clean separation between lossy recurrent compression and exact associative lookup.",
        fontsize=11.2,
        color=ink,
        ha="left",
        va="top",
    )

    add_chip(ax, 0.05, 0.835, "token stream", fc=navy)
    add_chip(ax, 0.215, 0.835, "compression", fc=warm)
    add_chip(ax, 0.380, 0.835, "retrieval", fc=sea)

    add_box(ax, 0.05, 0.696, 0.12, 0.083, "Byte Tokens", subtitle="$x_{1:T}$", fc=gold, title_size=11, subtitle_size=10)
    add_box(ax, 0.22, 0.696, 0.15, 0.083, "Token + Pos", subtitle="shared features", fc=navy, title_size=11, subtitle_size=9)
    add_box(ax, 0.42, 0.696, 0.13, 0.083, "Conv Mix", subtitle="optional local path", fc=sea, title_size=11, subtitle_size=9)
    add_box(ax, 0.60, 0.679, 0.17, 0.118, "Triple-Latent Stack", subtitle="repeat x L layers", fc=coral, title_size=12, subtitle_size=9)
    add_box(ax, 0.82, 0.696, 0.14, 0.083, "Final Norm + LM Head", subtitle="tied output logits", fc=navy, title_size=10, subtitle_size=8.6)

    add_arrow(ax, (0.17, 0.738), (0.22, 0.738))
    add_arrow(ax, (0.37, 0.738), (0.42, 0.738))
    add_arrow(ax, (0.55, 0.738), (0.60, 0.738))
    add_arrow(ax, (0.77, 0.738), (0.82, 0.738))

    add_panel(ax, 0.05, 0.12, 0.51, 0.46, "Base Triple-Latent Compression", fc=panel)
    add_panel(ax, 0.60, 0.12, 0.35, 0.46, "Recall-Focused Gated Retrieval", fc=panel)

    add_box(ax, 0.09, 0.355, 0.10, 0.080, "LayerNorm", subtitle="shared stream", fc=navy, title_size=10, subtitle_size=8)
    add_box(ax, 0.23, 0.407, 0.14, 0.086, "Write Proj", subtitle="$a_t, b_t$", fc=warm, title_size=10, subtitle_size=9)
    add_box(ax, 0.23, 0.232, 0.14, 0.086, "Read Proj", subtitle="$q_t^\\ell, q_t^r$", fc=warm, title_size=10, subtitle_size=9)
    add_box(ax, 0.42, 0.407, 0.12, 0.086, "Running State", subtitle="$s_t$", fc=mint, title_size=10, subtitle_size=10)
    add_stack_box(ax, 0.42, 0.232, 0.12, 0.086, "Pair Memory", subtitle="$P_t$ or slots", fc=mint, title_size=10, subtitle_size=9)
    add_box(ax, 0.31, 0.141, 0.18, 0.068, "Bilinear Readout", subtitle="$(P_t q_t^\\ell) \\odot q_t^r$", fc=sand, title_size=10, subtitle_size=8)
    add_box(ax, 0.09, 0.152, 0.15, 0.086, "Residual + FFN", subtitle="next layer stream", fc=sea, title_size=10, subtitle_size=8)

    add_arrow(ax, (0.19, 0.395), (0.23, 0.447))
    add_arrow(ax, (0.19, 0.395), (0.23, 0.272))
    add_arrow(ax, (0.37, 0.450), (0.42, 0.450))
    add_arrow(ax, (0.37, 0.275), (0.42, 0.275))
    add_arrow(ax, (0.48, 0.400), (0.48, 0.340), connectionstyle="arc3")
    add_arrow(ax, (0.48, 0.340), (0.48, 0.225), connectionstyle="arc3")
    add_arrow(ax, (0.42, 0.275), (0.49, 0.215), connectionstyle="arc3,rad=-0.08")
    add_arrow(ax, (0.31, 0.175), (0.24, 0.190), connectionstyle="arc3,rad=0.0")
    add_arrow(ax, (0.165, 0.245), (0.165, 0.350), connectionstyle="arc3")

    add_label(ax, 0.49, 0.320, "update uses previous $s_{t-1}$ and current $b_t$", size=8.7)
    add_label(ax, 0.09, 0.485, "lossy recurrent compression", size=9.1)
    add_label(ax, 0.09, 0.135, "dense / slot memory stays inside the recurrent path", size=8.8)

    add_box(ax, 0.64, 0.407, 0.13, 0.086, "Key / Value Write", subtitle="$(k_{t-1}, v_t)$", fc=sea, title_size=10, subtitle_size=9)
    add_box(ax, 0.64, 0.237, 0.13, 0.086, "Previous Query", subtitle="$q_t = W_q e_{t-1}$", fc=navy, title_size=10, subtitle_size=8)
    add_box(ax, 0.82, 0.344, 0.12, 0.102, "Top-k Lookup", subtitle="normalized key match", fc=mint, title_size=10, subtitle_size=8)
    add_diamond(ax, 0.64, 0.145, 0.085, 0.085, "Read Gate", subtitle="$g_t$", fc=coral, title_size=9.0, subtitle_size=8.8)
    add_box(ax, 0.73, 0.131, 0.16, 0.068, "Late Logit Fusion", subtitle="add retrieval only at output", fc=sand, title_size=10, subtitle_size=8)

    add_arrow(ax, (0.77, 0.450), (0.82, 0.395))
    add_arrow(ax, (0.77, 0.280), (0.82, 0.365))
    add_arrow(ax, (0.88, 0.344), (0.85, 0.199), connectionstyle="arc3,rad=0.02")
    add_arrow(ax, (0.705, 0.237), (0.6825, 0.230), connectionstyle="arc3,rad=0.0")
    add_arrow(ax, (0.725, 0.1875), (0.73, 0.165), connectionstyle="arc3,rad=0.0")

    add_arrow(ax, (0.295, 0.690), (0.295, 0.545), connectionstyle="arc3")
    add_arrow(ax, (0.295, 0.545), (0.64, 0.450), connectionstyle="arc3,rad=-0.10")
    add_arrow(ax, (0.305, 0.690), (0.305, 0.345), connectionstyle="arc3")
    add_arrow(ax, (0.305, 0.345), (0.64, 0.280), connectionstyle="arc3,rad=0.10")
    add_arrow(ax, (0.85, 0.165), (0.89, 0.165))
    add_arrow(ax, (0.89, 0.165), (0.89, 0.696), connectionstyle="arc3,rad=0.0")

    add_label(ax, 0.645, 0.115, "best current recipe: previous-token query + last-layer retrieval + gated late fusion", size=8.7)

    ax.text(
        0.05,
        0.06,
        "Compression path handles generic sequence modeling; the retrieval branch is invoked only when exact key-value access is useful.",
        fontsize=10.7,
        color=ink,
        ha="left",
        va="center",
    )

    fig.tight_layout()
    fig.savefig(PDF_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    draw()
