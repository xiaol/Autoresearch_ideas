from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "arch")


COLORS = {
    "neutral": "#f2f2f2",
    "matrix": "#dbe7f6",
    "ffn": "#f6e3c6",
    "accent": "#e8f4e8",
    "attention": "#cfe0f7",
}


def _add_box(ax, x, y, w, h, text, fc, ec="black", lw=1.4, fontsize=10, dashed=False):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(box)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def _add_circle(ax, x, y, r, text, lw=1.4, fontsize=10):
    circ = Circle((x, y), r, edgecolor="black", facecolor="white", linewidth=lw)
    ax.add_patch(circ)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize)


def _add_arrow(ax, x1, y1, x2, y2, lw=1.2):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, linewidth=lw, color="black")
    ax.add_patch(arr)


def _add_curve(ax, x1, y1, x2, y2, rad=0.3, lw=1.1):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=lw,
        color="black",
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)


def _panel_base(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def _draw_ut_panel(ax, mixer_label, mixer_color, variant=None):
    # Embedding and output
    _add_box(ax, 0.18, 0.10, 0.64, 0.08, "Embedding", COLORS["neutral"])
    _add_box(ax, 0.18, 0.82, 0.64, 0.08, "Output", COLORS["neutral"])

    # Recurrent depth boundary
    _add_box(ax, 0.08, 0.22, 0.84, 0.56, "", "none", dashed=True, lw=1.0)
    ax.text(0.10, 0.76, "Recurrent Depth (shared)", ha="left", va="center", fontsize=8)

    # Core blocks
    _add_box(ax, 0.22, 0.34, 0.56, 0.12, mixer_label, mixer_color, fontsize=9)
    _add_box(ax, 0.22, 0.52, 0.56, 0.12, "FFN", COLORS["ffn"], fontsize=9)
    _add_circle(ax, 0.50, 0.70, 0.03, "+", fontsize=11)

    # Flow arrows
    _add_arrow(ax, 0.50, 0.18, 0.50, 0.34)
    _add_arrow(ax, 0.50, 0.46, 0.50, 0.52)
    _add_arrow(ax, 0.50, 0.64, 0.50, 0.67)
    _add_arrow(ax, 0.50, 0.73, 0.50, 0.82)

    # Residual path from embedding to add
    _add_arrow(ax, 0.20, 0.18, 0.12, 0.18)
    _add_arrow(ax, 0.12, 0.18, 0.12, 0.70)
    _add_arrow(ax, 0.12, 0.70, 0.47, 0.70)

    # Recurrent loop arrow
    _add_curve(ax, 0.90, 0.74, 0.90, 0.26, rad=-0.5, lw=1.0)
    ax.text(0.92, 0.50, "repeat K", rotation=-90, ha="center", va="center", fontsize=8)

    # Variant-specific decorations
    if variant == "dynamic":
        _add_box(ax, 0.02, 0.36, 0.16, 0.08, "Timescale", COLORS["accent"], fontsize=8)
        _add_arrow(ax, 0.18, 0.40, 0.22, 0.40)
    elif variant == "rosa":
        _add_box(ax, 0.78, 0.60, 0.16, 0.08, "ROSA", COLORS["accent"], fontsize=8)
        _add_arrow(ax, 0.78, 0.62, 0.53, 0.70)
    elif variant == "deepembed":
        _add_box(ax, 0.78, 0.52, 0.16, 0.08, "DeepEmbed", COLORS["accent"], fontsize=8)
        _add_arrow(ax, 0.78, 0.56, 0.76, 0.56)
    elif variant == "structured":
        _add_box(ax, 0.26, 0.36, 0.24, 0.08, "Low-Rank", COLORS["accent"], fontsize=7)
        _add_box(ax, 0.52, 0.36, 0.24, 0.08, "Diagonal", COLORS["accent"], fontsize=7)
    elif variant == "hybrid":
        _add_box(ax, 0.22, 0.44, 0.56, 0.10, "Attention", COLORS["attention"], fontsize=8)
    elif variant == "dual":
        _add_box(ax, 0.24, 0.36, 0.24, 0.08, "Fast", COLORS["accent"], fontsize=8)
        _add_box(ax, 0.52, 0.36, 0.24, 0.08, "Slow", COLORS["accent"], fontsize=8)
        _add_box(ax, 0.38, 0.46, 0.24, 0.05, "Mix", COLORS["neutral"], fontsize=7)
    elif variant == "rulemix":
        _add_box(ax, 0.02, 0.52, 0.16, 0.08, "RuleMix", COLORS["accent"], fontsize=8)
        _add_arrow(ax, 0.18, 0.56, 0.22, 0.56)
        _add_box(ax, 0.02, 0.40, 0.08, 0.05, "Δ1", COLORS["neutral"], fontsize=7)
        _add_box(ax, 0.10, 0.40, 0.08, 0.05, "Δ2", COLORS["neutral"], fontsize=7)
        _add_box(ax, 0.06, 0.34, 0.08, 0.05, "Δ3", COLORS["neutral"], fontsize=7)
    elif variant == "skewstable":
        _add_box(ax, 0.78, 0.36, 0.16, 0.08, "Skew Update", COLORS["accent"], fontsize=7)
        _add_box(ax, 0.78, 0.46, 0.16, 0.06, "Stable", COLORS["neutral"], fontsize=7)
        _add_arrow(ax, 0.78, 0.40, 0.76, 0.40)
    elif variant == "convmix":
        _add_box(ax, 0.02, 0.56, 0.16, 0.08, "Local Conv", COLORS["accent"], fontsize=8)
        _add_box(ax, 0.02, 0.44, 0.16, 0.08, "Mix Gate", COLORS["neutral"], fontsize=8)
        _add_arrow(ax, 0.18, 0.56, 0.22, 0.56)
    elif variant == "stepcond":
        _add_box(ax, 0.78, 0.60, 0.16, 0.08, "Step Emb", COLORS["accent"], fontsize=8)
        _add_box(ax, 0.78, 0.50, 0.16, 0.06, "Step Gate", COLORS["neutral"], fontsize=8)
        _add_arrow(ax, 0.78, 0.60, 0.74, 0.60)
    elif variant == "spectral":
        _add_box(ax, 0.78, 0.36, 0.16, 0.08, "EigenClamp", COLORS["accent"], fontsize=7)
        _add_box(ax, 0.78, 0.46, 0.16, 0.06, "Spec Reg", COLORS["neutral"], fontsize=7)
        _add_arrow(ax, 0.78, 0.40, 0.76, 0.40)
    elif variant == "discovery":
        _add_box(ax, 0.02, 0.52, 0.16, 0.08, "RuleMix", COLORS["accent"], fontsize=8)
        _add_box(ax, 0.78, 0.60, 0.16, 0.08, "ROSA", COLORS["accent"], fontsize=8)
        _add_box(ax, 0.78, 0.50, 0.16, 0.08, "DeepEmbed", COLORS["accent"], fontsize=7)
        _add_box(ax, 0.02, 0.36, 0.16, 0.08, "Timescale", COLORS["accent"], fontsize=7)
        _add_box(ax, 0.78, 0.36, 0.16, 0.08, "EigenClamp", COLORS["neutral"], fontsize=7)
        _add_arrow(ax, 0.18, 0.56, 0.22, 0.56)
        _add_arrow(ax, 0.78, 0.62, 0.53, 0.70)
        _add_arrow(ax, 0.78, 0.54, 0.76, 0.54)
    elif variant == "sparsepointer":
        _add_box(ax, 0.02, 0.56, 0.16, 0.08, "Write Gate", COLORS["accent"], fontsize=8)
        _add_box(ax, 0.02, 0.44, 0.16, 0.08, "LRU / Evict", COLORS["neutral"], fontsize=8)
        _add_box(ax, 0.78, 0.60, 0.16, 0.08, "Top-k Slots", COLORS["accent"], fontsize=8)
        _add_box(ax, 0.78, 0.48, 0.16, 0.08, "Pointer Fuse", COLORS["neutral"], fontsize=8)
        _add_arrow(ax, 0.18, 0.60, 0.22, 0.60)
        _add_arrow(ax, 0.18, 0.48, 0.22, 0.40)
        _add_arrow(ax, 0.78, 0.62, 0.53, 0.70)
        _add_arrow(ax, 0.78, 0.52, 0.76, 0.52)
    elif variant == "productkey":
        _add_box(ax, 0.02, 0.58, 0.16, 0.08, "Query $q^A$", COLORS["accent"], fontsize=8)
        _add_box(ax, 0.02, 0.46, 0.16, 0.08, "Query $q^B$", COLORS["neutral"], fontsize=8)
        _add_box(ax, 0.78, 0.60, 0.16, 0.08, "Codebook A", COLORS["accent"], fontsize=8)
        _add_box(ax, 0.78, 0.48, 0.16, 0.08, "Codebook B", COLORS["accent"], fontsize=8)
        _add_box(ax, 0.78, 0.36, 0.16, 0.08, "Bucket Pair", COLORS["neutral"], fontsize=8)
        _add_arrow(ax, 0.18, 0.62, 0.22, 0.56)
        _add_arrow(ax, 0.18, 0.50, 0.22, 0.40)
        _add_arrow(ax, 0.78, 0.62, 0.53, 0.70)
        _add_arrow(ax, 0.78, 0.50, 0.53, 0.70)
        _add_arrow(ax, 0.78, 0.40, 0.76, 0.40)
    elif variant == "relay":
        _add_box(ax, 0.02, 0.56, 0.16, 0.08, "Hop 1", COLORS["accent"], fontsize=8)
        _add_box(ax, 0.02, 0.44, 0.16, 0.08, "Hop 2", COLORS["neutral"], fontsize=8)
        _add_box(ax, 0.78, 0.60, 0.16, 0.08, "Anchor Mem", COLORS["accent"], fontsize=8)
        _add_box(ax, 0.78, 0.48, 0.16, 0.08, "Relay Link", COLORS["neutral"], fontsize=8)
        _add_box(ax, 0.78, 0.36, 0.16, 0.08, "Value Read", COLORS["accent"], fontsize=8)
        _add_arrow(ax, 0.18, 0.60, 0.22, 0.56)
        _add_arrow(ax, 0.18, 0.48, 0.22, 0.40)
        _add_arrow(ax, 0.78, 0.62, 0.53, 0.70)
        _add_arrow(ax, 0.78, 0.50, 0.53, 0.70)
        _add_arrow(ax, 0.78, 0.40, 0.76, 0.40)


def _draw_triptych(path, title, variant):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=150)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    label_map = {
        "dynamic": "Dynamic",
        "rosa": "ROSA",
        "deepembed": "DeepEmbed",
        "structured": "Structured",
        "hybrid": "Hybrid",
        "dual": "DualTimescale",
        "rulemix": "RuleMix",
        "skewstable": "SkewStable",
        "convmix": "ConvMix",
        "stepcond": "StepConditioned",
        "spectral": "Spectral",
        "discovery": "Discovery",
        "sparsepointer": "SparsePointer",
        "productkey": "ProductKey",
        "relay": "Relay",
    }

    _panel_base(axes[0])
    _draw_ut_panel(axes[0], "Self-Attention", COLORS["attention"])
    axes[0].text(0.5, 0.02, "(a) Universal Transformer", ha="center", va="center", fontsize=10)

    _panel_base(axes[1])
    _draw_ut_panel(axes[1], "Matrix State", COLORS["matrix"])
    axes[1].text(0.5, 0.02, "(b) UniMatrix Core", ha="center", va="center", fontsize=10)

    _panel_base(axes[2])
    _draw_ut_panel(axes[2], "Matrix State", COLORS["matrix"], variant=variant)
    axes[2].text(0.5, 0.02, f"(c) UniMatrix-{label_map[variant]}", ha="center", va="center", fontsize=10)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    _draw_triptych(os.path.join(OUT_DIR, "umt_dynamic.png"), "UniMatrix-Dynamic (Per-Head Timescale)", "dynamic")
    _draw_triptych(os.path.join(OUT_DIR, "umt_rosa.png"), "UniMatrix-ROSA (Suffix Memory)", "rosa")
    _draw_triptych(os.path.join(OUT_DIR, "umt_deepembed.png"), "UniMatrix-DeepEmbed (Token Modulation)", "deepembed")
    _draw_triptych(os.path.join(OUT_DIR, "umt_structured.png"), "UniMatrix-Structured (Low-Rank + Diag)", "structured")
    _draw_triptych(os.path.join(OUT_DIR, "umt_hybrid.png"), "UniMatrix-Hybrid (Interleaved Attention)", "hybrid")
    _draw_triptych(os.path.join(OUT_DIR, "umt_dual.png"), "UniMatrix-DualTimescale (Fast + Slow)", "dual")
    _draw_triptych(os.path.join(OUT_DIR, "umt_rulemix.png"), "UniMatrix-RuleMix (Hybrid Update Rules)", "rulemix")
    _draw_triptych(os.path.join(OUT_DIR, "umt_skewstable.png"), "UniMatrix-SkewStable (Skew-Symmetric Update)", "skewstable")
    _draw_triptych(os.path.join(OUT_DIR, "umt_convmix.png"), "UniMatrix-ConvMix (Local + Global Memory)", "convmix")
    _draw_triptych(os.path.join(OUT_DIR, "umt_stepcond.png"), "UniMatrix-StepConditioned (UT Step Gates)", "stepcond")
    _draw_triptych(os.path.join(OUT_DIR, "umt_spectral.png"), "UniMatrix-Spectral (Eigenvalue Control)", "spectral")
    _draw_triptych(os.path.join(OUT_DIR, "umt_discovery.png"), "UniMatrix-Discovery (Combined)", "discovery")
    _draw_triptych(os.path.join(OUT_DIR, "umt_sparsepointer.png"), "UniMatrix-SparsePointer (Sparse Slot Cache)", "sparsepointer")
    _draw_triptych(os.path.join(OUT_DIR, "umt_productkey.png"), "UniMatrix-ProductKey (Factorized Retrieval)", "productkey")
    _draw_triptych(os.path.join(OUT_DIR, "umt_relay.png"), "UniMatrix-Relay (Two-Hop Retrieval)", "relay")


if __name__ == "__main__":
    main()
