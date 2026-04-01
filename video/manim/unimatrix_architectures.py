from __future__ import annotations

from pathlib import Path
import re
import numpy as np
from manim import (
    Circle,
    ImageMobject,
    Line,
    Rectangle,
    RoundedRectangle,
    Scene,
    Text,
    MathTex,
    VGroup,
    SurroundingRectangle,
    WHITE,
    FadeIn,
    FadeOut,
    Write,
    Create,
    Transform,
    UP,
    DOWN,
    LEFT,
    RIGHT,
)


ROOT = Path(__file__).resolve().parents[2]
ARCH_DIR = ROOT / "assets" / "arch"


ARCHS = [
    {
        "class": "UniMatrixDynamic",
        "title": "UniMatrix-Dynamic",
        "subtitle": "Per-Head Timescale",
        "image": "umt_dynamic.png",
        "baseline": (
            "Universal Transformer",
            ["Shared parameters across depth", "Repeat K steps for refinement"],
            r"h^{(k+1)} = f_\theta(h^{(k)}, x)",
        ),
        "core": (
            "UniMatrix Core",
            ["Matrix-valued recurrence", "Linear-time update, no KV cache"],
            r"S_t = (1-g_t)\odot S_{t-1} + g_t \odot (u_t v_t^\top)",
        ),
        "variant": (
            "Dynamic Timescale",
            ["Per-head decay controls memory", "Different heads, different horizons"],
            r"S_t = e^{-\tau_h}\odot S_{t-1} + (1-e^{-\tau_h})\odot (u_t v_t^\top)",
        ),
    },
    {
        "class": "UniMatrixROSA",
        "title": "UniMatrix-ROSA",
        "subtitle": "Suffix Memory",
        "image": "umt_rosa.png",
        "baseline": (
            "Universal Transformer",
            ["Shared parameters across depth", "Repeat K steps for refinement"],
            r"h^{(k+1)} = f_\theta(h^{(k)}, x)",
        ),
        "core": (
            "UniMatrix Core",
            ["Matrix-state recurrence", "Attention-free linear-time core"],
            r"y_t = S_t q_t,\quad S_t\ \text{updated linearly}",
        ),
        "variant": (
            "ROSA Fusion",
            ["Inject suffix memory into residual", "Lossless long-range routing"],
            r"x_t' = x_t + W_r r_t,\quad r_t=\mathrm{ROSA}(x_{\le t})",
        ),
    },
    {
        "class": "UniMatrixDeepEmbed",
        "title": "UniMatrix-DeepEmbed",
        "subtitle": "Token Modulation",
        "image": "umt_deepembed.png",
        "baseline": (
            "Universal Transformer",
            ["Shared parameters across depth", "Repeat K steps for refinement"],
            r"h^{(k+1)} = f_\theta(h^{(k)}, x)",
        ),
        "core": ("UniMatrix Core", ["Matrix-state recurrence", "Attention-free linear-time core"], r"y_t = S_t q_t"),
        "variant": (
            "DeepEmbed Modulation",
            ["Token embedding gates FFN", "Improves rare-token routing"],
            r"\mathrm{FFN}(x) = W_2\,\sigma(W_1 x \odot (1+W_d d_t))",
        ),
    },
    {
        "class": "UniMatrixStructured",
        "title": "UniMatrix-Structured",
        "subtitle": "Low-Rank + Diagonal",
        "image": "umt_structured.png",
        "baseline": (
            "Universal Transformer",
            ["Shared parameters across depth", "Repeat K steps for refinement"],
            r"h^{(k+1)} = f_\theta(h^{(k)}, x)",
        ),
        "core": ("UniMatrix Core", ["Matrix-state recurrence", "Attention-free linear-time core"], r"y_t = S_t q_t"),
        "variant": (
            "Structured State",
            ["Low-rank + diagonal factorization", "More memory at lower cost"],
            r"S_t = L_t R_t^\top + \mathrm{diag}(d_t)",
        ),
    },
    {
        "class": "UniMatrixHybrid",
        "title": "UniMatrix-Hybrid",
        "subtitle": "Interleaved Attention",
        "image": "umt_hybrid.png",
        "baseline": (
            "Universal Transformer",
            ["Shared parameters across depth", "Repeat K steps for refinement"],
            r"h^{(k+1)} = f_\theta(h^{(k)}, x)",
        ),
        "core": ("UniMatrix Core", ["Matrix-state recurrence", "Attention-free linear-time core"], r"y_t = S_t q_t"),
        "variant": (
            "Hybrid Step",
            ["Occasional attention steps", "Recover hard dependencies"],
            r"h^{(k+1)} = f_\text{attn}(f_\text{mat}(h^{(k)}))",
        ),
    },
    {
        "class": "UniMatrixDualTimescale",
        "title": "UniMatrix-DualTimescale",
        "subtitle": "Fast + Slow States",
        "image": "umt_dual.png",
        "baseline": (
            "Universal Transformer",
            ["Shared parameters across depth", "Repeat K steps for refinement"],
            r"h^{(k+1)} = f_\theta(h^{(k)}, x)",
        ),
        "core": ("UniMatrix Core", ["Matrix-state recurrence", "Attention-free linear-time core"], r"y_t = S_t q_t"),
        "variant": (
            "Dual Timescale",
            ["Fast + slow state mixture", "Separates short and long memory"],
            r"S_t = \alpha S_t^\text{fast} + (1-\alpha) S_t^\text{slow}",
        ),
    },
    {
        "class": "UniMatrixRuleMix",
        "title": "UniMatrix-RuleMix",
        "subtitle": "Hybrid Update Rules",
        "image": "umt_rulemix.png",
        "baseline": (
            "Universal Transformer",
            ["Shared parameters across depth", "Repeat K steps for refinement"],
            r"h^{(k+1)} = f_\theta(h^{(k)}, x)",
        ),
        "core": ("UniMatrix Core", ["Matrix-state recurrence", "Attention-free linear-time core"], r"y_t = S_t q_t"),
        "variant": (
            "RuleMix",
            ["Mixture over update rules", "Model discovers best rule"],
            r"\Delta_t=\sum_i \alpha_i(x_t)\Delta_i,\quad S_t=S_{t-1}+\Delta_t",
        ),
    },
    {
        "class": "UniMatrixSkewStable",
        "title": "UniMatrix-SkewStable",
        "subtitle": "Skew-Symmetric Update",
        "image": "umt_skewstable.png",
        "baseline": (
            "Universal Transformer",
            ["Shared parameters across depth", "Repeat K steps for refinement"],
            r"h^{(k+1)} = f_\theta(h^{(k)}, x)",
        ),
        "core": ("UniMatrix Core", ["Matrix-state recurrence", "Attention-free linear-time core"], r"y_t = S_t q_t"),
        "variant": (
            "Skew-Stable",
            ["Skew-symmetric update", "Eigenvalues near imaginary axis"],
            r"K_t = A_t - A_t^\top,\quad S_t = (I+\tau K_t)S_{t-1} + U_t V_t^\top",
        ),
    },
    {
        "class": "UniMatrixConvMix",
        "title": "UniMatrix-ConvMix",
        "subtitle": "Local + Global Memory",
        "image": "umt_convmix.png",
        "baseline": (
            "Universal Transformer",
            ["Shared parameters across depth", "Repeat K steps for refinement"],
            r"h^{(k+1)} = f_\theta(h^{(k)}, x)",
        ),
        "core": ("UniMatrix Core", ["Matrix-state recurrence", "Attention-free linear-time core"], r"y_t = S_t q_t"),
        "variant": (
            "ConvMix",
            ["Local conv + global matrix", "Gate controls blend"],
            r"\Delta_t = \lambda\,\Delta_\text{mat} + (1-\lambda)\,\Delta_\text{conv}",
        ),
    },
    {
        "class": "UniMatrixStepConditioned",
        "title": "UniMatrix-StepConditioned",
        "subtitle": "UT Step Gates",
        "image": "umt_stepcond.png",
        "baseline": (
            "Universal Transformer",
            ["Shared parameters across depth", "Repeat K steps for refinement"],
            r"h^{(k+1)} = f_\theta(h^{(k)}, x)",
        ),
        "core": ("UniMatrix Core", ["Matrix-state recurrence", "Attention-free linear-time core"], r"y_t = S_t q_t"),
        "variant": (
            "Step Conditioned",
            ["Step-index gate g(k)", "Early refine, later consolidate"],
            r"S_t = g(k)\odot S_{t-1} + (1-g(k))\odot (u_t v_t^\top)",
        ),
    },
    {
        "class": "UniMatrixSpectral",
        "title": "UniMatrix-Spectral",
        "subtitle": "Eigenvalue Control",
        "image": "umt_spectral.png",
        "baseline": (
            "Universal Transformer",
            ["Shared parameters across depth", "Repeat K steps for refinement"],
            r"h^{(k+1)} = f_\theta(h^{(k)}, x)",
        ),
        "core": ("UniMatrix Core", ["Matrix-state recurrence", "Attention-free linear-time core"], r"y_t = S_t q_t"),
        "variant": (
            "Spectral Control",
            ["Eigenvalue radius constraint", "Stability over long contexts"],
            r"\rho(S_t)\le \rho_\text{max}\quad\text{or}\quad \mathcal{L}_\text{spec}=\max(0,\rho-\rho_\text{max})",
        ),
    },
    {
        "class": "UniMatrixDiscovery",
        "title": "UniMatrix-Discovery",
        "subtitle": "Combined Mechanisms",
        "image": "umt_discovery.png",
        "baseline": (
            "Universal Transformer",
            ["Shared parameters across depth", "Repeat K steps for refinement"],
            r"h^{(k+1)} = f_\theta(h^{(k)}, x)",
        ),
        "core": ("UniMatrix Core", ["Matrix-state recurrence", "Attention-free linear-time core"], r"y_t = S_t q_t"),
        "variant": (
            "Discovery",
            ["RuleMix + ROSA + DeepEmbed", "Spectral guard for stability"],
            r"S_t=\mathrm{Spec}(\mathrm{RuleMix}(S_{t-1}))\ ,\ x_t' = x_t + W_r r_t",
        ),
    },
]


def _panel_rect(image: ImageMobject, index: int) -> Rectangle:
    panel_w = image.width / 3.0
    center_x = image.get_left()[0] + panel_w * (index + 0.5)
    center = np.array([center_x, image.get_center()[1], 0.0])
    rect = Rectangle(width=panel_w * 0.95, height=image.height * 0.85)
    rect.move_to(center)
    return rect


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return slug.strip("_")


def _build_scene(cfg: dict):
    class _Scene(Scene):
        def construct(self):
            self.camera.background_color = WHITE
            header = Text(f"{cfg['title']} — {cfg['subtitle']}", font_size=40, color="#000000")
            header.to_edge(UP)

            image_path = ARCH_DIR / cfg["image"]
            img = ImageMobject(str(image_path))
            img.scale_to_fit_width(9.4)
            img.next_to(header, DOWN, buff=0.2)
            img.to_edge(LEFT, buff=0.5)

            card = RoundedRectangle(corner_radius=0.08, width=img.width + 0.4, height=img.height + 0.4)
            card.set_stroke(color="#111111", width=1.2)
            card.set_fill(color="#ffffff", opacity=1.0)
            card.move_to(img)

            shadow = RoundedRectangle(corner_radius=0.08, width=card.width, height=card.height)
            shadow.set_fill(color="#000000", opacity=0.08)
            shadow.set_stroke(width=0)
            shadow.shift(DOWN * 0.06 + RIGHT * 0.06)
            shadow.move_to(card)

            self.play(Write(header))
            self.play(FadeIn(shadow), FadeIn(card), FadeIn(img, shift=0.1), run_time=0.8)
            self.wait(0.3)

            def callout(title_text, bullet_list, eq_text):
                title = Text(title_text, font_size=26, color="#000000")
                bullets = VGroup(
                    *[Text(f"• {b}", font_size=20, color="#333333") for b in bullet_list]
                ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
                eq = MathTex(eq_text, font_size=28, color="#000000")
                group = VGroup(title, bullets, eq).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
                box = RoundedRectangle(corner_radius=0.06, width=5.2, height=2.6)
                box.set_stroke(color="#111111", width=1.0)
                box.set_fill(color="#ffffff", opacity=1.0)
                box.move_to(group)
                group.move_to(box)
                call = VGroup(box, group)
                call.to_edge(RIGHT, buff=0.5)
                call.shift(DOWN * 0.1)
                return call

            def indicator(active_index):
                labels = ["Baseline", "Core", "Variant"]
                dots = VGroup()
                for i, label in enumerate(labels):
                    circ = Circle(radius=0.12)
                    if i == active_index:
                        circ.set_fill(color="#2C5AA0", opacity=1.0)
                        circ.set_stroke(color="#2C5AA0", width=2)
                    else:
                        circ.set_fill(color="#ffffff", opacity=1.0)
                        circ.set_stroke(color="#999999", width=1.2)
                    text = Text(label, font_size=16, color="#333333")
                    item = VGroup(circ, text).arrange(RIGHT, buff=0.15)
                    dots.add(item)
                dots.arrange(RIGHT, buff=0.6)
                dots.to_edge(DOWN, buff=0.35)
                return dots

            rect = _panel_rect(img, 0)
            call = callout(*cfg["baseline"])
            prog = indicator(0)
            self.play(Create(rect), FadeIn(call), FadeIn(prog), run_time=0.8)
            self.wait(2.6)
            self.play(FadeOut(call), run_time=0.4)

            self.play(Transform(rect, _panel_rect(img, 1)), run_time=0.6)
            call = callout(*cfg["core"])
            self.play(Transform(prog, indicator(1)), FadeIn(call), run_time=0.6)
            self.wait(2.6)
            self.play(FadeOut(call), run_time=0.4)

            self.play(Transform(rect, _panel_rect(img, 2)), run_time=0.6)
            call = callout(*cfg["variant"])
            self.play(Transform(prog, indicator(2)), FadeIn(call), run_time=0.6)
            self.wait(3.0)
            self.play(FadeOut(call), FadeOut(rect), FadeOut(prog), run_time=0.4)
            self.wait(0.4)

    _Scene.__name__ = cfg["class"]
    return _Scene


for cfg in ARCHS:
    globals()[cfg["class"]] = _build_scene(cfg)
