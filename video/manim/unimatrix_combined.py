from __future__ import annotations

from pathlib import Path
import json
import numpy as np
from manim import (
    Arrow,
    Dot,
    Circle,
    FadeIn,
    FadeOut,
    ImageMobject,
    LaggedStart,
    Text,
    Square,
    Rectangle,
    RoundedRectangle,
    Scene,
    VGroup,
    WHITE,
    config,
    Create,
    Transform,
    GrowArrow,
    Indicate,
    Line,
    MoveAlongPath,
    Succession,
    Write,
    UP,
    DOWN,
    LEFT,
    RIGHT,
)


ROOT = Path(__file__).resolve().parents[2]
ARCH_DIR = ROOT / "assets" / "arch"
FONT = "Avenir Next"
SEGMENT_JSON = ROOT / "video" / "voiceover" / "segments.json"


VARIANTS = [
    {
        "key": "rosa",
        "title": "ROSA",
        "bullets": ["Suffix memory injected", "Lossless long-range routing"],
        "eq": "Add suffix memory into the token stream",
        "image": "umt_rosa.png",
    },
    {
        "key": "deepembed",
        "title": "DeepEmbed",
        "bullets": ["Token-conditioned FFN", "Better rare-token routing"],
        "eq": "Rare tokens steer the FFN path",
        "image": "umt_deepembed.png",
    },
    {
        "key": "convmix",
        "title": "ConvMix",
        "bullets": ["Local + global memory", "Gate blends two updates"],
        "eq": "Blend local conv updates with matrix memory",
        "image": "umt_convmix.png",
    },
    {
        "key": "dynamic",
        "title": "Dynamic",
        "bullets": ["Per-head decay", "Different memory horizons"],
        "eq": "Each head learns its own decay speed",
        "image": "umt_dynamic.png",
    },
    {
        "key": "dual",
        "title": "DualTimescale",
        "bullets": ["Fast + slow states", "Mix short and long memory"],
        "eq": "Fast scratchpad plus slow archive",
        "image": "umt_dual.png",
    },
    {
        "key": "skewstable",
        "title": "SkewStable",
        "bullets": ["Skew-symmetric update", "Eigenvalues near imaginary axis"],
        "eq": "Rotation-like updates keep energy bounded",
        "image": "umt_skewstable.png",
    },
    {
        "key": "spectral",
        "title": "Spectral",
        "bullets": ["Spectral radius control", "Stability for long contexts"],
        "eq": "Cap memory strength for stable long context",
        "image": "umt_spectral.png",
    },
    {
        "key": "stepcond",
        "title": "StepConditioned",
        "bullets": ["Gate by UT step", "Early refine, late consolidate"],
        "eq": "Early steps explore, late steps consolidate",
        "image": "umt_stepcond.png",
    },
    {
        "key": "structured",
        "title": "Structured",
        "bullets": ["Low-rank + diagonal", "More memory at low cost"],
        "eq": "Low-rank memory with a diagonal boost",
        "image": "umt_structured.png",
    },
    {
        "key": "rulemix",
        "title": "RuleMix",
        "bullets": ["Mixture of update rules", "Model discovers best rule"],
        "eq": "Learn which update rule to trust",
        "image": "umt_rulemix.png",
    },
    {
        "key": "hybrid",
        "title": "Hybrid",
        "bullets": ["Occasional attention", "Recover hard dependencies"],
        "eq": "Call attention only when needed",
        "image": "umt_hybrid.png",
    },
    {
        "key": "discovery",
        "title": "Discovery",
        "bullets": ["RuleMix + ROSA + DeepEmbed", "Spectral guard for stability"],
        "eq": "Merge routing, stability, and structure",
        "image": "umt_discovery.png",
    },
]


VARIANT_LOOKUP = {variant["key"]: variant for variant in VARIANTS}


def label(text: str, font_size: int, color: str) -> Text:
    return Text(
        text,
        font=FONT,
        font_size=font_size,
        color=color,
        disable_ligatures=False,
    )


def gradient_background() -> Rectangle:
    bg = Rectangle(width=config.frame_width, height=config.frame_height)
    bg.set_fill(color=["#f7f1ea", "#efe4d9"], opacity=1.0)
    bg.set_stroke(width=0)
    bg.set_z_index(-10)
    return bg


def focus_overlay(opacity: float = 0.16) -> Rectangle:
    overlay = Rectangle(width=config.frame_width, height=config.frame_height)
    overlay.set_fill(color="#000000", opacity=opacity)
    overlay.set_stroke(width=0)
    overlay.set_z_index(-5)
    return overlay


def chapter_card(title: str, idx: int, total: int) -> VGroup:
    card = RoundedRectangle(corner_radius=0.12, width=7.2, height=1.8)
    card.set_stroke(color="#222222", width=1.1)
    card.set_fill(color="#ffffff", opacity=0.96)

    title_text = label(title, 28, "#111111")

    bar_base = Rectangle(width=5.8, height=0.12)
    bar_base.set_fill(color="#e6dcd2", opacity=1.0)
    bar_base.set_stroke(color="#c9bdb1", width=0.6)

    fill_w = max(0.2, bar_base.width * (idx / max(1, total)))
    bar_fill = Rectangle(width=fill_w, height=0.12)
    bar_fill.set_fill(color="#2C5AA0", opacity=0.9)
    bar_fill.set_stroke(width=0)
    bar_fill.align_to(bar_base, LEFT)

    bar_group = VGroup(bar_base, bar_fill)
    content = VGroup(title_text, bar_group).arrange(DOWN, buff=0.25)
    content.move_to(card)
    group = VGroup(card, content)
    group.move_to([0, 1.8, 0])
    return group


def legend_card() -> VGroup:
    box = RoundedRectangle(corner_radius=0.08, width=4.4, height=2.0)
    box.set_stroke(color="#222222", width=1.0)
    box.set_fill(color="#ffffff", opacity=0.96)

    arrow = Arrow([-0.6, 0, 0], [0.4, 0, 0], buff=0.02, color="#2C5AA0", stroke_width=2)
    arrow_label = label("Flow", 16, "#333333")
    row1 = VGroup(arrow, arrow_label).arrange(RIGHT, buff=0.2)

    grid = VGroup(*[Square(side_length=0.18) for _ in range(4)])
    grid.arrange_in_grid(rows=2, cols=2, buff=0.04)
    for cell in grid:
        cell.set_stroke(color="#1f1f1f", width=0.8)
        cell.set_fill(color="#e9eff7", opacity=1.0)
    grid_label = label("Matrix memory", 16, "#333333")
    row2 = VGroup(grid, grid_label).arrange(RIGHT, buff=0.2)

    token = Dot(radius=0.06, color="#2C5AA0")
    token.set_fill(color="#2C5AA0", opacity=1.0)
    token_label = label("Token", 16, "#333333")
    row3 = VGroup(token, token_label).arrange(RIGHT, buff=0.2)

    rows = VGroup(row1, row2, row3).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
    rows.move_to(box)
    group = VGroup(box, rows)
    group.to_edge(LEFT, buff=0.5)
    group.to_edge(DOWN, buff=0.4)
    return group


def flow_token_animation(arrows: VGroup, run_time: float = 2.8) -> tuple[Dot, Succession]:
    token = Dot(radius=0.06, color="#2C5AA0")
    token.set_fill(color="#2C5AA0", opacity=1.0)
    token.move_to(arrows[0].get_start())
    token.set_z_index(5)
    path_anims = [MoveAlongPath(token, arrow) for arrow in arrows[:4]]
    return token, Succession(*path_anims, run_time=run_time)


def panel_rect(image: ImageMobject, index: int) -> Rectangle:
    panel_w = image.width / 3.0
    center_x = image.get_left()[0] + panel_w * (index + 0.5)
    center = np.array([center_x, image.get_center()[1], 0.0])
    rect = Rectangle(width=panel_w * 0.95, height=image.height * 0.85)
    rect.move_to(center)
    return rect


def flow_arrows(rect: Rectangle) -> VGroup:
    w = rect.width
    h = rect.height
    cx = rect.get_center()[0]
    left = rect.get_left()[0] + 0.12 * w
    y_positions = np.linspace(rect.get_bottom()[1] + 0.14 * h, rect.get_top()[1] - 0.14 * h, 5)
    y0, y1, y2, y3, y4 = y_positions
    color = "#2C5AA0"

    arrows = VGroup(
        Arrow([cx, y0, 0], [cx, y1, 0], buff=0.02, color=color, stroke_width=3),
        Arrow([cx, y1, 0], [cx, y2, 0], buff=0.02, color=color, stroke_width=3),
        Arrow([cx, y2, 0], [cx, y3, 0], buff=0.02, color=color, stroke_width=3),
        Arrow([cx, y3, 0], [cx, y4, 0], buff=0.02, color=color, stroke_width=3),
        Arrow([left, y0, 0], [left, y3, 0], buff=0.02, color=color, stroke_width=3),
        Arrow([left, y3, 0], [left + 0.3 * w, y3, 0], buff=0.02, color=color, stroke_width=3),
    )
    return arrows


def callout(title: str, bullets: list[str], eq: str) -> VGroup:
    title_text = label(title, 28, "#000000")
    bullet_text = VGroup(
        *[label(f"• {b}", 20, "#333333") for b in bullets]
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
    eq_text = label(eq, 19, "#111111")
    group = VGroup(title_text, bullet_text, eq_text).arrange(DOWN, aligned_edge=LEFT, buff=0.25)

    box = RoundedRectangle(corner_radius=0.06, width=5.4, height=2.8)
    box.set_stroke(color="#111111", width=1.0)
    box.set_fill(color="#ffffff", opacity=1.0)
    box.move_to(group)
    group.move_to(box)
    call = VGroup(box, group)
    call.to_edge(RIGHT, buff=0.5)
    call.shift(DOWN * 0.1)
    return call


def indicator(active_index: int) -> VGroup:
    labels = ["Baseline", "Core", "Variant"]
    items = VGroup()
    for i, name in enumerate(labels):
        circ = Circle(radius=0.12)
        if i == active_index:
            circ.set_fill(color="#2C5AA0", opacity=1.0)
            circ.set_stroke(color="#2C5AA0", width=2)
        else:
            circ.set_fill(color="#ffffff", opacity=1.0)
            circ.set_stroke(color="#999999", width=1.2)
        text = label(name, 16, "#333333")
        items.add(VGroup(circ, text).arrange(RIGHT, buff=0.15))
    items.arrange(RIGHT, buff=0.6)
    items.to_edge(DOWN, buff=0.35)
    return items


def load_segment_durations() -> dict[str, float]:
    if not SEGMENT_JSON.exists():
        return {}
    try:
        payload = json.loads(SEGMENT_JSON.read_text())
        return payload.get("segments", {})
    except json.JSONDecodeError:
        return {}


def info_card(title: str, bullets: list[str], width: float = 8.0, height: float = 3.2) -> VGroup:
    title_text = label(title, 30, "#111111")
    bullet_text = VGroup(
        *[label(f"• {b}", 22, "#333333") for b in bullets]
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
    content = VGroup(title_text, bullet_text).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
    box = RoundedRectangle(corner_radius=0.08, width=width, height=height)
    box.set_stroke(color="#111111", width=1.1)
    box.set_fill(color="#ffffff", opacity=1.0)
    box.move_to(content)
    content.move_to(box)
    card = VGroup(box, content)
    card.move_to([0, -0.2, 0])
    return card


def complexity_panel() -> VGroup:
    title = label("Compute & Memory", 30, "#111111")
    attn_bar = Rectangle(width=1.4, height=2.6)
    attn_bar.set_fill(color="#E07A5F", opacity=0.85)
    attn_bar.set_stroke(color="#B85C3F", width=1.0)
    mat_bar = Rectangle(width=1.4, height=1.4)
    mat_bar.set_fill(color="#81B29A", opacity=0.9)
    mat_bar.set_stroke(color="#5B8A74", width=1.0)
    attn_label = label("Attention\nquadratic", 18, "#222222")
    mat_label = label("UniMatrix\nlinear", 18, "#222222")
    attn_group = VGroup(attn_bar, attn_label).arrange(DOWN, buff=0.2)
    mat_group = VGroup(mat_bar, mat_label).arrange(DOWN, buff=0.2)
    bars = VGroup(attn_group, mat_group).arrange(RIGHT, buff=1.2)
    panel = VGroup(title, bars).arrange(DOWN, buff=0.5)
    panel.move_to([0, -0.1, 0])
    return panel


def performance_panel() -> VGroup:
    title = label("Performance Signals", 30, "#111111")
    labels = ["Throughput", "Latency", "Long Recall"]
    values = [0.85, 0.75, 0.8]
    rows = VGroup()
    for metric, value in zip(labels, values):
        base = Rectangle(width=4.0, height=0.24)
        base.set_stroke(color="#cccccc", width=1.0)
        base.set_fill(color="#f3f3f3", opacity=1.0)
        fill = Rectangle(width=4.0 * value, height=0.24)
        fill.set_stroke(width=0)
        fill.set_fill(color="#2C5AA0", opacity=0.9)
        fill.align_to(base, LEFT)
        metric_text = label(metric, 20, "#222222")
        row = VGroup(metric_text, VGroup(base, fill)).arrange(RIGHT, buff=0.5)
        rows.add(row)
    rows.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
    panel = VGroup(title, rows).arrange(DOWN, buff=0.5)
    panel.move_to([0, -0.1, 0])
    return panel


def scaling_curve() -> VGroup:
    title = label("Scaling Intuition", 30, "#111111")
    axis_x = Line([-3, -1.2, 0], [3, -1.2, 0], color="#222222")
    axis_y = Line([-3, -1.2, 0], [-3, 1.8, 0], color="#222222")
    curve = Line([-3, -1.1, 0], [2.4, 1.4, 0], color="#2C5AA0", stroke_width=4)
    dot = Circle(radius=0.08, color="#2C5AA0").set_fill("#2C5AA0", 1.0)
    dot.move_to([2.4, 1.4, 0])
    x_label = label("Context length", 18, "#222222").next_to(axis_x, DOWN, buff=0.15)
    y_label = label("Quality", 18, "#222222").next_to(axis_y, LEFT, buff=0.2).rotate(1.5708)
    panel = VGroup(title, axis_x, axis_y, curve, dot, x_label, y_label)
    panel.move_to([0, -0.1, 0])
    return panel


def matrix_demo() -> dict[str, VGroup]:
    title = label("Matrix Memory", 24, "#111111")
    rows, cols = 4, 4
    grid = VGroup(
        *[Square(side_length=0.28) for _ in range(rows * cols)]
    )
    grid.arrange_in_grid(rows=rows, cols=cols, buff=0.05)
    for cell in grid:
        cell.set_stroke(color="#1f1f1f", width=1.0)
        cell.set_fill(color="#e9eff7", opacity=1.0)

    token = Circle(radius=0.14, color="#2C5AA0")
    token.set_fill(color="#2C5AA0", opacity=1.0)
    token_label = label("token", 12, WHITE)
    token_label.move_to(token)
    token_group = VGroup(token, token_label)

    context_box = RoundedRectangle(corner_radius=0.06, width=1.2, height=0.5)
    context_box.set_stroke(color="#333333", width=1.0)
    context_box.set_fill(color="#f7f7f7", opacity=1.0)
    context_label = label("context", 12, "#333333")
    context_label.move_to(context_box)
    context_group = VGroup(context_box, context_label)

    layout = VGroup(token_group, grid, context_group).arrange(RIGHT, buff=0.35)
    demo = VGroup(title, layout).arrange(DOWN, buff=0.2)
    demo.to_edge(RIGHT, buff=0.6)
    demo.shift(DOWN * 0.6)

    write_arrow = Arrow(token_group.get_right(), grid.get_left(), buff=0.08, color="#2C5AA0", stroke_width=3)
    read_arrow = Arrow(grid.get_right(), context_group.get_left(), buff=0.08, color="#2C5AA0", stroke_width=3)

    write_cell = grid[5]
    read_col = VGroup(grid[2], grid[6], grid[10], grid[14])

    return {
        "group": demo,
        "write_arrow": write_arrow,
        "read_arrow": read_arrow,
        "write_cell": write_cell,
        "read_col": read_col,
    }


class UniMatrixCombined(Scene):
    def construct(self):
        self.camera.background_color = "#f3eee8"
        bg = gradient_background()
        self.add(bg)
        durations = load_segment_durations()

        chapters = [
            ("Motivation", "Motivation"),
            ("Compute and Memory", "Compute & Memory"),
            ("Performance Signals", "Performance Signals"),
            ("Training Cost", "Training Cost"),
            ("Scaling Intuition", "Scaling Intuition"),
            ("Universal Transformer", "Universal Transformer"),
            ("UniMatrix Core", "UniMatrix Core"),
            ("Matrix Memory Demo", "Matrix Memory Demo"),
            ("Routing Variants", "Routing Variants"),
            ("Stability and Timescales", "Stability & Timescales"),
            ("Structure and Search", "Structure & Search"),
            ("Discovery Merge", "Discovery Merge"),
            ("Serving Cost", "Serving Cost"),
            ("Scaling Strategy", "Scaling Strategy"),
            ("Evaluation", "Evaluation"),
            ("Summary", "Summary"),
        ]
        chapter_lookup = {key: (idx + 1, title) for idx, (key, title) in enumerate(chapters)}
        chapter_total = len(chapters)

        def segment_wait(key: str, used: float, tail: float = 0.0, min_wait: float = 0.2) -> None:
            if key not in durations:
                self.wait(min_wait)
                return
            remaining = durations[key] - used - tail
            if remaining < min_wait:
                remaining = min_wait
            self.wait(remaining)

        def show_chapter(key: str) -> float:
            idx, title = chapter_lookup[key]
            card = chapter_card(title, idx, chapter_total)
            self.play(FadeIn(card), run_time=0.5)
            self.wait(0.2)
            self.play(FadeOut(card), run_time=0.3)
            return 1.0

        header = label("UniMatrix Architecture Suite", 44, "#000000")
        sub = label("Unified Overview and Intuition", 26, "#444444")
        VGroup(header, sub).arrange(DOWN, buff=0.2).to_edge(UP)
        self.play(Write(header), FadeIn(sub), run_time=1.2)
        intro_used = 1.2
        intro_total = durations.get("Intro", 20.0)

        opening = info_card(
            "What you'll learn",
            [
                "Why long context makes attention slow",
                "How matrix memory keeps cost steady",
                "Which variants improve routing and stability",
            ],
            width=8.4,
            height=3.1,
        )
        roadmap = info_card(
            "Roadmap",
            [
                "The scaling problem",
                "The UniMatrix update",
                "Variants and scaling strategy",
            ],
            width=8.0,
            height=3.0,
        )
        fade_in = 0.8
        fade_out = 0.4
        fixed = intro_used + (fade_in * 2) + (fade_out * 2)
        remaining = max(0.0, intro_total - fixed)
        hold_one = remaining * 0.55
        hold_two = max(0.0, remaining - hold_one)

        dim = focus_overlay(0.12)
        self.play(FadeIn(dim), FadeIn(opening), run_time=fade_in)
        self.bring_to_front(opening)
        intro_used += fade_in
        self.wait(hold_one)
        intro_used += hold_one
        self.play(FadeOut(opening), run_time=fade_out)
        intro_used += fade_out
        self.play(FadeIn(roadmap), run_time=fade_in)
        self.bring_to_front(roadmap)
        intro_used += fade_in
        self.wait(hold_two)
        intro_used += hold_two
        self.play(FadeOut(roadmap), FadeOut(dim), run_time=fade_out)
        intro_used += fade_out

        segment_wait("Intro", used=intro_used)

        # early motivation cards
        used = show_chapter("Motivation")
        motivation = info_card(
            "Why change the backbone?",
            [
                "Long context costs explode with attention",
                "KV cache grows linearly with tokens and layers",
                "We want streamable memory with steady cost",
            ],
        )
        dim = focus_overlay()
        self.play(FadeIn(dim), FadeIn(motivation), run_time=1.0)
        self.bring_to_front(motivation)
        used += 1.0
        segment_wait("Motivation", used=used, tail=0.6)
        self.play(FadeOut(motivation), FadeOut(dim), run_time=0.6)

        used = show_chapter("Compute and Memory")
        compute_panel = complexity_panel()
        dim = focus_overlay()
        self.play(FadeIn(dim), FadeIn(compute_panel), run_time=1.0)
        self.bring_to_front(compute_panel)
        used += 1.0
        segment_wait("Compute and Memory", used=used, tail=0.6)
        self.play(FadeOut(compute_panel), FadeOut(dim), run_time=0.6)

        used = show_chapter("Performance Signals")
        perf_panel = performance_panel()
        dim = focus_overlay()
        self.play(FadeIn(dim), FadeIn(perf_panel), run_time=1.0)
        self.bring_to_front(perf_panel)
        used += 1.0
        segment_wait("Performance Signals", used=used, tail=0.6)
        self.play(FadeOut(perf_panel), FadeOut(dim), run_time=0.6)

        used = show_chapter("Training Cost")
        training = info_card(
            "Training Cost",
            [
                "Linear memory keeps batch sizes healthy",
                "Stable updates reduce wasted steps",
                "Long context without VRAM explosions",
            ],
            width=7.6,
            height=3.2,
        )
        dim = focus_overlay()
        self.play(FadeIn(dim), FadeIn(training), run_time=0.8)
        self.bring_to_front(training)
        used += 0.8
        segment_wait("Training Cost", used=used, tail=0.6)
        self.play(FadeOut(training), FadeOut(dim), run_time=0.6)

        used = show_chapter("Scaling Intuition")
        scale_panel = scaling_curve()
        dim = focus_overlay()
        self.play(FadeIn(dim), FadeIn(scale_panel), run_time=1.0)
        self.bring_to_front(scale_panel)
        used += 1.0
        segment_wait("Scaling Intuition", used=used, tail=0.6)
        self.play(FadeOut(scale_panel), FadeOut(dim), run_time=0.6)

        # base image
        img = ImageMobject(str(ARCH_DIR / "umt_dynamic.png"))
        img.scale_to_fit_width(9.4)
        img.next_to(sub, DOWN, buff=0.2)
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

        used = 0.0
        self.play(FadeIn(shadow), FadeIn(card), FadeIn(img, shift=0.1), run_time=1.0)
        used += 1.0
        used += show_chapter("Universal Transformer")

        legend = legend_card()
        self.play(FadeIn(legend), run_time=0.6)
        self.wait(1.0)
        self.play(FadeOut(legend), run_time=0.4)
        used += 2.0

        # Baseline
        rect = panel_rect(img, 0)
        call = callout(
            "Universal Transformer",
            ["Shared parameters across depth", "Repeat K steps for refinement"],
            "Refine a shared hidden canvas step by step",
        )
        prog = indicator(0)
        arrows = flow_arrows(rect)
        dim = focus_overlay(0.12)
        self.play(Create(rect), FadeIn(call), FadeIn(prog), FadeIn(dim), run_time=1.0)
        self.bring_to_front(card, img, rect, prog, call)
        used += 1.0
        token, token_anim = flow_token_animation(arrows, run_time=3.2)
        self.add(token)
        self.bring_to_front(arrows, token)
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.12), token_anim, run_time=3.2)
        used += 3.2
        segment_wait("Universal Transformer", used=used, tail=0.4)
        self.play(FadeOut(call), FadeOut(arrows), FadeOut(dim), FadeOut(token), run_time=0.4)

        # Core
        used = show_chapter("UniMatrix Core")
        self.play(Transform(rect, panel_rect(img, 1)), Transform(prog, indicator(1)), run_time=0.7)
        used += 0.7
        call = callout(
            "UniMatrix Core",
            ["Matrix-state recurrence", "Linear-time update, no KV cache"],
            "Update a shared memory matrix each token",
        )
        arrows = flow_arrows(panel_rect(img, 1))
        dim = focus_overlay(0.12)
        self.play(FadeIn(call), FadeIn(dim), run_time=0.7)
        self.bring_to_front(card, img, rect, prog, call)
        used += 0.7
        token, token_anim = flow_token_animation(arrows, run_time=3.2)
        self.add(token)
        self.bring_to_front(arrows, token)
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.12), token_anim, run_time=3.2)
        used += 3.2
        segment_wait("UniMatrix Core", used=used, tail=0.4)
        self.play(FadeOut(call), FadeOut(arrows), FadeOut(dim), FadeOut(token), run_time=0.4)

        used = show_chapter("Matrix Memory Demo")
        demo = matrix_demo()
        dim = focus_overlay(0.14)
        self.play(FadeIn(dim), FadeIn(demo["group"]), run_time=0.5)
        self.bring_to_front(demo["group"])
        used += 0.5
        self.play(GrowArrow(demo["write_arrow"]), Indicate(demo["write_cell"], scale_factor=1.15), run_time=2.0)
        used += 2.0
        self.play(GrowArrow(demo["read_arrow"]), Indicate(demo["read_col"], scale_factor=1.08), run_time=2.0)
        used += 2.0
        segment_wait("Matrix Memory Demo", used=used, tail=0.25)
        self.play(
            FadeOut(demo["group"]),
            FadeOut(demo["write_arrow"]),
            FadeOut(demo["read_arrow"]),
            FadeOut(dim),
            run_time=0.25,
        )

        def play_variant_group(
            keys: list[str],
            segment_key: str,
            chapter_time: float = 0.0,
            tail_fadeout: bool = False,
        ) -> None:
            segment_total = durations.get(segment_key)
            base_per_variant = 0.6 + 0.6 + 0.6 + 2.8 + 0.35
            tail = 0.6 if tail_fadeout else 0.0
            if segment_total:
                segment_total = max(0.0, segment_total - chapter_time)
                extra = max(0.5, (segment_total - base_per_variant * len(keys) - tail) / max(1, len(keys)))
            else:
                extra = 0.8

            for key in keys:
                variant = VARIANT_LOOKUP[key]
                new_img = ImageMobject(str(ARCH_DIR / variant["image"]))
                new_img.scale_to_fit_width(img.width)
                new_img.move_to(img)
                self.play(Transform(img, new_img), run_time=0.6)

                rect_target = panel_rect(img, 2)
                self.play(Transform(rect, rect_target), Transform(prog, indicator(2)), run_time=0.6)
                call = callout(variant["title"], variant["bullets"], variant["eq"])
                arrows = flow_arrows(rect_target)
                dim = focus_overlay(0.12)
                self.play(FadeIn(call), FadeIn(dim), run_time=0.6)
                self.bring_to_front(card, img, rect, prog, call)
                token, token_anim = flow_token_animation(arrows, run_time=2.8)
                self.add(token)
                self.bring_to_front(arrows, token)
                self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.12), token_anim, run_time=2.8)
                self.wait(extra)
                self.play(FadeOut(call), FadeOut(arrows), FadeOut(dim), FadeOut(token), run_time=0.35)

            if tail_fadeout:
                self.play(FadeOut(rect), FadeOut(prog), FadeOut(img), FadeOut(card), FadeOut(shadow), run_time=0.6)

        chapter_time = show_chapter("Routing Variants")
        play_variant_group(["rosa", "deepembed", "convmix"], "Routing Variants", chapter_time=chapter_time)
        chapter_time = show_chapter("Stability and Timescales")
        play_variant_group(
            ["dynamic", "dual", "skewstable", "spectral", "stepcond"],
            "Stability and Timescales",
            chapter_time=chapter_time,
        )
        chapter_time = show_chapter("Structure and Search")
        play_variant_group(["structured", "rulemix", "hybrid"], "Structure and Search", chapter_time=chapter_time)
        chapter_time = show_chapter("Discovery Merge")
        play_variant_group(["discovery"], "Discovery Merge", chapter_time=chapter_time, tail_fadeout=True)

        used = show_chapter("Serving Cost")
        serving = info_card(
            "Serving Cost",
            [
                "No massive KV cache to manage",
                "Lower latency and steadier throughput",
                "Streaming stays stable at long context",
            ],
            width=7.8,
            height=3.2,
        )
        dim = focus_overlay()
        self.play(FadeIn(dim), FadeIn(serving), run_time=0.8)
        self.bring_to_front(serving)
        used += 0.8
        segment_wait("Serving Cost", used=used, tail=0.6)
        self.play(FadeOut(serving), FadeOut(dim), run_time=0.6)

        used = show_chapter("Scaling Strategy")
        scaling = info_card(
            "Scaling Strategy",
            [
                "Start short context, grow length gradually",
                "Add routing variants after core stabilizes",
                "Use long context data to train persistence",
            ],
            width=7.9,
            height=3.2,
        )
        dim = focus_overlay()
        self.play(FadeIn(dim), FadeIn(scaling), run_time=0.8)
        self.bring_to_front(scaling)
        used += 0.8
        segment_wait("Scaling Strategy", used=used, tail=0.6)
        self.play(FadeOut(scaling), FadeOut(dim), run_time=0.6)

        used = show_chapter("Evaluation")
        eval_card = info_card(
            "Evaluation",
            [
                "Long-context retrieval and recall",
                "Latency under streaming decode",
                "Quality at scale and efficiency",
            ],
            width=7.6,
            height=3.2,
        )
        dim = focus_overlay()
        self.play(FadeIn(dim), FadeIn(eval_card), run_time=0.8)
        self.bring_to_front(eval_card)
        used += 0.8
        segment_wait("Evaluation", used=used, tail=0.6)
        self.play(FadeOut(eval_card), FadeOut(dim), run_time=0.6)

        used = show_chapter("Summary")
        summary = info_card(
            "Takeaways",
            [
                "UniMatrix keeps cost steady as context grows",
                "Variants tune routing, stability, and structure",
                "Discovery merges the best ideas for scale",
            ],
            width=7.8,
            height=3.2,
        )
        dim = focus_overlay()
        self.play(FadeIn(dim), FadeIn(summary), run_time=0.8)
        self.bring_to_front(summary)
        used += 0.8
        bullets = summary[1][1]
        self.play(Indicate(bullets[0], scale_factor=1.05), run_time=3.0)
        used += 3.0
        self.play(Indicate(bullets[1], scale_factor=1.05), run_time=3.0)
        used += 3.0
        self.play(Indicate(bullets[2], scale_factor=1.05), run_time=3.0)
        used += 3.0
        segment_wait("Summary", used=used, tail=0.6)
        self.play(FadeOut(summary), FadeOut(dim), run_time=0.6)
        self.wait(1.2)
