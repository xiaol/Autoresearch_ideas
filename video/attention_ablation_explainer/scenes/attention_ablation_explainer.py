"""Manim Community explainer: auditing a 4-variant sparse-attention pretraining ablation.

Render preview from /home/xiaol/X:

    MANIM_PREVIEW=1 MANIM_LAYOUT_STRICT=1 \
    /home/xiaol/X/ai_hunt_replicate/.conda-manim/bin/manim -ql --disable_caching \
        attention_ablation_manim/scenes/attention_ablation_explainer.py AttentionAblationExplainer

Render final 4K visual pass:

    MANIM_LAYOUT_STRICT=1 \
    /home/xiaol/X/ai_hunt_replicate/.conda-manim/bin/manim -qh --disable_caching \
        attention_ablation_manim/scenes/attention_ablation_explainer.py AttentionAblationExplainer
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from manim import (
    BLACK,
    BLUE_C,
    BLUE_D,
    BOLD,
    Brace,
    Create,
    DashedLine,
    DOWN,
    FadeIn,
    FadeOut,
    GOLD_C,
    GREEN_C,
    GREY_A,
    GREY_B,
    GREY_C,
    GREY_D,
    Group,
    GrowArrow,
    Indicate,
    LaggedStart,
    LEFT,
    Line,
    MathTex,
    ORANGE,
    ORIGIN,
    PURPLE_A,
    RED_C,
    RIGHT,
    RoundedRectangle,
    Scene,
    SurroundingRectangle,
    TEAL_C,
    Text,
    TransformFromCopy,
    UP,
    VGroup,
    WHITE,
    Write,
    YELLOW_C,
    Arrow,
    Axes,
    Rectangle,
    ReplacementTransform,
    config,
)

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
ASSETS = ROOT / "assets"

config.background_color = BLACK
config.frame_width = 16
config.frame_height = 9
if os.environ.get("MANIM_PREVIEW") == "1":
    config.pixel_width = 854
    config.pixel_height = 480
    config.frame_rate = 15
else:
    config.pixel_width = 3840
    config.pixel_height = 2160
    config.frame_rate = 30
config.media_dir = str(ROOT / "renders")

FONT = "DejaVu Sans"
MONO = "DejaVu Sans Mono"

INK = GREY_A
MUTED = GREY_B
FAINT = GREY_D
BLUE = BLUE_C
DEEP_BLUE = BLUE_D
GREEN = GREEN_C
CYAN = TEAL_C
AMBER = YELLOW_C
GOLD = GOLD_C
RED = RED_C
PURPLE = PURPLE_A
PANEL = "#10171f"
PANEL_DARK = "#0b1118"
PANEL_WARM = "#19140e"

VARIANT_COLORS = {"dsa": ORANGE, "lsa": BLUE, "csa": GREEN, "hca": PURPLE}
VARIANT_NAMES = {
    "dsa": "DSA  oracle top-k",
    "lsa": "LSA  local + recall",
    "csa": "CSA  compressed 4:1",
    "hca": "HCA  compressed 8:1",
}

# Segment end-times (seconds). FINALIZED from narration/natural_durations.json —
# each window = cumulative natural MiniMax duration + intentional holds.
# Populated by scripts/apply_windows.py; keep in sync with narration_tts.md.
BEATS_PATH = ROOT / "narration" / "beat_windows.json"
if BEATS_PATH.is_file():
    BEATS = {int(k): float(v) for k, v in json.loads(BEATS_PATH.read_text()).items()}
else:  # provisional fallback so previews render before TTS measurement
    BEATS = {
        1: 52.0, 2: 86.0, 3: 132.0, 4: 178.0, 5: 224.0, 6: 266.0,
        7: 314.0, 8: 352.0, 9: 400.0, 10: 450.0, 11: 500.0,
    }

RESULTS_PATH = ANALYSIS / "comparison.json"

# natural narration length per beat (for word-proportional intra-beat sync points)
DUR_PATH = ROOT / "narration" / "natural_durations.json"
if DUR_PATH.is_file():
    NATURALS = {
        seg["index"]: float(seg["natural_sec"])
        for seg in json.loads(DUR_PATH.read_text())["segments"]
        if seg.get("status") == "ok"
    }
else:
    NATURALS = {}


def txt(content, size=28, color=INK, weight=None, font=FONT, line_spacing=0.9):
    kwargs = {
        "font": font,
        "font_size": size,
        "color": color,
        "line_spacing": line_spacing,
        "disable_ligatures": True,
    }
    if weight is not None:
        kwargs["weight"] = weight
    return Text(content, **kwargs)


def mono(content, size=22, color=INK, weight=None):
    return txt(content, size=size, color=color, weight=weight, font=MONO)


def eq(content, size=34, color=INK):
    return MathTex(content, font_size=size, color=color)


def fit_width(mob, width, shrink=0.98):
    if mob.width > width:
        mob.scale(width / mob.width * shrink)
    return mob


def fit_height(mob, height, shrink=0.98):
    if mob.height > height:
        mob.scale(height / mob.height * shrink)
    return mob


def panel(width, height, color=GREY_C, fill=PANEL, opacity=0.86):
    return RoundedRectangle(
        width=width,
        height=height,
        corner_radius=0.08,
        stroke_width=1.35,
        stroke_color=color,
        fill_color=fill,
        fill_opacity=opacity,
    )


def chip(label, color=BLUE, width=None, height=0.5, size=18):
    word = txt(label, size=size, color=WHITE, weight=BOLD)
    box_width = max(width or 0.0, word.width + 0.34)
    box = RoundedRectangle(
        width=box_width,
        height=height,
        corner_radius=0.08,
        stroke_width=1.2,
        stroke_color=color,
        fill_color=color,
        fill_opacity=0.18,
    )
    fit_width(word, box_width - 0.18)
    word.move_to(box)
    return VGroup(box, word)


def arrow_between(left, right, color=INK, buff=0.10, stroke=3.0):
    return Arrow(
        start=left.get_right() + RIGHT * buff,
        end=right.get_left() + LEFT * buff,
        color=color,
        stroke_width=stroke,
        buff=0,
        max_tip_length_to_length_ratio=0.18,
        max_stroke_width_to_length_ratio=8,
    )


def attention_grid(n=18, cell=0.17, base_color=GREY_D):
    """n x n causal grid; returns (VGroup grid, function cell_at(q, k))."""
    cells = {}
    group = VGroup()
    for q in range(n):
        for k in range(n):
            rect = Rectangle(
                width=cell,
                height=cell,
                stroke_width=0.55,
                stroke_color="#2a323c",
                fill_color=base_color if k <= q else "#0a0e13",
                fill_opacity=0.55 if k <= q else 0.25,
            )
            rect.move_to(np.array([k * cell, -q * cell, 0.0]))
            group.add(rect)
            cells[(q, k)] = rect
    group.center()
    return group, cells


def paint(cells, coords, color, opacity=0.92):
    anims = []
    for qk in coords:
        if qk in cells:
            anims.append(cells[qk].animate.set_fill(color, opacity=opacity))
    return anims


def mask_thumbnail(kind, n=12, cell=0.13):
    grid, cells = attention_grid(n=n, cell=cell)
    rng = np.random.default_rng(7)
    color = VARIANT_COLORS[kind]
    for q in range(n):
        allowed = list(range(q + 1))
        chosen = []
        if kind == "dsa":
            chosen = list(rng.choice(allowed, size=min(4, len(allowed)), replace=False))
        elif kind == "lsa":
            chosen = [k for k in allowed if q - k < 3]
            if q >= 6:
                chosen += [k for k in allowed[: q - 4][:2]]
        elif kind == "csa":
            chosen = [k for k in allowed if q - k < 3] + [k for k in allowed if k % 4 == 0 and q - k >= 3][:3]
        else:
            chosen = [k for k in allowed if q - k < 3] + [k for k in allowed if k % 5 == 0 and q - k >= 3]
        for k in set(chosen):
            cells[(q, k)].set_fill(color, opacity=0.9)
    return grid


class AttentionAblationExplainer(Scene):
    """Audit-then-rerun story for the DSA/LSA/CSA/HCA pretraining ablation."""

    def construct(self):
        self.camera.background_color = BLACK
        self.layout_checks = []
        self.results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.is_file() else None
        self.cold_open()
        self.harness()
        self.dsa_mechanism()
        self.lsa_mechanism()
        self.csa_hca_mechanism()
        self.audit_causality()
        self.latent_leak()
        self.harness_bugs()
        self.budget_confound()
        self.results_beat()
        self.synthesis()
        self.hold_until(BEATS[11])
        self.write_layout_report()

    # ---------- timeline helpers ----------
    def hold_until(self, end_time):
        remaining = end_time - float(self.time)
        if remaining > 0.02:
            self.wait(remaining)

    def finish_beat(self, end_time, clear=False):
        if clear:
            self.hold_until(max(float(self.time), end_time - 0.5))
            self.clear_scene(run_time=min(0.5, max(0.05, end_time - float(self.time))))
        self.hold_until(end_time)

    def beat_sync(self, beat, frac):
        """Hold until `frac` of beat's spoken narration has elapsed (word-proportional)."""
        start = BEATS.get(beat - 1, 0.0) if beat > 1 else 0.0
        natural = NATURALS.get(beat, BEATS[beat] - start)
        self.hold_until(start + frac * natural)

    def clear_scene(self, run_time=0.45):
        if self.mobjects:
            self.play(FadeOut(Group(*self.mobjects)), run_time=run_time)

    def top_title(self, title, subtitle=None):
        title_mob = txt(title, size=38, color=WHITE, weight=BOLD)
        fit_width(title_mob, 14.8)
        title_mob.to_edge(UP, buff=0.22)
        if subtitle:
            sub = txt(subtitle, size=19, color=MUTED)
            fit_width(sub, 14.2)
            sub.next_to(title_mob, DOWN, buff=0.08)
            return VGroup(title_mob, sub)
        return VGroup(title_mob)

    def register_inside(self, name, outer, inner, margin=0.04):
        outer_bounds = {
            "left": float(outer.get_left()[0] + margin),
            "right": float(outer.get_right()[0] - margin),
            "bottom": float(outer.get_bottom()[1] + margin),
            "top": float(outer.get_top()[1] - margin),
        }
        inner_bounds = {
            "left": float(inner.get_left()[0]),
            "right": float(inner.get_right()[0]),
            "bottom": float(inner.get_bottom()[1]),
            "top": float(inner.get_top()[1]),
        }
        overflow = {
            "left": max(0.0, outer_bounds["left"] - inner_bounds["left"]),
            "right": max(0.0, inner_bounds["right"] - outer_bounds["right"]),
            "bottom": max(0.0, outer_bounds["bottom"] - inner_bounds["bottom"]),
            "top": max(0.0, inner_bounds["top"] - outer_bounds["top"]),
        }
        ok = max(overflow.values()) <= 1e-3
        self.layout_checks.append(
            {"name": name, "ok": ok, "outer": outer_bounds, "inner": inner_bounds, "overflow": overflow}
        )
        if not ok and os.environ.get("MANIM_LAYOUT_STRICT") == "1":
            raise RuntimeError(f"Layout overflow: {json.dumps(self.layout_checks[-1], sort_keys=True)}")

    def write_layout_report(self):
        report_path = Path(os.environ.get("MANIM_LAYOUT_REPORT", str(ANALYSIS / "layout_report.json")))
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "scene": self.__class__.__name__,
            "ok": all(bool(item["ok"]) for item in self.layout_checks),
            "checks": self.layout_checks,
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def caption_panel(self, title, body, width, height, color=BLUE):
        box = panel(width, height, color=color, fill=PANEL, opacity=0.88)
        head = txt(title, size=21, color=WHITE, weight=BOLD)
        body_mob = txt(body, size=16, color=INK, line_spacing=0.86)
        fit_width(head, width - 0.32)
        fit_width(body_mob, width - 0.36)
        stack = VGroup(head, body_mob).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to(box)
        self.register_inside(title, box, stack, margin=0.08)
        return VGroup(box, stack)

    # ---------- beat 1: cold open ----------
    def cold_open(self):
        title = self.top_title(
            "Four Sparse Attentions, One Rigged Race",
            "auditing a pretraining ablation before believing it",
        )
        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=0.9)

        thumbs = Group()
        old_scores = {"dsa": 2.6187, "lsa": 2.6139, "csa": 2.6170, "hca": 2.6051}
        for key in ("dsa", "lsa", "csa", "hca"):
            card = panel(3.15, 3.35, color=VARIANT_COLORS[key], fill=PANEL_DARK)
            thumb = mask_thumbnail(key)
            fit_width(thumb, 2.55)
            thumb.move_to(card.get_center() + UP * 0.42)
            name = txt(VARIANT_NAMES[key], size=16, color=WHITE, weight=BOLD)
            fit_width(name, 2.8)
            name.next_to(thumb, DOWN, buff=0.16)
            score = mono(f"val {old_scores[key]:.4f}", size=17, color=MUTED)
            score.next_to(name, DOWN, buff=0.1)
            self.register_inside(f"cold-open {key}", card, VGroup(thumb, name, score), margin=0.09)
            thumbs.add(Group(card, thumb, name, score))
        thumbs.arrange(RIGHT, buff=0.38).next_to(title, DOWN, buff=0.5)
        self.beat_sync(1, 0.12)  # "Four sparse attention mechanisms..."
        self.play(
            LaggedStart(*[FadeIn(t, shift=UP * 0.18) for t in thumbs], lag_ratio=0.18),
            run_time=2.2,
        )

        brace = Brace(thumbs, DOWN, buff=0.18, color=AMBER)
        gap = txt("after 1,000 steps: max gap 0.014 nats — statistically nothing", size=22, color=AMBER)
        fit_width(gap, 12.5)
        gap.next_to(brace, DOWN, buff=0.14)
        self.beat_sync(1, 0.50)  # "...land within one hundredth of a nat"
        self.play(Create(brace), FadeIn(gap), run_time=1.2)

        verdict = txt(
            "reading 1: \"attention design doesn't matter\"      reading 2: the instrument was broken",
            size=21,
            color=INK,
        )
        fit_width(verdict, 14.4)
        verdict.next_to(gap, DOWN, buff=0.3)
        self.beat_sync(1, 0.63)  # "The obvious reading..."
        self.play(Write(verdict), run_time=1.6)
        strike = Line(verdict.get_left() + LEFT * 0.02, verdict.get_center() + RIGHT * 0.6, color=RED, stroke_width=4)
        self.beat_sync(1, 0.78)  # "The correct reading: this race was rigged"
        self.play(Create(strike), run_time=0.7)
        self.finish_beat(BEATS[1], clear=True)

    # ---------- beat 2: harness ----------
    def harness(self):
        title = self.top_title("One Testbed, One Knob", "byte-level LM - everything fixed except the attention rule")
        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=0.8)

        stream = VGroup()
        for ch in ["T", "h", "e", "_", "c", "a", "t", "…", "EOT"]:
            color = GOLD if ch == "EOT" else BLUE
            stream.add(chip(ch, color=color, height=0.46, size=15))
        stream.arrange(RIGHT, buff=0.1).shift(UP * 1.55 + LEFT * 2.3)
        vocab = self.caption_panel(
            "vocabulary = 257",
            "256 possible byte values + 1 end-of-text marker.\nNo tokenizer to blame - the model reads raw bytes.",
            5.0,
            1.5,
            color=GOLD,
        )
        vocab.next_to(stream, RIGHT, buff=0.45).align_to(stream, UP).shift(DOWN * 0.4)
        self.beat_sync(2, 0.10)  # "A byte-level language model..."
        self.play(LaggedStart(*[FadeIn(c, shift=RIGHT * 0.1) for c in stream], lag_ratio=0.08), run_time=1.4)
        self.beat_sync(2, 0.28)  # "...vocabulary is just 257 symbols"
        self.play(FadeIn(vocab), run_time=0.9)

        blocks = VGroup()
        for i in range(3):
            blk = VGroup(
                panel(4.6, 1.05, color=GREY_C, fill=PANEL),
                txt("attention", size=17, color=WHITE, weight=BOLD),
                txt("MLP + norms", size=14, color=MUTED),
            )
            blk[1].move_to(blk[0].get_center() + UP * 0.2)
            blk[2].move_to(blk[0].get_center() + DOWN * 0.24)
            blocks.add(blk)
        dots = txt("x 6 layers", size=16, color=MUTED)
        stack = VGroup(*blocks, dots).arrange(DOWN, buff=0.18).shift(DOWN * 1.35 + LEFT * 3.4)
        self.beat_sync(2, 0.46)  # "A few transformer layers..."
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.15) for b in blocks], lag_ratio=0.2), FadeIn(dots), run_time=1.6)

        knob = SurroundingRectangle(VGroup(*[b[1] for b in blocks]), color=AMBER, buff=0.14, stroke_width=2.4)
        knob_label = txt("the ONLY thing that changes:\nwho may look at whom", size=19, color=AMBER, line_spacing=0.95)
        knob_label.next_to(knob, RIGHT, buff=0.55)
        fixed = self.caption_panel(
            "held fixed",
            "data mix - model size - optimizer - seeds - training budget",
            6.2,
            1.15,
            color=GREEN,
        )
        fixed.next_to(knob_label, DOWN, buff=0.45).align_to(knob_label, LEFT)
        self.beat_sync(2, 0.62)  # "trained on a fixed mix..."
        self.play(FadeIn(fixed), run_time=0.9)
        self.beat_sync(2, 0.80)  # "except one knob"
        self.play(Create(knob), FadeIn(knob_label), run_time=1.1)
        self.finish_beat(BEATS[2], clear=True)

    # ---------- beat 3: DSA ----------
    def dsa_mechanism(self):
        title = self.top_title("Variant 1 - DSA-style: attend where it matters", "per-query top-k ... selected by an oracle")
        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=0.8)

        grid, cells = attention_grid(n=18, cell=0.21)
        grid.shift(LEFT * 3.9 + DOWN * 0.55)
        glabel = txt("causal score grid  (rows = queries, cols = keys)", size=15, color=MUTED)
        glabel.next_to(grid, DOWN, buff=0.18)
        self.beat_sync(3, 0.05)
        self.play(Create(grid), FadeIn(glabel), run_time=1.6)

        q = 13
        row = [(q, k) for k in range(q + 1)]
        row_note = txt("query 13 scores ALL of its past ...", size=18, color=INK)
        row_note.shift(RIGHT * 3.6 + UP * 1.7)
        self.beat_sync(3, 0.22)  # "Each query scores every earlier position"
        self.play(*paint(cells, row, DEEP_BLUE, 0.75), run_time=0.9)
        self.play(FadeIn(row_note), run_time=0.7)

        rng = np.random.default_rng(3)
        keep = sorted(rng.choice(np.arange(q + 1), size=5, replace=False).tolist())
        keep_note = txt("... keeps only the top-k, softmax over the shortlist", size=18, color=ORANGE)
        keep_note.next_to(row_note, DOWN, buff=0.28).align_to(row_note, LEFT)
        fit_width(keep_note, 7.4)
        self.beat_sync(3, 0.28)  # "keeps only the top scorers"
        self.play(*paint(cells, [(q, k) for k in keep], ORANGE, 0.98), run_time=0.9)
        self.play(FadeIn(keep_note), run_time=0.7)

        formula = eq(r"\mathrm{att}=\mathrm{softmax}\!\big(\mathrm{top}k(QK^{\top}/\sqrt{d})\big)V", size=32, color=WHITE)
        formula.next_to(keep_note, DOWN, buff=0.42).align_to(keep_note, LEFT)
        fit_width(formula, 7.4)
        self.beat_sync(3, 0.36)  # "runs softmax over that shortlist"
        self.play(Write(formula), run_time=1.2)

        oracle = self.caption_panel(
            "but it is an ORACLE",
            "to pick the top-k it computes the FULL score matrix -\nthe exact thing a deployable indexer cannot afford.\nSo it answers: what is PERFECT selection worth?",
            7.4,
            1.85,
            color=RED,
        )
        oracle.next_to(formula, DOWN, buff=0.4).align_to(formula, LEFT)
        self.beat_sync(3, 0.52)  # "But look closely at the implementation..."
        self.play(FadeIn(oracle, shift=UP * 0.15), run_time=1.0)
        self.finish_beat(BEATS[3], clear=True)

    # ---------- beat 4: LSA ----------
    def lsa_mechanism(self):
        title = self.top_title("Variant 2 - LSA-style: guaranteed local + coarse recall", "sliding window - block summaries - reuse the mask every other layer")
        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=0.8)

        grid, cells = attention_grid(n=18, cell=0.21)
        grid.shift(LEFT * 3.9 + DOWN * 0.55)
        self.play(Create(grid), run_time=1.2)

        q = 15
        local = [(q, k) for k in range(q + 1) if q - k < 4]
        local_note = txt("local window: last 64 tokens, always visible", size=18, color=BLUE)
        local_note.shift(RIGHT * 3.3 + UP * 1.85)
        fit_width(local_note, 6.6)
        self.beat_sync(4, 0.13)  # "A guaranteed local window..."
        self.play(*paint(cells, local, BLUE, 0.95), run_time=0.8)
        self.play(FadeIn(local_note), run_time=0.7)

        blocks = [(0, 3), (4, 7), (8, 11)]
        block_rects = VGroup()
        for b0, b1 in blocks:
            rect = SurroundingRectangle(
                VGroup(*[cells[(q, k)] for k in range(b0, b1 + 1)]), color=CYAN, buff=0.02, stroke_width=2.0
            )
            block_rects.add(rect)
        block_note = txt("older past -> blocks of 16, summarized by their mean key", size=18, color=CYAN)
        block_note.next_to(local_note, DOWN, buff=0.26).align_to(local_note, LEFT)
        fit_width(block_note, 6.6)
        self.beat_sync(4, 0.35)  # "older context is grouped into blocks of sixteen"
        self.play(LaggedStart(*[Create(r) for r in block_rects], lag_ratio=0.25), run_time=1.1)
        self.play(FadeIn(block_note), run_time=0.7)

        fine = [(q, 5), (q, 6)]
        fine_note = txt("recall top blocks -> pick individual tokens inside", size=18, color=GOLD)
        fine_note.next_to(block_note, DOWN, buff=0.26).align_to(block_note, LEFT)
        fit_width(fine_note, 6.6)
        self.beat_sync(4, 0.55)  # "picks its four favorite blocks..."
        self.play(block_rects[1].animate.set_stroke(GOLD, width=3.5), run_time=0.6)
        self.play(*paint(cells, fine, GOLD, 0.98), run_time=0.7)
        self.play(FadeIn(fine_note), run_time=0.7)

        l_even = chip("layer 2k: build mask", color=BLUE, size=16, height=0.52)
        l_odd = chip("layer 2k+1: REUSE it", color=GOLD, size=16, height=0.52)
        stack = VGroup(l_even, l_odd).arrange(DOWN, buff=0.55)
        stack.next_to(fine_note, DOWN, buff=0.5).align_to(fine_note, LEFT).shift(RIGHT * 0.9)
        reuse_arrow = Arrow(
            l_even[0].get_bottom() + DOWN * 0.02,
            l_odd[0].get_top() + UP * 0.02,
            color=GOLD,
            stroke_width=3.2,
            buff=0.05,
            max_tip_length_to_length_ratio=0.3,
        )
        reuse_note = txt("a cheap stand-in for cross-layer indexing", size=15, color=MUTED)
        reuse_note.next_to(stack, RIGHT, buff=0.4)
        fit_width(reuse_note, 4.6)
        self.beat_sync(4, 0.76)  # "One more move: every other layer..."
        self.play(FadeIn(l_even), FadeIn(l_odd), GrowArrow(reuse_arrow), FadeIn(reuse_note), run_time=1.2)
        self.finish_beat(BEATS[4], clear=True)

    # ---------- beat 5: CSA / HCA ----------
    def csa_hca_mechanism(self):
        title = self.top_title("Variants 3 & 4 - compressed memory", "the older past exists only as gated block summaries")
        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=0.8)

        tokens = VGroup(*[chip(f"k{i}", color=GREY_C, height=0.44, size=13, width=0.62) for i in range(4)])
        tokens.arrange(RIGHT, buff=0.12).shift(LEFT * 4.6 + UP * 1.35)
        gate = chip("learned gate", color=AMBER, size=15)
        gate.next_to(tokens, DOWN, buff=0.5)
        pooled = chip("K, V summary", color=GREEN, size=15, width=1.9)
        pooled.next_to(gate, DOWN, buff=0.5)
        a1 = Arrow(tokens.get_bottom(), gate[0].get_top(), color=AMBER, stroke_width=2.8, buff=0.08)
        a2 = Arrow(gate[0].get_bottom(), pooled[0].get_top(), color=GREEN, stroke_width=2.8, buff=0.08)
        pool_note = txt("4 tokens -> 1 vector (CSA)\n8 tokens -> 1 vector (HCA)", size=16, color=MUTED, line_spacing=0.95)
        pool_note.next_to(pooled, DOWN, buff=0.3)
        self.beat_sync(5, 0.08)
        self.play(FadeIn(tokens), run_time=0.8)
        self.beat_sync(5, 0.30)  # "a tiny learned gate decides..."
        self.play(GrowArrow(a1), FadeIn(gate), run_time=0.8)
        self.play(GrowArrow(a2), FadeIn(pooled), FadeIn(pool_note), run_time=0.9)

        soft_box = panel(8.3, 2.5, color=GREY_C, fill=PANEL_DARK)
        soft_box.shift(RIGHT * 2.9 + UP * 0.75)
        local_chips = VGroup(*[chip(f"t{i}", color=BLUE, height=0.4, size=12, width=0.56) for i in range(5)])
        comp_chips = VGroup(*[chip(f"B{i}", color=GREEN, height=0.4, size=12, width=0.56) for i in range(4)])
        local_chips.arrange(RIGHT, buff=0.08)
        comp_chips.arrange(RIGHT, buff=0.08)
        divider = DashedLine(UP * 0.55, DOWN * 0.55, color=GREY_C)
        inner = VGroup(local_chips, divider, comp_chips).arrange(RIGHT, buff=0.3)
        inner.move_to(soft_box.get_center() + UP * 0.45)
        soft_label = txt("ONE softmax spans both branches:", size=18, color=WHITE, weight=BOLD)
        soft_label.next_to(soft_box.get_top(), DOWN, buff=0.16)
        soft_eq = eq(r"\mathrm{softmax}\big(\,[\,S_{\mathrm{local}}\;\big\|\;S_{\mathrm{compressed}}\,]\,\big)", size=30, color=INK)
        soft_eq.move_to(soft_box.get_center() + DOWN * 0.62)
        fit_width(soft_eq, 7.6)
        self.register_inside("softmax panel", soft_box, VGroup(soft_label, inner, soft_eq), margin=0.12)

        diff = self.caption_panel(
            "CSA vs HCA",
            "CSA: pool 4:1, keep only the best 32 blocks (top-k on summaries).\nHCA: pool 8:1, keep every completed block - no selection at all.",
            8.3,
            1.45,
            color=GREEN,
        )
        diff.next_to(soft_box, DOWN, buff=0.4)
        note = txt("crisp recent tokens compete directly with blurred history", size=17, color=AMBER)
        note.next_to(diff, DOWN, buff=0.28)
        fit_width(note, 8.3)

        self.beat_sync(5, 0.48)  # "The compressed variant pools four..."
        self.play(FadeIn(diff), run_time=0.9)
        self.beat_sync(5, 0.76)  # "...compete inside a single softmax"
        self.play(FadeIn(soft_box), FadeIn(soft_label), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(c) for c in [*local_chips, divider, *comp_chips]], lag_ratio=0.06), run_time=1.2)
        self.play(Write(soft_eq), run_time=1.0)
        self.play(FadeIn(note), run_time=0.7)
        self.finish_beat(BEATS[5], clear=True)

    # ---------- beat 6: causality audit ----------
    def audit_causality(self):
        title = self.top_title("The Audit, Part 1 - causality is testable", "gradients to the future must be EXACTLY zero")
        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=0.8)

        law = eq(r"\frac{\partial\, y_i}{\partial\, x_j} \;=\; 0 \quad \forall\, j > i", size=44, color=WHITE)
        law.shift(UP * 1.35)
        law_note = txt("not small. not approximately. exactly zero.", size=20, color=AMBER)
        law_note.next_to(law, DOWN, buff=0.3)
        self.beat_sync(6, 0.25)  # "Take the gradient of output i..."
        self.play(Write(law), run_time=1.3)
        self.beat_sync(6, 0.40)  # "Not small. Zero."
        self.play(FadeIn(law_note), run_time=0.7)

        rows = ["dsa", "lsa", "csa", "hca"]
        cols = ["t=33", "t=77", "t=101", "t=128", "dense limit"]
        table = VGroup()
        header = VGroup(*[txt(c, size=15, color=MUTED) for c in cols]).arrange(RIGHT, buff=0.62)
        table.add(header)
        for r in rows:
            row_cells = VGroup(*[txt("PASS", size=15, color=GREEN, weight=BOLD) for _ in cols]).arrange(RIGHT, buff=0.78)
            name = txt(r.upper(), size=15, color=VARIANT_COLORS[r], weight=BOLD)
            table.add(VGroup(name, row_cells).arrange(RIGHT, buff=0.5))
        table.arrange(DOWN, buff=0.24, aligned_edge=RIGHT).shift(DOWN * 1.25)
        box = panel(table.width + 0.7, table.height + 0.5, color=GREY_C, fill=PANEL_DARK)
        box.move_to(table)
        self.register_inside("audit table", box, table, margin=0.1)
        self.beat_sync(6, 0.50)  # "A sixty-eight-check suite runs this test..."
        self.play(FadeIn(box), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(r) for r in table], lag_ratio=0.12), run_time=2.0)

        tally = txt("68 checks: gradient causality - block padding at odd lengths - dense-limit equivalence", size=17, color=INK)
        fit_width(tally, 13.8)
        tally.next_to(box, DOWN, buff=0.3)
        self.beat_sync(6, 0.85)  # "Every gradient comes back exactly zero"
        self.play(FadeIn(tally), run_time=0.8)
        self.finish_beat(BEATS[6], clear=True)

    # ---------- beat 7: latent leak ----------
    def latent_leak(self):
        title = self.top_title("The Audit, Part 2 - a trap sleeping in the defaults", "softmax of an all-masked row is UNIFORM, not zero")
        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=0.8)

        n = 8
        cells = VGroup()
        for i in range(n):
            cell = VGroup(
                Rectangle(width=1.15, height=0.62, stroke_width=1.1, stroke_color=GREY_C, fill_color=PANEL, fill_opacity=0.9),
                mono("-1e9", size=15, color=RED),
            )
            cell[1].move_to(cell[0])
            cells.add(cell)
        cells.arrange(RIGHT, buff=0.06).shift(UP * 1.5)
        row_label = txt("one query row after masking: every score forbidden", size=18, color=INK)
        row_label.next_to(cells, UP, buff=0.22)
        self.beat_sync(7, 0.12)  # "forbidden scores are overwritten with minus ten to the ninth"
        self.play(FadeIn(row_label), LaggedStart(*[FadeIn(c) for c in cells], lag_ratio=0.07), run_time=1.4)

        arrow = Arrow(cells.get_bottom() + DOWN * 0.05, cells.get_bottom() + DOWN * 0.85, color=AMBER, stroke_width=3.4, buff=0)
        soft = eq(r"\mathrm{softmax}(\text{all equal}) = \tfrac{1}{n}\ \text{everywhere}", size=32, color=AMBER)
        soft.next_to(arrow, DOWN, buff=0.12)
        self.beat_sync(7, 0.42)  # "Softmax of a row where everything is equal..."
        self.play(GrowArrow(arrow), Write(soft), run_time=1.2)

        bars = VGroup()
        for i in range(n):
            bar = Rectangle(width=0.62, height=0.72, stroke_width=0.8, stroke_color=GREY_C, fill_color=RED if i >= 4 else BLUE, fill_opacity=0.85)
            bars.add(bar)
        bars.arrange(RIGHT, buff=0.16).next_to(soft, DOWN, buff=0.34)
        past_lab = txt("past", size=15, color=BLUE)
        future_lab = txt("FUTURE - read anyway", size=15, color=RED, weight=BOLD)
        past_lab.next_to(bars[1], DOWN, buff=0.12)
        future_lab.next_to(bars[6], DOWN, buff=0.12).shift(LEFT * 0.3)
        self.beat_sync(7, 0.52)  # "It is uniform attention"
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.12) for b in bars], lag_ratio=0.08), run_time=1.1)
        self.beat_sync(7, 0.58)  # "...including the future"
        self.play(FadeIn(past_lab), FadeIn(future_lab), run_time=0.8)

        consequences = self.caption_panel(
            "why it is dangerous",
            "triggered by topk=0 or window=0 - loss IMPROVES - nothing crashes, nothing warns.\nfix: reject any config that can produce an empty row, at construction time.",
            11.6,
            1.5,
            color=RED,
        )
        consequences.next_to(bars, DOWN, buff=0.75)
        self.beat_sync(7, 0.78)  # "The fix is boring and absolute"
        self.play(FadeIn(consequences, shift=UP * 0.15), run_time=1.0)
        self.finish_beat(BEATS[7], clear=True)

    # ---------- beat 8: harness bugs ----------
    def harness_bugs(self):
        title = self.top_title("The Audit, Part 3 - instrument errors", "invisible in any plot, fatal to the record")
        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=0.8)

        strip = VGroup(*[
            Rectangle(width=0.58, height=0.5, stroke_width=1.0, stroke_color=GREY_C, fill_color=DEEP_BLUE, fill_opacity=0.55)
            for _ in range(16)
        ]).arrange(RIGHT, buff=0.045)
        strip.shift(UP * 1.45 + LEFT * 1.6)
        strip_label = txt("dataset (byte stream)", size=15, color=MUTED)
        strip_label.next_to(strip, UP, buff=0.15).align_to(strip, LEFT)
        window = SurroundingRectangle(VGroup(*strip[10:15]), color=GREEN, buff=0.035, stroke_width=2.6)
        last_window = SurroundingRectangle(VGroup(*strip[11:16]), color=RED, buff=0.035, stroke_width=2.6)
        win_lab = txt("sampled windows", size=14, color=GREEN)
        win_lab.next_to(window, DOWN, buff=0.12)
        last_lab = txt("the LAST window: unreachable (off-by-one)", size=14, color=RED)
        last_lab.next_to(last_window, UP, buff=0.5).shift(RIGHT * 1.2)
        code = mono("randint(0, N - L - 1)   # exclusive high: final start never drawn", size=15, color=INK)
        code.next_to(strip, DOWN, buff=0.75).align_to(strip, LEFT)
        self.beat_sync(8, 0.08)
        self.play(FadeIn(strip_label), LaggedStart(*[FadeIn(c) for c in strip], lag_ratio=0.04), run_time=1.1)
        self.play(Create(window), FadeIn(win_lab), run_time=0.8)
        self.beat_sync(8, 0.22)  # "the final training window could never be drawn"
        self.play(Create(last_window), FadeIn(last_lab), run_time=0.9)
        self.play(FadeIn(code), run_time=0.8)

        log_box = panel(6.4, 2.2, color=GREY_C, fill=PANEL_DARK)
        log_box.shift(DOWN * 1.85 + LEFT * 3.9)
        lines = VGroup(
            mono('{"step": 200, "val": 2.91}', size=13, color=BLUE),
            mono('{"step": 400, "val": 2.74}', size=13, color=BLUE),
            mono('{"step": 200, "val": 3.05}   <- rerun, appended', size=13, color=RED),
            mono('{"step": 400, "val": 2.88}   <- two curves, one file', size=13, color=RED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.13)
        log_title = txt("metrics.jsonl opened in append mode", size=16, color=WHITE, weight=BOLD)
        inner = VGroup(log_title, lines).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to(log_box)
        self.register_inside("metrics log", log_box, inner, margin=0.12)
        self.beat_sync(8, 0.52)  # "the metrics file was opened in append mode"
        self.play(FadeIn(log_box), FadeIn(inner), run_time=1.1)

        fixes = self.caption_panel(
            "fixes",
            "sampler bound corrected (every window reachable)\nmetrics truncated per run - tok/s no longer counts eval time",
            7.2,
            1.6,
            color=GREEN,
        )
        fixes.next_to(log_box, RIGHT, buff=0.5).align_to(log_box, UP)
        self.beat_sync(8, 0.82)  # "Neither bug favors any variant..."
        self.play(FadeIn(fixes, shift=UP * 0.15), run_time=0.9)
        self.finish_beat(BEATS[8], clear=True)

    # ---------- beat 9: budget confound ----------
    def budget_confound(self):
        title = self.top_title("The Deepest Problem Was Not a Bug", "count what each query may touch - the budgets differ")
        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=0.8)

        budgets = {"dsa": 64, "lsa": 96, "csa": 96, "hca": 72}
        bars = Group()
        bar_map = {}
        base_y = -2.1
        for i, key in enumerate(("dsa", "lsa", "csa", "hca")):
            h = budgets[key] / 96 * 3.3
            bar = Rectangle(width=1.5, height=h, stroke_width=1.4, stroke_color=VARIANT_COLORS[key], fill_color=VARIANT_COLORS[key], fill_opacity=0.55)
            bar.move_to(np.array([-5.0 + i * 3.0, base_y + h / 2, 0]))
            num = txt(str(budgets[key]), size=26, color=WHITE, weight=BOLD)
            num.next_to(bar, UP, buff=0.14)
            name = txt(key.upper(), size=17, color=VARIANT_COLORS[key], weight=BOLD)
            name.next_to(bar, DOWN, buff=0.16)
            sub = txt(
                {"dsa": "64 oracle tokens", "lsa": "64 local + 32 recalled", "csa": "64 local + 32 blocks", "hca": "64 local + 8 blocks"}[key],
                size=13,
                color=MUTED,
            )
            sub.next_to(name, DOWN, buff=0.08)
            bars.add(Group(bar, num, name, sub))
            bar_map[key] = (bar, num, sub)
        self.beat_sync(9, 0.10)  # "Count what each query may actually touch..."
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in bars], lag_ratio=0.16), run_time=1.8)

        q = txt("if LSA wins ... is recall clever, or just RICHER?", size=24, color=AMBER, weight=BOLD)
        fit_width(q, 12.0)
        q.shift(UP * 1.9)
        self.beat_sync(9, 0.56)  # "is its recall clever, or is it just richer?"
        self.play(Write(q), run_time=1.4)

        new_vals = {"dsa": 96, "hca": 96}
        anims = []
        for key, val in new_vals.items():
            bar, num, sub = bar_map[key]
            new_h = val / 96 * 3.3
            new_bar = Rectangle(width=1.5, height=new_h, stroke_width=1.4, stroke_color=VARIANT_COLORS[key], fill_color=VARIANT_COLORS[key], fill_opacity=0.55)
            new_bar.move_to(np.array([bar.get_center()[0], base_y + new_h / 2, 0]))
            new_num = txt("96", size=26, color=GREEN, weight=BOLD)
            new_num.next_to(new_bar, UP, buff=0.14)
            anims += [ReplacementTransform(bar, new_bar), ReplacementTransform(num, new_num)]
        fix_lab = txt("matched: every variant gets 96 slots at the final query", size=20, color=GREEN)
        fit_width(fix_lab, 12.5)
        fix_lab.next_to(q, DOWN, buff=0.3)
        flags = mono("--dsa-topk 96   --hca-ratio 8   (every knob is now an explicit flag)", size=16, color=MUTED)
        fit_width(flags, 12.5)
        flags.next_to(fix_lab, DOWN, buff=0.2)
        self.beat_sync(9, 0.70)  # "So the budgets were matched"
        self.play(*anims, run_time=1.4)
        self.beat_sync(9, 0.85)  # "every sparsity knob became an explicit flag"
        self.play(FadeIn(fix_lab), FadeIn(flags), run_time=1.0)
        self.finish_beat(BEATS[9], clear=True)

    # ---------- beat 10: results ----------
    def results_beat(self):
        title = self.top_title("The Fixed Rerun - 12x data, matched budgets, two seeds", "50M-char 7-source mix - 6 layers - 4.9M params - 49M tokens/run")
        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=0.8)

        if not self.results:
            placeholder = txt("[ results pending training completion ]", size=26, color=MUTED)
            self.play(FadeIn(placeholder), run_time=0.6)
            self.finish_beat(BEATS[10], clear=True)
            return

        curves = self.results["curves"]
        table_rows = self.results["table"]
        all_tokens = curves["dsa"]["tokens"]
        max_tok = all_tokens[-1]
        y_vals = [v for c in curves.values() for v in c["val_loss_mean"]]
        y_min, y_max = min(y_vals), min(3.2, max(y_vals))

        axes = Axes(
            x_range=[0, max_tok * 1.02, max_tok / 5],
            y_range=[y_min - 0.05, y_max + 0.1, 0.3],
            x_length=7.6,
            y_length=4.3,
            axis_config={"stroke_color": GREY_C, "stroke_width": 1.6, "include_ticks": True, "include_tip": False},
        )
        axes.shift(LEFT * 3.5 + DOWN * 0.75)
        x_lab = txt("training tokens", size=15, color=MUTED).next_to(axes.x_axis, DOWN, buff=0.18)
        y_lab = txt("val loss (nats)", size=15, color=MUTED).rotate(np.pi / 2).next_to(axes.y_axis, LEFT, buff=0.18)
        self.beat_sync(10, 0.10)
        self.play(Create(axes), FadeIn(x_lab), FadeIn(y_lab), run_time=1.2)

        plots = VGroup()
        for key in ("dsa", "lsa", "csa", "hca"):
            pts = [axes.coords_to_point(t, v) for t, v in zip(curves[key]["tokens"], curves[key]["val_loss_mean"]) if v <= y_max + 0.09]
            line = VGroup()
            for a, b in zip(pts[:-1], pts[1:]):
                line.add(Line(a, b, color=VARIANT_COLORS[key], stroke_width=3.0))
            plots.add(line)
        self.beat_sync(10, 0.24)  # "the four curves finally tell a story"
        self.play(LaggedStart(*[Create(p) for p in plots], lag_ratio=0.15), run_time=2.6)

        tbl_box = panel(7.0, 3.4, color=GREY_C, fill=PANEL_DARK)
        tbl_box.shift(RIGHT * 4.3 + DOWN * 0.75)
        header = VGroup(
            txt("variant", size=15, color=MUTED),
            txt("final val (2 seeds)", size=15, color=MUTED),
            txt("spread", size=15, color=MUTED),
        ).arrange(RIGHT, buff=0.75)
        rows = VGroup(header)
        for row in table_rows:
            key = row["variant"]
            r = VGroup(
                txt(key.upper(), size=17, color=VARIANT_COLORS[key], weight=BOLD),
                mono(f"{row['final_val_mean']:.4f}", size=17, color=WHITE),
                mono(f"±{row['final_val_spread'] / 2:.4f}", size=15, color=MUTED),
            ).arrange(RIGHT, buff=1.05)
            rows.add(r)
        rows.arrange(DOWN, buff=0.28, aligned_edge=LEFT).move_to(tbl_box)
        self.register_inside("results table", tbl_box, rows, margin=0.14)
        self.beat_sync(10, 0.35)  # "Both seeds agree on exactly the same order"
        self.play(FadeIn(tbl_box), LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.12), run_time=1.8)

        best = table_rows[0]
        worst = table_rows[-1]
        gap = worst["final_val_mean"] - best["final_val_mean"]
        spread = max(r["final_val_spread"] for r in table_rows)
        verdict_text = (
            f"gap {best['variant'].upper()} to {worst['variant'].upper()}: {gap:.4f} nats vs cross-seed spread {spread:.4f}"
        )
        verdict = txt(verdict_text, size=19, color=AMBER)
        fit_width(verdict, 6.8)
        verdict.next_to(tbl_box, DOWN, buff=0.35)
        self.beat_sync(10, 0.55)  # "more than twice the largest seed-to-seed wobble"
        self.play(FadeIn(verdict), run_time=0.9)
        self.finish_beat(BEATS[10], clear=True)

    # ---------- beat 11: synthesis ----------
    def synthesis(self):
        title = self.top_title("Calibrate the Instrument Before Reading the Dial")
        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=0.8)

        bounds = self.caption_panel(
            "boundaries - stated plainly",
            "4.9M params is a toy - byte-level loss is not benchmark accuracy\noracle top-k is an upper bound, not a mechanism - tok/s here is masking overhead, not kernels",
            12.6,
            1.7,
            color=MUTED,
        )
        bounds.shift(UP * 1.35)
        self.beat_sync(11, 0.05)
        self.play(FadeIn(bounds, shift=UP * 0.15), run_time=1.0)

        checks = VGroup(
            chip("1  prove causality with gradients - exactly zero", color=BLUE, size=17, height=0.62),
            chip("2  make silent failure modes unrepresentable", color=RED, size=17, height=0.62),
            chip("3  match the budgets - vary ONE thing", color=GREEN, size=17, height=0.62),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        checks.next_to(bounds, DOWN, buff=0.55)
        self.beat_sync(11, 0.52)  # "you calibrate it: prove causality..."
        self.play(LaggedStart(*[FadeIn(c, shift=RIGHT * 0.2) for c in checks], lag_ratio=0.3), run_time=2.0)

        closing = txt(
            "the most dangerous experiment is not the one that crashes -\nit is the one that runs perfectly and measures the wrong thing",
            size=23,
            color=AMBER,
            line_spacing=1.0,
        )
        fit_width(closing, 13.4)
        closing.next_to(checks, DOWN, buff=0.55)
        self.beat_sync(11, 0.82)  # "The most dangerous experiment..."
        self.play(Write(closing), run_time=2.2)
        self.finish_beat(BEATS[11], clear=False)
