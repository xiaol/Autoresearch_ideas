#!/usr/bin/env python3
"""Render a 4K Manim-style conclusion card with Cairo and ffmpeg.

Manim Community is not available on this machine because the native pangocairo
development package is missing. This renderer keeps the Manim animation
discipline: timed visual beats, persistent visual carriers, and a layout report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import cairo


WIDTH = 3840
HEIGHT = 2160
FPS = 30
DURATION = 28.0
BG = (0.035, 0.06, 0.12)
PANEL = (0.055, 0.085, 0.165)
PANEL_DARK = (0.025, 0.04, 0.085)
LINE = (0.24, 0.32, 0.46)
TEXT = (0.94, 0.97, 1.0)
MUTED = (0.63, 0.70, 0.82)
GREEN = (0.18, 0.90, 0.62)
RED = (0.92, 0.25, 0.28)
BLUE = (0.34, 0.55, 0.95)
CYAN = (0.25, 0.80, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the 4K MLP efficiency conclusion insert.")
    parser.add_argument("--out", type=Path, default=Path("narration-assets/conclusion_insight_4k_silent.mp4"))
    parser.add_argument("--poster", type=Path, default=Path("narration-assets/conclusion_insight_poster.png"))
    parser.add_argument("--layout-report", type=Path, default=Path("narration-assets/conclusion_insight_layout.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.poster.parent.mkdir(parents=True, exist_ok=True)
    args.layout_report.parent.mkdir(parents=True, exist_ok=True)
    layout = build_layout()
    verify_layout(layout)
    args.layout_report.write_text(json.dumps(layout, indent=2), encoding="utf-8")
    render_video(args.out)
    render_frame(DURATION * 0.72).write_to_png(str(args.poster))
    print(f"Wrote {args.out}")
    print(f"Wrote {args.poster}")
    print(f"Wrote {args.layout_report}")
    return 0


def build_layout() -> dict[str, Any]:
    objects = [
        {"name": "title", "bbox": [260, 170, 2100, 150]},
        {"name": "scope-scale", "bbox": [260, 380, 3320, 220]},
        {"name": "qwen-card", "bbox": [340, 780, 1450, 690]},
        {"name": "rwkv-card", "bbox": [2050, 780, 1450, 690]},
        {"name": "checklist", "bbox": [520, 1620, 2800, 330]},
    ]
    return {"frame": {"width": WIDTH, "height": HEIGHT}, "objects": objects}


def verify_layout(layout: dict[str, Any]) -> None:
    width = layout["frame"]["width"]
    height = layout["frame"]["height"]
    failures = []
    for item in layout["objects"]:
        x, y, w, h = item["bbox"]
        ok = x >= 0 and y >= 0 and x + w <= width and y + h <= height
        item["withinFrame"] = ok
        if not ok:
            failures.append(item["name"])
    if failures and os.environ.get("MANIM_LAYOUT_STRICT") == "1":
        raise RuntimeError(f"Layout overflow: {', '.join(failures)}")


def render_video(out_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgra",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    frame_count = int(DURATION * FPS)
    for frame in range(frame_count):
        surface = render_frame(frame / FPS)
        process.stdin.write(surface.get_data())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed while rendering conclusion insert.")


def render_frame(t: float) -> cairo.ImageSurface:
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    ctx = cairo.Context(surface)
    draw_background(ctx, t)
    draw_scope(ctx, t)
    draw_network_carrier(ctx, t)
    draw_metric_cards(ctx, t)
    draw_checklist(ctx, t)
    surface.flush()
    return surface


def draw_background(ctx: cairo.Context, t: float) -> None:
    set_source(ctx, BG)
    ctx.rectangle(0, 0, WIDTH, HEIGHT)
    ctx.fill()
    grid_alpha = 0.08
    set_source(ctx, (0.20, 0.28, 0.44), grid_alpha)
    ctx.set_line_width(2)
    offset = (t * 18) % 160
    for x in range(-160, WIDTH + 160, 160):
        ctx.move_to(x + offset, 0)
        ctx.line_to(x + offset - 420, HEIGHT)
    for y in range(160, HEIGHT, 160):
        ctx.move_to(0, y)
        ctx.line_to(WIDTH, y)
    ctx.stroke()

    glow = cairo.RadialGradient(WIDTH * 0.5, HEIGHT * 0.38, 80, WIDTH * 0.5, HEIGHT * 0.38, 1400)
    glow.add_color_stop_rgba(0, 0.06, 0.22, 0.30, 0.55)
    glow.add_color_stop_rgba(1, 0.02, 0.03, 0.08, 0.0)
    ctx.set_source(glow)
    ctx.rectangle(0, 0, WIDTH, HEIGHT)
    ctx.fill()


def draw_scope(ctx: cairo.Context, t: float) -> None:
    a = fade(t, 0.0, 1.2)
    draw_text(ctx, "Conclusion: efficiency is a layer diagnostic", 260, 235, 84, TEXT, alpha=a, weight=True)
    draw_text(
        ctx,
        "Useful for selected blocks. Not yet a full model benchmark.",
        265,
        325,
        42,
        MUTED,
        alpha=fade(t, 1.2, 1.4),
    )

    scale_alpha = fade(t, 2.0, 1.6)
    x, y, w, h = 260, 430, 3320, 130
    rounded_rect(ctx, x, y, w, h, 36)
    set_source(ctx, PANEL_DARK, 0.92 * scale_alpha)
    ctx.fill_preserve()
    set_source(ctx, LINE, 0.9 * scale_alpha)
    ctx.set_line_width(3)
    ctx.stroke()
    draw_text(ctx, "Per-layer diagnostic", x + 70, y + 80, 40, TEXT, alpha=scale_alpha, weight=True)
    draw_text(ctx, "Full model benchmark", x + w - 650, y + 80, 40, MUTED, alpha=scale_alpha, weight=True)
    set_source(ctx, GREEN, scale_alpha)
    ctx.set_line_width(12)
    ctx.set_line_cap(cairo.LineCap.ROUND)
    ctx.move_to(x + 520, y + 66)
    ctx.line_to(x + 2140, y + 66)
    ctx.stroke()
    set_source(ctx, (1, 1, 1), scale_alpha)
    ctx.arc(x + 520, y + 66, 17, 0, math.tau)
    ctx.fill()
    set_source(ctx, MUTED, scale_alpha * 0.45)
    ctx.set_line_width(5)
    ctx.move_to(x + 2140, y + 66)
    ctx.line_to(x + w - 740, y + 66)
    ctx.stroke()


def draw_network_carrier(ctx: cairo.Context, t: float) -> None:
    alpha = 0.18 * fade(t, 0.8, 2.0)
    centers = [760, 1060, 1360, 1660]
    rows = 13
    top = 760
    gap = 42
    for col, x in enumerate(centers):
        for row in range(rows):
            wave = 0.5 + 0.5 * math.sin(t * 1.2 + row * 0.9 + col * 1.7)
            radius = 9 + 9 * wave
            y = top + row * gap
            set_source(ctx, (0.80, 0.84, 0.88), alpha * (0.42 + wave * 0.58))
            ctx.arc(x + math.sin(row * 1.3 + col) * 16, y, radius, 0, math.tau)
            ctx.fill()
    line_alpha = 0.12 * fade(t, 1.6, 1.5)
    for c in range(len(centers) - 1):
        for row in range(0, rows, 2):
            y0 = top + row * gap
            y1 = top + ((row * 3 + c * 4) % rows) * gap
            color = GREEN if (row + c) % 3 else RED
            set_source(ctx, color, line_alpha)
            ctx.set_line_width(3)
            ctx.move_to(centers[c], y0)
            ctx.line_to(centers[c + 1], y1)
            ctx.stroke()


def draw_metric_cards(ctx: cairo.Context, t: float) -> None:
    q_alpha = fade(t, 7.5, 1.5)
    r_alpha = fade(t, 9.5, 1.5)
    draw_model_card(
        ctx,
        "Qwen3.5 0.8B",
        "Transformer FFN block",
        "22.0",
        "MFLOPs/token",
        "0.028",
        "Delta/MFLOP",
        340,
        780,
        q_alpha,
        BLUE,
    )
    draw_model_card(
        ctx,
        "RWKV-7 0.1B",
        "Recurrent mix block",
        "14.2",
        "MFLOPs/token",
        "0.043",
        "Delta/MFLOP",
        2050,
        780,
        r_alpha,
        GREEN,
    )
    arrow_alpha = fade(t, 15.0, 1.2)
    set_source(ctx, GREEN, arrow_alpha)
    ctx.set_line_width(12)
    ctx.set_line_cap(cairo.LineCap.ROUND)
    ctx.move_to(1790, 1110)
    ctx.curve_to(1880, 1030, 1950, 1030, 2050, 1110)
    ctx.stroke()
    draw_text(ctx, "higher movement per estimated compute in this snapshot", 1445, 1035, 34, GREEN, alpha=arrow_alpha, weight=True)


def draw_model_card(
    ctx: cairo.Context,
    name: str,
    subtitle: str,
    mflops: str,
    mflops_label: str,
    delta: str,
    delta_label: str,
    x: float,
    y: float,
    alpha: float,
    accent: tuple[float, float, float],
) -> None:
    shift = 80 * (1 - ease(alpha))
    rounded_rect(ctx, x, y + shift, 1450, 690, 34)
    set_source(ctx, PANEL, 0.90 * alpha)
    ctx.fill_preserve()
    set_source(ctx, accent, 0.80 * alpha)
    ctx.set_line_width(4)
    ctx.stroke()
    draw_text(ctx, name, x + 70, y + 95 + shift, 58, TEXT, alpha=alpha, weight=True)
    draw_text(ctx, subtitle, x + 70, y + 155 + shift, 34, MUTED, alpha=alpha)
    draw_metric(ctx, x + 70, y + 245 + shift, 600, 190, mflops, mflops_label, accent, alpha)
    draw_metric(ctx, x + 780, y + 245 + shift, 600, 190, delta, delta_label, GREEN if delta == "0.043" else CYAN, alpha)
    draw_bar(ctx, x + 80, y + 545 + shift, 1280, 36, float(mflops) / 24.0, accent, alpha, "Compute estimate")
    draw_bar(ctx, x + 80, y + 610 + shift, 1280, 36, float(delta) / 0.05, GREEN, alpha, "Delta per MFLOP")


def draw_metric(
    ctx: cairo.Context,
    x: float,
    y: float,
    w: float,
    h: float,
    value: str,
    label: str,
    color: tuple[float, float, float],
    alpha: float,
) -> None:
    rounded_rect(ctx, x, y, w, h, 24)
    set_source(ctx, PANEL_DARK, 0.88 * alpha)
    ctx.fill_preserve()
    set_source(ctx, LINE, 0.70 * alpha)
    ctx.set_line_width(3)
    ctx.stroke()
    draw_text(ctx, value, x + 42, y + 88, 72, color, alpha=alpha, weight=True)
    draw_text(ctx, label, x + 42, y + 142, 34, MUTED, alpha=alpha, weight=True)


def draw_bar(
    ctx: cairo.Context,
    x: float,
    y: float,
    w: float,
    h: float,
    value: float,
    color: tuple[float, float, float],
    alpha: float,
    label: str,
) -> None:
    draw_text(ctx, label, x, y - 14, 28, MUTED, alpha=alpha)
    rounded_rect(ctx, x + 310, y - 32, w - 310, h, h / 2)
    set_source(ctx, PANEL_DARK, 0.75 * alpha)
    ctx.fill()
    rounded_rect(ctx, x + 310, y - 32, (w - 310) * clamp(value), h, h / 2)
    set_source(ctx, color, 0.88 * alpha)
    ctx.fill()


def draw_checklist(ctx: cairo.Context, t: float) -> None:
    alpha = fade(t, 20.5, 1.4)
    x, y, w, h = 520, 1620, 2800, 330
    rounded_rect(ctx, x, y, w, h, 34)
    set_source(ctx, PANEL_DARK, 0.90 * alpha)
    ctx.fill_preserve()
    set_source(ctx, LINE, 0.75 * alpha)
    ctx.set_line_width(3)
    ctx.stroke()
    draw_text(ctx, "Before claiming global efficiency, capture:", x + 70, y + 80, 44, TEXT, alpha=alpha, weight=True)
    items = ["real intermediate activations", "latency", "memory"]
    for index, item in enumerate(items):
        ix = x + 95 + index * 880
        iy = y + 205
        set_source(ctx, GREEN, alpha)
        ctx.set_line_width(9)
        ctx.move_to(ix, iy - 5)
        ctx.line_to(ix + 26, iy + 25)
        ctx.line_to(ix + 78, iy - 45)
        ctx.stroke()
        draw_text(ctx, item, ix + 110, iy + 5, 36, TEXT, alpha=alpha)


def rounded_rect(ctx: cairo.Context, x: float, y: float, w: float, h: float, r: float) -> None:
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    ctx.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    ctx.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    ctx.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    ctx.close_path()


def draw_text(
    ctx: cairo.Context,
    text: str,
    x: float,
    y: float,
    size: float,
    color: tuple[float, float, float],
    alpha: float = 1.0,
    weight: bool = False,
) -> None:
    ctx.select_font_face("Sans", cairo.FontSlant.NORMAL, cairo.FontWeight.BOLD if weight else cairo.FontWeight.NORMAL)
    ctx.set_font_size(size)
    set_source(ctx, color, alpha)
    ctx.move_to(x, y)
    ctx.show_text(text)


def set_source(ctx: cairo.Context, color: tuple[float, float, float], alpha: float = 1.0) -> None:
    ctx.set_source_rgba(color[0], color[1], color[2], alpha)


def fade(t: float, start: float, duration: float) -> float:
    return ease(clamp((t - start) / duration))


def ease(x: float) -> float:
    x = clamp(x)
    return x * x * (3 - 2 * x)


def clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


if __name__ == "__main__":
    raise SystemExit(main())
