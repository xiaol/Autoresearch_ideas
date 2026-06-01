from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


W, H = 1920, 1080
FPS = 24
BG = "#f7f8fb"
INK = "#17202a"
MUTED = "#425466"
BLUE = "#195fb8"
TEAL = "#039b8e"
ORANGE = "#c85a17"
GRAY = "#808a9a"
LIGHT = "#e7ebf2"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf" if bold else "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TITLE = font(58, bold=True)
FONT_H1 = font(44, bold=True)
FONT_H2 = font(34, bold=True)
FONT_BODY = font(28)
FONT_SMALL = font(22)
FONT_TINY = font(18)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    fnt: ImageFont.ImageFont,
    fill: str = INK,
    width: int = 900,
    line_gap: int = 10,
) -> int:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if text_size(draw, trial, fnt)[0] <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    x, y = xy
    line_height = text_size(draw, "Ag", fnt)[1] + line_gap
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += line_height
    return y


def draw_centered(
    draw: ImageDraw.ImageDraw,
    y: int,
    text: str,
    *,
    fnt: ImageFont.ImageFont,
    fill: str = INK,
) -> int:
    tw, th = text_size(draw, text, fnt)
    draw.text(((W - tw) // 2, y), text, font=fnt, fill=fill)
    return y + th


def rounded_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str,
    outline: str = "#cfd6e3",
    radius: int = 18,
    width: int = 2,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def base_frame() -> Image.Image:
    return Image.new("RGB", (W, H), BG)


def curve_points(
    center: tuple[float, float],
    rx: float,
    ry: float,
    n: int = 160,
    wobble: float = 0.18,
) -> list[tuple[int, int]]:
    points = []
    cx, cy = center
    for i in range(n):
        t = 2 * math.pi * i / (n - 1)
        rmod = 1.0 + wobble * math.sin(2 * t + 0.6)
        x = cx + rx * rmod * math.cos(t)
        y = cy + ry * math.sin(t) + 35 * math.sin(3 * t)
        points.append((int(x), int(y)))
    return points


def interp(points: list[tuple[int, int]], alpha: float) -> tuple[int, int]:
    alpha = max(0.0, min(1.0, alpha))
    pos = alpha * (len(points) - 1)
    i = int(pos)
    j = min(i + 1, len(points) - 1)
    f = pos - i
    x = points[i][0] * (1 - f) + points[j][0] * f
    y = points[i][1] * (1 - f) + points[j][1] * f
    return int(x), int(y)


def line_interp(a: tuple[int, int], b: tuple[int, int], alpha: float) -> tuple[int, int]:
    return int(a[0] * (1 - alpha) + b[0] * alpha), int(a[1] * (1 - alpha) + b[1] * alpha)


def draw_marker(draw: ImageDraw.ImageDraw, xy: tuple[int, int], *, kind: str, color: str) -> None:
    x, y = xy
    if kind == "square":
        draw.rectangle((x - 13, y - 13, x + 13, y + 13), fill=color, outline="#000000", width=2)
    else:
        draw.ellipse((x - 13, y - 13, x + 13, y + 13), fill=color, outline="#333333", width=2)


@lru_cache(maxsize=16)
def load_gif_frames(path_str: str, size: tuple[int, int]) -> tuple[Image.Image, ...]:
    path = Path(path_str)
    gif = Image.open(path)
    frames = []
    for idx in range(getattr(gif, "n_frames", 1)):
        gif.seek(idx)
        frame = gif.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        frames.append(frame.copy())
    return tuple(frames)


class VideoWriter:
    def __init__(self, out_path: Path, *, fps: int = FPS):
        self.out_path = out_path
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg not found")
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{W}x{H}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.frame_count = 0

    def write(self, image: Image.Image) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("ffmpeg stdin closed")
        self.proc.stdin.write(image.convert("RGB").tobytes())
        self.frame_count += 1

    def close(self) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        code = self.proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg exited with code {code}")


def add_segment(
    writer: VideoWriter,
    duration: float,
    draw_fn: Callable[[Image.Image, ImageDraw.ImageDraw, float], None],
) -> None:
    total = int(round(duration * FPS))
    for frame_idx in range(total):
        alpha = frame_idx / max(1, total - 1)
        image = base_frame()
        draw = ImageDraw.Draw(image)
        draw_fn(image, draw, alpha)
        writer.write(image)


def title_slide(image: Image.Image, draw: ImageDraw.ImageDraw, alpha: float) -> None:
    draw_centered(draw, 250, "The world inside neural networks", fnt=FONT_TITLE)
    draw_centered(draw, 330, "Why curved geometry matters for steering", fnt=FONT_H2, fill=BLUE)
    y = draw_wrapped(
        draw,
        (445, 455),
        "Goodfire's neural geometry article argues that models often learn internal shapes that mirror structured concepts in the world: days, months, colors, maps, and more.",
        fnt=FONT_BODY,
        fill=INK,
        width=1030,
        line_gap=12,
    )
    draw_wrapped(
        draw,
        (445, y + 28),
        "Our question: can the same idea work in a state-based model like RWKV?",
        fnt=FONT_BODY,
        fill=MUTED,
        width=1030,
        line_gap=12,
    )
    draw.text((60, 1000), "Source: Goodfire, The World Inside Neural Networks", font=FONT_SMALL, fill=MUTED)


def spaces_slide(image: Image.Image, draw: ImageDraw.ImageDraw, alpha: float) -> None:
    draw_centered(draw, 70, "Two spaces, one behavior", fnt=FONT_H1)
    draw_wrapped(
        draw,
        (210, 150),
        "Activation space is where hidden states live. Behavior space is what the model does next: its output probabilities.",
        fnt=FONT_BODY,
        width=1500,
        fill=MUTED,
    )
    left_box = (150, 285, 820, 810)
    right_box = (1100, 285, 1770, 810)
    rounded_rect(draw, left_box, fill="#ffffff")
    rounded_rect(draw, right_box, fill="#ffffff")
    draw.text((300, 315), "Activation space", font=FONT_H2, fill=INK)
    draw.text((1245, 315), "Behavior space", font=FONT_H2, fill=BLUE)
    curve = curve_points((485, 570), 210, 130)
    draw.line(curve, fill="#111111", width=7, joint="curve")
    for i, label in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
        x, y = interp(curve, i / 7)
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=BLUE)
        draw.text((x + 12, y - 10), label, font=FONT_TINY, fill=MUTED)
    bars = [0.62, 0.24, 0.06, 0.03, 0.02, 0.02, 0.01]
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for i, (name, value) in enumerate(zip(names, bars, strict=True)):
        y = 430 + i * 48
        draw.text((1215, y - 10), name, font=FONT_SMALL, fill=INK)
        draw.rounded_rectangle((1300, y, 1660, y + 24), radius=12, fill=LIGHT)
        draw.rounded_rectangle((1300, y, int(1300 + 360 * value), y + 24), radius=12, fill=BLUE)
    draw.line((845, 545, 1075, 545), fill=MUTED, width=4)
    draw.polygon([(1075, 545), (1045, 530), (1045, 560)], fill=MUTED)
    draw_wrapped(
        draw,
        (720, 610),
        "Patch one hidden state, then read the final output distribution.",
        fnt=FONT_SMALL,
        fill=MUTED,
        width=500,
    )


def steering_paths_slide(image: Image.Image, draw: ImageDraw.ImageDraw, alpha: float) -> None:
    draw_centered(draw, 70, "The core test: straight line or manifold path?", fnt=FONT_H1)
    draw_wrapped(
        draw,
        (280, 145),
        "If concepts live on a curve, the shortest-looking straight line can pass through hidden states the model almost never uses.",
        fnt=FONT_BODY,
        width=1360,
        fill=MUTED,
    )
    curve = curve_points((960, 560), 420, 210)
    draw.line(curve, fill="#111111", width=8, joint="curve")
    start = interp(curve, 0.08)
    end = interp(curve, 0.58)
    draw.line((start[0], start[1], end[0], end[1]), fill=GRAY, width=7)
    for i in range(7):
        x, y = interp(curve, i / 7)
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=BLUE)
    draw_marker(draw, interp(curve, 0.08 + 0.50 * alpha), kind="square", color="#111111")
    draw_marker(draw, line_interp(start, end, alpha), kind="circle", color="#777777")
    draw.text((520, 840), "square: follows the concept manifold", font=FONT_BODY, fill=INK)
    draw.text((520, 890), "circle: linear steering cuts across", font=FONT_BODY, fill=GRAY)


def method_slide(image: Image.Image, draw: ImageDraw.ImageDraw, alpha: float) -> None:
    draw_centered(draw, 70, "Goodfire's method in plain language", fnt=FONT_H1)
    steps = [
        ("1", "Collect activations", "Run many prompts and save hidden states."),
        ("2", "Fit a manifold", "Average by concept, then fit a smooth curve."),
        ("3", "Patch paths", "Replace the hidden state along a path."),
        ("4", "Measure behavior", "Check whether outputs stay on the natural behavior manifold."),
    ]
    for idx, (num, title, body) in enumerate(steps):
        x = 170 + idx * 435
        rounded_rect(draw, (x, 300, x + 360, 680), fill="#ffffff")
        draw.ellipse((x + 130, 330, x + 230, 430), fill=BLUE)
        tw, th = text_size(draw, num, FONT_H1)
        draw.text((x + 180 - tw // 2, 378 - th // 2), num, font=FONT_H1, fill="#ffffff")
        draw.text((x + 35, 475), title, font=FONT_H2, fill=INK)
        draw_wrapped(draw, (x + 35, 535), body, fnt=FONT_SMALL, fill=MUTED, width=290)
        if idx < len(steps) - 1:
            draw.line((x + 370, 490, x + 425, 490), fill=MUTED, width=4)
            draw.polygon([(x + 425, 490), (x + 405, 478), (x + 405, 502)], fill=MUTED)
    draw_wrapped(
        draw,
        (310, 765),
        "The claim is not just that internal states are curved. The stronger claim is that activation geometry and output behavior geometry are approximately aligned.",
        fnt=FONT_BODY,
        fill=INK,
        width=1300,
    )


def legend_slide(image: Image.Image, draw: ImageDraw.ImageDraw, alpha: float) -> None:
    draw_centered(draw, 70, "How to read the steering GIFs", fnt=FONT_H1)
    draw_wrapped(
        draw,
        (250, 145),
        "Each moving marker is the model's output distribution projected into behavior space. The natural concept centroids form a reference path.",
        fnt=FONT_BODY,
        fill=MUTED,
        width=1420,
    )
    curve = curve_points((760, 565), 310, 175)
    draw.line(curve, fill="#111111", width=7, joint="curve")
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for i, label in enumerate(labels):
        x, y = interp(curve, i / 7)
        draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill=BLUE, outline="#0f3565", width=2)
        draw.text((x + 16, y - 11), label, font=FONT_SMALL, fill=INK)
    start = interp(curve, 0.08)
    end = interp(curve, 0.58)
    draw.line((start[0], start[1], end[0], end[1]), fill=GRAY, width=6)
    draw_marker(draw, interp(curve, 0.08 + 0.50 * alpha), kind="square", color="#111111")
    draw_marker(draw, line_interp(start, end, alpha), kind="circle", color="#777777")

    legend_x = 1190
    rounded_rect(draw, (legend_x, 330, 1690, 765), fill="#ffffff")
    draw.ellipse((legend_x + 45, 385, legend_x + 75, 415), fill=BLUE, outline="#0f3565", width=2)
    draw.text((legend_x + 105, 380), "blue dots and labels", font=FONT_H2, fill=INK)
    draw_wrapped(draw, (legend_x + 105, 430), "Natural weekday or month outputs, connected as the fitted behavior manifold.", fnt=FONT_SMALL, fill=MUTED, width=340)
    draw.rectangle((legend_x + 45, 530, legend_x + 75, 560), fill="#111111", outline="#000000", width=2)
    draw.text((legend_x + 105, 522), "black square", font=FONT_H2, fill=INK)
    draw_wrapped(draw, (legend_x + 105, 572), "Manifold steering: patch hidden states along the fitted activation curve.", fnt=FONT_SMALL, fill=MUTED, width=340)
    draw.ellipse((legend_x + 45, 665, legend_x + 75, 695), fill="#777777", outline="#333333", width=2)
    draw.text((legend_x + 105, 657), "gray dot", font=FONT_H2, fill=INK)
    draw_wrapped(draw, (legend_x + 105, 707), "Linear steering: patch hidden states along the straight chord between the same endpoints.", fnt=FONT_SMALL, fill=MUTED, width=340)
    draw_wrapped(
        draw,
        (260, 865),
        "If the square stays on the reference path while the dot cuts away, the curved activation path is preserving more natural output behavior.",
        fnt=FONT_BODY,
        fill=BLUE,
        width=1400,
    )


def gif_slide(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    alpha: float,
    *,
    headline: str,
    caption: str,
    gif_paths: list[Path],
    labels: list[str],
) -> None:
    draw_centered(draw, 50, headline, fnt=FONT_H1)
    draw_wrapped(draw, (220, 118), caption, fnt=FONT_SMALL, fill=MUTED, width=1480)
    if len(gif_paths) == 1:
        size = (1450, 754)
        frames = load_gif_frames(str(gif_paths[0]), size)
        frame = frames[int(alpha * (len(frames) - 1))]
        image.paste(frame, ((W - size[0]) // 2, 245))
        draw.text((245, 990), labels[0], font=FONT_SMALL, fill=INK)
    else:
        size = (860, 447)
        for idx, path in enumerate(gif_paths):
            frames = load_gif_frames(str(path), size)
            frame = frames[int(alpha * (len(frames) - 1))]
            x = 80 + idx * 940
            y = 260
            image.paste(frame, (x, y))
            draw.text((x + 10, y - 36), labels[idx], font=FONT_SMALL, fill=INK)
    draw.text((70, 1010), "In these GIFs: square = manifold steering, circle = linear steering.", font=FONT_SMALL, fill=MUTED)


def metrics_slide(image: Image.Image, draw: ImageDraw.ImageDraw, alpha: float, rows: list[dict]) -> None:
    draw_centered(draw, 70, "What changed in our reproduction?", fnt=FONT_H1)
    draw_wrapped(
        draw,
        (260, 145),
        "We reran the experiments with matched endpoints: linear and manifold steering start and end at exactly the same hidden states. Only the intermediate path differs.",
        fnt=FONT_BODY,
        fill=MUTED,
        width=1400,
    )
    cols = [210, 540, 800, 1045, 1290, 1535]
    headers = ["Task", "Model", "Layer", "r manifold", "r linear", "Endpoint delta"]
    y0 = 300
    rounded_rect(draw, (170, y0 - 35, 1770, y0 + 355), fill="#ffffff")
    for x, header in zip(cols, headers, strict=True):
        draw.text((x, y0), header, font=FONT_SMALL, fill=INK)
    draw.line((200, y0 + 42, 1730, y0 + 42), fill="#cfd6e3", width=2)
    for r_idx, row in enumerate(rows):
        y = y0 + 78 + r_idx * 67
        values = [
            row["task"],
            row["model"],
            str(row["layer"]),
            f"{row['isometry_geometric_r']:.3f}",
            f"{row['isometry_linear_r']:.3f}",
            "0",
        ]
        for x, value in zip(cols, values, strict=True):
            color = BLUE if value == f"{row['isometry_geometric_r']:.3f}" else INK
            draw.text((x, y), value, font=FONT_SMALL, fill=color)
    draw_wrapped(
        draw,
        (260, 760),
        "The geometry signal survives in RWKV: activation-manifold distances correlate strongly with behavior-manifold distances. But these tiny models are not good at the tasks yet, so this is a geometry sanity check, not a capability claim.",
        fnt=FONT_BODY,
        fill=INK,
        width=1400,
    )


def insight_slide(image: Image.Image, draw: ImageDraw.ImageDraw, alpha: float) -> None:
    draw_centered(draw, 90, "New insight from the RWKV version", fnt=FONT_H1)
    bullets = [
        ("Where to patch", "For RWKV, the closest analogue to a transformer residual stream is the last-token block output after time-mix and channel-mix residual updates."),
        ("Endpoint matching matters", "If endpoints differ, the behavior-space picture can mislead. Our corrected runs remove that confound."),
        ("State models can be tested", "The manifold-steering idea is not transformer-only. The same activation-to-behavior geometry can be probed in RWKV."),
    ]
    y = 250
    for title, body in bullets:
        rounded_rect(draw, (260, y, 1660, y + 170), fill="#ffffff")
        draw.ellipse((300, y + 58, 345, y + 103), fill=TEAL)
        draw.text((380, y + 35), title, font=FONT_H2, fill=INK)
        draw_wrapped(draw, (380, y + 90), body, fnt=FONT_SMALL, fill=MUTED, width=1160)
        y += 205
    draw.text((420, 935), "Next: larger RWKV/Qwen models, sequential concepts, and pullback paths.", font=FONT_BODY, fill=BLUE)


def closing_slide(image: Image.Image, draw: ImageDraw.ImageDraw, alpha: float) -> None:
    draw_centered(draw, 230, "Takeaway", fnt=FONT_TITLE)
    draw_wrapped(
        draw,
        (410, 360),
        "Linear steering asks: what direction should I push? Manifold steering asks a better question: what path stays inside the model's natural concept geometry?",
        fnt=FONT_H2,
        fill=INK,
        width=1100,
        line_gap=14,
    )
    draw_wrapped(
        draw,
        (470, 620),
        "Our RWKV experiment suggests that this geometry-first view can extend beyond transformers.",
        fnt=FONT_BODY,
        fill=MUTED,
        width=980,
    )
    draw.text((465, 900), "References: goodfire.ai/research/the-world-inside-neural-networks", font=FONT_SMALL, fill=MUTED)
    draw.text((465, 935), "Local report: reports/manifold_report/index.html", font=FONT_SMALL, fill=MUTED)


def write_contact_sheet(video: Path, sheet: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-vf",
            "fps=1/8,scale=420:-1,tile=4x3",
            "-frames:v",
            "1",
            str(sheet),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def build_video(repo: Path, out_dir: Path, *, profile: str = "short") -> dict[str, object]:
    if profile not in {"short", "narrated"}:
        raise ValueError(f"unknown profile: {profile}")
    metrics = json.loads((repo / "reports/manifold_report/metrics.json").read_text(encoding="utf-8"))
    gif_root = repo / "reports/manifold_report/visuals"
    paths = {
        "rwkv_weekday": gif_root / "weekday_rwkv-7_0_1b/steering_movement.gif",
        "qwen_weekday": gif_root / "weekday_qwen3_5_0_8b/steering_movement.gif",
        "rwkv_month": gif_root / "month_rwkv-7_0_1b/steering_movement.gif",
        "qwen_month": gif_root / "month_qwen3_5_0_8b/steering_movement.gif",
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing {name}: {path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    if profile == "narrated":
        out_path = out_dir / "neural_geometry_rwkv_insight_long_silent.mp4"
    else:
        out_path = out_dir / "neural_geometry_rwkv_insight.mp4"
    writer = VideoWriter(out_path)
    segment_records: list[dict[str, object]] = []

    def segment(name: str, duration: float, draw_fn: Callable[[Image.Image, ImageDraw.ImageDraw, float], None]) -> None:
        start = writer.frame_count / FPS
        add_segment(writer, duration, draw_fn)
        segment_records.append({"name": name, "start": start, "end": writer.frame_count / FPS, "duration": duration})

    try:
        if profile == "narrated":
            segment("title", 40.0, title_slide)
            segment("spaces", 40.0, spaces_slide)
            segment("steering_paths", 35.0, steering_paths_slide)
            segment("method", 39.0, method_slide)
            segment("legend", 37.0, legend_slide)
            rwkv_weekday_duration = 36.0
            qwen_weekday_duration = 34.0
            months_duration = 34.0
            metrics_duration = 42.0
            metrics_month_duration = 44.0
            insight_duration = 40.0
            closing_duration = 40.0
        else:
            segment("title", 7.0, title_slide)
            segment("spaces", 9.0, spaces_slide)
            segment("steering_paths", 9.0, steering_paths_slide)
            segment("method", 10.0, method_slide)
            rwkv_weekday_duration = 7.0
            qwen_weekday_duration = 7.0
            months_duration = 8.0
            metrics_duration = 10.0
            metrics_month_duration = 0.0
            insight_duration = 10.0
            closing_duration = 7.0

        segment(
            "rwkv_weekday",
            rwkv_weekday_duration,
            lambda image, draw, alpha: gif_slide(
                image,
                draw,
                alpha,
                headline="Our RWKV weekday run",
                caption="The square follows the fitted activation manifold; the circle is the straight-line baseline. Behavior space is final output probabilities, not hidden layers.",
                gif_paths=[paths["rwkv_weekday"]],
                labels=["RWKV-7 0.1B, Monday to Thursday"],
            ),
        )
        segment(
            "qwen_weekday",
            qwen_weekday_duration,
            lambda image, draw, alpha: gif_slide(
                image,
                draw,
                alpha,
                headline="Transformer comparison: Qwen",
                caption="Same matched-endpoint protocol, but on a small transformer. This lets us compare whether the geometry signal is architecture-specific.",
                gif_paths=[paths["qwen_weekday"]],
                labels=["Qwen3.5 0.8B, Monday to Thursday"],
            ),
        )
        segment(
            "months",
            months_duration,
            lambda image, draw, alpha: gif_slide(
                image,
                draw,
                alpha,
                headline="Months: same idea, harder geometry",
                caption="We also tested January to April. The important detail is that all corrected runs share identical path endpoints.",
                gif_paths=[paths["rwkv_month"], paths["qwen_month"]],
                labels=["RWKV-7 0.1B", "Qwen3.5 0.8B"],
            ),
        )
        segment("metrics", metrics_duration, lambda image, draw, alpha: metrics_slide(image, draw, alpha, metrics))
        if metrics_month_duration > 0:
            segment("metrics_months", metrics_month_duration, lambda image, draw, alpha: metrics_slide(image, draw, alpha, metrics))
        segment("insight", insight_duration, insight_slide)
        segment("closing", closing_duration, closing_slide)
    finally:
        writer.close()

    if profile == "narrated":
        sheet = out_dir / "neural_geometry_rwkv_insight_long_silent_contact_sheet.jpg"
        manifest_path = out_dir / "audience_video_manifest_narrated.json"
    else:
        sheet = out_dir / "neural_geometry_rwkv_insight_contact_sheet.jpg"
        manifest_path = out_dir / "audience_video_manifest.json"
    write_contact_sheet(out_path, sheet)
    manifest = {
        "video": str(out_path),
        "contact_sheet": str(sheet),
        "duration_seconds": writer.frame_count / FPS,
        "fps": FPS,
        "profile": profile,
        "segments": segment_records,
        "source_article": "https://www.goodfire.ai/research/the-world-inside-neural-networks",
        "generated_gifs": {name: str(path) for name, path in paths.items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--out-dir", default="reports/manifold_report/audience_video")
    parser.add_argument("--profile", choices=["short", "narrated"], default="short")
    args = parser.parse_args()
    manifest = build_video(Path(args.repo).resolve(), (Path(args.repo) / args.out_dir).resolve(), profile=args.profile)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
