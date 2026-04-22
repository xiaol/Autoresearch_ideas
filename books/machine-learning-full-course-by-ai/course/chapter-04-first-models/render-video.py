#!/usr/bin/env python3

from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CHAPTER_DIR = Path(__file__).resolve().parent
BOOK_ROOT = CHAPTER_DIR.parents[1]
ASSETS_DIR = CHAPTER_DIR / "assets"
OUTPUT_DIR = CHAPTER_DIR / "video-output"
SLIDES_DIR = OUTPUT_DIR / "slides"
AUDIO_DIR = OUTPUT_DIR / "audio"

DELIVERY_BRIEF = BOOK_ROOT / "examples" / "delivery-time-prediction" / "artifacts" / "brief.md"
DELIVERY_OUTPUT = BOOK_ROOT / "examples" / "delivery-time-prediction" / "artifacts" / "run-output.txt"

WIDTH = 1920
HEIGHT = 1080
FPS = 30
VOICE = "Alex"
RATE = "182"
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD_PATH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


SCENES = [
    {
        "slug": "scene-01-title",
        "type": "svg",
        "source": ASSETS_DIR / "title-card.svg",
        "script": (
            "Chapter four. First models. "
            "Why do first models matter so much in machine learning? "
            "Because if you do not know what a simple model can already do, "
            "then you do not know what your complexity is buying."
        ),
    },
    {
        "slug": "scene-02-baseline-ladder",
        "type": "svg",
        "source": ASSETS_DIR / "process-figure.svg",
        "script": (
            "This lesson is about baseline discipline. "
            "A naive baseline tells us what happens when we barely model anything. "
            "An interpretable baseline tells us what happens when we let the data speak through a simple structure. "
            "A richer model should only come after we understand whether more complexity has earned its place."
        ),
    },
    {
        "slug": "scene-03-case-brief",
        "type": "generated",
        "title": "Delivery-Time Prediction Case",
        "subtitle": "Problem framing before model choice",
        "body": [
            "Decision supported:",
            "Inform the customer of the expected arrival time.",
            "Identify likely late deliveries for intervention.",
            "",
            "Prediction moment:",
            "At dispatch time, after food preparation finishes and the courier has been assigned.",
            "",
            "Constraint:",
            "Prediction should be cheap, fast, and interpretable enough to trust.",
        ],
        "script": (
            "Our case is delivery-time prediction. "
            "We want to estimate total delivery time before the courier reaches the customer. "
            "This prediction supports customer communication and late-delivery intervention. "
            "And it comes with practical constraints. "
            "The prediction should be fast, cheap, and interpretable enough to trust."
        ),
    },
    {
        "slug": "scene-04-demo-command",
        "type": "generated",
        "title": "Runnable Example",
        "subtitle": "One small script, one honest comparison",
        "body": [
            "$ cd /Users/xiaol/x/PaperX/books/machine-learning-full-course-by-ai",
            "$ python3 examples/delivery-time-prediction/scripts/run_baseline.py",
            "",
            "Compare:",
            "Naive mean baseline",
            "Linear regression baseline",
        ],
        "script": (
            "Now we run one tiny local example. "
            "The point is not scale. The point is judgment. "
            "We compare a naive mean baseline against a linear regression baseline "
            "on the same delivery-time problem."
        ),
    },
    {
        "slug": "scene-05-evidence-panel",
        "type": "svg",
        "source": ASSETS_DIR / "evidence-panel.svg",
        "script": (
            "Here is the key result. "
            "The naive mean baseline has a mean absolute error of nine point one two. "
            "The linear regression baseline has a mean absolute error of one point nine four. "
            "That is a large improvement, and it suggests the features contain real usable structure."
        ),
    },
    {
        "slug": "scene-06-weight-reading",
        "type": "generated",
        "title": "What The Weights Suggest",
        "subtitle": "Interpretable structure matters",
        "body": [
            "distance_km: 11.706",
            "prep_minutes: -2.477",
            "courier_load: 1.639",
            "weather_score: 3.495",
            "is_rush_hour: 7.014",
            "",
            "Operational intuition:",
            "Distance, weather, and rush hour all push delivery time upward.",
        ],
        "script": (
            "We can also inspect the learned weights. "
            "Distance is strongly positive. "
            "Weather severity is positive. "
            "Rush hour is positive. "
            "Those directions make operational sense. "
            "That does not prove the model is perfect, but it does show that the baseline is interpretable and useful."
        ),
    },
    {
        "slug": "scene-07-skill-prompt",
        "type": "generated",
        "title": "Reusable Judgment",
        "subtitle": "Apply the ML Baseline Builder skill",
        "body": [
            "Use $ml-baseline-builder to review whether the linear baseline",
            "is meaningfully better than the naive baseline in this delivery-time example.",
            "",
            "Ask for:",
            "metric interpretation",
            "evidence of meaningful gain",
            "what the result still does not prove",
            "whether richer models are justified yet",
        ],
        "script": (
            "This is where the ML Baseline Builder skill becomes useful. "
            "The skill forces us to write down the metric, the evidence, and the next justified escalation. "
            "That turns baseline work into a reusable engineering discipline instead of a one-off opinion."
        ),
    },
    {
        "slug": "scene-08-close",
        "type": "generated",
        "title": "Closing Takeaway",
        "subtitle": "One sentence to keep",
        "body": [
            "Complexity should earn its place.",
            "",
            "Before you reach for a more powerful model,",
            "make sure a simpler one has already taught you something important.",
        ],
        "script": (
            "The point is not to stay simple forever. "
            "The point is to know what complexity is buying. "
            "If you keep one sentence from this lesson, let it be this. "
            "Complexity should earn its place."
        ),
    },
]


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, **kwargs)


def ffprobe_duration(path: Path) -> float:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        capture_output=True,
    )
    return float(result.stdout.strip())


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD_PATH if bold else FONT_PATH, size=size)


def wrap_block(lines: list[str], width: int) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        if not line:
            wrapped.append("")
            continue
        wrapped.extend(textwrap.wrap(line, width=width) or [""])
    return wrapped


def render_generated_slide(scene: dict[str, object], destination: Path) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#0B1324")
    draw = ImageDraw.Draw(image)

    # Background accents
    draw.rectangle((0, 0, WIDTH, HEIGHT), fill="#0B1324")
    draw.ellipse((1200, -120, 1900, 580), fill="#123B69")
    draw.ellipse((1320, 620, 1850, 1150), fill="#0E7C66")
    draw.rectangle((64, 64, WIDTH - 64, HEIGHT - 64), outline="#3A526E", width=3)
    draw.rectangle((88, 88, 102, HEIGHT - 88), fill="#F59E0B")

    title_font = load_font(64, bold=True)
    subtitle_font = load_font(30)
    body_font = load_font(34)

    title = str(scene["title"])
    subtitle = str(scene["subtitle"])
    body_lines = wrap_block([str(line) for line in scene["body"]], width=54)

    draw.text((140, 120), title, fill="#F8FAFC", font=title_font)
    draw.text((140, 208), subtitle, fill="#93C5FD", font=subtitle_font)

    y = 300
    for line in body_lines:
        if line == "":
            y += 26
            continue
        fill = "#E2E8F0"
        if line.endswith(":"):
            fill = "#FBBF24"
        draw.text((160, y), line, fill=fill, font=body_font)
        y += 48

    image.save(destination)


def render_svg_slide(source: Path, destination: Path) -> None:
    run(
        [
            "rsvg-convert",
            "-w",
            str(WIDTH),
            "-h",
            str(HEIGHT),
            str(source),
            "-o",
            str(destination),
        ]
    )


def build_narration(scene: dict[str, object], audio_path: Path) -> float:
    run(
        [
            "say",
            "-v",
            VOICE,
            "-r",
            RATE,
            "-o",
            str(audio_path),
            str(scene["script"]),
        ]
    )
    return ffprobe_duration(audio_path)


def make_segment(scene_image: Path, scene_audio: Path, duration: float, output_path: Path) -> None:
    hold_duration = max(duration + 0.35, 2.5)
    run(
        [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            str(scene_image),
            "-i",
            str(scene_audio),
            "-c:v",
            "libx264",
            "-tune",
            "stillimage",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            f"fps={FPS},format=yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-t",
            f"{hold_duration:.2f}",
            str(output_path),
        ],
        capture_output=True,
    )


def write_concat_manifest(segment_paths: list[Path], manifest_path: Path) -> None:
    lines = [f"file '{segment.as_posix()}'" for segment in segment_paths]
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_thumbnail(scene_path: Path, thumbnail_path: Path) -> None:
    shutil.copyfile(scene_path, thumbnail_path)


def main() -> None:
    ensure_clean_dir(OUTPUT_DIR)
    SLIDES_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    segments: list[Path] = []
    manifest_records: list[dict[str, object]] = []

    for index, scene in enumerate(SCENES, start=1):
        slug = str(scene["slug"])
        slide_path = SLIDES_DIR / f"{index:02d}-{slug}.png"
        audio_path = AUDIO_DIR / f"{index:02d}-{slug}.aiff"
        segment_path = OUTPUT_DIR / f"{index:02d}-{slug}.mp4"

        if scene["type"] == "svg":
            render_svg_slide(Path(scene["source"]), slide_path)
        else:
            render_generated_slide(scene, slide_path)

        duration = build_narration(scene, audio_path)
        make_segment(slide_path, audio_path, duration, segment_path)

        segments.append(segment_path)
        manifest_records.append(
            {
                "index": index,
                "slug": slug,
                "duration_seconds": round(duration, 2),
                "slide": slide_path.name,
                "audio": audio_path.name,
                "segment": segment_path.name,
            }
        )

    concat_manifest = OUTPUT_DIR / "concat.txt"
    write_concat_manifest(segments, concat_manifest)

    final_video = OUTPUT_DIR / "chapter-04-example-video.mp4"
    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_manifest),
            "-c",
            "copy",
            str(final_video),
        ],
        capture_output=True,
    )

    create_thumbnail(SLIDES_DIR / "01-scene-01-title.png", OUTPUT_DIR / "chapter-04-thumbnail.png")

    metadata = {
        "title": "Machine Learning Full Course by AI | Chapter 04 Example Video",
        "voice": VOICE,
        "rate": RATE,
        "fps": FPS,
        "source_brief": str(DELIVERY_BRIEF),
        "source_output": str(DELIVERY_OUTPUT),
        "scenes": manifest_records,
        "final_video": str(final_video),
    }
    (OUTPUT_DIR / "video-manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(final_video)


if __name__ == "__main__":
    main()
