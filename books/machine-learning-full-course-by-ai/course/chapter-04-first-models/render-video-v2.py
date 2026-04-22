#!/usr/bin/env python3

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont


CHAPTER_DIR = Path(__file__).resolve().parent
BOOK_ROOT = CHAPTER_DIR.parents[1]
REPO_ROOT = BOOK_ROOT.parents[1]
ASSETS_DIR = CHAPTER_DIR / "assets"
OUTPUT_DIR = CHAPTER_DIR / "video-output-v2"
SLIDES_DIR = OUTPUT_DIR / "slides"
AUDIO_DIR = OUTPUT_DIR / "audio"

DELIVERY_BRIEF = BOOK_ROOT / "examples" / "delivery-time-prediction" / "artifacts" / "brief.md"
DELIVERY_OUTPUT = BOOK_ROOT / "examples" / "delivery-time-prediction" / "artifacts" / "run-output.txt"

WIDTH = 1920
HEIGHT = 1080
FPS = 30
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD_PATH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

MINIMAX_MODEL = os.getenv("MINIMAX_TTS_MODEL", "speech-2.8-hd")
MINIMAX_VOICE = os.getenv("MINIMAX_TTS_VOICE_ID", "English_AttractiveGirl")
MINIMAX_SPEED = float(os.getenv("MINIMAX_TTS_SPEED", "1.08"))
MINIMAX_VOLUME = float(os.getenv("MINIMAX_TTS_VOLUME", "1.08"))
MINIMAX_PITCH = int(os.getenv("MINIMAX_TTS_PITCH", "1"))
MINIMAX_EMOTION = os.getenv("MINIMAX_TTS_EMOTION", "happy")
MINIMAX_FALLBACK_VOICES = [
    "English_captivating_female1",
    "English_LovelyGirl",
    "English_SereneWoman",
    "English_ConfidentWoman",
    "English_CalmWoman",
]
MINIMAX_FALLBACK_MODELS = ["speech-2.6-hd", "speech-01-turbo"]
SILENCE_BETWEEN_SCENES = 1.1


SCENES = [
    {
        "slug": "scene-01-title",
        "type": "svg",
        "source": ASSETS_DIR / "title-card.svg",
        "subtitle": "First models are not warm-up exercises. They are instruments for judgment.",
        "script": (
            "Chapter four. First models. "
            "In this lesson, we will treat the first model as a professional instrument, not a warm-up exercise. "
            "A beginner often asks, what model should I use? "
            "A professional asks a sharper question: what have I learned from the simplest model that could possibly work? "
            "That first model gives us a reference point. "
            "Without it, we cannot honestly say what complexity is buying."
        ),
    },
    {
        "slug": "scene-02-hook",
        "type": "generated",
        "title": "The Professional Question",
        "subtitle": "What did complexity buy?",
        "caption": "A first model gives the team a measurement baseline.",
        "body": [
            "Weak question:",
            "Can we train a powerful model?",
            "",
            "Professional question:",
            "What decision improves because this model exists?",
            "",
            "First-model mindset:",
            "Start with evidence, not model prestige.",
        ],
        "script": (
            "Here is the professional question. "
            "Not, can we train a powerful model. "
            "The question is, what decision improves because this model exists? "
            "A first model gives the team a measurement baseline. "
            "It also gives the team a shared language. "
            "Product can ask whether the prediction changes the customer experience. "
            "Operations can ask whether the prediction changes intervention timing. "
            "Engineering can ask whether the improvement is worth the serving cost. "
            "That is how model choice moves from taste into evidence."
        ),
    },
    {
        "slug": "scene-03-baseline-ladder",
        "type": "svg",
        "source": ASSETS_DIR / "process-figure.svg",
        "subtitle": "Naive baseline, interpretable baseline, then richer model. Each rung earns the next.",
        "script": (
            "This lesson is baseline discipline. "
            "A naive baseline shows what happens when we barely model anything. "
            "An interpretable baseline lets the data speak through a simple structure. "
            "A richer model comes later, only after the simple model reveals what it can and cannot explain. "
            "This ladder is important because each step earns the next. "
            "If a linear model already solves the business problem, a neural network may add cost without adding value. "
            "If the simple model fails in clear ways, those failures tell us what to try next."
        ),
    },
    {
        "slug": "scene-04-case-brief",
        "type": "generated",
        "title": "Delivery-Time Prediction Case",
        "subtitle": "Problem framing before model choice",
        "caption": "The model supports ETA communication and late-delivery intervention.",
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
            "At dispatch time, we estimate when the order will arrive. "
            "That supports customer communication and late-delivery intervention. "
            "This is important because the model is not just predicting a number. "
            "It is shaping expectations, staffing decisions, and escalation workflows. "
            "If the estimate is too optimistic, customers become frustrated. "
            "If it is too pessimistic, the business may over-correct and waste operational attention. "
            "So even in a small example, the prediction sits inside a real decision loop."
        ),
    },
    {
        "slug": "scene-05-prediction-moment",
        "type": "generated",
        "title": "Prediction Moment",
        "subtitle": "The exact moment defines the legal information",
        "caption": "If information is not available at dispatch time, the model cannot use it.",
        "body": [
            "Allowed at dispatch:",
            "distance_km",
            "prep_minutes",
            "courier_load",
            "weather_score",
            "is_rush_hour",
            "",
            "Not allowed:",
            "anything learned after the delivery finishes",
        ],
        "script": (
            "The prediction moment matters. "
            "At dispatch time, the model may know distance, preparation time, courier load, weather, and rush hour. "
            "It may not know anything that happens after delivery. "
            "This single boundary prevents many accidental leakage problems. "
            "For example, if we accidentally include the actual arrival delay, or a support ticket created after the delivery, the model may look amazing offline and fail in production. "
            "A first model should make this boundary explicit before training starts."
        ),
    },
    {
        "slug": "scene-06-dataset-snapshot",
        "type": "generated",
        "title": "Feature Snapshot",
        "subtitle": "Small features, strong signal",
        "caption": "Begin with features a human operator can inspect.",
        "body": [
            "Rows:",
            "past deliveries",
            "",
            "Target:",
            "total delivery minutes",
            "",
            "Features:",
            "distance, prep time, courier load, weather, rush hour",
        ],
        "script": (
            "The dataset is intentionally small and inspectable. "
            "Each row is a past delivery. "
            "The target is total delivery minutes. "
            "The features are simple operational signals. "
            "This is a good first-model setup because a learner can reason about every input. "
            "Distance should usually increase time. "
            "Bad weather should usually increase time. "
            "Rush hour should usually increase time. "
            "When a feature has an unexpected direction, that does not automatically mean the model is wrong, but it gives us a concrete question to investigate."
        ),
    },
    {
        "slug": "scene-07-leakage-guardrail",
        "type": "generated",
        "title": "Leakage Guardrail",
        "subtitle": "A good score can still be fake",
        "caption": "The first model should make leakage easier to detect.",
        "body": [
            "Ask before training:",
            "Would this feature be known at prediction time?",
            "Was the label created after the outcome?",
            "Did train and test share future information?",
            "",
            "Suspicious result:",
            "too good, too early, too hard to explain",
        ],
        "script": (
            "Before training, we ask three leakage questions. "
            "Would this feature be known at prediction time? "
            "Was the label created after the outcome? "
            "Did train and test share future information? "
            "A simple first model makes suspicious results easier to notice. "
            "If the first model is unbelievably accurate, we should not celebrate immediately. "
            "We should ask whether the data split, timestamp logic, or feature pipeline accidentally gave the model information from the future."
        ),
    },
    {
        "slug": "scene-08-demo-command",
        "type": "generated",
        "title": "Runnable Example",
        "subtitle": "One small script, one honest comparison",
        "caption": "Run the example first. Interpret the result second.",
        "body": [
            "$ cd /Users/xiaol/x/PaperX/books/machine-learning-full-course-by-ai",
            "$ python3 examples/delivery-time-prediction/scripts/run_baseline.py",
            "",
            "Compare:",
            "Naive mean baseline",
            "Linear regression baseline",
        ],
        "script": (
            "Now run the local example. "
            "The goal is not scale; it is judgment. "
            "We compare a naive mean baseline with linear regression on the same delivery-time task. "
            "Both models see the same split and the same metric, so the comparison is fair. "
            "This is the habit to copy in real work: keep the data, metric, and evaluation window fixed while the model changes."
        ),
    },
    {
        "slug": "scene-09-naive-baseline",
        "type": "generated",
        "title": "Naive Baseline",
        "subtitle": "The model that barely models",
        "caption": "Naive baselines are not embarrassing. They are measurement tools.",
        "body": [
            "Naive rule:",
            "predict the average delivery time",
            "",
            "What it answers:",
            "How much error exists before learning structure?",
            "",
            "Why it matters:",
            "Every richer model must beat this cleanly.",
        ],
        "script": (
            "The naive baseline predicts the average delivery time. "
            "It ignores distance, weather, courier load, and rush hour. "
            "That sounds weak, but it answers an essential question. "
            "How much error exists before the model learns any structure? "
            "This number protects us from self-deception. "
            "If a fancy model barely beats the average, then the model may be impressive technically but weak for this decision."
        ),
    },
    {
        "slug": "scene-10-linear-baseline",
        "type": "generated",
        "title": "Interpretable Baseline",
        "subtitle": "A simple model with readable structure",
        "caption": "Linear regression is useful because both score and explanation are inspectable.",
        "body": [
            "Model:",
            "linear regression",
            "",
            "Why here:",
            "fast training",
            "cheap serving",
            "readable coefficients",
            "",
            "Question:",
            "Does simple structure explain the target?",
        ],
        "script": (
            "Next we train a linear regression baseline. "
            "It is still simple, but now the model can use structure. "
            "It can learn that distance, weather, and rush hour change expected time. "
            "The value is that performance and explanation remain inspectable. "
            "Training is fast. "
            "Serving is cheap. "
            "The coefficients can be read by a human. "
            "For many production problems, that combination is not a compromise; it is a serious engineering advantage."
        ),
    },
    {
        "slug": "scene-11-evidence-panel",
        "type": "svg",
        "source": ASSETS_DIR / "evidence-panel.svg",
        "subtitle": "MAE falls from 9.12 to 1.94. The features contain useful structure.",
        "script": (
            "Here is the key result. "
            "The naive mean baseline has mean absolute error nine point one two. "
            "Linear regression drops it to one point nine four. "
            "That is not a small cosmetic improvement. "
            "It is evidence that the simple operational features contain usable structure. "
            "Mean absolute error is easy to explain: on average, how many minutes are we wrong? "
            "So the improvement is not just a leaderboard number. "
            "It translates into a much tighter estimate for the people using the system."
        ),
    },
    {
        "slug": "scene-12-weight-reading",
        "type": "generated",
        "title": "What The Weights Suggest",
        "subtitle": "Interpretable structure matters",
        "caption": "Performance plus interpretability is a strong first-model signal.",
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
            "We can inspect the learned weights. "
            "Distance, weather severity, and rush hour all push delivery time upward. "
            "That direction makes operational sense. "
            "The negative preparation-time weight deserves review, because it may reflect process timing or dataset quirks. "
            "Interpretability helps us find those questions early. "
            "This is why simple models are not only for beginners. "
            "A professional engineer uses them to find bugs, assumptions, and business questions before the system becomes harder to understand."
        ),
    },
    {
        "slug": "scene-13-skill-prompt",
        "type": "generated",
        "title": "Reusable Judgment",
        "subtitle": "Apply the ML Baseline Builder skill",
        "caption": "The skill turns one result into a repeatable review habit.",
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
            "Now use the ML Baseline Builder skill. "
            "It forces us to name the metric, the evidence, and the next justified escalation. "
            "Instead of saying, the score looks good, we ask what the score proves, what it does not prove, and whether a richer model is justified. "
            "For this case, the skill should point out that the linear baseline is meaningfully better than naive, but it has not yet proven robustness across restaurants, weather conditions, or courier-load extremes."
        ),
    },
    {
        "slug": "scene-14-decision-gate",
        "type": "generated",
        "title": "Decision Gate",
        "subtitle": "What should happen after the first model?",
        "caption": "A first model should change the next experiment, not end the discussion.",
        "body": [
            "Do next:",
            "check error slices",
            "inspect outliers",
            "test calibration",
            "review leakage again",
            "",
            "Only then:",
            "try richer trees or ensembles",
        ],
        "script": (
            "After the first model, do not jump straight to a bigger model. "
            "Check error slices. "
            "Inspect outliers. "
            "Test whether predictions are calibrated enough for decisions. "
            "Review leakage again. "
            "Only then should we try richer trees or ensembles. "
            "A decision gate is not a delay tactic. "
            "It is how we avoid spending a week on complexity when the next useful action is a one-hour data review."
        ),
    },
    {
        "slug": "scene-15-professional-mistakes",
        "type": "generated",
        "title": "Professional Mistakes",
        "subtitle": "What beginners and teams often skip",
        "caption": "The danger is not simplicity. The danger is unmeasured complexity.",
        "body": [
            "Common mistakes:",
            "skip the naive baseline",
            "change metric between models",
            "celebrate aggregate score only",
            "ignore slice failures",
            "explain results after choosing the model",
        ],
        "script": (
            "Here are common mistakes. "
            "Skipping the naive baseline. "
            "Changing the metric between models. "
            "Celebrating only the aggregate score. "
            "Ignoring slice failures. "
            "Explaining results after the model has already been chosen. "
            "First-model discipline protects us from all five. "
            "The danger is not simplicity. "
            "The danger is unmeasured complexity, where the team cannot tell whether a larger model improved the decision or merely made the pipeline harder to reason about."
        ),
    },
    {
        "slug": "scene-16-practice",
        "type": "generated",
        "title": "Practice Task",
        "subtitle": "Turn the lesson into a repeatable habit",
        "caption": "Use the same baseline ladder on your own dataset.",
        "body": [
            "Your turn:",
            "pick one prediction moment",
            "write one naive baseline",
            "train one interpretable baseline",
            "compare one metric",
            "review one failure slice",
            "",
            "Then ask:",
            "what did complexity buy?",
        ],
        "script": (
            "Your practice task is simple. "
            "Choose one prediction moment. "
            "Write one naive baseline. "
            "Train one interpretable baseline. "
            "Compare one metric. "
            "Review one failure slice. "
            "Then ask the professional question: what did complexity buy? "
            "If you can answer that clearly, you are not just running machine learning code. "
            "You are building engineering judgment that transfers to fraud detection, recommendation, search, forecasting, and almost every applied machine learning system."
        ),
    },
    {
        "slug": "scene-17-close",
        "type": "generated",
        "title": "Closing Takeaway",
        "subtitle": "One sentence to keep",
        "caption": "A simpler model should teach you something before a bigger model enters the room.",
        "body": [
            "Complexity should earn its place.",
            "",
            "Before you reach for a more powerful model,",
            "make sure a simpler one has already taught you something important.",
        ],
        "script": (
            "The goal is not to stay simple forever. "
            "The goal is to know what complexity buys. "
            "A first model should teach you something before a larger model enters the room. "
            "Keep one sentence from this chapter: complexity should earn its place. "
            "In the next lessons, that sentence will guide how we evaluate models, diagnose failures, and decide when a more powerful method is truly worth it."
        ),
    },
]


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, **kwargs)


def load_env_files() -> None:
    for env_path in (REPO_ROOT / ".env", BOOK_ROOT / ".env", CHAPTER_DIR / ".env"):
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def group_id_from_jwt(token: str) -> str:
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload.encode("utf-8"))
        data = json.loads(decoded)
        return str(data.get("GroupID") or data.get("GroupId") or data.get("group_id") or "")
    except Exception:
        return ""


def minimax_urls(api_key: str) -> list[str]:
    group_id = os.getenv("MINIMAX_GROUP_ID") or group_id_from_jwt(api_key)
    configured_base = os.getenv("MINIMAX_BASE_URL")
    base_urls = [
        configured_base,
        "https://api.minimax.io/v1/t2a_v2",
        "https://api.minimaxi.chat/v1/t2a_v2",
        "https://api.minimax.chat/v1/t2a_v2",
    ]
    urls: list[str] = []
    for base_url in base_urls:
        if not base_url or base_url in urls:
            continue
        if group_id and "GroupId=" not in base_url:
            separator = "&" if "?" in base_url else "?"
            urls.append(f"{base_url}{separator}GroupId={group_id}")
        urls.append(base_url)
    return list(dict.fromkeys(urls))


def minimax_model_candidates() -> list[str]:
    candidates = [MINIMAX_MODEL, *MINIMAX_FALLBACK_MODELS]
    return list(dict.fromkeys(candidates))


def minimax_voice_candidates() -> list[str]:
    candidates = [MINIMAX_VOICE, *MINIMAX_FALLBACK_VOICES]
    return list(dict.fromkeys(candidates))


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

    draw.rectangle((0, 0, WIDTH, HEIGHT), fill="#0B1324")
    draw.ellipse((1180, -140, 1920, 620), fill="#123B69")
    draw.ellipse((1320, 620, 1850, 1150), fill="#0E7C66")
    draw.rectangle((64, 64, WIDTH - 64, HEIGHT - 64), outline="#3A526E", width=3)
    draw.rectangle((88, 88, 102, HEIGHT - 88), fill="#F59E0B")

    title_font = load_font(64, bold=True)
    subtitle_font = load_font(30)
    body_font = load_font(34)
    caption_font = load_font(28)

    title = str(scene["title"])
    subtitle = str(scene["subtitle"])
    caption = str(scene.get("caption", ""))
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

    if caption:
        draw.rounded_rectangle((140, 928, WIDTH - 140, 1014), radius=24, fill="#020617", outline="#334155", width=2)
        draw.text((176, 956), caption, fill="#F8FAFC", font=caption_font)

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


def generate_minimax_audio(text: str, audio_path: Path) -> tuple[bool, str, str]:
    load_env_files()
    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        print("MiniMax API key missing. Set MINIMAX_API_KEY in /Users/xiaol/x/PaperX/.env.")
        return False, "", ""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error = "No MiniMax endpoint was attempted."

    for voice_id in minimax_voice_candidates():
        for model in minimax_model_candidates():
            payload = {
                "model": model,
                "text": text,
                "stream": False,
                "language_boost": "English",
                "output_format": "hex",
                "voice_setting": {
                    "voice_id": voice_id,
                    "speed": MINIMAX_SPEED,
                    "vol": MINIMAX_VOLUME,
                    "pitch": MINIMAX_PITCH,
                    "emotion": MINIMAX_EMOTION,
                },
                "audio_setting": {
                    "sample_rate": 32000,
                    "bitrate": 128000,
                    "format": "mp3",
                    "channel": 1,
                },
            }

            for url in minimax_urls(api_key):
                safe_url = url.split("?", 1)[0]
                try:
                    response = requests.post(url, headers=headers, json=payload, timeout=120)
                except requests.RequestException as exc:
                    last_error = f"{safe_url} request failed: {exc}"
                    print(f"MiniMax TTS request failed for {safe_url}: {exc}")
                    continue

                if response.status_code != 200:
                    last_error = f"{safe_url} HTTP {response.status_code}: {response.text[:400]}"
                    print(f"MiniMax TTS failed for {safe_url}: HTTP {response.status_code}: {response.text[:300]}")
                    continue

                try:
                    data = response.json()
                    audio_hex = data["data"]["audio"]
                    audio_path.write_bytes(binascii.unhexlify(audio_hex))
                except Exception as exc:
                    last_error = f"{safe_url} returned unusable audio: {exc}"
                    print(f"MiniMax TTS response did not contain usable audio from {safe_url}: {exc}")
                    print(response.text[:300])
                    continue

                print(f"MiniMax TTS succeeded with voice={voice_id}, model={model}, endpoint={safe_url}")
                return True, voice_id, model

    print(f"MiniMax TTS exhausted all voice/model/endpoint candidates. Last error: {last_error}")
    return False, "", ""


def generate_say_audio(text: str, audio_path: Path) -> bool:
    run(
        [
            "say",
            "-v",
            "Samantha",
            "-r",
            "178",
            "-o",
            str(audio_path),
            text,
        ]
    )
    return True


def build_narration(scene: dict[str, object], audio_path: Path) -> tuple[float, str]:
    script = str(scene["script"])
    minimax_path = audio_path.with_suffix(".mp3")
    ok, voice_id, model = generate_minimax_audio(script, minimax_path)
    if ok:
        scene["resolved_voice_id"] = voice_id
        scene["resolved_tts_model"] = model
        return ffprobe_duration(minimax_path), "minimax"

    fallback_path = audio_path.with_suffix(".aiff")
    generate_say_audio(script, fallback_path)
    scene["resolved_voice_id"] = "Samantha"
    scene["resolved_tts_model"] = "macos-say"
    return ffprobe_duration(fallback_path), "say-fallback"


def audio_path_for_scene(base_path: Path) -> Path:
    mp3_path = base_path.with_suffix(".mp3")
    if mp3_path.exists():
        return mp3_path
    return base_path.with_suffix(".aiff")


def make_segment(scene_image: Path, scene_audio: Path, duration: float, output_path: Path) -> None:
    hold_duration = max(duration + SILENCE_BETWEEN_SCENES, 3.0)
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


def srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def subtitle_text(scene: dict[str, object]) -> str:
    return re.sub(r"\s+", " ", str(scene["script"])).strip()


def split_subtitle_chunks(text: str) -> list[str]:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > 96:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks or [text]


def write_subtitles(scene_records: list[dict[str, object]], subtitle_path: Path) -> None:
    cursor = 0.0
    blocks: list[str] = []
    subtitle_index = 1
    for record in scene_records:
        duration = float(record["segment_duration_seconds"])
        chunks = split_subtitle_chunks(str(record["subtitle"]))
        per_chunk = duration / len(chunks)
        for chunk_index, chunk in enumerate(chunks):
            start = cursor + (chunk_index * per_chunk)
            end = cursor + ((chunk_index + 1) * per_chunk)
            text = "\n".join(textwrap.wrap(chunk, width=58))
            blocks.append(f"{subtitle_index}\n{srt_time(start)} --> {srt_time(end)}\n{text}\n")
            subtitle_index += 1
        cursor = end
    subtitle_path.write_text("\n".join(blocks), encoding="utf-8")


def burn_subtitles(input_video: Path, subtitles: Path, output_video: Path) -> None:
    escaped = str(subtitles).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    style = "FontName=Arial,FontSize=14,PrimaryColour=&H00FFFFFF,OutlineColour=&HAA000000,BorderStyle=3,Outline=1,Shadow=0,MarginV=52"
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-vf",
            f"subtitles='{escaped}':force_style='{style}'",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-c:a",
            "copy",
            str(output_video),
        ],
        capture_output=True,
    )


def mix_background(input_video: Path, output_video: Path, duration: float) -> None:
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=196:sample_rate=44100:duration={duration:.2f}",
            "-filter_complex",
            "[1:a]volume=0.018,afade=t=in:st=0:d=2,afade=t=out:st="
            f"{max(duration - 3, 0):.2f}:d=3[bed];[0:a][bed]amix=inputs=2:duration=first:normalize=0[a]",
            "-map",
            "0:v",
            "-map",
            "[a]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_video),
        ],
        capture_output=True,
    )


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
        audio_base_path = AUDIO_DIR / f"{index:02d}-{slug}"
        segment_path = OUTPUT_DIR / f"{index:02d}-{slug}.mp4"

        if scene["type"] == "svg":
            render_svg_slide(Path(scene["source"]), slide_path)
        else:
            render_generated_slide(scene, slide_path)

        narration_duration, provider = build_narration(scene, audio_base_path)
        scene_audio = audio_path_for_scene(audio_base_path)
        make_segment(slide_path, scene_audio, narration_duration, segment_path)
        segment_duration = ffprobe_duration(segment_path)

        segments.append(segment_path)
        manifest_records.append(
            {
                "index": index,
                "slug": slug,
                "tts_provider": provider,
                "voice_id": scene.get("resolved_voice_id", MINIMAX_VOICE if provider == "minimax" else "Samantha"),
                "tts_model": scene.get("resolved_tts_model", MINIMAX_MODEL if provider == "minimax" else "macos-say"),
                "narration_duration_seconds": round(narration_duration, 2),
                "segment_duration_seconds": round(segment_duration, 2),
                "subtitle": subtitle_text(scene),
                "slide": slide_path.name,
                "audio": scene_audio.relative_to(OUTPUT_DIR).as_posix(),
                "segment": segment_path.name,
            }
        )

    concat_manifest = OUTPUT_DIR / "concat.txt"
    write_concat_manifest(segments, concat_manifest)

    raw_video = OUTPUT_DIR / "chapter-04-v2-raw.mp4"
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
            str(raw_video),
        ],
        capture_output=True,
    )

    subtitles_path = OUTPUT_DIR / "chapter-04-v2-subtitles.srt"
    write_subtitles(manifest_records, subtitles_path)

    subtitled_video = OUTPUT_DIR / "chapter-04-v2-subtitled.mp4"
    burn_subtitles(raw_video, subtitles_path, subtitled_video)

    final_video = OUTPUT_DIR / "chapter-04-v2-minimax-female-subtitled.mp4"
    final_duration = ffprobe_duration(subtitled_video)
    mix_background(subtitled_video, final_video, final_duration)

    create_thumbnail(SLIDES_DIR / "01-scene-01-title.png", OUTPUT_DIR / "chapter-04-v2-thumbnail.png")

    metadata = {
        "title": "Machine Learning Full Course by AI | Chapter 04 Refined Example",
        "tts_provider_requested": "minimax",
        "minimax_voice_id": MINIMAX_VOICE,
        "minimax_model": MINIMAX_MODEL,
        "minimax_speed": MINIMAX_SPEED,
        "minimax_emotion": MINIMAX_EMOTION,
        "fps": FPS,
        "resolution": f"{WIDTH}x{HEIGHT}",
        "source_brief": str(DELIVERY_BRIEF),
        "source_output": str(DELIVERY_OUTPUT),
        "subtitles": str(subtitles_path),
        "background_bed": "ffmpeg generated subtle sine bed at low volume",
        "scenes": manifest_records,
        "final_video": str(final_video),
    }
    (OUTPUT_DIR / "video-manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(final_video)


if __name__ == "__main__":
    main()
