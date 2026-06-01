#!/usr/bin/env python3
"""Generate MiniMax narration and mux it onto the long audience explainer."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = Path(os.environ.get("PAPERX_ENV", "~/.codex/env/PaperX.env")).expanduser()
VIDEO_DIR = ROOT / "reports" / "manifold_report" / "audience_video"
NARRATION = VIDEO_DIR / "narration_minimax.md"
SOURCE_VIDEO = VIDEO_DIR / "neural_geometry_rwkv_insight_long_silent.mp4"
FINAL_VIDEO = VIDEO_DIR / "neural_geometry_rwkv_insight_minimax.mp4"
FINAL_SRT = VIDEO_DIR / "neural_geometry_rwkv_insight_minimax.srt"
FINAL_WAV = VIDEO_DIR / "minimax_audio" / "neural_geometry_rwkv_insight_minimax.wav"
FINAL_MP3 = VIDEO_DIR / "minimax_audio" / "neural_geometry_rwkv_insight_minimax.mp3"
REPORT = VIDEO_DIR / "render_report_minimax.json"
CONTACT_SHEET = VIDEO_DIR / "neural_geometry_rwkv_insight_minimax_contact_sheet.jpg"
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE", "ffprobe")
MODEL = "speech-2.8-hd"
VOICE_A = os.environ.get("MANIFOLD_MINIMAX_VOICE_A", "English_captivating_female1")
VOICE_B = os.environ.get("MANIFOLD_MINIMAX_VOICE_B", "English_CaptivatingStoryteller")
VOICE_SPEED = float(os.environ.get("MANIFOLD_MINIMAX_SPEED", "1.0"))
TAG_RE = re.compile(r"\((?:breath|chuckle|sighs|laughs)\)|<#[0-9.]+#>")


@dataclass(frozen=True)
class Segment:
    index: int
    start: float
    end: float
    text: str
    speaker_id: str = "A"

    @property
    def duration(self) -> float:
        return max(0.1, self.end - self.start)

    @property
    def speaker(self) -> str:
        return self.speaker_id

    @property
    def voice(self) -> str:
        return VOICE_B if self.speaker == "B" else VOICE_A

    @property
    def subtitle_text(self) -> str:
        text = TAG_RE.sub("", self.text)
        text = re.sub(r"\b[AB]:\s*", "", text)
        return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class Turn:
    speaker: str
    text: str

    @property
    def voice(self) -> str:
        return VOICE_B if self.speaker == "B" else VOICE_A

    @property
    def subtitle_text(self) -> str:
        text = TAG_RE.sub("", self.text)
        text = re.sub(r"\b[AB]:\s*", "", text)
        return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class SceneBlock:
    index: int
    start: float
    end: float
    turns: tuple[Turn, ...]


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def load_env() -> None:
    if not ENV_PATH.exists():
        return
    for raw in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_time(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"Bad timestamp: {value}")


def format_srt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    whole = int(seconds)
    ms = int(round((seconds - whole) * 1000))
    if ms == 1000:
        whole += 1
        ms = 0
    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def audio_duration(path: Path) -> float:
    result = run(
        [
            FFPROBE,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    return float(result.stdout.strip())


def video_duration(path: Path) -> float:
    return audio_duration(path)


def parse_turns(text: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"(?m)(^|\n)([AB]):\s*", text))
    if not matches:
        return [("A", re.sub(r"\s+", " ", text).strip())]
    turns: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        speaker = match.group(2)
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = re.sub(r"\s+", " ", text[body_start:body_end].strip())
        if body:
            turns.append((speaker, body))
    return turns


def parse_scene_blocks() -> list[SceneBlock]:
    source = NARRATION.read_text(encoding="utf-8")
    pattern = re.compile(r"^##\s+([0-9:.]+)-([0-9:.]+)\s*$", re.M)
    matches = list(pattern.finditer(source))
    if not matches:
        raise RuntimeError(f"No timestamp headings found in {NARRATION}")
    blocks: list[SceneBlock] = []
    total_duration = video_duration(SOURCE_VIDEO)
    for i, match in enumerate(matches, start=1):
        scene_start = parse_time(match.group(1))
        if scene_start >= total_duration:
            continue
        scene_end = min(parse_time(match.group(2)), total_duration)
        body_start = match.end()
        body_end = matches[i].start() if i < len(matches) else len(source)
        turns = tuple(Turn(speaker=speaker, text=text) for speaker, text in parse_turns(source[body_start:body_end].strip()))
        blocks.append(SceneBlock(index=i, start=scene_start, end=scene_end, turns=turns))
    return blocks


def minimax_url() -> str:
    base = os.environ.get("MINIMAX_TTS_BASE_URL") or os.environ.get("MINIMAX_BASE_URL") or "https://api.minimax.io/v1/t2a_v2"
    group_id = os.environ.get("MINIMAX_GROUP_ID")
    if group_id and "GroupId=" not in base:
        sep = "&" if "?" in base else "?"
        return f"{base}{sep}GroupId={group_id}"
    return base


def cache_key(segment: Segment | Turn) -> str:
    payload = {
        "text": segment.text,
        "voice": segment.voice,
        "speed": VOICE_SPEED,
        "model": MODEL,
        "version": 2,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def cached(path: Path, key: str) -> bool:
    meta = path.with_suffix(path.suffix + ".json")
    if not path.exists() or path.stat().st_size < 1000 or not meta.exists():
        return False
    try:
        return json.loads(meta.read_text(encoding="utf-8")).get("cache_key") == key
    except json.JSONDecodeError:
        return False


def synth_segment(segment: Segment) -> Path:
    out_dir = FINAL_WAV.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"segment_{segment.index:02d}_{segment.speaker}_raw.mp3"
    key = cache_key(segment)
    if cached(raw_path, key):
        return raw_path
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise RuntimeError("MINIMAX_API_KEY is not set")
    if re.match(r"\s*[AB]:", segment.text):
        raise RuntimeError(f"speaker label leaked into TTS text for segment {segment.index}: {segment.text[:40]!r}")
    payload = {
        "model": MODEL,
        "text": segment.text,
        "stream": False,
        "language_boost": "English",
        "output_format": "hex",
        "voice_setting": {"voice_id": segment.voice, "speed": VOICE_SPEED, "vol": 1.0, "pitch": 0},
        "audio_setting": {"sample_rate": 32000, "bitrate": 128000, "format": "mp3", "channel": 1},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data: dict | None = None
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = requests.post(minimax_url(), headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(4 * attempt)
    if data is None:
        raise RuntimeError(f"MiniMax TTS failed for segment {segment.index}: {last_error!r}")
    status = (data.get("base_resp") or {}).get("status_code", 0)
    if status not in (0, "0", None):
        raise RuntimeError(f"MiniMax TTS error for segment {segment.index}: {data.get('base_resp')}")
    audio_hex = (data.get("data") or {}).get("audio")
    audio_url = (data.get("data") or {}).get("audio_url")
    if audio_hex:
        raw_path.write_bytes(bytes.fromhex(audio_hex))
    elif audio_url:
        audio = requests.get(audio_url, timeout=180)
        audio.raise_for_status()
        raw_path.write_bytes(audio.content)
    else:
        raise RuntimeError(f"No audio in MiniMax response for segment {segment.index}: {json.dumps(data)[:500]}")
    raw_path.with_suffix(raw_path.suffix + ".json").write_text(
        json.dumps(
            {
                "cache_key": key,
                "provider": "MiniMax",
                "model": MODEL,
                "voice": segment.voice,
                "voice_speed": VOICE_SPEED,
                "speaker": segment.speaker,
                "segment": segment.index,
                "start": segment.start,
                "end": segment.end,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return raw_path


def synth_turn(turn: Turn, index: int) -> Path:
    synthetic_segment = Segment(index=index, start=0.0, end=1.0, text=turn.text, speaker_id=turn.speaker)
    return synth_segment(synthetic_segment)


def fit_segment_audio(segment: Segment, raw_path: Path) -> tuple[Path, dict[str, float]]:
    fitted = FINAL_WAV.parent / f"segment_{segment.index:02d}_{segment.speaker}_fitted.wav"
    raw_duration = audio_duration(raw_path)
    target = max(0.1, segment.duration - 0.25)
    speed = 1.0
    filters: list[str] = []
    if raw_duration > target:
        speed = min(1.18, raw_duration / target)
        filters.append(f"atempo={speed:.5f}")
    filters.extend(["loudnorm=I=-20:TP=-1.5:LRA=11", "apad"])
    run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(raw_path),
            "-af",
            ",".join(filters),
            "-t",
            f"{segment.duration:.3f}",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(fitted),
        ]
    )
    return fitted, {"raw_duration": raw_duration, "target_duration": segment.duration, "speed_factor": speed}


def fit_turn_audio(segment: Segment, raw_path: Path, *, raw_duration: float) -> tuple[Path, dict[str, float]]:
    fitted = FINAL_WAV.parent / f"segment_{segment.index:02d}_{segment.speaker}_fitted.wav"
    filters = ["loudnorm=I=-20:TP=-1.5:LRA=11", "apad"]
    run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(raw_path),
            "-af",
            ",".join(filters),
            "-t",
            f"{segment.duration:.3f}",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(fitted),
        ]
    )
    return fitted, {"raw_duration": raw_duration, "target_duration": segment.duration, "speed_factor": 1.0}


def silence(path: Path, duration: float) -> None:
    run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-t",
            f"{duration:.3f}",
            str(path),
        ]
    )


def build_timed_segments(blocks: list[SceneBlock]) -> tuple[list[Segment], list[Path], list[dict[str, object]]]:
    timed_segments: list[Segment] = []
    fitted_paths: list[Path] = []
    reports: list[dict[str, object]] = []
    segment_index = 1
    timeline_cursor = 0.0
    for block in blocks:
        if block.start > timeline_cursor + 0.01:
            pre = FINAL_WAV.parent / f"before_scene_{block.index:02d}_silence.wav"
            silence(pre, block.start - timeline_cursor)
            fitted_paths.append(pre)
            timeline_cursor = block.start
        turn_infos: list[tuple[Turn, Path, float]] = []
        for turn in block.turns:
            raw = synth_turn(turn, segment_index)
            turn_infos.append((turn, raw, audio_duration(raw)))
            segment_index += 1
        raw_total = sum(raw_duration for _turn, _raw, raw_duration in turn_infos)
        block_duration = block.end - block.start
        gap = max(0.15, min(1.0, (block_duration - raw_total) / (len(turn_infos) + 1))) if turn_infos else block_duration
        cursor = block.start + gap
        for turn, raw, raw_duration in turn_infos:
            turns_already_in_block = len([s for s in timed_segments if block.start <= s.start < block.end])
            remaining_turns = len(turn_infos) - turns_already_in_block
            end_limit = block.end - gap * max(0, remaining_turns - 1)
            duration = min(raw_duration + 0.25, max(0.3, end_limit - cursor))
            if cursor + duration > block.end:
                duration = max(0.3, block.end - cursor)
            segment = Segment(index=len(timed_segments) + 1, start=cursor, end=cursor + duration, text=turn.text, speaker_id=turn.speaker)
            fitted, metrics = fit_turn_audio(segment, raw, raw_duration=raw_duration)
            if segment.start > timeline_cursor + 0.01:
                between = FINAL_WAV.parent / f"before_segment_{segment.index:02d}_silence.wav"
                silence(between, segment.start - timeline_cursor)
                fitted_paths.append(between)
            timed_segments.append(segment)
            fitted_paths.append(fitted)
            reports.append({"segment": segment.index, "speaker": segment.speaker, "voice": segment.voice, **metrics})
            cursor = segment.end + gap
            timeline_cursor = segment.end
        trailing = max(0.0, block.end - timeline_cursor)
        if trailing > 0.05:
            tail = FINAL_WAV.parent / f"scene_{block.index:02d}_tail_silence.wav"
            silence(tail, trailing)
            fitted_paths.append(tail)
            timeline_cursor = block.end
    return timed_segments, fitted_paths, reports


def clean_previous_timeline_audio() -> None:
    out_dir = FINAL_WAV.parent
    if not out_dir.exists():
        return
    for pattern in ("*_fitted.wav", "*_silence.wav", "scene_*_tail_silence.wav", "before_scene_*_silence.wav", "before_segment_*_silence.wav"):
        for path in out_dir.glob(pattern):
            path.unlink(missing_ok=True)


def write_concat_audio(paths: list[Path], duration: float) -> None:
    concat = FINAL_WAV.parent / "concat.txt"
    concat.write_text("".join(f"file '{path.resolve()}'\n" for path in paths), encoding="utf-8")
    run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
            "-af",
            "loudnorm=I=-20:TP=-1.5:LRA=11,apad",
            "-t",
            f"{duration:.3f}",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(FINAL_WAV),
        ]
    )
    run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(FINAL_WAV),
            "-c:a",
            "libmp3lame",
            "-b:a",
            "128k",
            str(FINAL_MP3),
        ]
    )


def split_subtitle_text(text: str, start: float, end: float, first_index: int) -> tuple[list[str], int]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= 105:
            current = current + " " + sentence
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    if not chunks:
        chunks = [text]
    step = (end - start) / len(chunks)
    entries: list[str] = []
    index = first_index
    for i, chunk in enumerate(chunks):
        a = start + i * step
        b = end if i == len(chunks) - 1 else start + (i + 1) * step
        entries.append(f"{index}\n{format_srt_time(a)} --> {format_srt_time(b)}\n{chunk}\n\n")
        index += 1
    return entries, index


def write_srt(segments: list[Segment]) -> None:
    entries: list[str] = []
    index = 1
    for segment in segments:
        segment_entries, index = split_subtitle_text(segment.subtitle_text, segment.start, segment.end, index)
        entries.extend(segment_entries)
    FINAL_SRT.write_text("".join(entries), encoding="utf-8")


def mux_video() -> dict[str, object]:
    run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(SOURCE_VIDEO),
            "-i",
            str(FINAL_WAV),
            "-i",
            str(FINAL_SRT),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-map",
            "2:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-c:s",
            "mov_text",
            "-metadata:s:s:0",
            "language=eng",
            str(FINAL_VIDEO),
        ]
    )
    result = run(
        [
            FFPROBE,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=index,codec_type,codec_name,width,height,r_frame_rate",
            "-of",
            "json",
            str(FINAL_VIDEO),
        ]
    )
    return json.loads(result.stdout)


def write_contact_sheet() -> None:
    run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(FINAL_VIDEO),
            "-vf",
            "fps=1/30,scale=420:-1,tile=4x3",
            "-frames:v",
            "1",
            str(CONTACT_SHEET),
        ]
    )


def main() -> int:
    load_env()
    if not SOURCE_VIDEO.exists():
        raise FileNotFoundError(f"missing source video: {SOURCE_VIDEO}")
    clean_previous_timeline_audio()
    duration = video_duration(SOURCE_VIDEO)
    blocks = parse_scene_blocks()
    segments, fitted_paths, segment_reports = build_timed_segments(blocks)
    for segment in segments:
        print(json.dumps({"segment": segment.index, "speaker": segment.speaker, "start": segment.start, "end": segment.end}, sort_keys=True))
    write_concat_audio(fitted_paths, duration)
    write_srt(segments)
    streams = mux_video()
    write_contact_sheet()
    max_speed = max(float(item["speed_factor"]) for item in segment_reports)
    report = {
        "video": str(FINAL_VIDEO),
        "source_video": str(SOURCE_VIDEO),
        "srt": str(FINAL_SRT),
        "wav": str(FINAL_WAV),
        "mp3": str(FINAL_MP3),
        "contact_sheet": str(CONTACT_SHEET),
        "narration_markdown": str(NARRATION),
        "tts": {
            "provider": "MiniMax",
            "model": MODEL,
            "voice_a": VOICE_A,
            "voice_b": VOICE_B,
            "speed": VOICE_SPEED,
            "pitch": 0,
            "volume": 1.0,
            "normalization": "ffmpeg loudnorm=I=-20:TP=-1.5:LRA=11",
        },
        "duration_seconds": duration,
        "audio_duration_seconds": audio_duration(FINAL_WAV),
        "segments": len(segments),
        "max_speed_factor": max_speed,
        "speaker_labels_stripped_for_tts": True,
        "supported_tts_tags": ["(breath)", "(chuckle)", "(sighs)", "(laughs)", "<#0.4#>"],
        "streams": streams,
        "segment_reports": segment_reports,
        "source_article": "https://www.goodfire.ai/research/the-world-inside-neural-networks",
        "source_paper": "arXiv:2605.05115",
        "skills": {
            "paper_blackboard_video": "paper-blackboard-video skill used during local production",
        },
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
