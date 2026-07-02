#!/usr/bin/env python3
"""Generate MiniMax narration, subtitles, and final muxed 4K video."""

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
ENV_PATH = Path("/home/xiaol/.codex/env/PaperX.env")
NARRATION = ROOT / "narration" / "narration_tts.md"
OUT_DIR = ROOT / "narration" / "minimax_audio"
FINAL_DIR = ROOT / "outputs"
SOURCE_VIDEO = Path(
    os.environ.get(
        "ATTNABL_SOURCE_VIDEO",
        str(ROOT / "renders" / "videos" / "attention_ablation_explainer" / "2160p30" / "AttentionAblationExplainer.mp4"),
    )
)
FINAL_VIDEO = Path(os.environ.get("ATTNABL_FINAL_VIDEO", str(FINAL_DIR / "AttentionAblationExplainer_4k_minimax_narrated.mp4")))
FINAL_SRT = Path(os.environ.get("ATTNABL_FINAL_SRT", str(FINAL_DIR / "AttentionAblationExplainer_4k_minimax_narrated.srt")))
REPORT = Path(os.environ.get("ATTNABL_NARRATION_REPORT", str(FINAL_DIR / "minimax_narration_report.json")))
VOICE_CHECK = OUT_DIR / "minimax_voice_check.json"
FINAL_WAV = OUT_DIR / "AttentionAblationExplainer_minimax.wav"
FINAL_MP3 = OUT_DIR / "AttentionAblationExplainer_minimax.mp3"
FFMPEG = os.environ.get("FFMPEG", "/home/xiaol/.local/bin/ffmpeg")
FFPROBE = os.environ.get("FFPROBE", "/home/xiaol/.local/bin/ffprobe")
VOICE_A = os.environ.get("ATTNABL_VOICE_A", "English_captivating_female1")
VOICE_B = os.environ.get("ATTNABL_VOICE_B", "English_CaptivatingStoryteller")
MODEL = os.environ.get("ATTNABL_MINIMAX_MODEL", "speech-2.8-hd")
VOICE_SPEED = float(os.environ.get("ATTNABL_MINIMAX_SPEED", "1.0"))
TAG_RE = re.compile(r"\((?:breath|chuckle|sighs|laughs)\)|<#[0-9.]+#>")


@dataclass(frozen=True)
class Turn:
    segment_index: int
    turn_index: int
    start: float
    end: float
    speaker: str
    text: str

    @property
    def voice(self) -> str:
        return VOICE_A if self.speaker == "A" else VOICE_B

    @property
    def subtitle_text(self) -> str:
        return re.sub(r"\s+", " ", TAG_RE.sub("", self.text)).strip()


@dataclass(frozen=True)
class Segment:
    index: int
    start: float
    end: float
    turns: list[Turn]

    @property
    def duration(self) -> float:
        return self.end - self.start


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


def minimax_url() -> str:
    base = os.environ.get("MINIMAX_TTS_BASE_URL") or os.environ.get("MINIMAX_BASE_URL") or "https://api.minimax.io/v1/t2a_v2"
    group_id = os.environ.get("MINIMAX_GROUP_ID")
    if group_id and "GroupId=" not in base:
        sep = "&" if "?" in base else "?"
        return f"{base}{sep}GroupId={group_id}"
    return base


def parse_time(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"Bad timestamp: {value}")


def srt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    whole = int(seconds)
    ms = int(round((seconds - whole) * 1000))
    if ms == 1000:
        whole += 1
        ms = 0
    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def duration(path: Path) -> float:
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


def parse_turns(body: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"(?m)(^|\s)([AB]):\s+", body))
    turns: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        speaker = match.group(2)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        text = re.sub(r"\s+", " ", body[start:end].strip())
        if text:
            turns.append((speaker, text))
    return turns


def parse_segments() -> list[Segment]:
    text = NARRATION.read_text(encoding="utf-8")
    matches = list(re.finditer(r"^##\s+([0-9:.]+)-([0-9:.]+)\s*$", text, re.M))
    segments: list[Segment] = []
    for seg_idx, match in enumerate(matches, start=1):
        start = parse_time(match.group(1))
        end = parse_time(match.group(2))
        body_start = match.end()
        body_end = matches[seg_idx].start() if seg_idx < len(matches) else len(text)
        parsed = parse_turns(text[body_start:body_end].strip())
        weights = [max(1, len(turn_text.split())) for _, turn_text in parsed]
        total = sum(weights)
        cursor = start
        turns: list[Turn] = []
        for turn_idx, (speaker, turn_text) in enumerate(parsed, start=1):
            turn_end = end if turn_idx == len(parsed) else start + (end - start) * sum(weights[:turn_idx]) / total
            turns.append(Turn(seg_idx, turn_idx, cursor, turn_end, speaker, turn_text))
            cursor = turn_end
        segments.append(Segment(seg_idx, start, end, turns))
    return segments


def query_voice() -> None:
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise RuntimeError("MINIMAX_API_KEY is not set")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data: dict | None = None
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = requests.post("https://api.minimax.io/v1/get_voice", headers=headers, json={"voice_type": "all"}, timeout=120)
            response.raise_for_status()
            data = response.json()
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(4 * attempt)
    if data is None:
        raise RuntimeError(f"MiniMax voice query failed: {last_error!r}")
    status = (data.get("base_resp") or {}).get("status_code", 0)
    if status not in (0, "0", None):
        raise RuntimeError(f"MiniMax voice query error: {data.get('base_resp')}")
    text = json.dumps(data)
    VOICE_CHECK.write_text(
        json.dumps(
            {
                "provider": "MiniMax",
                "voice_a": VOICE_A,
                "voice_b": VOICE_B,
                "voice_a_available": VOICE_A in text,
                "voice_b_available": VOICE_B in text,
                "top_level_keys": sorted(data.keys()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def cache_key(turn: Turn) -> str:
    payload = {"text": turn.text, "voice": turn.voice, "model": MODEL, "speed": VOICE_SPEED, "version": 1}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def cached(path: Path, key: str) -> bool:
    meta = path.with_suffix(path.suffix + ".json")
    if not path.exists() or path.stat().st_size < 1000 or not meta.exists():
        return False
    try:
        return json.loads(meta.read_text(encoding="utf-8")).get("cache_key") == key
    except json.JSONDecodeError:
        return False


def synth_turn(turn: Turn) -> Path:
    raw_path = OUT_DIR / f"segment_{turn.segment_index:02d}_turn_{turn.turn_index:02d}_{turn.speaker}_raw.mp3"
    key = cache_key(turn)
    if cached(raw_path, key):
        return raw_path
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise RuntimeError("MINIMAX_API_KEY is not set")
    payload = {
        "model": MODEL,
        "text": turn.text,
        "stream": False,
        "language_boost": "English",
        "output_format": "hex",
        "voice_setting": {"voice_id": turn.voice, "speed": VOICE_SPEED, "vol": 1.0, "pitch": 0},
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
        raise RuntimeError(f"MiniMax TTS failed for segment {turn.segment_index}: {last_error!r}")
    status = (data.get("base_resp") or {}).get("status_code", 0)
    if status not in (0, "0", None):
        raise RuntimeError(f"MiniMax TTS error for segment {turn.segment_index}: {data.get('base_resp')}")
    payload_data = data.get("data") or {}
    audio_hex = payload_data.get("audio")
    audio_url = payload_data.get("audio_url")
    if audio_hex:
        raw_path.write_bytes(bytes.fromhex(audio_hex))
    elif audio_url:
        audio = requests.get(audio_url, timeout=180)
        audio.raise_for_status()
        raw_path.write_bytes(audio.content)
    else:
        raise RuntimeError(f"No audio in MiniMax response: {json.dumps(data)[:500]}")
    raw_path.with_suffix(raw_path.suffix + ".json").write_text(
        json.dumps(
            {
                "cache_key": key,
                "provider": "MiniMax",
                "model": MODEL,
                "voice": turn.voice,
                "speaker": turn.speaker,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return raw_path


def loudnorm(path: Path) -> Path:
    norm = OUT_DIR / f"{path.stem}_norm.wav"
    run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
            "-af",
            "loudnorm=I=-20:TP=-1.5:LRA=11",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(norm),
        ]
    )
    return norm


def concat_files(paths: list[Path], out_path: Path) -> None:
    concat = out_path.with_suffix(".concat.txt")
    concat.write_text("".join(f"file '{path.resolve()}'\n" for path in paths), encoding="utf-8")
    run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-ar", "48000", "-ac", "2", str(out_path)])


def fit_segment(segment: Segment, turn_paths: list[Path]) -> tuple[Path, dict[str, float]]:
    joined = OUT_DIR / f"segment_{segment.index:02d}_joined.wav"
    concat_files(turn_paths, joined)
    raw_duration = duration(joined)
    speed = 1.0  # voice is NEVER compressed; windows are derived from natural TTS durations
    if raw_duration > segment.duration - 0.02:
        raise RuntimeError(
            f"Segment {segment.index} audio ({raw_duration:.2f}s) does not fit its window "
            f"({segment.duration:.2f}s). Re-run scripts/apply_windows.py and re-render the scene; "
            "speech must not be sped up."
        )
    filters: list[str] = ["loudnorm=I=-20:TP=-1.5:LRA=11", "apad"]
    fitted = OUT_DIR / f"segment_{segment.index:02d}_fitted.wav"
    run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(joined),
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


def write_final_audio(paths: list[Path], target_duration: float) -> None:
    concat_files(paths, FINAL_WAV)
    tmp = FINAL_WAV.with_suffix(".tmp.wav")
    FINAL_WAV.rename(tmp)
    run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(tmp),
            "-af",
            "loudnorm=I=-20:TP=-1.5:LRA=11,apad",
            "-t",
            f"{target_duration:.3f}",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(FINAL_WAV),
        ]
    )
    tmp.unlink(missing_ok=True)
    run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(FINAL_WAV), "-c:a", "libmp3lame", "-b:a", "160k", str(FINAL_MP3)])


def write_srt(segments: list[Segment], target_duration: float) -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[str] = []
    index = 1
    for segment in segments:
        for turn in segment.turns:
            end = min(turn.end, max(turn.start + 0.05, target_duration - 0.05))
            entries.extend([str(index), f"{srt_time(turn.start)} --> {srt_time(end)}", turn.subtitle_text, ""])
            index += 1
    start = max(0.0, target_duration - 0.04)
    end = max(start + 0.02, target_duration - 0.01)
    entries.extend([str(index), f"{srt_time(start)} --> {srt_time(end)}", " ", ""])
    FINAL_SRT.write_text("\n".join(entries) + "\n", encoding="utf-8")


def mux_video() -> dict[str, object]:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
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
            "160k",
            "-c:s",
            "mov_text",
            "-metadata:s:s:0",
            "language=eng",
            "-shortest",
            str(FINAL_VIDEO),
        ]
    )
    return json.loads(
        run(
            [
                FFPROBE,
                "-v",
                "error",
                "-show_entries",
                "format=duration,size:stream=index,codec_type,codec_name,width,height,r_frame_rate,duration,nb_frames",
                "-of",
                "json",
                str(FINAL_VIDEO),
            ]
        ).stdout
    )


def main() -> int:
    load_env()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    query_voice()
    segments = parse_segments()
    if not segments:
        raise RuntimeError("No narration segments found")
    if not SOURCE_VIDEO.exists():
        raise RuntimeError(f"Source video missing: {SOURCE_VIDEO}")
    video_duration = duration(SOURCE_VIDEO)
    target_duration = min(video_duration, max(segment.end for segment in segments))
    fitted_segments: list[Path] = []
    segment_reports: list[dict[str, object]] = []
    for segment in segments:
        print(json.dumps({"segment": segment.index, "turns": len(segment.turns), "duration": segment.duration}, sort_keys=True))
        turn_paths: list[Path] = []
        turn_reports: list[dict[str, object]] = []
        for turn in segment.turns:
            raw = synth_turn(turn)
            norm = loudnorm(raw)
            turn_paths.append(norm)
            turn_reports.append(
                {
                    "segment": turn.segment_index,
                    "turn": turn.turn_index,
                    "speaker": turn.speaker,
                    "voice": turn.voice,
                    "raw_path": str(raw),
                    "normalized_path": str(norm),
                    "raw_duration": duration(raw),
                    "normalized_duration": duration(norm),
                }
            )
        fitted, metrics = fit_segment(segment, turn_paths)
        fitted_segments.append(fitted)
        segment_reports.append({"segment": segment.index, **metrics, "turns": turn_reports})
    write_final_audio(fitted_segments, target_duration)
    write_srt(segments, target_duration)
    ffprobe = mux_video()
    max_speed = max(float(item["speed_factor"]) for item in segment_reports)
    report = {
        "title": "Four Sparse Attentions, One Rigged Race: Auditing a Pretraining Ablation",
        "tts": {
            "provider": "MiniMax",
            "model": MODEL,
            "voice_a": VOICE_A,
            "voice_b": VOICE_B,
            "speed": VOICE_SPEED,
            "pitch": 0,
            "volume": 1.0,
            "voice_modify": None,
            "normalization": "ffmpeg loudnorm=I=-20:TP=-1.5:LRA=11 per turn, per segment, and final mix",
            "voice_check": str(VOICE_CHECK),
        },
        "max_speed_factor": max_speed,
        "source_video": str(SOURCE_VIDEO),
        "video_duration": video_duration,
        "target_duration": target_duration,
        "final_video": str(FINAL_VIDEO),
        "wav": str(FINAL_WAV),
        "mp3": str(FINAL_MP3),
        "srt": str(FINAL_SRT),
        "audio_duration": duration(FINAL_WAV),
        "ffprobe": ffprobe,
        "segment_reports": segment_reports,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"final_video": str(FINAL_VIDEO), "report": str(REPORT), "max_speed_factor": max_speed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

