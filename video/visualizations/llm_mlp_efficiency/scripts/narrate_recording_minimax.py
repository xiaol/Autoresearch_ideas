#!/usr/bin/env python3
"""Generate a narrated MP4 for the recorded visualizer demo.

The script reads an SRT file, generates one MiniMax TTS segment per subtitle
cue, pads or lightly speeds up each segment to match the cue duration, then
muxes the narration and subtitles into the source video.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import requests


DEFAULT_VIDEO = Path("/home/xiaol/2026-06-03 23-19-10.mp4")
DEFAULT_SRT = Path("narration-assets/mlp_efficiency_narration.srt")
DEFAULT_OUT = Path("narration-assets/mlp_efficiency_narrated.mp4")
DEFAULT_ENV = Path("/home/xiaol/.codex/env/PaperX.env")
VOICE_ID = "English_CaptivatingStoryteller"
MODEL = "speech-2.8-hd"
MAX_SPEED_FACTOR = 1.18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Narrate the LLM visualizer recording with MiniMax TTS.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="Source MP4.")
    parser.add_argument("--srt", type=Path, default=DEFAULT_SRT, help="Subtitle/narration SRT.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Final narrated MP4.")
    parser.add_argument("--work-dir", type=Path, default=Path("narration-assets/voiceover"), help="Generated audio cache.")
    parser.add_argument("--voice-id", default=VOICE_ID, help="MiniMax voice id.")
    parser.add_argument("--speed", type=float, default=1.0, help="MiniMax speech speed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env(DEFAULT_ENV)
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        print("Missing MINIMAX_API_KEY. Put it in the shell or /home/xiaol/.codex/env/PaperX.env.", file=sys.stderr)
        return 2

    video = args.video.resolve()
    srt = args.srt.resolve()
    out_file = args.out.resolve()
    work_dir = args.work_dir.resolve()
    raw_dir = work_dir / "raw"
    fitted_dir = work_dir / "fitted"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fitted_dir.mkdir(parents=True, exist_ok=True)

    cues = parse_srt(srt.read_text(encoding="utf-8"))
    if not cues:
        print(f"No subtitle cues found in {srt}", file=sys.stderr)
        return 2

    voice_query_path = work_dir / "minimax_voice_query.json"
    query_voices(api_key, voice_query_path)

    segment_reports: list[dict[str, Any]] = []
    fitted_segments: list[Path] = []
    for cue in cues:
        raw = generate_segment(cue, raw_dir, api_key, args.voice_id, args.speed)
        fitted = fitted_dir / f"{cue['index']:02d}_fit.wav"
        raw_duration = media_duration(raw)
        target_duration = cue["end"] - cue["start"]
        speed_factor = max(1.0, raw_duration / target_duration) if target_duration > 0 else 1.0
        if speed_factor > MAX_SPEED_FACTOR:
            raise RuntimeError(
                f"Cue {cue['index']} is too long for its visual slot: "
                f"{raw_duration:.2f}s audio into {target_duration:.2f}s needs {speed_factor:.2f}x."
            )
        fit_segment(raw, fitted, target_duration, speed_factor)
        fitted_duration = media_duration(fitted)
        fitted_segments.append(fitted)
        segment_reports.append(
            {
                "index": cue["index"],
                "targetDuration": target_duration,
                "rawDuration": raw_duration,
                "fittedDuration": fitted_duration,
                "speedFactor": speed_factor,
                "text": cue["text"],
                "raw": str(raw),
                "fitted": str(fitted),
            }
        )

    narration_wav = work_dir / "mlp_efficiency_narration.wav"
    narration_padded_wav = work_dir / "mlp_efficiency_narration_padded.wav"
    concat_wav(fitted_segments, narration_wav)
    pad_audio_to_duration(narration_wav, narration_padded_wav, media_duration(video))
    mux_video(video, narration_padded_wav, srt, out_file)
    ffprobe = probe_streams(out_file)
    report = {
        "sourceVideo": str(video),
        "srt": str(srt),
        "voiceProvider": "MiniMax",
        "voiceModel": MODEL,
        "voiceId": args.voice_id,
        "voiceQuery": str(voice_query_path),
        "normalization": "ffmpeg loudnorm=I=-20:TP=-1.5:LRA=11",
        "maxSpeedFactor": max(item["speedFactor"] for item in segment_reports),
        "rawNarrationWav": str(narration_wav),
        "narrationWav": str(narration_padded_wav),
        "finalMp4": str(out_file),
        "segments": segment_reports,
        "ffprobe": ffprobe,
    }
    report_path = work_dir / "render_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_file}")
    print(f"Wrote {report_path}")
    return 0


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export "):]
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def parse_srt(text: str) -> list[dict[str, Any]]:
    cues: list[dict[str, Any]] = []
    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            continue
        index = int(lines[0])
        start_text, end_text = [part.strip() for part in lines[1].split("-->")]
        cues.append(
            {
                "index": index,
                "start": parse_timestamp(start_text),
                "end": parse_timestamp(end_text),
                "text": " ".join(lines[2:]),
            }
        )
    return cues


def parse_timestamp(value: str) -> float:
    hours, minutes, rest = value.split(":")
    seconds, millis = rest.split(",")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def minimax_tts_url() -> str:
    base = os.environ.get("MINIMAX_TTS_BASE_URL") or os.environ.get("MINIMAX_BASE_URL") or "https://api.minimax.io/v1/t2a_v2"
    group_id = os.environ.get("MINIMAX_GROUP_ID")
    if group_id and "GroupId=" not in base:
        separator = "&" if "?" in base else "?"
        return f"{base}{separator}GroupId={group_id}"
    return base


def query_voices(api_key: str, out_path: Path) -> None:
    if out_path.exists():
        return
    response = requests.post(
        "https://api.minimax.io/v1/get_voice",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"voice_type": "all"},
        timeout=60,
    )
    response.raise_for_status()
    out_path.write_text(json.dumps(response.json(), indent=2), encoding="utf-8")


def generate_segment(cue: dict[str, Any], raw_dir: Path, api_key: str, voice_id: str, speed: float) -> Path:
    text = cue["text"]
    key = hashlib.sha256(f"{MODEL}\n{voice_id}\n{speed}\n{text}".encode("utf-8")).hexdigest()[:16]
    out_path = raw_dir / f"{cue['index']:02d}_{key}.mp3"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    payload = {
        "model": MODEL,
        "text": text,
        "stream": False,
        "language_boost": "English",
        "output_format": "hex",
        "voice_setting": {"voice_id": voice_id, "speed": speed, "vol": 1.0, "pitch": 0},
        "audio_setting": {"sample_rate": 32000, "bitrate": 128000, "format": "mp3", "channel": 1},
    }
    response = requests.post(
        minimax_tts_url(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    base_resp = data.get("base_resp") or {}
    status = base_resp.get("status_code", 0)
    if status not in (0, "0", None):
        raise RuntimeError(base_resp)
    payload_data = data.get("data") or {}
    audio_hex = payload_data.get("audio")
    audio_url = payload_data.get("audio_url")
    if audio_hex:
        out_path.write_bytes(bytes.fromhex(audio_hex))
    elif audio_url:
        audio_response = requests.get(audio_url, timeout=180)
        audio_response.raise_for_status()
        out_path.write_bytes(audio_response.content)
    else:
        raise RuntimeError("MiniMax response did not include audio data.")
    return out_path


def fit_segment(raw: Path, out_path: Path, target_duration: float, speed_factor: float) -> None:
    filters = ["loudnorm=I=-20:TP=-1.5:LRA=11"]
    if speed_factor > 1.001:
        filters.insert(0, f"atempo={speed_factor:.6f}")
    filters.append("apad")
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(raw),
            "-af",
            ",".join(filters),
            "-t",
            f"{target_duration:.3f}",
            "-c:a",
            "pcm_s16le",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(out_path),
        ]
    )


def concat_wav(files: list[Path], out_file: Path) -> None:
    concat_file = out_file.with_suffix(".concat.txt")
    concat_file.write_text("\n".join(f"file '{path.as_posix()}'" for path in files), encoding="utf-8")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c:a", "pcm_s16le", str(out_file)])


def pad_audio_to_duration(audio: Path, out_file: Path, duration: float) -> None:
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(audio),
            "-af",
            "apad",
            "-t",
            f"{duration:.3f}",
            "-c:a",
            "pcm_s16le",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(out_file),
        ]
    )


def mux_video(video: Path, audio: Path, srt: Path, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-i",
            str(audio),
            "-i",
            str(srt),
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
            "192k",
            "-c:s",
            "mov_text",
            "-metadata:s:s:0",
            "language=eng",
            str(out_file),
        ]
    )


def media_duration(path: Path) -> float:
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
        capture=True,
    )
    return float(result.stdout.strip())


def probe_streams(path: Path) -> dict[str, Any]:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=index,codec_type,codec_name,width,height,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        capture=True,
    )
    return json.loads(result.stdout)


def run(command: list[str], capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=capture,
    )


if __name__ == "__main__":
    raise SystemExit(main())
