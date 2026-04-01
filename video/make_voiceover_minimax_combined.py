from __future__ import annotations

from pathlib import Path
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from manim_lab.minimax_tts_client import MiniMaxTTSClient


ROOT_DIR = Path(__file__).resolve().parent
NARRATION = ROOT_DIR / "narration_combined.md"
OUT_DIR = ROOT_DIR / "voiceover"
SEGMENT_DIR = OUT_DIR / "segments"
SEGMENT_JSON = OUT_DIR / "segments.json"


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    return cleaned.strip("-") or "segment"


def split_segments(text: str) -> list[dict[str, str]]:
    segments: list[dict[str, str]] = []
    current_title = "intro"
    current_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current_lines:
                segments.append({"title": current_title, "text": "\n".join(current_lines).strip()})
            current_title = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        segments.append({"title": current_title, "text": "\n".join(current_lines).strip()})
    return [seg for seg in segments if seg["text"]]


def duration(path: Path) -> float:
    result = subprocess.run(
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
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def concat_mp3(files: list[Path], out_file: Path) -> None:
    concat_list = out_file.with_suffix(".concat.txt")
    lines = [f"file '{f.as_posix()}'" for f in files]
    concat_list.write_text("\n".join(lines))
    temp_wav = out_file.with_suffix(".tmp.wav")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c:a",
            "pcm_s16le",
            "-ar",
            "32000",
            "-ac",
            "1",
            str(temp_wav),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(temp_wav),
            "-c:a",
            "libmp3lame",
            "-b:a",
            "128k",
            str(out_file),
        ],
        check=True,
    )


def main() -> None:
    text = NARRATION.read_text().strip()
    if not text:
        raise SystemExit("Narration file is empty.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SEGMENT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "unimatrix_combined.mp3"

    segments = split_segments(text)
    if not segments:
        raise SystemExit("No segments found. Use headings like '## Title'.")

    tts = MiniMaxTTSClient()
    segment_files: list[Path] = []
    segment_meta: dict[str, float] = {}

    for index, segment in enumerate(segments, start=1):
        title = segment["title"]
        seg_text = segment["text"].strip()
        if not seg_text:
            continue
        filename = f"{index:02d}_{slugify(title)}.mp3"
        seg_path = SEGMENT_DIR / filename
        ok = tts.generate_audio(
            text=seg_text,
            output_filename=str(seg_path),
            voice_id="English_CaptivatingStoryteller",
            emotion="neutral",
            speed=1.0,
            volume=1.0,
            pitch=0,
            model="speech-2.6-hd",
        )
        if not ok:
            raise SystemExit("MiniMax TTS generation failed. Check MINIMAX_API_KEY.")
        segment_files.append(seg_path)
        segment_meta[title] = duration(seg_path)

    concat_mp3(segment_files, out_file)
    total = duration(out_file)
    SEGMENT_JSON.write_text(
        json.dumps(
            {"segments": segment_meta, "order": [s["title"] for s in segments], "total": total},
            indent=2,
        )
    )

    print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
