from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
MANIM_FILE = ROOT / "manim" / "unimatrix_combined.py"
VOICE_FILE = ROOT / "voiceover" / "unimatrix_combined.mp3"
RENDER_DIR = ROOT / "renders"
MEDIA_ROOT = Path.cwd()


def find_rendered(scene_name: str) -> Path:
    media_dir = MEDIA_ROOT / "media" / "videos" / "unimatrix_combined"
    candidates = list(media_dir.glob(f"**/{scene_name}.mp4"))
    if not candidates:
        raise FileNotFoundError(f"rendered video not found for {scene_name}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


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


def mux_with_audio(video_path: Path, audio_path: Path, out_path: Path) -> None:
    vid_dur = duration(video_path)
    aud_dur = duration(audio_path)
    temp_audio = out_path.with_suffix(".tmp.m4a")

    if aud_dur > vid_dur:
        raise SystemExit(
            f"Audio ({aud_dur:.2f}s) is longer than video ({vid_dur:.2f}s). "
            "Add more animation time so the visuals match the narration."
        )

    if aud_dur < vid_dur:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(audio_path),
                "-filter_complex",
                f"apad,atrim=0:{vid_dur:.2f}",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                str(temp_audio),
            ],
            check=True,
        )
        mux_audio = temp_audio
    else:
        mux_audio = audio_path

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(mux_audio),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            str(out_path),
        ],
        check=True,
    )


def main() -> None:
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    if not VOICE_FILE.exists():
        raise SystemExit("Voiceover missing. Run make_voiceover_minimax_combined.py first.")

    subprocess.run(
        [
            "manim",
            "-q",
            "h",
            "-r",
            "1920,1080",
            "--fps",
            "30",
            str(MANIM_FILE),
            "UniMatrixCombined",
        ],
        check=True,
    )

    video_path = find_rendered("UniMatrixCombined")
    out_path = RENDER_DIR / "unimatrix_combined.mp4"
    mux_with_audio(video_path, VOICE_FILE, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
