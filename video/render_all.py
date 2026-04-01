from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
MANIM_FILE = ROOT / "manim" / "unimatrix_architectures.py"
VOICE_DIR = ROOT / "voiceover"
RENDER_DIR = ROOT / "renders"
MEDIA_ROOT = Path.cwd()

SCENES = [
    ("UniMatrixDynamic", "unimatrix_dynamic"),
    ("UniMatrixROSA", "unimatrix_rosa"),
    ("UniMatrixDeepEmbed", "unimatrix_deepembed"),
    ("UniMatrixStructured", "unimatrix_structured"),
    ("UniMatrixHybrid", "unimatrix_hybrid"),
    ("UniMatrixDualTimescale", "unimatrix_dual_timescale"),
    ("UniMatrixRuleMix", "unimatrix_rulemix"),
    ("UniMatrixSkewStable", "unimatrix_skewstable"),
    ("UniMatrixConvMix", "unimatrix_convmix"),
    ("UniMatrixStepConditioned", "unimatrix_stepconditioned"),
    ("UniMatrixSpectral", "unimatrix_spectral"),
    ("UniMatrixDiscovery", "unimatrix_discovery"),
]


def find_rendered(scene_name: str) -> Path:
    media_dir = MEDIA_ROOT / "media" / "videos" / "unimatrix_architectures"
    # pick highest resolution dir present
    candidates = list(media_dir.glob(f"**/{scene_name}.mp4"))
    if not candidates:
        raise FileNotFoundError(f"rendered video not found for {scene_name}")
    # choose shortest path (usually lowest quality) to keep consistent
    return sorted(candidates, key=lambda p: len(p.parts))[0]


def main() -> None:
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    for scene, out_name in SCENES:
        print(f"Rendering {scene} ...")
        subprocess.run(
            [
                "manim",
                "-q",
                "l",
                "-r",
                "1280,720",
                str(MANIM_FILE),
                scene,
            ],
            check=True,
        )

        video_path = find_rendered(scene)
        audio_path = VOICE_DIR / f"{out_name}.m4a"
        if not audio_path.exists() and out_name == "unimatrix_dual_timescale":
            audio_path = VOICE_DIR / "unimatrix_dualtimescale.m4a"
        if not audio_path.exists():
            print(f"audio missing for {scene}, skipping mux")
            continue

        out_path = RENDER_DIR / f"{out_name}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-i",
                str(audio_path),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(out_path),
            ],
            check=True,
        )
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
