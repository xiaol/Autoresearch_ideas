from __future__ import annotations

from pathlib import Path
import re
import subprocess


VOICE = "Samantha"
ROOT = Path(__file__).resolve().parent
NARRATION = ROOT / "narration.md"
OUT_DIR = ROOT / "voiceover"


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return slug.strip("_")


def say_available(voice: str) -> bool:
    try:
        res = subprocess.run(["say", "-v", voice, "test"], check=False, capture_output=True)
        return res.returncode == 0
    except FileNotFoundError:
        return False


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = NARRATION.read_text().strip()
    sections = re.split(r"^##\s+", text, flags=re.M)
    sections = [s for s in sections if s.strip()]

    use_voice = VOICE if say_available(VOICE) else None

    for section in sections:
        lines = section.splitlines()
        title = lines[0].strip()
        body = " ".join(line.strip() for line in lines[1:] if line.strip())
        if not body:
            continue

        out_file = OUT_DIR / f"{slugify(title)}.m4a"
        cmd = ["say"]
        if use_voice:
            cmd += ["-v", use_voice]
        cmd += ["-o", str(out_file), body]
        subprocess.run(cmd, check=True)
        print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
