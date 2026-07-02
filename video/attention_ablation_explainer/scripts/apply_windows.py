#!/usr/bin/env python3
"""Set narration segment windows = measured natural TTS duration + intentional hold.

Reads narration/natural_durations.json (from measure_tts.py), rewrites the
`## MM:SS.ss-MM:SS.ss` headers in narration/narration_tts.md cumulatively, and
writes narration/beat_windows.json (segment index -> absolute end time) for the
scene's BEATS table. Voice is never sped up: window >= natural by construction.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NARRATION = ROOT / "narration" / "narration_tts.md"
DURATIONS = ROOT / "narration" / "natural_durations.json"
WINDOWS = ROOT / "narration" / "beat_windows.json"

HOLD = 3.0        # intentional breathing hold per beat (seconds)
FINAL_EXTRA = 2.0  # extra hold on the closing beat


def fmt(seconds: float) -> str:
    m, s = divmod(seconds, 60.0)
    return f"{int(m):02d}:{s:05.2f}"


def main() -> None:
    report = json.loads(DURATIONS.read_text())
    naturals = {}
    for seg in report["segments"]:
        if seg["status"] != "ok":
            raise SystemExit(f"segment {seg['index']} not measured: {seg['status']}")
        naturals[seg["index"]] = seg["natural_sec"]

    n = len(naturals)
    text = NARRATION.read_text(encoding="utf-8")
    headers = list(re.finditer(r"^##\s+([0-9:.]+)-([0-9:.]+)\s*$", text, re.M))
    if len(headers) != n:
        raise SystemExit(f"{len(headers)} headers vs {n} measured segments")

    cursor = 0.0
    ends: dict[int, float] = {}
    new_text = []
    last = 0
    for idx, match in enumerate(headers, start=1):
        hold = HOLD + (FINAL_EXTRA if idx == n else 0.0)
        start = cursor
        cursor = round(cursor + naturals[idx] + hold, 2)
        ends[idx] = cursor
        new_text.append(text[last : match.start()])
        new_text.append(f"## {fmt(start)}-{fmt(cursor)}")
        last = match.end()
    new_text.append(text[last:])
    NARRATION.write_text("".join(new_text), encoding="utf-8")
    WINDOWS.write_text(json.dumps(ends, indent=2), encoding="utf-8")
    print(f"total video duration: {cursor:.2f}s ({fmt(cursor)})")
    for idx in sorted(ends):
        print(f"  beat {idx:02d}: natural {naturals[idx]:6.2f}s  ends at {fmt(ends[idx])}")
    print(f"wrote {WINDOWS}")


if __name__ == "__main__":
    main()
