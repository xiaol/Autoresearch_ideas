#!/usr/bin/env python3
"""Synthesize each narration turn with MiniMax and report NATURAL durations.

Narration-first contract: these measured durations (plus intentional holds)
become the segment windows in narration_tts.md, and the scene timeline is coded
against them. Segments containing PLACEHOLDER are skipped (measured later).

Writes narration/natural_durations.json.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "scripts" / "generate_minimax_narration.py"

spec = importlib.util.spec_from_file_location("gen_narration", GEN)
gen = importlib.util.module_from_spec(spec)
sys.modules["gen_narration"] = gen
spec.loader.exec_module(gen)


def main() -> int:
    gen.load_env()
    gen.OUT_DIR.mkdir(parents=True, exist_ok=True)
    segments = gen.parse_segments()
    report: dict[str, object] = {"segments": []}
    total = 0.0
    for segment in segments:
        seg_text = " ".join(t.text for t in segment.turns)
        if "PLACEHOLDER" in seg_text:
            report["segments"].append(
                {"index": segment.index, "status": "skipped_placeholder", "window": [segment.start, segment.end]}
            )
            print(f"segment {segment.index:02d}: SKIPPED (placeholder)")
            continue
        turn_durations = []
        for turn in segment.turns:
            path = gen.synth_turn(turn)
            turn_durations.append(gen.duration(path))
        natural = sum(turn_durations)
        total += natural
        report["segments"].append(
            {
                "index": segment.index,
                "status": "ok",
                "window": [segment.start, segment.end],
                "window_len": round(segment.duration, 2),
                "natural_sec": round(natural, 2),
                "turn_sec": [round(d, 2) for d in turn_durations],
                "fits": natural <= segment.duration - 0.05,
            }
        )
        print(
            f"segment {segment.index:02d}: natural {natural:6.2f}s vs window {segment.duration:6.2f}s "
            f"{'OK' if natural <= segment.duration - 0.05 else 'WINDOW TOO SHORT'}"
        )
    report["total_natural_sec"] = round(total, 2)
    out = ROOT / "narration" / "natural_durations.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
