#!/usr/bin/env python3
"""Compare attention-variant training runs: table + val-loss curves + JSON export.

Usage:
    python compare_runs.py RUN_DIR [RUN_DIR2 ...] [--out OUT_DIR]

Each RUN_DIR is an output of train_attention_variants.py (contains
<variant>/metrics.jsonl). Multiple run dirs are treated as seeds/replicas of
the same experiment and aggregated per variant.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARIANT_LABELS = {
    "dsa": "DSA (oracle top-k)",
    "lsa": "LSA (local + block recall)",
    "csa": "CSA (compressed + top blocks)",
    "hca": "HCA (compressed, all blocks)",
}
VARIANT_COLORS = {"dsa": "#e8873a", "lsa": "#4d9de0", "csa": "#7bc96f", "hca": "#c678dd"}


def load_run(run_dir: Path) -> dict[str, list[dict]]:
    curves: dict[str, list[dict]] = {}
    for sub in sorted(run_dir.iterdir()):
        metrics = sub / "metrics.jsonl"
        if not metrics.is_file():
            continue
        rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
        if rows:
            curves[sub.name] = rows
    if not curves:
        raise SystemExit(f"no metrics.jsonl found under {run_dir}")
    return curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.out or args.run_dirs[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = {rd.name: load_run(rd) for rd in args.run_dirs}

    # aggregate final/best per variant across runs
    agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run_name, curves in runs.items():
        for variant, rows in curves.items():
            last = rows[-1]
            agg[variant]["final_val"].append(last["val_loss"])
            agg[variant]["best_val"].append(last["best_val_loss"])
            agg[variant]["tokens_per_sec"].append(last["tokens_per_sec"])
            agg[variant]["params"].append(last["params"])

    print(f"{'variant':<8}{'final_val (mean over runs)':<30}{'best_val':<12}{'params':<10}{'tok/s':<10}")
    table = []
    for variant in sorted(agg, key=lambda v: sum(agg[v]["final_val"]) / len(agg[v]["final_val"])):
        vals = agg[variant]
        n = len(vals["final_val"])
        mean_final = sum(vals["final_val"]) / n
        spread = (max(vals["final_val"]) - min(vals["final_val"])) if n > 1 else 0.0
        mean_best = sum(vals["best_val"]) / n
        row = {
            "variant": variant,
            "runs": n,
            "final_val_mean": mean_final,
            "final_val_spread": spread,
            "best_val_mean": mean_best,
            "params": int(vals["params"][0]),
            "tokens_per_sec_mean": sum(vals["tokens_per_sec"]) / n,
            "final_vals": vals["final_val"],
        }
        table.append(row)
        print(
            f"{variant:<8}{mean_final:.4f} (spread {spread:.4f}, n={n})     "
            f"{mean_best:<12.4f}{row['params']:<10}{row['tokens_per_sec_mean']:<10.0f}"
        )

    # curves plot: one panel, val loss vs tokens, mean across runs per variant
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    export_curves: dict[str, dict] = {}
    for variant in sorted(agg):
        per_run = [runs[rn][variant] for rn in runs if variant in runs[rn]]
        steps = [r["tokens"] for r in per_run[0]]
        vals = [[r["val_loss"] for r in rows] for rows in per_run]
        n_pts = min(len(v) for v in vals)
        mean_curve = [sum(v[i] for v in vals) / len(vals) for i in range(n_pts)]
        export_curves[variant] = {"tokens": steps[:n_pts], "val_loss_mean": mean_curve, "val_loss_runs": [v[:n_pts] for v in vals]}
        ax.plot(steps[:n_pts], mean_curve, label=VARIANT_LABELS.get(variant, variant), color=VARIANT_COLORS.get(variant), linewidth=2)
        for v in vals:
            ax.plot(steps[:n_pts], v[:n_pts], color=VARIANT_COLORS.get(variant), alpha=0.25, linewidth=1)
    ax.set_xlabel("training tokens", color="white")
    ax.set_ylabel("validation loss (nats/byte-token)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#555555")
    ax.legend(facecolor="black", labelcolor="white", edgecolor="#555555")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    curve_path = out_dir / "val_loss_curves.png"
    fig.savefig(curve_path, facecolor="black")

    # zoomed tail plot (last 60% of tokens) where variants separate
    ax.set_xlim(left=export_curves[min(export_curves)]["tokens"][-1] * 0.4)
    tail_vals = [
        v
        for c in export_curves.values()
        for series in ([c["val_loss_mean"]] + c["val_loss_runs"])
        for tok, v in zip(c["tokens"], series)
        if tok >= c["tokens"][-1] * 0.4
    ]
    pad = (max(tail_vals) - min(tail_vals)) * 0.1 + 1e-4
    ax.set_ylim(min(tail_vals) - pad, max(tail_vals) + pad)
    tail_path = out_dir / "val_loss_curves_tail.png"
    fig.savefig(tail_path, facecolor="black")

    export = {"runs": sorted(runs), "table": table, "curves": export_curves}
    export_path = out_dir / "comparison.json"
    export_path.write_text(json.dumps(export, indent=2))
    print(f"\nwrote {curve_path}\nwrote {tail_path}\nwrote {export_path}")


if __name__ == "__main__":
    main()
