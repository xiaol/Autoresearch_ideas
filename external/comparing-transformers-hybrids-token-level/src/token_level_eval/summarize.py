from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from token_level_eval.common import read_jsonl
from token_level_eval.tagging import CONTENT_COARSE


def load_rows(path: str | Path):
    import pandas as pd

    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return pd.DataFrame(rows)


def _explode_tags(df, column: str = "tags"):
    exploded = df.copy()
    exploded[column] = exploded[column].apply(lambda value: value if isinstance(value, list) and value else ["Untagged"])
    return exploded.explode(column).rename(columns={column: "tag"})


def tag_summary(df):
    import numpy as np

    exploded = _explode_tags(df, "tags")
    grouped = exploded.groupby(["domain", "tag"], dropna=False)
    summary = grouped.agg(
        count=("loss_gap", "size"),
        mean_gap=("loss_gap", "mean"),
        std_gap=("loss_gap", "std"),
        mean_transformer_nll=("loss_transformer", "mean"),
        mean_hybrid_nll=("loss_hybrid", "mean"),
        distinct_token_types=("token_id", "nunique"),
    ).reset_index()
    summary["sem_gap"] = summary["std_gap"] / np.sqrt(summary["count"].clip(lower=1))
    summary["hybrid_probability_ratio"] = np.exp(summary["mean_gap"])
    summary = summary.sort_values(["domain", "mean_gap"], ascending=[True, False])
    return summary


def aggregate_tag_summary(df):
    exploded = _explode_tags(df, "aggregate_tags")
    grouped = exploded.groupby(["domain", "tag"], dropna=False)
    return grouped.agg(
        count=("loss_gap", "size"),
        mean_gap=("loss_gap", "mean"),
        mean_transformer_nll=("loss_transformer", "mean"),
        mean_hybrid_nll=("loss_hybrid", "mean"),
    ).reset_index().sort_values(["domain", "mean_gap"], ascending=[True, False])


def copy_summary(df, max_copy_ngram: int):
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for n in range(1, max_copy_ngram + 1):
        column = f"copy_{n}"
        if column not in df:
            continue
        subset = df[df[column].astype(bool)]
        rows.append(
            {
                "ngram": n,
                "count": int(len(subset)),
                "mean_gap": float(subset["loss_gap"].mean()) if len(subset) else None,
                "mean_transformer_nll": float(subset["loss_transformer"].mean()) if len(subset) else None,
                "mean_hybrid_nll": float(subset["loss_hybrid"].mean()) if len(subset) else None,
            }
        )
    return pd.DataFrame(rows)


def bracket_summary(df):
    import pandas as pd

    exploded = _explode_tags(df, "tags")
    subset = exploded[exploded["tag"].isin(["Open Bracket", "Close Bracket"])]
    if subset.empty:
        return pd.DataFrame(columns=["domain", "tag", "count", "mean_gap"])
    return subset.groupby(["domain", "tag"]).agg(
        count=("loss_gap", "size"),
        mean_gap=("loss_gap", "mean"),
        mean_transformer_nll=("loss_transformer", "mean"),
        mean_hybrid_nll=("loss_hybrid", "mean"),
    ).reset_index()


def _loss_record(name: str, subset) -> dict[str, Any]:
    return {
        "filter": name,
        "count": int(len(subset)),
        "transformer_nll": float(subset["loss_transformer"].mean()) if len(subset) else None,
        "hybrid_nll": float(subset["loss_hybrid"].mean()) if len(subset) else None,
        "gap": float(subset["loss_gap"].mean()) if len(subset) else None,
    }


def filtered_losses(df, *, top_k: int, no_copy_upto: int, copy_only_n: int) -> list[dict[str, Any]]:
    import numpy as np

    records = [_loss_record("all_tokens", df)]
    summary = tag_summary(df)
    open_class_tags = [tag for tag in summary["tag"].tolist() if tag in CONTENT_COARSE]
    top_tags = open_class_tags[:top_k]
    if top_tags:
        no_copy_mask = np.ones(len(df), dtype=bool)
        for n in range(1, no_copy_upto + 1):
            column = f"copy_{n}"
            if column in df:
                no_copy_mask &= ~df[column].fillna(False).astype(bool).to_numpy()
        tag_mask = df["tags"].apply(lambda tags: bool(set(tags) & set(top_tags)))
        records.append(_loss_record(f"top_{top_k}_open_class_no_copy_{no_copy_upto}", df[tag_mask & no_copy_mask]))
    copy_column = f"copy_{copy_only_n}"
    if copy_column in df:
        records.append(_loss_record(f"copy_{copy_only_n}_only", df[df[copy_column].fillna(False).astype(bool)]))
    return records


def _prepare_regression_frame(df, max_copy_ngram: int):
    import pandas as pd

    reg = df.copy()
    reg["log_prev_distance"] = reg["prev_distance"].apply(lambda value: math.log1p(float(value)) if pd.notna(value) else 0.0)
    counts = reg["token_id"].value_counts().to_dict()
    reg["log_freq"] = reg["token_id"].map(lambda token_id: math.log1p(counts.get(token_id, 0)))
    reg["mean_loss_sq"] = reg["mean_loss"].astype(float) ** 2
    for n in range(1, max_copy_ngram + 1):
        column = f"copy_{n}"
        if column in reg:
            reg[column] = reg[column].fillna(False).astype(float)
    return reg


def run_ols_regression(df, max_copy_ngram: int):
    import numpy as np
    import pandas as pd

    reg = _prepare_regression_frame(df, max_copy_ngram)
    numeric_cols = ["rel_pos", "mean_loss", "mean_loss_sq", "log_prev_distance", "log_freq"]
    numeric_cols.extend([f"copy_{n}" for n in range(1, max_copy_ngram + 1) if f"copy_{n}" in reg])
    cat_cols = ["domain", "primary_tag", "word_position"]
    matrix_parts = [pd.Series(1.0, index=reg.index, name="intercept")]
    matrix_parts.extend([reg[col].astype(float) for col in numeric_cols])
    for col in cat_cols:
        dummies = pd.get_dummies(reg[col].fillna("missing"), prefix=col, drop_first=True, dtype=float)
        matrix_parts.append(dummies)
    x = pd.concat(matrix_parts, axis=1).fillna(0.0)
    y = reg["loss_gap"].astype(float).to_numpy()
    beta, residuals, rank, singular_values = np.linalg.lstsq(x.to_numpy(dtype=float), y, rcond=None)
    return pd.DataFrame(
        {
            "feature": x.columns,
            "coefficient": beta,
            "rank": rank,
            "residual_sum_squares": float(residuals[0]) if len(residuals) else None,
            "min_singular_value": float(np.min(singular_values)) if len(singular_values) else None,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize token-level transformer/hybrid rows.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-copy-ngram", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-copy-upto", type=int, default=4)
    parser.add_argument("--copy-only-n", type=int, default=5)
    parser.add_argument("--run-regression", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_rows(args.input_jsonl)

    tag_summary(df).to_csv(output_dir / "tag_summary.csv", index=False)
    aggregate_tag_summary(df).to_csv(output_dir / "aggregate_tag_summary.csv", index=False)
    copy_summary(df, args.max_copy_ngram).to_csv(output_dir / "copy_summary.csv", index=False)
    bracket_summary(df).to_csv(output_dir / "bracket_summary.csv", index=False)
    filters = filtered_losses(
        df,
        top_k=args.top_k,
        no_copy_upto=args.no_copy_upto,
        copy_only_n=args.copy_only_n,
    )
    (output_dir / "filtered_losses.json").write_text(json.dumps(filters, indent=2), encoding="utf-8")

    if args.run_regression:
        run_ols_regression(df, args.max_copy_ngram).to_csv(output_dir / "regression_coefficients.csv", index=False)

    print(f"Wrote summaries to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
