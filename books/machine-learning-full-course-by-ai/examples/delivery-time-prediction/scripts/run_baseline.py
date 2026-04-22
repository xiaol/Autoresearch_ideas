#!/usr/bin/env python3

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "delivery_time_data.csv"


def load_rows(path: Path) -> list[dict[str, float]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def mae(y_true: list[float], y_pred: list[float]) -> float:
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def mean_baseline(train_rows: list[dict[str, float]], test_rows: list[dict[str, float]]) -> list[float]:
    avg = mean(r["delivery_minutes"] for r in train_rows)
    return [avg for _ in test_rows]


def linear_regression_fit(train_rows: list[dict[str, float]]) -> tuple[list[float], float]:
    feature_names = ["distance_km", "prep_minutes", "courier_load", "weather_score", "is_rush_hour"]
    weights = [0.0 for _ in feature_names]
    bias = mean(r["delivery_minutes"] for r in train_rows)
    learning_rate = 0.001

    for _ in range(10000):
        grad_w = [0.0 for _ in feature_names]
        grad_b = 0.0
        for row in train_rows:
            x = [row[name] for name in feature_names]
            y = row["delivery_minutes"]
            pred = sum(w * xi for w, xi in zip(weights, x)) + bias
            err = pred - y
            for i, xi in enumerate(x):
                grad_w[i] += err * xi
            grad_b += err
        scale = 2.0 / len(train_rows)
        for i in range(len(weights)):
            weights[i] -= learning_rate * scale * grad_w[i]
        bias -= learning_rate * scale * grad_b
    return weights, bias


def linear_regression_predict(rows: list[dict[str, float]], weights: list[float], bias: float) -> list[float]:
    feature_names = ["distance_km", "prep_minutes", "courier_load", "weather_score", "is_rush_hour"]
    preds = []
    for row in rows:
        x = [row[name] for name in feature_names]
        preds.append(sum(w * xi for w, xi in zip(weights, x)) + bias)
    return preds


def main() -> None:
    rows = load_rows(DATA_PATH)
    split_idx = int(len(rows) * 0.7)
    train_rows = rows[:split_idx]
    test_rows = rows[split_idx:]
    y_test = [r["delivery_minutes"] for r in test_rows]

    mean_preds = mean_baseline(train_rows, test_rows)
    weights, bias = linear_regression_fit(train_rows)
    linear_preds = linear_regression_predict(test_rows, weights, bias)

    print("Delivery-Time Prediction Example")
    print(f"Train rows: {len(train_rows)}")
    print(f"Test rows: {len(test_rows)}")
    print()
    print(f"Naive mean baseline MAE: {mae(y_test, mean_preds):.2f}")
    print(f"Linear regression MAE: {mae(y_test, linear_preds):.2f}")
    print()
    print("Linear weights:")
    for name, weight in zip(
        ["distance_km", "prep_minutes", "courier_load", "weather_score", "is_rush_hour"],
        weights,
    ):
        print(f"  {name}: {weight:.3f}")
    print(f"  bias: {bias:.3f}")


if __name__ == "__main__":
    main()
