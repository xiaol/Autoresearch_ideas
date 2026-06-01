from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.decomposition import PCA

from .spline import PeriodicCubicSpline1D, TAU, hellinger_sqrt, pad_other_bin, weekday_thetas
from .tasks import WEEKDAYS


def _load_artifacts(path: Path) -> dict[str, np.ndarray]:
    if path.is_dir():
        path = path / "artifacts.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _control_thetas(artifacts: dict[str, np.ndarray], control_points: np.ndarray) -> np.ndarray:
    if "activation_control_thetas" in artifacts:
        return artifacts["activation_control_thetas"].astype(np.float32)
    if control_points.shape[0] == len(WEEKDAYS):
        return weekday_thetas()
    return np.linspace(0.0, TAU, control_points.shape[0], endpoint=False, dtype=np.float32)


def _class_names(artifacts: dict[str, np.ndarray], count: int) -> list[str]:
    if "class_names" in artifacts:
        return [str(item) for item in artifacts["class_names"].tolist()]
    if count == len(WEEKDAYS):
        return WEEKDAYS
    return [f"class {i}" for i in range(count)]


def _behavior_curve(
    control_thetas: np.ndarray,
    control_points: np.ndarray,
    samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, TAU, samples, endpoint=False, dtype=np.float32)
    manifold = PeriodicCubicSpline1D(control_thetas, control_points.astype(np.float32), period=TAU)
    curve = manifold.evaluate(theta)
    curve = curve / np.clip(np.linalg.norm(curve, axis=-1, keepdims=True), 1e-8, None)
    return theta, curve.astype(np.float32)


def _behavior_path(probs: np.ndarray) -> np.ndarray:
    mean_probs = probs.mean(axis=1)
    return hellinger_sqrt(pad_other_bin(mean_probs)).astype(np.float32)


def _project_combined(*arrays: np.ndarray) -> list[np.ndarray]:
    lengths = [array.shape[0] for array in arrays]
    stacked = np.vstack(arrays)
    projected = PCA(n_components=3, random_state=0).fit_transform(stacked).astype(np.float32)
    out: list[np.ndarray] = []
    start = 0
    for length in lengths:
        out.append(projected[start : start + length])
        start += length
    return out


def _hellinger_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=-1) / np.sqrt(2.0)


def write_behavior_comparison(
    out_dir: Path,
    *,
    rwkv_artifacts: Path,
    qwen_artifacts: Path,
    rwkv_label: str = "RWKV",
    qwen_label: str = "Qwen",
    samples: int = 256,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rwkv = _load_artifacts(rwkv_artifacts)
    qwen = _load_artifacts(qwen_artifacts)
    label_count = int(rwkv["output_control_points"].shape[0])
    if int(qwen["output_control_points"].shape[0]) != label_count:
        raise ValueError("model artifacts have different label counts")
    labels = _class_names(rwkv, label_count)
    rwkv_theta = _control_thetas(rwkv, rwkv["output_control_points"])
    qwen_theta = _control_thetas(qwen, qwen["output_control_points"])

    _, rwkv_curve = _behavior_curve(rwkv_theta, rwkv["output_control_points"], samples)
    _, qwen_curve = _behavior_curve(qwen_theta, qwen["output_control_points"], samples)
    rwkv_geo = _behavior_path(rwkv["geometric_probs"])
    rwkv_lin = _behavior_path(rwkv["linear_probs"])
    qwen_geo = _behavior_path(qwen["geometric_probs"])
    qwen_lin = _behavior_path(qwen["linear_probs"])

    projected = _project_combined(
        rwkv_curve,
        qwen_curve,
        rwkv["output_control_points"],
        qwen["output_control_points"],
        rwkv_geo,
        rwkv_lin,
        qwen_geo,
        qwen_lin,
    )
    (
        rwkv_curve_3d,
        qwen_curve_3d,
        rwkv_centroids_3d,
        qwen_centroids_3d,
        rwkv_geo_3d,
        rwkv_lin_3d,
        qwen_geo_3d,
        qwen_lin_3d,
    ) = projected

    color_cycle = [
        "#5b5f97",
        "#f45b69",
        "#2e86ab",
        "#f6ae2d",
        "#33658a",
        "#55a630",
        "#8f2d56",
        "#6a4c93",
        "#1982c4",
        "#8ac926",
        "#ffca3a",
        "#ff595e",
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=rwkv_curve_3d[:, 0],
            y=rwkv_curve_3d[:, 1],
            z=rwkv_curve_3d[:, 2],
            mode="lines",
            line=dict(color="#111111", width=5),
            name=f"{rwkv_label} behavior manifold",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=qwen_curve_3d[:, 0],
            y=qwen_curve_3d[:, 1],
            z=qwen_curve_3d[:, 2],
            mode="lines",
            line=dict(color="#2e86ab", width=5),
            name=f"{qwen_label} behavior manifold",
        )
    )
    for idx, label in enumerate(labels):
        color = color_cycle[idx % len(color_cycle)]
        fig.add_trace(
            go.Scatter3d(
                x=[rwkv_centroids_3d[idx, 0], qwen_centroids_3d[idx, 0]],
                y=[rwkv_centroids_3d[idx, 1], qwen_centroids_3d[idx, 1]],
                z=[rwkv_centroids_3d[idx, 2], qwen_centroids_3d[idx, 2]],
                mode="markers+lines+text",
                marker=dict(size=7, color=color),
                line=dict(color=color, width=2, dash="dot"),
                text=[f"{rwkv_label} {label}", f"{qwen_label} {label}"],
                textposition="top center",
                name=label,
                showlegend=False,
            )
        )
    for name, path, color, dash in [
        (f"{rwkv_label} geometric path", rwkv_geo_3d, "#111111", "dash"),
        (f"{rwkv_label} linear path", rwkv_lin_3d, "#777777", "dash"),
        (f"{qwen_label} geometric path", qwen_geo_3d, "#2e86ab", "dash"),
        (f"{qwen_label} linear path", qwen_lin_3d, "#82b7d1", "dash"),
    ]:
        fig.add_trace(
            go.Scatter3d(
                x=path[:, 0],
                y=path[:, 1],
                z=path[:, 2],
                mode="lines",
                line=dict(color=color, width=4, dash=dash),
                name=name,
            )
        )
    fig.update_layout(
        title=f"{rwkv_label} vs {qwen_label} behavior manifolds in shared Hellinger space",
        margin=dict(l=0, r=0, t=45, b=0),
        scene=dict(aspectmode="data"),
    )
    html_path = out_dir / "behavior_space_compare.html"
    html_path.write_text(
        pio.to_html(fig, full_html=True, include_plotlyjs="cdn", config=dict(responsive=True, displaylogo=False)),
        encoding="utf-8",
    )

    centroid_distances = _hellinger_distance(
        rwkv["output_control_points"],
        qwen["output_control_points"],
    )
    curve_distances = _hellinger_distance(rwkv_curve, qwen_curve)
    summary = {
        "rwkv_artifacts": str(rwkv_artifacts),
        "qwen_artifacts": str(qwen_artifacts),
        "rwkv_label": rwkv_label,
        "qwen_label": qwen_label,
        "sample_count": samples,
        "mean_weekday_centroid_hellinger_distance": float(centroid_distances.mean()),
        "max_weekday_centroid_hellinger_distance": float(centroid_distances.max()),
        "label_count": label_count,
        "centroid_hellinger_distance_by_label": {
            label: float(distance)
            for label, distance in zip(labels, centroid_distances, strict=True)
        },
        "mean_curve_hellinger_distance": float(curve_distances.mean()),
        "max_curve_hellinger_distance": float(curve_distances.max()),
    }
    summary_path = out_dir / "behavior_space_compare.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return html_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rwkv", default="outputs/weekday_manifold")
    parser.add_argument("--qwen", default="outputs/qwen_weekday_manifold")
    parser.add_argument("--out-dir", default="outputs/model_compare")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--rwkv-label", default="RWKV")
    parser.add_argument("--qwen-label", default="Qwen")
    args = parser.parse_args()
    html_path, summary_path = write_behavior_comparison(
        Path(args.out_dir),
        rwkv_artifacts=Path(args.rwkv),
        qwen_artifacts=Path(args.qwen),
        rwkv_label=args.rwkv_label,
        qwen_label=args.qwen_label,
        samples=args.samples,
    )
    print(f"Wrote {html_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
