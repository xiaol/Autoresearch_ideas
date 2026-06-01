from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

from .isometry import compute_and_plot_isometry
from .rwkv7a_model import HiddenPatch, RWKV7AModel, resolve_model_path
from .spline import (
    PeriodicCubicSpline1D,
    TAU,
    bhattacharyya_from_hellinger,
    hellinger_distance,
    hellinger_sqrt,
    pad_other_bin,
    shortest_arc,
    weekday_thetas,
)
from .tasks import WEEKDAYS, WeekdayExample, make_weekday_examples
from .tokenizer import RWKVTokenizer
from .visualize_3d import (
    parse_component_indices,
    select_components,
    write_isometry_3d_html,
    write_steering_movement_3d_html,
    write_steering_movement_gif,
)


def weekday_token_ids(tokenizer: RWKVTokenizer) -> list[int]:
    ids: list[int] = []
    for weekday in WEEKDAYS:
        encoded = tokenizer.encode(" " + weekday)
        if len(encoded) != 1:
            raise ValueError(
                f"weekday {weekday!r} is not a single leading-space token: {encoded}"
            )
        ids.append(encoded[0])
    return ids


def collect_hidden_vectors(
    model: RWKV7AModel,
    tokenizer: RWKVTokenizer,
    examples: list[WeekdayExample],
) -> tuple[np.ndarray, np.ndarray]:
    all_hidden: list[list[np.ndarray]] = []
    all_probs: list[np.ndarray] = []
    token_ids = weekday_token_ids(tokenizer)

    for example in tqdm(examples, desc="collect"):
        output = model.forward(tokenizer.encode(example.prompt), collect_layers=True)
        if output.hidden_by_layer is None:
            raise RuntimeError("missing hidden collection")
        all_hidden.append([hidden.numpy() for hidden in output.hidden_by_layer])
        probs = torch.softmax(output.logits.float(), dim=-1)[token_ids]
        all_probs.append(probs.detach().cpu().numpy())

    return (
        np.asarray(all_hidden, dtype=np.float32),
        np.asarray(all_probs, dtype=np.float32),
    )


def nearest_centroid_accuracy(coords: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    centroids = np.stack([coords[labels == i].mean(axis=0) for i in range(len(WEEKDAYS))])
    distances = ((coords[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
    prediction = distances.argmin(axis=1)
    accuracy = float((prediction == labels).mean())

    within = np.mean(
        [
            np.linalg.norm(coords[row] - centroids[labels[row]]) ** 2
            for row in range(coords.shape[0])
        ]
    )
    center = coords.mean(axis=0)
    between = np.mean([np.linalg.norm(centroid - center) ** 2 for centroid in centroids])
    return accuracy, float(between / (within + 1e-8))


def choose_layer(
    hidden: np.ndarray,
    labels: np.ndarray,
    *,
    pca_dim: int,
) -> tuple[int, list[dict[str, float]]]:
    metrics: list[dict[str, float]] = []
    for layer in range(hidden.shape[1]):
        layer_hidden = hidden[:, layer, :]
        dim = min(pca_dim, layer_hidden.shape[0] - 1, layer_hidden.shape[1])
        coords = PCA(n_components=dim, random_state=0).fit_transform(layer_hidden)
        accuracy, ratio = nearest_centroid_accuracy(coords, labels)
        metrics.append(
            {
                "layer": float(layer),
                "nearest_centroid_accuracy": accuracy,
                "between_within_ratio": ratio,
            }
        )
    best = max(
        range(hidden.shape[1]),
        key=lambda layer: (
            metrics[layer]["nearest_centroid_accuracy"],
            metrics[layer]["between_within_ratio"],
        ),
    )
    return best, metrics


def fit_layer_geometry(
    hidden: np.ndarray,
    labels: np.ndarray,
    layer: int,
    *,
    pca_dim: int,
) -> tuple[PCA, np.ndarray, np.ndarray, np.ndarray, np.ndarray, PeriodicCubicSpline1D]:
    layer_hidden = hidden[:, layer, :]
    dim = min(pca_dim, layer_hidden.shape[0] - 1, layer_hidden.shape[1])
    pca = PCA(n_components=dim, random_state=0)
    coords = pca.fit_transform(layer_hidden).astype(np.float32)
    centroids = np.stack([coords[labels == i].mean(axis=0) for i in range(len(WEEKDAYS))])
    mean = coords.mean(axis=0)
    std = coords.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    standardized_centroids = ((centroids - mean) / std).astype(np.float32)
    manifold = PeriodicCubicSpline1D(
        weekday_thetas(),
        standardized_centroids,
        period=TAU,
    )
    return pca, coords, centroids, mean.astype(np.float32), std, manifold


def inverse_pca_hidden(
    pca: PCA,
    coords: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    standardized: bool,
) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    if standardized:
        coords = coords * std + mean
    return pca.inverse_transform(coords).astype(np.float32)


def build_geometric_hidden_path(
    pca: PCA,
    manifold: PeriodicCubicSpline1D,
    mean: np.ndarray,
    std: np.ndarray,
    start: int,
    end: int,
    samples: int,
    *,
    oversteer_frac: float = 0.0,
    oversteer_steps: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = weekday_thetas()
    intrinsic_path = shortest_arc(
        float(theta[start]),
        float(theta[end]),
        samples,
        oversteer_frac=oversteer_frac,
        oversteer_steps=oversteer_steps,
    )
    standardized_path = manifold.evaluate(intrinsic_path)
    hidden = inverse_pca_hidden(pca, standardized_path, mean, std, standardized=True)
    return hidden, intrinsic_path, standardized_path


def build_linear_hidden_path(
    start_hidden: np.ndarray,
    end_hidden: np.ndarray,
    samples: int,
    *,
    oversteer_frac: float = 0.0,
    oversteer_steps: int = 0,
) -> np.ndarray:
    normal = np.linspace(0.0, 1.0, samples, dtype=np.float32)
    if oversteer_frac > 0.0 and oversteer_steps > 0:
        step = oversteer_frac / oversteer_steps
        extra = np.linspace(1.0 + step, 1.0 + oversteer_frac, oversteer_steps)
        alphas = np.concatenate([normal, extra.astype(np.float32)])
    else:
        alphas = normal
    return np.asarray(
        [(1.0 - alpha) * start_hidden + alpha * end_hidden for alpha in alphas],
        dtype=np.float32,
    )


def project_activation_path_3d(
    hidden_path: np.ndarray,
    activation_vertices: np.ndarray,
    activation_mds,
) -> np.ndarray:
    distances = np.linalg.norm(
        hidden_path[:, None, :] - activation_vertices[None, :, :],
        axis=-1,
    )
    return activation_mds.project(distances).astype(np.float32)


def project_behavior_path_3d(
    path_distributions: np.ndarray,
    behavior_vertices: np.ndarray,
    behavior_mds,
) -> np.ndarray:
    mean_probs = path_distributions.mean(axis=1)
    padded = pad_other_bin(mean_probs)
    sqrt_probs = hellinger_sqrt(padded)
    distances = np.linalg.norm(
        sqrt_probs[:, None, :] - behavior_vertices[None, :, :],
        axis=-1,
    ) / np.sqrt(2.0)
    return behavior_mds.project(distances).astype(np.float32)


def selected_carriers(
    examples: list[WeekdayExample],
    *,
    n_prompts: int,
) -> list[WeekdayExample]:
    if n_prompts <= 0:
        raise ValueError("n_prompts must be positive")
    # Match Causalab path_steering's use of carrier prompts: fixed base prompts
    # reused for every path point, independent of the path endpoints.
    return examples[: min(n_prompts, len(examples))]


def patch_distributions(
    model: RWKV7AModel,
    tokenizer: RWKVTokenizer,
    hidden_path: np.ndarray,
    carriers: list[WeekdayExample],
    *,
    layer: int,
) -> np.ndarray:
    token_ids = weekday_token_ids(tokenizer)
    all_steps: list[np.ndarray] = []
    for hidden in tqdm(hidden_path, desc=f"patch layer {layer}", leave=False):
        patch = HiddenPatch(layer=layer, hidden=torch.from_numpy(hidden))
        per_prompt: list[np.ndarray] = []
        for carrier in carriers:
            output = model.forward(tokenizer.encode(carrier.prompt), patch=patch)
            probs = torch.softmax(output.logits.float(), dim=-1)[token_ids]
            per_prompt.append(probs.detach().cpu().numpy())
        all_steps.append(np.asarray(per_prompt, dtype=np.float32))
    return np.asarray(all_steps, dtype=np.float32)


def output_manifold_from_base_probs(
    base_probs: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, PeriodicCubicSpline1D]:
    padded = pad_other_bin(base_probs)
    sqrt_probs = hellinger_sqrt(padded)
    centroids = np.stack([sqrt_probs[labels == i].mean(axis=0) for i in range(len(WEEKDAYS))])
    centroids = centroids / np.clip(np.linalg.norm(centroids, axis=-1, keepdims=True), 1e-8, None)
    return centroids.astype(np.float32), PeriodicCubicSpline1D(
        weekday_thetas(),
        centroids.astype(np.float32),
        period=TAU,
    )


def _nearest_manifold_distance(
    sqrt_probs: np.ndarray,
    output_manifold: PeriodicCubicSpline1D,
    *,
    grid_size: int = 512,
) -> np.ndarray:
    grid = np.linspace(0.0, TAU, grid_size, endpoint=False, dtype=np.float32)
    refs = output_manifold.evaluate(grid)
    refs = refs / np.clip(np.linalg.norm(refs, axis=-1, keepdims=True), 1e-8, None)
    flat = sqrt_probs.reshape(-1, sqrt_probs.shape[-1])
    # This grid approximation mirrors the original metric's distance to the
    # continuous output manifold, while staying dependency-light in this repo.
    distances = []
    chunk = 2048
    for start in range(0, flat.shape[0], chunk):
        rows = flat[start : start + chunk]
        d = np.linalg.norm(rows[:, None, :] - refs[None, :, :], axis=-1) / np.sqrt(2.0)
        distances.append(d.min(axis=1))
    return np.concatenate(distances).reshape(sqrt_probs.shape[:-1])


def _matched_geodesic_distance(
    sqrt_probs: np.ndarray,
    output_manifold: PeriodicCubicSpline1D,
    intrinsic_path: np.ndarray,
) -> np.ndarray:
    refs = output_manifold.evaluate(intrinsic_path)
    refs = refs / np.clip(np.linalg.norm(refs, axis=-1, keepdims=True), 1e-8, None)
    return hellinger_distance(sqrt_probs, refs[:, None, :])


def path_metrics(
    probs: np.ndarray,
    output_manifold: PeriodicCubicSpline1D,
    intrinsic_path: np.ndarray | None = None,
) -> dict[str, float]:
    on_target = np.clip(probs.sum(axis=-1), 0.0, 1.0)
    padded = pad_other_bin(probs)
    sqrt_probs = hellinger_sqrt(padded)
    d_manifold = _nearest_manifold_distance(sqrt_probs, output_manifold)
    db_manifold = bhattacharyya_from_hellinger(d_manifold)

    metrics = {
        "coherence_mean": float(on_target.mean()),
        "coherence_worst": float(on_target.min(axis=0).mean()),
        "distance_from_behavior_manifold_mean": float(db_manifold.sum(axis=0).mean()),
    }
    if intrinsic_path is not None:
        d_geo = _matched_geodesic_distance(sqrt_probs, output_manifold, intrinsic_path)
        metrics["distance_from_geodesic_mean"] = float(
            bhattacharyya_from_hellinger(d_geo).sum(axis=0).mean()
        )
    return metrics


def plot_path_probs(
    out_path: Path,
    path_distributions: dict[str, np.ndarray],
    *,
    start: int,
    end: int,
    layer: int,
) -> None:
    fig, axes = plt.subplots(
        1,
        len(path_distributions),
        figsize=(7 * len(path_distributions), 5),
        sharey=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    for axis, (label, values) in zip(axes, path_distributions.items()):
        mean = values.mean(axis=1)
        std = values.std(axis=1)
        x_axis = np.linspace(0.0, 1.0, values.shape[0])
        for idx, weekday in enumerate(WEEKDAYS):
            axis.plot(x_axis, mean[:, idx], label=weekday)
            axis.fill_between(
                x_axis,
                np.clip(mean[:, idx] - std[:, idx], 0.0, 1.0),
                np.clip(mean[:, idx] + std[:, idx], 0.0, 1.0),
                alpha=0.12,
            )
        axis.set_title(label)
        axis.set_xlabel(f"{WEEKDAYS[start]} to {WEEKDAYS[end]}")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("next-token probability, mean over carriers")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(f"RWKV block-output path steering at layer {layer}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_pca_map(
    out_path: Path,
    coords: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    geometric_coords: np.ndarray,
    linear_coords: np.ndarray,
    *,
    layer: int,
) -> None:
    fig, axis = plt.subplots(figsize=(8, 6))
    for idx, weekday in enumerate(WEEKDAYS):
        rows = coords[labels == idx]
        axis.scatter(rows[:, 0], rows[:, 1], s=38, alpha=0.7, label=weekday)
        axis.scatter(centroids[idx, 0], centroids[idx, 1], s=130, marker="x", linewidths=2.5)
    closed = np.vstack([centroids, centroids[0]])
    axis.plot(closed[:, 0], closed[:, 1], color="black", alpha=0.3, label="centroids")
    axis.plot(geometric_coords[:, 0], geometric_coords[:, 1], color="black", linewidth=2.0, label="geometric")
    axis.plot(linear_coords[:, 0], linear_coords[:, 1], color="gray", linestyle="--", linewidth=2.0, label="linear")
    axis.set_title(f"RWKV layer {layer} weekday activation manifold")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.grid(alpha=0.25)
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_summary(
    out_dir: Path,
    *,
    examples: list[WeekdayExample],
    carriers: list[WeekdayExample],
    base_probs: np.ndarray,
    layer_metrics: list[dict[str, float]],
    best_layer: int,
    start: int,
    end: int,
    path_distributions: dict[str, np.ndarray],
    path_scores: dict[str, dict[str, float]],
    model_path: Path,
    pca_dim: int,
    selected_layer: str,
    num_steps: int,
    isometry: dict[str, float | int | str],
    viz: dict[str, str],
) -> None:
    labels = np.asarray([example.result_index for example in examples], dtype=np.int64)
    predictions = base_probs.argmax(axis=1)
    payload = {
        "model_path": str(model_path),
        "method": {
            "original_transformer_target": "residual_stream/block_output at last_token via PyVene",
            "rwkv_target": "x[-1] after the selected RWKV block's time-mix and channel-mix residual updates",
            "activation_manifold": "PCA -> standardize -> periodic cubic spline over weekday angle theta",
            "path_modes": {
                "geometric": "shortest arc in intrinsic weekday theta, decoded through spline",
                "linear": "straight line between raw hidden centroids",
            },
        },
        "pca_dim": pca_dim,
        "selected_layer_mode": selected_layer,
        "best_layer": best_layer,
        "num_steps": num_steps,
        "n_carrier_prompts": len(carriers),
        "base_weekday_accuracy": float((predictions == labels).mean()),
        "layer_metrics": layer_metrics,
        "isometry": isometry,
        "visualization": viz,
        "path": {
            "start": WEEKDAYS[start],
            "end": WEEKDAYS[end],
            "scores": path_scores,
            "argmax_by_mode": {
                label: [WEEKDAYS[int(i)] for i in probs.mean(axis=1).argmax(axis=1)]
                for label, probs in path_distributions.items()
            },
        },
        "carrier_prompts": [asdict(example) for example in carriers],
        "examples": [
            {
                **asdict(example),
                "base_prediction": WEEKDAYS[int(predictions[i])],
                "base_weekday_probs": {
                    weekday: float(base_probs[i, j]) for j, weekday in enumerate(WEEKDAYS)
                },
            }
            for i, example in enumerate(examples)
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Path to RWKV-7 .pth file.")
    parser.add_argument("--out-dir", default="outputs/weekday_manifold")
    parser.add_argument("--pca-dim", type=int, default=16)
    parser.add_argument("--num-steps", "--samples", dest="num_steps", type=int, default=50)
    parser.add_argument("--n-prompts", type=int, default=16)
    parser.add_argument("--start", default="Monday", choices=WEEKDAYS)
    parser.add_argument("--end", default="Thursday", choices=WEEKDAYS)
    parser.add_argument(
        "--selected-layer",
        default="best",
        help="Layer index to patch, or 'best' to select by centroid separability.",
    )
    parser.add_argument("--oversteer-frac", type=float, default=0.0)
    parser.add_argument("--oversteer-steps", type=int, default=10)
    parser.add_argument("--isometry-n-interior-per-pair", type=int, default=2)
    parser.add_argument("--isometry-n-arc-steps", type=int, default=150)
    parser.add_argument("--activation-3d-components", default="0,1,2")
    parser.add_argument("--behavior-3d-components", default="0,1,2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose-kernel", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = RWKVTokenizer()
    model_path = resolve_model_path(args.model)
    print(f"Loading RWKV model: {model_path}")
    model = RWKV7AModel(model_path, device=args.device, verbose_kernel=args.verbose_kernel)

    examples = make_weekday_examples()
    carriers = selected_carriers(examples, n_prompts=args.n_prompts)
    labels = np.asarray([example.result_index for example in examples], dtype=np.int64)
    hidden, base_probs = collect_hidden_vectors(model, tokenizer, examples)
    hidden_centroids = np.stack(
        [hidden[labels == i].mean(axis=0) for i in range(len(WEEKDAYS))]
    )  # (weekday, layer, embd)

    best_layer_auto, layer_metrics = choose_layer(hidden, labels, pca_dim=args.pca_dim)
    if args.selected_layer == "best":
        best_layer = best_layer_auto
    else:
        best_layer = int(args.selected_layer)

    pca, coords, centroids, pca_mean, pca_std, activation_manifold = fit_layer_geometry(
        hidden,
        labels,
        best_layer,
        pca_dim=args.pca_dim,
    )
    output_centroids, output_manifold = output_manifold_from_base_probs(base_probs, labels)

    start = WEEKDAYS.index(args.start)
    end = WEEKDAYS.index(args.end)
    oversteer_steps = args.oversteer_steps if args.oversteer_frac > 0 else 0
    geometric_hidden, geometric_intrinsic, geometric_std_coords = build_geometric_hidden_path(
        pca,
        activation_manifold,
        pca_mean,
        pca_std,
        start,
        end,
        args.num_steps,
        oversteer_frac=args.oversteer_frac,
        oversteer_steps=oversteer_steps,
    )
    linear_hidden = build_linear_hidden_path(
        hidden_centroids[start, best_layer],
        hidden_centroids[end, best_layer],
        args.num_steps,
        oversteer_frac=args.oversteer_frac,
        oversteer_steps=oversteer_steps,
    )

    path_hidden = {
        "geometric": geometric_hidden,
        "linear": linear_hidden,
    }
    path_distributions = {
        label: patch_distributions(model, tokenizer, path, carriers, layer=best_layer)
        for label, path in path_hidden.items()
    }

    theta = weekday_thetas()
    linear_intrinsic = shortest_arc(
        float(theta[start]),
        float(theta[end]),
        args.num_steps,
        oversteer_frac=args.oversteer_frac,
        oversteer_steps=oversteer_steps,
    )
    path_scores = {
        "geometric": path_metrics(
            path_distributions["geometric"],
            output_manifold,
            intrinsic_path=geometric_intrinsic,
        ),
        "linear": path_metrics(
            path_distributions["linear"],
            output_manifold,
            intrinsic_path=linear_intrinsic,
        ),
    }

    def activation_decode_fn(theta_values: np.ndarray) -> np.ndarray:
        standardized = activation_manifold.evaluate(np.asarray(theta_values, dtype=np.float32))
        coords = standardized * pca_std + pca_mean
        return pca.inverse_transform(coords).astype(np.float32)

    def output_decode_fn(theta_values: np.ndarray) -> np.ndarray:
        probs = output_manifold.evaluate(np.asarray(theta_values, dtype=np.float32))
        return probs / np.clip(np.linalg.norm(probs, axis=-1, keepdims=True), 1e-8, None)

    isometry_result = compute_and_plot_isometry(
        activation_decode_fn=activation_decode_fn,
        output_decode_fn=output_decode_fn,
        n_interior_per_pair=args.isometry_n_interior_per_pair,
        n_arc_steps=args.isometry_n_arc_steps,
        out_dir=out_dir,
    )

    geometric_pca_coords = geometric_std_coords * pca_std + pca_mean
    linear_pca_coords = pca.transform(linear_hidden).astype(np.float32)
    activation_components = parse_component_indices(args.activation_3d_components)
    behavior_components = parse_component_indices(args.behavior_3d_components)
    activation_mds_3d = select_components(
        isometry_result.activation_mds_3d.coords,
        activation_components,
    )
    behavior_mds_3d = select_components(
        isometry_result.behavior_mds_3d.coords,
        behavior_components,
    )
    activation_path_geometric_3d = select_components(
        project_activation_path_3d(
            geometric_hidden,
            isometry_result.activation_vertices,
            isometry_result.activation_mds_3d,
        ),
        activation_components,
    )
    activation_path_linear_3d = select_components(
        project_activation_path_3d(
            linear_hidden,
            isometry_result.activation_vertices,
            isometry_result.activation_mds_3d,
        ),
        activation_components,
    )
    behavior_path_geometric_3d = select_components(
        project_behavior_path_3d(
            path_distributions["geometric"],
            isometry_result.behavior_vertices,
            isometry_result.behavior_mds_3d,
        ),
        behavior_components,
    )
    behavior_path_linear_3d = select_components(
        project_behavior_path_3d(
            path_distributions["linear"],
            isometry_result.behavior_vertices,
            isometry_result.behavior_mds_3d,
        ),
        behavior_components,
    )
    np.savez_compressed(
        out_dir / "artifacts.npz",
        hidden=hidden,
        base_probs=base_probs,
        pca_coords=coords,
        pca_centroids=centroids,
        pca_mean=pca_mean,
        pca_std=pca_std,
        activation_control_thetas=weekday_thetas(),
        activation_control_points=activation_manifold.y.astype(np.float32),
        output_control_points=output_centroids,
        geometric_hidden=geometric_hidden,
        linear_hidden=linear_hidden,
        geometric_pca_coords=geometric_pca_coords,
        linear_pca_coords=linear_pca_coords,
        activation_mds_3d=activation_mds_3d,
        behavior_mds_3d=behavior_mds_3d,
        activation_path_mds_3d_geometric=activation_path_geometric_3d,
        activation_path_mds_3d_linear=activation_path_linear_3d,
        behavior_path_mds_3d_geometric=behavior_path_geometric_3d,
        behavior_path_mds_3d_linear=behavior_path_linear_3d,
        geometric_probs=path_distributions["geometric"],
        linear_probs=path_distributions["linear"],
        labels=labels,
        isometry_report_path=str(isometry_result.report_path),
    )

    isometry_3d_path = write_isometry_3d_html(
        out_dir / "isometry_3d.html",
        theta=isometry_result.theta,
        activation_mds=activation_mds_3d,
        behavior_mds=behavior_mds_3d,
        pearson_r_geometric=isometry_result.pearson_r_geometric,
        pearson_r_linear=isometry_result.pearson_r_linear,
    )
    steering_3d_path = write_steering_movement_3d_html(
        out_dir / "steering_3d.html",
        theta=theta,
        activation_vertices_3d=activation_mds_3d,
        behavior_vertices_3d=behavior_mds_3d,
        activation_paths={
            "geometric": activation_path_geometric_3d,
            "linear": activation_path_linear_3d,
        },
        behavior_paths={
            "geometric": behavior_path_geometric_3d,
            "linear": behavior_path_linear_3d,
        },
        start=start,
        end=end,
        layer=best_layer,
        start_name=args.start,
        end_name=args.end,
    )
    steering_gif_path = write_steering_movement_gif(
        out_dir / "steering_movement.gif",
        activation_paths={
            "geometric": activation_path_geometric_3d,
            "linear": activation_path_linear_3d,
        },
        behavior_paths={
            "geometric": behavior_path_geometric_3d,
            "linear": behavior_path_linear_3d,
        },
        start=start,
        end=end,
        start_name=args.start,
        end_name=args.end,
    )
    plot_pca_map(
        out_dir / "activation_paths.png",
        coords,
        labels,
        centroids,
        geometric_pca_coords,
        linear_pca_coords,
        layer=best_layer,
    )
    plot_path_probs(
        out_dir / "path_probabilities.png",
        path_distributions,
        start=start,
        end=end,
        layer=best_layer,
    )
    write_summary(
        out_dir,
        examples=examples,
        carriers=carriers,
        base_probs=base_probs,
        layer_metrics=layer_metrics,
        best_layer=best_layer,
        start=start,
        end=end,
        path_distributions=path_distributions,
        path_scores=path_scores,
        model_path=model_path,
        pca_dim=args.pca_dim,
        selected_layer=args.selected_layer,
        num_steps=args.num_steps,
        isometry={
            "pearson_r_geometric": isometry_result.pearson_r_geometric,
            "pearson_r_linear": isometry_result.pearson_r_linear,
            "n_pairs": isometry_result.n_pairs,
            "n_vertices": isometry_result.n_vertices,
            "n_excluded_same_geodesic": isometry_result.n_excluded_same_geodesic,
            "report_path": str(isometry_result.report_path),
        },
        viz={
            "isometry_3d_html": str(isometry_3d_path),
            "steering_3d_html": str(steering_3d_path),
            "steering_gif": str(steering_gif_path),
            "activation_3d_components": ",".join(map(str, activation_components)),
            "behavior_3d_components": ",".join(map(str, behavior_components)),
        },
    )

    base_acc = float((base_probs.argmax(axis=1) == labels).mean())
    print(f"Selected layer: {best_layer}")
    print(f"Base weekday-token accuracy over 49 prompts: {base_acc:.3f}")
    for label, scores in path_scores.items():
        print(
            f"{label}: coherence={scores['coherence_mean']:.6f}, "
            f"worst={scores['coherence_worst']:.6f}, "
            f"behavior_distance={scores['distance_from_behavior_manifold_mean']:.6f}"
        )
    print(f"Wrote {out_dir / 'summary.json'}")
    print(f"Wrote {out_dir / 'activation_paths.png'}")
    print(f"Wrote {out_dir / 'path_probabilities.png'}")
    print(f"Wrote {isometry_result.report_path}")
    print(f"Wrote {isometry_3d_path}")
    print(f"Wrote {steering_3d_path}")
    print(f"Wrote {steering_gif_path}")


if __name__ == "__main__":
    main()
