from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

from .hf_causal_model import DEFAULT_QWEN_MODEL, HFCausalTransformer
from .hf_causal_model import HiddenPatch as HFHiddenPatch
from .isometry import compute_and_plot_isometry
from .rwkv7a_model import HiddenPatch as RWKVHiddenPatch
from .rwkv7a_model import RWKV7AModel, resolve_model_path
from .spline import (
    PeriodicCubicSpline1D,
    TAU,
    bhattacharyya_from_hellinger,
    hellinger_distance,
    hellinger_sqrt,
    pad_other_bin,
    shortest_arc,
)
from .tasks import CyclicTask, WeekdayExample, get_cyclic_task
from .tokenizer import RWKVTokenizer
from .visualize_3d import (
    parse_component_indices,
    select_components,
    write_isometry_3d_html,
    write_steering_movement_3d_html,
    write_steering_movement_gif,
)
from .weekday_manifold import (
    build_linear_hidden_path,
    project_activation_path_3d,
    project_behavior_path_3d,
    selected_carriers,
)


class CyclicModel(Protocol):
    n_embd: int

    def encode(self, text: str) -> list[int]: ...

    def forward(self, tokens, *, collect_layers: bool = False, patch=None): ...


def cyclic_thetas(labels: list[str]) -> np.ndarray:
    return np.linspace(0.0, TAU, len(labels), endpoint=False, dtype=np.float32)


def nearest_centroid_accuracy(
    coords: np.ndarray,
    example_labels: np.ndarray,
    *,
    label_count: int,
) -> tuple[float, float]:
    centroids = np.stack([coords[example_labels == i].mean(axis=0) for i in range(label_count)])
    distances = ((coords[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
    prediction = distances.argmin(axis=1)
    accuracy = float((prediction == example_labels).mean())
    within = np.mean(
        [
            np.linalg.norm(coords[row] - centroids[example_labels[row]]) ** 2
            for row in range(coords.shape[0])
        ]
    )
    center = coords.mean(axis=0)
    between = np.mean([np.linalg.norm(centroid - center) ** 2 for centroid in centroids])
    return accuracy, float(between / (within + 1e-8))


def choose_layer_for_task(
    hidden: np.ndarray,
    example_labels: np.ndarray,
    *,
    label_count: int,
    pca_dim: int,
) -> tuple[int, list[dict[str, float]]]:
    metrics: list[dict[str, float]] = []
    for layer in range(hidden.shape[1]):
        layer_hidden = hidden[:, layer, :]
        dim = min(pca_dim, layer_hidden.shape[0] - 1, layer_hidden.shape[1])
        coords = PCA(n_components=dim, random_state=0).fit_transform(layer_hidden)
        accuracy, ratio = nearest_centroid_accuracy(
            coords,
            example_labels,
            label_count=label_count,
        )
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


def label_token_ids(model: CyclicModel, labels: list[str]) -> list[int]:
    ids: list[int] = []
    for label in labels:
        encoded = model.encode(" " + label)
        if len(encoded) != 1:
            raise ValueError(
                f"label {label!r} is not a single leading-space token: {encoded}"
            )
        ids.append(encoded[0])
    return ids


def collect_hidden_vectors(
    model: CyclicModel,
    examples: list[WeekdayExample],
    labels: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    all_hidden: list[list[np.ndarray]] = []
    all_probs: list[np.ndarray] = []
    token_ids = label_token_ids(model, labels)
    for example in tqdm(examples, desc="collect"):
        output = model.forward(model.encode(example.prompt), collect_layers=True)
        if output.hidden_by_layer is None:
            raise RuntimeError("missing hidden collection")
        all_hidden.append([hidden.numpy() for hidden in output.hidden_by_layer])
        probs = torch.softmax(output.logits.float(), dim=-1)[token_ids]
        all_probs.append(probs.detach().cpu().numpy())
    return np.asarray(all_hidden, dtype=np.float32), np.asarray(all_probs, dtype=np.float32)


def fit_layer_geometry(
    hidden: np.ndarray,
    example_labels: np.ndarray,
    layer: int,
    *,
    label_count: int,
    pca_dim: int,
) -> tuple[PCA, np.ndarray, np.ndarray, np.ndarray, np.ndarray, PeriodicCubicSpline1D]:
    layer_hidden = hidden[:, layer, :]
    dim = min(pca_dim, layer_hidden.shape[0] - 1, layer_hidden.shape[1])
    pca = PCA(n_components=dim, random_state=0)
    coords = pca.fit_transform(layer_hidden).astype(np.float32)
    centroids = np.stack([coords[example_labels == i].mean(axis=0) for i in range(label_count)])
    mean = coords.mean(axis=0)
    std = coords.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    standardized_centroids = ((centroids - mean) / std).astype(np.float32)
    manifold = PeriodicCubicSpline1D(
        np.linspace(0.0, TAU, label_count, endpoint=False, dtype=np.float32),
        standardized_centroids,
        period=TAU,
    )
    return pca, coords, centroids, mean.astype(np.float32), std, manifold


def output_manifold_from_base_probs(
    base_probs: np.ndarray,
    example_labels: np.ndarray,
    *,
    label_count: int,
) -> tuple[np.ndarray, PeriodicCubicSpline1D]:
    padded = pad_other_bin(base_probs)
    sqrt_probs = hellinger_sqrt(padded)
    centroids = np.stack([sqrt_probs[example_labels == i].mean(axis=0) for i in range(label_count)])
    centroids = centroids / np.clip(np.linalg.norm(centroids, axis=-1, keepdims=True), 1e-8, None)
    return centroids.astype(np.float32), PeriodicCubicSpline1D(
        np.linspace(0.0, TAU, label_count, endpoint=False, dtype=np.float32),
        centroids.astype(np.float32),
        period=TAU,
    )


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
    control_thetas: np.ndarray,
    start: int,
    end: int,
    samples: int,
    *,
    oversteer_frac: float = 0.0,
    oversteer_steps: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    intrinsic_path = shortest_arc(
        float(control_thetas[start]),
        float(control_thetas[end]),
        samples,
        oversteer_frac=oversteer_frac,
        oversteer_steps=oversteer_steps,
    )
    standardized_path = manifold.evaluate(intrinsic_path)
    hidden = inverse_pca_hidden(pca, standardized_path, mean, std, standardized=True)
    return hidden, intrinsic_path, standardized_path


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


def endpoint_diagnostics(
    *,
    geometric_hidden: np.ndarray,
    linear_hidden: np.ndarray,
    geometric_probs: np.ndarray,
    linear_probs: np.ndarray,
) -> dict[str, float]:
    """Measure endpoint agreement between the two steering paths."""
    out: dict[str, float] = {}
    for name, idx in [("start", 0), ("end", -1)]:
        hidden_delta = geometric_hidden[idx] - linear_hidden[idx]
        geo_mean = geometric_probs[idx].mean(axis=0)
        lin_mean = linear_probs[idx].mean(axis=0)
        prob_delta = geo_mean - lin_mean
        out[f"{name}_hidden_l2"] = float(np.linalg.norm(hidden_delta))
        out[f"{name}_behavior_prob_l1"] = float(np.abs(prob_delta).sum())
        out[f"{name}_behavior_prob_l2"] = float(np.linalg.norm(prob_delta))
        out[f"{name}_geometric_concept_mass"] = float(geo_mean.sum())
        out[f"{name}_linear_concept_mass"] = float(lin_mean.sum())
    return out


def patch_distributions(
    model: CyclicModel,
    hidden_path: np.ndarray,
    carriers: list[WeekdayExample],
    *,
    labels: list[str],
    layer: int,
    backend: str,
) -> np.ndarray:
    token_ids = label_token_ids(model, labels)
    all_steps: list[np.ndarray] = []
    patch_cls = RWKVHiddenPatch if backend == "rwkv" else HFHiddenPatch
    for hidden in tqdm(hidden_path, desc=f"patch layer {layer}", leave=False):
        patch = patch_cls(layer=layer, hidden=torch.from_numpy(hidden))
        per_prompt: list[np.ndarray] = []
        for carrier in carriers:
            output = model.forward(model.encode(carrier.prompt), patch=patch)
            probs = torch.softmax(output.logits.float(), dim=-1)[token_ids]
            per_prompt.append(probs.detach().cpu().numpy())
        all_steps.append(np.asarray(per_prompt, dtype=np.float32))
    return np.asarray(all_steps, dtype=np.float32)


def plot_path_probs(
    out_path: Path,
    path_distributions: dict[str, np.ndarray],
    *,
    labels: list[str],
    task_name: str,
    start: int,
    end: int,
    layer: int,
    backend: str,
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
        for idx, item in enumerate(labels):
            axis.plot(x_axis, mean[:, idx], label=item)
            axis.fill_between(
                x_axis,
                np.clip(mean[:, idx] - std[:, idx], 0.0, 1.0),
                np.clip(mean[:, idx] + std[:, idx], 0.0, 1.0),
                alpha=0.12,
            )
        axis.set_title(label)
        axis.set_xlabel(f"{labels[start]} to {labels[end]}")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("next-token probability, mean over carriers")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(f"{backend.upper()} {task_name} path steering at layer {layer}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_pca_map(
    out_path: Path,
    coords: np.ndarray,
    example_labels: np.ndarray,
    centroids: np.ndarray,
    geometric_coords: np.ndarray,
    linear_coords: np.ndarray,
    *,
    labels: list[str],
    task_name: str,
    layer: int,
    backend: str,
) -> None:
    fig, axis = plt.subplots(figsize=(8, 6))
    for idx, item in enumerate(labels):
        rows = coords[example_labels == idx]
        axis.scatter(rows[:, 0], rows[:, 1], s=38, alpha=0.7, label=item)
        axis.scatter(centroids[idx, 0], centroids[idx, 1], s=130, marker="x", linewidths=2.5)
    closed = np.vstack([centroids, centroids[0]])
    axis.plot(closed[:, 0], closed[:, 1], color="black", alpha=0.3, label="centroids")
    axis.plot(geometric_coords[:, 0], geometric_coords[:, 1], color="black", linewidth=2.0, label="geometric")
    axis.plot(linear_coords[:, 0], linear_coords[:, 1], color="gray", linestyle="--", linewidth=2.0, label="linear")
    axis.set_title(f"{backend.upper()} layer {layer} {task_name} activation manifold")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.grid(alpha=0.25)
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_experiment(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    task = get_cyclic_task(args.task)
    labels = task.labels
    theta = cyclic_thetas(labels)

    if args.backend == "rwkv":
        tokenizer = RWKVTokenizer()
        model_path = resolve_model_path(args.model)
        print(f"Loading RWKV model: {model_path}")
        model: CyclicModel = RWKV7AModel(model_path, device=args.device, verbose_kernel=args.verbose_kernel)
        model_ref = str(model_path)
        encode = tokenizer.encode
        model.encode = encode  # type: ignore[method-assign]
    else:
        model_name = args.model or DEFAULT_QWEN_MODEL
        print(f"Loading HF model: {model_name}")
        model = HFCausalTransformer(model_name, device=args.device, dtype=args.dtype)
        model_ref = model_name

    examples = task.examples
    carriers = selected_carriers(examples, n_prompts=args.n_prompts)
    example_labels = np.asarray([example.result_index for example in examples], dtype=np.int64)
    hidden, base_probs = collect_hidden_vectors(model, examples, labels)
    hidden_centroids = np.stack(
        [hidden[example_labels == i].mean(axis=0) for i in range(len(labels))]
    )

    best_layer_auto, layer_metrics = choose_layer_for_task(
        hidden,
        example_labels,
        label_count=len(labels),
        pca_dim=args.pca_dim,
    )
    best_layer = best_layer_auto if args.selected_layer == "best" else int(args.selected_layer)
    pca, coords, centroids, pca_mean, pca_std, activation_manifold = fit_layer_geometry(
        hidden,
        example_labels,
        best_layer,
        label_count=len(labels),
        pca_dim=args.pca_dim,
    )
    output_centroids, output_manifold = output_manifold_from_base_probs(
        base_probs,
        example_labels,
        label_count=len(labels),
    )

    start = labels.index(args.start)
    end = labels.index(args.end)
    oversteer_steps = args.oversteer_steps if args.oversteer_frac > 0 else 0
    geometric_hidden, geometric_intrinsic, geometric_std_coords = build_geometric_hidden_path(
        pca,
        activation_manifold,
        pca_mean,
        pca_std,
        theta,
        start,
        end,
        args.num_steps,
        oversteer_frac=args.oversteer_frac,
        oversteer_steps=oversteer_steps,
    )
    linear_start = hidden_centroids[start, best_layer]
    linear_end = hidden_centroids[end, best_layer]
    if args.linear_endpoint_mode == "matched":
        linear_start = geometric_hidden[0]
        linear_end = geometric_hidden[args.num_steps - 1]
    linear_hidden = build_linear_hidden_path(
        linear_start,
        linear_end,
        args.num_steps,
        oversteer_frac=args.oversteer_frac,
        oversteer_steps=oversteer_steps,
    )
    path_distributions = {
        "geometric": patch_distributions(
            model,
            geometric_hidden,
            carriers,
            labels=labels,
            layer=best_layer,
            backend=args.backend,
        ),
        "linear": patch_distributions(
            model,
            linear_hidden,
            carriers,
            labels=labels,
            layer=best_layer,
            backend=args.backend,
        ),
    }
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
    endpoint_report = endpoint_diagnostics(
        geometric_hidden=geometric_hidden,
        linear_hidden=linear_hidden,
        geometric_probs=path_distributions["geometric"],
        linear_probs=path_distributions["linear"],
    )

    def activation_decode_fn(theta_values: np.ndarray) -> np.ndarray:
        standardized = activation_manifold.evaluate(np.asarray(theta_values, dtype=np.float32))
        coords_decoded = standardized * pca_std + pca_mean
        return pca.inverse_transform(coords_decoded).astype(np.float32)

    def output_decode_fn(theta_values: np.ndarray) -> np.ndarray:
        probs = output_manifold.evaluate(np.asarray(theta_values, dtype=np.float32))
        return probs / np.clip(np.linalg.norm(probs, axis=-1, keepdims=True), 1e-8, None)

    isometry_result = compute_and_plot_isometry(
        activation_decode_fn=activation_decode_fn,
        output_decode_fn=output_decode_fn,
        n_interior_per_pair=args.isometry_n_interior_per_pair,
        n_arc_steps=args.isometry_n_arc_steps,
        out_dir=out_dir,
        control_thetas=theta,
        labels=labels,
        title=f"{args.backend.upper()} {task.name} activation vs behavior manifold isometry",
    )

    geometric_pca_coords = geometric_std_coords * pca_std + pca_mean
    linear_pca_coords = pca.transform(linear_hidden).astype(np.float32)
    activation_components = parse_component_indices(args.activation_3d_components)
    behavior_components = parse_component_indices(args.behavior_3d_components)
    activation_mds_3d = select_components(isometry_result.activation_mds_3d.coords, activation_components)
    behavior_mds_3d = select_components(isometry_result.behavior_mds_3d.coords, behavior_components)
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
        activation_control_thetas=theta,
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
        labels=example_labels,
        class_names=np.asarray(labels, dtype=str),
        isometry_report_path=str(isometry_result.report_path),
    )

    page_title = f"{args.backend.upper()} {task.name} manifold"
    isometry_3d_path = write_isometry_3d_html(
        out_dir / "isometry_3d.html",
        theta=isometry_result.theta,
        activation_mds=activation_mds_3d,
        behavior_mds=behavior_mds_3d,
        pearson_r_geometric=isometry_result.pearson_r_geometric,
        pearson_r_linear=isometry_result.pearson_r_linear,
        labels=labels,
        page_title=f"{page_title} isometry",
        concept_name=task.name,
    )
    steering_3d_path = write_steering_movement_3d_html(
        out_dir / "steering_3d.html",
        theta=theta,
        activation_vertices_3d=activation_mds_3d,
        behavior_vertices_3d=behavior_mds_3d,
        activation_paths={"geometric": activation_path_geometric_3d, "linear": activation_path_linear_3d},
        behavior_paths={"geometric": behavior_path_geometric_3d, "linear": behavior_path_linear_3d},
        start=start,
        end=end,
        layer=best_layer,
        start_name=args.start,
        end_name=args.end,
        labels=labels,
        page_title=f"{page_title} steering",
        concept_name=task.name,
    )
    steering_gif_path = write_steering_movement_gif(
        out_dir / "steering_movement.gif",
        activation_paths={"geometric": activation_path_geometric_3d, "linear": activation_path_linear_3d},
        behavior_paths={"geometric": behavior_path_geometric_3d, "linear": behavior_path_linear_3d},
        start=start,
        end=end,
        start_name=args.start,
        end_name=args.end,
        labels=labels,
        page_title=f"{page_title} steering",
    )
    plot_pca_map(
        out_dir / "activation_paths.png",
        coords,
        example_labels,
        centroids,
        geometric_pca_coords,
        linear_pca_coords,
        labels=labels,
        task_name=task.name,
        layer=best_layer,
        backend=args.backend,
    )
    plot_path_probs(
        out_dir / "path_probabilities.png",
        path_distributions,
        labels=labels,
        task_name=task.name,
        start=start,
        end=end,
        layer=best_layer,
        backend=args.backend,
    )

    predictions = base_probs.argmax(axis=1)
    summary = {
        "backend": args.backend,
        "model": model_ref,
        "task": task.name,
        "labels": labels,
        "target": (
            "x[-1] after selected RWKV block"
            if args.backend == "rwkv"
            else "decoder block output / residual stream after selected transformer block"
        ),
        "best_layer": best_layer,
        "pca_dim": args.pca_dim,
        "num_steps": args.num_steps,
        "n_carrier_prompts": len(carriers),
        "linear_endpoint_mode": args.linear_endpoint_mode,
        "base_accuracy": float((predictions == example_labels).mean()),
        "layer_metrics": layer_metrics,
        "isometry": {
            "pearson_r_geometric": isometry_result.pearson_r_geometric,
            "pearson_r_linear": isometry_result.pearson_r_linear,
            "n_pairs": isometry_result.n_pairs,
            "n_vertices": isometry_result.n_vertices,
            "n_excluded_same_geodesic": isometry_result.n_excluded_same_geodesic,
            "report_path": str(isometry_result.report_path),
        },
        "visualization": {
            "isometry_3d_html": str(isometry_3d_path),
            "steering_3d_html": str(steering_3d_path),
            "steering_gif": str(steering_gif_path),
            "activation_3d_components": ",".join(map(str, activation_components)),
            "behavior_3d_components": ",".join(map(str, behavior_components)),
        },
        "path": {
            "start": args.start,
            "end": args.end,
            "scores": path_scores,
            "endpoint_diagnostics": endpoint_report,
            "argmax_by_mode": {
                label: [labels[int(i)] for i in probs.mean(axis=1).argmax(axis=1)]
                for label, probs in path_distributions.items()
            },
        },
        "carrier_prompts": [asdict(example) for example in carriers],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Selected layer: {best_layer}")
    print(f"Base {task.name} token accuracy over {len(examples)} prompts: {summary['base_accuracy']:.3f}")
    for label, scores in path_scores.items():
        print(
            f"{label}: coherence={scores['coherence_mean']:.6f}, "
            f"worst={scores['coherence_worst']:.6f}, "
            f"behavior_distance={scores['distance_from_behavior_manifold_mean']:.6f}"
        )
    print(
        "endpoint deltas: "
        f"hidden_start={endpoint_report['start_hidden_l2']:.6g}, "
        f"hidden_end={endpoint_report['end_hidden_l2']:.6g}, "
        f"behavior_start_l1={endpoint_report['start_behavior_prob_l1']:.6g}, "
        f"behavior_end_l1={endpoint_report['end_behavior_prob_l1']:.6g}"
    )
    print(f"Wrote {out_dir / 'summary.json'}")
    print(f"Wrote {isometry_result.report_path}")
    print(f"Wrote {isometry_3d_path}")
    print(f"Wrote {steering_3d_path}")
    print(f"Wrote {steering_gif_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["rwkv", "qwen"], required=True)
    parser.add_argument("--task", choices=["weekday", "month"], default="weekday")
    parser.add_argument("--model", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pca-dim", type=int, default=16)
    parser.add_argument("--num-steps", "--samples", dest="num_steps", type=int, default=40)
    parser.add_argument("--n-prompts", type=int, default=12)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--selected-layer", default="best")
    parser.add_argument(
        "--linear-endpoint-mode",
        choices=["matched", "raw"],
        default="matched",
        help=(
            "matched uses the same decoded start/end activations as manifold "
            "steering; raw uses raw hidden centroids."
        ),
    )
    parser.add_argument("--oversteer-frac", type=float, default=0.0)
    parser.add_argument("--oversteer-steps", type=int, default=10)
    parser.add_argument("--isometry-n-interior-per-pair", type=int, default=2)
    parser.add_argument("--isometry-n-arc-steps", type=int, default=150)
    parser.add_argument("--activation-3d-components", default="0,1,2")
    parser.add_argument("--behavior-3d-components", default="0,1,2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--verbose-kernel", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
