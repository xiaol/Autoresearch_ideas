from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from tqdm import tqdm

from .hf_causal_model import DEFAULT_QWEN_MODEL, HiddenPatch, HFCausalTransformer
from .isometry import compute_and_plot_isometry
from .spline import hellinger_sqrt, pad_other_bin, shortest_arc, weekday_thetas
from .tasks import WEEKDAYS, WeekdayExample, make_weekday_examples
from .visualize_3d import (
    parse_component_indices,
    select_components,
    write_isometry_3d_html,
    write_steering_movement_3d_html,
    write_steering_movement_gif,
)
from .weekday_manifold import (
    build_geometric_hidden_path,
    build_linear_hidden_path,
    choose_layer,
    fit_layer_geometry,
    output_manifold_from_base_probs,
    path_metrics,
    plot_path_probs,
    plot_pca_map,
    project_activation_path_3d,
    project_behavior_path_3d,
    selected_carriers,
)


def weekday_token_ids(model: HFCausalTransformer) -> list[int]:
    ids: list[int] = []
    for weekday in WEEKDAYS:
        encoded = model.encode(" " + weekday)
        if len(encoded) != 1:
            raise ValueError(
                f"weekday {weekday!r} is not a single leading-space token for this tokenizer: {encoded}"
            )
        ids.append(encoded[0])
    return ids


def collect_hidden_vectors(
    model: HFCausalTransformer,
    examples: list[WeekdayExample],
) -> tuple[np.ndarray, np.ndarray]:
    all_hidden: list[list[np.ndarray]] = []
    all_probs: list[np.ndarray] = []
    token_ids = weekday_token_ids(model)
    for example in tqdm(examples, desc="collect"):
        output = model.forward(model.encode(example.prompt), collect_layers=True)
        if output.hidden_by_layer is None:
            raise RuntimeError("missing hidden collection")
        all_hidden.append([hidden.numpy() for hidden in output.hidden_by_layer])
        probs = torch.softmax(output.logits.float(), dim=-1)[token_ids]
        all_probs.append(probs.detach().cpu().numpy())
    return np.asarray(all_hidden, dtype=np.float32), np.asarray(all_probs, dtype=np.float32)


def patch_distributions(
    model: HFCausalTransformer,
    hidden_path: np.ndarray,
    carriers: list[WeekdayExample],
    *,
    layer: int,
) -> np.ndarray:
    token_ids = weekday_token_ids(model)
    all_steps: list[np.ndarray] = []
    for hidden in tqdm(hidden_path, desc=f"patch layer {layer}", leave=False):
        patch = HiddenPatch(layer=layer, hidden=torch.from_numpy(hidden))
        per_prompt: list[np.ndarray] = []
        for carrier in carriers:
            output = model.forward(model.encode(carrier.prompt), patch=patch)
            probs = torch.softmax(output.logits.float(), dim=-1)[token_ids]
            per_prompt.append(probs.detach().cpu().numpy())
        all_steps.append(np.asarray(per_prompt, dtype=np.float32))
    return np.asarray(all_steps, dtype=np.float32)


def _behavior_path_3d(
    probs: np.ndarray,
    behavior_vertices: np.ndarray,
    behavior_mds,
    components: tuple[int, int, int],
) -> np.ndarray:
    distances = np.linalg.norm(
        hellinger_sqrt(pad_other_bin(probs.mean(axis=1)))[:, None, :]
        - behavior_vertices[None, :, :],
        axis=-1,
    ) / np.sqrt(2.0)
    return select_components(behavior_mds.project(distances), components)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--out-dir", default="outputs/qwen_weekday_manifold")
    parser.add_argument("--pca-dim", type=int, default=16)
    parser.add_argument("--num-steps", "--samples", dest="num_steps", type=int, default=50)
    parser.add_argument("--n-prompts", type=int, default=16)
    parser.add_argument("--start", default="Monday", choices=WEEKDAYS)
    parser.add_argument("--end", default="Thursday", choices=WEEKDAYS)
    parser.add_argument("--selected-layer", default="best")
    parser.add_argument("--oversteer-frac", type=float, default=0.0)
    parser.add_argument("--oversteer-steps", type=int, default=10)
    parser.add_argument("--isometry-n-interior-per-pair", type=int, default=2)
    parser.add_argument("--isometry-n-arc-steps", type=int, default=150)
    parser.add_argument("--activation-3d-components", default="0,1,2")
    parser.add_argument("--behavior-3d-components", default="0,1,2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading HF model: {args.model}")
    model = HFCausalTransformer(args.model, device=args.device, dtype=args.dtype)
    examples = make_weekday_examples()
    carriers = selected_carriers(examples, n_prompts=args.n_prompts)
    labels = np.asarray([example.result_index for example in examples], dtype=np.int64)
    hidden, base_probs = collect_hidden_vectors(model, examples)
    hidden_centroids = np.stack(
        [hidden[labels == i].mean(axis=0) for i in range(len(WEEKDAYS))]
    )

    best_layer_auto, layer_metrics = choose_layer(hidden, labels, pca_dim=args.pca_dim)
    best_layer = best_layer_auto if args.selected_layer == "best" else int(args.selected_layer)
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
    path_distributions = {
        "geometric": patch_distributions(model, geometric_hidden, carriers, layer=best_layer),
        "linear": patch_distributions(model, linear_hidden, carriers, layer=best_layer),
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
        return pca.inverse_transform(standardized * pca_std + pca_mean).astype(np.float32)

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
    behavior_path_geometric_3d = _behavior_path_3d(
        path_distributions["geometric"],
        isometry_result.behavior_vertices,
        isometry_result.behavior_mds_3d,
        behavior_components,
    )
    behavior_path_linear_3d = _behavior_path_3d(
        path_distributions["linear"],
        isometry_result.behavior_vertices,
        isometry_result.behavior_mds_3d,
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
        activation_paths={"geometric": activation_path_geometric_3d, "linear": activation_path_linear_3d},
        behavior_paths={"geometric": behavior_path_geometric_3d, "linear": behavior_path_linear_3d},
        start=start,
        end=end,
        layer=best_layer,
        start_name=args.start,
        end_name=args.end,
    )
    steering_gif_path = write_steering_movement_gif(
        out_dir / "steering_movement.gif",
        activation_paths={"geometric": activation_path_geometric_3d, "linear": activation_path_linear_3d},
        behavior_paths={"geometric": behavior_path_geometric_3d, "linear": behavior_path_linear_3d},
        start=start,
        end=end,
        start_name=args.start,
        end_name=args.end,
    )
    plot_pca_map(out_dir / "activation_paths.png", coords, labels, centroids, geometric_pca_coords, linear_pca_coords, layer=best_layer)
    plot_path_probs(out_dir / "path_probabilities.png", path_distributions, start=start, end=end, layer=best_layer)
    summary = {
        "model": args.model,
        "target": "decoder block output / residual stream after selected transformer block",
        "best_layer": best_layer,
        "base_weekday_accuracy": float((base_probs.argmax(axis=1) == labels).mean()),
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
        },
        "path": {"start": args.start, "end": args.end, "scores": path_scores},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Selected layer: {best_layer}")
    print(f"Base weekday-token accuracy over 49 prompts: {summary['base_weekday_accuracy']:.3f}")
    print(f"Wrote {out_dir / 'summary.json'}")
    print(f"Wrote {isometry_3d_path}")
    print(f"Wrote {steering_3d_path}")
    print(f"Wrote {steering_gif_path}")


if __name__ == "__main__":
    main()
