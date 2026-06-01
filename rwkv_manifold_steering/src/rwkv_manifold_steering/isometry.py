from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

from .spline import TAU, shortest_arc, weekday_thetas
from .tasks import WEEKDAYS


@dataclass(frozen=True)
class IsometryResult:
    theta: np.ndarray
    pearson_r_geometric: float
    pearson_r_linear: float
    n_pairs: int
    n_vertices: int
    n_excluded_same_geodesic: int
    report_path: Path
    activation_vertices: np.ndarray
    behavior_vertices: np.ndarray
    activation_mds_3d: "MDSEmbedding"
    behavior_mds_3d: "MDSEmbedding"


@dataclass(frozen=True)
class MDSEmbedding:
    coords: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    row_means: np.ndarray
    grand_mean: float

    def project(self, distances: np.ndarray) -> np.ndarray:
        return project_mds_embedding(self, distances)


def build_vertex_set(
    n_interior_per_pair: int,
    *,
    control_thetas: np.ndarray | None = None,
) -> tuple[np.ndarray, list[frozenset[int]]]:
    theta = (
        weekday_thetas().astype(np.float32)
        if control_thetas is None
        else np.asarray(control_thetas, dtype=np.float32)
    )
    vertices = [np.float32(x) for x in theta]
    supports = [frozenset({i}) for i in range(len(theta))]
    fractions = (
        [(k + 1) / (n_interior_per_pair + 1) for k in range(n_interior_per_pair)]
        if n_interior_per_pair > 0
        else []
    )
    for i in range(len(theta)):
        for j in range(i + 1, len(theta)):
            arc = shortest_arc(float(theta[i]), float(theta[j]), 2)
            delta = np.float32(arc[1] - arc[0])
            for frac in fractions:
                vertices.append(np.float32(theta[i] + frac * delta))
                supports.append(frozenset({i, j}))
    return np.asarray(vertices, dtype=np.float32), supports


def _shortest_arc_delta(u_a: np.ndarray, u_b: np.ndarray) -> np.ndarray:
    delta = np.asarray(u_b - u_a, dtype=np.float64)
    delta = ((delta + TAU / 2.0) % TAU) - TAU / 2.0
    return delta.astype(np.float32)


def _pairwise_chord_distance(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)


def _correlation_vectors(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size == 0 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), int(x.size)
    return float(np.corrcoef(x, y)[0, 1]), int(x.size)


def classical_mds_fit(distance_matrix: np.ndarray, n_components: int = 2) -> MDSEmbedding:
    d2 = np.asarray(distance_matrix, dtype=np.float64) ** 2
    n = d2.shape[0]
    j = np.eye(n) - np.ones((n, n), dtype=np.float64) / n
    b = -0.5 * j @ d2 @ j
    eigvals, eigvecs = np.linalg.eigh(b)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order][:n_components]
    eigvecs = eigvecs[:, order][:, :n_components]
    coords = (eigvecs * np.sqrt(np.clip(eigvals, 0.0, None))).astype(np.float32)
    return MDSEmbedding(
        coords=coords,
        eigenvalues=eigvals.astype(np.float64),
        eigenvectors=eigvecs.astype(np.float64),
        row_means=d2.mean(axis=1).astype(np.float64),
        grand_mean=float(d2.mean()),
    )


def classical_mds(distance_matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    return classical_mds_fit(distance_matrix, n_components=n_components).coords


def project_mds_embedding(embedding: MDSEmbedding, distances: np.ndarray) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim == 1:
        distances = distances[None, :]
    if distances.shape[1] != embedding.row_means.shape[0]:
        raise ValueError(
            f"expected {embedding.row_means.shape[0]} distances, got {distances.shape[1]}"
        )
    d2 = distances**2
    centered = -0.5 * (
        d2
        - d2.mean(axis=1, keepdims=True)
        - embedding.row_means[None, :]
        + embedding.grand_mean
    )
    denom = np.sqrt(np.clip(embedding.eigenvalues, 1e-12, None))
    coords = centered @ embedding.eigenvectors / denom
    return coords.astype(np.float32)


def _fit_slope(x: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.dot(x, x))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(x, y) / denom)


def _theta_colors(theta: np.ndarray) -> np.ndarray:
    cmap = plt.get_cmap("twilight_shifted")
    norm = mcolors.Normalize(vmin=0.0, vmax=TAU)
    return cmap(norm(np.mod(theta, TAU)))


def _path_length_from_theta(theta_a: np.ndarray, theta_b: np.ndarray, decode_fn, n_steps: int) -> np.ndarray:
    if theta_a.shape != theta_b.shape:
        raise ValueError("theta_a and theta_b must have matching shapes")
    out = np.empty(theta_a.shape[0], dtype=np.float64)
    t = np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float32)
    for i, (a, b) in enumerate(zip(theta_a, theta_b)):
        delta = _shortest_arc_delta(np.array(a, dtype=np.float32), np.array(b, dtype=np.float32))
        path = np.asarray(a + t * delta, dtype=np.float32)
        decoded = decode_fn(path.astype(np.float32))
        diffs = decoded[1:] - decoded[:-1]
        out[i] = np.linalg.norm(diffs, axis=-1).sum()
    return out


def compute_and_plot_isometry(
    *,
    activation_decode_fn,
    output_decode_fn,
    n_interior_per_pair: int,
    n_arc_steps: int,
    out_dir: Path,
    control_thetas: np.ndarray | None = None,
    labels: list[str] | None = None,
    title: str = "RWKV weekday activation vs behavior manifold isometry",
) -> IsometryResult:
    theta, supports = build_vertex_set(
        n_interior_per_pair,
        control_thetas=control_thetas,
    )
    n_vertices = len(theta)
    labels = list(labels or WEEKDAYS)
    centroid_count = len(labels)

    act_vertices = activation_decode_fn(theta)
    out_vertices = output_decode_fn(theta)
    out_vertices = out_vertices / np.clip(
        np.linalg.norm(out_vertices, axis=-1, keepdims=True), 1e-8, None
    )

    same_geo = np.zeros((n_vertices, n_vertices), dtype=bool)
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            if supports[i] <= supports[j] or supports[j] <= supports[i]:
                same_geo[i, j] = same_geo[j, i] = True

    pair_a, pair_b = np.triu_indices(n_vertices, k=1)
    keep = ~same_geo[pair_a, pair_b]
    n_excluded = int((~keep).sum())

    dx_geo = np.zeros((n_vertices, n_vertices), dtype=np.float64)
    dy = np.zeros((n_vertices, n_vertices), dtype=np.float64)
    if pair_a.size > 0:
        geo_vals = _path_length_from_theta(theta[pair_a], theta[pair_b], activation_decode_fn, n_arc_steps)
        out_vals = _path_length_from_theta(theta[pair_a], theta[pair_b], output_decode_fn, n_arc_steps) / math.sqrt(2.0)
        dx_geo[pair_a, pair_b] = geo_vals
        dx_geo[pair_b, pair_a] = geo_vals
        dy[pair_a, pair_b] = out_vals
        dy[pair_b, pair_a] = out_vals

    dx_lin = _pairwise_chord_distance(act_vertices)
    flat_geo = dx_geo[pair_a, pair_b][keep]
    flat_lin = dx_lin[pair_a, pair_b][keep]
    flat_out = dy[pair_a, pair_b][keep]
    pearson_geo, n_pairs = _correlation_vectors(flat_geo, flat_out)
    pearson_lin, _ = _correlation_vectors(flat_lin, flat_out)
    if n_pairs == 0:
        raise ValueError("isometry needs at least one non-excluded pair")

    act_mds = classical_mds(dx_geo, 2)
    out_mds = classical_mds(dy, 2)
    act_mds_3d = classical_mds_fit(dx_geo, 3)
    out_mds_3d = classical_mds_fit(dy, 3)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])

    def scatter_panel(ax, x, y, title):
        slope = _fit_slope(x, y)
        ax.scatter(x, y, s=8, alpha=0.25, c="steelblue", edgecolors="none")
        if np.isfinite(slope):
            xr = np.linspace(0.0, float(np.max(x)) * 1.05, 100)
            ax.plot(xr, slope * xr, color="crimson", linewidth=1.5)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_title(title)
        ax.set_xlabel("Activation manifold path length")
        ax.set_ylabel("Behavior manifold path length (Hellinger)")

    ax1 = fig.add_subplot(gs[0, 0])
    scatter_panel(ax1, flat_geo, flat_out, f"Geometric path mode, r={pearson_geo:.3f}")
    ax1.text(0.04, 0.96, f"n_pairs={n_pairs}\nexcluded={n_excluded}", transform=ax1.transAxes, va="top", fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    ax2 = fig.add_subplot(gs[0, 1])
    scatter_panel(ax2, flat_lin, flat_out, f"Linear control, r={pearson_lin:.3f}")
    ax2.text(0.04, 0.96, f"n_pairs={n_pairs}\nexcluded={n_excluded}", transform=ax2.transAxes, va="top", fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    colors = _theta_colors(theta)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(act_mds[centroid_count:, 0], act_mds[centroid_count:, 1], s=28, c=colors[centroid_count:], alpha=0.72, edgecolors="none")
    ax3.scatter(act_mds[:centroid_count, 0], act_mds[:centroid_count, 1], s=85, c=colors[:centroid_count], marker="D", edgecolors="black", linewidths=0.8)
    for i, label in enumerate(labels):
        ax3.annotate(label, (act_mds[i, 0], act_mds[i, 1]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax3.set_title("Activation manifold MDS")
    ax3.set_xlabel("MDS-1")
    ax3.set_ylabel("MDS-2")
    ax3.grid(alpha=0.2)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(out_mds[centroid_count:, 0], out_mds[centroid_count:, 1], s=28, c=colors[centroid_count:], alpha=0.72, edgecolors="none")
    ax4.scatter(out_mds[:centroid_count, 0], out_mds[:centroid_count, 1], s=85, c=colors[:centroid_count], marker="D", edgecolors="black", linewidths=0.8)
    for i, label in enumerate(labels):
        ax4.annotate(label, (out_mds[i, 0], out_mds[i, 1]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax4.set_title("Behavior manifold MDS")
    ax4.set_xlabel("MDS-1")
    ax4.set_ylabel("MDS-2")
    ax4.grid(alpha=0.2)

    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    report_path = out_dir / "isometry_report.png"
    fig.savefig(report_path, dpi=170)
    plt.close(fig)

    np.savez_compressed(
        out_dir / "isometry_artifacts.npz",
        theta=theta,
        dx_geo=dx_geo,
        dx_lin=dx_lin,
        dy=dy,
        act_vertices=act_vertices,
        out_vertices=out_vertices,
        act_mds=act_mds,
        out_mds=out_mds,
        act_mds_3d_coords=act_mds_3d.coords,
        out_mds_3d_coords=out_mds_3d.coords,
    )

    return IsometryResult(
        theta=theta,
        pearson_r_geometric=float(pearson_geo),
        pearson_r_linear=float(pearson_lin),
        n_pairs=int(n_pairs),
        n_vertices=int(n_vertices),
        n_excluded_same_geodesic=n_excluded,
        report_path=report_path,
        activation_vertices=act_vertices,
        behavior_vertices=out_vertices,
        activation_mds_3d=act_mds_3d,
        behavior_mds_3d=out_mds_3d,
    )
