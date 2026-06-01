from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


RUNS = [
    ("weekday", "RWKV-7 0.1B", "outputs/report_weekday_rwkv_matched"),
    ("weekday", "Qwen3.5 0.8B", "outputs/report_weekday_qwen_matched"),
    ("month", "RWKV-7 0.1B", "outputs/report_month_rwkv_matched"),
    ("month", "Qwen3.5 0.8B", "outputs/report_month_qwen_matched"),
]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def _metric_rows(repo: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task, model, run_dir_str in RUNS:
        run_dir = repo / run_dir_str
        summary = _read_json(run_dir / "summary.json")
        scores = summary["path"]["scores"]
        endpoints = summary["path"]["endpoint_diagnostics"]
        rows.append(
            {
                "task": task,
                "model": model,
                "layer": int(summary["best_layer"]),
                "base_accuracy": float(summary["base_accuracy"]),
                "isometry_geometric_r": float(summary["isometry"]["pearson_r_geometric"]),
                "isometry_linear_r": float(summary["isometry"]["pearson_r_linear"]),
                "geometric_behavior_distance": float(
                    scores["geometric"]["distance_from_behavior_manifold_mean"]
                ),
                "linear_behavior_distance": float(
                    scores["linear"]["distance_from_behavior_manifold_mean"]
                ),
                "geometric_geodesic_distance": float(
                    scores["geometric"]["distance_from_geodesic_mean"]
                ),
                "linear_geodesic_distance": float(
                    scores["linear"]["distance_from_geodesic_mean"]
                ),
                "endpoint_hidden_l2_start": float(endpoints["start_hidden_l2"]),
                "endpoint_hidden_l2_end": float(endpoints["end_hidden_l2"]),
                "endpoint_behavior_l1_start": float(
                    endpoints["start_behavior_prob_l1"]
                ),
                "endpoint_behavior_l1_end": float(endpoints["end_behavior_prob_l1"]),
                "run_dir": run_dir_str,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns = list(rows[0].keys())
    lines = [",".join(columns)]
    for row in rows:
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.10g}")
            else:
                text = str(value).replace('"', '""')
                if "," in text:
                    text = f'"{text}"'
                values.append(text)
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(rows: list[dict[str, object]]) -> str:
    headers = [
        "task",
        "model",
        "layer",
        "acc",
        "r geom",
        "r linear",
        "dist geom",
        "dist linear",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["task"]),
                    str(row["model"]),
                    str(row["layer"]),
                    f"{row['base_accuracy']:.3f}",
                    f"{row['isometry_geometric_r']:.3f}",
                    f"{row['isometry_linear_r']:.3f}",
                    f"{row['geometric_behavior_distance']:.4f}",
                    f"{row['linear_behavior_distance']:.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _html_table(rows: list[dict[str, object]]) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{row['task']}</td>"
            f"<td>{row['model']}</td>"
            f"<td>{row['layer']}</td>"
            f"<td>{row['base_accuracy']:.3f}</td>"
            f"<td>{row['isometry_geometric_r']:.3f}</td>"
            f"<td>{row['isometry_linear_r']:.3f}</td>"
            f"<td>{row['geometric_behavior_distance']:.4f}</td>"
            f"<td>{row['linear_behavior_distance']:.4f}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Task</th><th>Model</th><th>Layer</th><th>Base acc</th>"
        "<th>r geom</th><th>r linear</th><th>Dist geom</th><th>Dist linear</th>"
        "</tr></thead><tbody>"
        + "\n".join(body)
        + "</tbody></table>"
    )


def _copy_visuals(repo: Path, out_dir: Path) -> dict[str, dict[str, str]]:
    visual_root = out_dir / "visuals"
    visual_root.mkdir(parents=True, exist_ok=True)
    copied: dict[str, dict[str, str]] = {}
    for task, model, run_dir_str in RUNS:
        run_dir = repo / run_dir_str
        key = f"{task}_{model.lower().replace(' ', '_').replace('.', '_')}"
        dst_dir = visual_root / key
        dst_dir.mkdir(parents=True, exist_ok=True)
        copied[key] = {}
        for name in [
            "activation_paths.png",
            "path_probabilities.png",
            "isometry_report.png",
            "isometry_3d.html",
            "steering_3d.html",
            "steering_movement.gif",
            "summary.json",
        ]:
            src = run_dir / name
            if src.exists():
                dst = dst_dir / name
                shutil.copy2(src, dst)
                copied[key][name] = _rel(dst, out_dir)
    for compare_name in ["report_compare_weekday_matched", "report_compare_month_matched"]:
        src_dir = repo / "outputs" / compare_name
        if not src_dir.exists():
            continue
        dst_dir = visual_root / compare_name
        dst_dir.mkdir(exist_ok=True)
        copied[compare_name] = {}
        for name in ["behavior_space_compare.html", "behavior_space_compare.json"]:
            src = src_dir / name
            if src.exists():
                dst = dst_dir / name
                shutil.copy2(src, dst)
                copied[compare_name][name] = _rel(dst, out_dir)
    return copied


def _short_model(model: str) -> str:
    return "RWKV" if "RWKV" in model else "Qwen"


def write_markdown_report(repo: Path, out_dir: Path, rows: list[dict[str, object]], visuals: dict[str, dict[str, str]]) -> Path:
    table = _markdown_table(rows)
    lines = [
        "# RWKV Manifold Steering Report",
        "",
        "This report summarizes the matched-endpoint reproduction of Goodfire-style manifold steering on RWKV-7 0.1B and Qwen3.5 0.8B. The matched runs force linear and manifold steering to share the same start and end hidden states, so endpoint differences in behavior space are removed.",
        "",
        "## Main Results",
        "",
        table,
        "",
        "The tiny models are near chance on the raw next-token arithmetic tasks. The useful signal here is geometric: activation-manifold distances remain strongly correlated with behavior-manifold distances, especially for RWKV months and RWKV weekdays.",
        "",
        "## Interpretation",
        "",
        "- `r geom` is the correlation between activation-manifold path length and behavior-manifold path length.",
        "- `r linear` is the same comparison when activation distances are measured by straight Euclidean paths.",
        "- `Dist geom` and `Dist linear` are mean Bhattacharyya distances from the induced output trajectory to the fitted behavior manifold. Lower is more natural.",
        "- All matched report runs have zero start/end hidden and behavior deltas between linear and manifold paths.",
        "",
        "## Visual Artifacts",
        "",
    ]
    for row in rows:
        key = f"{row['task']}_{str(row['model']).lower().replace(' ', '_').replace('.', '_')}"
        item = visuals.get(key, {})
        title = f"{row['task']} / {row['model']}"
        lines.extend(
            [
                f"### {title}",
                "",
                f"- [2D activation/path probabilities]({item.get('activation_paths.png', '')})",
                f"- [Isometry report image]({item.get('isometry_report.png', '')})",
                f"- [Interactive 3D isometry]({item.get('isometry_3d.html', '')})",
                f"- [Interactive 3D steering]({item.get('steering_3d.html', '')})",
                f"- [Steering movement GIF]({item.get('steering_movement.gif', '')})",
                "",
            ]
        )
    lines.extend(
        [
            "## Animation",
            "",
            "- [Manim scene source](animation/manifold_report_scene.py)",
            "- [Rendered Manim MP4](animation/manim_matched_endpoint_report.mp4)",
            "- [Manim contact sheet](animation/manim_contact_sheet.jpg)",
            "- [Data-driven fallback MP4](animation/matched_endpoint_report.mp4)",
            "- [Fallback contact sheet](animation/matched_endpoint_contact_sheet.jpg)",
            "- [Audience explainer video](audience_video/neural_geometry_rwkv_insight.mp4)",
            "- [Audience explainer contact sheet](audience_video/neural_geometry_rwkv_insight_contact_sheet.jpg)",
            "",
            "## Cross-Model Behavior Space",
            "",
            f"- [Weekday RWKV vs Qwen behavior-space comparison]({visuals['report_compare_weekday_matched']['behavior_space_compare.html']})",
            f"- [Month RWKV vs Qwen behavior-space comparison]({visuals['report_compare_month_matched']['behavior_space_compare.html']})",
            "",
            "## Caveats",
            "",
            "These are small models. Base task accuracy is chance-level, so this should be treated as a geometry and intervention sanity check, not as evidence that the models solve weekday/month arithmetic. The RWKV intervention target is the last-token block output after the selected RWKV block's residual updates, used as the closest analogue to transformer residual-stream block output patching.",
        ]
    )
    path = out_dir / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_html_report(repo: Path, out_dir: Path, rows: list[dict[str, object]], visuals: dict[str, dict[str, str]]) -> Path:
    cards = []
    for row in rows:
        key = f"{row['task']}_{str(row['model']).lower().replace(' ', '_').replace('.', '_')}"
        item = visuals.get(key, {})
        cards.append(
            f"""
            <section class="run">
              <h3>{row['task']} / {row['model']}</h3>
              <div class="grid">
                <a href="{item.get('isometry_report.png', '#')}"><img src="{item.get('isometry_report.png', '')}" alt="isometry report"></a>
                <a href="{item.get('activation_paths.png', '#')}"><img src="{item.get('activation_paths.png', '')}" alt="activation paths"></a>
                <a href="{item.get('path_probabilities.png', '#')}"><img src="{item.get('path_probabilities.png', '')}" alt="path probabilities"></a>
                <a class="gif" href="{item.get('steering_movement.gif', '#')}"><img src="{item.get('steering_movement.gif', '')}" alt="steering movement"></a>
              </div>
              <p>
                <a href="{item.get('isometry_3d.html', '#')}">3D isometry</a>
                <a href="{item.get('steering_3d.html', '#')}">3D steering</a>
                <a href="{item.get('summary.json', '#')}">summary JSON</a>
              </p>
            </section>
            """
        )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RWKV Manifold Steering Report</title>
  <style>
    body {{ margin: 0; font-family: Inter, Arial, sans-serif; color: #17202a; background: #f7f8fb; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 22px 56px; }}
    h1 {{ font-size: 30px; margin: 0 0 12px; }}
    h2 {{ margin-top: 34px; border-bottom: 1px solid #d9dde7; padding-bottom: 8px; }}
    h3 {{ margin: 0 0 14px; }}
    p, li {{ line-height: 1.55; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d9dde7; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e5e8ef; text-align: left; font-size: 14px; }}
    th {{ background: #eef2f7; }}
    .run {{ margin-top: 24px; background: white; border: 1px solid #d9dde7; border-radius: 8px; padding: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    .grid img {{ width: 100%; display: block; border: 1px solid #e5e8ef; background: white; }}
    a {{ color: #195fb8; margin-right: 16px; }}
    .note {{ background: #fff8e6; border: 1px solid #ead391; padding: 12px 14px; border-radius: 6px; }}
    @media (max-width: 760px) {{ .grid {{ grid-template-columns: 1fr; }} table {{ font-size: 12px; }} }}
  </style>
</head>
<body>
<main>
  <h1>RWKV Manifold Steering Report</h1>
  <p>This package uses matched-endpoint runs, so linear and manifold steering share identical start/end hidden states. Behavior-space endpoint disagreement is therefore not a confound in these visuals.</p>
  <h2>Main Results</h2>
  {_html_table(rows)}
  <p class="note">The models are tiny and task accuracy is chance-level. Interpret the result as evidence about geometry and patching behavior, not as solved arithmetic.</p>
  <h2>Visuals</h2>
  {''.join(cards)}
  <h2>Animation</h2>
  <p>
    <a href="animation/manifold_report_scene.py">Manim scene source</a>
    <a href="animation/manim_matched_endpoint_report.mp4">Rendered Manim MP4</a>
    <a href="animation/manim_contact_sheet.jpg">Manim contact sheet</a>
    <a href="animation/matched_endpoint_report.mp4">Data-driven fallback MP4</a>
    <a href="animation/matched_endpoint_contact_sheet.jpg">Fallback contact sheet</a>
    <a href="audience_video/neural_geometry_rwkv_insight.mp4">Audience explainer MP4</a>
    <a href="audience_video/neural_geometry_rwkv_insight_contact_sheet.jpg">Audience contact sheet</a>
  </p>
  <h2>Cross-Model Comparisons</h2>
  <p>
    <a href="{visuals['report_compare_weekday_matched']['behavior_space_compare.html']}">Weekday behavior-space comparison</a>
    <a href="{visuals['report_compare_month_matched']['behavior_space_compare.html']}">Month behavior-space comparison</a>
  </p>
</main>
</body>
</html>
"""
    path = out_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def write_manim_scene(repo: Path, out_dir: Path, rows: list[dict[str, object]]) -> Path:
    scene_dir = out_dir / "animation"
    scene_dir.mkdir(parents=True, exist_ok=True)
    scene = scene_dir / "manifold_report_scene.py"
    payload = json.dumps(rows, indent=2)
    scene.write_text(
        f'''from __future__ import annotations

import numpy as np
from manim import *

ROWS = {payload}


class ManifoldSteeringReport(Scene):
    def construct(self):
        self.camera.background_color = "#f7f8fb"
        title = Text("Matched-endpoint manifold steering", font_size=36, color="#17202a")
        subtitle = Text("RWKV-7 0.1B vs Qwen3.5 0.8B", font_size=23, color="#425466")
        header = VGroup(title, subtitle).arrange(DOWN, buff=0.18).to_edge(UP)
        self.play(FadeIn(header, shift=0.2 * DOWN))

        left = self.make_space("Activation space", "#17202a").shift(LEFT * 3.4 + DOWN * 0.25)
        right = self.make_space("Behavior space", "#195fb8").shift(RIGHT * 3.4 + DOWN * 0.25)
        arrow = Arrow(left.get_right() + RIGHT * 0.25, right.get_left() + LEFT * 0.25, buff=0, color="#425466")
        label = Text("patch hidden state, read output probabilities", font_size=18, color="#425466").next_to(arrow, UP, buff=0.15)
        self.play(LaggedStart(FadeIn(left), GrowArrow(arrow), FadeIn(label), FadeIn(right), lag_ratio=0.18))

        square = Square(0.16, color=BLACK, fill_opacity=1).move_to(left[1].point_from_proportion(0.08))
        dot = Dot(left[2].point_from_proportion(0.08), radius=0.07, color="#777777")
        sq_b = Square(0.16, color=BLACK, fill_opacity=1).move_to(right[1].point_from_proportion(0.08))
        dot_b = Dot(right[2].point_from_proportion(0.08), radius=0.07, color="#777777")
        legend = VGroup(
            VGroup(Square(0.14, color=BLACK, fill_opacity=1), Text("manifold path", font_size=18, color="#17202a")).arrange(RIGHT, buff=0.15),
            VGroup(Dot(radius=0.07, color="#777777"), Text("linear path", font_size=18, color="#17202a")).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).to_edge(DOWN).shift(LEFT * 3.2)
        self.play(FadeIn(legend), FadeIn(square), FadeIn(dot), FadeIn(sq_b), FadeIn(dot_b))
        self.play(
            MoveAlongPath(square, left[1]),
            MoveAlongPath(dot, left[2]),
            MoveAlongPath(sq_b, right[1]),
            MoveAlongPath(dot_b, right[2]),
            run_time=4.5,
            rate_func=smooth,
        )

        table = self.make_metric_table().to_edge(DOWN).shift(RIGHT * 1.25)
        self.play(FadeIn(table, shift=0.25 * UP))
        self.wait(1.5)

    def make_space(self, title, color):
        title_obj = Text(title, font_size=24, color=color)
        manifold = ParametricFunction(
            lambda t: np.array([1.25 * np.cos(t), 0.72 * np.sin(t) + 0.18 * np.sin(2 * t), 0]),
            t_range=[0, TAU],
            color=BLACK,
        )
        linear = Line(manifold.point_from_proportion(0.08), manifold.point_from_proportion(0.58), color="#888888").set_stroke(width=4, opacity=0.85)
        manifold.set_stroke(width=5)
        concepts = VGroup(*[Dot(manifold.point_from_proportion(i / 7), radius=0.05, color="#195fb8") for i in range(7)])
        group = VGroup(title_obj, manifold, linear, concepts).arrange(DOWN, buff=0.22)
        return group

    def make_metric_table(self):
        rows = [row for row in ROWS if row["task"] == "weekday"]
        title = Text("Weekday geometry", font_size=20, color="#17202a")
        lines = VGroup(title)
        for row in rows:
            text = f'{{row["model"]}}: r(manifold)={{row["isometry_geometric_r"]:.3f}}, r(linear)={{row["isometry_linear_r"]:.3f}}'
            lines.add(Text(text, font_size=17, color="#17202a"))
        return lines.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
''',
        encoding="utf-8",
    )
    return scene


def _normalize_xy(path: np.ndarray, center: tuple[float, float], scale: float) -> np.ndarray:
    xy = path[:, :2].astype(np.float64)
    xy = xy - xy.mean(axis=0, keepdims=True)
    span = np.maximum(xy.max(axis=0) - xy.min(axis=0), 1e-6)
    xy = xy / span.max() * scale
    xy[:, 0] += center[0]
    xy[:, 1] += center[1]
    return xy


def write_fallback_video(repo: Path, out_dir: Path) -> Path:
    """Render a lightweight MP4 directly from saved trajectories."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    artifacts = np.load(repo / "outputs/report_weekday_rwkv_matched/artifacts.npz")
    act_geo = artifacts["activation_path_mds_3d_geometric"]
    act_lin = artifacts["activation_path_mds_3d_linear"]
    beh_geo = artifacts["behavior_path_mds_3d_geometric"]
    beh_lin = artifacts["behavior_path_mds_3d_linear"]
    n = act_geo.shape[0]
    act_geo_xy = _normalize_xy(act_geo, (-2.6, 0.0), 2.2)
    act_lin_xy = _normalize_xy(act_lin, (-2.6, 0.0), 2.2)
    beh_geo_xy = _normalize_xy(beh_geo, (2.6, 0.0), 2.2)
    beh_lin_xy = _normalize_xy(beh_lin, (2.6, 0.0), 2.2)

    out_path = out_dir / "animation" / "matched_endpoint_report.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.patch.set_facecolor("#f7f8fb")
    ax.set_facecolor("#f7f8fb")
    ax.set_xlim(-5.2, 5.2)
    ax.set_ylim(-3.0, 3.0)
    ax.axis("off")
    writer = FFMpegWriter(fps=24, metadata={"title": "Matched endpoint manifold steering"})
    with writer.saving(fig, str(out_path), dpi=160):
        for frame in range(n + 36):
            i = min(frame, n - 1)
            ax.clear()
            ax.set_xlim(-5.2, 5.2)
            ax.set_ylim(-3.0, 3.0)
            ax.axis("off")
            ax.text(0, 2.65, "Matched-endpoint manifold steering", ha="center", fontsize=20, weight="bold", color="#17202a")
            ax.text(0, 2.35, "RWKV-7 0.1B weekday run: square = manifold path, circle = linear path", ha="center", fontsize=11, color="#425466")
            ax.text(-2.6, 1.9, "Activation space", ha="center", fontsize=15, color="#17202a")
            ax.text(2.6, 1.9, "Behavior space", ha="center", fontsize=15, color="#195fb8")
            ax.annotate("", xy=(0.75, 0), xytext=(-0.75, 0), arrowprops=dict(arrowstyle="->", color="#425466", lw=2))
            ax.text(0, 0.18, "patch hidden state", ha="center", fontsize=10, color="#425466")
            for xy_geo, xy_lin in [(act_geo_xy, act_lin_xy), (beh_geo_xy, beh_lin_xy)]:
                ax.plot(xy_geo[:, 0], xy_geo[:, 1], color="black", lw=2.2, alpha=0.55)
                ax.plot(xy_lin[:, 0], xy_lin[:, 1], color="#888888", lw=2.0, ls="--", alpha=0.75)
                ax.scatter(xy_geo[i, 0], xy_geo[i, 1], marker="s", s=80, color="#111111", zorder=5)
                ax.scatter(xy_lin[i, 0], xy_lin[i, 1], marker="o", s=65, color="#777777", zorder=5)
            ax.text(-4.65, -2.5, "Endpoints are identical in hidden state and behavior probabilities.", fontsize=11, color="#17202a")
            ax.text(-4.65, -2.75, "The comparison is about the intermediate trajectory, not different targets.", fontsize=11, color="#17202a")
            writer.grab_frame()
    plt.close(fig)
    return out_path


def write_contact_sheet(out_dir: Path, video_path: Path) -> Path:
    import subprocess

    sheet = out_dir / "animation" / "matched_endpoint_contact_sheet.jpg"
    frames_dir = out_dir / "animation" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            "fps=1,scale=360:-1,tile=4x2",
            "-frames:v",
            "1",
            str(sheet),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--out-dir", default="reports/manifold_report")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    out_dir = (repo / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _metric_rows(repo)
    _write_csv(out_dir / "metrics.csv", rows)
    (out_dir / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    visuals = _copy_visuals(repo, out_dir)
    report_md = write_markdown_report(repo, out_dir, rows, visuals)
    report_html = write_html_report(repo, out_dir, rows, visuals)
    scene = write_manim_scene(repo, out_dir, rows)
    video = write_fallback_video(repo, out_dir)
    sheet = write_contact_sheet(out_dir, video)
    manifest = {
        "report_markdown": str(report_md),
        "report_html": str(report_html),
        "metrics_csv": str(out_dir / "metrics.csv"),
        "metrics_json": str(out_dir / "metrics.json"),
        "manim_scene": str(scene),
        "manim_video": str(out_dir / "animation" / "manim_matched_endpoint_report.mp4"),
        "manim_contact_sheet": str(out_dir / "animation" / "manim_contact_sheet.jpg"),
        "fallback_video": str(video),
        "contact_sheet": str(sheet),
        "note": "Manim scene source is provided; fallback MP4 is rendered directly from saved experiment trajectories.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
