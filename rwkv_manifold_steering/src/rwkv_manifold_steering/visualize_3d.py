from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib import animation
from matplotlib import colors as mcolors

from .spline import TAU
from .tasks import WEEKDAYS


_FONT_FAMILY = "DejaVu Sans, Arial, sans-serif"
_CAMERA_PRESETS: dict[str, dict[str, dict[str, float]]] = {
    "default": {"eye": {"x": 1.7, "y": 1.6, "z": 1.25}},
    "front": {"eye": {"x": 0.0, "y": 2.8, "z": 0.15}},
    "side": {"eye": {"x": 2.8, "y": 0.0, "z": 0.15}},
    "top": {"eye": {"x": 0.1, "y": 0.1, "z": 2.9}, "up": {"x": 0.0, "y": 1.0, "z": 0.0}},
    "diagonal": {"eye": {"x": 2.2, "y": 1.25, "z": 1.65}},
}


def parse_component_indices(spec: str) -> tuple[int, int, int]:
    tokens = spec.replace(",", " ").split()
    if len(tokens) != 3:
        raise ValueError(f"expected three indices like '0,1,2', got {spec!r}")
    values = tuple(int(token) for token in tokens)
    if any(value < 0 for value in values):
        raise ValueError(f"component indices must be non-negative, got {values}")
    if len(set(values)) != 3:
        raise ValueError(f"component indices must be unique, got {values}")
    return values


def select_components(points: np.ndarray, components: tuple[int, int, int]) -> np.ndarray:
    coords = np.asarray(points, dtype=np.float32)
    if coords.shape[-1] <= max(components):
        raise ValueError(
            f"expected at least {max(components) + 1} coordinates, got {coords.shape[-1]}"
        )
    return coords[..., list(components)].astype(np.float32)


def _theta_colors(theta: np.ndarray) -> list[str]:
    cmap = plt.get_cmap("twilight_shifted")
    norm = mcolors.Normalize(vmin=0.0, vmax=TAU)
    rgba = cmap(norm(np.mod(np.asarray(theta, dtype=np.float32), TAU)))
    return [
        f"rgba({int(round(r * 255))},{int(round(g * 255))},{int(round(b * 255))},{a:.3f})"
        for r, g, b, a in rgba
    ]


def _plotly_rgba_to_mpl(color: str) -> tuple[float, float, float, float]:
    if color.startswith("rgba(") and color.endswith(")"):
        parts = color[5:-1].split(",")
        if len(parts) != 4:
            raise ValueError(f"invalid RGBA color: {color!r}")
        r, g, b = [float(part) / 255.0 for part in parts[:3]]
        a = float(parts[3])
        return (r, g, b, a)
    return mcolors.to_rgba(color)


def _curve(points: np.ndarray, theta: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(theta, dtype=np.float32))
    return np.vstack([np.asarray(points, dtype=np.float32)[order], np.asarray(points, dtype=np.float32)[order][:1]])


def _label_traces(coords: np.ndarray, theta: np.ndarray, labels: list[str]) -> list[go.Scatter3d]:
    traces: list[go.Scatter3d] = []
    colors = _theta_colors(theta)
    for idx, label in enumerate(labels):
        traces.append(
            go.Scatter3d(
                x=[float(coords[idx, 0])],
                y=[float(coords[idx, 1])],
                z=[float(coords[idx, 2])],
                mode="markers+text",
                marker=dict(size=10, color=colors[idx], symbol="diamond"),
                text=[label],
                textposition="top center",
                textfont=dict(size=11, color=colors[idx], family=_FONT_FAMILY),
                hovertemplate=f"<b>{label}</b><extra></extra>",
                showlegend=False,
            )
        )
    return traces


def _manifold_figure(
    *,
    title: str,
    coords: np.ndarray,
    theta: np.ndarray,
    labels: list[str] | None = None,
) -> go.Figure:
    coords = np.asarray(coords, dtype=np.float32)
    theta = np.asarray(theta, dtype=np.float32)
    labels = list(labels or WEEKDAYS)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers",
            marker=dict(size=4, color=_theta_colors(theta), opacity=0.22),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    curve = _curve(coords, theta)
    fig.add_trace(
        go.Scatter3d(
            x=curve[:, 0],
            y=curve[:, 1],
            z=curve[:, 2],
            mode="lines",
            line=dict(color="rgba(55,55,55,0.7)", width=4),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    for trace in _label_traces(coords[: len(labels)], theta[: len(labels)], labels):
        fig.add_trace(trace)
    fig.update_layout(
        margin=dict(l=0, r=0, t=48, b=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family=_FONT_FAMILY, size=11),
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(size=16, family=_FONT_FAMILY, color="#222"),
        ),
        scene=dict(
            aspectmode="data",
            xaxis=dict(title="MDS-1", gridcolor="#eee", showbackground=False),
            yaxis=dict(title="MDS-2", gridcolor="#eee", showbackground=False),
            zaxis=dict(title="MDS-3", gridcolor="#eee", showbackground=False),
            camera=_CAMERA_PRESETS["default"],
        ),
    )
    return fig


def _plot_div(fig: go.Figure, *, div_id: str, include_plotlyjs: bool) -> str:
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        div_id=div_id,
        config=dict(responsive=True, displaylogo=False, scrollZoom=True),
    )


def _camera_buttons_html() -> str:
    labels = ["Default", "Front", "Side", "Top", "Diagonal"]
    keys = ["default", "front", "side", "top", "diagonal"]
    return "".join(
        f'<button type="button" data-camera="{key}">{label}</button>'
        for label, key in zip(labels, keys, strict=True)
    )


def _page_shell(*, title: str, subtitle: str, body_html: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #fff;
      color: #222;
      font-family: {_FONT_FAMILY};
    }}
    .page {{ padding: 8px 10px 14px; }}
    .title {{
      margin: 0 0 4px;
      text-align: center;
      font-size: 18px;
      font-weight: 600;
    }}
    .subtitle {{
      margin: 0 0 10px;
      text-align: center;
      font-size: 12px;
      color: #666;
    }}
    .controls {{
      display: flex;
      justify-content: center;
      margin: 6px 0 10px;
    }}
    .button-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: center;
    }}
    .button-row button, .timeline-row button {{
      border: 1px solid #bbb;
      background: #fff;
      color: #222;
      padding: 6px 10px;
      font-size: 12px;
      cursor: pointer;
    }}
    .button-row button:hover, .timeline-row button:hover {{ background: #f3f3f3; }}
    .plots {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: stretch;
      justify-content: center;
    }}
    .plot {{
      min-width: 360px;
      flex: 1 1 560px;
    }}
    .timeline {{
      max-width: 1180px;
      margin: 0 auto 10px;
      padding: 0 8px;
    }}
    .timeline-row {{
      display: flex;
      align-items: center;
      gap: 12px;
      justify-content: center;
      flex-wrap: wrap;
    }}
    .timeline-row input[type=range] {{
      width: min(680px, 72vw);
      cursor: pointer;
    }}
    .step-label {{
      font-size: 12px;
      color: #555;
      min-width: 92px;
      text-align: center;
    }}
  </style>
</head>
<body>
  <div class="page">
    <h1 class="title">{title}</h1>
    <p class="subtitle">{subtitle}</p>
    <div class="controls">
      <div class="button-row" id="camera-buttons">
        {_camera_buttons_html()}
      </div>
    </div>
    {body_html}
  </div>
</body>
</html>"""


def write_isometry_3d_html(
    out_path: Path,
    *,
    theta: np.ndarray,
    activation_mds: np.ndarray,
    behavior_mds: np.ndarray,
    pearson_r_geometric: float,
    pearson_r_linear: float,
    labels: list[str] | None = None,
    page_title: str = "RWKV weekday manifold isometry",
    concept_name: str = "weekday",
) -> Path:
    labels = list(labels or WEEKDAYS)
    act_fig = _manifold_figure(
        title=f"Activation manifold | r={pearson_r_geometric:.3f}",
        coords=np.asarray(activation_mds, dtype=np.float32),
        theta=np.asarray(theta, dtype=np.float32),
        labels=labels,
    )
    beh_fig = _manifold_figure(
        title=f"Behavior manifold | r={pearson_r_linear:.3f}",
        coords=np.asarray(behavior_mds, dtype=np.float32),
        theta=np.asarray(theta, dtype=np.float32),
        labels=labels,
    )
    body_html = f"""
    <div class="plots">
      <div class="plot">{_plot_div(act_fig, div_id="isometry-act", include_plotlyjs=True)}</div>
      <div class="plot">{_plot_div(beh_fig, div_id="isometry-beh", include_plotlyjs=False)}</div>
    </div>
    <script>
    window.addEventListener('load', function() {{
      var cameras = {json.dumps(_CAMERA_PRESETS)};
      var act = document.getElementById('isometry-act');
      var beh = document.getElementById('isometry-beh');
      var buttons = document.querySelectorAll('#camera-buttons button');
      function setCamera(name) {{
        var camera = cameras[name];
        if (!camera) return;
        Plotly.relayout(act, {{'scene.camera': camera}});
        Plotly.relayout(beh, {{'scene.camera': camera}});
      }}
      buttons.forEach(function(button) {{
        button.addEventListener('click', function() {{
          setCamera(this.getAttribute('data-camera'));
        }});
      }});
      setCamera('default');
    }});
    </script>
    """
    html = _page_shell(
        title=page_title,
        subtitle=(
            "Approximate isometry between activation and behavior manifolds "
            f"for {concept_name} "
            f"(geometric r={pearson_r_geometric:.3f}, linear r={pearson_r_linear:.3f})"
        ),
        body_html=body_html,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _path_figure(
    *,
    title: str,
    coords: np.ndarray,
    theta: np.ndarray,
    geometric_path: np.ndarray,
    linear_path: np.ndarray,
    layer: int,
    labels: list[str] | None = None,
) -> tuple[go.Figure, dict[str, int]]:
    fig = _manifold_figure(title=title, coords=coords, theta=theta, labels=labels)
    geo_line_idx = len(fig.data)
    fig.add_trace(
        go.Scatter3d(
            x=geometric_path[:, 0],
            y=geometric_path[:, 1],
            z=geometric_path[:, 2],
            mode="lines",
            line=dict(color="black", width=5, dash="dash"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    geo_marker_idx = len(fig.data)
    fig.add_trace(
        go.Scatter3d(
            x=[float(geometric_path[0, 0])],
            y=[float(geometric_path[0, 1])],
            z=[float(geometric_path[0, 2])],
            mode="markers",
            marker=dict(size=11, color="#000000", symbol="square", line=dict(color="#111", width=1)),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    lin_line_idx = len(fig.data)
    fig.add_trace(
        go.Scatter3d(
            x=linear_path[:, 0],
            y=linear_path[:, 1],
            z=linear_path[:, 2],
            mode="lines",
            line=dict(color="#8d8d8d", width=4, dash="dash"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    lin_marker_idx = len(fig.data)
    fig.add_trace(
        go.Scatter3d(
            x=[float(linear_path[0, 0])],
            y=[float(linear_path[0, 1])],
            z=[float(linear_path[0, 2])],
            mode="markers",
            marker=dict(size=10, color="#8d8d8d", symbol="circle", line=dict(color="#444", width=1)),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>layer {layer}</sup>",
            x=0.5,
            xanchor="center",
            font=dict(size=16, family=_FONT_FAMILY, color="#222"),
        ),
        margin=dict(l=0, r=0, t=56, b=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family=_FONT_FAMILY, size=11),
        scene=dict(
            aspectmode="data",
            xaxis=dict(title="MDS-1", gridcolor="#eee", showbackground=False),
            yaxis=dict(title="MDS-2", gridcolor="#eee", showbackground=False),
            zaxis=dict(title="MDS-3", gridcolor="#eee", showbackground=False),
            camera=_CAMERA_PRESETS["default"],
        ),
    )
    return fig, {
        "geo_line": geo_line_idx,
        "geo_marker": geo_marker_idx,
        "lin_line": lin_line_idx,
        "lin_marker": lin_marker_idx,
    }


def write_steering_movement_3d_html(
    out_path: Path,
    *,
    theta: np.ndarray,
    activation_vertices_3d: np.ndarray,
    behavior_vertices_3d: np.ndarray,
    activation_paths: dict[str, np.ndarray],
    behavior_paths: dict[str, np.ndarray],
    start: int,
    end: int,
    layer: int,
    start_name: str | None = None,
    end_name: str | None = None,
    labels: list[str] | None = None,
    page_title: str = "RWKV manifold steering",
    concept_name: str = "weekday",
) -> Path:
    labels = list(labels or WEEKDAYS)
    start_name = start_name or labels[start]
    end_name = end_name or labels[end]
    act = np.asarray(activation_vertices_3d, dtype=np.float32)
    beh = np.asarray(behavior_vertices_3d, dtype=np.float32)
    geo_act = np.asarray(activation_paths["geometric"], dtype=np.float32)
    lin_act = np.asarray(activation_paths["linear"], dtype=np.float32)
    geo_beh = np.asarray(behavior_paths["geometric"], dtype=np.float32)
    lin_beh = np.asarray(behavior_paths["linear"], dtype=np.float32)
    max_steps = max(len(geo_act), len(lin_act), len(geo_beh), len(lin_beh))
    normal_steps = min(len(geo_act), len(lin_act), len(geo_beh), len(lin_beh))
    colors = _theta_colors(np.asarray(theta, dtype=np.float32))
    start_color = colors[start]
    end_color = colors[end]

    act_fig, act_idx = _path_figure(
        title="Activation space",
        coords=act,
        theta=np.asarray(theta, dtype=np.float32),
        geometric_path=geo_act,
        linear_path=lin_act,
        layer=layer,
        labels=labels,
    )
    beh_fig, beh_idx = _path_figure(
        title="Behavior space",
        coords=beh,
        theta=np.asarray(theta, dtype=np.float32),
        geometric_path=geo_beh,
        linear_path=lin_beh,
        layer=layer,
        labels=labels,
    )
    body_html = f"""
    <div class="plots">
      <div class="plot">{_plot_div(act_fig, div_id="steer-act", include_plotlyjs=True)}</div>
      <div class="plot">{_plot_div(beh_fig, div_id="steer-beh", include_plotlyjs=False)}</div>
    </div>
    <div class="timeline">
      <div class="timeline-row">
        <button type="button" id="play-button">Play</button>
        <input type="range" id="step-slider" min="0" max="{max_steps - 1}" value="0">
        <div class="step-label" id="step-label">Step 1 / {max_steps}</div>
      </div>
    </div>
    <script>
    window.addEventListener('load', function() {{
      var cameras = {json.dumps(_CAMERA_PRESETS)};
      var act = document.getElementById('steer-act');
      var beh = document.getElementById('steer-beh');
      var slider = document.getElementById('step-slider');
      var label = document.getElementById('step-label');
      var play = document.getElementById('play-button');
      var buttons = document.querySelectorAll('#camera-buttons button');
      var maxSteps = {max_steps};
      var normalSteps = {normal_steps};
      var geoAct = {json.dumps(geo_act.tolist())};
      var linAct = {json.dumps(lin_act.tolist())};
      var geoBeh = {json.dumps(geo_beh.tolist())};
      var linBeh = {json.dumps(lin_beh.tolist())};
      var actIdx = {json.dumps(act_idx)};
      var behIdx = {json.dumps(beh_idx)};
      var startColor = {json.dumps(start_color)};
      var endColor = {json.dumps(end_color)};
      var timer = null;
      var playing = false;

      function hexToRgb(color) {{
        if (color.indexOf('rgba') === 0) {{
          var raw = color.substring(color.indexOf('(') + 1, color.indexOf(')')).split(',');
          return [parseFloat(raw[0]), parseFloat(raw[1]), parseFloat(raw[2])];
        }}
        if (color.indexOf('#') === 0) {{
          var hex = color.replace('#', '');
          return [
            parseInt(hex.substring(0, 2), 16),
            parseInt(hex.substring(2, 4), 16),
            parseInt(hex.substring(4, 6), 16)
          ];
        }}
        return [0, 0, 0];
      }}

      function rgbToCss(r, g, b) {{
        return 'rgb(' + [r, g, b].map(function(v) {{
          return Math.max(0, Math.min(255, Math.round(v)));
        }}).join(',') + ')';
      }}

      function lerpColor(a, b, t) {{
        var ca = hexToRgb(a), cb = hexToRgb(b);
        return rgbToCss(
          ca[0] + (cb[0] - ca[0]) * t,
          ca[1] + (cb[1] - ca[1]) * t,
          ca[2] + (cb[2] - ca[2]) * t
        );
      }}

      function setCamera(name) {{
        var camera = cameras[name];
        if (!camera) return;
        Plotly.relayout(act, {{'scene.camera': camera}});
        Plotly.relayout(beh, {{'scene.camera': camera}});
      }}

      function clampStep(step, path) {{
        return Math.max(0, Math.min(step, path.length - 1));
      }}

      function updateMarker(div, idx, path, step, color) {{
        var s = clampStep(step, path);
        Plotly.restyle(div, {{
          x: [[path[s][0]]],
          y: [[path[s][1]]],
          z: [[path[s][2]]],
          'marker.color': [color]
        }}, [idx]);
      }}

      function update(step) {{
        slider.value = step;
        label.textContent = 'Step ' + (step + 1) + ' / ' + maxSteps;
        var t = normalSteps > 1 ? Math.min(step, normalSteps - 1) / (normalSteps - 1) : 0;
        var color = lerpColor(startColor, endColor, t);
        updateMarker(act, actIdx.geo_marker, geoAct, step, color);
        updateMarker(act, actIdx.lin_marker, linAct, step, color);
        updateMarker(beh, behIdx.geo_marker, geoBeh, step, color);
        updateMarker(beh, behIdx.lin_marker, linBeh, step, color);
      }}

      function stop() {{
        if (timer) {{
          clearInterval(timer);
          timer = null;
        }}
        playing = false;
        play.textContent = 'Play';
      }}

      function start() {{
        if (playing) return;
        playing = true;
        play.textContent = 'Pause';
        timer = setInterval(function() {{
          var step = parseInt(slider.value, 10);
          if (step >= maxSteps - 1) {{
            stop();
            update(0);
            return;
          }}
          update(step + 1);
        }}, 110);
      }}

      buttons.forEach(function(button) {{
        button.addEventListener('click', function() {{
          setCamera(this.getAttribute('data-camera'));
        }});
      }});
      slider.addEventListener('input', function() {{
        stop();
        update(parseInt(this.value, 10));
      }});
      play.addEventListener('click', function() {{
        if (playing) {{
          stop();
        }} else {{
          start();
        }}
      }});
      setCamera('default');
      update(0);
    }});
    </script>
    """
    html = _page_shell(
        title=page_title,
        subtitle=(
            f"Geometric path: shortest arc in intrinsic {concept_name} space; "
            f"linear path: raw hidden interpolation | {start_name} -> {end_name} | layer {layer}"
        ),
        body_html=body_html,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def write_steering_movement_gif(
    out_path: Path,
    *,
    activation_paths: dict[str, np.ndarray],
    behavior_paths: dict[str, np.ndarray],
    start: int,
    end: int,
    start_name: str | None = None,
    end_name: str | None = None,
    labels: list[str] | None = None,
    page_title: str = "RWKV manifold steering",
    fps: int = 10,
) -> Path:
    labels = list(labels or WEEKDAYS)
    start_name = start_name or labels[start]
    end_name = end_name or labels[end]
    geo_act = np.asarray(activation_paths["geometric"], dtype=np.float32)
    lin_act = np.asarray(activation_paths["linear"], dtype=np.float32)
    geo_beh = np.asarray(behavior_paths["geometric"], dtype=np.float32)
    lin_beh = np.asarray(behavior_paths["linear"], dtype=np.float32)
    n_frames = max(len(geo_act), len(lin_act), len(geo_beh), len(lin_beh))
    colors = _theta_colors(np.linspace(0.0, TAU, n_frames, endpoint=False))
    mpl_colors = [_plotly_rgba_to_mpl(color) for color in colors]

    fig = plt.figure(figsize=(12.5, 6.5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    for ax, title, geo_path, lin_path in [
        (ax1, "Activation space", geo_act, lin_act),
        (ax2, "Behavior space", geo_beh, lin_beh),
    ]:
        ax.set_title(title, fontsize=13, pad=12)
        ax.plot(geo_path[:, 0], geo_path[:, 1], geo_path[:, 2], color="black", linewidth=2.5, linestyle="--", alpha=0.55)
        ax.plot(lin_path[:, 0], lin_path[:, 1], lin_path[:, 2], color="#8d8d8d", linewidth=2.3, linestyle="--", alpha=0.65)
        ax.scatter(
            geo_path[0, 0],
            geo_path[0, 1],
            geo_path[0, 2],
            color=mpl_colors[0],
            marker="s",
            s=70,
            edgecolors="black",
            linewidths=0.8,
        )
        ax.scatter(
            lin_path[0, 0],
            lin_path[0, 1],
            lin_path[0, 2],
            color=mpl_colors[0],
            marker="o",
            s=55,
            edgecolors="#444",
            linewidths=0.8,
        )
        ax.set_xlabel("MDS-1")
        ax.set_ylabel("MDS-2")
        ax.set_zlabel("MDS-3")
        ax.view_init(elev=22, azim=35)
        ax.grid(True, alpha=0.2)
    fig.suptitle(
        f"{page_title}: {start_name} -> {end_name}",
        fontsize=15,
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        "square = geometric path, circle = linear path",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#444",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    geo_act_marker = ax1.scatter([], [], [], color="black", marker="s", s=80)
    lin_act_marker = ax1.scatter([], [], [], color="#8d8d8d", marker="o", s=60)
    geo_beh_marker = ax2.scatter([], [], [], color="black", marker="s", s=80)
    lin_beh_marker = ax2.scatter([], [], [], color="#8d8d8d", marker="o", s=60)

    def _set_marker(scatter, point: np.ndarray) -> None:
        scatter._offsets3d = (
            np.asarray([float(point[0])]),
            np.asarray([float(point[1])]),
            np.asarray([float(point[2])]),
        )

    def _update(frame: int):
        step = min(frame, n_frames - 1)
        _set_marker(geo_act_marker, geo_act[step])
        _set_marker(lin_act_marker, lin_act[step])
        _set_marker(geo_beh_marker, geo_beh[step])
        _set_marker(lin_beh_marker, lin_beh[step])
        color = mpl_colors[step]
        geo_act_marker.set_color(color)
        lin_act_marker.set_color(color)
        geo_beh_marker.set_color(color)
        lin_beh_marker.set_color(color)
        ax1.set_title(f"Activation space | step {step + 1}/{n_frames}", fontsize=13, pad=12)
        ax2.set_title(f"Behavior space | step {step + 1}/{n_frames}", fontsize=13, pad=12)
        return geo_act_marker, lin_act_marker, geo_beh_marker, lin_beh_marker

    anim = animation.FuncAnimation(fig, _update, frames=n_frames, interval=1000 / fps, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return out_path
