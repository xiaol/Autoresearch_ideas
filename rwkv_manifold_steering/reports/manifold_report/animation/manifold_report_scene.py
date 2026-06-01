from __future__ import annotations

import numpy as np
from manim import *

ROWS = [
  {
    "task": "weekday",
    "model": "RWKV-7 0.1B",
    "layer": 11,
    "base_accuracy": 0.14285714285714285,
    "isometry_geometric_r": 0.9485235558702364,
    "isometry_linear_r": 0.8906151342126383,
    "geometric_behavior_distance": 0.012833359939714727,
    "linear_behavior_distance": 0.013266041837243099,
    "geometric_geodesic_distance": 0.01619405299425125,
    "linear_geodesic_distance": 0.016682764515280724,
    "endpoint_hidden_l2_start": 0.0,
    "endpoint_hidden_l2_end": 0.0,
    "endpoint_behavior_l1_start": 0.0,
    "endpoint_behavior_l1_end": 0.0,
    "run_dir": "outputs/report_weekday_rwkv_matched"
  },
  {
    "task": "weekday",
    "model": "Qwen3.5 0.8B",
    "layer": 2,
    "base_accuracy": 0.14285714285714285,
    "isometry_geometric_r": 0.8567248122258615,
    "isometry_linear_r": 0.5373864897263517,
    "geometric_behavior_distance": 0.520252145466359,
    "linear_behavior_distance": 0.5227992033498463,
    "geometric_geodesic_distance": 0.5535328388214111,
    "linear_geodesic_distance": 0.5561314821243286,
    "endpoint_hidden_l2_start": 0.0,
    "endpoint_hidden_l2_end": 0.0,
    "endpoint_behavior_l1_start": 0.0,
    "endpoint_behavior_l1_end": 0.0,
    "run_dir": "outputs/report_weekday_qwen_matched"
  },
  {
    "task": "month",
    "model": "RWKV-7 0.1B",
    "layer": 5,
    "base_accuracy": 0.08333333333333333,
    "isometry_geometric_r": 0.9893030157380409,
    "isometry_linear_r": 0.21034045012998698,
    "geometric_behavior_distance": 1.922766400130256,
    "linear_behavior_distance": 1.9215775464900562,
    "geometric_geodesic_distance": 1.9427059888839722,
    "linear_geodesic_distance": 1.9415408372879028,
    "endpoint_hidden_l2_start": 0.0,
    "endpoint_hidden_l2_end": 0.0,
    "endpoint_behavior_l1_start": 0.0,
    "endpoint_behavior_l1_end": 0.0,
    "run_dir": "outputs/report_month_rwkv_matched"
  },
  {
    "task": "month",
    "model": "Qwen3.5 0.8B",
    "layer": 20,
    "base_accuracy": 0.08333333333333333,
    "isometry_geometric_r": 0.9154983046636843,
    "isometry_linear_r": 0.7439587048559193,
    "geometric_behavior_distance": 0.06381324915532204,
    "linear_behavior_distance": 0.0639100544927662,
    "geometric_geodesic_distance": 0.07148326933383942,
    "linear_geodesic_distance": 0.07175631076097488,
    "endpoint_hidden_l2_start": 0.0,
    "endpoint_hidden_l2_end": 0.0,
    "endpoint_behavior_l1_start": 0.0,
    "endpoint_behavior_l1_end": 0.0,
    "run_dir": "outputs/report_month_qwen_matched"
  }
]


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
            text = f'{row["model"]}: r(manifold)={row["isometry_geometric_r"]:.3f}, r(linear)={row["isometry_linear_r"]:.3f}'
            lines.add(Text(text, font_size=17, color="#17202a"))
        return lines.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
