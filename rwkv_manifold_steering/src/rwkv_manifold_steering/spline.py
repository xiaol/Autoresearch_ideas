from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


TAU = 2.0 * math.pi


@dataclass(frozen=True)
class PeriodicCubicSpline1D:
    """Exact periodic cubic spline for a 1D cyclic intrinsic coordinate."""

    x: np.ndarray
    y: np.ndarray
    period: float = TAU

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=np.float64).reshape(-1)
        y = np.asarray(self.y, dtype=np.float64)
        if y.ndim != 2:
            raise ValueError(f"y must have shape (n, d), got {y.shape}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x has {x.shape[0]} rows but y has {y.shape[0]}")
        if x.shape[0] < 3:
            raise ValueError("periodic cubic spline needs at least 3 control points")

        order = np.argsort(x)
        x = x[order]
        y = y[order]
        if np.any(np.diff(x) <= 0):
            raise ValueError("control points must be unique")
        if float(self.period) <= float(x[-1] - x[0]):
            raise ValueError("period must be larger than the knot span")

        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "_h", self._intervals(x, float(self.period)))
        object.__setattr__(self, "_m", self._solve_second_derivatives(x, y, float(self.period)))

    @staticmethod
    def _intervals(x: np.ndarray, period: float) -> np.ndarray:
        return np.concatenate([np.diff(x), np.array([x[0] + period - x[-1]])])

    @classmethod
    def _solve_second_derivatives(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        period: float,
    ) -> np.ndarray:
        n = x.shape[0]
        h = cls._intervals(x, period)
        matrix = np.zeros((n, n), dtype=np.float64)
        rhs = np.zeros((n, y.shape[1]), dtype=np.float64)

        for i in range(n):
            prev_i = (i - 1) % n
            next_i = (i + 1) % n
            h_prev = h[prev_i]
            h_next = h[i]
            matrix[i, prev_i] = h_prev
            matrix[i, i] = 2.0 * (h_prev + h_next)
            matrix[i, next_i] = h_next
            slope_next = (y[next_i] - y[i]) / h_next
            slope_prev = (y[i] - y[prev_i]) / h_prev
            rhs[i] = 6.0 * (slope_next - slope_prev)

        try:
            return np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(matrix, rhs, rcond=None)[0]

    def evaluate(self, u: np.ndarray) -> np.ndarray:
        query = np.asarray(u, dtype=np.float64).reshape(-1)
        x0 = float(self.x[0])
        wrapped = ((query - x0) % float(self.period)) + x0
        # The final wrap segment is represented as [x[-1], x[0] + period].
        wrapped = np.where(wrapped < self.x[0], wrapped + self.period, wrapped)

        out = np.empty((wrapped.shape[0], self.y.shape[1]), dtype=np.float64)
        x_ext = np.concatenate([self.x, np.array([self.x[0] + self.period])])
        y_ext = np.concatenate([self.y, self.y[:1]], axis=0)
        m_ext = np.concatenate([self._m, self._m[:1]], axis=0)

        for row, value in enumerate(wrapped):
            if value < self.x[0]:
                value += self.period
            segment = int(np.searchsorted(x_ext, value, side="right") - 1)
            segment = min(max(segment, 0), self.x.shape[0] - 1)
            left = x_ext[segment]
            right = x_ext[segment + 1]
            h = right - left
            a = right - value
            b = value - left
            out[row] = (
                m_ext[segment] * (a**3) / (6.0 * h)
                + m_ext[segment + 1] * (b**3) / (6.0 * h)
                + (y_ext[segment] / h - m_ext[segment] * h / 6.0) * a
                + (y_ext[segment + 1] / h - m_ext[segment + 1] * h / 6.0) * b
            )

        return out.astype(np.float32)


def weekday_thetas() -> np.ndarray:
    return np.linspace(0.0, TAU, 7, endpoint=False, dtype=np.float32)


def shortest_arc(
    start: float,
    end: float,
    steps: int,
    *,
    period: float = TAU,
    oversteer_frac: float = 0.0,
    oversteer_steps: int = 0,
) -> np.ndarray:
    if steps < 2:
        raise ValueError("steps must be at least 2")
    normal = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    if oversteer_frac > 0.0 and oversteer_steps > 0:
        step = oversteer_frac / oversteer_steps
        extra = np.linspace(1.0 + step, 1.0 + oversteer_frac, oversteer_steps)
        alphas = np.concatenate([normal, extra.astype(np.float32)])
    else:
        alphas = normal
    delta = ((end - start + period / 2.0) % period) - period / 2.0
    return (start + alphas * delta).astype(np.float32)


def pad_other_bin(p: np.ndarray) -> np.ndarray:
    other = np.clip(1.0 - p.sum(axis=-1, keepdims=True), 0.0, 1.0)
    return np.concatenate([p, other], axis=-1)


def hellinger_sqrt(p: np.ndarray) -> np.ndarray:
    root = np.sqrt(np.clip(p, 0.0, 1.0))
    norm = np.linalg.norm(root, axis=-1, keepdims=True)
    return root / np.clip(norm, 1e-8, None)


def hellinger_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=-1) / math.sqrt(2.0)


def bhattacharyya_from_hellinger(d_h: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(1.0 - d_h**2, 1e-7, None))
