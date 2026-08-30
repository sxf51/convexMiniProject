"""Double integrator vehicle model.

The state is ``x = [p1, p2, v1, v2]`` and the control input is ``u = [a1, a2]``.
The continuous dynamics are discretised with the Euler-approximated matrices used in
the assignment:

    x_{k+1} = A x_k + B u_k + w_k
"""

from __future__ import annotations

import numpy as np


class DoubleIntegrator:
    """Discrete-time double integrator with optional additive process noise."""

    def __init__(self, sampling_time: float = 0.1) -> None:
        if sampling_time <= 0:
            msg = "sampling_time must be positive"
            raise ValueError(msg)
        self.sampling_time = float(sampling_time)

    @property
    def A(self) -> np.ndarray:
        """State transition matrix."""
        ts = self.sampling_time
        return np.array(
            [
                [1.0, 0.0, ts, 0.0],
                [0.0, 1.0, 0.0, ts],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    @property
    def B(self) -> np.ndarray:
        """Control input matrix."""
        ts = self.sampling_time
        return np.array(
            [
                [0.5 * ts**2, 0.0],
                [0.0, 0.5 * ts**2],
                [ts, 0.0],
                [0.0, ts],
            ]
        )

    def step(self, x: np.ndarray, u: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
        """Advance the state by one time step."""
        x = np.asarray(x, dtype=float).reshape(4)
        u = np.asarray(u, dtype=float).reshape(2)
        x_next = self.A @ x + self.B @ u
        if w is not None:
            x_next = x_next + np.asarray(w, dtype=float).reshape(4)
        return x_next
