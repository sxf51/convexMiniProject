"""Closed-loop simulation of the vehicle under MPC control."""

from __future__ import annotations

import numpy as np

from controllers.mpc import MPCController
from models.double_integrator import DoubleIntegrator


class Simulator:
    """Runs the vehicle model in closed loop with an MPC controller."""

    def __init__(
        self,
        model: DoubleIntegrator,
        controller: MPCController,
        noise_std: float = 0.0,
        rng: int | np.random.Generator | None = None,
    ) -> None:
        self.model = model
        self.controller = controller
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(rng)

    def run(
        self,
        x0: np.ndarray,
        reference: np.ndarray,
        n_steps: int | None = None,
        wrap: bool = False,
        state_constraints: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate the closed-loop system.

        Returns ``(states, controls)`` where ``states`` has one more row than
        ``controls`` (the initial state plus one row per applied input).
        ``wrap`` treats ``reference`` (and ``state_constraints``) as a closed loop.
        """
        reference = np.asarray(reference, dtype=float)
        nx = reference.shape[1]
        horizon = self.controller.horizon

        if n_steps is None:
            n_steps = reference.shape[0] if wrap else max(0, reference.shape[0] - horizon)

        states = [np.asarray(x0, dtype=float).reshape(-1)]
        controls: list[np.ndarray] = []

        for step in range(n_steps):
            current = states[-1]
            indices = np.arange(step, step + horizon + 1)
            if wrap:
                indices = indices % reference.shape[0]
            else:
                indices = np.clip(indices, 0, reference.shape[0] - 1)

            window = reference[indices]

            constraints = None
            if state_constraints is not None:
                C, d_min, d_max = state_constraints
                constraints = (C[indices], d_min[indices], d_max[indices])

            u = self.controller.control(current, window, constraints)

            w = self.rng.normal(0.0, self.noise_std, size=nx) if self.noise_std > 0 else None

            next_state = self.model.step(current, u, w)
            states.append(next_state)
            controls.append(u)

        return np.asarray(states), np.asarray(controls)
