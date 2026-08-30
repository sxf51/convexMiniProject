"""Model predictive control (MPC) controller."""

from __future__ import annotations

from typing import Any

import numpy as np

from controllers.utils import condense_qp, solve_box_qp_pgd, solve_qp_gurobi


class MPCController:
    """Receding-horizon MPC controller for the double integrator."""

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        horizon: int,
        Q: np.ndarray,
        R: np.ndarray,
        u_min: np.ndarray | None = None,
        u_max: np.ndarray | None = None,
        solver: str = "pgd",
        **solver_kwargs: Any,
    ) -> None:
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.horizon = int(horizon)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.u_min = None if u_min is None else np.asarray(u_min, dtype=float).reshape(-1)
        self.u_max = None if u_max is None else np.asarray(u_max, dtype=float).reshape(-1)
        self.solver = solver
        self.solver_kwargs = solver_kwargs
        self._previous_input = np.zeros(self.B.shape[1])

    def control(self, x: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        """Return the first control input ``u0`` for the given state.

        ``x_ref`` has shape ``(horizon + 1, nx)`` and covers reference steps ``0..N``.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        x_ref = np.asarray(x_ref, dtype=float)
        if x_ref.shape[0] != self.horizon + 1:
            msg = f"x_ref must have {self.horizon + 1} rows, got {x_ref.shape[0]}"
            raise ValueError(msg)

        qp = condense_qp(
            self.A,
            self.B,
            self.Q,
            self.R,
            x,
            x_ref[1:],
            self.u_min,
            self.u_max,
        )
        z = self._solve(qp)
        u0 = np.asarray(z[: self.B.shape[1]])
        self._previous_input = u0
        return u0

    def _solve(self, qp: dict[str, Any]) -> np.ndarray:
        if self.solver == "pgd":
            return solve_box_qp_pgd(
                qp["H"],
                qp["h"],
                qp["lb"],
                qp["ub"],
                **self.solver_kwargs,
            )
        if self.solver == "gurobi":
            return solve_qp_gurobi(qp["H"], qp["h"], lb=qp["lb"], ub=qp["ub"])
        msg = f"unknown solver: {self.solver}"
        raise ValueError(msg)
