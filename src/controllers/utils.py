"""Helper functions for the MPC controllers."""

from __future__ import annotations

from typing import Any

import numpy as np


def condense_qp(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    x0: np.ndarray,
    x_ref: np.ndarray,
    u_min: np.ndarray | None = None,
    u_max: np.ndarray | None = None,
    C: np.ndarray | None = None,
    d_min: np.ndarray | None = None,
    d_max: np.ndarray | None = None,
) -> dict[str, Any]:
    """Condense the MPC problem into a QP in ``z = [u_0; ...; u_{N-1}]``.

    ``x_ref`` has shape ``(N, nx)`` and contains the reference states for time steps
    ``1..N``. The current state ``x0`` is fixed and is not a decision variable.

    Optional state constraints ``d_min <= C x <= d_max`` are converted to ``G z <= g``.
    ``C`` has shape ``(N, nx)`` and ``d_min``/``d_max`` have shape ``(N,)``.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    x0 = np.asarray(x0, dtype=float).reshape(-1)

    nx = A.shape[0]
    nu = B.shape[1]
    x_ref = np.asarray(x_ref, dtype=float)
    if x_ref.shape[1] != nx:
        msg = f"x_ref must have {nx} columns, got {x_ref.shape[1]}"
        raise ValueError(msg)
    horizon = x_ref.shape[0]

    # A_bar = [A; A^2; ...; A^N]
    a_pow = np.eye(nx)
    a_rows = []
    for _ in range(horizon):
        a_pow = a_pow @ A
        a_rows.append(a_pow)
    a_bar = np.vstack(a_rows)

    # B_bar maps z to [x_1; ...; x_N] with x_k = A^k x0 + sum_j A^{k-j} B u_{j-1}.
    b_bar = np.zeros((horizon * nx, horizon * nu))
    for k in range(1, horizon + 1):
        for j in range(1, k + 1):
            block = np.linalg.matrix_power(A, k - j) @ B
            b_bar[(k - 1) * nx : k * nx, (j - 1) * nu : j * nu] = block

    q_bar = np.kron(np.eye(horizon), Q)
    r_bar = np.kron(np.eye(horizon), R)

    error = a_bar @ x0 - x_ref.reshape(-1)

    H = 2.0 * (b_bar.T @ q_bar @ b_bar + r_bar)
    h = 2.0 * (b_bar.T @ (q_bar @ error))

    lb = None
    ub = None
    if u_min is not None:
        lb = np.tile(np.asarray(u_min, dtype=float).reshape(-1), horizon)
    if u_max is not None:
        ub = np.tile(np.asarray(u_max, dtype=float).reshape(-1), horizon)

    G = None
    g = None
    if C is not None:
        C = np.asarray(C, dtype=float)
        d_min = np.asarray(d_min, dtype=float).reshape(-1)
        d_max = np.asarray(d_max, dtype=float).reshape(-1)
        if C.shape != (horizon, nx):
            msg = f"C must have shape ({horizon}, {nx}), got {C.shape}"
            raise ValueError(msg)

        c_bar = np.zeros((horizon, horizon * nx))
        for i in range(horizon):
            c_bar[i, i * nx : (i + 1) * nx] = C[i]

        c_dyn = c_bar @ b_bar
        c_const = c_bar @ (a_bar @ x0)

        G = np.vstack([c_dyn, -c_dyn])
        g = np.concatenate([d_max - c_const, -(d_min - c_const)])

    return {"H": H, "h": h, "lb": lb, "ub": ub, "G": G, "g": g}


def solve_box_qp_pgd(
    H: np.ndarray,
    h: np.ndarray,
    lb: np.ndarray | None = None,
    ub: np.ndarray | None = None,
    x_init: np.ndarray | None = None,
    max_iter: int = 20_000,
    tol: float = 1e-10,
    step: float | None = None,
) -> np.ndarray:
    """Solve ``min 0.5 z^T H z + h^T z`` subject to ``lb <= z <= ub``.

    Uses projected gradient descent. ``H`` is positive definite for this MPC problem,
    so the constant step ``1 / lambda_max(H)`` gives linear convergence.
    """
    H = np.asarray(H, dtype=float)
    h = np.asarray(h, dtype=float).reshape(-1)
    n = H.shape[0]

    lb = np.full(n, -np.inf) if lb is None else np.broadcast_to(np.asarray(lb, dtype=float), (n,))
    ub = np.full(n, np.inf) if ub is None else np.broadcast_to(np.asarray(ub, dtype=float), (n,))

    if x_init is None:
        z = np.clip(np.zeros(n), lb, ub)
    else:
        z = np.clip(np.asarray(x_init, dtype=float).reshape(-1), lb, ub)

    if step is None:
        largest_eig = float(np.linalg.eigvalsh(H).max())
        step = 1.0 / largest_eig if largest_eig > 0 else 1.0

    for _ in range(max_iter):
        gradient = H @ z + h
        z_next = np.clip(z - step * gradient, lb, ub)
        if np.linalg.norm(z_next - z) <= tol:
            z = z_next
            break
        z = z_next
    return z


def solve_qp_ineq_pgd(
    H: np.ndarray,
    h: np.ndarray,
    G: np.ndarray | None = None,
    g: np.ndarray | None = None,
    lb: np.ndarray | None = None,
    ub: np.ndarray | None = None,
    x_init: np.ndarray | None = None,
    max_iter: int = 20_000,
    tol: float = 1e-8,
    step: float | None = None,
) -> np.ndarray:
    """Solve ``min 0.5 z^T H z + h^T z`` with ``G z <= g`` and box constraints.

    This is a projected-gradient ascent on the dual problem, which keeps the projection
    trivial (``lambda >= 0``). Box constraints are appended to ``G z <= g`` before the
    dual is formed. ``x_init`` is accepted for interface compatibility but ignored; the
    dual multipliers always start at zero.
    """
    del x_init

    H = np.asarray(H, dtype=float)
    h = np.asarray(h, dtype=float).reshape(-1)
    n = H.shape[0]

    rows_G: list[np.ndarray] = []
    rows_g: list[np.ndarray] = []
    if G is not None and G.size:
        rows_G.append(np.asarray(G, dtype=float))
        rows_g.append(np.asarray(g, dtype=float).reshape(-1))
    if lb is not None:
        lb = np.asarray(lb, dtype=float)
        rows_G.append(-np.eye(n))
        rows_g.append(-lb)
    if ub is not None:
        ub = np.asarray(ub, dtype=float)
        rows_G.append(np.eye(n))
        rows_g.append(ub)

    if not rows_G:
        return solve_box_qp_pgd(H, h, lb, ub, max_iter=max_iter, tol=tol, step=step)

    G_aug = np.vstack(rows_G)
    g_aug = np.concatenate(rows_g)

    H_inv = np.linalg.inv(H)
    dual_hessian = G_aug @ H_inv @ G_aug.T
    largest_eig = float(np.linalg.eigvalsh(dual_hessian).max())

    if step is None:
        step = 1.0 / largest_eig if largest_eig > 1e-12 else 1.0

    lam = np.zeros(G_aug.shape[0])
    for _ in range(max_iter):
        z = -H_inv @ (h + G_aug.T @ lam)
        gradient = G_aug @ z - g_aug
        lam_next = np.maximum(0.0, lam + step * gradient)
        if np.linalg.norm(lam_next - lam) <= tol:
            lam = lam_next
            break
        lam = lam_next

    return -H_inv @ (h + G_aug.T @ lam)


def solve_qp_gurobi(
    H: np.ndarray,
    h: np.ndarray,
    G: np.ndarray | None = None,
    g: np.ndarray | None = None,
    lb: np.ndarray | None = None,
    ub: np.ndarray | None = None,
) -> np.ndarray:
    """Solve the QP with Gurobi (requires a valid Gurobi license)."""
    import gurobipy as gp

    H = np.asarray(H, dtype=float)
    h = np.asarray(h, dtype=float).reshape(-1)
    n = H.shape[0]

    model = gp.Model("condensed_qp")
    model.setParam("OutputFlag", 0)

    z = model.addMVar(n, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="z")
    if lb is not None:
        z.lb = np.asarray(lb, dtype=float)
    if ub is not None:
        z.ub = np.asarray(ub, dtype=float)

    objective = 0.5 * (z @ H @ z) + gp.quicksum(h[i] * z[i] for i in range(n))
    model.setObjective(objective, gp.GRB.MINIMIZE)

    if G is not None and g is not None:
        model.addMConstr(
            np.asarray(G, dtype=float), z, gp.GRB.LESS_EQUAL, np.asarray(g, dtype=float)
        )

    model.optimize()
    if model.Status != gp.GRB.OPTIMAL:
        msg = f"Gurobi did not reach optimality (status {model.Status})"
        raise RuntimeError(msg)
    return np.asarray(z.X, dtype=float)


def circular_track_reference(
    radius: float = 1.0,
    angular_velocity: float = 0.2,
    n_steps: int = 100,
    sampling_time: float = 0.1,
    centre: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Build a reference trajectory around a circle.

    Returns an ``(n_steps, 4)`` array whose columns are ``[p1, p2, v1, v2]``.
    """
    times = np.arange(n_steps) * sampling_time
    angles = angular_velocity * times
    px = centre[0] + radius * np.cos(angles)
    py = centre[1] + radius * np.sin(angles)
    vx = -radius * angular_velocity * np.sin(angles)
    vy = radius * angular_velocity * np.cos(angles)
    return np.column_stack([px, py, vx, vy])


def plot_trajectory(
    reference: np.ndarray,
    states: np.ndarray,
    ax: Any | None = None,
) -> Any:
    """Plot the reference path and the simulated trajectory."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        msg = "matplotlib is required for plotting; install it with `uv add matplotlib`"
        raise ImportError(msg) from exc

    reference = np.asarray(reference)
    states = np.asarray(states)

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(reference[:, 0], reference[:, 1], "--", label="Reference")
    ax.plot(states[:, 0], states[:, 1], "-o", label="Vehicle")
    ax.set_xlabel("p1")
    ax.set_ylabel("p2")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    return ax
