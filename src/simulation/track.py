"""Track generation, ported from the MATLAB driving-scenario setup."""

from __future__ import annotations

import numpy as np


def build_track(
    waypoints: np.ndarray,
    width: float,
    step: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the inner/outer boundaries and centre path of a closed road.

    Mirrors the MATLAB ``road``/``roadBoundaries`` setup: the centre line follows the
    waypoints and the two boundaries are offset by ``width / 2`` on each side. The
    returned centre path is ``0.5 * (inner + outer)``, exactly as in the MATLAB script.

    Returns ``(inner, outer, centre)`` as ``(N, 2)`` arrays.
    """
    waypoints = np.asarray(waypoints, dtype=float)
    if waypoints.ndim != 2 or waypoints.shape[1] != 2:
        msg = "waypoints must be an (M, 2) array"
        raise ValueError(msg)

    points = waypoints[:-1] if np.allclose(waypoints[0], waypoints[-1]) else waypoints
    if len(points) < 3:
        msg = "at least three distinct waypoints are required"
        raise ValueError(msg)

    centre_points: list[np.ndarray] = []
    for i in range(len(points)):
        start = points[i]
        end = points[(i + 1) % len(points)]
        length = float(np.linalg.norm(end - start))
        count = max(1, int(np.ceil(length / step)))
        for fraction in np.linspace(0.0, 1.0, count + 1)[:-1]:
            centre_points.append(start + fraction * (end - start))
    centre = np.asarray(centre_points, dtype=float)

    half_width = 0.5 * width
    inner: list[np.ndarray] = []
    outer: list[np.ndarray] = []
    n = len(centre)

    for i in range(n):
        prev = centre[(i - 1) % n]
        nxt = centre[(i + 1) % n]

        tangent_in = centre[i] - prev
        tangent_in /= np.linalg.norm(tangent_in)
        tangent_out = nxt - centre[i]
        tangent_out /= np.linalg.norm(tangent_out)

        tangent = tangent_in + tangent_out
        tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])

        inner.append(centre[i] - half_width * normal)
        outer.append(centre[i] + half_width * normal)

    inner = np.asarray(inner)
    outer = np.asarray(outer)
    return inner, outer, 0.5 * (inner + outer)


def task_track() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(inner, outer, centre)`` for the assignment's example track."""
    x1_wp = [15.0, 25.0, -15.0, -15.0, -15.0, 15.0]
    x2_wp = [0.1, 25.0, 25.0, -0.1, -25.0, 0.1]
    waypoints = np.column_stack([x1_wp, x2_wp])

    # pathWidth = 10 in the MATLAB script: the augmented boundary with safety margin.
    return build_track(waypoints, width=10.0)


def state_constraints(
    inner: np.ndarray,
    outer: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(C, d_min, d_max)`` for boundary-crossing avoidance.

    Each row of ``C`` is the inward normal of the track at that point, so
    ``d_min <= C x <= d_max`` keeps the position between the two tangent boundary
    lines. This is equivalent to the MATLAB ``c/dMin/dMax`` construction but avoids
    division by zero on vertical boundary segments.
    """
    inner = np.asarray(inner, dtype=float)
    outer = np.asarray(outer, dtype=float)
    if inner.shape != outer.shape or inner.shape[1] != 2:
        msg = "inner and outer must both be (N, 2) arrays"
        raise ValueError(msg)

    normal = outer - inner
    n = normal.shape[0]
    C = np.zeros((n, 4))
    C[:, :2] = normal

    d_inner = np.einsum("ij,ij->i", normal, inner)
    d_outer = np.einsum("ij,ij->i", normal, outer)
    d_min = np.minimum(d_inner, d_outer)
    d_max = np.maximum(d_inner, d_outer)
    return C, d_min, d_max
