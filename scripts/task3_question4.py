"""Port of ``docs/Task3_Question4.m`` (input + state constrained path following)."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from controllers.mpc import MPCController  # noqa: E402
from models.double_integrator import DoubleIntegrator  # noqa: E402
from simulation.simulator import Simulator  # noqa: E402
from simulation.track import state_constraints, task_track  # noqa: E402


def main() -> None:
    inner, outer, centre = task_track()

    ni = 2
    ts = 2.01

    A = np.array(
        [
            [1.0, 0.0, ts, 0.0],
            [0.0, 1.0, 0.0, ts],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    B = np.array(
        [
            [ts**2 / 2, 0.0],
            [0.0, ts**2 / 2],
            [ts, 0.0],
            [0.0, ts],
        ]
    )
    Q = np.diag([1.0, 1.0, 0.0, 0.0])
    R = 2.2e3 * np.eye(ni)
    u_min = np.array([-1.0, -1.0])
    u_max = np.array([1.0, 1.0])

    # Task 3 reference: centre positions plus forward-difference velocities.
    next_centre = np.roll(centre, -1, axis=0)
    x_ref = np.column_stack([centre, (next_centre - centre) / ts])

    C, d_min, d_max = state_constraints(inner, outer)

    model = DoubleIntegrator(sampling_time=ts)
    controller = MPCController(
        A, B, horizon=10, Q=Q, R=R, u_min=u_min, u_max=u_max, solver="ineq_pgd"
    )
    simulator = Simulator(model, controller, noise_std=0.01, rng=0)

    states, _ = simulator.run(x_ref[0], x_ref, wrap=True, state_constraints=(C, d_min, d_max))

    plt.figure()
    plt.plot(inner[:, 0], inner[:, 1], "-", color="green", label="Augmented boundary")
    plt.plot(outer[:, 0], outer[:, 1], "-", color="green")
    plt.plot(centre[:, 0], centre[:, 1], ".", color="red", label="Centre path")
    plt.plot(states[:, 0], states[:, 1], "-o", label="Vehicle")
    plt.xlabel("p1")
    plt.ylabel("p2")
    plt.axis("equal")
    plt.legend()
    plt.title("Task 3 - Path following with input and state constraints")
    plt.show()


if __name__ == "__main__":
    main()
