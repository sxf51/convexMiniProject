"""Port of ``docs/Task2_Question3.m`` (input-constrained path following)."""

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
from simulation.track import task_track  # noqa: E402


def main() -> None:
    _, _, centre = task_track()

    ns = 4
    ni = 2
    ts = 2.01
    n_steps = centre.shape[0]

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

    # Task 2 reference: centre positions with zero reference velocity.
    x_ref = np.zeros((n_steps, ns))
    x_ref[:, :2] = centre

    model = DoubleIntegrator(sampling_time=ts)
    controller = MPCController(A, B, horizon=10, Q=Q, R=R, u_min=u_min, u_max=u_max, solver="pgd")
    simulator = Simulator(model, controller, noise_std=0.01, rng=0)

    states, _ = simulator.run(x_ref[0], x_ref, wrap=True)

    plt.figure()
    plt.plot(centre[:, 0], centre[:, 1], "--", label="Reference")
    plt.plot(states[:, 0], states[:, 1], "-o", label="Vehicle")
    plt.xlabel("p1")
    plt.ylabel("p2")
    plt.axis("equal")
    plt.legend()
    plt.title("Task 2 - Path following with input constraints")
    plt.show()


if __name__ == "__main__":
    main()
