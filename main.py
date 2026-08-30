"""Entry point for the MPC mini-project."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from controllers.mpc import MPCController  # noqa: E402
from controllers.utils import circular_track_reference  # noqa: E402
from models.double_integrator import DoubleIntegrator  # noqa: E402
from simulation.simulator import Simulator  # noqa: E402


def run_demo() -> None:
    import numpy as np

    model = DoubleIntegrator(sampling_time=0.1)
    horizon = 10
    q = np.diag([10.0, 10.0, 1.0, 1.0])
    r = np.diag([0.1, 0.1])

    controller = MPCController(
        model.A,
        model.B,
        horizon,
        q,
        r,
        u_min=np.array([-1.0, -1.0]),
        u_max=np.array([1.0, 1.0]),
        solver="pgd",
    )

    reference = circular_track_reference(
        radius=1.0,
        angular_velocity=0.2,
        n_steps=120,
        sampling_time=model.sampling_time,
    )
    simulator = Simulator(model, controller, noise_std=0.001, rng=0)

    states, controls = simulator.run(x0=reference[0], reference=reference)

    print(f"Simulated {len(states) - 1} steps")
    print(f"Final position: {states[-1, 0]:.3f}, {states[-1, 1]:.3f}")
    print(f"Controls applied: {controls.shape[0]}")


if __name__ == "__main__":
    run_demo()
