import numpy as np

from controllers.mpc import MPCController
from models.double_integrator import DoubleIntegrator
from simulation.simulator import Simulator


def test_simulator_produces_expected_shapes():
    model = DoubleIntegrator(sampling_time=0.1)
    controller = MPCController(
        model.A,
        model.B,
        horizon=5,
        Q=np.eye(4),
        R=np.eye(2),
        solver="pgd",
    )
    reference = np.zeros((20, 4))
    simulator = Simulator(model, controller, noise_std=0.0, rng=0)

    states, controls = simulator.run(x0=np.zeros(4), reference=reference)

    # n_steps defaults to 20 - 5 = 15, so states has 16 rows and controls has 15.
    assert states.shape == (16, 4)
    assert controls.shape == (15, 2)
