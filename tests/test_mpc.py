import numpy as np

from controllers.mpc import MPCController
from models.double_integrator import DoubleIntegrator


def test_controller_returns_first_input_within_bounds():
    model = DoubleIntegrator(sampling_time=0.1)
    horizon = 5
    u_min = np.array([-1.0, -1.0])
    u_max = np.array([1.0, 1.0])

    controller = MPCController(
        model.A,
        model.B,
        horizon,
        Q=np.eye(4),
        R=np.eye(2),
        u_min=u_min,
        u_max=u_max,
        solver="pgd",
    )

    x = np.array([0.1, 0.0, 0.0, 0.0])
    reference = np.zeros((horizon + 1, 4))
    u0 = controller.control(x, reference)

    assert u0.shape == (2,)
    assert np.all(u0 >= u_min - 1e-9)
    assert np.all(u0 <= u_max + 1e-9)
