import numpy as np

from controllers.mpc import MPCController
from controllers.utils import solve_qp_ineq_pgd
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


def test_solve_qp_ineq_pgd_respects_constraints():
    H = np.array([[2.0, 0.0], [0.0, 2.0]])
    h = np.array([-2.0, -2.0])
    G = np.array([[1.0, 0.0]])
    g = np.array([0.5])
    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])

    z = solve_qp_ineq_pgd(H, h, G, g, lb, ub)

    assert G @ z <= g + 1e-6
    assert np.all(z >= lb - 1e-6)
    assert np.all(z <= ub + 1e-6)
