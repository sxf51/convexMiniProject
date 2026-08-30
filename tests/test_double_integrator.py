import numpy as np
from pytest import approx

from models.double_integrator import DoubleIntegrator


def test_matrices_have_expected_shapes():
    model = DoubleIntegrator(sampling_time=0.1)
    assert model.A.shape == (4, 4)
    assert model.B.shape == (4, 2)


def test_step_applies_dynamics_and_noise():
    model = DoubleIntegrator(sampling_time=0.2)
    x = np.array([0.0, 0.0, 1.0, 0.0])
    u = np.array([1.0, 0.0])
    w = np.array([0.0, 0.0, 0.0, 0.0])

    x_next = model.step(x, u, w)

    # p1 += v1 * Ts + 0.5 * a1 * Ts^2, v1 += a1 * Ts
    assert x_next[0] == approx(1.0 * 0.2 + 0.5 * 1.0 * 0.2**2)
    assert x_next[2] == approx(1.0 + 1.0 * 0.2)
