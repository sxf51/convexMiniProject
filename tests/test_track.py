import numpy as np

from simulation.track import build_track, state_constraints, task_track


def test_task_track_returns_consistent_boundaries():
    inner, outer, centre = task_track()

    assert inner.shape == outer.shape == centre.shape
    assert inner.shape[1] == 2
    assert centre.shape[0] > 10
    assert np.allclose(centre, 0.5 * (inner + outer))


def test_state_constraints_have_expected_shapes_and_order():
    inner, outer, _ = task_track()
    C, d_min, d_max = state_constraints(inner, outer)

    assert C.shape == (inner.shape[0], 4)
    assert d_min.shape == d_max.shape == (inner.shape[0],)
    assert np.all(d_min <= d_max)


def test_build_track_rejects_too_few_waypoints():
    try:
        build_track(np.array([[0.0, 0.0], [1.0, 1.0]]), width=10.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a degenerate track")
