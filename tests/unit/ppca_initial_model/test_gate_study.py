"""Gate study options of stochastic_update: the oracle table in place of the gate and the second-moment summaries."""

import jax.numpy as jnp
import numpy as np
import pytest

from relax.ppca_initial_model.update import FIRST_MOMENT_DECAY, empty_moments, stochastic_update

pytestmark = pytest.mark.unit

BOX = 8


def _inputs(seed=0, n=40, q=2):
    rng = np.random.default_rng(seed)
    theta = (rng.standard_normal((n, q + 1)) + 1j * rng.standard_normal((n, q + 1))).astype(np.complex64)
    directions = (rng.standard_normal((2, n, q + 1)) + 1j * rng.standard_normal((2, n, q + 1))).astype(np.complex64)
    shells = (np.arange(n) % 4).astype(np.int32)
    return jnp.asarray(theta), jnp.asarray(directions), np.ones((2, n), bool), shells


def _update(theta, directions, coverage, shells, moments=None, **kwargs):
    return stochastic_update(
        theta,
        empty_moments(theta) if moments is None else moments,
        directions,
        coverage,
        shells,
        step=0.3,
        fudge=1.0,
        image_size=BOX,
        disagreement_scale=0.5,
        **kwargs,
    )


def test_the_oracle_table_replaces_the_gate_and_the_estimate_is_still_reported():
    theta, directions, coverage, shells = _inputs()
    table = np.asarray([[1.0, 0.9], [0.8, 0.5], [0.3, 0.1], [0.0, 0.0]])
    out, moments, info = _update(theta, directions, coverage, shells, gate_override=table)
    _, _, default = _update(theta, directions, coverage, shells)
    assert np.array_equal(info["gates"], table.astype(np.float32))
    assert np.allclose(info["gates_estimated"], default["gates"])
    # The update with that gate: theta + step (phi * mean first / sqrt(u) - (1 - phi) theta); first call: u = 0.999 +
    # 0.001 ratio, first = directions.
    phi = np.concatenate([table[shells][:, :1], np.repeat(table[shells][:, 1:], 2, axis=1)], axis=1)
    mean_first = np.asarray(directions).mean(0)
    divisor = np.sqrt(np.asarray(moments.second))[:, [0, 1, 1]] + 1e-12
    expected = np.asarray(theta) + 0.3 * (phi * mean_first / divisor - (1 - phi) * np.asarray(theta))
    assert np.allclose(np.asarray(out), expected, rtol=1e-5, atol=1e-6)


def test_a_table_with_fewer_shells_closes_the_missing_shells_and_bad_tables_are_refused():
    theta, directions, coverage, shells = _inputs()
    _, _, info = _update(theta, directions, coverage, shells, gate_override=np.ones((2, 2)))
    assert np.array_equal(info["gates"], [[1, 1], [1, 1], [0, 0], [0, 0]])
    with pytest.raises(ValueError):
        _update(theta, directions, coverage, shells, gate_override=np.full((4, 2), 1.5))
    with pytest.raises(ValueError):
        _update(theta, directions, coverage, shells, gate_override=np.ones((4, 3)))


def test_default_arguments_leave_the_update_unchanged():
    theta, directions, coverage, shells = _inputs(seed=3)
    a = _update(theta, directions, coverage, shells)
    b = _update(theta, directions, coverage, shells, gate_override=None, log_moments=True)
    assert np.array_equal(np.asarray(a[0]), np.asarray(b[0]))
    assert "sqrt_u_mean" not in a[2] and "sqrt_u_mean" in b[2]


def test_the_effective_gate_is_the_fixed_point_weight_of_the_update():
    """With constant directions d_h = target - theta... the fixed point of theta <- theta + eta (phi m / sqrt(u) -
    (1 - phi) theta) with m = target - theta is theta* = phi_eff target, phi_eff = phi / (phi + (1 - phi) sqrt(u))."""
    n = 8
    shells = np.zeros(n, np.int32)
    target = jnp.asarray(np.full((n, 2), 2.0 + 1.0j, np.complex64))
    theta = jnp.zeros((n, 2), jnp.complex64)
    moments = empty_moments(theta)
    table = np.asarray([[0.6, 0.6]])
    for _ in range(4000):
        noise = np.zeros((2, n, 2), np.complex64)
        noise[0] += 0.5
        noise[1] -= 0.5  # fixed half disagreement: u settles at |d1 - d0|^2 / |d_bar|^2
        directions = jnp.asarray(np.asarray(target - theta)[None] + noise)
        theta, moments, info = _update(
            theta, directions, np.ones((2, n), bool), shells, moments=moments, gate_override=table, log_moments=True
        )
    effective = float(info["effective_gate_mean"][0, 0])
    assert abs(float(np.real(theta[0, 0])) / 2.0 - effective) < 2e-3
    assert FIRST_MOMENT_DECAY == 0.9
