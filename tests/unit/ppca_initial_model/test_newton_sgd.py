"""Newton SGD (coupled direction, PC gate, learning-rate schedule) and the translation options: CPU contracts."""

import argparse
import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from relax.commands.ppca_initial_model import add_args, resolve_sgd_options, translation_options
from relax.ppca_initial_model.config import Config
from relax.ppca_initial_model.sgd_recipe import (
    LearningRateSchedule,
    NewtonSGD,
    PCGate,
    damped_newton_direction,
    split_curvature_step,
)

pytestmark = pytest.mark.unit


def _pack(matrix):
    i, j = np.triu_indices(matrix.shape[-1])
    return matrix[..., i, j]


def _spd(rng, n, p):
    a = rng.standard_normal((n, p, p)).astype(np.float32)
    return np.einsum("fij,fkj->fik", a, a) + 0.5 * np.eye(p, dtype=np.float32)


def _gradient(rng, n, p):
    return (rng.standard_normal((n, p)) + 1j * rng.standard_normal((n, p))).astype(np.complex64)


def test_newton_direction_solves_each_frequency_and_skips_inactive_rows():
    rng = np.random.default_rng(0)
    h, g = _spd(rng, 6, 3), _gradient(rng, 6, 3)
    active = np.array([1, 1, 0, 1, 1, 0], bool)
    direction, info = damped_newton_direction(_pack(h), g, active, 1e-6)
    want = np.where(active[:, None], np.linalg.solve(h.astype(np.float64), g.astype(np.complex128)[..., None])[..., 0], 0)
    np.testing.assert_allclose(np.asarray(direction), want, rtol=2e-4, atol=1e-5)
    assert info["indefinite_frequencies"] == 0 and info["max_damping"] == 0


def test_an_indefinite_metric_is_damped_not_refused():
    h = np.diag(np.array([2.0, 1.0, -0.5], np.float32))[None]
    g = np.ones((1, 3), np.complex64)
    direction, info = damped_newton_direction(_pack(h), g, np.ones(1, bool), 0.01)
    shift = 0.01 + 0.5  # floor minus the smallest eigenvalue, added to every eigenvalue
    np.testing.assert_allclose(np.asarray(direction)[0], 1 / np.maximum(np.diag(h[0]) + shift, 0.01), rtol=1e-5)
    assert info["indefinite_frequencies"] == 1 and info["max_damping"] == pytest.approx(shift)


def test_split_step_divides_the_mean_and_the_pcs_by_their_own_band_maxima():
    diagonal = np.array([[4.0, 1.0, 3.0], [2.0, 5.0, 1.0], [9.0, 9.0, 9.0]], np.float32)
    g = np.ones((3, 3), np.complex64)
    step, c_mean, c_pc = split_curvature_step(g, diagonal, np.array([1, 1, 0], bool), 0.5, 1e-6)
    assert c_mean == 4.0 and c_pc == 3.0  # the inactive row's 9s do not count
    np.testing.assert_allclose(np.asarray(step)[:2], [[0.5 / 4, 0.5 / 3, 0.5 / 3]] * 2, rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(step)[2], 0)


def test_pc_gate_is_the_moving_half_batch_correlation_clipped_at_the_floor():
    rng = np.random.default_rng(1)
    shells = np.array([0, 0, 1, 1])
    gate = PCGate(0.25, 2)
    signal = _gradient(rng, 4, 3)
    for _ in range(200):  # agreeing halves in shell 0, independent noise in shell 1
        noise0, noise1 = _gradient(rng, 4, 3), _gradient(rng, 4, 3)
        g0 = np.where(shells[:, None] == 0, signal, noise0)
        g1 = np.where(shells[:, None] == 0, signal, noise1)
        gate.update([g0, g1], shells)
    np.testing.assert_allclose(np.asarray(gate.factors())[0], 1.0, rtol=1e-5)
    assert np.asarray(gate.factors())[1] == pytest.approx(0.25)
    assert abs(float(np.asarray(gate.raw())[1])) < 0.25
    restored = PCGate(0.25, 2, gate.state())
    np.testing.assert_allclose(np.asarray(restored.raw()), np.asarray(gate.raw()))


def test_pc_gate_ignores_the_mean_channel():
    shells = np.array([0, 0])
    g0 = np.array([[5, 1, 1], [-5, 1, 1]], np.complex64)
    g1 = np.array([[-5, 1, 1], [5, 1, 1]], np.complex64)  # the means disagree, the PCs agree
    assert float(np.asarray(PCGate(0.25, 1).update([g0, g1], shells))[0]) == pytest.approx(1.0)


def test_learning_rate_schedule_follows_the_file_relative_to_its_first_step(tmp_path):
    np.save(tmp_path / "s.npy", np.array([0.5, 0.5, 0.25, 0.1]))
    lr = LearningRateSchedule(0.7, tmp_path / "s.npy")
    assert [lr(t) for t in (1, 2, 3, 4, 9)] == pytest.approx([0.7, 0.7, 0.35, 0.14, 0.14])
    assert LearningRateSchedule(0.7, None)(1234) == 0.7


def _config(**kw):
    base = dict(optimizer="momentum_sgd", sgd_momentum=0.0, sgd_direction="coupled", sgd_pc_gate_floor=0.25)
    return Config(**{**base, **kw})


def test_newton_step_matches_the_split_norm_at_update_one_and_keeps_k(tmp_path):
    rng = np.random.default_rng(2)
    h = [_spd(rng, 8, 3) for _ in range(2)]
    stats = [SimpleNamespace(lhs_tri=_pack(m), residual_gradient=_gradient(rng, 8, 3)) for m in h]
    active, shells = np.ones(8, bool), np.repeat([0, 1], 4)
    newton = NewtonSGD(_config(sgd_pc_gate_floor=None), 2)
    step, info = newton.step(stats, active, shells, 1, 1e-6, 2)
    split, _, _ = split_curvature_step(
        sum(s.residual_gradient for s in stats), np.diagonal(sum(h), axis1=1, axis2=2), active, 0.4, 1e-6
    )
    assert float(jnp.linalg.norm(step)) == pytest.approx(float(jnp.linalg.norm(split)), rel=1e-4)
    k = newton.k
    newton.step(stats, active, shells, 2, 1e-6, 2)
    assert newton.k == k
    newton.save(tmp_path / "n.npz")
    assert NewtonSGD(_config(sgd_pc_gate_floor=None), 2, NewtonSGD.load(tmp_path / "n.npz")).k == k


def test_config_refuses_newton_options_outside_plain_sgd():
    with pytest.raises(ValueError, match="momentum 0"):
        _config(sgd_momentum=0.9)
    with pytest.raises(ValueError, match="momentum 0"):
        Config(optimizer="vdam", sgd_pc_gate_floor=0.25)
    with pytest.raises(ValueError, match="floor"):
        _config(sgd_pc_gate_floor=0.0)
    with pytest.raises(ValueError, match="shift schedule"):
        Config(shift_schedule=((5, 8, 2),))


def test_shift_schedule_and_held_sigma_from_angstrom():
    args = SimpleNamespace(shift_schedule=json.dumps([[1, 48, 12], [1626, 48, 6]]), offset_sigma_hold=20.0)
    options = translation_options(args, 6.0)
    assert options == {"shift_schedule": ((1, 8.0, 2.0), (1626, 8.0, 1.0)), "offset_sigma_hold_px": 20 / 6}
    config = Config(**options)
    assert config.shifts(1) == (8.0, 2.0) and config.shifts(1625) == (8.0, 2.0) and config.shifts(1626) == (8.0, 1.0)
    assert Config().shifts(7) == (Config().shift_range, Config().shift_step)
    # Steps round to whole pixels and the window to whole steps: 20 A is 2 px, 55 A is 2.75 steps, so 3 steps.
    assert translation_options(SimpleNamespace(shift_schedule="[[1, 55, 20]]", offset_sigma_hold=None), 10.0) == {
        "shift_schedule": ((1, 6.0, 2.0),)
    }


def test_cli_maps_the_recipe_options():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args(
        ["--i", "p.star", "--particle-diameter", "100", "-o", "out", "--optimizer", "momentum_sgd",
         "--sgd-momentum", "0", "--sgd-direction", "coupled", "--sgd-pc-gate-floor", "0.25",
         "--uniform-direction-prior"]
    )
    options = resolve_sgd_options(args)
    assert options["sgd_direction"] == "coupled" and options["sgd_pc_gate_floor"] == 0.25
    assert args.uniform_direction_prior


def test_matched_rate_makes_the_mean_step_equal_the_shared_trace_step(tmp_path):
    rng = np.random.default_rng(3)
    h = [_spd(rng, 8, 3) for _ in range(2)]
    stats = [SimpleNamespace(lhs_tri=_pack(m), residual_gradient=_gradient(rng, 8, 3)) for m in h]
    active, shells = np.ones(8, bool), np.repeat([0, 1], 4)
    newton = NewtonSGD(_config(sgd_direction="trace", sgd_pc_gate_floor=None, sgd_lr_match=1.95), 2)
    step, info = newton.step(stats, active, shells, 1, 1e-6, 2)
    g = sum(s.residual_gradient for s in stats)
    trace_max = np.max(np.trace(sum(h), axis1=1, axis2=2))
    np.testing.assert_allclose(np.asarray(step)[:, 0], 1.95 * g[:, 0] / trace_max, rtol=1e-4)
    newton.save(tmp_path / "n.npz")
    again = NewtonSGD(_config(sgd_direction="trace", sgd_pc_gate_floor=None, sgd_lr_match=1.95), 2, NewtonSGD.load(tmp_path / "n.npz"))
    assert again.lr.base == pytest.approx(info["learning_rate_base"])


def test_relative_floor_bounds_every_eigenvalue_by_the_band_maximum():
    h = np.stack([np.diag(np.array([100.0, 1e-6, 1e-6], np.float32)), np.diag(np.array([4.0, 2.0, 1.0], np.float32))])
    g = np.ones((2, 3), np.complex64)
    direction, info = damped_newton_direction(_pack(h), g, np.ones(2, bool), 1e-8, relative_floor=0.01)
    assert info["eigenvalue_floor"] == pytest.approx(1.0)  # 0.01 x the band's largest eigenvalue, 100
    np.testing.assert_allclose(np.asarray(direction).real, [[0.01, 1.0, 1.0], [0.25, 0.5, 1.0]], rtol=1e-5)
    plain, _ = damped_newton_direction(_pack(h), g, np.ones(2, bool), 1e-8)
    assert np.asarray(plain).real[0, 1] > 1e5  # without it the near-zero curvature gives a huge step


def test_pc_min_shell_leaves_low_shells_of_the_pcs_unchanged():
    rng = np.random.default_rng(3)
    n, q = 8, 2
    shells = np.array([0, 1, 1, 2, 2, 3, 3, 3])
    stats = [SimpleNamespace(lhs_tri=_pack(_spd(rng, n, q + 1)), residual_gradient=_gradient(rng, n, q + 1)) for _ in range(2)]
    step, _ = NewtonSGD(_config(sgd_pc_min_shell=2), 4).step(stats, np.ones(n, bool), shells, 1, 1e-6, q)
    step = np.asarray(step)
    assert not np.any(step[shells < 2, 1:]) and np.all(step[shells >= 2, 1:] != 0) and np.all(step[:, 0] != 0)
    with pytest.raises(ValueError):
        Config(optimizer="momentum_sgd", sgd_momentum=0.0, sgd_pc_min_shell=1)  # needs the coupled direction
