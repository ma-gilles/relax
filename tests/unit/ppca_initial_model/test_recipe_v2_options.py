"""Recipe v2 options of ppca_initial_model: the fixed Newton step and the oversampling start / rotation-only children."""

import argparse
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from relax.commands.ppca_initial_model import add_args, resolve_sgd_options
from relax.ppca_initial_model import checkpoint
from relax.ppca_initial_model.config import Config
from relax.ppca_initial_model.iteration_loop import oversampled_update
from relax.ppca_initial_model.sgd_recipe import NewtonSGD, damped_newton_direction

pytestmark = pytest.mark.unit


def _pack(matrix):
    i, j = np.triu_indices(matrix.shape[-1])
    return matrix[..., i, j]


def _stats(rng, n=8, p=3):
    a = rng.standard_normal((n, p, p)).astype(np.float32)
    h = np.einsum("fij,fkj->fik", a, a) + 0.5 * np.eye(p, dtype=np.float32)
    g = (rng.standard_normal((n, p)) + 1j * rng.standard_normal((n, p))).astype(np.complex64)
    return [SimpleNamespace(lhs_tri=_pack(h), residual_gradient=g)], h, g


def _config(**kw):
    base = dict(optimizer="momentum_sgd", sgd_momentum=0.0, sgd_direction="coupled")
    return Config(**{**base, **kw})


def test_fixed_step_is_t_times_the_schedule_shape_times_the_newton_direction(tmp_path):
    steps = np.array([2.0, 2.0, 1.0, 0.5])
    np.save(tmp_path / "s.npy", steps)
    stats, h, g = _stats(np.random.default_rng(0))
    active, shells = np.ones(8, bool), np.repeat([0, 1], 4)
    direction, _ = damped_newton_direction(_pack(h), g, active, 1e-6)
    for rate in (0.3, 7.0):  # the learning rate cancels
        newton = NewtonSGD(_config(sgd_fixed_step=0.1, sgd_lr_schedule=str(tmp_path / "s.npy"), sgd_learning_rate=rate), 2)
        for t, shape in ((1, 1.0), (3, 0.5), (9, 0.25)):
            step, info = newton.step(stats, active, shells, t, 1e-6, 2)
            np.testing.assert_allclose(np.asarray(step), 0.1 * shape * np.asarray(direction), rtol=1e-5, atol=1e-7)
            assert info["fixed_step_used"] == pytest.approx(0.1 * shape)
        assert newton.k is None  # no update-1 norm match


def test_fixed_step_without_a_schedule_is_constant():
    stats, h, g = _stats(np.random.default_rng(1))
    active, shells = np.ones(8, bool), np.zeros(8, int)
    newton = NewtonSGD(_config(sgd_fixed_step=0.25), 1)
    direction, _ = damped_newton_direction(_pack(h), g, active, 1e-6)
    for t in (1, 50):
        step, _ = newton.step(stats, active, shells, t, 1e-6, 2)
        np.testing.assert_allclose(np.asarray(step), 0.25 * np.asarray(direction), rtol=1e-5, atol=1e-7)


def test_fixed_step_validation():
    with pytest.raises(ValueError, match="fixed step"):
        _config(sgd_fixed_step=0.1, sgd_direction="trace", sgd_pc_gate_floor=0.25)
    with pytest.raises(ValueError, match="fixed step"):
        _config(sgd_fixed_step=0.1, sgd_lr_match=1.95)
    with pytest.raises(ValueError, match="fixed step"):
        _config(sgd_fixed_step=-1.0)
    assert _config(sgd_fixed_step=0.1).newton_sgd()


def test_oversampling_starts_at_the_last_stage_or_the_given_update():
    stages = ((1, 11, 1), (126, 12, 2), (1626, 24, 3), (2500, 31, 3))
    last = Config(oversampling=1, stages=stages)
    assert not oversampled_update(last, 2499) and oversampled_update(last, 2500)
    early = Config(oversampling=1, stages=stages, oversampling_from_update=1626)
    assert not oversampled_update(early, 1625) and oversampled_update(early, 1626)
    assert not oversampled_update(Config(oversampling=0, stages=stages), 3000)
    with pytest.raises(ValueError, match="need --oversampling 1"):
        Config(oversampling=0, oversampling_from_update=1626)
    with pytest.raises(ValueError, match="need --oversampling 1"):
        Config(oversampling=0, oversampling_rotations_only=True)


def test_cli_maps_the_recipe_v2_options():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args(["--i", "p.star", "-o", "out", "--particle-diameter", "300", "--optimizer", "momentum_sgd", "--sgd-momentum", "0",
                              "--sgd-direction", "coupled", "--sgd-fixed-step", "0.1", "--oversampling", "1",
                              "--oversampling-from-update", "1626", "--oversampling-rotations-only"])
    assert resolve_sgd_options(args)["sgd_fixed_step"] == 0.1
    assert args.oversampling_from_update == 1626 and args.oversampling_rotations_only


def test_checkpoints_without_the_recipe_v2_fields_keep_their_identity():
    added = ("sgd_fixed_step", "oversampling_from_update", "oversampling_rotations_only")
    full = dataclasses.asdict(Config())
    old = {k: v for k, v in full.items() if k not in added}
    assert checkpoint._identity_config(old) == checkpoint._identity_config(full)
