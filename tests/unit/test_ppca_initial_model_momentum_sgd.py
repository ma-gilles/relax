"""Independent CPU contracts for PPCA's opt-in augmented momentum step."""

import argparse
import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches, assert_trees_match
from recovar.core import fourier_transform_utils as ftu

from relax.commands.ppca_initial_model import add_args
from relax.ppca_initial_model import iteration_loop
from relax.ppca_initial_model.checkpoint import load, save
from relax.ppca_initial_model.config import Config
from relax.ppca_initial_model.initialization import bandlimit_and_mask, support_mask
from relax.ppca_initial_model.sgd_update import momentum_step
from relax.ppca_initial_model.state import State
from relax.ppca_refinement.residual_statistics import residual_image_statistics

pytestmark = pytest.mark.unit


def _pack(matrix):
    return matrix[..., np.triu_indices(matrix.shape[-1])[0], np.triu_indices(matrix.shape[-1])[1]]


def _loss(theta, hessian, rhs):
    return np.real(0.5 * np.einsum("fi,fij,fj->", theta.conj(), hessian, theta) - np.vdot(theta, rhs))


def test_augmented_quadratic_descent_and_duplicate_batch_invariance():
    h = np.array(
        [[[3, 0.4, -0.2], [0.4, 2, 0.3], [-0.2, 0.3, 1]], [[1.2, -0.1, 0.2], [-0.1, 1.5, 0.1], [0.2, 0.1, 1.1]]],
        dtype=np.float32,
    )
    theta = np.array([[0.1 + 0.2j, -0.2 + 0.1j, 0.3 - 0.1j], [0.2, 0.1j, -0.1]], np.complex64)
    target = np.array([[1, 0.5j, -0.6], [0.3, -0.4j, 0.2]], np.complex64)
    rhs = np.einsum("fij,fj->fi", h, target)
    gradient = rhs - np.einsum("fij,fj->fi", h, theta)
    packed = _pack(h)
    before = _loss(theta, h, rhs)
    updated, velocity, info = momentum_step(
        theta, np.zeros_like(theta), gradient, packed, np.ones(2, bool), learning_rate=0.4, floor=1e-7
    )
    expected = np.float32(0.04 / np.max(np.trace(h, axis1=-2, axis2=-1))) * gradient
    np.testing.assert_allclose(velocity, expected, rtol=1e-6, atol=1e-7)
    assert _loss(np.asarray(updated), h, rhs) < before
    assert info["curvature_trace_max"] == pytest.approx(float(np.max(np.trace(h, axis1=-2, axis2=-1))))
    duplicated, _, _ = momentum_step(
        theta, np.zeros_like(theta), 2 * gradient, 2 * packed, np.ones(2, bool), learning_rate=0.4, floor=1e-7
    )
    np.testing.assert_allclose(duplicated, updated, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("reflection", [False, True])
def test_latent_orthogonal_rotation_moves_model_and_momentum_together(reflection):
    rng = np.random.default_rng(4)
    a = rng.normal(size=(4, 3, 3)).astype(np.float32)
    h = np.einsum("fki,fkj->fij", a, a) + 0.3 * np.eye(3, dtype=np.float32)
    theta = (rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))).astype(np.complex64)
    old = (rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))).astype(np.complex64) * 0.01
    gradient = (rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))).astype(np.complex64)
    angle = 0.71
    q = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], np.float32)
    if reflection:
        q[:, 0] *= -1
    transform = np.eye(3, dtype=np.float32)
    transform[1:, 1:] = q
    active = np.array([True, True, False, True])
    updated, velocity, info = momentum_step(theta, old, gradient, _pack(h), active, learning_rate=0.4, floor=1e-7)
    rotated, rotated_velocity, rotated_info = momentum_step(
        theta @ transform,
        old @ transform,
        gradient @ transform,
        _pack(np.einsum("ip,fij,jq->fpq", transform, h, transform)),
        active,
        learning_rate=0.4,
        floor=1e-7,
    )
    np.testing.assert_allclose(rotated, np.asarray(updated) @ transform, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(rotated_velocity, np.asarray(velocity) @ transform, rtol=1e-6, atol=1e-6)
    assert rotated_info["curvature_scalar"] == pytest.approx(info["curvature_scalar"], rel=1e-6)
    np.testing.assert_allclose(np.asarray(velocity)[2], np.zeros(3), atol=1e-7)


def test_q0_units_zero_curvature_and_signed_step():
    theta = np.zeros((2, 1), np.complex64)
    gradient = np.array([[-12], [100]], np.complex64)
    curvature = np.array([[6], [0]], np.float32)
    result, velocity, info = momentum_step(
        theta, theta, gradient, curvature, np.array([True, False]), learning_rate=0.5, floor=1e-7
    )
    np.testing.assert_allclose(result[:, 0], [-0.1, 0], atol=1e-7)
    np.testing.assert_allclose(velocity[:, 0], [-0.1, 0], atol=1e-7)
    assert info["curvature_scalar"] == pytest.approx(6)
    no_data, no_velocity, _ = momentum_step(
        theta,
        np.ones_like(theta),
        gradient,
        np.zeros_like(curvature),
        np.ones(2, bool),
        learning_rate=0.5,
        floor=1e-7,
    )
    np.testing.assert_allclose(no_data, theta, atol=1e-7)
    np.testing.assert_allclose(no_velocity, theta, atol=1e-7)


def test_velocity_uses_the_same_fourier_band_as_model():
    shape = (8, 8, 8)
    n_freq = shape[0] * shape[1] * (shape[2] // 2 + 1)
    radii = np.asarray(ftu.get_grid_of_radial_distances_real(shape, rounded=False)).reshape(-1)
    gradient = jnp.ones((n_freq, 3), jnp.complex64)
    curvature = np.tile(_pack(np.eye(3, dtype=np.float32)), (n_freq, 1))
    theta = jnp.zeros_like(gradient)
    _, raw_velocity, _ = momentum_step(theta, theta, gradient, curvature, radii <= 2, learning_rate=0.4, floor=1e-7)
    mask = support_mask(8, diameter_px=5)
    velocity = bandlimit_and_mask(raw_velocity, shape, 2, mask)
    model = bandlimit_and_mask(theta + velocity, shape, 2, mask)
    assert np.max(np.abs(np.asarray(velocity)[radii > 2])) < 1e-7
    assert np.max(np.abs(np.asarray(model)[radii > 2])) < 1e-7


def test_existing_direct_residual_retains_posterior_covariance():
    # One image, translation and pose. Expected residual is Y1*alpha-L*projection.
    mean = np.array([0.25, -0.5], np.float32)
    covariance = np.array([[0.8, 0.2], [0.2, 0.4]], np.float32)
    alpha = np.array([1.0, *mean], np.float32)
    second = np.outer(mean, mean) + covariance
    augmented = np.block([[np.ones((1, 1)), mean[None]], [mean[:, None], second]]).astype(np.float32)
    projected = np.array([0.5, -0.3, 0.7], np.complex64)
    y1 = np.array([[[2.0]]], np.complex64)
    residual, _, _ = residual_image_statistics(
        jnp.zeros((1, 1, 1), jnp.float32),
        jnp.asarray(alpha[None, None, None, :]),
        jnp.asarray(_pack(augmented)[None, None, None, :]),
        jnp.zeros(1, jnp.float32),
        jnp.asarray(y1),
        jnp.ones((1, 1), jnp.float32),
        jnp.asarray(projected[None, :, None]),
    )
    expected = alpha * y1[0, 0, 0] - augmented @ projected
    np.testing.assert_allclose(np.asarray(residual)[:, 0, 0], expected, rtol=1e-6, atol=1e-6)
    assert not np.allclose(expected, alpha * y1[0, 0, 0] - np.outer(alpha, alpha) @ projected)


def test_cli_config_and_optimizer_checkpoint(tmp_path):
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args(
        ["manifest.json", "-o", str(tmp_path), "--optimizer", "momentum_sgd", "--sgd-learning-rate", "0.25"]
    )
    config = Config(optimizer=args.optimizer, sgd_learning_rate=args.sgd_learning_rate)
    assert (config.optimizer, config.sgd_learning_rate) == ("momentum_sgd", 0.25)
    assert Config().optimizer == "vdam"
    with pytest.raises(ValueError, match="learning rate"):
        Config(optimizer="momentum_sgd", sgd_learning_rate=np.nan)
    theta = jnp.ones((5, 3), jnp.complex64)
    velocity = jnp.full((5, 3), 0.125 + 0.25j, jnp.complex64)
    state = State(
        theta,
        None,
        jnp.ones(3, jnp.float32),
        1,
        np.arange(8),
        np.random.default_rng(11).bit_generator.state,
        2.0,
        4,
        {"seed": 11},
        sgd_momentum=velocity,
    )
    path = tmp_path / "sgd.npz"
    save(path, state, config, {"fixture": "test"})
    restored = load(path, config, {"fixture": "test"})
    assert restored.moments is None
    np.testing.assert_allclose(restored.sgd_momentum, velocity, rtol=1e-6, atol=1e-7)
    with pytest.raises(ValueError, match="identity mismatch"):
        load(path, Config(), {"fixture": "test"})


def test_controller_pools_direct_residual_and_restarts_with_velocity(tmp_path, monkeypatch):
    shape = (4, 4, 4)
    n_freq = shape[0] * shape[1] * (shape[2] // 2 + 1)
    dc = int(np.argmin(np.asarray(ftu.get_grid_of_radial_distances_real(shape)).reshape(-1)))
    zero = jnp.zeros((n_freq, 3), jnp.complex64)
    h = np.zeros((n_freq, 3, 3), np.float32)
    h[dc] = np.diag([2, 1, 1])
    gradient = np.zeros((n_freq, 3), np.complex64)
    gradient[dc, 0] = -1
    monkeypatch.setattr(iteration_loop, "initialize", lambda *_args, **_kwargs: (zero, jnp.ones(3, jnp.float32), {"seed": 11}))
    monkeypatch.setattr(iteration_loop, "support_mask", lambda *_args: jnp.ones(shape, jnp.float32))
    # A soft support operation must act once on each output, not twice on the new step.
    monkeypatch.setattr(iteration_loop, "bandlimit_and_mask", lambda value, *_args: value * 0.5)
    monkeypatch.setattr(
        iteration_loop.sampling, "get_relion_hidden_rotation_grid", lambda *_args, **_kwargs: np.zeros((1, 3))
    )

    def expectation(_dataset, _state, _config, ids, _iteration, *, embeddings_only=False):
        if embeddings_only:
            return SimpleNamespace(original_image_ids=np.asarray(ids), embeddings=jnp.zeros((len(ids), 2)))
        return SimpleNamespace(
            residual_gradient=jnp.asarray(gradient),
            lhs_tri=jnp.asarray(_pack(h)),
            # Deliberately incompatible RHS: the controller must use direct residual.
            rhs=jnp.full((n_freq, 3), 100, jnp.complex64),
            residual_num=jnp.ones(3, jnp.float32) * 2,
            residual_den=jnp.ones(3, jnp.float32) * 2,
            original_image_ids=np.asarray(ids),
            n_images=len(ids),
            log_likelihood=0.0,
            diagnostics={
                "offset_second_sum_px2": 2.0,
                "rotation_mass": np.array([float(len(ids))]),
                "latent_covariance_trace_mean": 1.0,
                "pose_entropy_mean": 0.0,
                "pmax_mean": 1.0,
            },
        )

    monkeypatch.setattr(iteration_loop, "expectation", expectation)
    dataset = SimpleNamespace(n_images=4, grid_size=4, volume_shape=shape, voxel_size=1.0)
    config = Config(iterations=1, stages=((1, 1, 0),), optimizer="momentum_sgd", sgd_learning_rate=0.4)
    identity = {"fixture": "tiny"}
    state = iteration_loop.run(dataset, config, tmp_path, identity, diameter_ang=2.0)
    # Each half contributes G=-1 and trace H=4; pooled G=-2, trace H=8.
    assert np.asarray(state.theta)[dc, 0].real == pytest.approx(-0.005, abs=1e-7)
    assert state.moments is None
    np.testing.assert_allclose(state.sgd_momentum, state.theta, rtol=1e-6, atol=1e-7)
    restored = load(tmp_path / "checkpoint_0001.npz", config, identity)
    np.testing.assert_allclose(restored.sgd_momentum, state.sgd_momentum, rtol=1e-6, atol=1e-7)
    compact = iteration_loop.run(
        dataset, config, tmp_path / "compact", identity, diameter_ang=2.0,
        log_direction_prior=False,
    )
    assert_matches(compact.theta, state.theta)
    assert_matches(compact.sgd_momentum, state.sgd_momentum)
    assert_matches(compact.direction_prior, state.direction_prior)
    full_log = json.loads((tmp_path / "iterations.jsonl").read_text().strip())
    compact_log = json.loads((tmp_path / "compact/iterations.jsonl").read_text().strip())
    assert "direction_prior" in full_log
    assert "direction_prior" not in compact_log
    for log in (full_log, compact_log):
        log.pop("elapsed_seconds")
    assert_trees_match({key: value for key, value in full_log.items() if key != "direction_prior"}, compact_log)
