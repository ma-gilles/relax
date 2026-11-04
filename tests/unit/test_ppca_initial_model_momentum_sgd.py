"""Independent CPU contracts for PPCA's opt-in augmented momentum step."""

import argparse
import dataclasses
import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches, assert_trees_match
from recovar.core import fourier_transform_utils as ftu

from relax.commands.ppca_initial_model import add_args
from relax.ppca_initial_model import checkpoint, iteration_loop
from relax.ppca_initial_model.checkpoint import load, save
from relax.ppca_initial_model.config import Config
from relax.ppca_initial_model.initialization import bandlimit_and_mask, support_mask
from relax.ppca_initial_model.sgd_update import metric_curvature, momentum_step
from relax.ppca_initial_model.state import State
from relax.ppca_refinement.residual_statistics import residual_image_statistics

pytestmark = pytest.mark.unit


def _pack(matrix):
    return matrix[..., np.triu_indices(matrix.shape[-1])[0], np.triu_indices(matrix.shape[-1])[1]]


def _curvature(matrix):
    """The ``(mean entry, loading trace)`` of each ``(1 + q)`` square metric, float32."""
    loading = np.trace(matrix[..., 1:, 1:], axis1=-2, axis2=-1)
    return np.stack([matrix[..., 0, 0], loading], axis=-1).astype(np.float32)


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
    curvature = _curvature(h)
    before = _loss(theta, h, rhs)
    updated, velocity, info = momentum_step(
        theta, np.zeros_like(theta), gradient, curvature, np.ones(2, bool), learning_rate=0.4, floor=1e-7
    )
    scale = np.max(curvature, axis=0)  # the mean column and the loading columns each have their own divisor
    expected = np.float32(0.04) * gradient / np.array([scale[0], scale[1], scale[1]], np.float32)
    np.testing.assert_allclose(velocity, expected, rtol=1e-6, atol=1e-7)
    assert _loss(np.asarray(updated), h, rhs) < before
    assert info["curvature_mean"] == pytest.approx(float(scale[0]))
    assert info["curvature_loading"] == pytest.approx(float(scale[1]))
    duplicated, _, _ = momentum_step(
        theta, np.zeros_like(theta), 2 * gradient, 2 * curvature, np.ones(2, bool), learning_rate=0.4, floor=1e-7
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
    updated, velocity, info = momentum_step(theta, old, gradient, _curvature(h), active, learning_rate=0.4, floor=1e-7)
    rotated, rotated_velocity, rotated_info = momentum_step(
        theta @ transform,
        old @ transform,
        gradient @ transform,
        _curvature(np.einsum("ip,fij,jq->fpq", transform, h, transform)),
        active,
        learning_rate=0.4,
        floor=1e-7,
    )
    np.testing.assert_allclose(rotated, np.asarray(updated) @ transform, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(rotated_velocity, np.asarray(velocity) @ transform, rtol=1e-6, atol=1e-6)
    assert rotated_info["curvature_mean"] == pytest.approx(info["curvature_mean"], rel=1e-6)
    assert rotated_info["curvature_loading"] == pytest.approx(info["curvature_loading"], rel=1e-6)
    np.testing.assert_allclose(np.asarray(velocity)[2], np.zeros(3), atol=1e-7)


def test_q0_units_zero_curvature_and_signed_step():
    theta = np.zeros((2, 1), np.complex64)
    gradient = np.array([[-12], [100]], np.complex64)
    curvature = np.array([[6, 0], [0, 0]], np.float32)
    result, velocity, info = momentum_step(
        theta, theta, gradient, curvature, np.array([True, False]), learning_rate=0.5, floor=1e-7
    )
    np.testing.assert_allclose(result[:, 0], [-0.1, 0], atol=1e-7)
    np.testing.assert_allclose(velocity[:, 0], [-0.1, 0], atol=1e-7)
    assert info["curvature_mean"] == pytest.approx(6)
    assert info["curvature_loading"] is None
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


@pytest.mark.parametrize("p", [1, 4])
def test_metric_curvature_splits_the_packed_diagonal_into_the_mean_and_the_loading_trace(p):
    rng = np.random.default_rng(9)
    a = rng.normal(size=(5, p, p)).astype(np.float32)
    h = a @ np.swapaxes(a, -1, -2)
    np.testing.assert_allclose(metric_curvature(_pack(h), p), _curvature(h), rtol=1e-6)


def test_the_mean_step_does_not_depend_on_the_number_of_loadings():
    """Loadings with their own curvature leave the mean column's step as it is without them."""
    rng = np.random.default_rng(3)
    mean_curvature = rng.uniform(0.5, 2.0, size=6).astype(np.float32)
    mean_gradient = (rng.normal(size=6) + 1j * rng.normal(size=6)).astype(np.complex64)
    active = np.ones(6, bool)
    steps = []
    for q in (0, 10):
        theta = np.zeros((6, 1 + q), np.complex64)
        gradient = np.zeros_like(theta)
        gradient[:, 0] = mean_gradient
        gradient[:, 1:] = 1 + 1j
        curvature = np.stack([mean_curvature, np.full(6, 10.0 * q, np.float32)], axis=-1)
        _, velocity, _ = momentum_step(theta, theta, gradient, curvature, active, learning_rate=0.4, floor=1e-7)
        steps.append(np.asarray(velocity)[:, 0])
    np.testing.assert_allclose(steps[1], steps[0], rtol=1e-6)
    expected = np.float32(0.04) * mean_gradient / mean_curvature.max()
    np.testing.assert_allclose(steps[0], expected, rtol=1e-6)


def test_a_loading_group_without_curvature_takes_no_step_while_the_mean_does():
    theta = np.zeros((3, 3), np.complex64)
    gradient = np.ones((3, 3), np.complex64)
    curvature = np.stack([np.ones(3, np.float32), np.zeros(3, np.float32)], axis=-1)
    _, velocity, _ = momentum_step(
        theta, np.ones_like(theta), gradient, curvature, np.ones(3, bool), learning_rate=0.5, floor=1e-7
    )
    assert np.all(np.asarray(velocity)[:, 0] != 0)
    np.testing.assert_allclose(np.asarray(velocity)[:, 1:], 0, atol=1e-7)


def test_loading_scale_multiplies_the_loading_gradient_term_and_leaves_the_mean_and_momentum_alone():
    rng = np.random.default_rng(5)
    theta = np.zeros((4, 4), np.complex64)
    previous = (rng.normal(size=theta.shape) + 1j * rng.normal(size=theta.shape)).astype(np.complex64)
    gradient = (rng.normal(size=theta.shape) + 1j * rng.normal(size=theta.shape)).astype(np.complex64)
    curvature = np.stack([rng.uniform(1, 2, 4), rng.uniform(10, 20, 4)], axis=-1).astype(np.float32)
    active = np.ones(4, bool)

    def step(scale, prev, grad):
        return np.asarray(
            momentum_step(theta, prev, grad, curvature, active, learning_rate=0.4, floor=1e-7, loading_scale=scale)[1]
        )

    base, scaled = step(1.0, previous, gradient), step(0.25, previous, gradient)
    np.testing.assert_allclose(scaled[:, 0], base[:, 0], rtol=1e-6)
    momentum_only = 0.9 * previous[:, 1:]
    np.testing.assert_allclose(scaled[:, 1:] - momentum_only, 0.25 * (base[:, 1:] - momentum_only), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(step(0.25, previous, np.zeros_like(gradient)), 0.9 * previous, rtol=1e-6)
    with pytest.raises(ValueError, match="Loading scale"):
        step(0.0, previous, gradient)


def test_velocity_uses_the_same_fourier_band_as_model():
    shape = (8, 8, 8)
    n_freq = shape[0] * shape[1] * (shape[2] // 2 + 1)
    radii = np.asarray(ftu.get_grid_of_radial_distances_real(shape, rounded=False)).reshape(-1)
    gradient = jnp.ones((n_freq, 3), jnp.complex64)
    curvature = np.full((n_freq, 2), 3, np.float32)
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
        ["--i", "particles.star", "--particle-diameter", "100", "-o", str(tmp_path), "--optimizer", "momentum_sgd", "--sgd-learning-rate", "0.25", "--sgd-loading-scale", "0.5"]
    )
    config = Config(
        optimizer=args.optimizer, sgd_learning_rate=args.sgd_learning_rate, sgd_loading_scale=args.sgd_loading_scale
    )
    assert (config.optimizer, config.sgd_learning_rate, config.sgd_loading_scale) == ("momentum_sgd", 0.25, 0.5)
    assert Config().optimizer == "vdam" and Config().sgd_loading_scale == 1 / 16
    default_args = parser.parse_args(["--i", "particles.star", "--particle-diameter", "100", "-o", str(tmp_path)])
    assert default_args.sgd_loading_scale == Config().sgd_loading_scale
    with pytest.raises(ValueError, match="learning rate"):
        Config(optimizer="momentum_sgd", sgd_learning_rate=np.nan)
    with pytest.raises(ValueError, match="loading scale"):
        Config(optimizer="momentum_sgd", sgd_loading_scale=0.0)
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
    with pytest.raises(ValueError, match="identity mismatch"):
        load(path, dataclasses.replace(config, sgd_loading_scale=1.0), {"fixture": "test"})


def test_a_checkpoint_from_before_the_loading_scale_is_scale_one():
    older = checkpoint._identity_config(dataclasses.asdict(Config(sgd_loading_scale=1.0)))
    assert "sgd_loading_scale" not in older
    assert checkpoint._identity_config(dataclasses.asdict(Config()))["sgd_loading_scale"] == pytest.approx(1 / 16)


def test_controller_pools_direct_residual_and_restarts_with_velocity(tmp_path, monkeypatch):
    shape = (4, 4, 4)
    n_freq = shape[0] * shape[1] * (shape[2] // 2 + 1)
    dc = int(np.argmin(np.asarray(ftu.get_grid_of_radial_distances_real(shape)).reshape(-1)))
    zero = jnp.zeros((n_freq, 3), jnp.complex64)
    h = np.zeros((n_freq, 3, 3), np.float32)
    h[dc] = np.diag([2, 1, 1])
    gradient = np.zeros((n_freq, 3), np.complex64)
    gradient[dc, 0] = -1
    monkeypatch.setattr(
        iteration_loop, "initialize", lambda *_args, **_kwargs: (zero, jnp.ones(3, jnp.float32), {"seed": 11})
    )
    monkeypatch.setattr(iteration_loop, "support_mask", lambda *_args: jnp.ones(shape, jnp.float32))
    # A soft support operation must act once on each output, not twice on the new step.
    monkeypatch.setattr(iteration_loop, "bandlimit_and_mask", lambda value, *_args: value * 0.5)
    monkeypatch.setattr(
        iteration_loop.grids, "get_relion_hidden_rotation_grid", lambda *_args, **_kwargs: np.zeros((1, 3))
    )
    # The one-rotation stand-in grid has one direction.
    monkeypatch.setattr(iteration_loop.grids, "rotation_direction_ids", lambda *_a, **_k: np.zeros(1, np.int64))

    def expectation(_dataset, _state, _config, ids, _iteration, *, embeddings_only=False, diameter_ang=None):
        if embeddings_only:
            zeros = np.zeros(len(ids), np.float32)
            return SimpleNamespace(
                original_image_ids=np.asarray(ids), embeddings=jnp.zeros((len(ids), 2)),
                log_evidence=zeros, best_pose_log_score=zeros, max_posterior=zeros + 1, null_log_likelihood=zeros,
            )
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
    # The host-mask engine: the patched expectation stands in for it.
    config = Config(
        iterations=1, stages=((1, 1, 0),), optimizer="momentum_sgd", sgd_learning_rate=0.4, stream_coarse_recompute=False
    )
    identity = {"fixture": "tiny"}
    state = iteration_loop.run(dataset, config, tmp_path, identity, diameter_ang=2.0)
    # Each half contributes G=-1 and mean curvature H_00=2; pooled G=-2, H_00=4: step 0.1 * 0.4 * G / H_00,
    # then the half-strength support.
    assert np.asarray(state.theta)[dc, 0].real == pytest.approx(-0.01, abs=1e-7)
    assert state.moments is None
    np.testing.assert_allclose(state.sgd_momentum, state.theta, rtol=1e-6, atol=1e-7)
    restored = load(tmp_path / "checkpoint_0001.npz", config, identity)
    np.testing.assert_allclose(restored.sgd_momentum, state.sgd_momentum, rtol=1e-6, atol=1e-7)
    compact = iteration_loop.run(
        dataset,
        config,
        tmp_path / "compact",
        identity,
        diameter_ang=2.0,
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
