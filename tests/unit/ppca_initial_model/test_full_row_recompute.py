"""Bounded-memory coarse PPCA routing and online normalization contracts."""

import argparse
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from relax.commands.ppca_initial_model import add_args
from relax.ppca_initial_model import iteration_loop
from relax.ppca_initial_model.config import Config
from relax.ppca_refinement.full_row_stream import _add_block_partition_online

pytestmark = pytest.mark.unit


def test_coarse_recompute_cli_is_opt_in():
    parser = argparse.ArgumentParser()
    add_args(parser)
    required = ["manifest.json", "--output", "out"]
    assert not parser.parse_args(required).stream_coarse_recompute
    assert parser.parse_args([*required, "--stream-coarse-recompute"]).stream_coarse_recompute
    fixed = parser.parse_args([*required, "--stochastic-batch-size", "300", "--stochastic-all-iterations"])
    assert fixed.stochastic_batch_size == 300 and fixed.stochastic_all_iterations
    controls = parser.parse_args(
        [
            *required,
            "--stochastic-batch-size",
            "300",
            "--stochastic-all-iterations",
            "--balanced-stochastic-halves",
            "--checkpoint-interval",
            "10",
            "--skip-final-embeddings",
        ]
    )
    assert controls.balanced_stochastic_halves and controls.checkpoint_interval == 10
    assert controls.skip_final_embeddings


@pytest.mark.parametrize("optimizer", ["vdam", "momentum_sgd"])
def test_fixed_stochastic_batch_includes_last_update_only_when_opted_in(optimizer):
    base = dict(iterations=600, optimizer=optimizer, stochastic_batch_size=300)
    assert Config(**base).schedule(600, 20_000)[0] == 20_000
    fixed = Config(**base, stochastic_all_iterations=True)
    assert fixed.schedule(1, 20_000)[0] == 300
    assert fixed.schedule(599, 20_000)[0] == 300
    assert fixed.schedule(600, 20_000)[0] == 300
    with pytest.raises(ValueError, match="requires a fixed stochastic batch size"):
        Config(stochastic_all_iterations=True)


def test_balanced_stochastic_halves_are_fixed_and_reproducible():
    config = Config(
        iterations=600,
        stochastic_batch_size=300,
        stochastic_all_iterations=True,
        balanced_stochastic_halves=True,
    )
    assert config.schedule(600, 20_000)[0] == 300
    order = np.random.default_rng(9).permutation(20_000)
    selected, halves = iteration_loop._select_halves(np.random.default_rng(11), order, 300, True)
    _, repeated = iteration_loop._select_halves(np.random.default_rng(11), order, 300, True)
    assert [len(ids) for ids in halves] == [150, 150]
    for half, (ids, again) in enumerate(zip(halves, repeated, strict=True)):
        assert np.all(ids % 2 == half)
        np.testing.assert_array_equal(ids, again)
    assert len(np.unique(np.concatenate(halves))) == 300
    np.testing.assert_array_equal(selected, np.concatenate(halves))
    shuffled = np.random.default_rng(11).permutation(order)
    for half, ids in enumerate(halves):
        np.testing.assert_array_equal(ids, shuffled[shuffled % 2 == half][:150])
    default_selected, default = iteration_loop._select_halves(np.random.default_rng(11), order, 300, False)
    np.testing.assert_array_equal(default_selected, shuffled[:300])
    for half, ids in enumerate(default):
        np.testing.assert_array_equal(ids, default_selected[default_selected % 2 == half])
    with pytest.raises(ValueError, match="even fixed batch"):
        Config(stochastic_batch_size=301, stochastic_all_iterations=True, balanced_stochastic_halves=True)
    with pytest.raises(ValueError, match="pseudo-half population"):
        iteration_loop._select_halves(np.random.default_rng(11), np.arange(4), 6, True)
    with pytest.raises(ValueError, match="even effective image count"):
        iteration_loop._select_halves(np.random.default_rng(11), np.arange(3), 3, True)


def test_online_partition_normalizes_late_support_and_small_tails():
    blocks = [
        np.array([[[-np.inf, -np.inf], [-np.inf, -np.inf]], [[-100, -101], [-102, -103]]], np.float32),
        np.array([[[-12, -14], [-20, -40]], [[0, -2], [-4, -8]]], np.float32),
        np.array([[[0, -0.1], [-8, -60]], [[-30, -50], [-70, -90]]], np.float32),
    ]
    center = jnp.full(2, -jnp.inf, jnp.float32)
    partition = jnp.zeros(2, jnp.float32)
    for block in blocks:
        new_center = jnp.maximum(center, jnp.max(block, axis=(1, 2)))
        partition = _add_block_partition_online(partition, jnp.asarray(block), center, new_center)
        center = new_center
    full = np.concatenate([b.reshape(2, -1) for b in blocks], axis=1)
    expected_center = np.max(full, axis=1)
    expected_partition = np.sum(np.exp(full - expected_center[:, None]), axis=1)
    np.testing.assert_allclose(np.asarray(center), expected_center, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.asarray(partition), expected_partition, rtol=1e-6, atol=1e-7)
    normalized = np.sum(np.exp(full - np.asarray(center)[:, None]) / np.asarray(partition)[:, None], axis=1)
    np.testing.assert_allclose(normalized, np.ones(2), rtol=1e-6, atol=1e-7)


def test_coarse_route_uses_one_parent_and_keeps_default_dense(monkeypatch):
    n = 8
    n_freq = n * n * (n // 2 + 1)
    rotations = np.stack([np.eye(3, dtype=np.float32)] * 3)
    eulers = np.zeros((3, 3), np.float32)
    shifts = np.zeros((2, 2), np.float32)
    monkeypatch.setattr(
        iteration_loop.sampling,
        "get_relion_hidden_rotation_grid",
        lambda _hp, *, matrices: rotations if matrices else eulers,
    )
    monkeypatch.setattr(iteration_loop.sampling, "get_relion_translation_grid", lambda **_kw: shifts)
    monkeypatch.setattr(iteration_loop, "make_radial_noise", lambda _noise, _shape: np.ones((n, n // 2 + 1)))
    data = SimpleNamespace(grid_size=n, image_shape=(n, n), volume_shape=(n, n, n))
    state = SimpleNamespace(
        theta=jnp.zeros((n_freq, 3), jnp.complex64),
        noise=jnp.ones(5, jnp.float32),
        direction_prior=None,
        direction_order=-1,
        offset_variance=4.0,
    )
    captured = []
    sentinel = SimpleNamespace(diagnostics={})

    def prepare(*_args, **kwargs):
        captured.append(kwargs)
        return sentinel

    def part(_stream, ids, support, *, factor_once, recompute):
        assert _stream is sentinel and support == [None] * len(ids) and recompute
        assert factor_once == (len(ids) > 1)
        return SimpleNamespace(
            rhs=jnp.zeros((n_freq, 3), jnp.complex64),
            lhs_tri=jnp.zeros((n_freq, 6), jnp.float32),
            residual_gradient=jnp.zeros((n_freq, 3), jnp.complex64),
            residual_num=jnp.ones(5, jnp.float32),
            residual_den=jnp.ones(5, jnp.float32),
            embeddings=jnp.zeros((len(ids), 2), jnp.float32),
            original_image_ids=np.asarray(ids),
            n_images=len(ids),
            log_likelihood=0.0,
            diagnostics={
                "rotation_mass": np.ones(3, np.float32) * len(ids) / 3,
                "offset_second_sum_px2": 0.0,
                "latent_covariance_trace_mean": 1.0,
                "pose_entropy_mean": 0.0,
                "pmax_mean": 0.5,
                "scored_image_rows": len(ids) * 3,
                "supported_image_rows": len(ids) * 3,
            },
        )

    monkeypatch.setattr(iteration_loop, "prepare_full_row_stream", prepare)
    monkeypatch.setattr(iteration_loop, "accumulate_full_row_tile", part)
    ids = np.arange(5)
    config = Config(
        stages=((1, 2, 0),),
        oversampling=0,
        image_batch_size=2,
        rotation_block_size=2,
        stream_coarse_recompute=True,
    )
    stats = iteration_loop.expectation(data, state, config, ids, 1)
    assert len(captured) == 1
    assert captured[0]["n_coarse_rotations"] == captured[0]["n_coarse_translations"] == 1
    np.testing.assert_array_equal(captured[0]["rotation_parent"], np.zeros(3, np.int32))
    np.testing.assert_array_equal(captured[0]["translation_parent"], np.zeros(2, np.int32))
    np.testing.assert_allclose(stats.diagnostics["rotation_mass"], np.ones(3) * 5 / 3, rtol=1e-6)
    assert stats.diagnostics["engine"] == "full_row_coarse_recompute"
    assert stats.n_images == 5

    monkeypatch.setattr(
        iteration_loop,
        "full_row_tile_embeddings",
        lambda _stream, tile_ids, support, *, recompute: SimpleNamespace(
            embeddings=jnp.ones((len(tile_ids), 2), jnp.float32),
            original_image_ids=np.asarray(tile_ids),
            n_images=len(tile_ids),
        ),
    )
    embedded = iteration_loop.expectation(data, state, config, ids, 1, embeddings_only=True)
    np.testing.assert_array_equal(embedded.original_image_ids, ids)
    assert embedded.embeddings.shape == (5, 2)

    dense_called = []
    monkeypatch.setattr(
        iteration_loop,
        "accumulate_dense_ppca_statistics",
        lambda *_args, **_kwargs: dense_called.append(True) or sentinel,
    )
    assert iteration_loop.expectation(data, state, Config(stages=((1, 2, 0),), oversampling=0), ids, 1) is sentinel
    assert dense_called == [True]
    with pytest.raises(ValueError, match="oversampling=0"):
        Config(oversampling=1, stream_coarse_recompute=True)
