"""Bounded-memory coarse PPCA routing and online normalization contracts."""

import argparse
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from relax.commands.ppca_initial_model import add_args
from relax.ppca_initial_model import iteration_loop
from relax.ppca_initial_model.config import Config
from relax.ppca_refinement.full_row_stream import _normalize

pytestmark = pytest.mark.unit


def test_coarse_recompute_cli_is_the_default():
    parser = argparse.ArgumentParser()
    add_args(parser)
    required = ["manifest.json", "--output", "out"]
    assert parser.parse_args(required).stream_coarse_recompute
    assert parser.parse_args([*required, "--stream-coarse-recompute"]).stream_coarse_recompute
    assert not parser.parse_args([*required, "--no-stream-coarse-recompute"]).stream_coarse_recompute
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


def test_tile_normalization_handles_late_support_and_ties():
    # Three (B=2, T=2, R=2) rotation blocks; image 0 has no support in block 0.
    blocks = [
        np.array([[[-np.inf, -np.inf], [-np.inf, -np.inf]], [[-50, -51], [-52, -53]]], np.float32),
        np.array([[[-12, -14], [-20, -40]], [[0, -2], [-4, -8]]], np.float32),
        np.array([[[0, -0.1], [-8, -200]], [[-30, -50], [-70, 0]]], np.float32),
    ]
    # The engine keeps scores rotation-major, (R, B, T).
    kept = jnp.asarray(np.concatenate(blocks + [np.full((2, 2, 2), -np.inf, np.float32)], axis=2).transpose(2, 0, 1))
    rows = jnp.asarray([10, 11, 12, 13, 14, 15, 99, 99], jnp.int32)
    posterior = _normalize(kept, rows, n_blocks=3, block_size=2)
    full = np.concatenate([b.reshape(2, -1) for b in blocks], axis=1).astype(np.float64)
    expected_center = np.max(full, axis=1)
    expected_partition = np.sum(np.exp(full - expected_center[:, None]), axis=1)
    np.testing.assert_allclose(np.asarray(posterior.center), expected_center, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.exp(np.asarray(posterior.centered_logZ)), expected_partition, rtol=1e-6)
    # First maximum in block order: image 1 ties at 0 in block 1 (row 12, t 0) and block 2 (row 15, t 1).
    assert np.asarray(posterior.top_rotation).tolist() == [14, 12]
    assert np.asarray(posterior.top_translation).tolist() == [0, 0]


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

    def tiles(_stream, items):
        return [part(_stream, ids, support) for ids, support in items]

    def part(_stream, ids, support):
        assert _stream is sentinel and support == [None] * len(ids)
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
    monkeypatch.setattr(iteration_loop, "accumulate_full_row_tiles", tiles)
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
        lambda _stream, tile_ids, support: SimpleNamespace(
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
    host_mask = Config(stages=((1, 2, 0),), stream_coarse_recompute=False)
    assert iteration_loop.expectation(data, state, host_mask, ids, 1) is sentinel
    assert dense_called == [True]


def test_defaults_are_the_dense_stream_and_oversampling_is_refused():
    """The default engine is the qualified dense stream; oversampling and its fine pass are refused."""
    config = Config()
    assert config.oversampling == 0 and config.stream_coarse_recompute
    assert (config.image_batch_size, config.rotation_block_size) == (150, 512)
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args(["manifest.json", "--output", "out"])
    assert args.oversampling == 0 and args.stream_coarse_recompute
    assert (args.image_batch_size, args.rotation_block_size) == (150, 512)
    host_mask = parser.parse_args(["manifest.json", "--output", "out", "--no-stream-coarse-recompute"])
    assert not host_mask.stream_coarse_recompute
    for change in ({"oversampling": 1}, {"stream_full_fine_rows": True}, {"fine_devices": 2}):
        with pytest.raises(ValueError, match="section 14"):
            Config(stream_coarse_recompute=False, **change)


def test_gemm_precision_flag_config_and_checkpoint_default(tmp_path):
    """--ppca-gemm-precision defaults to auto, tf32 needs a streamed engine; precision is not checkpoint identity."""
    import dataclasses
    import json

    from relax.ppca_initial_model import checkpoint

    parser = argparse.ArgumentParser()
    add_args(parser)
    required = ["manifest.json", "--output", "out"]
    assert parser.parse_args(required).ppca_gemm_precision == "auto"
    assert parser.parse_args([*required, "--ppca-gemm-precision", "fp32"]).ppca_gemm_precision == "fp32"
    assert parser.parse_args([*required, "--ppca-gemm-precision", "tf32"]).ppca_gemm_precision == "tf32"
    with pytest.raises(SystemExit):
        parser.parse_args([*required, "--ppca-gemm-precision", "bf16"])
    with pytest.raises(ValueError, match="streamed engines"):
        Config(gemm_precision="tf32", stream_coarse_recompute=False)
    with pytest.raises(ValueError, match="fp32 or tf32"):
        Config(gemm_precision="bf16", oversampling=0, stream_coarse_recompute=True)
    assert Config().gemm_precision == "auto"  # the host-mask engines ignore it
    config = Config(q=2, oversampling=0, stream_coarse_recompute=True, gemm_precision="fp32")
    shape = (4, 4, 4)
    n_freq = shape[0] * shape[1] * (shape[2] // 2 + 1)
    state = iteration_loop.State(
        jnp.zeros((n_freq, 3), jnp.complex64), iteration_loop.empty_moments(jnp.zeros((n_freq, 3), jnp.complex64)),
        jnp.ones(3, jnp.float32), 0, np.arange(8), np.random.default_rng(1).bit_generator.state, 2.0, 1, {"seed": 1},
    )
    path = tmp_path / "old.npz"
    checkpoint.save(path, state, config, {"fixture": "tiny"})
    # A checkpoint written before the field existed: drop it from the stored configuration.
    with np.load(path, allow_pickle=False) as arrays:
        saved = {k: arrays[k] for k in arrays.files}
    meta = json.loads(str(saved["metadata"]))
    meta["config"].pop("gemm_precision")
    saved["metadata"] = np.asarray(json.dumps(meta))
    np.savez(path, **saved)
    assert checkpoint.saved_gemm_precision(path) == "fp32"
    for precision in ("fp32", "auto", "tf32"):  # a runtime setting: any value resumes
        assert checkpoint.load(path, dataclasses.replace(config, gemm_precision=precision), {"fixture": "tiny"})
    # Model-defining fields still have to match.
    with pytest.raises(ValueError, match="identity mismatch"):
        checkpoint.load(path, dataclasses.replace(config, rotation_block_size=4), {"fixture": "tiny"})


def test_auto_gemm_precision_resolves_by_device():
    import jax

    from relax.ppca_refinement.full_row_stream import resolve_gemm_precision

    cpu = jax.devices("cpu")[0]
    assert resolve_gemm_precision("auto", cpu) == "fp32"
    assert resolve_gemm_precision("fp32", cpu) == "fp32"
    with pytest.raises(ValueError, match="compute capability"):
        resolve_gemm_precision("tf32", cpu)
    with pytest.raises(ValueError, match="gemm_precision"):
        resolve_gemm_precision("bf16", cpu)
    assert iteration_loop._gemm_precision_used(Config(stream_coarse_recompute=False)) == "fp32"  # host-mask engine


def test_resume_under_another_gemm_precision_is_logged(tmp_path, monkeypatch, caplog):
    """An fp32 checkpoint resumes under auto; the warning and every update record name the change."""
    import dataclasses
    import json
    import logging

    from relax.ppca_initial_model import checkpoint

    shape = (4, 4, 4)
    n_freq = shape[0] * shape[1] * (shape[2] // 2 + 1)
    monkeypatch.setattr(iteration_loop, "support_mask", lambda *_args: jnp.ones(shape, jnp.float32))
    monkeypatch.setattr(iteration_loop, "bandlimit_and_mask", lambda value, *_args: value)
    monkeypatch.setattr(iteration_loop.sampling, "get_relion_hidden_rotation_grid", lambda *_a, **_k: np.zeros((1, 3)))

    def groups(_dataset, _state, _config, halves, _iteration, **_kwargs):
        return [
            SimpleNamespace(
                residual_gradient=jnp.zeros((n_freq, 3), jnp.complex64), metric_trace=jnp.ones(n_freq, jnp.float32),
                lhs_tri=None, rhs=None, residual_num=jnp.ones(3, jnp.float32), residual_den=jnp.ones(3, jnp.float32),
                original_image_ids=np.asarray(ids), n_images=len(ids), log_likelihood=0.0,
                diagnostics={"offset_second_sum_px2": 2.0, "rotation_mass": np.array([float(len(ids))]),
                             "latent_covariance_trace_mean": 1.0, "pose_entropy_mean": 0.0, "pmax_mean": 1.0},
            )
            for ids in halves
        ]

    monkeypatch.setattr(iteration_loop, "expectation_groups", groups)
    monkeypatch.setattr(iteration_loop, "_curvature_trace", lambda stats, p: sum(s.metric_trace for s in stats))
    dataset = SimpleNamespace(n_images=40, grid_size=4, volume_shape=shape, voxel_size=1.0)
    fp32 = Config(iterations=3, stages=((1, 1, 0),), optimizer="momentum_sgd", oversampling=0,
                  stream_coarse_recompute=True, stochastic_batch_size=8, stochastic_all_iterations=True,
                  skip_final_embeddings=True, gemm_precision="fp32")
    theta = jnp.zeros((n_freq, 3), jnp.complex64)
    state = iteration_loop.State(theta, None, jnp.ones(3, jnp.float32), 1, np.arange(40),
                                 np.random.default_rng(1).bit_generator.state, 2.0, 1, {"seed": 1},
                                 sgd_momentum=jnp.zeros_like(theta))
    identity = {"fixture": "tiny"}
    checkpoint.save(tmp_path / "checkpoint_0001.npz", state, fp32, identity)
    auto = dataclasses.replace(fp32, gemm_precision="auto")
    with caplog.at_level(logging.WARNING, logger=iteration_loop.__name__):
        iteration_loop.run(dataset, auto, tmp_path / "run", identity, diameter_ang=2.0,
                           resume=tmp_path / "checkpoint_0001.npz")
    resolved = iteration_loop._gemm_precision_used(auto)  # tf32 on an sm_80+ default device, else fp32
    assert f"resumed fp32 checkpoint under auto ({resolved})" in caplog.text
    records = [json.loads(line) for line in (tmp_path / "run" / "iterations.jsonl").read_text().splitlines()]
    assert [r["iteration"] for r in records] == [2, 3]
    assert all(r["resumed_from_gemm_precision"] == "fp32" and r["gemm_precision"] == resolved for r in records)
    assert checkpoint.saved_gemm_precision(tmp_path / "run" / "checkpoint_0003.npz") == "auto"
