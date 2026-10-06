"""Gated momentum SGD (half-set shell gate, exact-diagonal step scale, noise annealing): CPU contracts."""

import argparse
import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from recovar.core import fourier_transform_utils as ftu

from relax.commands.ppca_initial_model import add_args, resolve_sgd_options
from relax.ppca_initial_model import checkpoint, iteration_loop
from relax.ppca_initial_model.config import Config
from relax.ppca_initial_model.sgd_update import gated_gradient, metric_diagonal, momentum_step
from relax.ppca_initial_model.state import State
from relax.ppca_initial_model.update import Moments, empty_moments, shell_gates
from relax.ppca_refinement.full_row_stream import ExactTracePPCAStats

pytestmark = pytest.mark.unit


def _pack(matrix):
    i, j = np.triu_indices(matrix.shape[-1])
    return matrix[..., i, j]


def test_gated_direction_converges_to_gate_times_the_unregularised_solution():
    """For a diagonal quadratic the gated step's fixed point is ``gate * theta_ml`` per channel."""
    diagonal = np.array([[3.0, 2.0, 1.5], [1.0, 2.5, 2.0]], np.float32)
    solution = np.array([[1 + 0.5j, -0.4j, 0.3], [-0.2, 0.6 + 0.1j, -0.5j]], np.complex64)
    gate = np.array([[0.3, 0.7, 1.0], [0.9, 0.5, 0.0]], np.float32)
    theta = np.zeros_like(solution)
    velocity = np.zeros_like(solution)
    for _ in range(3000):
        gradient = (diagonal * (solution - theta)).astype(np.complex64)
        direction = gated_gradient(theta, gradient, diagonal, gate)
        theta, velocity, _ = momentum_step(
            theta, velocity, direction, diagonal.sum(-1), np.ones(2, bool), learning_rate=1.0, floor=1e-7
        )
    np.testing.assert_allclose(np.asarray(theta), gate * solution, atol=1e-5)


def test_gate_one_is_the_plain_gradient_and_shapes_are_checked():
    rng = np.random.default_rng(3)
    theta = (rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3))).astype(np.complex64)
    gradient = (rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3))).astype(np.complex64)
    diagonal = rng.uniform(1, 2, size=(5, 3)).astype(np.float32)
    assert_matches(gated_gradient(theta, gradient, diagonal, np.ones((5, 3), np.float32)), gradient)
    # Gate 0 is pure shrinkage toward zero along the metric diagonal.
    assert_matches(gated_gradient(theta, gradient, diagonal, np.zeros((5, 3), np.float32)), -diagonal * theta)
    with pytest.raises(ValueError, match="same"):
        gated_gradient(theta, gradient, diagonal[:, :2], np.ones((5, 3), np.float32))
    with pytest.raises(TypeError, match="float32"):
        gated_gradient(theta, gradient, diagonal.astype(np.float64), np.ones((5, 3), np.float32))


def test_metric_diagonal_reads_the_packed_diagonal():
    rng = np.random.default_rng(9)
    a = rng.normal(size=(4, 3, 3)).astype(np.float32)
    h = a @ np.swapaxes(a, -1, -2)
    np.testing.assert_allclose(metric_diagonal(_pack(h), 3), np.diagonal(h, axis1=-2, axis2=-1), rtol=1e-6)


def test_shell_gates_by_hand():
    """Two shells; rho = 2 fudge <|theta|> / <|first_0 - first_1|>, the loading amplitude a norm over sqrt(q)."""
    theta = jnp.asarray([[1.0, 0.6, 0.8], [2.0, 0.0, 0.0], [3.0, 1.0, 0.0]], jnp.complex64)
    first = jnp.asarray(
        [[[0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.3, 0.4]], [[0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
        jnp.complex64,
    )
    shells = np.array([0, 0, 1])
    gates = shell_gates(theta, first, jnp.asarray([True, True, True]), shells, fudge=2.0)
    # Shell 0: signal (1.5, 0.5 / sqrt 2), disagreement (0.2, 0); shell 1: (3, 1 / sqrt 2) and (0, 0.5 / sqrt 2).
    rho_mean0 = 2 * 2.0 * 1.5 / 0.2
    rho_load1 = 2 * 2.0 * (1 / np.sqrt(2)) / (0.5 / np.sqrt(2))
    expected = np.array([[rho_mean0 / (1 + rho_mean0), 0.0], [0.0, rho_load1 / (1 + rho_load1)]], np.float32)
    np.testing.assert_allclose(np.asarray(gates.gates), expected, rtol=1e-6)
    assert gates.diagnostics()["zero_disagreement_shell_channels"] == 2
    np.testing.assert_allclose(np.asarray(gates.per_frequency(shells, 2)), expected[shells][:, [0, 1, 1]], rtol=1e-6)


def test_noise_inflation_schedule_and_option_checks():
    annealed = Config(optimizer="momentum_sgd", noise_anneal_factor=8.0, noise_anneal_updates=750)
    assert annealed.noise_inflation(1) == pytest.approx(1 + 7 * (1 - 1 / 750))
    assert annealed.noise_inflation(375) == pytest.approx(4.5)
    assert annealed.noise_inflation(750) == 1.0 and annealed.noise_inflation(2000) == 1.0
    assert all(Config().noise_inflation(t) == 1.0 for t in (0, 1, 750, 6000))
    for bad in ({"noise_anneal_factor": 8.0}, {"noise_anneal_updates": 10}, {"noise_anneal_factor": 0.5}):
        with pytest.raises(ValueError, match="annealing"):
            Config(**bad)
    with pytest.raises(ValueError, match="momentum SGD options"):
        Config(sgd_halfset_gate=True)
    with pytest.raises(ValueError, match="momentum SGD options"):
        Config(sgd_step_scale="exact_diagonal")
    with pytest.raises(ValueError, match="step scale must"):
        Config(optimizer="momentum_sgd", sgd_step_scale="hessian")
    with pytest.raises(ValueError, match="streamed engine"):
        Config(optimizer="momentum_sgd", sgd_step_scale="exact_diagonal", stream_coarse_recompute=False)
    assert Config().keeps_halfset_moments() and not Config(optimizer="momentum_sgd").keeps_halfset_moments()
    assert Config(optimizer="momentum_sgd", sgd_halfset_gate=True).keeps_halfset_moments()


def _options(*argv):
    parser = argparse.ArgumentParser()
    add_args(parser)
    return resolve_sgd_options(parser.parse_args(["--i", "particles.star", "--particle-diameter", "100", "-o", "out", *argv]))


def test_variant_presets_and_component_precedence():
    assert _options() == {} and _options("--optimizer", "momentum_sgd") == {}
    gated = _options("--optimizer", "momentum_sgd", "--sgd-variant", "gated")
    assert gated == {
        "sgd_learning_rate": 0.8,
        "sgd_halfset_gate": True,
        "sgd_step_scale": "exact_diagonal",
        "noise_anneal_factor": 8.0,
        "noise_anneal_updates": 750,
    }
    Config(optimizer="momentum_sgd", **gated)
    # A component given explicitly beats the variant's value.
    ablated = _options(
        "--optimizer", "momentum_sgd", "--sgd-variant", "gated", "--sgd-gate", "none", "--noise-anneal", "1,0",
        "--sgd-step-scale", "trace", "--sgd-learning-rate", "2.4",
    )  # fmt: skip
    assert ablated == {
        "sgd_learning_rate": 2.4,
        "sgd_halfset_gate": False,
        "sgd_step_scale": "trace",
        "noise_anneal_factor": 1.0,
        "noise_anneal_updates": 0,
    }
    assert _options("--optimizer", "momentum_sgd", "--sgd-gate", "halfset") == {"sgd_halfset_gate": True}
    # Noise annealing is an E-step option of either optimizer; the SGD components are not.
    assert _options("--noise-anneal", "4,100") == {"noise_anneal_factor": 4.0, "noise_anneal_updates": 100}
    for argv in (("--sgd-variant", "gated"), ("--sgd-gate", "none"), ("--sgd-step-scale", "trace")):
        with pytest.raises(ValueError, match="need --optimizer momentum_sgd"):
            _options(*argv)
    with pytest.raises(ValueError, match="F,N"):
        _options("--noise-anneal", "8")


def _state(theta, config):
    return State(
        theta,
        empty_moments(theta) if config.keeps_halfset_moments() else None,
        jnp.ones(3, jnp.float32),
        1,
        np.arange(8),
        np.random.default_rng(11).bit_generator.state,
        2.0,
        4,
        {"seed": 11},
        sgd_momentum=jnp.zeros_like(theta) if config.optimizer == "momentum_sgd" else None,
    )


def test_gated_checkpoint_keeps_moments_and_momentum_and_old_checkpoints_resume(tmp_path):
    gated = Config(optimizer="momentum_sgd", sgd_halfset_gate=True)
    theta = jnp.ones((5, 3), jnp.complex64)
    state = _state(theta, gated)
    state.sgd_momentum = jnp.full((5, 3), 0.25j, jnp.complex64)
    state.moments = Moments(jnp.full((2, 5, 3), 0.5, jnp.complex64), state.moments.second, jnp.ones((2, 5), bool))
    checkpoint.save(tmp_path / "gated.npz", state, gated, {"fixture": "t"})
    restored = checkpoint.load(tmp_path / "gated.npz", gated, {"fixture": "t"})
    assert_matches(restored.sgd_momentum, state.sgd_momentum)
    assert_matches(restored.moments.first, state.moments.first)
    with pytest.raises(ValueError, match="identity mismatch"):
        checkpoint.load(tmp_path / "gated.npz", Config(optimizer="momentum_sgd"), {"fixture": "t"})
    with pytest.raises(ValueError, match="half-set moments"):
        checkpoint.save(tmp_path / "bad.npz", _state(theta, Config(optimizer="momentum_sgd")), gated, {"fixture": "t"})
    # A checkpoint written before the options existed holds the model their defaults give.
    plain = Config(optimizer="momentum_sgd")
    checkpoint.save(tmp_path / "old.npz", _state(theta, plain), plain, {"fixture": "t"})
    with np.load(tmp_path / "old.npz") as arrays:
        saved = dict(arrays)
    metadata = json.loads(str(saved["metadata"]))
    for key in ("sgd_halfset_gate", "sgd_step_scale", "noise_anneal_factor", "noise_anneal_updates"):
        metadata["config"].pop(key)
    saved["metadata"] = json.dumps(metadata)
    np.savez(tmp_path / "old.npz", **saved)
    assert checkpoint.load(tmp_path / "old.npz", plain, {"fixture": "t"}).moments is None


def _exact_stats(ids, lhs_tri, gradient, exact, metric_trace=None):
    return ExactTracePPCAStats(
        rhs=None,
        lhs_tri=None if lhs_tri is None else jnp.asarray(lhs_tri),
        metric_trace=None if metric_trace is None else jnp.asarray(metric_trace),
        exact_metric_trace=jnp.asarray(exact),
        residual_gradient=jnp.asarray(gradient),
        residual_num=jnp.ones(3, jnp.float32),
        residual_den=jnp.ones(3, jnp.float32),
        original_image_ids=np.asarray(ids),
        n_images=len(ids),
        log_likelihood=0.0,
        embeddings=jnp.zeros((len(ids), 2)),
        diagnostics={
            "offset_second_sum_px2": 2.0,
            "rotation_mass": np.array([float(len(ids))]),
            "latent_covariance_trace_mean": 1.0,
            "pose_entropy_mean": 0.0,
            "pmax_mean": 1.0,
        },
    )


def test_merge_and_curvature_read_the_exact_trace():
    n = 4
    trace_only = [
        _exact_stats([0], None, np.zeros((n, 3), np.complex64), np.full(n, k + 1.0), np.full(n, 9.0)) for k in range(2)
    ]
    merged = iteration_loop._merge_statistics(trace_only)
    assert isinstance(merged, ExactTracePPCAStats) and merged.lhs_tri is None
    assert_matches(merged.exact_metric_trace, np.full(n, 3.0, np.float32))
    assert_matches(merged.metric_trace, np.full(n, 18.0, np.float32))
    packed = [
        _exact_stats([0], np.ones((n, 6), np.float32), np.zeros((n, 3), np.complex64), np.ones(n)) for _ in range(2)
    ]
    merged = iteration_loop._merge_statistics(packed)
    assert merged.metric_trace is None
    assert_matches(merged.lhs_tri, np.full((n, 6), 2.0, np.float32))
    assert_matches(iteration_loop._curvature_trace(packed, 3, exact=True), np.full(n, 2.0, np.float32))
    assert_matches(iteration_loop._curvature_trace(packed, 3, exact=False), np.full(n, 6.0, np.float32))
    with pytest.raises(ValueError, match="exact metric trace"):
        iteration_loop._curvature_trace([SimpleNamespace(lhs_tri=np.ones((n, 6)))], 3, exact=True)


def test_controller_gated_update_by_hand(tmp_path, monkeypatch):
    """One gated update with the exact-diagonal scale and annealing, from hand-computed half directions and gate."""
    shape = (4, 4, 4)
    n_freq = shape[0] * shape[1] * (shape[2] // 2 + 1)
    dc = int(np.argmin(np.asarray(ftu.get_grid_of_radial_distances_real(shape)).reshape(-1)))
    theta0 = np.zeros((n_freq, 3), np.complex64)
    theta0[dc] = [1.0, 0.5, 0.0]
    h = np.zeros((n_freq, 3, 3), np.float32)
    h[dc] = np.diag([2.0, 1.0, 1.0])
    half_gradients = [np.zeros((n_freq, 3), np.complex64) for _ in range(2)]
    half_gradients[0][dc] = [-1.0, 0.2, 0.0]
    half_gradients[1][dc] = [-0.6, 0.0, 0.0]
    exact = np.zeros(n_freq, np.float32)
    exact[dc] = 1.5
    monkeypatch.setattr(
        iteration_loop, "initialize", lambda *_a, **_k: (jnp.asarray(theta0), jnp.ones(3, jnp.float32), {"seed": 11})
    )
    monkeypatch.setattr(iteration_loop, "support_mask", lambda *_args: jnp.ones(shape, jnp.float32))
    monkeypatch.setattr(iteration_loop, "bandlimit_and_mask", lambda value, *_args: value)
    monkeypatch.setattr(iteration_loop.grids, "get_relion_hidden_rotation_grid", lambda *_a, **_k: np.zeros((1, 3)))
    monkeypatch.setattr(iteration_loop.grids, "rotation_direction_ids", lambda *_a, **_k: np.zeros(1, np.int64))
    iteration_loop._rotation_grid.cache_clear()
    iteration_loop._direction_ids.cache_clear()
    seen = []

    def groups(_dataset, _state, _config, halves, _iteration, *, noise_inflation=1.0, **_kwargs):
        seen.append(noise_inflation)
        return [_exact_stats(ids, _pack(h), g, exact) for ids, g in zip(halves, half_gradients, strict=True)]

    monkeypatch.setattr(iteration_loop, "expectation_groups", groups)
    dataset = SimpleNamespace(n_images=8, grid_size=4, volume_shape=shape, voxel_size=1.0)
    config = Config(
        iterations=1,
        stages=((1, 1, 0),),
        optimizer="momentum_sgd",
        sgd_learning_rate=0.8,
        sgd_halfset_gate=True,
        sgd_step_scale="exact_diagonal",
        noise_anneal_factor=8.0,
        noise_anneal_updates=750,
        skip_final_embeddings=True,
    )
    state = iteration_loop.run(dataset, config, tmp_path, {"fixture": "tiny"}, diameter_ang=2.0)
    assert seen == [pytest.approx(1 + 7 * (1 - 1 / 750))]
    fudge = config.schedule(1, 8)[2]
    # Half directions H^-1 G: (-0.5, 0.2, 0) and (-0.3, 0, 0); the first moments start at them.
    rho = 2 * fudge * np.array([1.0, 0.5 / np.sqrt(2)]) / np.array([0.2, 0.2 / np.sqrt(2)])
    gate = rho / (1 + rho)
    direction = np.array([gate[0] * -1.6 - (1 - gate[0]) * 4 * 1.0, gate[1] * 0.2 - (1 - gate[1]) * 2 * 0.5, 0])
    velocity = 0.1 * 0.8 * direction / 3.0  # exact trace 1.5 per half
    np.testing.assert_allclose(np.asarray(state.theta)[dc], theta0[dc] + velocity, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.asarray(state.sgd_momentum)[dc], velocity, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.asarray(state.moments.first)[:, dc], [[-0.5, 0.2, 0], [-0.3, 0, 0]], atol=1e-7)
    record = json.loads((tmp_path / "iterations.jsonl").read_text())
    assert record["noise_inflation"] == pytest.approx(seen[0]) and record["sgd_step_scale"] == "exact_diagonal"
    np.testing.assert_allclose(np.asarray(record["gates"])[0], gate, rtol=1e-6)
    assert record["curvature_scalar"] == pytest.approx(3.0)
    restored = checkpoint.load(tmp_path / "checkpoint_0001.npz", config, {"fixture": "tiny"})
    assert_matches(restored.moments.first, state.moments.first)
    assert_matches(restored.sgd_momentum, state.sgd_momentum)
