"""Per-class diagnostics and failure evidence do not alter the model update."""

import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from test_mixture_controller import tiny_config, tomo_data

from relax.ppca_initial_class3d import iteration_loop
from relax.ppca_initial_class3d.diagnostics import MetricFailure, PosteriorDiagnostics, model_summary

pytestmark = pytest.mark.unit


def test_model_summary_matches_independent_host_algebra():
    rng = np.random.default_rng(29)
    model = (rng.normal(size=(15, 3)) + 1j * rng.normal(size=(15, 3))).astype(np.complex64)
    actual = model_summary(jnp.asarray(model))
    assert_matches(actual["loading_singular_values"], np.linalg.svd(model[:, 1:], compute_uv=False))
    assert_matches(actual["mean_power"], np.sum(np.abs(model[:, 0]) ** 2))
    assert_matches(actual["loading_power"], np.sum(np.abs(model[:, 1:]) ** 2))


def test_posterior_aggregates_joint_contributions_by_particle_count():
    diagnostics = PosteriorDiagnostics()
    for n, scale in ((3, 1), (1, 2)):
        diagnostics.add(SimpleNamespace(n_images=n, diagnostics={
            "pmax_mean": 0.1 * scale, "pose_entropy_mean": 0.2 * scale,
            "latent_covariance_trace_mean": 0.3 * scale,
            "scored_rows": 20, "pass2_rows": 10 * scale,
            "tile_size": n, "gemm_precision": "fp32",
        }))
    record = diagnostics.record()
    assert record["particle_count"] == 4
    assert_matches(record["pmax_mean"], 0.125)
    assert_matches(record["pose_entropy_mean"], 0.25)
    assert_matches(record["latent_covariance_trace_mean"], 0.375)
    assert_matches(record["pass2_row_fraction"], 0.75)
    assert record["tile_sizes"] == [1, 3]
    assert record["gemm_precision"] == ["fp32"]


def test_k2_records_optimizer_and_posterior_diagnostics(tmp_path):
    iteration_loop.run(tomo_data(), tiny_config(2), tmp_path, {"case": "diagnostics"}, 6.)
    records = [json.loads(line) for line in (tmp_path / "iterations.jsonl").read_text().splitlines()]
    assert len(records) == 2
    for record in records:
        assert len(record["class_diagnostics"]) == 2
        assert record["gemm_precision"] == ["fp32"]
        assert record["vdam_step_factor"] > 0
        assert np.all(np.isfinite(record["direction_prior"]))
        for k, component in enumerate(record["class_diagnostics"]):
            assert component["class_index"] == k
            for name in ("gates", "signal", "disagreement", "loading_singular_values", "mean_power", "loading_power"):
                assert np.all(np.isfinite(component[name]))
            assert component["zero_disagreement_shell_channels"] >= 0
        for half in record["posterior"]:
            assert len(half) == 2
            for component in half:
                assert component["particle_count"] == 2
                assert np.isfinite(component["pose_entropy_mean"])
                assert component["tile_sizes"]


def test_metric_failure_saves_offending_class_and_half(tmp_path, monkeypatch):
    original = iteration_loop.coupled_direction
    calls = []

    def fail_fourth(lhs, gradient, support, *, floor):
        calls.append((np.asarray(lhs), np.asarray(gradient)))
        if len(calls) == 4:
            raise ValueError("injected metric failure")
        return original(lhs, gradient, support, floor=floor)

    monkeypatch.setattr(iteration_loop, "coupled_direction", fail_fourth)
    with pytest.raises(MetricFailure, match="Class 1, half 1: injected metric failure"):
        iteration_loop.run(tomo_data(), tiny_config(2), tmp_path, {"case": "metric-failure"}, 6.)
    assert (tmp_path / "failure_before_0001.npz").is_file()
    with np.load(tmp_path / "failure_metric_0001_class001_half1.npz") as failure:
        assert_matches(failure["lhs_tri"], calls[-1][0])
        assert_matches(failure["residual_gradient"], calls[-1][1])
        assert np.all(failure["half_particle_ids"] % 2 == 1)
        assert failure["support"].dtype == np.bool_
    assert not (tmp_path / "failure_noise_0001.npz").exists()
