"""Donor reconstruction ownership and authoritative per-half shell priors."""

import dataclasses
import inspect

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from recovar.core import fourier_transform_utils as ftu
from recovar.reconstruction import relion_functions as rf

from relax.reconstruction import regularization_relion

pytestmark = pytest.mark.unit

VOLUME_SHAPE = (8, 8, 8)
VOLUME_SIZE = 512


def test_mean_reconstruction_variants_share_run_level_settings():
    from relax.refinement import mean_helpers as mean_helpers_module

    assert tuple(inspect.signature(mean_helpers_module.reconstruct_numbered_k1_halfmaps).parameters) == (
        "numerators_by_half", "denominators_by_half", "tau_by_half", "settings", "iteration",
        "current_size", "accumulator_volume_shape",
        "relion_firstiter_cc_this_iter", "retained_first_numerator", "observer",
    )
    assert tuple(inspect.signature(mean_helpers_module.reconstruct_numbered_class_maps).parameters) == (
        "combined_numerators", "combined_denominators", "tau_by_class", "settings", "n_classes",
        "iteration", "current_size", "accumulator_volume_shape",
        "relion_firstiter_cc_this_iter", "observer",
    )
    assert tuple(field.name for field in dataclasses.fields(mean_helpers_module.ReconstructionSettings)) == (
        "box_size", "voxel_size", "volume_shape", "padding_factor",
        "projection_padding_factor", "minres_map", "width_mask_edge", "fmask_edge",
        "tau2_fudge", "particle_diameter_angstrom", "first_iteration_lowpass_angstrom",
        "gridding_kernel", "shell_pair_counting",
        "solvent_mask", "solvent_correct_fsc", "solvent_fsc_seed",
    )
    for name in (
        "MeanReconstructionData", "MeanAccumulatorState", "MeanPriorSpec",
        "MeanGeometrySpec", "MeanPostprocessPolicy",
    ):
        assert not hasattr(mean_helpers_module, name)

    assert not hasattr(mean_helpers_module, "_postprocess_numbered_maps")


_NUMBERED_SEQUENCES = {
    1: ("k1_maximization", "reconstruct_numbered_k1_halfmaps", "_reconstruct_k1_maps",
        "_apply_relion_initial_lowpass_filter", "_apply_relion_solvent_flatten_k1"),
    2: ("class_maximization", "reconstruct_numbered_class_maps", "_reconstruct_class_maps",
        "_lowpass_class_stack", "_flatten_class_stack"),
}


@pytest.mark.parametrize("n_classes", [1, 2])
def test_numbered_reconstruction_sequence_and_owners(n_classes, monkeypatch, tmp_path):
    """One run-level ReconstructionSettings reaches every M-step; each mode's operation solves, then per slot
    captures, low-pass filters (CC iteration only), masks and flattens, and reports the CC low-pass once."""
    from helpers.tiny_refinement import CallTrace, run_tiny_refinement

    from relax.diagnostics import observers
    from relax.refinement import iteration_loop as iteration_loop_module
    from relax.refinement import mean_helpers as mean_helpers_module

    maximization, operation, solve, lowpass, flatten = _NUMBERED_SEQUENCES[n_classes]
    monkeypatch.setattr(observers, "write_premask_mean", lambda *args, **kwargs: None)
    trace = CallTrace(monkeypatch)
    trace.wrap(iteration_loop_module, "ReconstructionSettings", "settings")
    trace.wrap(iteration_loop_module, maximization, "maximization")
    trace.wrap(iteration_loop_module, operation, "operation")
    for name, label in (
        (solve, "solve"), (lowpass, "lowpass"),
        ("_numbered_solvent_mask", "mask"), ("_make_relion_solvent_mask", "mask_builder"), (flatten, "flatten"),
        ("_log_first_cc_lowpass", "log"),
    ):
        trace.wrap(mean_helpers_module, name, label)
    trace.wrap(observers.PremaskObserver, "map_solved", "capture")
    trace.wrap(observers, "write_premask_mean", "dump")
    run_tiny_refinement(
        monkeypatch, n_classes=n_classes, final_after_max_iter=False, observer=observers.PremaskObserver(tmp_path),
        schedule=dict(particle_diameter_ang=6.0),
        parity=dict(emulate_relion_firstiter_cc=True, relion_firstiter_ini_high_angstrom=8.0),
    )

    (settings,) = trace.calls("settings")
    assert [call.args[3] for call in trace.calls("maximization")] == [settings.result] * 2
    assert all(call.inside == ("maximization",) for call in trace.calls("operation"))
    slot = ["capture", "dump", "mask", "mask_builder", "flatten"]
    cc_slot = ["capture", "dump", "lowpass", "mask", "mask_builder", "flatten"]
    in_operation = [call.label for call in trace.calls_seen if "operation" in call.inside]
    assert in_operation == ["solve", *cc_slot, *cc_slot, "log", "solve", *slot, *slot]
    assert all(call.inside[-1] == "capture" for call in trace.calls("dump"))
    assert [call.kwargs["half_index"] for call in trace.calls("dump")] == [0, 1, 0, 1]
    # The particle-diameter mask: radius in pixels, with the soft edge outside it.
    for call in trace.calls("mask_builder"):
        assert call.inside[-1] == "mask"
        assert call.kwargs["radius"] == 6.0 / (2.0 * settings.result.voxel_size)
        assert call.kwargs["radius_p"] == call.kwargs["radius"] + settings.result.width_mask_edge


def test_unregularized_reconstruction_variants_expose_dependencies():
    from relax.refinement import mean_helpers as mean_helpers_module

    assert tuple(inspect.signature(mean_helpers_module.reconstruct_unregularized_k1_halfmaps).parameters) == (
        "Ft_y_per_half", "Ft_ctf_per_half", "settings",
        "accumulator_volume_shape",
    )
    assert tuple(inspect.signature(mean_helpers_module.reconstruct_unregularized_class_means).parameters) == (
        "Ft_y_combined", "Ft_ctf_combined", "settings", "n_classes",
        "accumulator_volume_shape",
    )
    assert tuple(inspect.signature(mean_helpers_module.align_k1_volume_signs).parameters) == (
        "means", "previous_means", "unregularized_means", "volume_shape",
    )
    assert tuple(inspect.signature(mean_helpers_module.share_kclass_volume_signs).parameters) == (
        "means", "unregularized_means",
    )
    for name in (
        "UnregularizedMeanState",
        "UnregularizedAccumulatorState",
        "UnregularizedReconstructionPolicy",
    ):
        assert not hasattr(mean_helpers_module, name)


class TestReconstructionOwnership:
    def test_k1_reconstruction_uses_per_half_1d_tau_shell_prior(self, monkeypatch):
        """K=1 reconstruction should not round-trip tau2 through full volumes."""

        from relax.refinement import mean_helpers as mean_helpers_module

        calls = []
        events = []

        def fake_reconstruct(*_args, retained_device_numerator=None, **kwargs):
            kwargs["retained_device_numerator"] = retained_device_numerator
            calls.append(kwargs)
            events.append("reconstruct")
            return jnp.ones(VOLUME_SIZE, dtype=jnp.complex128)

        def fake_finish(result, *_accumulators):
            events.append("finish")
            return result

        monkeypatch.setattr(mean_helpers_module, "_reconstruct_volume_eager", fake_reconstruct)
        monkeypatch.setattr(mean_helpers_module, "_finish_host_staged_reconstruction", fake_finish)
        n_shells = VOLUME_SHAPE[0] // 2 + 1
        tau_shells = [jnp.arange(n_shells, dtype=jnp.float32) + 101.0, jnp.arange(n_shells, dtype=jnp.float32) + 201.0]
        retained_half0 = object()
        settings = mean_helpers_module.ReconstructionSettings(
            box_size=8,
            voxel_size=1.0,
            volume_shape=VOLUME_SHAPE,
            padding_factor=1,
            projection_padding_factor=1,
            minres_map=0,
            width_mask_edge=5,
            fmask_edge=2,
            tau2_fudge=1.0,
            particle_diameter_angstrom=None,
            first_iteration_lowpass_angstrom=None,
        )
        means = mean_helpers_module.reconstruct_numbered_k1_halfmaps(
            (jnp.ones(VOLUME_SIZE, dtype=jnp.complex64), jnp.ones(VOLUME_SIZE, dtype=jnp.complex64)),
            (jnp.ones(VOLUME_SIZE, dtype=jnp.float32), jnp.ones(VOLUME_SIZE, dtype=jnp.float32)),
            tau_shells,
            settings,
            iteration=0,
            current_size=8,
            accumulator_volume_shape=None,
            relion_firstiter_cc_this_iter=False,
            retained_first_numerator=retained_half0,
        )
        assert len(calls) == 2
        assert events == ["reconstruct", "finish", "reconstruct", "finish"]
        assert calls[0]["retained_device_numerator"] is retained_half0
        assert calls[1]["retained_device_numerator"] is None
        assert all((call["tau_is_1d"] is True for call in calls))
        assert all((call["tau"].dtype == jnp.float64 for call in calls))
        assert_matches(np.asarray(calls[0]["tau"]), np.asarray(tau_shells[0]))
        assert_matches(np.asarray(calls[1]["tau"]), np.asarray(tau_shells[1]))
        assert means[0].shape == (VOLUME_SIZE,)
        assert means[1].shape == (VOLUME_SIZE,)

    def test_host_staged_k1_reconstruction_blocks_before_next_half(self, monkeypatch):
        """Host-staged box-scale reconstruction must serialize its FFT workspace."""
        from relax.refinement import mean_helpers as mean_helpers_module

        events = []

        class Result:
            def block_until_ready(self):
                events.append("block")

        monkeypatch.setattr(mean_helpers_module.gc, "collect", lambda: events.append("collect"))
        result = Result()
        returned = mean_helpers_module._finish_host_staged_reconstruction(
            result, np.ones(1, dtype=np.float32), jnp.ones(1, dtype=jnp.float32)
        )
        assert returned is result
        assert events == ["block", "collect"]
        events.clear()
        returned = mean_helpers_module._finish_host_staged_reconstruction(
            result, jnp.ones(1, dtype=jnp.float32), jnp.ones(1, dtype=jnp.float32)
        )
        assert returned is result
        assert events == []

    def test_large_host_reconstruction_reuses_retained_numerator_and_releases_stage_a(self, monkeypatch, caplog):
        """The retained half-0 buffer must feed Stage A and release before the iFFT."""
        from recovar.reconstruction import relion_functions

        from relax.reconstruction import relion_functions_relion
        from relax.refinement import mean_helpers as mean_helpers_module

        events = []
        host_boundary = np.ones((5, 5, 3), dtype=np.complex64)

        class DeviceBoundary:
            def block_until_ready(self):
                events.append("block")

            def __del__(self):
                events.append("release")

        host_ctf = np.ones((5, 5, 3), dtype=np.float32)
        host_numerator = np.ones((5, 5, 3), dtype=np.complex64)
        retained_numerator = jnp.ones((5, 5, 3), dtype=jnp.complex64)
        regularized_filter = jnp.ones(host_ctf.shape, dtype=jnp.float32)

        def fake_regularize(stage_filter, *_args):
            events.append("regularize")
            stage_filter.delete()
            return regularized_filter

        def fake_divide(stage_numerator, stage_filter, *_args):
            events.append("divide")
            assert stage_numerator is retained_numerator
            assert stage_filter is regularized_filter
            return DeviceBoundary()

        def fake_device_get(_value):
            events.append("device_get")
            return host_boundary

        def fake_finish(value, *_args, **kwargs):
            events.append("finish")
            assert events == ["regularize", "divide", "block", "device_get", "release", "collect", "collect", "finish"]
            assert value.shape == (4, 4, 3)
            assert_matches(value, np.ones((4, 4, 3), dtype=np.complex64))
            assert kwargs["gridding_correct"] == "radial"
            return host_boundary

        monkeypatch.setattr(relion_functions, "_large_grid_postprocess_single_precision_enabled", lambda _voxels: True)
        monkeypatch.setattr(relion_functions_relion, "_regularize_large_relion_half_filter_donate_ctf", fake_regularize)
        monkeypatch.setattr(relion_functions_relion, "_divide_large_relion_half_numerator_donate_numerator", fake_divide)
        monkeypatch.setattr(relion_functions_relion, "_finish_large_relion_postprocess_from_fftw_half", fake_finish)
        monkeypatch.setattr(mean_helpers_module.jax, "device_get", fake_device_get)
        monkeypatch.setattr(mean_helpers_module.gc, "collect", lambda: events.append("collect"))
        caplog.set_level("INFO", logger=mean_helpers_module.__name__)
        volume_shape = (2, 2, 2)
        accumulator_shape = (5, 5, 5)
        half_shape = ftu.volume_shape_to_half_volume_shape(accumulator_shape)
        assert half_shape == host_numerator.shape
        returned = mean_helpers_module._reconstruct_volume_eager(
            host_ctf,
            host_numerator,
            volume_shape,
            2,
            tau=np.ones(np.prod(volume_shape), dtype=np.float32),
            tau2_fudge=1.0,
            projection_padding_factor=1,
            accumulator_volume_shape=accumulator_shape,
            retained_device_numerator=retained_numerator,
        )
        assert returned is host_boundary
        assert events == ["regularize", "divide", "block", "device_get", "release", "collect", "collect", "finish"]
        assert (
            "RELION split pre-IFFT host boundary: accumulator_shape=(5, 5, 5) reconstruction_shape=(4, 4, 4) packed_half_bytes=384"
            in caplog.text
        )

    def test_large_host_reconstruction_stages_numpy_numerator_for_donation(self, monkeypatch, caplog):
        """Half 2 must see half 1 freed, then stage/delete its host numerator."""
        from recovar.reconstruction import relion_functions

        from relax.reconstruction import relion_functions_relion
        from relax.refinement import mean_helpers as mean_helpers_module

        volume_shape = (2, 2, 2)
        accumulator_shape = (5, 5, 5)
        half_shape = ftu.volume_shape_to_half_volume_shape(accumulator_shape)
        host_ctf = np.ones(half_shape, dtype=np.float32)
        host_numerator = np.ones(half_shape, dtype=np.complex64)
        retained_numerator = jnp.ones(half_shape, dtype=jnp.complex64)
        stage_inputs = []
        stage_outputs = []
        regularized_filters = []
        sentinel = jnp.asarray([7.0 + 0j], dtype=jnp.complex64)

        def fake_regularize(stage_filter, *_args):
            assert isinstance(stage_filter, mean_helpers_module.jax.Array)
            stage_filter.delete()
            regularized = jnp.ones(half_shape, dtype=jnp.float32)
            regularized_filters.append(regularized)
            return regularized

        def fake_divide(stage_numerator, regularized_filter, *_args):
            assert regularized_filter is regularized_filters[-1]
            if not stage_inputs:
                assert stage_numerator is retained_numerator
            else:
                assert retained_numerator.is_deleted()
                assert stage_outputs[0].is_deleted()
                assert isinstance(stage_numerator, mean_helpers_module.jax.Array)
                assert not isinstance(stage_numerator, np.ndarray)
            assert not stage_numerator.is_deleted()
            stage_inputs.append(stage_numerator)
            stage_outputs.append(jnp.ones(half_shape, dtype=jnp.complex64))
            return stage_outputs[-1]

        def fake_finish(value, *_args, **_kwargs):
            assert isinstance(value, np.ndarray)
            assert value.shape == (4, 4, 3)
            return sentinel

        monkeypatch.setattr(relion_functions, "_large_grid_postprocess_single_precision_enabled", lambda _voxels: True)
        monkeypatch.setattr(relion_functions_relion, "_regularize_large_relion_half_filter_donate_ctf", fake_regularize)
        monkeypatch.setattr(relion_functions_relion, "_divide_large_relion_half_numerator_donate_numerator", fake_divide)
        monkeypatch.setattr(relion_functions_relion, "_finish_large_relion_postprocess_from_fftw_half", fake_finish)
        caplog.set_level("INFO", logger=mean_helpers_module.__name__)
        half0 = mean_helpers_module._reconstruct_volume_eager(
            host_ctf,
            host_numerator,
            volume_shape,
            2,
            tau=np.ones(np.prod(volume_shape), dtype=np.float32),
            tau2_fudge=1.0,
            projection_padding_factor=1,
            accumulator_volume_shape=accumulator_shape,
            retained_device_numerator=retained_numerator,
        )
        half1 = mean_helpers_module._reconstruct_volume_eager(
            host_ctf,
            host_numerator,
            volume_shape,
            2,
            tau=np.ones(np.prod(volume_shape), dtype=np.float32),
            tau2_fudge=1.0,
            projection_padding_factor=1,
            accumulator_volume_shape=accumulator_shape,
        )
        assert half0 is sentinel
        assert half1 is sentinel
        assert len(stage_inputs) == len(stage_outputs) == 2
        assert len(regularized_filters) == 2
        assert all((value.is_deleted() for value in stage_inputs))
        assert all((value.is_deleted() for value in stage_outputs))
        assert all((value.is_deleted() for value in regularized_filters))
        assert "RELION Stage A staging host numerator for donation" in caplog.text
        assert "source=staged_numpy output_deleted=True numerator_deleted=True filter_deleted=True" in caplog.text


def test_relion_reconstruction_tau_shells_match_full_prior_bitwise():
    """Authoritative tau2 shells reproduce the legacy full-prior result exactly."""
    volume_shape = (8, 8, 8)
    accumulator_shape = (16, 16, 16)
    half_shape = (16, 16, 9)
    rng = np.random.default_rng(20260831)
    weight = (0.5 + rng.random(half_shape)).astype(np.float32)
    numerator = (rng.standard_normal(half_shape) + 1j * rng.standard_normal(half_shape)).astype(np.complex64)
    fsc = np.linspace(0.9, 0.1, volume_shape[0] // 2 + 1, dtype=np.float64)
    tau_full, _, details = regularization_relion.compute_relion_tau2_from_weights(
        weight,
        weight,
        fsc,
        volume_shape,
        padding_factor=2,
        r_max=volume_shape[0] // 2,
        return_details=True,
        accumulator_volume_shape=accumulator_shape,
    )
    common = {
        "kernel": "triangular",
        "use_spherical_mask": False,
        "grid_correct": False,
        "current_size": volume_shape[0],
        "accumulator_volume_shape": accumulator_shape,
        "input_half_volume": True,
        "preserve_output_precision": True,
    }
    from_full = rf.post_process_from_filter_v2(
        jnp.asarray(weight),
        jnp.asarray(numerator),
        volume_shape,
        2,
        tau=jnp.asarray(tau_full, dtype=jnp.float64),
        tau_is_1d=False,
        **common,
    )
    from_shells = rf.post_process_from_filter_v2(
        jnp.asarray(weight),
        jnp.asarray(numerator),
        volume_shape,
        2,
        tau=jnp.asarray(details["prior_shells"], dtype=jnp.float64),
        tau_is_1d=True,
        **common,
    )
    assert_matches(np.asarray(from_shells), np.asarray(from_full))


def test_k1_numpy_join_reservation_reaches_first_stage_a_only(monkeypatch):
    """Production host join must hand one live exact buffer to half-0 Stage A."""

    from relax.refinement import mean_helpers as mean_helpers_module

    accumulator_shape = (9, 9, 9)
    half_shape = ftu.volume_shape_to_half_volume_shape(accumulator_shape)
    rng = np.random.default_rng(20260901)
    ft_y_0 = (rng.standard_normal(half_shape) + 1j * rng.standard_normal(half_shape)).astype(np.complex64)
    ft_y_1 = (rng.standard_normal(half_shape) + 1j * rng.standard_normal(half_shape)).astype(np.complex64)
    ft_ctf_0 = rng.uniform(0.5, 1.5, half_shape).astype(np.float32)
    ft_ctf_1 = rng.uniform(0.5, 1.5, half_shape).astype(np.float32)
    monkeypatch.setenv("RELAX_LOWRES_JOIN_HOST_FALLBACK", "always")
    joined = regularization_relion.join_halves_at_low_resolution(
        ft_y_0,
        ft_y_1,
        ft_ctf_0,
        ft_ctf_1,
        volume_shape=accumulator_shape,
        voxel_size=10.0,
        grid_size=4,
        low_resol_join_halves_angstrom=40.0,
        padding_factor=2,
        preserve_inputs=False,
        return_retained_first_numerator=True,
    )
    retained_half0 = joined[4]
    assert retained_half0 is not None
    assert_matches(np.asarray(retained_half0), joined[0])
    calls = []

    def fake_reconstruct(*args, retained_device_numerator=None, **kwargs):
        kwargs["retained_device_numerator"] = retained_device_numerator
        calls.append((args, kwargs))
        return jnp.ones(4**3, dtype=jnp.complex128)

    monkeypatch.setattr(mean_helpers_module, "_reconstruct_volume_eager", fake_reconstruct)
    monkeypatch.setattr(
        mean_helpers_module, "_finish_host_staged_reconstruction", lambda result, *_accumulators: result
    )
    settings = mean_helpers_module.ReconstructionSettings(
        box_size=4,
        voxel_size=1.0,
        volume_shape=(4, 4, 4),
        padding_factor=2,
        projection_padding_factor=1,
        minres_map=0,
        width_mask_edge=5,
        fmask_edge=2,
        tau2_fudge=1.0,
        particle_diameter_angstrom=None,
        first_iteration_lowpass_angstrom=None,
    )
    means = mean_helpers_module.reconstruct_numbered_k1_halfmaps(
        (joined[0], joined[1]),
        (joined[2], joined[3]),
        [jnp.ones(3, dtype=jnp.float32), jnp.ones(3, dtype=jnp.float32)],
        settings,
        iteration=0,
        current_size=4,
        accumulator_volume_shape=accumulator_shape,
        relion_firstiter_cc_this_iter=False,
        retained_first_numerator=retained_half0,
    )
    assert len(calls) == 2
    assert calls[0][0][1] is joined[0]
    assert calls[0][1]["retained_device_numerator"] is retained_half0
    assert calls[1][1]["retained_device_numerator"] is None


def test_host_join_threshold_counts_the_physical_grid_of_a_packed_half(monkeypatch):
    """EMPIAR-10202 it3 (14456981): a 611^3 accumulator stored as a packed half has
    114M elements but 228M voxels. The join counted stored elements, moved the host
    halves to the device, and the reconstruction took its monolithic device 1600^3
    iFFT. The join now counts the physical grid, the reconstruction's unit."""

    accumulator_shape = (9, 9, 9)
    half_shape = ftu.volume_shape_to_half_volume_shape(accumulator_shape)
    stored, voxels = int(np.prod(half_shape)), int(np.prod(accumulator_shape))
    threshold = (stored + voxels) // 2
    assert stored < threshold <= voxels
    monkeypatch.setenv("RELAX_LOWRES_JOIN_HOST_FALLBACK", "auto")
    monkeypatch.setenv("RELAX_LOWRES_JOIN_HOST_FALLBACK_MIN_ELEMENTS", str(threshold))
    rng = np.random.default_rng(20260926)
    ft_y_0 = (rng.standard_normal(half_shape) + 1j * rng.standard_normal(half_shape)).astype(np.complex64)
    ft_y_1 = (rng.standard_normal(half_shape) + 1j * rng.standard_normal(half_shape)).astype(np.complex64)
    ft_ctf_0 = rng.uniform(0.5, 1.5, half_shape).astype(np.float32)
    ft_ctf_1 = rng.uniform(0.5, 1.5, half_shape).astype(np.float32)
    kwargs = dict(
        volume_shape=accumulator_shape,
        voxel_size=10.0,
        grid_size=4,
        low_resol_join_halves_angstrom=40.0,
        padding_factor=2,
    )
    host = regularization_relion.join_halves_at_low_resolution(ft_y_0, ft_y_1, ft_ctf_0, ft_ctf_1, **kwargs)
    assert all(isinstance(value, np.ndarray) for value in host)
    monkeypatch.setenv("RELAX_LOWRES_JOIN_HOST_FALLBACK", "never")
    device = regularization_relion.join_halves_at_low_resolution(ft_y_0, ft_y_1, ft_ctf_0, ft_ctf_1, **kwargs)
    for h, d in zip(host, device, strict=True):
        np.testing.assert_allclose(np.asarray(h), np.asarray(d), rtol=1e-6, atol=0)


@pytest.mark.parametrize("old_dim,new_dim", [(9, 16), (10, 16), (11, 15)])
def test_host_wiener_pad_equals_the_device_fftw_pad(old_dim, new_dim):
    """The host pad of a large accumulator's Wiener half (EMPIAR-10202, 823^3-1163^3
    into 1600^3) must write exactly what recovar's device pad writes."""

    from relax.refinement import mean_helpers

    old_shape, new_shape = (old_dim,) * 3, (new_dim,) * 3
    half_shape = ftu.volume_shape_to_half_volume_shape(old_shape)
    rng = np.random.default_rng(old_dim * 100 + new_dim)
    wiener = (rng.standard_normal(half_shape) + 1j * rng.standard_normal(half_shape)).astype(np.complex64)
    device = np.asarray(rf._relion_pad_centered_half_fourier_to_fftw(jnp.asarray(wiener), old_shape, new_shape))
    host = mean_helpers._pad_relion_wiener_half_to_fftw_host(wiener, old_shape, new_shape, rf)
    assert host.dtype == device.dtype and host.shape == device.shape
    assert_matches(host, device)
    # Zero pattern (support and placement) is discrete: exact.
    assert np.array_equal(host == 0, device == 0)
