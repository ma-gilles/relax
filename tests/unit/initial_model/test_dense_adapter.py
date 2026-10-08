"""InitialModel dense K-class E-step adapter tests."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.local.local_layout import LocalHypothesisLayout
from relax.vdam.adaptive_estep import _resolve_sparse_pass1_current_size, _safe_coarse_significance_image_batch_size
from relax.vdam.bootstrap_iref import initialise_denovo_state
from relax.vdam.dense_adapter import (
    _resolve_class_inputs,
    prepare_relion_projector_class_inputs_and_power,
    run_dense_initial_model_estep,
)
from relax.vdam.estep_common import (
    DenseInitialModelEstepConfig,
    DenseInitialModelEstepResult,
    _arrays_to_accumulators,
    _estep_meta,
    relion_bpref_frame_scales,
)
from relax.vdam.state import InitialModelState

pytestmark = pytest.mark.unit


def test_coarse_significance_batch_is_unchanged_for_small_pose_grids():
    assert _safe_coarse_significance_image_batch_size(
        500,
        n_classes=1,
        n_rotations=4608,
        n_translations=45,
    ) == 500


@pytest.mark.parametrize(
    ("n_translations", "expected"),
    [(37, 146), (45, 120)],
)
def test_coarse_significance_batch_caps_gui_default_oom_grids(
    n_translations,
    expected,
):
    assert _safe_coarse_significance_image_batch_size(
        500,
        n_classes=1,
        n_rotations=36864,
        n_translations=n_translations,
    ) == expected


def test_coarse_significance_batch_respects_smaller_user_request_and_k_axis():
    assert _safe_coarse_significance_image_batch_size(
        64,
        n_classes=1,
        n_rotations=36864,
        n_translations=45,
    ) == 64
    assert _safe_coarse_significance_image_batch_size(
        500,
        n_classes=4,
        n_rotations=36864,
        n_translations=45,
    ) == 30


class _Dataset:
    n_images = 4

    def subset(self, image_indices):
        n_images = int(np.asarray(image_indices).size)
        return SimpleNamespace(n_images=n_images, n_units=n_images)


class _ReplaceableNamespace(SimpleNamespace):
    """Small mutable result double with the NamedTuple ``_replace`` contract."""

    def _replace(self, **updates):
        values = vars(self).copy()
        values.update(updates)
        return type(self)(**values)




def _identity_local_layout(*, n_images, rotations_per_image, n_translations):
    n_rotations = n_images * rotations_per_image
    return LocalHypothesisLayout(
        n_global_rotations=1,
        n_pixels=1,
        n_psi=1,
        rotation_offsets=np.arange(n_images + 1, dtype=np.int64) * rotations_per_image,
        rotation_ids_flat=np.zeros(n_rotations, dtype=np.int32),
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (n_rotations, 3, 3)).copy(),
        rotation_log_priors_flat=np.zeros(n_rotations, dtype=np.float32),
        rotation_counts=np.full(n_images, rotations_per_image, dtype=np.int32),
        translation_grid=np.zeros((n_translations, 2), dtype=np.float32),
        translation_log_priors=np.zeros((n_images, n_translations), dtype=np.float32),
        rotation_posterior_ids_flat=np.zeros(n_rotations, dtype=np.int32),
    )


def _fake_noise_stats(offset: float, sumw: float, wsum_noise, img_power):
    return SimpleNamespace(
        wsum_sigma2_offset=float(offset),
        sumw=float(sumw),
        wsum_sigma2_noise=np.asarray(wsum_noise, dtype=np.float32),
        wsum_img_power=np.asarray(img_power, dtype=np.float32),
    )


def test_arrays_to_accumulators_inverts_relion_x_public_layout_without_projector_flip():
    from relax.helpers.half_volume_mstep import (
        enforce_relion_half_volume_x0_hermitian_host,
        relion_x_half_volume_to_full,
    )
    state = initialise_denovo_state(
        ori_size=8,
        pixel_size=1.0,
        K=1,
        nr_iter=1,
        n_directions=4,
    )
    state.current_size = 4
    compact_shape = (7, 7, 7)
    half_shape = (7, 7, 4)
    rng = np.random.default_rng(93)
    bp_data = (
        rng.standard_normal(half_shape) + 1j * rng.standard_normal(half_shape)
    ).astype(np.complex64)
    bp_weight = rng.uniform(1e-3, 2.0, size=half_shape).astype(np.float32)
    bp_data = enforce_relion_half_volume_x0_hermitian_host(
        bp_data.reshape(-1), compact_shape
    ).reshape(half_shape)
    bp_weight = enforce_relion_half_volume_x0_hermitian_host(
        bp_weight.reshape(-1), compact_shape
    ).reshape(half_shape)
    public_data = relion_x_half_volume_to_full(bp_data.reshape(-1), compact_shape)
    public_weight = relion_x_half_volume_to_full(bp_weight.reshape(-1), compact_shape)

    actual = _arrays_to_accumulators(
        [public_data],
        [public_weight],
        state,
        halfset_idx=0,
        padding_factor=1,
    )[0]

    data_scale, weight_scale = relion_bpref_frame_scales(state.ori_size)
    assert_matches(actual.data, bp_data.astype(np.complex128) * data_scale)
    assert_matches(actual.weight, bp_weight.astype(np.float64) * weight_scale)


def test_arrays_to_accumulators_splits_grouped_halfsets():
    state = initialise_denovo_state(
        ori_size=8,
        pixel_size=1.0,
        K=1,
        nr_iter=1,
        n_directions=4,
    )
    state.current_size = 4
    public_size = 7 * 7 * 7
    grouped_data = np.stack(
        [
            np.full(public_size, 1.0 + 2.0j, dtype=np.complex64),
            np.full(public_size, 3.0 + 4.0j, dtype=np.complex64),
        ]
    )
    grouped_weight = np.stack(
        [
            np.full(public_size, 5.0, dtype=np.float32),
            np.full(public_size, 7.0, dtype=np.float32),
        ]
    )

    actual = _arrays_to_accumulators(
        grouped_data[None, ...],
        grouped_weight[None, ...],
        state,
        halfset_idx=None,
        reconstruction_group_count=2,
        padding_factor=1,
    )

    assert [value.halfset_idx for value in actual] == [0, 1]
    assert [value.class_idx for value in actual] == [0, 0]
    assert not np.array_equal(actual[0].data, actual[1].data)
    assert not np.array_equal(actual[0].weight, actual[1].weight)


def _capture_adaptive_route(monkeypatch):
    """Replace the adaptive E-step and return the keyword arguments it was called with."""
    calls = []

    def fake_route(dataset, state, config, **kwargs):
        calls.append(kwargs)
        return DenseInitialModelEstepResult(accumulators=[], meta={})

    monkeypatch.setattr("relax.vdam.dense_adapter.run_adaptive_initial_model_estep", fake_route)
    return calls


def _projector_config(n_classes, **overrides):
    values = dict(
        noise_variance=np.ones(8 * 8, dtype=np.float32),
        translations=np.zeros((1, 2), dtype=np.float32),
        relion_projector_half_by_class=np.zeros((n_classes, 4), dtype=np.complex64),
        relion_projector_r_max=3,
    )
    values.update(overrides)
    return DenseInitialModelEstepConfig(**values)


def test_initial_model_estep_hands_the_subset_halves_and_class_priors_to_the_adaptive_route(monkeypatch):
    calls = _capture_adaptive_route(monkeypatch)
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=2, nr_iter=1, n_directions=4)
    state.current_size = 8
    state.pdf_class = np.asarray([0.75, 0.25])

    result = run_dense_initial_model_estep(
        _Dataset(),
        state,
        _projector_config(2),
        particle_ids=np.asarray([3, 0, 2]),
        halfset_ids=np.asarray([1, 0, 1], dtype=np.int8),
    )

    # One pass over the subset: each particle's pseudo-halfset goes with it.
    assert len(calls) == 1
    assert_matches(calls[0]["joint_particle_ids"], [3, 0, 2])
    assert_matches(calls[0]["joint_halfset_ids"], [1, 0, 1])
    # Class priors travel in the joint class/direction prior: the engine's own class term is zero.
    assert_matches(calls[0]["class_log_priors"], np.zeros(2))
    assert calls[0]["engine_kwargs"]["current_size"] == 8
    assert calls[0]["relion_projector_r_max"] == 3
    assert "pass2_engines" in result.meta and "coarse_engine_calls" in result.meta


def test_initial_model_estep_defaults_to_every_particle_in_alternating_halves(monkeypatch):
    calls = _capture_adaptive_route(monkeypatch)
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=1, nr_iter=1, n_directions=4)
    run_dense_initial_model_estep(_Dataset(), state, _projector_config(1))
    assert_matches(calls[0]["joint_particle_ids"], [0, 1, 2, 3])
    assert_matches(calls[0]["joint_halfset_ids"], [0, 1, 0, 1])

    single = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=1, nr_iter=1, n_directions=4, pseudo_halfsets=False)
    run_dense_initial_model_estep(_Dataset(), single, _projector_config(1))
    assert calls[1]["joint_halfset_ids"] is None


def test_estep_meta_aggregates_noise_stats_for_model_updates():
    halfset_results = {
        0: SimpleNamespace(
            class_posterior_sums=np.asarray([1.0, 2.0], dtype=np.float32),
            class_mstep_posterior_sums=np.asarray([1.0, 2.0], dtype=np.float32),
            noise_stats=(
                _fake_noise_stats(0.0, 0.25, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                _fake_noise_stats(0.0, 0.75, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ),
            aggregate_noise_stats=_fake_noise_stats(10.0, 3.0, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
        ),
        1: SimpleNamespace(
            class_posterior_sums=np.asarray([3.0, 4.0], dtype=np.float32),
            class_mstep_posterior_sums=np.asarray([3.0, 4.0], dtype=np.float32),
            noise_stats=(
                _fake_noise_stats(0.0, 1.25, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                _fake_noise_stats(0.0, 1.75, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ),
            aggregate_noise_stats=_fake_noise_stats(20.0, 7.0, [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]),
        ),
    }

    meta = _estep_meta(halfset_results)

    assert meta["wsum_sigma2_offset"] == pytest.approx(30.0)
    assert meta["sigma2_offset_sumw"] == pytest.approx(10.0)
    assert meta["noise_sumw"] == pytest.approx(10.0)
    np.testing.assert_allclose(meta["class_reconstruction_support_sums"], [1.5, 2.5])
    np.testing.assert_allclose(meta["halfset_0_class_reconstruction_support_sums"], [0.25, 0.75])
    np.testing.assert_allclose(meta["halfset_1_class_reconstruction_support_sums"], [1.25, 1.75])
    np.testing.assert_allclose(meta["wsum_sigma2_noise"], [8.0, 10.0, 12.0])
    np.testing.assert_allclose(meta["wsum_img_power"], [14.0, 16.0, 18.0])
    np.testing.assert_allclose(meta["halfset_0_wsum_sigma2_noise"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(meta["halfset_1_wsum_img_power"], [10.0, 11.0, 12.0])


def test_estep_meta_refuses_a_result_without_mstep_class_mass():
    halfset_results = {0: SimpleNamespace(class_posterior_sums=np.asarray([1.0, 2.0]), class_mstep_posterior_sums=None)}

    with pytest.raises(ValueError, match="no M-step class mass"):
        _estep_meta(halfset_results)


def test_estep_meta_uses_significant_mstep_mass_for_relion_probability_updates():
    halfset_results = {
        0: SimpleNamespace(
            class_posterior_sums=np.asarray([1.0, 2.0], dtype=np.float32),
            class_mstep_posterior_sums=np.asarray([0.8, 1.9], dtype=np.float32),
        ),
        1: SimpleNamespace(
            class_posterior_sums=np.asarray([3.0, 4.0], dtype=np.float32),
            class_mstep_posterior_sums=np.asarray([2.7, 3.6], dtype=np.float32),
        ),
    }

    meta = _estep_meta(halfset_results)

    np.testing.assert_allclose(meta["class_posterior_sums"], [3.5, 5.5])
    np.testing.assert_allclose(meta["class_posterior_sums_full"], [4.0, 6.0])
    np.testing.assert_allclose(meta["halfset_0_class_posterior_sums"], [0.8, 1.9])
    np.testing.assert_allclose(meta["halfset_0_class_posterior_sums_full"], [1.0, 2.0])


def test_estep_meta_keeps_each_halfset_profile_summary():
    def halfset(profile_summary):
        return SimpleNamespace(
            class_posterior_sums=np.asarray([1.0], dtype=np.float32),
            class_mstep_posterior_sums=np.asarray([1.0], dtype=np.float32),
            profile_summary=profile_summary,
        )

    meta = _estep_meta({0: halfset({"em_time_s": 1.25, "batches": 1}), 1: halfset(None)})

    assert meta["halfset_0_profile_summary"] == {"em_time_s": 1.25, "batches": 1}
    assert "halfset_1_profile_summary" not in meta


def test_initial_model_estep_with_a_projector_passes_no_dense_means(monkeypatch):
    calls = _capture_adaptive_route(monkeypatch)
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=2, nr_iter=1, n_directions=4)
    run_dense_initial_model_estep(_Dataset(), state, _projector_config(2))
    # The resident route reads only K and the dtype of the dense means: a NaN stand-in.
    assert np.asarray(calls[0]["means"]).shape == (2, 1) and np.all(np.isnan(calls[0]["means"]))
    assert np.all(np.isnan(calls[0]["mean_variance"]))


@pytest.mark.requires_relion_bind
def test_projector_conversion_uses_relion_frame(monkeypatch):
    calls = []

    def fake_recovar_volume_to_relion(ref):
        return np.asarray(ref) + 10.0

    def fake_compute_fourier_transform_map(
        vol, ori_size, padding_factor, interpolator, current_size, do_gridding, data_dim
    ):
        calls.append(
            {
                "vol": np.asarray(vol).copy(),
                "ori_size": ori_size,
                "padding_factor": padding_factor,
                "interpolator": interpolator,
                "current_size": current_size,
                "do_gridding": do_gridding,
                "data_dim": data_dim,
            }
        )
        return np.ones((3, 3, 2), dtype=np.complex128), np.zeros(1), ori_size, padding_factor, 1, 0, interpolator

    monkeypatch.setattr("recovar.utils.helpers.recovar_volume_to_relion", fake_recovar_volume_to_relion)
    monkeypatch.setattr(
        "relax.relion_bind._relion_bind_core.compute_fourier_transform_map",
        fake_compute_fourier_transform_map,
    )

    refs = np.zeros((1, 4, 4, 4), dtype=np.float32)
    from scripts.lib.native_projector_setup import native_reference_to_relion_projector_half_maps

    projector_maps, _ = native_reference_to_relion_projector_half_maps(
        refs, current_size=2, padding_factor=1, projector_data_dtype=np.complex64
    )

    assert np.asarray(projector_maps).shape == (1, 3, 3, 2)
    assert np.asarray(projector_maps).dtype == np.complex64
    assert len(calls) == 1
    call = calls[0]
    np.testing.assert_allclose(call["vol"], np.full((4, 4, 4), 10.0, dtype=np.float64))
    assert {k: v for k, v in call.items() if k != "vol"} == {
        "ori_size": 4,
        "padding_factor": 1,
        "interpolator": 1,
        "current_size": 2,
        "do_gridding": True,
        "data_dim": 2,
    }


def test_relion_projector_projection_dense_scale_matches_embedded_means(monkeypatch):
    import jax.numpy as jnp

    from relax.helpers import projection as projection_helpers

    raw = jnp.asarray([[1.0 + 2.0j, -3.0 + 0.5j]], dtype=jnp.complex64)

    def fake_project(*args, **kwargs):
        return raw

    monkeypatch.setattr(
        projection_helpers,
        "project_relion_projector_half_spectrum_centered_rows",
        fake_project,
    )

    rotations = np.eye(3, dtype=np.float32)[None]
    proj, proj_abs2 = projection_helpers.compute_relion_projector_projections_block(
        np.zeros((3, 3, 2), dtype=np.complex64),
        rotations,
        (4, 4),
        r_max=1,
        centered_rows=True,
        dense_scale=True,
    )

    expected = np.asarray(raw) * -16.0
    # float32 round-trip through jnp / jax has ~1 ULP relative error; the
    # default rtol=1e-7 of assert_allclose is too tight for float32.
    np.testing.assert_allclose(np.asarray(proj), expected, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(np.asarray(proj_abs2), np.abs(expected) ** 2, rtol=1e-5, atol=1e-3)


def _assert_nan_stand_ins(means, mean_variance, n_classes):
    assert means.shape == mean_variance.shape == (n_classes, 1)
    assert means.dtype == np.complex64 and mean_variance.dtype == np.float32
    assert np.all(np.isnan(means)) and np.all(np.isnan(mean_variance))


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_resolve_class_inputs_takes_the_refreshed_projector_and_no_dense_means(monkeypatch, dtype):
    projector_half = np.ones((1, 3, 3, 2), dtype=dtype)
    monkeypatch.setattr(
        "relax.relion.relion_projector_setup.reference_to_relion_projector_half_maps_and_power",
        lambda *args, **kwargs: (projector_half, np.ones((1, 5)), 2),
    )
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=1, nr_iter=1, n_directions=4)
    (half, r_max), _ = prepare_relion_projector_class_inputs_and_power(state, padding_factor=1)
    config = DenseInitialModelEstepConfig(
        noise_variance=np.ones(8 * 8, dtype=np.float32),
        translations=np.zeros((1, 2), dtype=np.float32),
        relion_projector_half_by_class=half,
        relion_projector_r_max=r_max,
    )

    means, mean_variance, exact_half, exact_rmax = _resolve_class_inputs(state, config)

    _assert_nan_stand_ins(means, mean_variance, 1)
    assert_matches(exact_half, projector_half)
    assert exact_half.dtype == dtype
    assert exact_rmax == 2


def test_resolve_class_inputs_reuses_prebuilt_production_projector(monkeypatch):
    projector_half = np.ones((2, 3, 3, 2), dtype=np.complex64)
    monkeypatch.setattr(
        "relax.relion.relion_projector_setup.reference_to_relion_projector_half_maps",
        lambda *args, **kwargs: pytest.fail("prebuilt production projector was rebuilt"),
    )
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=2, nr_iter=1, n_directions=4)
    config = DenseInitialModelEstepConfig(
        noise_variance=np.ones(8 * 8, dtype=np.float32),
        translations=np.zeros((1, 2), dtype=np.float32),
        relion_projector_half_by_class=projector_half,
        relion_projector_r_max=2,
    )

    means, variance, exact_half, exact_rmax = _resolve_class_inputs(state, config)

    _assert_nan_stand_ins(means, variance, 2)
    assert_matches(exact_half, projector_half)
    assert exact_rmax == 2

    with pytest.raises(ValueError, match="needs the iteration's RELION projector"):
        _resolve_class_inputs(state, replace(config, relion_projector_r_max=None))


def test_projector_refresh_can_dump_exact_projector_operand(monkeypatch, tmp_path):
    projector_half = np.arange(54, dtype=np.float32).reshape(1, 3, 3, 6)[..., :2].astype(np.complex64)
    monkeypatch.setattr(
        "relax.relion.relion_projector_setup.reference_to_relion_projector_half_maps_and_power",
        lambda *args, **kwargs: (projector_half, np.ones((1, 5)), 2),
    )
    monkeypatch.setenv("RELAX_INITIAL_MODEL_PROJECTOR_DUMP_DIR", str(tmp_path))
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=1, nr_iter=1, n_directions=4)
    state.iter = 7
    state.current_size = 4

    prepare_relion_projector_class_inputs_and_power(state, padding_factor=1)

    with np.load(tmp_path / "iter007_relion_projector_half.npz") as dumped:
        assert_matches(dumped["projector_half"], projector_half)
        assert int(dumped["projector_r_max"]) == 2
        assert int(dumped["current_size"]) == 4
        assert int(dumped["iteration"]) == 7


def test_sparse_pass2_pass1_current_size_matches_relion_fixture_coarse_size():
    state = initialise_denovo_state(
        ori_size=64,
        pixel_size=8.5,
        K=1,
        nr_iter=1,
        n_directions=576,
        pseudo_halfsets=False,
    )
    assert state.current_size == 28

    pass1_current_size = _resolve_sparse_pass1_current_size(state, state.current_size, 544.0, 1)

    assert pass1_current_size == 10


def test_sparse_pass2_pass1_current_size_uses_pre_update_healpix_order():
    """InitialModel sizes pass 1 before RELION promotes the sampling order."""
    state = initialise_denovo_state(
        ori_size=128,
        pixel_size=4.25,
        K=1,
        nr_iter=25,
        n_directions=192,
        pseudo_halfsets=False,
    )
    state.current_size = 56

    pass1_current_size = _resolve_sparse_pass1_current_size(state, state.current_size, 200.0, 1)

    assert pass1_current_size == 26


def test_sparse_control_split_preserves_input_and_array_identity():
    from relax.vdam.adaptive_estep import _pop_sparse_pass2_options

    metadata = np.array([[0.125, 0.25]], dtype=np.float64)
    supplied = {"coarse_translations": metadata, "image_pre_shifts": metadata}
    cleaned, options = _pop_sparse_pass2_options(supplied)
    assert set(supplied) == {"coarse_translations", "image_pre_shifts"}
    assert set(cleaned) == {"image_pre_shifts"}
    assert set(options) == {"coarse_translations"}
    assert cleaned["image_pre_shifts"] is options["coarse_translations"] is metadata


def test_arrays_to_accumulators_k4_compact_and_full_layouts_match():
    state = InitialModelState(K=4, ori_size=8, current_size=4, Iref=None, Igrad1=None, Igrad2=None)
    r_max = state.current_size // 2
    compact_size = 2 * (r_max + 1) + 1
    full_center = state.ori_size // 2
    coordinates = np.arange(compact_size**3, dtype=np.float32).reshape((compact_size,) * 3)

    compact_data = []
    compact_weight = []
    full_data = []
    full_weight = []
    for class_index in range(state.K):
        data_cube = (coordinates + 1j * (coordinates[::-1] + 10 * class_index)).astype(np.complex64)
        weight_cube = (coordinates + 1 + 100 * class_index).astype(np.float32)
        data_full = np.zeros((state.ori_size,) * 3, dtype=np.complex64)
        weight_full = np.zeros((state.ori_size,) * 3, dtype=np.float32)
        full_cube = (slice(full_center - (r_max + 1), full_center + (r_max + 1) + 1),) * 3
        data_full[full_cube] = data_cube
        weight_full[full_cube] = weight_cube
        compact_data.append(data_cube.reshape(-1))
        compact_weight.append(weight_cube.reshape(-1))
        full_data.append(data_full.reshape(-1))
        full_weight.append(weight_full.reshape(-1))

    common = dict(state=state, halfset_idx=1, padding_factor=1)
    compact = _arrays_to_accumulators(compact_data, compact_weight, **common)
    full = _arrays_to_accumulators(full_data, full_weight, **common)

    assert [(value.halfset_idx, value.class_idx) for value in compact] == [
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
    ]
    for compact_accumulator, full_accumulator in zip(compact, full):
        assert_matches(compact_accumulator.data, full_accumulator.data)
        assert_matches(compact_accumulator.weight, full_accumulator.weight)


@pytest.mark.parametrize(
    ("data_class_count", "weight_class_count"),
    [(3, 4), (5, 4), (4, 3), (4, 5)],
)
def test_arrays_to_accumulators_rejects_missing_or_duplicated_k4_class_rows(
    data_class_count,
    weight_class_count,
):
    state = InitialModelState(K=4, ori_size=8, current_size=4, Iref=None, Igrad1=None, Igrad2=None)
    compact_voxels = 7**3
    data = [np.zeros(compact_voxels, dtype=np.complex64) for _ in range(data_class_count)]
    weight = [np.zeros(compact_voxels, dtype=np.float32) for _ in range(weight_class_count)]

    with pytest.raises(
        ValueError,
        match=(
            "K-class accumulator class axes must each contain exactly 4 classes, "
            f"got data={data_class_count} and weight={weight_class_count}"
        ),
    ):
        _arrays_to_accumulators(
            data,
            weight,
            state,
            halfset_idx=0,
                    padding_factor=1,
        )


def test_arrays_to_accumulators_accepts_compact_k4_backprojector_cubes():
    """Pin the real-data K=4 current-size bridge that failed on 59-cubed outputs."""

    state = InitialModelState(K=4, ori_size=256, current_size=56, Iref=None, Igrad1=None, Igrad2=None)
    compact_size = 59
    compact_voxels = compact_size**3
    data = np.stack(
        [np.full(compact_voxels, class_index + 1j, dtype=np.complex64) for class_index in range(4)],
    )
    weight = np.stack(
        [np.full(compact_voxels, class_index + 1, dtype=np.float32) for class_index in range(4)],
    )

    accumulators = _arrays_to_accumulators(
        data,
        weight,
        state,
        halfset_idx=0,
        padding_factor=1,
    )

    assert len(accumulators) == 4
    assert [(accum.halfset_idx, accum.class_idx) for accum in accumulators] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
    ]
    data_scale, weight_scale = relion_bpref_frame_scales(state.ori_size)
    for class_index, accumulator in enumerate(accumulators):
        assert accumulator.data.shape == (59, 59, 30)
        assert accumulator.weight.shape == (59, 59, 30)
        assert_matches(accumulator.data, np.complex128(class_index + 1j) * data_scale)
        assert_matches(accumulator.weight, np.float64(class_index + 1) * weight_scale)
