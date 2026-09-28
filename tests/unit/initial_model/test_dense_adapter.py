"""InitialModel dense K-class E-step adapter tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from relax.local.local_layout import LocalHypothesisLayout
from relax.relion.relion_projector_setup import reference_to_relion_projector_half_maps
from relax.vdam.adaptive_estep import _resolve_sparse_pass1_current_size, _safe_coarse_significance_image_batch_size
from relax.vdam.dense_adapter import (
    _relion_projector_to_dense_volume,
    _resolve_class_inputs,
    class_log_priors_from_state,
    reference_to_dense_means,
    relion_projector_half_maps_to_dense_means,
    run_dense_initial_model_estep,
)
from relax.vdam.estep_common import DenseInitialModelEstepConfig, _arrays_to_accumulators, _estep_meta
from relax.vdam.init import initialise_denovo_state
from helpers.float_compare import assert_matches

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


def _control_coarse_selector_audit(translation_count: int) -> dict:
    return {
        "score_mode": "gaussian",
        "translation_count": int(translation_count),
        "requested_fused": False,
        "effective_fused": False,
        "requested_workers": 0,
        "effective_workers": 0,
        "requested_atomic": False,
        "effective_atomic": False,
        "wrapper": None,
        "target": None,
        "counts": {
            "fused_calls": 0,
            "actual_rows": 0,
            "multistream_calls": 0,
            "native_atomic_selected_calls": 0,
        },
    }


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


def _fake_result(n_classes: int, n: int, *, n_images: int = 2, n_groups: int = 2):
    Ft_y = [np.full(n**3, k + 1, dtype=np.complex64) for k in range(n_classes)]
    Ft_ctf = [np.full(n**3, (k + 1) * 2, dtype=np.float32) for k in range(n_classes)]
    per_class_stats = tuple(
        SimpleNamespace(rotation_posterior_sums=np.full(3, k + 1, dtype=np.float32)) for k in range(n_classes)
    )
    return _ReplaceableNamespace(
        Ft_y=Ft_y,
        Ft_ctf=Ft_ctf,
        grouped_Ft_y=np.broadcast_to(np.asarray(Ft_y)[None, :, :], (n_groups, n_classes, n**3)).copy(),
        grouped_Ft_ctf=np.broadcast_to(np.asarray(Ft_ctf)[None, :, :], (n_groups, n_classes, n**3)).copy(),
        class_responsibilities=np.full((n_classes, n_images), 1.0 / n_classes, dtype=np.float32),
        class_posterior_sums=np.arange(n_classes, dtype=np.float32),
        class_assignments=np.zeros(n_images, dtype=np.int32),
        pose_assignments=np.arange(n_images, dtype=np.int32),
        best_pose_rotations=np.broadcast_to(np.eye(3, dtype=np.float32), (n_images, 3, 3)).copy(),
        best_pose_translations=np.arange(n_images * 2, dtype=np.float32).reshape(n_images, 2),
        best_pose_rotation_ids=np.arange(n_images, dtype=np.int32),
        stats=SimpleNamespace(max_posterior_per_image=np.linspace(0.25, 0.75, n_images, dtype=np.float32)),
        per_class_stats=per_class_stats,
        profile_summary=None,
    )


def _fake_noise_stats(offset: float, sumw: float, wsum_noise, img_power):
    return SimpleNamespace(
        wsum_sigma2_offset=float(offset),
        sumw=float(sumw),
        wsum_sigma2_noise=np.asarray(wsum_noise, dtype=np.float32),
        wsum_img_power=np.asarray(img_power, dtype=np.float32),
    )


def _fake_result_with_profile(n_classes: int, n: int, *, n_images: int = 2, n_groups: int = 2):
    result = _fake_result(n_classes, n, n_images=n_images, n_groups=n_groups)
    result.profile_summary = {"em_time_s": 1.25, "batches": 1}
    return result


def test_arrays_to_accumulators_inverts_relion_x_public_layout_without_projector_flip():
    from relax.helpers.half_volume_mstep import relion_x_half_volume_to_full
    from relax.helpers.half_volume_mstep import enforce_relion_half_volume_x0_hermitian_host
    from relax.vdam.layout import relion_bpref_frame_scales

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
        relion_bpref_frame=True,
        relion_projector_frame=True,
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
        relion_bpref_frame=True,
        relion_projector_frame=False,
        padding_factor=1,
    )

    assert [value.halfset_idx for value in actual] == [0, 1]
    assert [value.class_idx for value in actual] == [0, 0]
    assert not np.array_equal(actual[0].data, actual[1].data)
    assert not np.array_equal(actual[0].weight, actual[1].weight)


def test_class_log_priors_from_state_normalizes_weights():
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=2, nr_iter=1, n_directions=4)
    state.pdf_class = np.asarray([2.0, 1.0])
    np.testing.assert_allclose(class_log_priors_from_state(state), np.log([2.0 / 3.0, 1.0 / 3.0]))


def test_class_log_priors_from_state_allows_inactive_class():
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=2, nr_iter=1, n_directions=4)
    state.pdf_class = np.asarray([1.0, 0.0])
    out = class_log_priors_from_state(state)
    assert out[0] == 0.0
    assert out[1] < -1.0e20


def test_dense_initial_model_estep_runs_separate_k_class_calls_for_pseudo_halfsets(monkeypatch):
    from relax.vdam import dense_adapter

    calls = []
    conversions = []
    original_conversion = dense_adapter._dense_rotations_for_config

    def convert(rotations, config):
        conversions.append(rotations)
        return original_conversion(rotations, config)

    monkeypatch.setattr(dense_adapter, "_dense_rotations_for_config", convert)

    def fake_run_dense_k_class_em(
        dataset, means, mean_variance, noise_variance, rotations, translations, disc_type, **kwargs
    ):
        calls.append(
            {
                "means_shape": np.asarray(means).shape,
                "image_indices": np.asarray(kwargs["image_indices"]).copy(),
                "has_reconstruction_group_ids": "reconstruction_group_ids" in kwargs,
                "has_reconstruction_group_count": "reconstruction_group_count" in kwargs,
                "class_log_priors": np.asarray(kwargs["class_log_priors"]).copy(),
                "current_size": kwargs["current_size"],
            }
        )
        return _fake_result(n_classes=2, n=8, n_images=int(np.asarray(kwargs["image_indices"]).size), n_groups=1)

    monkeypatch.setattr(
        "relax.vdam.dense_adapter.run_dense_k_class_em",
        fake_run_dense_k_class_em,
    )
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=2, nr_iter=1, n_directions=4)
    state.current_size = 8
    state.pdf_class = np.asarray([0.75, 0.25])
    config = DenseInitialModelEstepConfig(
        means=np.zeros((2, 8**3), dtype=np.complex64),
        mean_variance=np.ones((2, 8**3), dtype=np.float32),
        noise_variance=np.ones(8 * 8, dtype=np.float32),
        rotations=np.eye(3, dtype=np.float32)[None],
        translations=np.zeros((1, 2), dtype=np.float32),
        relion_bpref_frame=False,
    )

    result = run_dense_initial_model_estep(
        _Dataset(),
        state,
        config,
        particle_ids=np.asarray([0, 1, 2, 3]),
        halfset_ids=np.asarray([0, 1, 0, 1], dtype=np.int8),
    )

    assert len(conversions) == 1
    assert conversions[0] is config.rotations
    assert_matches(config.rotations, np.eye(3, dtype=np.float32)[None])
    assert len(calls) == 2
    assert calls[0]["means_shape"] == (2, 8**3)
    assert calls[1]["means_shape"] == (2, 8**3)
    assert_matches(calls[0]["image_indices"], [0, 2])
    assert_matches(calls[1]["image_indices"], [1, 3])
    assert calls[0]["has_reconstruction_group_ids"] is False
    assert calls[1]["has_reconstruction_group_ids"] is False
    assert calls[0]["has_reconstruction_group_count"] is False
    assert calls[1]["has_reconstruction_group_count"] is False
    np.testing.assert_allclose(calls[0]["class_log_priors"], np.log([0.75, 0.25]))
    np.testing.assert_allclose(calls[1]["class_log_priors"], np.log([0.75, 0.25]))
    assert calls[0]["current_size"] == 8
    assert calls[1]["current_size"] == 8

    assert len(result.accumulators) == 4
    assert [(a.halfset_idx, a.class_idx) for a in result.accumulators] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    for accum in result.accumulators:
        assert accum.data.shape == (8, 8, 5)
        assert accum.weight.shape == (8, 8, 5)
    np.testing.assert_allclose(result.accumulators[0].data, 1.0)
    np.testing.assert_allclose(result.accumulators[1].weight, 4.0)
    assert result.meta["halfset_ids"] == (0, 1)
    assert "fused_pseudo_halfsets" not in result.meta
    np.testing.assert_allclose(result.meta["class_posterior_sums"], [0.0, 2.0])
    np.testing.assert_allclose(
        result.meta["class_direction_posterior_sums"],
        np.asarray([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]),
    )
    assert_matches(result.meta["selected_particle_ids"], [0, 2, 1, 3])
    assert_matches(result.meta["pose_assignments"], [0, 1, 0, 1])
    assert_matches(result.meta["class_assignments"], [0, 0, 0, 0])
    assert_matches(result.meta["best_pose_rotation_ids"], [0, 1, 0, 1])
    np.testing.assert_allclose(
        result.meta["best_pose_translations"],
        np.asarray([[0, 1], [2, 3], [0, 1], [2, 3]], dtype=np.float32),
    )
    assert result.meta["best_pose_rotations"].shape == (4, 3, 3)
    np.testing.assert_allclose(
        result.meta["max_posterior_per_image"],
        np.asarray([0.25, 0.75, 0.25, 0.75], dtype=np.float32),
    )


def test_estep_meta_aggregates_noise_stats_for_model_updates():
    halfset_results = {
        0: SimpleNamespace(
            class_posterior_sums=np.asarray([1.0, 2.0], dtype=np.float32),
            noise_stats=(
                _fake_noise_stats(0.0, 0.25, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                _fake_noise_stats(0.0, 0.75, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ),
            aggregate_noise_stats=_fake_noise_stats(10.0, 3.0, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
        ),
        1: SimpleNamespace(
            class_posterior_sums=np.asarray([3.0, 4.0], dtype=np.float32),
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


def test_dense_initial_model_estep_slices_full_translation_prior_for_pseudo_halfsets(monkeypatch):
    calls = []

    def fake_run_dense_k_class_em(*args, **kwargs):
        calls.append(np.asarray(kwargs["translation_log_prior"]).copy())
        return _fake_result(n_classes=1, n=8, n_images=int(np.asarray(kwargs["image_indices"]).size), n_groups=1)

    monkeypatch.setattr(
        "relax.vdam.dense_adapter.run_dense_k_class_em",
        fake_run_dense_k_class_em,
    )
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=1, nr_iter=1, n_directions=4)
    full_prior = np.arange(8, dtype=np.float32).reshape(4, 2)
    config = DenseInitialModelEstepConfig(
        means=np.zeros((1, 8**3), dtype=np.complex64),
        mean_variance=np.ones((1, 8**3), dtype=np.float32),
        noise_variance=np.ones(8 * 8, dtype=np.float32),
        rotations=np.eye(3, dtype=np.float32)[None],
        translations=np.zeros((2, 2), dtype=np.float32),
        relion_bpref_frame=False,
        engine_kwargs={"translation_log_prior": full_prior},
    )

    run_dense_initial_model_estep(
        _Dataset(),
        state,
        config,
        particle_ids=np.asarray([0, 1, 2, 3]),
        halfset_ids=np.asarray([0, 1, 0, 1], dtype=np.int8),
    )

    assert len(calls) == 2
    assert_matches(calls[0], full_prior[[0, 2]])
    assert_matches(calls[1], full_prior[[1, 3]])


def test_dense_initial_model_estep_meta_includes_optional_profiles(monkeypatch):
    def fake_run_dense_k_class_em(*args, **kwargs):
        # ``return_profile`` is filtered out by _dense_run_em_kwargs before
        # reaching run_dense_k_class_em — it's a sparse/local-engine-only
        # kwarg that run_dense_k_class_em explicitly _reject_kwargs's.
        # dense_adapter still extracts profile info from the return-value
        # meta regardless of whether the kwarg flowed through.
        assert "return_profile" not in kwargs
        return _fake_result_with_profile(
            n_classes=1,
            n=8,
            n_images=int(np.asarray(kwargs["image_indices"]).size),
        )

    monkeypatch.setattr(
        "relax.vdam.dense_adapter.run_dense_k_class_em",
        fake_run_dense_k_class_em,
    )
    state = initialise_denovo_state(
        ori_size=8,
        pixel_size=1.0,
        K=1,
        nr_iter=1,
        n_directions=4,
        pseudo_halfsets=False,
    )
    config = DenseInitialModelEstepConfig(
        means=np.zeros((1, 8**3), dtype=np.complex64),
        mean_variance=np.ones((1, 8**3), dtype=np.float32),
        noise_variance=np.ones(8 * 8, dtype=np.float32),
        rotations=np.eye(3, dtype=np.float32)[None],
        translations=np.zeros((1, 2), dtype=np.float32),
        relion_bpref_frame=False,
        engine_kwargs={"return_profile": True},
    )

    result = run_dense_initial_model_estep(_Dataset(), state, config)

    assert result.meta["halfset_0_profile_summary"] == {"em_time_s": 1.25, "batches": 1}
    np.testing.assert_allclose(result.meta["class_posterior_sums"], [0.0])
    np.testing.assert_allclose(result.meta["class_direction_posterior_sums"], [[1.0, 1.0, 1.0]])
    assert_matches(result.meta["selected_particle_ids"], [0, 1, 2, 3])
    assert_matches(result.meta["class_assignments"], [0, 0, 0, 0])
    np.testing.assert_allclose(
        result.meta["max_posterior_per_image"],
        np.linspace(0.25, 0.75, 4, dtype=np.float32),
    )


def test_dense_initial_model_estep_pseudo_halfset_meta_includes_per_halfset_profiles(monkeypatch):
    def fake_run_dense_k_class_em(*args, **kwargs):
        # See note in test_dense_initial_model_estep_meta_includes_optional_profiles
        # — return_profile is filtered out before the dense entry-point.
        assert "return_profile" not in kwargs
        return _fake_result_with_profile(
            n_classes=1,
            n=8,
            n_images=int(np.asarray(kwargs["image_indices"]).size),
            n_groups=1,
        )

    monkeypatch.setattr(
        "relax.vdam.dense_adapter.run_dense_k_class_em",
        fake_run_dense_k_class_em,
    )
    state = initialise_denovo_state(
        ori_size=8,
        pixel_size=1.0,
        K=1,
        nr_iter=1,
        n_directions=4,
        pseudo_halfsets=True,
    )
    config = DenseInitialModelEstepConfig(
        means=np.zeros((1, 8**3), dtype=np.complex64),
        mean_variance=np.ones((1, 8**3), dtype=np.float32),
        noise_variance=np.ones(8 * 8, dtype=np.float32),
        rotations=np.eye(3, dtype=np.float32)[None],
        translations=np.zeros((1, 2), dtype=np.float32),
        relion_bpref_frame=False,
        engine_kwargs={"return_profile": True},
    )

    result = run_dense_initial_model_estep(
        _Dataset(),
        state,
        config,
        particle_ids=np.asarray([0, 1, 2, 3]),
        halfset_ids=np.asarray([0, 1, 0, 1], dtype=np.int8),
    )

    assert result.meta["halfset_0_profile_summary"] == {"em_time_s": 1.25, "batches": 1}
    assert result.meta["halfset_1_profile_summary"] == {"em_time_s": 1.25, "batches": 1}
    assert "fused_profile_summary" not in result.meta


def test_dense_initial_model_estep_uses_current_state_reference_when_means_omitted(monkeypatch):
    calls = []

    def fake_reference_to_dense_means(references):
        refs = np.asarray(references)
        return np.full((refs.shape[0], refs.shape[1] ** 3), refs[0, 0, 0, 0], dtype=np.complex64)

    def fake_run_dense_k_class_em(
        dataset, means, mean_variance, noise_variance, rotations, translations, disc_type, **kwargs
    ):
        calls.append(
            {
                "means": np.asarray(means).copy(),
                "mean_variance": np.asarray(mean_variance).copy(),
            }
        )
        return _fake_result(n_classes=1, n=8)

    monkeypatch.setattr(
        "relax.vdam.dense_adapter.reference_to_dense_means",
        fake_reference_to_dense_means,
    )
    monkeypatch.setattr(
        "relax.vdam.dense_adapter.run_dense_k_class_em",
        fake_run_dense_k_class_em,
    )
    state = initialise_denovo_state(
        ori_size=8,
        pixel_size=1.0,
        K=1,
        nr_iter=1,
        n_directions=4,
        pseudo_halfsets=False,
    )
    state.Iref[0, 0, 0, 0] = 7.0
    config = DenseInitialModelEstepConfig(
        noise_variance=np.ones(8 * 8, dtype=np.float32),
        rotations=np.eye(3, dtype=np.float32)[None],
        translations=np.zeros((1, 2), dtype=np.float32),
        relion_bpref_frame=False,
    )

    run_dense_initial_model_estep(_Dataset(), state, config)

    assert len(calls) == 1
    np.testing.assert_allclose(calls[0]["means"], 7.0)
    np.testing.assert_allclose(calls[0]["mean_variance"], 49.0)


def test_reference_to_dense_means_uses_scoring_fourier_scale(monkeypatch):
    def fake_dft3(values):
        return np.asarray(values) + 2.0j

    def fake_gridding_correct(values, *args, **kwargs):
        return np.asarray(values) + 1.0, None

    monkeypatch.setattr("recovar.core.fourier_transform_utils.get_dft3", fake_dft3)
    monkeypatch.setattr("recovar.reconstruction.relion_functions.griddingCorrect", fake_gridding_correct)

    refs = np.zeros((1, 4, 4, 4), dtype=np.float32)

    means = reference_to_dense_means(refs)

    assert means.shape == (1, 4**3)
    np.testing.assert_allclose(means, 1.0 + 2.0j)


def test_relion_projector_to_dense_volume_embeds_cropped_slab(monkeypatch):
    captured = {}

    def fake_half_to_full(half, shape):
        captured["half"] = np.asarray(half)
        captured["shape"] = shape
        return np.asarray(half) + 1.0j

    monkeypatch.setattr("recovar.core.fourier_transform_utils.half_volume_to_full_volume", fake_half_to_full)

    slab = np.arange(3 * 3 * 2, dtype=np.float64).reshape(3, 3, 2).astype(np.complex128)
    out = _relion_projector_to_dense_volume(slab, 4)

    assert captured["shape"] == (4, 4, 4)
    half = captured["half"]
    assert half.shape == (4, 4, 3)
    assert_matches(half[1:4, 1:4, :2], slab[::-1, :, :])
    np.testing.assert_allclose(out, half + 1.0j)


def test_relion_projector_to_dense_volume_handles_ori_size_boundary(monkeypatch):
    """When current_size == ori_size, RELION's cropped projector has y/z dim
    2*r_max+1 = ori_size+1. The embedding loop must drop the redundant
    Nyquist row (Hermitian conjugate of index 0) without raising."""
    captured = {}

    def fake_half_to_full(half, shape):
        captured["half"] = np.asarray(half)
        return np.asarray(half)

    monkeypatch.setattr("recovar.core.fourier_transform_utils.half_volume_to_full_volume", fake_half_to_full)

    # ori_size=4 → r_max=2 → cropped shape (5, 5, 3)
    slab = np.arange(5 * 5 * 3, dtype=np.float64).reshape(5, 5, 3).astype(np.complex128)
    _relion_projector_to_dense_volume(slab, 4)

    half = captured["half"]
    assert half.shape == (4, 4, 3)
    # Index iz=4 (extra Nyquist) must be dropped, not raise.
    # The first 4 rows (iz=0..3) of the reversed slab map to half[0..3, :, :].
    assert_matches(half[0:4, 0:4, :3], slab[::-1, :, :][0:4, 0:4, :3])


def test_relion_projector_to_dense_volume_truncates_oversize(monkeypatch):
    """Slabs larger than the representable half-volume are truncated to the
    in-range subset rather than rejected. This handles VDAM iters where
    autosampling pushes current_size up to (or slightly above) ori_size and
    RELION emits cropped projectors of shape (2*r_max+1, 2*r_max+1, r_max+1)
    that exceed RECOVAR's (ori_size, ori_size, ori_size/2+1) layout."""
    captured = {}

    def fake_half_to_full(half, shape):
        captured["half"] = np.asarray(half)
        return np.asarray(half)

    monkeypatch.setattr("recovar.core.fourier_transform_utils.half_volume_to_full_volume", fake_half_to_full)
    # ori_size=4 → max half (4, 4, 3). Pass an even larger (7, 7, 4) slab.
    slab = np.arange(7 * 7 * 4, dtype=np.float64).reshape(7, 7, 4).astype(np.complex128)
    _relion_projector_to_dense_volume(slab, 4)
    half = captured["half"]
    assert half.shape == (4, 4, 3)
    # Center of slab (index 3) maps to center of half (index 2).
    # iz=3 → z = 3-3+2 = 2 ✓, iz=2 → z = 1, iz=4 → z = 3, iz=0/1/5/6 → out of range.
    # Reversed slab[::-1] at iz=2 = original slab[4]; at iz=3 = slab[3]; etc.
    rev = slab[::-1, :, :]
    assert_matches(half[1, 1, :3], rev[2, 2, :3])
    assert_matches(half[2, 2, :3], rev[3, 3, :3])


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

    def fake_embed(projector_data, ori_size):
        assert projector_data.shape == (3, 3, 2)
        return np.full((ori_size, ori_size, ori_size), 2.0 + 3.0j, dtype=np.complex128)

    monkeypatch.setattr("recovar.utils.helpers.recovar_volume_to_relion", fake_recovar_volume_to_relion)
    monkeypatch.setattr(
        "relax.relion_bind._relion_bind_core.compute_fourier_transform_map",
        fake_compute_fourier_transform_map,
    )
    monkeypatch.setattr("relax.vdam.dense_adapter._relion_projector_to_dense_volume", fake_embed)

    refs = np.zeros((1, 4, 4, 4), dtype=np.float32)
    projector_maps, _ = reference_to_relion_projector_half_maps(
        refs, current_size=2, padding_factor=1, projector_setup_backend="native"
    )
    means = relion_projector_half_maps_to_dense_means(projector_maps, refs.shape[-1])

    assert means.shape == (1, 4**3)
    assert means.dtype == np.complex64
    np.testing.assert_allclose(means, -16.0 * (2.0 + 3.0j))
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


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("override_variance", [False, True])
def test_resolve_class_inputs_relion_projector_uses_exact_path_by_default(monkeypatch, dtype, override_variance):
    projector_half = np.ones((1, 3, 3, 2), dtype=dtype)
    dense_means = np.full((1, 8**3), 2.0 + 0.5j, dtype=dtype)

    monkeypatch.setattr(
        "relax.relion.relion_projector_setup.reference_to_relion_projector_half_maps",
        lambda *args, **kwargs: (projector_half, 2),
    )
    monkeypatch.setattr(
        "relax.vdam.dense_adapter.relion_projector_half_maps_to_dense_means",
        lambda *args, **kwargs: dense_means,
    )
    variance_override = np.full(dense_means.shape, 7.0) if override_variance else None
    expected_variance = np.abs(dense_means) ** 2 if variance_override is None else variance_override
    original_abs = np.abs
    computed_variances = []

    def record_abs(value):
        result = original_abs(value)
        computed_variances.append(result)
        return result

    monkeypatch.setattr(np, "abs", record_abs)
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=1, nr_iter=1, n_directions=4)
    config = DenseInitialModelEstepConfig(
        noise_variance=np.ones(8 * 8, dtype=np.float32),
        rotations=np.eye(3, dtype=np.float32)[None],
        translations=np.zeros((1, 2), dtype=np.float32),
        relion_projector_frame=True,
        mean_variance=variance_override,
    )

    means, mean_variance, exact_half, exact_rmax = _resolve_class_inputs(state, config)

    assert len(computed_variances) == 1
    assert_matches(mean_variance, expected_variance)
    assert mean_variance.dtype == expected_variance.dtype
    if override_variance:
        assert mean_variance is variance_override
    assert_matches(means, dense_means)
    assert_matches(exact_half, projector_half)
    assert exact_rmax == 2

    monkeypatch.setenv("RELAX_INITIAL_MODEL_EXACT_RELION_PROJECTOR", "0")
    # assert_matches also calls np.abs, so count only this call's variances.
    calls_before = len(computed_variances)
    _means, _mean_variance, exact_half, exact_rmax = _resolve_class_inputs(state, config)

    assert len(computed_variances) == calls_before + 1
    assert_matches(_mean_variance, expected_variance)
    assert exact_half is None
    assert exact_rmax is None


def test_resolve_class_inputs_reuses_prebuilt_production_projector(monkeypatch):
    projector_half = np.ones((1, 3, 3, 2), dtype=np.complex64)
    dense_means = np.full((1, 8**3), 2.0 + 0.5j, dtype=np.complex64)
    mean_variance = np.abs(dense_means) ** 2
    monkeypatch.setattr(
        "relax.relion.relion_projector_setup.reference_to_relion_projector_half_maps",
        lambda *args, **kwargs: pytest.fail("prebuilt production projector was rebuilt"),
    )
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=1, nr_iter=1, n_directions=4)
    config = DenseInitialModelEstepConfig(
        means=dense_means,
        mean_variance=mean_variance,
        noise_variance=np.ones(8 * 8, dtype=np.float32),
        rotations=np.eye(3, dtype=np.float32)[None],
        translations=np.zeros((1, 2), dtype=np.float32),
        relion_projector_frame=True,
        relion_projector_half_by_class=projector_half,
        relion_projector_r_max=2,
    )

    means, variance, exact_half, exact_rmax = _resolve_class_inputs(state, config)

    assert means is dense_means
    assert variance is mean_variance
    assert_matches(exact_half, projector_half)
    assert exact_rmax == 2


def test_resolve_class_inputs_can_dump_exact_projector_operand(monkeypatch, tmp_path):
    projector_half = np.arange(54, dtype=np.float32).reshape(1, 3, 3, 6)[..., :2].astype(np.complex64)
    dense_means = np.zeros((1, 8**3), dtype=np.complex64)
    monkeypatch.setattr(
        "relax.relion.relion_projector_setup.reference_to_relion_projector_half_maps",
        lambda *args, **kwargs: (projector_half, 2),
    )
    monkeypatch.setattr(
        "relax.vdam.dense_adapter.relion_projector_half_maps_to_dense_means",
        lambda *args, **kwargs: dense_means,
    )
    monkeypatch.setenv("RELAX_INITIAL_MODEL_PROJECTOR_DUMP_DIR", str(tmp_path))
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=1, nr_iter=1, n_directions=4)
    state.iter = 7
    state.current_size = 4
    config = DenseInitialModelEstepConfig(
        noise_variance=np.ones(8 * 8, dtype=np.float32),
        rotations=np.eye(3, dtype=np.float32)[None],
        translations=np.zeros((1, 2), dtype=np.float32),
        relion_projector_frame=True,
    )

    _resolve_class_inputs(state, config)

    with np.load(tmp_path / "iter007_relion_projector_half.npz") as dumped:
        assert_matches(dumped["projector_half"], projector_half)
        assert int(dumped["projector_r_max"]) == 2
        assert int(dumped["current_size"]) == 4
        assert int(dumped["iteration"]) == 7


def test_dense_initial_model_estep_handles_empty_halfset(monkeypatch):
    calls = []

    def fake_run_dense_k_class_em(*args, **kwargs):
        calls.append(kwargs["image_indices"])
        result = _fake_result(n_classes=1, n=8, n_images=int(np.asarray(kwargs["image_indices"]).size), n_groups=2)
        result.grouped_Ft_y[1] = 0.0
        result.grouped_Ft_ctf[1] = 0.0
        return result

    monkeypatch.setattr(
        "relax.vdam.dense_adapter.run_dense_k_class_em",
        fake_run_dense_k_class_em,
    )
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=1, nr_iter=1, n_directions=4)
    config = DenseInitialModelEstepConfig(
        means=np.zeros((1, 8**3), dtype=np.complex64),
        mean_variance=np.ones((1, 8**3), dtype=np.float32),
        noise_variance=np.ones(8 * 8, dtype=np.float32),
        rotations=np.eye(3, dtype=np.float32)[None],
        translations=np.zeros((1, 2), dtype=np.float32),
        relion_bpref_frame=False,
    )

    result = run_dense_initial_model_estep(
        _Dataset(),
        state,
        config,
        particle_ids=np.asarray([0, 2]),
        halfset_ids=np.asarray([0, 0], dtype=np.int8),
    )

    assert len(calls) == 1
    assert_matches(calls[0], [0, 2])
    assert len(result.accumulators) == 2
    np.testing.assert_allclose(result.accumulators[1].data, 0.0)
    np.testing.assert_allclose(result.accumulators[1].weight, 0.0)


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

    pass1_current_size = _resolve_sparse_pass1_current_size(
        state,
        {"current_size": state.current_size},
        {"healpix_order": 1, "particle_diameter_ang": 544.0},
    )

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

    pass1_current_size = _resolve_sparse_pass1_current_size(
        state,
        {"current_size": state.current_size},
        {
            "healpix_order": 2,
            "pass1_healpix_order": 1,
            "particle_diameter_ang": 200.0,
        },
    )

    assert pass1_current_size == 26


def test_sparse_control_split_preserves_input_and_array_identity():
    from relax.vdam.adaptive_estep import _pop_sparse_pass2_options

    metadata = np.array([[0.125, 0.25]], dtype=np.float64)
    supplied = {"sparse_pass2": True, "coarse_translations": metadata, "image_pre_shifts": metadata}
    cleaned, options = _pop_sparse_pass2_options(supplied)
    assert set(supplied) == {"sparse_pass2", "coarse_translations", "image_pre_shifts"}
    assert set(cleaned) == {"image_pre_shifts"}
    assert set(options) == {"coarse_translations"}
    assert cleaned["image_pre_shifts"] is options["coarse_translations"] is metadata


def test_arrays_to_accumulators_k4_compact_and_full_layouts_match():
    state = SimpleNamespace(K=4, ori_size=8, current_size=4)
    r_max = state.current_size // 2
    compact_size = 2 * (r_max + 1) + 1
    compact_center = compact_size // 2
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
        full_slab = (
            slice(full_center - (r_max + 1), full_center + (r_max + 1) + 1),
            slice(full_center - (r_max + 1), full_center + (r_max + 1) + 1),
            slice(full_center, full_center + (r_max + 1) + 1),
        )
        data_full[full_slab] = data_cube[:, :, compact_center:]
        weight_full[full_slab] = weight_cube[:, :, compact_center:]
        compact_data.append(data_cube.reshape(-1))
        compact_weight.append(weight_cube.reshape(-1))
        full_data.append(data_full.reshape(-1))
        full_weight.append(weight_full.reshape(-1))

    common = dict(
        state=state,
        halfset_idx=1,
        relion_bpref_frame=False,
        relion_projector_frame=False,
        padding_factor=1,
    )
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
    state = SimpleNamespace(K=4, ori_size=8, current_size=4)
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
            relion_bpref_frame=False,
            relion_projector_frame=False,
            padding_factor=1,
        )


def test_arrays_to_accumulators_accepts_compact_k4_backprojector_cubes():
    """Pin the real-data K=4 current-size bridge that failed on 59-cubed outputs."""

    state = SimpleNamespace(K=4, ori_size=256, current_size=56)
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
        relion_bpref_frame=False,
        relion_projector_frame=False,
        padding_factor=1,
    )

    assert len(accumulators) == 4
    assert [(accum.halfset_idx, accum.class_idx) for accum in accumulators] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
    ]
    for class_index, accumulator in enumerate(accumulators):
        assert accumulator.data.shape == (59, 59, 30)
        assert accumulator.weight.shape == (59, 59, 30)
        assert_matches(accumulator.data, np.complex128(class_index + 1j))
        assert_matches(accumulator.weight, np.float64(class_index + 1))
