"""Half scoring over shape classes (optics groups on other pixel sizes and boxes, CPU).

Checks the RELION rules the per-class calls follow (noise shell remaps both ways,
translations in class pixels, remapped Fourier sizes, scaled projection) and that
per-image outputs return to the half's order by index while sums add.
"""

import dataclasses
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
from helpers.float_compare import assert_matches
from helpers.refinement_specs import local_half_owners

from relax.dense.score_outputs import ClassScoreSummary, HalfScoreResult, PerHalfOutputs
from relax.helpers.types import RelionStats, make_noise_stats
from relax.refinement import half_scoring, optics_shapes
from relax.refinement.half_inputs import HalfSet

REF_BOX, REF_PIX = 32, 4.0


def _half():
    ds_a = SimpleNamespace(image_shape=(32, 32), volume_shape=(32,) * 3, voxel_size=4.0)
    ds_b = SimpleNamespace(image_shape=(28, 28), volume_shape=(28,) * 3, voxel_size=4.0 * 32 / 24)
    classes = optics_shapes.make_shape_classes(
        [(ds_a, np.array([0, 3, 4])), (ds_b, np.array([1, 2]))], ref_box=REF_BOX, ref_pixel=REF_PIX
    )
    return optics_shapes.MultiShapeHalf(classes, image_shape=(32, 32), volume_shape=(32, 32, 32), voxel_size=4.0)


@pytest.mark.unit
def test_class_rotation_prior_is_sliced_only_when_per_image():
    half = _half()
    small = half.classes[1]
    per_class = np.arange(2 * 7.0).reshape(2, 7)
    per_image = np.arange(2 * 5 * 7.0).reshape(2, 5, 7)
    out = optics_shapes.class_kwargs({"class_rotation_log_prior_k": per_class}, small, 5)
    assert out["class_rotation_log_prior_k"] is per_class
    out = optics_shapes.class_kwargs({"class_rotation_log_prior_k": per_image}, small, 5)
    np.testing.assert_array_equal(out["class_rotation_log_prior_k"], per_image[:, [1, 2]])


@pytest.mark.unit
def test_class_noise_follows_relion_estep_remap():
    half = _half()
    shape_class = half.classes[1]
    s = shape_class.scale
    radial = np.stack([np.linspace(2.0, 1.0, 17), np.linspace(5.0, 3.0, 17)]) * REF_BOX**4
    table = optics_shapes.class_noise_table(radial, shape_class, REF_BOX)
    assert table.shape == (2, 28 * 28)
    # Pixel (ky, kx) = (0, i) sits on group shell i; RELION reads reference shell ROUND(i / s)
    # (ml_optimiser.cpp:6840) and the class image is in RELION sigma2 times box_g**4.
    image = table.reshape(2, 28, 28)
    for i in range(1, 14):
        j = int(np.floor(i / s + 0.5))
        np.testing.assert_allclose(image[:, 14, 14 + i], radial[:, j] / REF_BOX**4 * 28**4, rtol=1e-12)


@pytest.mark.unit
def test_noise_sums_follow_relion_mstep_remap():
    shape_class = _half().classes[1]
    s = shape_class.scale
    sums = np.arange(2 * 15, dtype=np.float64).reshape(2, 15) * 28**4
    out = optics_shapes.noise_sums_to_reference(sums, shape_class, REF_BOX)
    expected = np.zeros((2, 17))
    for g in range(2):
        for i in range(15):
            j = int(np.floor(i / s + 0.5))  # ml_optimiser.cpp:9100-9107
            if j < 17:
                expected[g, j] += sums[g, i] / 28**4 * REF_BOX**4
    np.testing.assert_allclose(out, expected, rtol=1e-12)


@pytest.mark.unit
def test_class_kwargs_units_and_sizes():
    half = _half()
    b = half.classes[1]
    kwargs = dict(
        experiment_dataset=half,
        image_corrections_k=np.arange(5.0),
        translation_search_base=np.arange(10.0).reshape(5, 2),
        current_translations=np.array([[0.0, 0.0], [1.0, -1.0]]),
        translation_log_prior=np.zeros(2),
        cs_for_engine=20,
        model_current_size_for_engine=None,
        optics_group_ids_k=np.array([0, 1, 1, 0, 0]),
        # A replay's translation grid (relion_replay), [T, 2], not per image.
        replay_prior_translations=np.array([[0.0, 0.0], [2.0, -2.0], [4.0, 0.0]]),
    )
    out = optics_shapes.class_kwargs(kwargs, b, 5)
    np.testing.assert_allclose(out["replay_prior_translations"], kwargs["replay_prior_translations"] * 0.75)
    assert_matches(out["image_corrections_k"], [1.0, 2.0])
    np.testing.assert_allclose(out["translation_search_base"], np.arange(10.0).reshape(5, 2)[[1, 2]] * 0.75)
    np.testing.assert_allclose(out["current_translations"], kwargs["current_translations"] * 0.75)
    assert_matches(out["translation_log_prior"], np.zeros(2))  # not per image
    assert out["cs_for_engine"] == out["model_current_size_for_engine"] == 2 * int(np.ceil(0.5 * b.scale * 20))
    assert out["reference_current_size"] == 20 and out["projection_scale"] == b.scale
    # The engines see the class's images on the reference volume grid.
    assert out["experiment_dataset"].volume_shape == (32, 32, 32)
    assert out["experiment_dataset"].image_shape == (28, 28) and out["experiment_dataset"]._dataset is b.dataset
    assert optics_shapes.class_kwargs(kwargs, half.classes[0], 5)["experiment_dataset"] is half.classes[0].dataset


def _fake_result(n, images, value, groups=2, box=32):
    stats = RelionStats(
        log_evidence_per_image=images.astype(float),
        best_log_score_per_image=-images.astype(float),
        max_posterior_per_image=np.full(n, value),
        rotation_posterior_sums=np.ones(3) * value,
    )
    noise = make_noise_stats(
        wsum_sigma2_noise=np.ones((groups, box // 2 + 1)) * box**4,
        wsum_img_power=np.zeros((groups, box // 2 + 1)),
        wsum_sigma2_offset=value,
        sumw=np.full(groups, value),
        wsum_norm_correction=images.astype(float) * 10,
    )
    return HalfScoreResult(
        ha=images.astype(np.int32),
        Ft_y=np.full(4, value),
        Ft_ctf=np.full(4, 2 * value),
        em_stats=stats,
        noise_stats=noise,
        best_pose_translations=np.ones((n, 2)),
        coarse_ha=images.astype(np.int32),
        mstep_accumulator_shape=(3, 3, 3),
    )




def _dense_owners(half, optics, *, class_batch_overrides=None):
    group_ids = np.empty(half.n_units, dtype=np.int32)
    for group, shape_class in enumerate(half.classes):
        group_ids[shape_class.image_indices] = group
    return (
        half_scoring.HalfScoringData(
            particles=HalfSet(
                index=0,
                dataset=half,
                optics_group_ids=group_ids,
                image_corrections=np.arange(half.n_units, dtype=float),
                scale_corrections=np.ones(half.n_units),
            ),
            reference=np.zeros(4),
            mean_variance=np.ones(4),
            noise_variance=None,
        ),
        half_scoring.DenseSamplingSpec(
            effective_rotations=np.eye(3)[None],
            current_translations=np.zeros((1, 2)),
            base_translations=np.zeros((1, 2)),
            current_healpix_order=0,
            oversampling_order=1,
            translation_step=1.0,
            random_perturbation=0.0,
            cs_for_engine=None,
        ),
        half_scoring.DensePriorSpec(
            rotation_log_prior_k=None,
            class_rotation_log_prior_k=None,
            translation_log_prior=None,
            translation_search_base=np.zeros((half.n_units, 2)),
            trans_prior_center_for_engine=np.zeros((half.n_units, 2)),
            class_log_priors=None,
        ),
        half_scoring.DenseBatchPolicy(
            image_batch_size=1,
            safe_batch_sizes=lambda *_args, **_kwargs: (1, 1),
            max_significants=-1,
            class_batch_overrides=class_batch_overrides,
        ),
        half_scoring.DenseVariantPolicy(
            firstiter_score_mode_this_iter="gaussian",
            firstiter_winner_take_all_this_iter=False,
            k_class_enabled=False,
            relion_firstiter_cc_this_iter=False,
        ),
        half_scoring.DenseExecutionPolicy(
            disc_type="linear_interp",
            disable_adjoint_y=False,
            disable_adjoint_ctf=False,
        ),
        optics,
    )


@pytest.mark.unit
def test_shape_scoring_releases_unused_class_summaries_before_the_next_shape(monkeypatch):
    half = _half()
    summaries = []

    def fake_score(half, *owners):
        assert all(reference() is None for reference in summaries)
        rows = np.asarray(half.particles.image_corrections).astype(int)
        result = _fake_result(rows.size, rows, 1.0, box=half.particles.dataset.image_shape[0])
        result.classes = ClassScoreSummary(
            assignments=np.zeros(rows.size, dtype=np.int32),
            mstep_mass=np.ones(2), evidence_mass=np.ones(2),
            rotation_mass=np.ones((2, 3)), noise_stats=object(),
        )
        summaries.append(weakref.ref(result.classes))
        return result

    monkeypatch.setattr(half_scoring, "_score_half_dense_one_shape", fake_score)
    owners = _dense_owners(
        half, optics_shapes.OpticsSpec(noise_radial_k=np.ones((2, 17)) * REF_BOX**4),
    )
    merged = half_scoring._score_half_dense(*owners)

    assert len(summaries) == 2
    assert all(reference() is None for reference in summaries)
    assert merged.classes is None


@pytest.mark.unit
def test_dense_owner_shape_derivation_passes_class_translations_through_the_merge(monkeypatch):
    half = _half()
    outputs = PerHalfOutputs()
    seen = []
    optics = optics_shapes.prepare_optics(
        half, noise_radial=np.ones((2, 17)) * REF_BOX**4, previous_translations=np.full((5, 2), 1.4),
        sigma_offset_angstrom=10.0, base_translations=np.zeros((1, 2)), current_translations=np.zeros((1, 2)),
        with_log_prior=True, zero_cold_center=False,
    )

    def fake_score(half, sampling, priors, batching, variant, execution, optics):
        seen.append((half, sampling, priors, batching, variant, execution, optics))
        images = np.asarray(half.particles.image_corrections).astype(int)
        return _fake_result(
            images.size,
            images,
            1.0,
            box=half.particles.dataset.image_shape[0],
        )

    monkeypatch.setattr(half_scoring, "_score_half_dense_one_shape", fake_score)
    owners = _dense_owners(half, optics)

    merged = half_scoring._score_half_dense(*owners)
    outputs.update_from(0, merged)

    assert [getattr(item[0].particles.dataset, "_dataset", item[0].particles.dataset) for item in seen] == [
        shape_class.dataset for shape_class in half.classes
    ]
    assert seen[1][0].noise_variance.shape == (2, 28 * 28)
    for owner, translations in zip(seen, optics.class_translations, strict=True):
        assert owner[2].translation_search_base is translations.search_base
        assert owner[2].trans_prior_center_for_engine is translations.engine_prior_center
        assert owner[2].translation_log_prior is translations.log_prior
        assert owner[6].class_translations is None
    assert_matches(merged.ha, np.arange(5))
    assert outputs.best_pose_translations[0] is merged.best_pose_translations
    assert_matches(merged.em_stats.log_evidence_per_image, np.arange(5.0))
    # Norm residuals are power sums: the 28-px class's go to the reference box x (32/28)^4.
    expected_norm = np.arange(5.0) * 10
    expected_norm[[1, 2]] *= (32 / 28) ** 4
    assert_matches(merged.noise_stats.wsum_norm_correction, expected_norm)
    # The 28-px class's sums join the 32-px reference's in its native units: data x (28/32)^2,
    # weight x (28/32)^4 (_to_reference_units).
    assert_matches(merged.Ft_y, np.full(4, 1.0 + (28 / 32) ** 2))
    assert_matches(merged.Ft_ctf, np.full(4, 2.0 + 2.0 * (28 / 32) ** 4))
    np.testing.assert_allclose(merged.noise_stats.sumw, [2.0, 2.0])
    np.testing.assert_allclose(merged.em_stats.rotation_posterior_sums, np.full(3, 2.0))
    # Class translations come back in reference pixels.
    np.testing.assert_allclose(merged.best_pose_translations[[1, 2]], 1.0 / half.classes[1].translation_factor)
    np.testing.assert_allclose(merged.best_pose_translations[[0, 3, 4]], 1.0)
    assert outputs.best_pose_translations[0] is merged.best_pose_translations
    # Both classes' noise shells land on the 17 reference shells.
    assert merged.noise_stats.wsum_sigma2_noise.shape == (2, 17)


def _write_fake_class_outputs(outputs, k, images, box):
    # Two Class3D classes; image i is assigned class i % 2, and the sums are per shape class.
    outputs.class_assignments[k] = (images % 2).astype(np.int32)
    outputs.class_posterior[k] = np.array([1.0, 2.0])
    outputs.class_full_posterior[k] = np.array([1.5, 2.5])
    outputs.class_rotation_posterior[k] = np.ones((2, 3))
    outputs.best_pose_rotations[k] = np.broadcast_to(np.eye(3), (images.size, 3, 3)) * images[:, None, None]
    outputs.best_pose_rotation_eulers[k] = np.zeros((images.size, 3)) + images[:, None]
    outputs.best_pose_translations[k] = np.ones((images.size, 2), dtype=np.float32)
    outputs.noise_stats_per_class[k] = [
        make_noise_stats(
            wsum_sigma2_noise=np.ones((2, box // 2 + 1)) * box**4,
            wsum_img_power=np.zeros((2, box // 2 + 1)),
            wsum_sigma2_offset=1.0,
            sumw=np.full(2, 1.0 + c),
            wsum_norm_correction=images.astype(float) * 10,
        )
        for c in range(2)
    ]


@pytest.mark.unit
@pytest.mark.parametrize("k_class_enabled", [False, True])
def test_dense_owner_shape_derivation_preserves_multi_shape_merge(monkeypatch, k_class_enabled):
    half = _half()
    outputs = PerHalfOutputs()
    seen = []

    def fake_score(half, sampling, priors, batching, variant, execution, optics):
        seen.append((half, sampling, priors, batching, variant, execution, optics))
        images = np.asarray(half.particles.image_corrections).astype(int)
        box = half.particles.dataset.image_shape[0]
        result = _fake_result(images.size, images, 1.0, box=box)
        if variant.k_class_enabled:
            class_outputs = PerHalfOutputs()
            _write_fake_class_outputs(class_outputs, 0, images, box)
            result.classes = ClassScoreSummary(
                assignments=class_outputs.class_assignments[0],
                mstep_mass=class_outputs.class_posterior[0],
                evidence_mass=class_outputs.class_full_posterior[0],
                rotation_mass=class_outputs.class_rotation_posterior[0],
                noise_stats=class_outputs.noise_stats_per_class[0],
            )
            result.best_pose_rotations = class_outputs.best_pose_rotations[0]
            result.best_pose_rotation_eulers = class_outputs.best_pose_rotation_eulers[0]
            result.best_pose_translations = class_outputs.best_pose_translations[0]
        return result

    monkeypatch.setattr(half_scoring, "_score_half_dense_one_shape", fake_score)
    owners = list(_dense_owners(half, optics_shapes.OpticsSpec(noise_radial_k=np.ones((2, 17)) * REF_BOX**4)))
    owners[4] = dataclasses.replace(owners[4], k_class_enabled=k_class_enabled)

    merged = half_scoring._score_half_dense(*owners)
    outputs.update_from(0, merged)

    if k_class_enabled:
        # Class assignments return to the half's image order; the class sums of both shape
        # classes add; each class's noise sums land on the reference shells.
        assert_matches(outputs.class_assignments[0], np.arange(5) % 2)
        np.testing.assert_allclose(outputs.class_posterior[0], [2.0, 4.0])
        np.testing.assert_allclose(outputs.class_full_posterior[0], [3.0, 5.0])
        np.testing.assert_allclose(outputs.class_rotation_posterior[0], np.full((2, 3), 2.0))
        per_class = outputs.noise_stats_per_class[0]
        assert len(per_class) == 2
        np.testing.assert_allclose(per_class[1].sumw, [4.0, 4.0])
        assert per_class[0].wsum_sigma2_noise.shape == (2, 17)
        expected_norm = np.arange(5.0) * 10
        expected_norm[[1, 2]] *= (32 / 28) ** 4
        assert_matches(per_class[0].wsum_norm_correction, expected_norm)
        # Best poses come from the class outputs, in the half's order and reference pixels.
        np.testing.assert_allclose(outputs.best_pose_rotation_eulers[0][:, 0], np.arange(5))
        np.testing.assert_allclose(outputs.best_pose_rotations[0][:, 0, 0], np.arange(5))
        np.testing.assert_allclose(outputs.best_pose_translations[0][[1, 2]], 1.0 / half.classes[1].translation_factor)
        np.testing.assert_allclose(outputs.best_pose_translations[0][[0, 3, 4]], 1.0)
        assert outputs.best_pose_translations[0].dtype == np.float32
    else:
        assert outputs.class_assignments[0] is None

    assert [getattr(item[0].particles.dataset, "_dataset", item[0].particles.dataset) for item in seen] == [
        shape_class.dataset for shape_class in half.classes
    ]
    assert seen[1][0].noise_variance.shape == (2, 28 * 28)
    assert_matches(merged.ha, np.arange(5))
    if not k_class_enabled:
        assert outputs.best_pose_translations[0] is merged.best_pose_translations


@pytest.mark.unit
def test_local_owner_shape_derivation_preserves_multi_shape_merge(monkeypatch):
    half = _half()
    outputs = PerHalfOutputs()
    seen = []
    optics = optics_shapes.prepare_optics(
        half, noise_radial=np.ones((2, 17)) * REF_BOX**4, previous_translations=np.full((5, 2), 1.4),
        sigma_offset_angstrom=10.0, base_translations=np.zeros((1, 2)), current_translations=np.zeros((1, 2)),
        with_log_prior=False, zero_cold_center=False,
    )

    def fake_score(half, sampling, priors, batching, execution, diagnostics, optics):
        seen.append((half, sampling, priors, batching, execution, diagnostics, optics))
        images = np.asarray(half.particles.image_corrections).astype(int)
        return _fake_result(
            images.size,
            images,
            1.0,
            box=half.particles.dataset.image_shape[0],
        )

    monkeypatch.setattr(half_scoring, "_score_half_local_one_shape", fake_score)
    owners = local_half_owners(
        k=0,
        experiment_dataset=half,
        means_k=np.zeros(4),
        noise_variance_k=None,
        previous_best_rotation_eulers_k=np.zeros((5, 3)),
        image_corrections_k=np.arange(5.0),
        scale_corrections_k=np.ones(5),
        optics_group_ids_k=np.array([0, 1, 1, 0, 0]),
        local_search_rotations=np.eye(3)[None],
        local_search_order=0,
        sigma_rot=0.1,
        sigma_psi=0.1,
        current_translations=np.zeros((1, 2)),
        base_translations=np.zeros((1, 2)),
        disc_type="linear_interp",
        cs_for_engine=None,
        local_pass1_current_size=None,
        local_search_random_perturbation=0.0,
        local_search_angular_sampling_deg=None,
        local_parent_oversampling_order=0,
        trans_prior_center=np.zeros((5, 2)),
        trans_prior_center_for_engine=np.zeros((5, 2)),
        current_sigma_offset_angstrom=1.0,
        translation_search_base=np.zeros((5, 2)),
        local_search_translation_prior_mode="current",
        replay_prior_translations=None,
        max_significants=-1,
        safe_batch_sizes=lambda *_args, **_kwargs: (1, 1),
        disable_adjoint_y=False,
        disable_adjoint_ctf=False,
        iteration=0,
        collect_local_search_profile=False,
        diagnostic_score_only=False,
        local_profile_history=[],
        noise_radial_k=np.ones((2, 17)) * REF_BOX**4,
        class_translations=optics.class_translations,
    )

    merged = half_scoring._score_half_local(*owners)
    outputs.update_from(0, merged)

    assert [
        getattr(item[0].particles.dataset, "_dataset", item[0].particles.dataset)
        for item in seen
    ] == [shape_class.dataset for shape_class in half.classes]
    assert seen[1][0].noise_variance.shape == (2, 28 * 28)
    for owner, translations in zip(seen, optics.class_translations, strict=True):
        assert owner[2].translation_search_base is translations.search_base
        assert owner[2].trans_prior_center is translations.local_prior_center
        assert owner[2].trans_prior_center_for_engine is translations.engine_prior_center
        assert owner[6].class_translations is None
    assert_matches(merged.ha, np.arange(5))
    assert outputs.best_pose_translations[0] is merged.best_pose_translations


@pytest.mark.unit
def test_class_coarse_size_is_relions_formula_at_the_class_grid():
    # image_coarse_size[g] = min(2 CEIL(remap * pixel_ref * ori / coarse_res), image_current_size[g])
    # (ml_optimiser.cpp:5761-5777); remap * pixel_ref * ori is the class's own pixel * box.
    half = _half()
    step, diameter = 34.5, 100.0
    coarse_res = (step / 360.0) * np.pi * diameter / 1.2
    kwargs = dict(experiment_dataset=half, cs_for_engine=20, firstiter_coarse_current_size=12,
                  coarse_sizing=(step, diameter))
    got = []
    for shape_class in half.classes:
        out = optics_shapes.class_kwargs(kwargs, shape_class, 5)
        assert "coarse_sizing" not in out
        expected = 2 * int(np.ceil(shape_class.pixel_size * shape_class.box_size / coarse_res))
        assert out["firstiter_coarse_current_size"] == min(expected, out["cs_for_engine"])
        got.append(out["firstiter_coarse_current_size"])
    # The reference class keeps its size; the other class's differs from a current-size remap (14).
    assert got == [12, 12]
    out = optics_shapes.class_kwargs(dict(kwargs, coarse_sizing=None), half.classes[1], 5)
    assert out["firstiter_coarse_current_size"] == 2 * int(np.ceil(0.5 * half.classes[1].scale * 12)) == 14


@pytest.mark.unit
def test_adaptive_batches_are_planned_per_class_box(monkeypatch):
    from relax.helpers.batch_planning import _AdaptiveDenseBatchSizes
    from relax.refinement import expectation_batches

    ds_a = SimpleNamespace(image_shape=(32, 32), volume_shape=(32,) * 3, voxel_size=4.0)
    ds_b = SimpleNamespace(image_shape=(40, 40), volume_shape=(40,) * 3, voxel_size=4.0)  # larger box
    classes = optics_shapes.make_shape_classes(
        [(ds_a, np.array([0, 2])), (ds_b, np.array([1]))], ref_box=REF_BOX, ref_pixel=REF_PIX
    )
    half = optics_shapes.MultiShapeHalf(classes, image_shape=(32, 32), volume_shape=(32,) * 3, voxel_size=4.0)
    seen = []

    def plan(*, image_shape, cs_for_engine, coarse_cs):
        seen.append((tuple(image_shape), cs_for_engine, coarse_cs))
        n = image_shape[0]
        return _AdaptiveDenseBatchSizes(1000 // n, 2000 // n, 3000 // n, 4000 // n)

    sizing = (34.5, 100.0)
    overrides = expectation_batches._class_adaptive_batch_overrides(
        half, plan=plan, cs_for_engine=20, coarse_cs=12, coarse_sizing=sizing
    )
    expected_sizes = [optics_shapes.class_adaptive_sizes(c, 20, 12, sizing) for c in classes]
    assert seen == [((32, 32),) + expected_sizes[0], ((40, 40),) + expected_sizes[1]]
    assert expected_sizes[1][0] == 26  # 2 ceil(0.5 * 1.25 * 20)
    assert overrides[1] == {
        "k_class_image_batch_size_override": 25,
        "k_class_rotation_block_size_override": 50,
        "significance_image_batch_size_override": 75,
        "significance_rotation_block_size_override": 100,
    }
    assert expectation_batches._largest_image_size(half) == 40

    # Each class's call receives its own plan.
    received = []
    def score(half, sampling, priors, batching, variant, execution, optics):
        received.append(batching.k_class_image_batch_size_override)
        images = np.asarray(half.particles.image_corrections).astype(int)
        return _fake_result(images.size, images, 1.0, box=half.particles.dataset.image_shape[0])

    monkeypatch.setattr(half_scoring, "_score_half_dense_one_shape", score)
    owners = _dense_owners(
        half, optics_shapes.OpticsSpec(noise_radial_k=np.ones((2, 17)) * REF_BOX**4),
        class_batch_overrides=overrides,
    )
    half_scoring._score_half_dense(*owners)
    assert received == [1000 // 32, 25]


@pytest.mark.unit
def test_full_box_reference_size_is_explicit_for_a_class_on_another_grid():
    # A full-box pass (no current size) still fills the backprojector at the reference box.
    half = _half()
    kwargs = dict(experiment_dataset=half, cs_for_engine=None)
    assert optics_shapes.class_kwargs(kwargs, half.classes[1], 5)["reference_current_size"] == REF_BOX
    assert optics_shapes.class_kwargs(kwargs, half.classes[0], 5)["reference_current_size"] is None


@pytest.mark.unit
def test_class_pre_shifts_are_rounded_in_the_class_pixels():
    # RELION rounds a particle's stored offset in its own image pixels (ml_optimiser.cpp:6085),
    # so a class's pre-shift is round(offset * factor), not round(offset) * factor.
    half = _half()
    previous = np.array([[1.4, -2.6], [2.5, 1.1], [-0.6, 3.3], [0.2, 0.2], [4.4, -4.4]])
    grid = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    out = optics_shapes.prepare_optics(
        half, noise_radial=np.ones((2, 17)), previous_translations=previous,
        sigma_offset_angstrom=10.0, base_translations=grid, current_translations=grid,
        with_log_prior=True, zero_cold_center=True,
    ).class_translations
    for shape_class, values in zip(half.classes, out):
        in_class = previous[shape_class.image_indices] * shape_class.translation_factor
        expected = np.sign(in_class) * np.floor(np.abs(in_class) + 0.5)
        assert_matches(values.search_base, expected.astype(np.float32))
        assert values.log_prior.shape == (shape_class.image_indices.size, 3)
    assert optics_shapes.prepare_optics(
        object(), noise_radial=None, previous_translations=previous,
        sigma_offset_angstrom=1.0, base_translations=grid, current_translations=grid,
        with_log_prior=False, zero_cold_center=False,
    ) == optics_shapes.OpticsSpec()


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("with_log_prior,zero_cold_center", [(False, False), (True, False), (True, True)])
def test_prepared_optics_preserves_cold_start_prior_policy(dtype, with_log_prior, zero_cold_center):
    half = _half()
    grid = np.array([[0.0, 0.0], [1.0, -1.0], [2.0, 0.0]], dtype=dtype)
    noise = np.ones((2, 17), dtype=np.float64)
    prepared = optics_shapes.prepare_optics(
        half, noise_radial=noise, previous_translations=None, sigma_offset_angstrom=10.0,
        base_translations=grid, current_translations=grid, with_log_prior=with_log_prior,
        zero_cold_center=zero_cold_center, coarse_step_deg=34.5, particle_diameter_ang=100.0, dtype=dtype,
    )
    assert prepared.noise_radial_k is noise
    assert prepared.coarse_sizing == (34.5, 100.0)
    for shape_class, translations in zip(half.classes, prepared.class_translations, strict=True):
        assert translations.search_base is translations.local_prior_center is None
        assert translations.engine_prior_center.dtype == dtype
        assert_matches(translations.engine_prior_center, np.zeros(2, dtype=dtype))
        if not with_log_prior:
            assert translations.log_prior is None
        else:
            assert translations.log_prior.dtype == dtype
            expected = np.zeros(3, dtype=dtype)
            if zero_cold_center:
                # pdf_offset evaluated in class pixels, with the existing angstrom scale.
                class_grid = grid * shape_class.translation_factor
                expected = -0.5 * np.sum(class_grid**2, axis=1) * shape_class.pixel_size**4 / 100.0
            assert_matches(translations.log_prior, expected)


@pytest.mark.unit
def test_single_shape_preparation_does_not_materialize_operands():
    class UnreadableArray:
        def __array__(self, *args, **kwargs):
            raise AssertionError("Single-shape optics preparation must not read arrays")

    array = UnreadableArray()
    assert optics_shapes.prepare_optics(
        object(), noise_radial=array, previous_translations=array, sigma_offset_angstrom=1.0,
        base_translations=array, current_translations=array, with_log_prior=True, zero_cold_center=True,
    ) == optics_shapes.OpticsSpec()


@pytest.mark.unit
def test_class_translation_step_is_in_class_pixels():
    # The adaptive pass 2 builds its oversampled translations from translation_step; a class
    # on another pixel size samples the same Angstrom offsets, so its step is in its pixels.
    half = _half()
    kwargs = dict(experiment_dataset=half, translation_step=2.0)
    assert optics_shapes.class_kwargs(kwargs, half.classes[1], 5)["translation_step"] == pytest.approx(2.0 * 0.75)
    assert optics_shapes.class_kwargs(kwargs, half.classes[0], 5)["translation_step"] == kwargs["translation_step"]


@pytest.mark.unit
def test_class_mstep_image_radius_is_reference_r_max_times_scale():
    from relax.refinement.optics_shapes import reconstruction_image_radius

    # RELION's backprojector bounds the reference-grid radius by r_max = cs // 2; a class
    # image pixel |k| lands at |k| / s, so the class keeps pixels out to r_max * s.
    shape_class = _half().classes[1]
    radius = reconstruction_image_radius(56, shape_class.scale)
    assert radius == pytest.approx(28 * shape_class.scale)
    assert radius != 28.0
    assert reconstruction_image_radius(None, shape_class.scale) is None


@pytest.mark.unit
def test_class_mstep_clip_keeps_the_reference_padding():
    import inspect

    from relax.helpers import adjoint
    from relax.sparse_pass2 import resident_pass2

    # One grid: the reference r_max, from which recovar infers RELION's pad size.
    assert adjoint.mstep_adjoint_max_r(56, None, 2) == pytest.approx(28.0)
    assert adjoint._recovar_clip_kwargs(28.0)["max_r"] == pytest.approx(28.0)
    # Another grid: the image radius r_max * s clips, and the padding is explicit (r_max * s
    # alone would make recovar infer padding 1 on a 115 grid for a 112 px image).
    clip = adjoint.mstep_adjoint_max_r(56, 28 * 1.12, 2)
    assert isinstance(clip, adjoint.ReferenceSphereClip)
    assert clip.image_radius == pytest.approx(28 * 1.12) and clip.upsampling == 2
    kwargs = adjoint._recovar_clip_kwargs(clip)
    assert kwargs["max_r"] == pytest.approx(28 * 1.12) and kwargs["upsampling"] == 2
    # The resident chunk loop builds its own program spec; it must be handed the clip.
    parameter = inspect.signature(resident_pass2._run_resident_chunk).parameters["mstep_max_r"]
    assert parameter.default is inspect.Parameter.empty


@pytest.mark.unit
def test_image_clip_at_r_max_times_scale_is_relions_rotated_radius_rule():
    """RELION drops a sample when |A k| > r_max for the rotated, scaled matrix A (BP.cuh:322).

    With A = R / s for an orthogonal R, that is |k| > r_max * s on the image lattice, pixel for
    pixel, which is what the class adjoint's image-side clip keeps.
    """
    from scipy.spatial.transform import Rotation

    r_max, scale, box = 28, 112 * 5.44 / (128 * 4.25), 112
    k = np.fft.fftfreq(box) * box
    ky, kx = np.meshgrid(k, np.arange(box // 2 + 1), indexing="ij")
    pixels = np.stack([kx.ravel(), ky.ravel(), np.zeros(kx.size)], axis=1)
    image_keep = np.hypot(kx, ky).ravel() <= r_max * scale
    for rotation in Rotation.random(5, random_state=3).as_matrix():
        rotated = pixels @ (rotation / scale).T
        relion_keep = np.linalg.norm(rotated, axis=1) <= r_max
        boundary = np.abs(np.linalg.norm(rotated, axis=1) - r_max) < 1e-9
        assert np.array_equal(image_keep[~boundary], relion_keep[~boundary])


@pytest.mark.unit
def test_a_group_spanning_sqrt2_times_the_reference_is_refused():
    # RELION's fine kernels project a moved pixel inside the sphere from sqrt(2) on
    # (relion_kernel_zero_rows); relax refuses rather than diverge silently.
    ds_ref = SimpleNamespace(image_shape=(32, 32), volume_shape=(32,) * 3, voxel_size=4.0)
    ds_wide = SimpleNamespace(image_shape=(32, 32), volume_shape=(32,) * 3, voxel_size=4.0 * 1.5)
    with pytest.raises(NotImplementedError, match="largest box x pixel size first"):
        optics_shapes.make_shape_classes([(ds_ref, np.array([0])), (ds_wide, np.array([1]))], ref_box=REF_BOX, ref_pixel=REF_PIX)


@pytest.mark.unit
def test_local_parent_pass_refuses_the_wrapped_coarse_band():
    ds_a = SimpleNamespace(image_shape=(32, 32), volume_shape=(32,) * 3, voxel_size=4.0)
    ds_b = SimpleNamespace(image_shape=(40, 40), volume_shape=(40,) * 3, voxel_size=4.0)  # s = 1.25
    classes = optics_shapes.make_shape_classes([(ds_a, np.array([0])), (ds_b, np.array([1]))], ref_box=REF_BOX, ref_pixel=REF_PIX)
    half = optics_shapes.MultiShapeHalf(classes, image_shape=(32, 32), volume_shape=(32,) * 3, voxel_size=4.0)
    # r_max 10; pass-1 18 -> class window 24: 12 lies between 10 and 1.25 * sqrt(101).
    with pytest.raises(NotImplementedError, match="fused coarse scorer"):
        optics_shapes.require_exact_local_parent_windows({"experiment_dataset": half, "cs_for_engine": 20, "local_pass1_current_size": 18})
    # Pass-1 20 -> class window 26 lies past the band.
    optics_shapes.require_exact_local_parent_windows({"experiment_dataset": half, "cs_for_engine": 20, "local_pass1_current_size": 20})


@pytest.mark.unit
def test_shape_class_engine_inputs_follow_the_class_rules(monkeypatch):
    from relax.relion import optics_aberrations

    monkeypatch.setattr(optics_aberrations, "dataset_projection_magnification", lambda dataset: None)
    half = _half()
    small = half.classes[1]
    rotations = np.stack([np.eye(3), 2 * np.eye(3)])
    inputs = optics_shapes.shape_class_engine_inputs(
        small,
        half,
        noise_radial=np.ones((2, 17)) * REF_BOX**4,
        coarse_rotations=rotations,
        fine_rotations=rotations,
        fine_mstep_rotations=None,
        coarse_translations=np.ones((3, 2)),
        fine_translations=np.full((4, 2), 2.0),
        coarse_current_size=16,
        fine_current_size=24,
        reference_current_size=None,
        engine_kwargs={
            "optics_group_ids": np.array([0, 1, 1, 0, 0]),
            "coarse_translation_log_prior": np.arange(15.0).reshape(5, 3),
            "translation_log_prior": np.zeros(4),
            "reconstruction_group_ids": np.array([0, 1, 2, 3, 0]),
        },
    )
    # applyScaleDifference: the projector divides the matrices by s.
    np.testing.assert_allclose(inputs.coarse_rotations, rotations / small.scale)
    np.testing.assert_allclose(inputs.fine_rotations, rotations / small.scale)
    assert inputs.fine_mstep_rotations is None
    np.testing.assert_allclose(inputs.fine_translations, 2.0 * small.translation_factor)
    # The same remap and class rows as the half-level route (class_kwargs).
    assert (inputs.fine_current_size, inputs.coarse_current_size) == optics_shapes.class_adaptive_sizes(
        small, 24, 16, None
    )
    np.testing.assert_array_equal(inputs.engine_kwargs["optics_group_ids"], [1, 1])
    np.testing.assert_array_equal(inputs.engine_kwargs["reconstruction_group_ids"], [1, 2])
    np.testing.assert_allclose(inputs.engine_kwargs["coarse_translation_log_prior"], [[3, 4, 5], [6, 7, 8]])
    assert inputs.engine_kwargs["translation_log_prior"].shape == (4,)
    # The backprojector stays on the reference model grid at the reference pass-2 size.
    assert inputs.engine_kwargs["reconstruction_volume_current_size"] == 24
    assert inputs.engine_kwargs["reconstruction_image_radius"] == pytest.approx(12 * small.scale)
    assert inputs.noise_variance.shape == (2, 28 * 28)
    with pytest.raises(ValueError, match="rebuilt in each shape class"):
        optics_shapes.shape_class_engine_inputs(
            small,
            half,
            noise_radial=np.ones((2, 17)),
            coarse_rotations=rotations,
            fine_rotations=rotations,
            fine_mstep_rotations=None,
            coarse_translations=np.ones((3, 2)),
            fine_translations=np.ones((4, 2)),
            coarse_current_size=None,
            fine_current_size=None,
            reference_current_size=None,
            engine_kwargs={"image_pre_shifts": np.ones((5, 2))},
        )


def _fake_engine_result(images, box, factor):
    from relax.classification.k_class_results import KClassEMResult

    n = images.size
    stats = RelionStats(
        log_evidence_per_image=images.astype(float),
        best_log_score_per_image=-images.astype(float),
        max_posterior_per_image=np.ones(n),
        rotation_posterior_sums=np.ones(3),
    )
    noise = make_noise_stats(
        wsum_sigma2_noise=np.ones((2, box // 2 + 1)) * box**4,
        wsum_img_power=np.zeros((2, box // 2 + 1)),
        wsum_sigma2_offset=1.0,
        sumw=np.ones(2),
        wsum_norm_correction=images.astype(float),
    )
    return KClassEMResult(
        new_means=None,
        Ft_y=np.ones((2, 4)),
        Ft_ctf=np.ones((2, 4)),
        per_class_hard_assignments=np.stack([images, images + 100]),
        class_assignments=images % 2,
        pose_assignments=images,
        class_responsibilities=np.stack([images * 0.1, 1 - images * 0.1]),
        class_posterior_sums=np.array([1.0, 2.0]),
        stats=stats,
        per_class_stats=(stats, stats),
        noise_stats=(noise, noise),
        aggregate_noise_stats=noise,
        per_class_best_pose_translations=(np.ones((n, 2)) * factor, np.ones((n, 2)) * factor),
        best_pose_translations=np.ones((n, 2)) * factor,
        significant_counts=images + 1,
        class_mstep_posterior_sums=np.array([0.5, 0.5]),
    )


@pytest.mark.unit
def test_merge_k_class_engine_results_places_images_and_adds_sums():
    half = _half()
    results = [
        _fake_engine_result(c.image_indices, c.box_size, c.translation_factor) for c in half.classes
    ]
    merged = optics_shapes.merge_k_class_engine_results(results, half.classes, half.n_units, REF_BOX)
    images = np.arange(5)
    np.testing.assert_array_equal(merged.pose_assignments, images)
    np.testing.assert_array_equal(merged.class_assignments, images % 2)
    # [K, N] fields return on their image axis.
    np.testing.assert_array_equal(merged.per_class_hard_assignments, np.stack([images, images + 100]))
    np.testing.assert_allclose(merged.class_responsibilities[0], images * 0.1)
    np.testing.assert_allclose(merged.class_posterior_sums, [2.0, 4.0])
    np.testing.assert_allclose(merged.class_mstep_posterior_sums, [1.0, 1.0])
    np.testing.assert_allclose(merged.stats.log_evidence_per_image, images)
    np.testing.assert_allclose(merged.per_class_stats[1].rotation_posterior_sums, np.full(3, 2.0))
    # Best translations back in reference pixels; accumulators in reference units.
    np.testing.assert_allclose(merged.best_pose_translations, 1.0)
    np.testing.assert_allclose(merged.per_class_best_pose_translations[1], 1.0)
    np.testing.assert_allclose(merged.Ft_y, 1.0 + (28 / 32) ** 2)
    np.testing.assert_allclose(merged.Ft_ctf, 1.0 + (28 / 32) ** 4)
    assert merged.aggregate_noise_stats.wsum_sigma2_noise.shape == (2, 17)
    np.testing.assert_allclose(merged.noise_stats[0].sumw, [2.0, 2.0])
    np.testing.assert_array_equal(merged.significant_counts, images + 1)


def test_bpref_cubes_on_different_windows_merge_on_the_smaller_centred_cube():
    # A model-grid class may return its BPref on a stable physical cube (7^3) while a class at its own
    # box returns the logical cube (5^3); the merge sums the logical voxels, as crop_public_full_volume
    # would select them, and leaves equal cubes alone.
    from relax.helpers.half_volume_mstep import crop_public_full_volume

    rng = np.random.default_rng(0)
    physical = rng.standard_normal((1, 2, 7**3))
    logical = rng.standard_normal((1, 2, 5**3))
    cut = optics_shapes._common_centered_cubes([physical, logical])
    expected = np.stack(
        [np.asarray(crop_public_full_volume(physical[0, h], (7, 7, 7), (5, 5, 5))) for h in range(2)]
    )[None]
    # An index-selection check: the cut moves elements without arithmetic, so exact equality is the contract.
    np.testing.assert_array_equal(cut[0], expected)
    assert cut[1] is logical
    same = [physical, physical.copy()]
    assert optics_shapes._common_centered_cubes(same) is same
    with pytest.raises(ValueError, match="not odd cubes"):
        optics_shapes._common_centered_cubes([physical, np.zeros((1, 2, 8**3))])
