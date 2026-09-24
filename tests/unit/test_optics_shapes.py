"""Half scoring over shape classes (optics groups on other pixel sizes and boxes, CPU).

Checks the RELION rules the per-class calls follow (noise shell remaps both ways,
translations in class pixels, remapped Fourier sizes, scaled projection) and that
per-image outputs return to the half's order by index while sums add.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from relax.dense.score_outputs import HalfScoreResult, PerHalfOutputs
from relax.helpers.types import RelionStats, make_noise_stats
from relax.refinement import optics_shapes

REF_BOX, REF_PIX = 32, 4.0


def _half():
    ds_a = SimpleNamespace(image_shape=(32, 32), voxel_size=4.0)
    ds_b = SimpleNamespace(image_shape=(28, 28), voxel_size=4.0 * 32 / 24)
    classes = optics_shapes.make_shape_classes(
        [(ds_a, np.array([0, 3, 4])), (ds_b, np.array([1, 2]))], ref_box=REF_BOX, ref_pixel=REF_PIX
    )
    return optics_shapes.MultiShapeHalf(classes, image_shape=(32, 32), volume_shape=(32, 32, 32), voxel_size=4.0)


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
    )
    out = optics_shapes.class_kwargs(kwargs, b, 5)
    np.testing.assert_array_equal(out["image_corrections_k"], [1.0, 2.0])
    np.testing.assert_allclose(out["translation_search_base"], np.arange(10.0).reshape(5, 2)[[1, 2]] * 0.75)
    np.testing.assert_allclose(out["current_translations"], kwargs["current_translations"] * 0.75)
    np.testing.assert_array_equal(out["translation_log_prior"], np.zeros(2))  # not per image
    assert out["cs_for_engine"] == out["model_current_size_for_engine"] == 2 * int(np.ceil(0.5 * b.scale * 20))
    assert out["reference_current_size"] == 20 and out["projection_scale"] == b.scale
    assert out["experiment_dataset"] is b.dataset


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


@pytest.mark.unit
def test_score_half_by_shape_places_images_and_adds_sums():
    half = _half()
    seen = []

    def fake_score(**kwargs):
        seen.append(kwargs)
        dataset = kwargs["experiment_dataset"]
        # Each class reports its images by the image corrections it was given.
        images = np.asarray(kwargs["image_corrections_k"]).astype(int)
        return _fake_result(images.size, images, 1.0, box=dataset.image_shape[0])

    outputs = PerHalfOutputs()
    merged = optics_shapes.score_half_by_shape(
        fake_score,
        dict(
            experiment_dataset=half,
            k=0,
            outputs=outputs,
            image_corrections_k=np.arange(5.0),
            optics_group_ids_k=np.array([0, 1, 1, 0, 0]),
            noise_variance_k=None,
            noise_radial_k=np.ones((2, 17)) * REF_BOX**4,
            cs_for_engine=None,
        ),
    )
    assert [kw["experiment_dataset"] for kw in seen] == [c.dataset for c in half.classes]
    assert seen[1]["noise_variance_k"].shape == (2, 28 * 28)
    np.testing.assert_array_equal(merged.ha, np.arange(5))
    np.testing.assert_array_equal(merged.em_stats.log_evidence_per_image, np.arange(5.0))
    np.testing.assert_array_equal(merged.noise_stats.wsum_norm_correction, np.arange(5.0) * 10)
    np.testing.assert_array_equal(merged.Ft_y, np.full(4, 2.0))
    np.testing.assert_allclose(merged.noise_stats.sumw, [2.0, 2.0])
    np.testing.assert_allclose(merged.em_stats.rotation_posterior_sums, np.full(3, 2.0))
    # Class translations come back in reference pixels.
    np.testing.assert_allclose(merged.best_pose_translations[[1, 2]], 1.0 / half.classes[1].translation_factor)
    np.testing.assert_allclose(merged.best_pose_translations[[0, 3, 4]], 1.0)
    assert outputs.best_pose_translations[0] is merged.best_pose_translations
    # Both classes' noise shells land on the 17 reference shells.
    assert merged.noise_stats.wsum_sigma2_noise.shape == (2, 17)
