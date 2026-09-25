"""Half scoring over shape classes (optics groups on other pixel sizes and boxes, CPU).

Checks the RELION rules the per-class calls follow (noise shell remaps both ways,
translations in class pixels, remapped Fourier sizes, scaled projection) and that
per-image outputs return to the half's order by index while sums add.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.dense.score_outputs import HalfScoreResult, PerHalfOutputs
from relax.helpers.types import RelionStats, make_noise_stats
from relax.refinement import optics_shapes

REF_BOX, REF_PIX = 32, 4.0


def _half():
    ds_a = SimpleNamespace(image_shape=(32, 32), volume_shape=(32,) * 3, voxel_size=4.0)
    ds_b = SimpleNamespace(image_shape=(28, 28), volume_shape=(28,) * 3, voxel_size=4.0 * 32 / 24)
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
    assert [getattr(kw["experiment_dataset"], "_dataset", kw["experiment_dataset"]) for kw in seen] == [
        c.dataset for c in half.classes
    ]
    assert seen[1]["noise_variance_k"].shape == (2, 28 * 28)
    assert_matches(merged.ha, np.arange(5))
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
def test_adaptive_batches_are_planned_per_class_box():
    from relax.helpers.batch_planning import _AdaptiveDenseBatchSizes
    from relax.refinement import iteration_loop

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
    overrides = iteration_loop._class_adaptive_batch_overrides(
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
    assert iteration_loop._largest_image_size(half) == 40

    # Each class's call receives its own plan.
    received = []
    optics_shapes.score_half_by_shape(
        lambda **kw: received.append(kw["k_class_image_batch_size_override"])
        or _fake_result(len(kw["image_corrections_k"]), np.asarray(kw["image_corrections_k"]).astype(int), 1.0,
                        box=kw["experiment_dataset"].image_shape[0]),
        dict(
            experiment_dataset=half, k=0, outputs=PerHalfOutputs(), image_corrections_k=np.arange(3.0),
            optics_group_ids_k=np.array([0, 1, 0]), noise_variance_k=None,
            noise_radial_k=np.ones((2, 17)) * REF_BOX**4, cs_for_engine=None, class_batch_overrides=overrides,
        ),
    )
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
    from relax.refinement import iteration_loop

    half = _half()
    previous = np.array([[1.4, -2.6], [2.5, 1.1], [-0.6, 3.3], [0.2, 0.2], [4.4, -4.4]])
    grid = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    out = iteration_loop._class_translation_kwargs(
        half, previous, sigma_offset_angstrom=10.0, base_translations=grid, current_translations=grid,
        with_log_prior=True, zero_cold_center=True,
    )["class_translation_overrides"]
    for shape_class, values in zip(half.classes, out):
        in_class = previous[shape_class.image_indices] * shape_class.translation_factor
        expected = np.sign(in_class) * np.floor(np.abs(in_class) + 0.5)
        assert_matches(values["translation_search_base"], expected.astype(np.float32))
        assert values["translation_log_prior"].shape == (shape_class.image_indices.size, 3)
    assert iteration_loop._class_translation_kwargs(
        object(), previous, sigma_offset_angstrom=1.0, base_translations=grid, current_translations=grid,
        with_log_prior=False, zero_cold_center=False,
    ) == {}


@pytest.mark.unit
def test_class_translation_step_is_in_class_pixels():
    # The adaptive pass 2 builds its oversampled translations from state.translation_step; a class
    # on another pixel size samples the same Angstrom offsets, so its step is in its pixels.
    import dataclasses

    @dataclasses.dataclass
    class State:
        translation_step: float
        adaptive_oversampling: int

    half = _half()
    kwargs = dict(experiment_dataset=half, state=State(2.0, 1))
    assert optics_shapes.class_kwargs(kwargs, half.classes[1], 5)["state"].translation_step == pytest.approx(2.0 * 0.75)
    assert optics_shapes.class_kwargs(kwargs, half.classes[0], 5)["state"] is kwargs["state"]


@pytest.mark.unit
def test_class_mstep_image_radius_is_reference_r_max_times_scale():
    from relax.refinement.half_scoring import _reconstruction_image_radius

    # RELION's backprojector bounds the reference-grid radius by r_max = cs // 2; a class
    # image pixel |k| lands at |k| / s, so the class keeps pixels out to r_max * s.
    shape_class = _half().classes[1]
    radius = _reconstruction_image_radius(56, shape_class.scale)
    assert radius == pytest.approx(28 * shape_class.scale)
    assert radius != 28.0
    assert _reconstruction_image_radius(None, shape_class.scale) is None


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
