"""Subtomogram ``--firstiter_cc`` from one reference per class (relax#75).

RELION restricts its normalized-CC iteration to the first reference only when it generates seeds
(``do_generate_seeds``: acc_ml_optimiser_impl.h:3696-3707, ml_model.cpp:1008-1010). With a reference per class it
scores every class and takes the device arg-min of the float32 ``diff2 = -CC`` over one class-major weight array
(acc_ml_optimiser_impl.h:1384-1392 and 2012-2026; cuda_utils_cub.cuh:66-84): a particle goes to the class of its
best coarse sample and an exact tie to the first class. relax composes that iteration from the K=1 tilt pass; these
tests cover the class choice, the particle subset and the merge of the per-class passes. The scored iteration
itself is compared with RELION on a GPU (the evidence of relax#75).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
from helpers.float_compare import assert_matches

from relax.helpers.types import SparsePass2Output, make_noise_stats, make_relion_stats
from relax.refinement import tomo_half, tomo_scoring
from relax.refinement.tomo_half import TomoHalf, TomoSampling
from relax.refinement.tomo_scoring import TomoScoreResult, classes_of_best_coarse_cc

pytestmark = pytest.mark.unit


def test_a_particle_goes_to_the_class_of_its_best_coarse_cc_and_a_tie_to_the_first_class():
    best_cc = np.asarray(
        [
            [0.30, 0.20, 0.25, 0.10],
            [0.10, 0.20, 0.26, 0.10],
            [0.30, 0.19, 0.26, 0.05],
        ]
    )
    # Particle 0: classes 0 and 2 tie -> 0; particle 1: classes 0 and 1 tie -> 0; particle 2: classes 1 and 2
    # tie -> 1; particle 3: classes 0 and 1 tie -> 0.
    assert classes_of_best_coarse_cc(best_cc).tolist() == [0, 0, 1, 0]


def test_the_class_tie_is_decided_in_float32_as_relion_compares_its_scores():
    cc = np.float64(np.float32(0.2))
    # Class 1 is larger in float64 by less than half a float32 step: equal in RELION's XFLOAT, so class 0 keeps it.
    best_cc = np.asarray([[cc], [cc + 0.25 * np.spacing(np.float32(0.2))]], dtype=np.float64)
    assert best_cc[1, 0] > best_cc[0, 0]
    assert classes_of_best_coarse_cc(best_cc).tolist() == [0]


class _Images:
    """A dataset stand-in: ``subset`` keeps the image rows it was asked for."""

    def __init__(self, rows):
        self.rows = np.asarray(rows, dtype=np.int64)
        self.n_units = int(self.rows.size)

    def subset(self, images):
        return _Images(self.rows[np.asarray(images, dtype=np.int64)])


def _half(tilts_per_unit=(2, 3, 1, 2)):
    offsets = np.concatenate([[0], np.cumsum(tilts_per_unit)])
    n_images = int(offsets[-1])
    return TomoHalf(
        _Images(np.arange(n_images)),
        unit_image_offsets=offsets,
        image_projections=np.arange(n_images, dtype=np.float64)[:, None, None] * np.ones((1, 3, 3)),
        image_left=-np.arange(n_images, dtype=np.float64)[:, None, None] * np.ones((1, 3, 3)),
        unit_optics_group=np.arange(len(tilts_per_unit)) % 2,
        rows=10 + np.arange(len(tilts_per_unit)),
        image_shape=(8, 8),
        volume_shape=(8, 8, 8),
        voxel_size=2.0,
        image_frames=np.arange(n_images) % 3,
        unit_tomogram=np.asarray([f"t{u}" for u in range(len(tilts_per_unit))]),
    )


def test_a_tomo_half_subset_keeps_each_particle_with_its_own_tilt_images_in_the_order_asked():
    half = _half()
    sub = half.subset([3, 1])
    assert sub.n_units == 2 and sub.n_images == 5
    assert sub.unit_image_offsets.tolist() == [0, 2, 5]
    # Particle 3's images are rows 6, 7 and particle 1's rows 2, 3, 4 of the half.
    assert sub.images.rows.tolist() == [6, 7, 2, 3, 4]
    assert sub.image_projections[:, 0, 0].tolist() == [6.0, 7.0, 2.0, 3.0, 4.0]
    assert sub.image_left[:, 0, 0].tolist() == [-6.0, -7.0, -2.0, -3.0, -4.0]
    assert sub.image_frames.tolist() == (np.asarray([6, 7, 2, 3, 4]) % 3).tolist()
    assert sub.unit_optics_group.tolist() == [1, 1]
    assert sub.unit_tomogram.tolist() == ["t3", "t1"]
    assert sub._index_layout.rows.tolist() == [13, 11]
    assert (sub.image_shape, sub.volume_shape, sub.voxel_size) == (half.image_shape, half.volume_shape, half.voxel_size)


N_ROT, N_SHELLS = 6, 4
SAMPLING = TomoSampling(
    healpix_order=1, oversampling_order=1, offset_range_angst=4.0, offset_step_angst=2.0, random_perturbation=0.0,
    coarse_size=8, fine_size=8,
)


def _class_result(seed, n, n_fine_trans):
    """A K=1 tilt pass over ``n`` particles, with recognisable values."""

    rng = np.random.default_rng(seed)
    rotation_ids = rng.integers(0, N_ROT, n)
    return TomoScoreResult(
        pass2=SparsePass2Output(
            Ft_y=np.full(5, complex(seed, -seed), dtype=np.complex64),
            Ft_ctf=np.full(5, float(seed), dtype=np.float32),
            hard_assignment=rng.integers(0, n_fine_trans, n),
            best_rotations=rng.normal(size=(n, 3, 3)).astype(np.float32),
            best_translations=rng.normal(size=(n, 3)).astype(np.float32),
            best_rotation_indices=rotation_ids,
            relion_stats=make_relion_stats(
                log_evidence_per_image=rng.normal(size=n), best_log_score_per_image=rng.normal(size=n),
                max_posterior_per_image=np.ones(n), rotation_posterior_sums=rng.random(N_ROT), host_arrays=True,
            ),
            noise_stats=make_noise_stats(
                wsum_sigma2_noise=rng.random(N_SHELLS), wsum_img_power=rng.random(N_SHELLS),
                wsum_sigma2_offset=float(rng.random()), sumw=float(n), host_arrays=True,
            ),
            source_eulers=rng.normal(size=(n, 3)),
        ),
        coarse_hard_assignment=rng.integers(0, 50, n).astype(np.int32),
        significant_counts=np.ones(n, dtype=np.int32),
    )


def test_per_class_passes_merge_into_the_k_class_pass_with_each_particle_in_its_own_class_only():
    _, _, fine_px, _ = tomo_half.tomo_translation_grids(SAMPLING, 2.0)
    n_fine_trans = int(fine_px.shape[0])
    members = [np.asarray([4, 0, 2]), np.zeros(0, dtype=np.int64), np.asarray([1, 3])]
    results = [_class_result(1, 3, n_fine_trans), None, _class_result(3, 2, n_fine_trans)]
    merged = tomo_scoring.winner_class_tomo_result(results, members, 5, SAMPLING, 2.0)
    out = merged.pass2
    unit_class = np.asarray([0, 2, 0, 2, 0])

    for k, result in enumerate(results):
        mine = members[k]
        others = np.setdiff1d(np.arange(5), mine)
        assert np.all(np.isneginf(out.class_log_evidence_per_image[k, others]))
        assert np.all(np.isneginf(out.class_best_log_score_per_image[k, others]))
        assert np.all(out.per_class_hard_assignments[k, others] == -1)
        if result is None:
            assert not np.any(np.asarray(out.Ft_y[k])) and not np.any(np.asarray(out.Ft_ctf[k]))
            assert out.class_reconstruction_posterior_sums[k] == 0.0
            assert not np.any(out.class_rotation_posterior_sums[k])
            continue
        p = result.pass2
        assert_matches(np.asarray(out.Ft_y[k]), np.asarray(p.Ft_y))
        assert_matches(np.asarray(out.Ft_ctf[k]), np.asarray(p.Ft_ctf))
        assert_matches(out.class_log_evidence_per_image[k, mine], np.asarray(p.relion_stats.log_evidence_per_image))
        assert_matches(out.class_best_log_score_per_image[k, mine], np.asarray(p.relion_stats.best_log_score_per_image))
        expected_hard = np.asarray(p.best_rotation_indices) * n_fine_trans + np.asarray(p.hard_assignment) % n_fine_trans
        assert out.per_class_hard_assignments[k, mine].tolist() == expected_hard.tolist()
        assert_matches(out.per_class_best_pose_rotations[k][mine], np.asarray(p.best_rotations))
        assert_matches(out.per_class_best_pose_translations[k][mine], np.asarray(p.best_translations))
        assert out.per_class_best_pose_rotation_ids[k][mine].tolist() == np.asarray(p.best_rotation_indices).tolist()
        assert_matches(out.per_class_best_pose_eulers_deg[k][mine], np.asarray(p.source_eulers))
        assert not np.any(out.per_class_best_pose_rotations[k][others])
        assert out.class_reconstruction_posterior_sums[k] == float(mine.size)
        assert_matches(out.class_rotation_posterior_sums[k], np.asarray(p.relion_stats.rotation_posterior_sums))
        assert merged.coarse_hard_assignment[mine].tolist() == np.asarray(result.coarse_hard_assignment).tolist()

    units = np.arange(5)
    assert_matches(np.asarray(out.stats.log_evidence_per_image), out.class_log_evidence_per_image[unit_class, units])
    assert_matches(np.asarray(out.stats.max_posterior_per_image), np.ones(5))
    assert_matches(
        np.asarray(out.stats.rotation_posterior_sums),
        np.asarray(results[0].pass2.relion_stats.rotation_posterior_sums)
        + np.asarray(results[2].pass2.relion_stats.rotation_posterior_sums),
    )
    noise = [results[0].pass2.noise_stats, results[2].pass2.noise_stats]
    assert_matches(np.asarray(out.noise_stats.wsum_sigma2_noise), sum(np.asarray(n.wsum_sigma2_noise) for n in noise))
    assert_matches(np.asarray(out.noise_stats.wsum_img_power), sum(np.asarray(n.wsum_img_power) for n in noise))
    assert float(np.sum(np.asarray(out.noise_stats.sumw))) == 5.0
    assert merged.significant_counts.tolist() == [1, 1, 1, 1, 1]
