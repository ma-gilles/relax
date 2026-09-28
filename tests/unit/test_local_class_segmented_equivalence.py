"""One segmented pass must reproduce the per-class calls it replaces.

The K-class local route runs a probe pass and an M-step pass per class. With the
classes laid out as segments of one bucket's row axis the engine scores the joint
class-by-pose posterior once. These tests hold that one pass to what the per-class
calls produce on the same inputs.

The fixture has to keep every class materially populated, or the comparison is
vacuous: with a peaked posterior one class takes all the mass and its rivals'
accumulators and angular sums are exactly zero, so comparing them proves nothing.
An earlier version of these tests had exactly that defect. The noise level below is
chosen so both classes keep real responsibility (minimum about 0.08 per image, class
masses about 2.5 and 3.5 of 6 images), the rows per class are unequal so the segment
padding is exercised, the translations are nontrivial, and every row carries
canonical source Euler metadata.

Tolerances come from the measured residuals rather than convention: they sit at the
reduction-order floor of each precision, about 1e-7 relative in float32 and about
3e-16 in float64, so the comparison would fail on a real behavior change.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from helpers.em_arrays import _hermitian_volume, _make_rotations
from relax.local.local_layout import LocalHypothesisLayout
from test_refine_relion_mode import IMAGE_SIZE, VOLUME_SHAPE
from helpers.float_compare import assert_matches

pytestmark = pytest.mark.unit

N_IMAGES = 6
# Both classes keep real posterior mass at this noise level; see the module docstring.
NOISE_VARIANCE = 1.0e3
TRANSLATIONS = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
CANONICAL_EULERS = np.asarray([360.0, 0.0, 0.0], dtype=np.float64)
# Enough candidate rows that the significant-only reconstruction route is actually
# selected: it needs local support of at least n_images / 0.25 rows. With the earlier
# 12 rows the engine took the dense in-bucket adjoint instead, so the tests never
# reached the route that had the class-routing bug.
ROWS_BY_CLASS = ([7, 5, 6, 6, 5, 7], [5, 7, 6, 6, 7, 5])


def _layout(counts, *, seed, n_global=4):
    counts = np.asarray(counts, dtype=np.int32)
    total = int(counts.sum())
    n_trans = TRANSLATIONS.shape[0]
    ids = np.concatenate([np.arange(count) % n_global for count in counts]).astype(np.int32)
    # RELION's canonical source angles for these rows; the engine must publish these
    # values, not angles recovered from the rotation matrices.
    source_eulers = np.tile(CANONICAL_EULERS, (total, 1))
    source_eulers[:, 2] = np.arange(total, dtype=np.float64)
    return LocalHypothesisLayout(
        n_global_rotations=n_global,
        n_pixels=n_global,
        n_psi=1,
        rotation_offsets=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        rotation_ids_flat=ids,
        rotations_flat=np.asarray(_make_rotations(total, seed=seed), dtype=np.float32),
        rotation_log_priors_flat=np.zeros(total, dtype=np.float32),
        rotation_counts=counts,
        translation_grid=TRANSLATIONS,
        translation_log_priors=np.zeros((N_IMAGES, n_trans), dtype=np.float32),
        rotation_posterior_ids_flat=ids,
        sample_mask_bits=np.packbits(
            np.ones((total, n_trans), dtype=bool), axis=1, bitorder="little"
        ),
        source_eulers_flat=source_eulers,
    )


def _fixture(float64: bool, n_classes=2):
    base = np.asarray(_hermitian_volume(VOLUME_SHAPE, seed=211))
    perturbation = np.asarray(_hermitian_volume(VOLUME_SHAPE, seed=307))
    means = jnp.stack([jnp.asarray(base + 0.05 * k * perturbation) for k in range(n_classes)], axis=0)
    if float64:
        means = means.astype(jnp.complex128)
    noise = jnp.full(IMAGE_SIZE, NOISE_VARIANCE, dtype=jnp.float64 if float64 else jnp.float32)
    layouts = [_layout(ROWS_BY_CLASS[k % 2], seed=17 + 12 * k) for k in range(n_classes)]
    return means, noise, layouts


def _assert_published_dtypes_match(segmented, baseline, expected_real, expected_normalization_real):
    """The result contract includes its precision; a joint pass must not promote it.

    The per-class route hands each M-step call the joint normalizer, which the engine
    casts to the scoring dtype, so its posteriors, accumulators and statistics stay at
    that precision. A pass that normalizes jointly computes the normalizer in the
    accumulation dtype and would publish promoted arrays unless it applies the same
    boundary.
    """
    expected_complex = np.complex64 if expected_real is np.float32 else np.complex128
    for name in ("Ft_y", "Ft_ctf"):
        assert np.asarray(getattr(segmented, name)).dtype == expected_complex, name
        assert np.asarray(getattr(segmented, name)).dtype == np.asarray(getattr(baseline, name)).dtype, name
    for name in ("class_responsibilities", "class_posterior_sums"):
        assert np.asarray(getattr(segmented, name)).dtype == expected_real, name
        assert np.asarray(getattr(segmented, name)).dtype == np.asarray(getattr(baseline, name)).dtype, name
    # The joint statistics keep the published contract: _assemble_result still calls
    # make_relion_stats with image_dtype=output_dtype, so all three stay at the
    # scoring dtype and that assertion is independent of anything below.
    #
    # Per class, log_evidence_per_image is a normalizer and follows the normalization
    # policy, because the value is formed in double and a scoring-dtype buffer
    # narrowed it per class before the driver reduced over classes. The winning score
    # and the maximum posterior are scores and stay at the scoring dtype. Both arms
    # must agree with each other on every field either way.
    for stats_name in ("log_evidence_per_image", "best_log_score_per_image", "max_posterior_per_image"):
        got = np.asarray(getattr(segmented.stats, stats_name)).dtype
        assert got == expected_real, (stats_name, got)
        assert got == np.asarray(getattr(baseline.stats, stats_name)).dtype, stats_name
        expected_field = expected_normalization_real if stats_name == "log_evidence_per_image" else expected_real
        for class_index in range(len(segmented.per_class_stats)):
            per_class = np.asarray(getattr(segmented.per_class_stats[class_index], stats_name)).dtype
            assert per_class == np.asarray(getattr(baseline.per_class_stats[class_index], stats_name)).dtype
            assert per_class == expected_field, (stats_name, class_index, per_class)
    for class_index in range(len(segmented.per_class_stats)):
        assert (
            np.asarray(segmented.per_class_stats[class_index].rotation_posterior_sums).dtype
            == np.asarray(baseline.per_class_stats[class_index].rotation_posterior_sums).dtype
        )
    for field in ("wsum_sigma2_noise", "wsum_img_power"):
        got = np.asarray(getattr(segmented.aggregate_noise_stats, field)).dtype
        assert got == expected_real, (field, got)
        assert got == np.asarray(getattr(baseline.aggregate_noise_stats, field)).dtype, field


def _assert_sparse_reconstruction_route_was_used(result):
    """Fail loudly if the fixture silently took the dense adjoint instead.

    The significant-only route is the one that packs surviving rows and scatters them
    outside the bucket program; it is selected only when the local support is large
    enough. A fixture below that size tests the other route and proves nothing about
    this one.
    """
    profile = result.profile_summary or {}
    chunks = int(profile.get("sparse_adjoint_chunk_count", 0))
    assert chunks > 0, f"the significant-only reconstruction route was not exercised: {chunks} sparse chunks"


def _assert_every_class_is_populated(result):
    """Guard the comparison itself: a collapsed class would make it vacuous."""
    responsibilities = np.asarray(result.class_responsibilities, dtype=np.float64)
    masses = np.asarray(result.class_posterior_sums, dtype=np.float64)
    assert responsibilities.shape == (len(result.per_class_stats), N_IMAGES)
    assert masses.min() > 0.5 * N_IMAGES / 10.0, masses
    assert responsibilities.min() > 1e-3, responsibilities
    for class_index in range(len(result.per_class_stats)):
        for name in ("Ft_y", "Ft_ctf"):
            values = np.abs(np.asarray(getattr(result, name)[class_index]))
            assert values.max() > 0.0 and np.count_nonzero(values) > values.size // 10, (name, class_index)
        angular = np.asarray(result.per_class_stats[class_index].rotation_posterior_sums, dtype=np.float64)
        assert angular.sum() > 0.1, (class_index, angular)


def test_class_packs_contain_only_their_own_rows():
    """Row identities, not aggregate weights: each class's pack is exactly its segment.

    The significant-only route packs surviving rows and scatters them outside the
    bucket program; with one joint pack every class landed in volume 0. Assert the
    property that failed, exactly and without tolerances: every packed row index of
    class k lies in class k's segment, the packs are disjoint, and together they hold
    precisely the rows a single joint pack would have selected.
    """
    from relax.local.local_bucket_stages import (
        _build_nonzero_reconstruction_pack_indices,
        build_class_segment_reconstruction_packs,
    )

    rng = np.random.default_rng(131)
    n_images, n_classes, seg, n_trans = 4, 3, 5, 2
    rows = n_classes * seg
    local_mask = rng.random((n_images, rows)) > 0.25
    significant = rng.random((n_images, rows)) > 0.4
    probs_sum_t = np.where(rng.random((n_images, rows)) > 0.3, rng.random((n_images, rows)), 0.0)

    packs = build_class_segment_reconstruction_packs(
        significant, local_mask, probs_sum_t, rotation_block_size=16,
        n_classes=n_classes, segment_rotation_count=seg,
    )
    assert len(packs) == n_classes
    selected_by_class = []
    for class_index, (take, mask, counts, row_count) in enumerate(packs):
        start_row, stop_row = class_index * seg, (class_index + 1) * seg
        chosen = take[mask]
        assert chosen.size == int(np.asarray(counts).sum()) == row_count
        # Every packed row belongs to this class's segment, exactly.
        assert chosen.size == 0 or (chosen.min() >= start_row and chosen.max() < stop_row), (class_index, chosen)
        # Per image, the packed rows are the surviving rows of this segment.
        for image in range(n_images):
            expected = np.flatnonzero(
                significant[image, start_row:stop_row] & local_mask[image, start_row:stop_row]
                & (probs_sum_t[image, start_row:stop_row] > 0.0)
            ) + start_row
            assert_matches(np.sort(take[image][mask[image]]), expected)
        selected_by_class.append(set(chosen.tolist()))

    # Disjoint, and together exactly the joint selection.
    for a in range(n_classes):
        for b in range(a + 1, n_classes):
            assert not (selected_by_class[a] & selected_by_class[b])
    joint_take, joint_mask, _counts, _rows = _build_nonzero_reconstruction_pack_indices(
        significant, local_mask, probs_sum_t, rotation_block_size=16,
    )
    joint_rows = {
        (image, int(row))
        for image in range(n_images)
        for row in joint_take[image][joint_mask[image]]
    }
    class_rows = {
        (image, int(row))
        for class_index, (take, mask, _c, _r) in enumerate(packs)
        for image in range(n_images)
        for row in take[image][mask[image]]
    }
    assert class_rows == joint_rows


