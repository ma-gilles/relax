"""K-class resident pass 2 (RELION Class3D) against the compact fused K-class engine.

Rows are image-major, then class-major (docs/development/resident_segments.md), so
one image's posterior segment spans every class: the minimum, the normalization
and the significance are RELION's joint ones (ml_optimiser.cpp:8411, :9225,
:9602-9660), each class backprojects into its own BPref (:10826) and the noise
is one total over classes (:10470, :11010). The compact engine computes the same
quantities per bucket; discrete state must match exactly and scores in the
default band, while maps and noise sums change reduction order.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from test_resident_pass2_driver import _driver_fixture_args, requires_resident_gpu

from relax.sparse_pass2 import resident_pass2 as rp
from relax.sparse_pass2.resident_candidates import CapacityChunk

pytestmark = pytest.mark.unit

_CLASS_PRIORS = (0.5, 0.3, 0.2)


def _chunk(row_counts_by_image_and_class, *, row_capacity, image_capacity):
    """Materialized-chunk stand-in: rows image-major, then class-major."""

    image, klass = [], []
    for b, counts in enumerate(row_counts_by_image_and_class):
        for k, count in enumerate(counts):
            image += [b] * count
            klass += [k] * count
    n_valid = len(image)
    host = {
        "row_image_local": np.full(row_capacity, image_capacity - 1, dtype=np.int32),
        "row_class": np.zeros(row_capacity, dtype=np.int32),
    }
    host["row_image_local"][:n_valid] = image
    host["row_class"][:n_valid] = klass
    chunk = CapacityChunk(
        image_start=0,
        image_stop=len(row_counts_by_image_and_class),
        row_start=0,
        row_stop=n_valid,
        row_capacity=row_capacity,
        image_capacity=image_capacity,
    )
    return host, chunk


def test_class_layout_sub_segments_and_mstep_order():
    counts = [(2, 0, 1), (0, 3, 2), (1, 1, 0)]
    host, chunk = _chunk(counts, row_capacity=16, image_capacity=4)
    layout = rp._chunk_class_layout(host, chunk, n_classes=3, n_fine_trans=5, place=rp._PLACE_ON_DEVICE)

    flat = np.zeros(4 * 3, dtype=np.int64)
    flat[:9] = np.asarray(counts).reshape(-1)
    expected_rows = np.concatenate([[0], np.cumsum(flat)])
    assert_matches(np.asarray(layout.segment_offsets), expected_rows * 5)
    assert_matches(np.asarray(layout.segment_row_start), expected_rows[:-1])
    n_valid = int(flat.sum())
    segment = np.asarray(layout.row_segment)
    assert_matches(segment[:n_valid], np.repeat(np.arange(12), flat))
    assert np.all(segment[n_valid:] == 12)

    # The M-step visits class 0's rows, then class 1's, then class 2's, each in
    # image order; padded rows stay last.
    order = np.asarray(layout.mstep_row_order)
    row_class = np.asarray(host["row_class"])
    assert_matches(row_class[order[:n_valid]], np.sort(row_class[:n_valid], kind="stable"))
    assert_matches(order[n_valid:], np.arange(n_valid, 16))
    assert_matches(np.asarray(layout.mstep_class_offsets), [0, 3, 7, 10])
    for k in range(3):
        rows = order[int(layout.mstep_class_offsets[k]) : int(layout.mstep_class_offsets[k + 1])]
        assert np.all(np.diff(host["row_image_local"][rows]) >= 0)


def test_class_layout_refuses_rows_out_of_hidden_space_order():
    host, chunk = _chunk([(1, 1)], row_capacity=4, image_capacity=2)
    host["row_class"][:2] = [1, 0]
    with pytest.raises(ValueError, match="image-major, then class-major"):
        rp._chunk_class_layout(host, chunk, n_classes=2, n_fine_trans=3, place=rp._PLACE_ON_DEVICE)


def test_class_mstep_blocks_cover_each_class_once():
    """Every class's rows lie in its blocks; a boundary block is shared, not skipped."""

    host, chunk = _chunk([(5, 2), (1, 9), (3, 0)], row_capacity=32, image_capacity=4)
    rows = rp._ChunkRowArrays(
        row_image_local=None,
        row_fine_rot=None,
        row_log_prior=None,
        row_mask_bits=None,
        row_mask_mode=None,
        image_ids=None,
        n_valid_rows=jnp.int32(20),
        n_valid_images=jnp.int32(3),
        segment_offsets=None,
        image_row_start=None,
        image_row_count=None,
        classes=rp._chunk_class_layout(host, chunk, n_classes=2, n_fine_trans=1, place=rp._PLACE_ON_DEVICE),
    )
    spec = type("Spec", (), {"mstep_block_rows": 4, "static_block_trip": False, "row_capacity": 32})()
    blocks = rp._MstepBlockInputs(None, None, None, None, None)
    covered = []
    for k in range(2):
        class_blocks, first, n_blocks = rp._class_mstep_blocks(blocks, rows, k, spec=spec)
        lo, hi = (int(v) for v in np.asarray(class_blocks.class_row_range))
        starts = [(int(first) + i) * 4 for i in range(int(n_blocks))]
        covered.append([r for s in starts for r in range(s, s + 4) if lo <= r < hi])
    assert covered[0] == list(range(0, 9)) and covered[1] == list(range(9, 20))


# ---------------------------------------------------------------------------
# GPU: the whole K-class pass against the compact fused engine
# ---------------------------------------------------------------------------


def _k_class_args(n_classes, seed=20260925):
    """The K=1 driver fixture with K class references, supports and priors."""

    from helpers.em_arrays import _hermitian_volume
    from test_sparse_pass2_bucketed_parity import VOLUME_SHAPE

    args = _driver_fixture_args()
    rng = np.random.default_rng(seed)
    n_images = len(args["significant_sample_indices"])
    n_coarse_rot = int(args["rotation_log_prior"].shape[0])
    total = n_coarse_rot * int(args["translations"].shape[0])
    supports = [args.pop("significant_sample_indices")]
    for _ in range(1, n_classes):
        class_supports = [None]
        for _ in range(1, n_images):
            count = int(rng.integers(1, total))
            class_supports.append(np.sort(rng.choice(total, size=count, replace=False).astype(np.int32)))
        supports.append(class_supports)
    rotation_log_prior = args.pop("rotation_log_prior")
    priors = [(rotation_log_prior + np.float32(np.log(_CLASS_PRIORS[k]))).astype(np.float32) for k in range(n_classes)]
    args.pop("volume")
    volumes = jnp.stack([_hermitian_volume(VOLUME_SHAPE, seed=17 + 13 * k) for k in range(n_classes)])
    for name in ("normalization_other_score_log_z", "normalization_score_mode"):
        args.pop(name)
    return args, volumes, supports, priors


def _compact(args, volumes, supports, priors):
    from relax.sparse_pass2.sparse_pass2_bucketed import compute_k_class_pass2_stats_sparse_fused

    compact_args = dict(args)
    for name in ("relion_fine_mstep_prune", "preserve_bpref_particle_order", "return_score_log_z"):
        compact_args.pop(name)
    experiment_dataset = compact_args.pop("experiment_dataset")
    noise_variance = compact_args.pop("noise_variance")
    translations = compact_args.pop("translations")
    return compute_k_class_pass2_stats_sparse_fused(
        experiment_dataset,
        volumes,
        noise_variance,
        translations,
        supports,
        rotation_log_priors_by_class=priors,
        relion_fine_mstep_prune_mode="joint",
        **compact_args,
    )


def _resident(args, volumes, supports, priors):
    resident_args = dict(args)
    experiment_dataset = resident_args.pop("experiment_dataset")
    noise_variance = resident_args.pop("noise_variance")
    translations = resident_args.pop("translations")
    return rp.compute_k_class_pass2_stats_resident(
        experiment_dataset,
        volumes,
        noise_variance,
        translations,
        supports,
        resident_args.pop("nside_level"),
        resident_args.pop("disc_type"),
        rotation_log_priors_by_class=priors,
        **resident_args,
    )


def _rel_l2(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    den = float(np.linalg.norm(a))
    return float(np.linalg.norm(a - b) / den) if den else float(np.linalg.norm(b))


@pytest.fixture
def _resident_production_env(monkeypatch):
    monkeypatch.setenv("RELAX_EM_PROTOTYPE_SOFT_POSTERIOR_BLOCK_BPREF", "1")
    monkeypatch.setenv("RELAX_RELION_WAVG_ATOMIC_SCALE_AA", "1")
    monkeypatch.setenv("RELAX_RELION_WAVG_ATOMIC_DIRECT_NOISE_ONLY", "1")
    monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_ROW_CAPACITIES", "256,1024,4096")
    monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_IMAGE_CAPACITIES", "4,16,64")
    monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_MSTEP_BLOCK_ROWS", "128")


@requires_resident_gpu
@pytest.mark.parametrize("n_classes", [2, 3])
def test_k_class_resident_matches_the_compact_fused_engine(_resident_production_env, n_classes):
    args, volumes, supports, priors = _k_class_args(n_classes)
    compact = _compact(args, volumes, supports, priors)
    resident = _resident(args, volumes, supports, priors)

    compact_evidence = np.asarray(compact.class_log_evidence, dtype=np.float64)
    has_class = np.isfinite(compact_evidence)
    assert_matches(np.isfinite(resident.class_log_evidence_per_image), has_class)
    assert_matches(resident.class_log_evidence_per_image[has_class], compact_evidence[has_class])
    assert_matches(
        np.asarray(resident.per_class_hard_assignments)[has_class],
        np.asarray(compact.per_class_hard_assignments)[has_class],
    )
    for k in range(n_classes):
        class_stats = compact.per_class_stats[k]
        rows = has_class[k]
        assert_matches(
            resident.class_best_log_score_per_image[k][rows],
            np.asarray(class_stats.best_log_score_per_image, dtype=np.float64)[rows],
            err_msg=f"best score, class {k}",
        )
        assert_matches(
            resident.class_rotation_posterior_sums[k],
            np.asarray(class_stats.rotation_posterior_sums, dtype=np.float64),
            err_msg=f"rotation mass, class {k}",
        )
    # Pmax of the joint posterior is its best class's share.
    compact_pmax = np.max(np.stack([np.asarray(s.max_posterior_per_image) for s in compact.per_class_stats]), axis=0)
    assert_matches(np.asarray(resident.stats.max_posterior_per_image), compact_pmax)
    assert_matches(resident.class_reconstruction_posterior_sums, np.asarray(compact.class_posterior_sums))

    # Float32 BPref atomics and blocked pixel-axis reductions, as for K=1.
    for k in range(n_classes):
        assert _rel_l2(compact.Ft_y[k], resident.Ft_y[k]) < 1e-6, f"Ft_y class {k}"
        assert _rel_l2(compact.Ft_ctf[k], resident.Ft_ctf[k]) < 1e-6, f"Ft_ctf class {k}"

    # The noise sums are not compared: the compact K-class engine has no RELION
    # direct low-shell Wavg residual (its historical K>1 arithmetic), so they
    # differ by construction; test_duplicated_class_is_the_k1_pass checks them.
    total_sumw = float(sum(float(stats.sumw) for stats in compact.noise_stats))
    assert abs(total_sumw - float(resident.noise_stats.sumw)) <= 1e-6 * abs(total_sumw)


@requires_resident_gpu
def test_duplicated_class_is_the_k1_pass(_resident_production_env):
    """Two copies of one class at prior 1/2 each are the K=1 pass split in half.

    Every duplicated cell carries half the K=1 posterior, so the joint pruning
    keeps the same cells, each class's BPref is half of the K=1 BPref, its mass
    half of ``sumw``, its evidence ``log 1/2`` below the K=1 evidence, and every
    noise, norm and scale sum equals the K=1 one. The K=1 driver is the one
    measured against the compact engine with this arithmetic, so this pins the
    K-class noise path, which the compact K-class engine cannot.
    """

    args = _driver_fixture_args()
    single = rp.compute_pass2_stats_resident(**args)
    k_args = dict(args)
    support = k_args.pop("significant_sample_indices")
    prior = k_args.pop("rotation_log_prior")
    volume = k_args.pop("volume")
    for name in ("normalization_other_score_log_z", "normalization_score_mode"):
        k_args.pop(name)
    half = (prior + np.float32(np.log(0.5))).astype(np.float32)
    doubled = _resident(k_args, jnp.stack([volume, volume]), [support, support], [half, half])

    for k in range(2):
        assert_matches(doubled.per_class_best_pose_rotation_ids[k], single.best_rotation_indices)
        assert_matches(
            doubled.class_log_evidence_per_image[k],
            np.asarray(single.relion_stats.log_evidence_per_image, dtype=np.float64) + np.log(0.5),
        )
        assert _rel_l2(0.5 * np.asarray(single.Ft_y), doubled.Ft_y[k]) < 1e-6, f"Ft_y class {k}"
        assert _rel_l2(0.5 * np.asarray(single.Ft_ctf), doubled.Ft_ctf[k]) < 1e-6, f"Ft_ctf class {k}"
    assert_matches(doubled.class_reconstruction_posterior_sums, np.full(2, 0.5 * float(single.noise_stats.sumw)))
    assert_matches(
        np.asarray(doubled.stats.max_posterior_per_image),
        0.5 * np.asarray(single.relion_stats.max_posterior_per_image),
    )
    # The K=1 engine-comparison bounds: the Wavg residual cancels most of its
    # magnitude, so its reduction order shows at 1e-4; the other sums at 1e-6.
    for field, bound in (
        ("wsum_sigma2_noise", 1e-4),
        ("wsum_img_power", 1e-6),
        ("wsum_norm_correction", 1e-6),
        ("wsum_scale_correction_xa", 1e-6),
        ("wsum_scale_correction_aa", 1e-6),
    ):
        measured = _rel_l2(getattr(single.noise_stats, field), getattr(doubled.noise_stats, field))
        print(f"duplicated-class {field} rel L2 {measured:.3e}")
        assert measured < bound, field
    assert abs(float(single.noise_stats.sumw) - float(doubled.noise_stats.sumw)) <= 1e-6 * float(
        single.noise_stats.sumw
    )
