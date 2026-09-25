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

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from test_resident_pass2_driver import _driver_fixture_args, requires_resident_gpu

from relax.sparse_pass2 import resident_pass2 as rp
from relax.sparse_pass2.resident_candidates import CapacityChunk

pytestmark = pytest.mark.unit

_CLASS_PRIORS = (0.5, 0.3, 0.2)


def _chunk(row_counts_by_image_and_class, *, row_capacity, image_capacity, image_group=None):
    """Materialized-chunk stand-in: rows image-major, then class-major.

    ``image_group`` gives each image an accumulator slot group (VDAM's pseudo-halfset);
    a row's slot is ``class + K * group`` as ``materialize_chunk`` writes it.
    """

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
    n_classes = len(row_counts_by_image_and_class[0]) if row_counts_by_image_and_class else 1
    group = np.zeros(len(row_counts_by_image_and_class), dtype=np.int32) if image_group is None else image_group
    host["row_slot"] = host["row_class"].copy()
    host["row_slot"][:n_valid] += n_classes * np.asarray(group, dtype=np.int32)[np.asarray(image, dtype=np.int64)]
    chunk = CapacityChunk(
        image_start=0,
        image_stop=len(row_counts_by_image_and_class),
        row_start=0,
        row_stop=n_valid,
        row_capacity=row_capacity,
        image_capacity=image_capacity,
    )
    return host, chunk


def test_class_layout_sub_segments():
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



def test_class_layout_refuses_rows_out_of_hidden_space_order():
    host, chunk = _chunk([(1, 1)], row_capacity=4, image_capacity=2)
    host["row_class"][:2] = [1, 0]
    with pytest.raises(ValueError, match="image-major, then class-major"):
        rp._chunk_class_layout(host, chunk, n_classes=2, n_fine_trans=3, place=rp._PLACE_ON_DEVICE)


def _block_inputs(host, chunk, live_cells, *, n_classes, n_slots=None, n_fine_trans=3):
    """``_make_mstep_block_inputs`` on a stand-in chunk whose listed rows carry weight."""

    n_slots = n_classes if n_slots is None else n_slots
    row_capacity = int(chunk.row_capacity)
    n_valid = int(chunk.row_stop)
    posterior = np.zeros((row_capacity, n_fine_trans), dtype=np.float32)
    for row, cell in live_cells:
        posterior[row, cell] = 0.25
    classes = (
        None
        if n_classes == 1
        else rp._chunk_class_layout(host, chunk, n_classes=n_classes, n_fine_trans=n_fine_trans, place=rp._PLACE_ON_DEVICE)
    )
    rows = rp._ChunkRowArrays(
        row_image_local=jnp.asarray(host["row_image_local"], dtype=jnp.int32),
        row_fine_rot=jnp.arange(row_capacity, dtype=jnp.int32) + 100,
        row_log_prior=None,
        row_mask_bits=None,
        row_mask_mode=None,
        image_ids=None,
        n_valid_rows=jnp.int32(n_valid),
        n_valid_images=None,
        segment_offsets=None,
        image_row_start=None,
        image_row_count=None,
        classes=classes,
        mstep=None if n_slots == 1 else rp._chunk_mstep_layout(host, place=rp._PLACE_ON_DEVICE),
    )
    row_is_valid = np.arange(row_capacity) < n_valid
    posterior_out = rp._ChunkPosterior(
        row_posterior=jnp.asarray(posterior),
        min_diff2=None,
        class_log_z=None,
        best_log_score=None,
        best_cell_index=None,
        max_posterior=None,
        kernel_row_image_ids=jnp.asarray(np.where(row_is_valid, host["row_image_local"], -1), dtype=jnp.int32),
        row_is_valid=jnp.asarray(row_is_valid),
    )
    return rp._make_mstep_block_inputs(rows, posterior_out, n_slots=n_slots)


def test_mstep_block_inputs_keep_only_live_rows_class_major():
    """Live rows of each class come first, in chunk order; dead and padded rows follow."""

    host, chunk = _chunk([(2, 0, 1), (0, 3, 2), (1, 1, 0)], row_capacity=16, image_capacity=4)
    # Rows 0-9 are valid: classes [0,0,2,1,1,1,2,2,0,1]. Rows 1, 4, 7 and 9 carry no weight.
    live = [(0, 0), (2, 1), (3, 2), (5, 0), (5, 1), (6, 2), (8, 0)]
    blocks = _block_inputs(host, chunk, live, n_classes=3)
    order = np.asarray(blocks.row_fine_rot) - 100
    assert_matches(np.asarray(blocks.slot_offsets), [0, 2, 4, 6])
    assert order[:6].tolist() == [0, 8, 3, 5, 2, 6]
    assert sorted(order[6:].tolist()) == sorted(set(range(16)) - {0, 8, 3, 5, 2, 6})
    posterior = np.asarray(blocks.row_posterior)
    assert np.all(posterior[:6].max(axis=1) > 0) and np.all(posterior[6:] == 0)
    assert_matches(np.asarray(blocks.row_image_local), host["row_image_local"][order])


def test_mstep_slots_split_each_class_by_pseudo_halfset():
    """VDAM: slot ``class + K * half`` (acc_ml_optimiser_impl.h:4800-4804); live rows keep image order."""

    counts = [(2, 0, 1), (0, 3, 2), (1, 1, 0), (2, 2, 2)]
    half = np.array([1, 0, 1, 0], dtype=np.int32)
    host, chunk = _chunk(counts, row_capacity=24, image_capacity=4, image_group=half)
    n_valid = int(np.sum(counts))
    # Every valid row but row 1 carries weight.
    live = [(row, 0) for row in range(n_valid) if row != 1]
    blocks = _block_inputs(host, chunk, live, n_classes=3, n_slots=6)
    order = np.asarray(blocks.row_fine_rot) - 100
    offsets = np.asarray(blocks.slot_offsets)
    row_image = np.asarray(host["row_image_local"])
    row_class = np.asarray(host["row_class"])
    per_slot = np.bincount(row_class[:n_valid] + 3 * half[row_image[:n_valid]], minlength=6)
    per_slot[row_class[1] + 3 * half[row_image[1]]] -= 1
    assert_matches(offsets, np.concatenate([[0], np.cumsum(per_slot)]))
    for a in range(6):
        rows = order[int(offsets[a]) : int(offsets[a + 1])]
        assert np.all(row_class[rows] == a % 3) and np.all(half[row_image[rows]] == a // 3)
        assert np.all(np.diff(rows) > 0)
    assert sorted(order[int(offsets[-1]) :].tolist()) == [1] + list(range(n_valid, 24))


def test_mstep_block_inputs_one_class_is_its_live_rows():
    host, chunk = _chunk([(3,), (2,)], row_capacity=8, image_capacity=4)
    blocks = _block_inputs(host, chunk, [(1, 2), (4, 0)], n_classes=1)
    assert_matches(np.asarray(blocks.slot_offsets), [0, 2])
    assert (np.asarray(blocks.row_fine_rot)[:2] - 100).tolist() == [1, 4]


def test_slot_mstep_blocks_cover_each_slot_once():
    """Every slot's live rows lie in its blocks; a boundary block is shared, not skipped."""

    spec = SimpleNamespace(mstep_block_rows=4, static_block_trip=False, row_capacity=32, n_slots=2)
    blocks = rp._MstepBlockInputs(None, None, None, None, None, slot_offsets=jnp.asarray([0, 9, 20], jnp.int32))
    covered = []
    for k in range(2):
        class_blocks, first, n_blocks = rp._slot_mstep_blocks(blocks, k, spec=spec)
        lo, hi = (int(v) for v in np.asarray(class_blocks.class_row_range))
        starts = [(int(first) + i) * 4 for i in range(int(n_blocks))]
        covered.append([r for s in starts for r in range(s, s + 4) if lo <= r < hi])
    assert covered[0] == list(range(0, 9)) and covered[1] == list(range(9, 20))


def test_live_block_spec_picks_the_smallest_block_holding_the_live_rows():
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Spec:
        mstep_block_rows: int

    spec = Spec(2048)
    assert rp._live_block_spec(spec, 0).mstep_block_rows == 32
    assert rp._live_block_spec(spec, 32).mstep_block_rows == 32
    assert rp._live_block_spec(spec, 33).mstep_block_rows == 256
    assert rp._live_block_spec(spec, 257).mstep_block_rows == 2048
    assert rp._live_block_spec(spec, 5000) is spec
    assert rp._live_block_spec(Spec(128), 3).mstep_block_rows == 16


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
    # Each class masks its scale sums with its own data_vs_prior_class > 3
    # (acc_ml_optimiser_impl.h:4908): class k keeps shells below n_shells - 1 - k.
    dvp = np.asarray(args["scale_correction_data_vs_prior"], dtype=np.float64)
    per_class = np.stack([dvp] * n_classes)
    for k in range(n_classes):
        per_class[k, dvp.size - 1 - k :] = 1.0
    args["scale_correction_data_vs_prior"] = per_class
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
    # The group scale sums take each class's own mask; both engines run the
    # atomic Wavg triplet here (_resident_production_env).
    for field in ("wsum_scale_correction_xa", "wsum_scale_correction_aa"):
        compact_sum = sum(np.asarray(getattr(stats, field), dtype=np.float64) for stats in compact.noise_stats)
        measured = _rel_l2(compact_sum, getattr(resident.noise_stats, field))
        print(f"K={n_classes} {field} rel L2 vs compact {measured:.3e}")
        assert measured < 1e-6, field
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
        # The class prior log 1/2 is folded into the float32 row prior, so the
        # scores it shifts round in float32: compare at float32 (the K=1 pass's
        # own score precision), not at the float64 of the log-sum-exp.
        assert_matches(
            np.float32(doubled.class_log_evidence_per_image[k]),
            np.float32(np.asarray(single.relion_stats.log_evidence_per_image, dtype=np.float64) + np.log(0.5)),
        )
        assert _rel_l2(0.5 * np.asarray(single.Ft_y), doubled.Ft_y[k]) < 1e-6, f"Ft_y class {k}"
        assert _rel_l2(0.5 * np.asarray(single.Ft_ctf), doubled.Ft_ctf[k]) < 1e-6, f"Ft_ctf class {k}"
    # Sums of float32 posteriors whose inputs differ by that float32 rounding.
    assert_matches(
        np.float32(doubled.class_reconstruction_posterior_sums),
        np.float32(np.full(2, 0.5 * float(single.noise_stats.sumw))),
    )
    # Pmax is RELION's float32 max weight over its float32 sum of every cell's
    # weight (sparse_pass2_segmented_posterior_f32); the duplicated pass sums twice
    # as many cells in another order, which moved 2 of 12 images by 5.2e-5
    # relative (14424611). The bound is that measurement with a 2x margin.
    assert_matches(
        np.asarray(doubled.stats.max_posterior_per_image),
        0.5 * np.asarray(single.relion_stats.max_posterior_per_image),
        rtol=1e-4,
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


# ---------------------------------------------------------------------------
# Engine selection: the K=1 flip's rules for the K-class pass
# ---------------------------------------------------------------------------


def _selection_run(monkeypatch, env_value, outcome):
    """Run the K-class route with a stub resident driver; return (result, engine entries)."""

    from relax.classification import k_class
    from relax.sparse_pass2 import engine_record
    from relax.sparse_pass2.sparse_pass2_policy import ResidentConfigurationUnsupported

    if env_value is None:
        monkeypatch.delenv(rp.RESIDENT_PASS2_ENV, raising=False)
    else:
        monkeypatch.setenv(rp.RESIDENT_PASS2_ENV, env_value)
    calls = []

    def stub(*args, **kwargs):
        calls.append(kwargs)
        if outcome == "refuse":
            raise ResidentConfigurationUnsupported(
                f"The device-resident K=1 sparse pass 2 ({rp.RESIDENT_PASS2_ENV}=1) does not implement "
                "this configuration: a stub refusal. Clear the flag to use the compact engine; this path "
                "never falls back silently."
            )
        return "resident-output"

    monkeypatch.setattr(rp, "compute_k_class_pass2_stats_resident", stub)
    monkeypatch.setattr(k_class, "_class_segmented_em_result", lambda output, **_: output)
    engine_record.take_pass_engines()
    result = k_class._run_resident_k_class_pass2(
        SimpleNamespace(n_units=3),
        np.zeros((2, 4, 4, 4), dtype=np.float32),
        None,
        None,
        [None, None],
        common={"nside_level": 1, "disc_type": "linear_interp", "relion_x_half_mstep": True},
        engine_kwargs={},
        class_rotation_priors=[None, None],
        relion_projector_half_by_class=None,
        relion_projector_r_max=None,
        accumulate_noise=True,
        mstep_accumulator_shape=None,
    )
    return result, engine_record.take_pass_engines(), calls


def test_k_class_default_falls_back_on_a_refusal_and_records_it(monkeypatch):
    result, entries, calls = _selection_run(monkeypatch, None, "refuse")
    assert result is None and len(calls) == 1
    assert entries == ["global:compact (a stub refusal)"]


def test_k_class_explicit_resident_makes_a_refusal_an_error(monkeypatch):
    from relax.sparse_pass2.sparse_pass2_policy import ResidentConfigurationUnsupported

    with pytest.raises(ResidentConfigurationUnsupported, match="a stub refusal"):
        _selection_run(monkeypatch, "1", "refuse")


def test_k_class_default_runs_resident_with_the_production_arithmetic(monkeypatch):
    result, entries, calls = _selection_run(monkeypatch, None, "run")
    assert result == "resident-output" and entries == ["global:resident"]
    (kwargs,) = calls
    for name in ("source_faithful_spectrum_norm", "preserve_bpref_particle_order", "relion_f32_fine_posterior"):
        assert kwargs[name] is True, name


def test_k_class_resident_off_records_the_compact_engine(monkeypatch):
    result, entries, calls = _selection_run(monkeypatch, "0", "run")
    assert result is None and not calls
    assert entries == [f"global:compact ({rp.RESIDENT_PASS2_ENV}=0)"]


def test_fold_class_scale_sums_masks_each_class_and_clears_its_channels():
    """RELION adds a class's XA/AA to the particle's scale sums under that class's mask only."""

    rng = np.random.default_rng(3)
    triplet = jnp.asarray(rng.normal(size=(3, 5, 3)), dtype=jnp.float32)
    carry = rp._ChunkMstepCarry(
        Ft_y=None,
        Ft_ctf=None,
        wavg_triplet_pixels=triplet,
        noise_shells=jnp.zeros(4, dtype=jnp.float64),
        a2_per_image=jnp.zeros(3),
        xa_per_image=jnp.zeros(3),
        scale_xa_per_image=jnp.ones(3, dtype=jnp.float64),
        scale_aa_per_image=jnp.zeros(3, dtype=jnp.float64),
    )
    mask = np.asarray([True, False, True, True, False])
    folded = rp._fold_class_scale_sums(carry, jnp.asarray(mask))
    host = np.asarray(triplet, dtype=np.float64)
    assert_matches(np.asarray(folded.scale_xa_per_image), 1.0 + host[:, mask, 0].sum(axis=1))
    assert_matches(np.asarray(folded.scale_aa_per_image), host[:, mask, 1].sum(axis=1))
    assert not np.asarray(folded.wavg_triplet_pixels[:, :, :2]).any()
    assert_matches(np.asarray(folded.wavg_triplet_pixels[:, :, 2]), np.asarray(triplet[:, :, 2]))
