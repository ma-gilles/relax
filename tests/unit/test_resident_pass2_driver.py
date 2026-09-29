"""Integration tests for the device-resident K=1 sparse pass-2 driver (T9b).

Ticket: ``em_parity_tickets_20260918/T9b_resident_pass2_integration.md``.
Design: ``em_device_resident_pass2_design_20260918.md``.

What the CPU tests cover
------------------------
Every stage of the resident driver that is CUDA-only skips on CPU: T6 scoring
(the flat-row fused-translate kernel), T7's segmented posterior, T8's flat-row
Wavg reducer and the x-half backprojection. What remains CPU-testable, and is
tested here, is:

* the production-configuration gate, one refusal per unsupported knob;
* the chunk segment offsets the segmented posterior is driven with;
* the flat-row twins of the three rectangular host helpers the driver replaces
  (``compute_local_mstep_sums``, ``_relion_wavg_atomic_triplet_terms`` and
  ``_relion_wavg_rectangle_triplet_terms``).

The whole-driver comparison against ``compute_pass2_stats_sparse_bucketed``
needs a GPU and the custom CUDA library; it is the GPU test at the end.
"""

from __future__ import annotations


import numpy as np
import pytest
from helpers.float_compare import assert_matches

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from relax.local.local_backprojection import compute_local_mstep_sums
from relax.sparse_pass2 import resident_pass2 as rp
from relax.sparse_pass2.resident_candidates import (
    CapacityChunk,
    ResidentCandidateTables,
    plan_capacity_chunks,
)
from relax.sparse_pass2.sparse_pass2_policy import ResidentConfigurationUnsupported
from relax.sparse_pass2.sparse_pass2_wavg import (
    _relion_wavg_rectangle_triplet_terms,
)

pytestmark = pytest.mark.unit


def _production_gate_kwargs(**overrides):
    kwargs = dict(
        relion_x_half_mstep=True,
        relion_exact_fine_gaussian=True,
        relion_firstiter_score_mode="gaussian",
        use_float64_scoring=False,
        relion_firstiter_winner_take_all=False,
        disable_adjoint_y=False,
        disable_adjoint_ctf=False,
        return_score_log_z_only=False,
        accumulate_noise=True,
        mstep_subtract_ctf_projection=False,
        normalization_log_z=None,
        normalization_other_score_log_z=None,
        relion_f32_normalization_sum_weight=None,
        relion_coarse_hard_assignment=None,
        preserve_bpref_particle_order=False,
        soft_posterior_block_bpref=False,
        fine_rotations_override=np.zeros((2, 3, 3), dtype=np.float32),
        fine_rotation_parent_override=np.zeros(2, dtype=np.int32),
        use_window=True,
        projection_cache_available=True,
        relion_wavg_atomic_scale_aa=True,
        relion_wavg_atomic_direct_noise=True,
        relion_wavg_atomic_direct_norm=False,
        relion_projector_texture=None,
    )
    kwargs.update(overrides)
    return kwargs


def test_production_configuration_is_accepted():
    rp.require_resident_production_configuration(**_production_gate_kwargs())


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"relion_x_half_mstep": False}, "x-half M-step"),
        ({"use_float64_scoring": True}, "float64 scoring"),
        ({"relion_firstiter_winner_take_all": True}, "winner-take-all"),
        ({"disable_adjoint_y": True, "disable_adjoint_ctf": True}, "score-only"),
        ({"accumulate_noise": False}, "noise statistics"),
        ({"normalization_log_z": np.zeros(3)}, "externally supplied log-Z"),
        (
            {"normalization_other_score_log_z": np.zeros(3)},
            "finite cross-class score normalization",
        ),
        ({"relion_f32_normalization_sum_weight": np.ones(3)}, "sum, winner and Pmax"),
        ({"preserve_bpref_particle_order": True}, "per-particle BPref launches"),
        ({"fine_rotations_override": None}, "fine_rotations_override"),
        ({"use_window": False}, "Nyquist row"),
        ({"projection_cache_available": False}, "projection cache"),
        ({"relion_wavg_atomic_scale_aa": False}, "atomic Wavg triplet"),
        ({"relion_wavg_atomic_direct_noise": False}, "direct low-shell residual"),
        ({"relion_wavg_atomic_direct_norm": True}, "stopped diagnostic"),
        ({"relion_firstiter_score_mode": "normalized_cc"}, "fine Gaussian"),
        ({"relion_projector_texture": object()}, "not a caller.s persistent texture"),
    ],
)
def test_gate_names_the_missing_piece(override, expected):
    with pytest.raises(NotImplementedError, match=expected):
        rp.require_resident_production_configuration(**_production_gate_kwargs(**override))


def test_gate_accepts_production_bpref_order_with_the_block_prototype():
    """``preserve_bpref_particle_order`` is production; the block prototype makes it block-wise."""

    rp.require_resident_production_configuration(
        **_production_gate_kwargs(
            preserve_bpref_particle_order=True, soft_posterior_block_bpref=True
        )
    )


def test_gate_refuses_a_diagnostic_dump(monkeypatch):
    monkeypatch.setenv("RELAX_PASS2_DUMP_DIR", "/tmp/does-not-matter")
    with pytest.raises(NotImplementedError, match="RELAX_PASS2_DUMP_DIR"):
        rp.require_resident_production_configuration(**_production_gate_kwargs())


def _tables(row_counts):
    row_counts = np.asarray(row_counts, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(row_counts)]).astype(np.int32)
    n_rows = int(offsets[-1])
    n_images = int(row_counts.size)
    return ResidentCandidateTables(
        n_images=n_images,
        n_rows=n_rows,
        n_fine_trans=4,
        n_coarse_trans=2,
        row_offsets=offsets,
        row_unit=np.repeat(np.arange(n_images, dtype=np.int32), row_counts),
        row_fine_rot=np.zeros(n_rows, dtype=np.int32),
        row_parent_local=np.zeros(n_rows, dtype=np.int32),
        row_log_prior=np.zeros(n_rows, dtype=np.float32),
        mask_mode=np.zeros(n_images, dtype=np.int8),
        parent_offsets=np.zeros(n_images + 1, dtype=np.int32),
        parent_trans_bits=np.zeros((0, 1), dtype=np.uint32),
    )


def test_chunk_segment_offsets_cover_each_image_once_and_pad_empty():
    tables = _tables([3, 5, 2, 7])
    chunk = CapacityChunk(
        image_start=1,
        image_stop=3,
        row_start=3,
        row_stop=10,
        row_capacity=16,
        image_capacity=4,
    )
    offsets = rp._chunk_segment_offsets(tables, chunk, n_fine_trans=4)
    assert offsets.dtype == np.int32
    assert offsets.shape == (5,)
    # Image 1 owns 5 rows, image 2 owns 2; both are contiguous from cell 0.
    assert_matches(offsets, np.asarray([0, 20, 28, 28, 28], dtype=np.int32))
    assert np.all(np.diff(offsets) >= 0)
    assert int(offsets[-1]) == chunk.n_valid_rows * 4
    # Padded slots are empty segments; the rows past n_valid_rows are covered
    # by no segment at all, which the handler treats as an all -inf row.
    assert int(offsets[3]) == int(offsets[4])


def test_flat_row_weighted_sums_agree_with_the_rectangular_mstep_sums():
    """The flat-row weighted sums reproduce ``compute_local_mstep_sums``.

    Both call ``compute_local_weighted_sums`` with its pinned
    ``Precision.HIGHEST``; the only change is that each flat row gathers its
    own image tile, which gives the translation contraction a singleton
    rotation axis. That reassociates a float32 GEMM, so the two are not
    bitwise equal on GPU. What is asserted is what matters: every output must
    agree with the rectangular one, and the contracted sums with a float64
    reference, to float32 relative accuracy. On an
    A100 the measured relative L2 against float64 is 1.8e-7 for the
    rectangular layout and 7.5e-8 for the flat-row layout at production
    shapes, so the flat-row layout is the slightly more accurate of the two.
    The per-element maximum relative difference reaches 5e-4, but only where
    the summed value itself has cancelled to near zero.
    """

    rng = np.random.default_rng(20260918)
    batch, n_rot, n_trans, n_pix = 8, 32, 12, 24
    probs = np.abs(rng.normal(size=(batch, n_rot, n_trans))).astype(np.float32)
    probs[0, 2, :] = 0.0  # a row with no posterior mass exercises the != 0 guard
    shifted = (
        rng.normal(size=(batch, n_trans, n_pix)) + 1j * rng.normal(size=(batch, n_trans, n_pix))
    ).astype(np.complex64)
    noise = (
        rng.normal(size=(batch, n_trans, n_pix)) + 1j * rng.normal(size=(batch, n_trans, n_pix))
    ).astype(np.complex64)
    ctf = np.abs(rng.normal(size=(batch, n_pix))).astype(np.float32)

    summed_rect, ctf_rect = compute_local_mstep_sums(
        jnp.asarray(probs),
        jnp.asarray(shifted),
        jnp.asarray(ctf),
        relion_x_half=True,
        sequential_translation_reduction=False,
    )
    masked_rect = compute_local_mstep_sums(
        jnp.asarray(probs),
        jnp.asarray(noise),
        jnp.asarray(ctf),
        relion_x_half=True,
        sequential_translation_reduction=False,
    )[0]

    row_image = np.repeat(np.arange(batch, dtype=np.int32), n_rot)
    summed, summed_masked, ctf_probs, probs_sum_t = rp._resident_block_weighted_sums(
        jnp.asarray(probs.reshape(batch * n_rot, n_trans)),
        jnp.asarray(row_image),
        jnp.asarray(shifted),
        jnp.asarray(noise),
        jnp.asarray(ctf),
    )

    def rel_l2(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        den = float(np.linalg.norm(a))
        return float(np.linalg.norm(a - b) / den) if den else 0.0

    for flat, rect, tile in (
        (summed, summed_rect, shifted),
        (summed_masked, masked_rect, noise),
    ):
        flat = np.asarray(flat)
        rect = np.asarray(rect).reshape(batch * n_rot, n_pix)
        reference = np.einsum(
            "brt,btp->brp", probs.astype(np.float64), tile.astype(np.complex128)
        ).reshape(batch * n_rot, n_pix)
        assert rel_l2(rect, flat) < 1e-6
        assert rel_l2(reference, flat) <= rel_l2(reference, rect) * 2.0
        assert rel_l2(reference, flat) < 1e-6

    # The rotation-posterior sum reduces the same translation axis, so it is
    # reassociated by the same shape change: measured at 2.2e-7 relative on an
    # A100. The CTF sum is that value times an elementwise row, so it inherits
    # the same bound rather than being bitwise.
    np.testing.assert_allclose(
        np.asarray(probs_sum_t),
        np.asarray(jnp.sum(jnp.asarray(probs), axis=-1)).reshape(batch * n_rot),
        rtol=1e-6,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(ctf_probs),
        np.asarray(ctf_rect).reshape(batch * n_rot, n_pix),
        rtol=1e-6,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "batch,n_rot,n_trans,n_rect,n_exact",
    [(2, 3, 4, 10, 6), (5, 64, 84, 97, 61), (1, 17, 3, 8, 8), (7, 1, 9, 33, 2)],
)
@pytest.mark.parametrize("power_at_exact_positions", [False, True])
def test_flat_row_wavg_rectangle_image_sums_match_the_rectangular_helper(
    batch, n_rot, n_trans, n_rect, n_exact, power_at_exact_positions
):
    """Per image, the rows' rectangle terms plus the marginal's power are the helper's sums.

    The rectangular helper fills the rectangle with each row's posterior-weighted
    image power and overwrites the exact positions; the Wavg accumulator sums
    those rows per image. The flat-row form returns the exact terms per row and
    the chunk adds the image power of each image's translation marginal once,
    so every image's sum over its rows must match: in the default band at the
    exact positions (the same terms, summed in row order), and in the float32
    band of the power's sum elsewhere, where only the grouping of the power's additions differs. On the
    algebraic route (``power_at_exact_positions``) the power covers the exact
    positions too. Only the logical rectangle is filled. Shapes cover the
    production ratio (many rows over few images), one row per image, a single
    rotation, an all-exact rectangle and a padded row.
    """

    rng = np.random.default_rng(4207 + n_rect)
    exact_positions = np.sort(
        rng.choice(n_rect, size=n_exact, replace=False).astype(np.int32)
    )
    exact_terms = rng.normal(size=(batch, n_rot, n_exact, 3)).astype(np.float32)
    raw_rect = (
        rng.normal(size=(batch, n_trans, n_rect)) + 1j * rng.normal(size=(batch, n_trans, n_rect))
    ).astype(np.complex64)
    posterior = np.abs(rng.normal(size=(batch, n_rot, n_trans))).astype(np.float32)

    rect = np.asarray(
        _relion_wavg_rectangle_triplet_terms(
            jnp.asarray(exact_terms),
            jnp.asarray(raw_rect),
            jnp.asarray(posterior),
            jnp.asarray(exact_positions),
        )
    )
    # One padding row (id -1) with weight must contribute nothing.
    row_ids = np.concatenate([np.repeat(np.arange(batch, dtype=np.int32), n_rot), [-1]])
    flat_exact = np.concatenate([exact_terms.reshape(batch * n_rot, n_exact, 3), exact_terms[:1, 0]])
    flat_posterior = np.concatenate([posterior.reshape(batch * n_rot, n_trans), posterior[:1, 0]])
    # The M-step blocks add the exact terms at their rectangle positions and
    # nothing elsewhere (relion_wavg_exact_atomic_flat_rows_triplet_add_f32).
    terms = np.zeros((flat_exact.shape[0], n_rect, 3), dtype=np.float32)
    terms[:, exact_positions, :] = flat_exact
    logical = n_rect - 1
    power = np.asarray(
        rp._add_wavg_rectangle_image_power(
            jnp.zeros((batch, n_rect, 3), dtype=jnp.float32),
            jnp.asarray(raw_rect),
            jnp.asarray(flat_posterior),
            jnp.asarray(row_ids),
            jnp.asarray(exact_positions),
            jnp.int32(logical),
            power_at_exact_positions=power_at_exact_positions,
        )
    )
    assert np.all(power[..., :2] == 0) and np.all(power[:, logical:, 2] == 0)
    power = power[..., 2]
    other = np.setdiff1d(np.arange(n_rect), exact_positions)
    assert np.all(terms[:, other, :] == 0)
    # The power's float64 truth: sum over the image's rows and translations of
    # w[r, t] |x_t[p]|^2, from the same float32 squares.
    square = (raw_rect.real * raw_rect.real).astype(np.float32) + (raw_rect.imag * raw_rect.imag).astype(np.float32)
    truth = np.einsum("brt,btp->bp", posterior.astype(np.float64), square.astype(np.float64))
    filled = np.arange(n_rect) < logical
    if not power_at_exact_positions:
        filled &= ~np.isin(np.arange(n_rect), exact_positions)
    # Both float32 groupings of these n_rot * n_trans positive terms are within
    # (n_rot + n_trans) float32 eps of their sum.
    bound = (n_rot + n_trans) * np.finfo(np.float32).eps * truth[:, filled]
    for image in range(batch):
        rows = terms[row_ids == image]
        assert_matches(
            np.cumsum(rows[:, exact_positions, :], axis=0, dtype=np.float32)[-1],
            np.cumsum(rect[image][:, exact_positions, :], axis=0, dtype=np.float32)[-1],
        )
        assert np.all(np.abs(power[image, filled] - truth[image, filled]) <= bound[image])
        assert np.all(power[image, ~filled] == 0)


def test_wavg_shifted_power_commutes_with_a_row_gather():
    """The identity the hoist rests on, stated on its own.

    ``square(gather(x)) == gather(square(x))`` in float32, including the
    optimization barrier that keeps the two products from contracting.
    """

    from relax.sparse_pass2.sparse_pass2_wavg import _relion_wavg_shifted_power

    rng = np.random.default_rng(5150)
    rect = (
        rng.normal(size=(6, 11, 29)) + 1j * rng.normal(size=(6, 11, 29))
    ).astype(np.complex64)
    take = rng.integers(0, 6, size=97).astype(np.int32)
    gather_then_square = np.asarray(
        _relion_wavg_shifted_power(jnp.asarray(rect)[jnp.asarray(take)])
    )
    square_then_gather = np.asarray(
        _relion_wavg_shifted_power(jnp.asarray(rect))[jnp.asarray(take)]
    )
    assert_matches(
        square_then_gather, gather_then_square
    )


def test_mstep_block_rows_divides_every_row_capacity():
    ladder = rp._DEFAULT_ROW_CAPACITY_LADDER
    block = rp._resolve_mstep_block_rows(
        n_recon_pixels=4324, max_block_bytes=513124859, row_capacity_ladder=ladder
    )
    assert block > 0 and block & (block - 1) == 0
    assert all(capacity % block == 0 for capacity in ladder)


def test_image_capacity_ladder_is_capped_by_the_translation_tile_budget():
    ladder = rp._cap_image_capacity_ladder(
        (32, 128, 512), n_fine_trans=100, n_recon_pixels=4324, max_tile_bytes=1_197_291_339
    )
    assert ladder and list(ladder) == sorted(ladder)
    assert max(ladder) <= 512
    # A budget below the smallest class plans the largest power of two of images
    # that fits, at least one (10202 it13, 14460403: 32-image tiles ran out of memory).
    assert rp._cap_image_capacity_ladder(
        (32, 128, 512), n_fine_trans=100, n_recon_pixels=4324, max_tile_bytes=1
    ) == (1,)
    per_image = 148 * 132095 * 8 * 3
    assert rp._cap_image_capacity_ladder(
        (32, 128, 512), n_fine_trans=148, n_recon_pixels=132095, max_tile_bytes=int(0.02 * 76e9)
    ) == (2,)
    assert rp._cap_image_capacity_ladder(
        (32, 128, 512), n_fine_trans=148, n_recon_pixels=132095, max_tile_bytes=31 * per_image
    ) == (16,)


def test_joint_chunk_plan_fits_the_10202_iteration_14_shape():
    """10202 it14 (14475506): T=148, 135755 recon pixels, local rows of 2.6 MiB. The
    rows, the translation tiles and a 64-row M-step block (9.7 GiB, sized without T)
    overcommitted the device together; one plan now sizes all three from one budget."""

    from relax.sparse_pass2.resident_scoring import resident_row_projection_bytes

    gib = 1024**3
    t, p_recon, p_score = 148, 135755, 137153
    row_bytes = resident_row_projection_bytes(n_score_pixels=p_score, n_recon_pixels=p_recon)
    fixed_tile = int(0.02 * 76e9)
    budget = 20 * gib
    images = rp.resident_image_capacity_start(
        (32, 128, 512), n_fine_trans=t, n_recon_pixels=p_recon, max_tile_bytes=fixed_tile,
        chunk_budget_bytes=budget,
    )
    assert images == (32, 128, 512)  # the fixed tile budget fits no class: the joint plan sizes them
    plan = rp.plan_resident_chunk_memory(
        row_capacity_ladder=(1024, 4096), image_capacity_ladder=images, mstep_block_rows=64,
        row_bytes=row_bytes, n_fine_trans=t, n_recon_pixels=p_recon, budget_bytes=budget,
    )
    assert plan.peak_bytes <= budget
    assert plan.peak_bytes == rp.resident_chunk_bytes(
        row_capacity=max(plan.row_capacity_ladder), image_capacity=max(plan.image_capacity_ladder),
        mstep_block_rows=plan.mstep_block_rows, row_bytes=row_bytes, n_fine_trans=t, n_recon_pixels=p_recon,
    )
    assert plan.mstep_block_rows < 64
    assert max(plan.image_capacity_ladder) >= 4  # more than the 2-image fixed fallback
    assert 1024 % plan.mstep_block_rows == 0
    # A single row class halves until the chunk fits (bigbox 14640954: the 10202 final pass of a
    # cold run had 13.81 GiB for a 13.90 GiB 1024-row chunk).
    halved = rp.plan_resident_chunk_memory(
        row_capacity_ladder=(1024,), image_capacity_ladder=(1,), mstep_block_rows=1,
        row_bytes=row_bytes, n_fine_trans=t, n_recon_pixels=p_recon, budget_bytes=1 * gib,
    )
    assert max(halved.row_capacity_ladder) < 1024 and halved.peak_bytes <= 1 * gib
    # A budget below the smallest chunk is a refusal (there is no other pass-2 engine to run it on).
    with pytest.raises(rp.ResidentConfigurationUnsupported, match="smallest chunk"):
        rp.plan_resident_chunk_memory(
            row_capacity_ladder=(1024,), image_capacity_ladder=(1,), mstep_block_rows=1,
            row_bytes=row_bytes, n_fine_trans=t, n_recon_pixels=p_recon, budget_bytes=row_bytes * 32,
        )


def test_float32_posterior_bucket_scratch_is_admitted_for_regular_pipeline_and_overflow():
    """The row-stage plan includes two bucket partials and output at each capacity."""

    from relax.sparse_pass2.resident_statistics import posterior_translation_bucket_scratch_bytes

    base = dict(row_bytes=1, n_fine_trans=8, n_recon_pixels=1, mstep_block_rows=1,
                held_tile_pixels=0, prepare_tile_pixels=0, projection_transient_bytes=0)
    row, image = 4096, 32
    scratch = posterior_translation_bucket_scratch_bytes(row, image, 8)
    plain = rp.resident_chunk_bytes(row_capacity=row, image_capacity=image, **base)
    bucket = rp.resident_chunk_bytes(
        row_capacity=row, image_capacity=image, float32_posterior_buckets=True, **base,
    )
    assert bucket == row + scratch
    assert bucket > plain
    pipelined = rp.resident_chunk_bytes(
        row_capacity=row, image_capacity=image, float32_posterior_buckets=True,
        pipelined=True, **base,
    )
    assert pipelined == bucket + row
    plan = rp.plan_resident_chunk_memory(
        row_capacity_ladder=(4096, 131072), image_capacity_ladder=(32,),
        budget_bytes=bucket + 100, float32_posterior_buckets=True, **base,
    )
    assert plan.row_capacity_ladder == (4096,)
    assert plan.peak_bytes == bucket
    minimum = rp.resident_chunk_bytes(
        row_capacity=row, image_capacity=1, float32_posterior_buckets=True, **base,
    )
    with pytest.raises(rp.ResidentConfigurationUnsupported, match="smallest chunk"):
        rp.plan_resident_chunk_memory(
            row_capacity_ladder=(4096,), image_capacity_ladder=(1,),
            budget_bytes=minimum - 1, float32_posterior_buckets=True, **base,
        )
    overflow_row = rp.overflow_row_capacity(5000, (4096,))
    overflow = rp.resident_chunk_bytes(
        row_capacity=overflow_row, image_capacity=image, float32_posterior_buckets=True, **base,
    )
    overflow_plan = rp.plan_resident_chunk_memory(
        row_capacity_ladder=(4096,), image_capacity_ladder=(32,),
        max_image_rows=5000, budget_bytes=overflow,
        float32_posterior_buckets=True, **base,
    )
    assert overflow_plan.peak_bytes == overflow


def test_joint_chunk_plan_counts_the_local_preparation_stage():
    """10202 it22 (14509861): T=84, 153857 recon and 155355 score pixels, a 626 x 314 Wavg
    rectangle. Three recon tiles per image put 32 images and an 18.55 GiB peak inside a
    22.2 GiB budget, but the local preparation of the translated tiles holds ~30 GiB at
    32 images and ran out of memory. Counting that stage shrinks the image classes until
    it fits; the default count (no preparation stage) reproduces the old plan, and the
    unshifted operands (T16), whose only translated arrays are the Wavg rectangle and its
    exact positions, keep the 32 images."""

    gib = 1024**3
    t, p_recon, p_score, n_rect = 84, 153857, 155355, 626 * 314
    row_bytes = int(3016.72 * 1024)
    budget = int(22.20 * gib)
    kwargs = dict(
        row_capacity_ladder=(1024,), image_capacity_ladder=(32,), mstep_block_rows=32,
        row_bytes=row_bytes, n_fine_trans=t, n_recon_pixels=p_recon, budget_bytes=budget,
    )
    old = rp.plan_resident_chunk_memory(**kwargs)
    assert old.image_capacity_ladder == (32,)
    assert old.peak_bytes / gib == pytest.approx(18.55, abs=0.01)

    tiles = rp.chunk_translated_tile_pixels(
        unshifted_operands=False,
        n_score_pixels=p_score, n_recon_pixels=p_recon, n_rect_pixels=n_rect, n_exact_rect_pixels=p_recon,
    )
    assert 32 * t * tiles["prepare_tile_pixels"] * 8 > budget
    plan = rp.plan_resident_chunk_memory(**kwargs, **tiles)
    assert plan.peak_bytes <= budget
    assert max(plan.image_capacity_ladder) * t * tiles["prepare_tile_pixels"] * 8 <= budget
    assert max(plan.image_capacity_ladder) >= 8
    assert plan.row_capacity_ladder == (1024,)  # only the image classes pay for the preparation stage
    assert plan.peak_bytes == rp.resident_chunk_bytes(
        row_capacity=1024, image_capacity=max(plan.image_capacity_ladder), mstep_block_rows=plan.mstep_block_rows,
        row_bytes=row_bytes, n_fine_trans=t, n_recon_pixels=p_recon, **tiles,
    )

    unshifted = rp.chunk_translated_tile_pixels(
        unshifted_operands=True,
        n_score_pixels=p_score, n_recon_pixels=p_recon, n_rect_pixels=n_rect, n_exact_rect_pixels=p_recon,
    )
    assert rp.plan_resident_chunk_memory(**kwargs, **unshifted).image_capacity_ladder == (32,)


def test_joint_chunk_plan_counts_the_global_per_chunk_preparation_with_its_rows():
    """VDAM K=1 ribosembly 100k (bench refresh 2026-09-27): a streamed global pass keeps the
    per-chunk preparation, and the plan counted three recon tiles, so 512-image chunks at
    current size 248 (T=84, a 248 x 125 Wavg rectangle) allocated the 9.93 GiB rectangle
    tile (512 x 84 x 31000 x 8 bytes) next to the chunk's projected rows and ran out of
    memory. The global pass projects the rows before it prepares the operands, so the
    preparation stage includes them."""

    gib = 1024**3
    # T and the rectangle from the failed allocation; the window sizes are the current-size-248
    # half disk (the run logged no plan line).
    t, p_recon, p_score, n_rect, n_exact = 84, 24000, 24200, 248 * 125, 24000
    assert 512 * t * n_rect * 8 == 10_665_984_000
    row_bytes = 3 * p_score * 8
    budget = 30 * gib
    kwargs = dict(
        row_capacity_ladder=(8192, 32768), image_capacity_ladder=(32, 128, 512), mstep_block_rows=1024,
        row_bytes=row_bytes, n_fine_trans=t, n_recon_pixels=p_recon, budget_bytes=budget,
    )
    tiles = rp.chunk_translated_tile_pixels(
        unshifted_operands=False,
        n_score_pixels=p_score, n_recon_pixels=p_recon, n_rect_pixels=n_rect, n_exact_rect_pixels=n_exact,
    )
    # The failing 512-image chunk's preparation, which three recon tiles per image did not count.
    assert 512 * t * tiles["prepare_tile_pixels"] * 8 > budget

    plan = rp.plan_resident_chunk_memory(**kwargs, **tiles, rows_live_during_prepare=True)
    assert plan.peak_bytes <= budget
    images, rows = max(plan.image_capacity_ladder), max(plan.row_capacity_ladder)
    assert images * t * tiles["prepare_tile_pixels"] * 8 + rows * row_bytes <= budget
    assert 32 <= images < 512
    assert plan.peak_bytes == rp.resident_chunk_bytes(
        row_capacity=rows, image_capacity=images, mstep_block_rows=plan.mstep_block_rows,
        row_bytes=row_bytes, n_fine_trans=t, n_recon_pixels=p_recon, rows_live_during_prepare=True, **tiles,
    )
    # With the half's resident operands the chunk holds only the rectangle and its exact positions.
    unshifted = rp.chunk_translated_tile_pixels(
        unshifted_operands=True,
        n_score_pixels=p_score, n_recon_pixels=p_recon, n_rect_pixels=n_rect, n_exact_rect_pixels=n_exact,
    )
    assert unshifted["prepare_tile_pixels"] < tiles["prepare_tile_pixels"] / 4
    assert max(rp.plan_resident_chunk_memory(
        **kwargs, **unshifted, rows_live_during_prepare=True
    ).image_capacity_ladder) >= images


def test_unshifted_mstep_blocks_gather_no_translated_tile():
    """An unshifted chunk's M-step block row holds only its sums, so the joint plan keeps
    larger blocks and images than the pre-shifted count at the 10097 full-box final pass
    (T=36, 25717 recon, 33024 rectangle pixels, 503 KiB rows, a 37.6 GiB budget)."""

    t, p, rect = 36, 25717, 33024
    assert rp._mstep_block_row_bytes(t, p, 0) == 44 * p
    assert rp._mstep_block_row_bytes(t, p) == 2 * t * p * 8 + 44 * p
    kwargs = dict(
        row_capacity_ladder=(1024, 4096, 16384), image_capacity_ladder=(32, 128, 512), mstep_block_rows=1024,
        row_bytes=503 * 1024, n_fine_trans=t, n_recon_pixels=p, budget_bytes=int(37.6 * 1024**3),
    )
    tiles = rp.chunk_translated_tile_pixels(
        unshifted_operands=False, n_score_pixels=p, n_recon_pixels=p, n_rect_pixels=rect, n_exact_rect_pixels=p
    )
    unshifted = rp.chunk_translated_tile_pixels(
        unshifted_operands=True, n_score_pixels=p, n_recon_pixels=p, n_rect_pixels=rect, n_exact_rect_pixels=p
    )
    assert unshifted["mstep_tile_pixels"] == 0 and tiles["mstep_tile_pixels"] == 2 * p
    pre_shifted = rp.plan_resident_chunk_memory(**kwargs, **tiles)
    plan = rp.plan_resident_chunk_memory(**kwargs, **unshifted)
    assert plan.peak_bytes <= kwargs["budget_bytes"]
    assert plan.mstep_block_rows >= pre_shifted.mstep_block_rows
    assert max(plan.image_capacity_ladder) >= max(pre_shifted.image_capacity_ladder)
    assert plan.peak_bytes == rp.resident_chunk_bytes(
        row_capacity=max(plan.row_capacity_ladder), image_capacity=max(plan.image_capacity_ladder),
        mstep_block_rows=plan.mstep_block_rows, row_bytes=503 * 1024, n_fine_trans=t, n_recon_pixels=p, **unshifted,
    )


def test_pipelined_chunk_plan_counts_the_previous_chunk():
    """EMPIAR-10202 iteration 22 (box 800, current size 626, T=84): the pipelined local loop
    holds chunk k's row projections and tiles while chunk k+1 runs, and the plan counted one
    chunk. It kept a 4096-row class (11.78 GiB of 3016.72 KiB rows, a 19.03 GiB modelled
    peak of a 22.12 GiB budget) and ran out of memory (bigbox 14564062)."""

    gib = 1024**3
    t, p = 84, 155356
    tiles = rp.chunk_translated_tile_pixels(
        unshifted_operands=True, n_score_pixels=p, n_recon_pixels=p, n_rect_pixels=158000,
        n_exact_rect_pixels=158000,
    )
    kwargs = dict(
        row_capacity_ladder=(1024, 4096), image_capacity_ladder=(32,), mstep_block_rows=32,
        row_bytes=int(3016.72 * 1024), n_fine_trans=t, n_recon_pixels=p, budget_bytes=int(22.12 * gib),
        **tiles,
    )
    serial = rp.plan_resident_chunk_memory(**kwargs)
    assert serial.row_capacity_ladder == (1024, 4096)
    pipelined = rp.plan_resident_chunk_memory(**kwargs, pipelined=True)
    assert pipelined.row_capacity_ladder == (1024,)
    assert pipelined.peak_bytes <= kwargs["budget_bytes"]
    rows = 1024 * kwargs["row_bytes"]
    held = 32 * t * tiles["held_tile_pixels"] * 8
    single = rp.resident_chunk_bytes(
        row_capacity=1024, image_capacity=32, mstep_block_rows=pipelined.mstep_block_rows,
        row_bytes=kwargs["row_bytes"], n_fine_trans=t, n_recon_pixels=p, **tiles,
    )
    assert pipelined.peak_bytes == single + rows + held


def test_k_class_projection_cache_is_built_in_place():
    """VDAM pdb K=2 5k (main 46b4f64) ran out of memory at iteration 96 concatenating the
    per-class union caches, which held the cache twice. The build writes every projector
    call's rows into one class-major cache, so at most one call's rows are live next to it."""

    n_classes, n_rows, n_pixels, per_call = 3, 10, 7, 4
    rng = np.random.default_rng(3)
    classes = [
        (rng.standard_normal((n_rows, n_pixels)) + 1j * rng.standard_normal((n_rows, n_pixels))).astype(np.complex64)
        for _ in range(n_classes)
    ]
    cache_bytes = n_classes * n_rows * n_pixels * 8
    peak = []

    def project_rows(class_index, start, stop):
        rows = jnp.asarray(classes[class_index][start:stop])
        peak.append(sum(a.nbytes for a in jax.live_arrays() if a.dtype == jnp.complex64))
        return rows

    baseline = sum(a.nbytes for a in jax.live_arrays() if a.dtype == jnp.complex64)
    cache = rp.build_projection_cache_in_place(
        project_rows, n_classes=n_classes, n_rows_per_class=n_rows, n_pixels=n_pixels,
        rows_per_call=per_call, dtype=jnp.complex64,
    )
    assert_matches(np.asarray(cache), np.concatenate(classes, axis=0))
    assert len(peak) == n_classes * 3
    assert max(peak) - baseline <= cache_bytes + per_call * n_pixels * 8


def test_full_box_chunk_plan_counts_the_projector_call_and_the_accumulators():
    """EMPIAR-10202 final pass (box 800, current size 800, padding 2, T=36): one projector call
    holds the texture crop (800 x 401 px) beside its gathered rows, so a constant reserve of
    one full-half-spectrum call (1673 rows x 320,800 px x 8 B = 4 GiB) was about a third of
    the call's real peak and the pass ran out of memory allocating the second 4 GiB array
    (bigbox 14560923). The call's bytes now enter the chunk plan, and the pass's accumulators
    (1603 x 1603 x 802 x (8 + 4) B) are its fixed bytes."""

    import numpy as np

    from relax.sparse_pass2.resident_scoring import resident_row_projection_bytes

    gib = 1024**3
    crop, n_score, n_recon, union = 800 * 401, 251966, 251313, 252000
    per_row = rp.projection_call_row_bytes(
        crop_pixels=crop, output_pixels=union, n_score_pixels=n_score, n_recon_pixels=n_recon
    )
    # The placement stage (union + zero column + both windows + |recon|^2) is this call's peak.
    assert per_row == 8 * (union + 1 + n_score + n_recon) + 4 * n_recon
    assert per_row > 8 * (crop + union) > 8 * crop
    old_rows = 4 * gib // (crop * 8)
    assert old_rows == 1673
    full_row = rp.projection_call_row_bytes(
        crop_pixels=crop, output_pixels=crop, n_score_pixels=n_score, n_recon_pixels=n_recon
    )
    assert old_rows * full_row > 2.5 * 4 * gib  # what the old reserve let one call hold
    rows = 4 * gib // per_row
    transient = rows * per_row
    assert transient <= 4 * gib

    t = 36
    tiles = rp.chunk_translated_tile_pixels(
        unshifted_operands=True, n_score_pixels=n_score, n_recon_pixels=n_recon, n_rect_pixels=crop,
        n_exact_rect_pixels=crop,
    )
    accumulators = rp.resident_accumulator_bytes(1603 * 1603 * 802, np.complex64, np.float32)
    assert accumulators == 1603 * 1603 * 802 * 12
    kwargs = dict(
        row_capacity_ladder=(1024, 4096), image_capacity_ladder=(16, 32), mstep_block_rows=32,
        row_bytes=resident_row_projection_bytes(n_score_pixels=n_score, n_recon_pixels=n_recon),
        n_fine_trans=t, n_recon_pixels=n_recon, budget_bytes=int(22.12 * gib), **tiles,
    )
    without = rp.plan_resident_chunk_memory(**kwargs)
    plan = rp.plan_resident_chunk_memory(**kwargs, projection_transient_bytes=transient, fixed_bytes=accumulators)
    assert plan.peak_bytes <= kwargs["budget_bytes"]
    mstep = plan.mstep_block_rows * rp._mstep_block_row_bytes(t, n_recon, tiles["mstep_tile_pixels"])
    assert transient > mstep
    assert plan.peak_bytes == rp.resident_chunk_bytes(
        row_capacity=max(plan.row_capacity_ladder), image_capacity=max(plan.image_capacity_ladder),
        mstep_block_rows=plan.mstep_block_rows, row_bytes=kwargs["row_bytes"], n_fine_trans=t,
        n_recon_pixels=n_recon, projection_transient_bytes=transient, **tiles,
    )
    assert plan.peak_bytes - without.peak_bytes == transient - mstep
    assert plan.pass_bytes == accumulators + plan.peak_bytes
    # The projector call and the M-step block are never live together: a call smaller than
    # the block adds nothing.
    assert rp.resident_chunk_bytes(
        row_capacity=1024, image_capacity=32, mstep_block_rows=32, row_bytes=kwargs["row_bytes"],
        n_fine_trans=t, n_recon_pixels=n_recon, projection_transient_bytes=mstep // 2, **tiles,
    ) == rp.resident_chunk_bytes(
        row_capacity=1024, image_capacity=32, mstep_block_rows=32, row_bytes=kwargs["row_bytes"],
        n_fine_trans=t, n_recon_pixels=n_recon, **tiles,
    )


def test_joint_chunk_plan_leaves_a_box_256_plan_unchanged():
    """Box-256 K=1 plans (10097, noise1 50k: T<=116, <=8.4k pixels) fit the joint budget
    as they stand; the planner changes nothing there, and an unknown budget never caps."""

    kwargs = dict(
        row_capacity_ladder=(8192, 32768, 131072), image_capacity_ladder=(32, 128), mstep_block_rows=256,
        row_bytes=8121 * 8, n_fine_trans=116, n_recon_pixels=7475,
    )
    for budget in (30 * 1024**3, None):
        plan = rp.plan_resident_chunk_memory(budget_bytes=budget, **kwargs)
        assert plan.row_capacity_ladder == (8192, 32768, 131072)
        assert plan.image_capacity_ladder == (32, 128)
        assert plan.mstep_block_rows == 256
    # Where the fixed tile budget fits a class, the starting image classes are its classes.
    fixed = dict(n_fine_trans=116, n_recon_pixels=7475, max_tile_bytes=int(0.02 * 80e9))
    assert rp.resident_image_capacity_start(
        (32, 128, 512), chunk_budget_bytes=30 * 1024**3, **fixed
    ) == rp._cap_image_capacity_ladder((32, 128, 512), **fixed)


# ---------------------------------------------------------------------------
# GPU: the whole driver
# ---------------------------------------------------------------------------


def _gpu_available():
    """Whether this process can run every CUDA-only resident stage."""

    if jax.default_backend() != "gpu":
        return False
    from recovar import cuda_backproject

    from relax.cuda import kernels as em_cuda_kernels

    return bool(
        cuda_backproject.custom_cuda_requested()
        and em_cuda_kernels.sparse_pass2_segmented_supported()
        and em_cuda_kernels.relion_wavg_sequential_runtime_flat_rows_triplet_f32_supported()
        and em_cuda_kernels.relion_wavg_rotation_atomic_runtime_flat_rows_triplet_add_f32_supported()
    )


requires_resident_gpu = pytest.mark.skipif(
    not _gpu_available(),
    reason=(
        "the resident driver's scoring, segmented posterior, flat-row Wavg and "
        "x-half backprojection stages are all CUDA FFI targets"
    ),
)


def _z_rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _driver_fixture_args(seed=20260918):
    """A small K=1 pass in the production configuration both engines accept.

    The fixture uses the 8x8 ``MockDataset`` of the bucketed parity tests with
    a current-size window (so the score window excludes the ``ky=-N/2`` Nyquist
    row), RELION's x-half M-step, the float32 fine posterior, the atomic Wavg
    triplet and one scale-correction group. It does not use RELION's exact
    BPref operands, which need a STAR-backed dataset, so the Wavg triplet takes
    the algebraic branch here; the production path takes the sequential CUDA
    branch, which the matched Slurm pair covers.
    """

    from helpers.em_arrays import _hermitian_volume
    from test_sparse_pass2_bucketed_parity import IMAGE_SHAPE, IMAGE_SIZE, VOLUME_SHAPE, MockDataset

    from relax.sampling import rotation_grid_size

    nside_level = 1
    n_coarse_rot = rotation_grid_size(nside_level)
    n_images = 12
    children = 2
    rng = np.random.default_rng(seed)

    fine_rotations = np.stack(
        [_z_rotation(0.031 * k) for k in range(n_coarse_rot * children)]
    ).astype(np.float32)
    fine_parent = np.repeat(np.arange(n_coarse_rot, dtype=np.int32), children)
    translations = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32
    )
    n_coarse_trans = translations.shape[0]
    fine_translations = np.concatenate([translations, translations + 0.5]).astype(np.float32)
    fine_translation_parent = np.concatenate(
        [np.arange(n_coarse_trans), np.arange(n_coarse_trans)]
    ).astype(np.int32)

    total = n_coarse_rot * n_coarse_trans
    samples = [None]
    for _ in range(1, n_images):
        count = int(rng.integers(1, total))
        samples.append(np.sort(rng.choice(total, size=count, replace=False).astype(np.int32)))

    n_shells = IMAGE_SHAPE[0] // 2 + 1
    return dict(
        experiment_dataset=MockDataset(n_images=n_images, seed=11),
        volume=_hermitian_volume(VOLUME_SHAPE, seed=17),
        noise_variance=jnp.ones(IMAGE_SIZE, dtype=jnp.float32) * 0.8,
        translations=translations,
        significant_sample_indices=samples,
        nside_level=nside_level,
        disc_type="linear_interp",
        oversampling_order=0,
        current_size=6,
        translation_step=1.0,
        rotation_log_prior=rng.normal(scale=0.1, size=n_coarse_rot).astype(np.float32),
        score_with_masked_images=False,
        return_stats=True,
        translation_log_prior=rng.normal(
            scale=0.05, size=(n_images, n_coarse_trans)
        ).astype(np.float32),
        accumulate_noise=True,
        half_spectrum_scoring=True,
        projection_padding_factor=2,
        reconstruction_padding_factor=2,
        image_corrections=None,
        scale_corrections=None,
        image_pre_shifts=None,
        use_float64_scoring=False,
        random_perturbation=0.0,
        group_ids=np.zeros(n_images, dtype=np.int32),
        scale_correction_group_count=1,
        scale_correction_data_vs_prior=np.full(n_shells, 5.0, dtype=np.float64),
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
        fine_translations_override=fine_translations,
        fine_translation_parent_override=fine_translation_parent,
        relion_x_half_mstep=True,
        relion_fine_mstep_prune=True,
        relion_f32_fine_posterior=True,
        relion_exact_fine_gaussian=True,
        # Production K=1 runs the exact rectangular CUDA reduction (k_class.py
        # sets this whenever the custom library is available). Without it the
        # compact engine takes the XLA 256-lane emulation, which differs from
        # every CUDA scorer by a few ULP and would confound the comparison.
        relion_fine_diff2_fused_ffi=True,
        preserve_bpref_particle_order=True,
        source_faithful_spectrum_norm=False,
        return_score_log_z=True,
        # Production K=1 reaches the M-step call through
        # k_class.py::_run_sparse_k_class_adaptive_pass2, which always supplies
        # the other classes' log-Z. At K=1 there are no other classes, so the
        # vector is all -inf; carry it here so the fixture exercises the same
        # branch the matched Slurm pairs do.
        normalization_other_score_log_z=np.full(n_images, -np.inf, dtype=np.float64),
        normalization_score_mode="gaussian",
        adaptive_fraction=0.999,
    )


def _float32_atomic_order_bound(args) -> float:
    """Relative-L2 bound on a map accumulated by float32 atomics in a run-dependent order.

    The x-half backprojection adds each image's weighted slice into the shared
    float32 accumulator with atomics, so a voxel is a float32 sum of up to one
    term per image whose order changes between identical runs. Reordering an
    n-term float32 sum moves it by about sqrt(n) * eps32 relative, which bounds
    the map's relative L2 difference.
    """

    n_images = int(args["experiment_dataset"].n_units)
    return float(np.sqrt(n_images) * np.finfo(np.float32).eps)


@pytest.fixture
def _resident_production_env(monkeypatch):
    monkeypatch.setenv("RELAX_EM_PROTOTYPE_SOFT_POSTERIOR_BLOCK_BPREF", "1")
    monkeypatch.setenv("RELAX_RELION_WAVG_ATOMIC_SCALE_AA", "1")
    monkeypatch.setenv("RELAX_RELION_WAVG_ATOMIC_DIRECT_NOISE_ONLY", "1")
    monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_ROW_CAPACITIES", "256,1024,4096")
    monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_IMAGE_CAPACITIES", "4,16,64")
    monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_MSTEP_BLOCK_ROWS", "128")


@requires_resident_gpu
@pytest.mark.parametrize("masked", [True, False], ids=["unshifted-operands", "per-chunk-tiles"])
def test_global_chunk_tile_count_matches_the_live_translated_arrays(_resident_production_env, masked):
    """The global pass's planned per-stage tile counts are the translated arrays a chunk
    holds. Unmasked scoring is a configuration the half's resident operands refuse after
    the plan, so that case also covers planning again for the per-chunk preparation
    (VDAM K=1 ribosembly 100k ran out of memory in that preparation, bench refresh
    2026-09-27)."""

    args = _driver_fixture_args()
    if masked:
        dataset = args["experiment_dataset"]
        dataset.image_source.image_mask = jnp.linspace(
            0.2, 1.0, dataset.image_size, dtype=jnp.float32
        ).reshape(dataset.image_shape)
    args["score_with_masked_images"] = masked

    planned, measured, baseline = [], [], set()
    live_log = []
    real_plan = rp.plan_resident_chunk_memory
    real_prepare = rp._prepare_chunk_reconstruction_operands
    real_rows = rp._chunk_operand_rows
    real_gather = rp.gather_resident_chunk_operands

    def translated_bytes(capacity, n_trans):
        # Complex arrays only: the fixture's 8 translations equal its 8x8 image rows, so a real
        # [images, 8, 8] image batch has the tiles' leading shape.
        total = 0
        for array in jax.live_arrays():
            if id(array) in baseline or not jnp.issubdtype(array.dtype, jnp.complexfloating):
                continue
            if (array.ndim == 2 and array.shape[0] == capacity * n_trans) or (
                array.ndim == 3 and tuple(array.shape[:2]) == (capacity, n_trans)
            ):
                total += array.nbytes
        live_log.append(
            (capacity, n_trans, sorted((tuple(a.shape), str(a.dtype)) for a in jax.live_arrays() if id(a) not in baseline))
        )
        return total

    def plan(**kwargs):
        planned.append(kwargs)
        return real_plan(**kwargs)

    def prepare(**kwargs):
        capacity, n_trans = int(kwargs["chunk"].image_capacity), int(kwargs["n_fine_trans"])
        baseline.clear()
        baseline.update(id(a) for a in jax.live_arrays())

        def rows(arrays, *a, **k):
            out = real_rows(arrays, *a, **k)
            measured.append(("prepare_tile_pixels", capacity * n_trans, translated_bytes(capacity, n_trans)))
            return out

        rp._chunk_operand_rows = rows
        try:
            result = real_prepare(**kwargs)
        finally:
            rp._chunk_operand_rows = real_rows
        measured.append(("held_tile_pixels", capacity * n_trans, translated_bytes(capacity, n_trans)))
        return result

    def gather(operands, image_slots, **kwargs):
        capacity, n_trans = int(np.asarray(image_slots).shape[0]), int(kwargs["translation_angles"].shape[0])
        baseline.clear()
        baseline.update(id(a) for a in jax.live_arrays())
        result = real_gather(operands, image_slots, **kwargs)
        for key in ("prepare_tile_pixels", "held_tile_pixels"):
            measured.append((key, capacity * n_trans, translated_bytes(capacity, n_trans)))
        return result

    patch = pytest.MonkeyPatch()
    patch.setattr(rp, "plan_resident_chunk_memory", plan)
    patch.setattr(rp, "_prepare_chunk_reconstruction_operands", prepare)
    patch.setattr(rp, "gather_resident_chunk_operands", gather)
    try:
        rp.compute_pass2_stats_resident(**args)
    finally:
        patch.undo()
    assert planned and measured
    final_plan = planned[-1]
    if masked:
        assert final_plan["prepare_tile_pixels"] == final_plan["held_tile_pixels"]
    else:
        assert final_plan["prepare_tile_pixels"] > final_plan["held_tile_pixels"]
    for (key, image_translations, live), arrays in zip(measured, live_log):
        assert live == image_translations * final_plan[key] * 8, (key, live, final_plan[key], arrays)


@requires_resident_gpu
def test_global_chunks_prepare_their_own_unshifted_operands(_resident_production_env, monkeypatch):
    """Without the half's resident operands, each chunk prepares its own images' unshifted
    operands, not the translated tiles, and the pass returns what the resident operands give.

    The operands are per-image pure, so only the M-step's float32 atomics may differ: the
    maps are held to the atomics' order bound, the ordered statistics to the default band.
    """

    args = _driver_fixture_args()
    dataset = args["experiment_dataset"]
    dataset.image_source.image_mask = jnp.linspace(
        0.2, 1.0, dataset.image_size, dtype=jnp.float32
    ).reshape(dataset.image_shape)
    args["score_with_masked_images"] = True
    resident = rp.compute_pass2_stats_resident(**args)

    chunk_calls, tile_calls = [], []
    real_chunk, real_tiles = rp.unshifted_chunk_operands, rp._prepare_chunk_reconstruction_operands
    monkeypatch.setattr(rp, "_resident_operands_fit", lambda *a, **k: False)
    monkeypatch.setattr(
        rp, "unshifted_chunk_operands", lambda *a, **k: chunk_calls.append(1) or real_chunk(*a, **k)
    )
    monkeypatch.setattr(
        rp, "_prepare_chunk_reconstruction_operands", lambda **k: tile_calls.append(1) or real_tiles(**k)
    )
    per_chunk = rp.compute_pass2_stats_resident(**args)
    assert chunk_calls and not tile_calls

    assert_matches(resident.hard_assignment, per_chunk.hard_assignment)
    assert_matches(resident.best_rotation_indices, per_chunk.best_rotation_indices)
    assert_matches(resident.best_translations, per_chunk.best_translations)
    assert_matches(np.asarray(resident.score_log_z), np.asarray(per_chunk.score_log_z))
    for field in (
        "log_evidence_per_image",
        "best_log_score_per_image",
        "max_posterior_per_image",
        "rotation_posterior_sums",
    ):
        assert_matches(
            np.asarray(getattr(resident.relion_stats, field)),
            np.asarray(getattr(per_chunk.relion_stats, field)),
            err_msg=field,
        )
    for field in ("wsum_sigma2_noise", "wsum_norm_correction"):
        assert_matches(
            np.asarray(getattr(per_chunk.noise_stats, field), dtype=np.float64),
            np.asarray(getattr(resident.noise_stats, field), dtype=np.float64),
            rtol=1e-6,
            err_msg=field,
        )

    def rel_l2(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        den = float(np.linalg.norm(a))
        return float(np.linalg.norm(a - b) / den) if den else 0.0

    bound = _float32_atomic_order_bound(args)
    assert rel_l2(resident.Ft_y, per_chunk.Ft_y) < bound
    assert rel_l2(resident.Ft_ctf, per_chunk.Ft_ctf) < bound


@requires_resident_gpu
def test_resident_driver_repeats_itself(_resident_production_env):
    """The resident driver's own repeat band, the reference for the table above."""

    args = _driver_fixture_args()
    first = rp.compute_pass2_stats_resident(**args)
    second = rp.compute_pass2_stats_resident(**args)
    assert_matches(first.hard_assignment, second.hard_assignment)
    # The statistics whose reductions are ordered are bit-reproducible.
    assert_matches(
        np.asarray(first.noise_stats.wsum_sigma2_noise),
        np.asarray(second.noise_stats.wsum_sigma2_noise),
        err_msg="wsum_sigma2_noise",
    )

    def rel_l2(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        den = float(np.linalg.norm(a))
        return float(np.linalg.norm(a - b) / den) if den else 0.0

    # Two reductions are not: the float32 BPref atomics, and the CUDA shell
    # binning behind the unweighted high image-power shell. Measured repeat
    # band on an A100: 1.5e-8 for the maps, 1.4e-8 for the image power. The map
    # bound is derived from the atomics (user approval 2026-09-27): an H100
    # repeat measured 1.036e-7 against the former 1e-7 (bigbox 14507539).
    assert rel_l2(first.Ft_y, second.Ft_y) < _float32_atomic_order_bound(args)
    assert rel_l2(first.noise_stats.wsum_img_power, second.noise_stats.wsum_img_power) < 1e-7
    # wsum_norm_correction adds the per-image relion_norm_high_shell term,
    # whose shell binning is the same racing scatter-add (resident_operands.py:
    # about 6e-8 relative between identical calls). H100 repeats missed by
    # 3.5e-8 on one image. Band approved by the user 2026-09-24; the exact
    # check is the deterministic-reductions arm below.
    assert rel_l2(
        first.noise_stats.wsum_norm_correction, second.noise_stats.wsum_norm_correction
    ) < 1e-7


@requires_resident_gpu
def test_glue_programs_match_the_loose_dispatch(_resident_production_env, monkeypatch):
    """P3-A: where the chunk loop's JIT boundary sits changes no output.

    ``RELAX_SPARSE_PASS2_RESIDENT_GLUE_JIT`` picks between one program per
    M-step block and the loose sequence of eager operations the per-stage path
    used before P3-A. The stage bodies are the same functions in both
    settings, so the discrete state must be equal and the ordered statistics
    match in the default band; the two reductions that are not bit-reproducible even between two
    identical runs -- the float32 BPref atomics and the CUDA shell binning --
    are held to the repeat band
    :func:`test_resident_driver_repeats_itself` measures.
    """

    from relax.helpers.deterministic_reduce import (
        deterministic_reductions_enabled,
    )

    args = _driver_fixture_args()
    monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_GLUE_JIT", "0")
    loose = rp.compute_pass2_stats_resident(**args)
    # The loose path's own repeat, in this process: the self-repeat band the
    # racing accumulators below are read against. It is measured, not assumed,
    # because the float32 BPref atomics and the CUDA shell binning do not
    # reproduce between two identical calls.
    loose_repeat = rp.compute_pass2_stats_resident(**args)
    monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_GLUE_JIT", "1")
    programs = rp.compute_pass2_stats_resident(**args)

    assert_matches(loose.hard_assignment, programs.hard_assignment)
    assert_matches(
        loose.best_rotation_indices, programs.best_rotation_indices
    )
    assert_matches(loose.best_rotations, programs.best_rotations)
    assert_matches(loose.best_translations, programs.best_translations)
    assert_matches(
        np.asarray(loose.score_log_z), np.asarray(programs.score_log_z)
    )
    for field in (
        "log_evidence_per_image",
        "best_log_score_per_image",
        "max_posterior_per_image",
        "rotation_posterior_sums",
    ):
        assert_matches(
            np.asarray(getattr(loose.relion_stats, field)),
            np.asarray(getattr(programs.relion_stats, field)),
            err_msg=field,
        )
    # ``wsum_sigma2_noise`` and ``wsum_norm_correction`` are shell sums fed by
    # the CUDA binning scatter of float32 terms, so they move at float32
    # precision between two identical runs: 3.5e-08 relative on one entry of
    # twelve in long GPU sessions. They get a 1e-6 relative band in every mode,
    # RELAX_EM_DETERMINISTIC_REDUCTIONS=1 included.
    for field in ("wsum_sigma2_noise", "wsum_norm_correction"):
        candidate = np.asarray(getattr(programs.noise_stats, field), dtype=np.float64)
        reference = np.asarray(getattr(loose.noise_stats, field), dtype=np.float64)
        assert_matches(candidate, reference, rtol=1e-6, err_msg=field)

    def rel_l2(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        den = float(np.linalg.norm(a))
        return float(np.linalg.norm(a - b) / den) if den else 0.0

    # The maps ride the same racing BPref atomics; their band is the loose
    # path's own repeat, measured in this process.
    for name, candidate, reference, repeat in (
        ("Ft_y", programs.Ft_y, loose.Ft_y, loose_repeat.Ft_y),
        ("Ft_ctf", programs.Ft_ctf, loose.Ft_ctf, loose_repeat.Ft_ctf),
        (
            "wsum_img_power",
            programs.noise_stats.wsum_img_power,
            loose.noise_stats.wsum_img_power,
            loose_repeat.noise_stats.wsum_img_power,
        ),
    ):
        if deterministic_reductions_enabled():
            assert_matches(
                np.asarray(candidate), np.asarray(reference), err_msg=name
            )
            continue
        band = rel_l2(reference, repeat)
        observed = rel_l2(reference, candidate)
        assert observed <= max(band, float(np.finfo(np.float32).eps)), (
            f"{name}: {observed} exceeds the repeat band {band}"
        )


@requires_resident_gpu
def test_gate_refuses_a_finite_cross_class_normalizer(_resident_production_env):
    args = _driver_fixture_args()
    args["normalization_other_score_log_z"] = np.zeros(
        args["experiment_dataset"].n_units, dtype=np.float64
    )
    with pytest.raises(NotImplementedError, match="finite cross-class"):
        rp.compute_pass2_stats_resident(**args)


@requires_resident_gpu
def test_resident_driver_refuses_an_unsupported_pass(_resident_production_env):
    """A configuration outside the gate raises instead of silently falling back."""

    args = _driver_fixture_args()
    args["relion_x_half_mstep"] = False
    with pytest.raises(NotImplementedError, match="x-half M-step"):
        rp.compute_pass2_stats_resident(**args)


def test_chunk_operands_are_padded_on_the_host():
    """Per-chunk shapes must not depend on a chunk's occupancy.

    Every traced program is keyed on its operand shapes, so an operand sized
    by the chunk's valid image count compiles once per distinct occupancy. The
    early-state cold arm traced 9352 programs in one iteration for this
    reason. The batch handed to the preparation is padded on the host instead.
    """

    import inspect

    source = inspect.getsource(rp._prepare_chunk_reconstruction_operands)
    assert "fetch_capacity_batch(" in source
    assert "_zero_padded_images" in source
    # jnp.pad with an occupancy-dependent width is what this replaced.
    assert "jnp.pad" not in source
    driver = inspect.getsource(rp)
    assert "def _pad_image_axis" not in driver


def test_capacity_batch_is_the_padded_fetch(monkeypatch):
    """The chunk fetch reads the host images in sub-batches into one reused capacity buffer
    (bigbox 14747301: fresh 1.3 GB fetches and pads were ~95 s per half of host time); the
    upload is the one-call fetch padded with its first image, as _pad_batch_to_capacity pads."""

    from types import SimpleNamespace

    stack = np.arange(7 * 3 * 3, dtype=np.float32).reshape(7, 3, 3)
    ctf = np.arange(7 * 2, dtype=np.float64).reshape(7, 2)
    calls = []

    def host_images(indices):
        calls.append(len(indices))
        return stack[np.asarray(indices)]

    dataset = SimpleNamespace(
        image_source=SimpleNamespace(host_images=host_images, tilt_series=False),
        metadata=SimpleNamespace(get_batch=lambda indices: (None, None, ctf[np.asarray(indices)])),
    )
    monkeypatch.setattr(rp, "_FETCH_SUB_BATCH_BYTES", 2 * stack[0].nbytes)
    indices = np.array([5, 1, 3, 0, 6], dtype=np.int64)
    for _ in range(2):  # the second call refills the reused buffer
        images, ctf_rows, fetched, padded = rp.fetch_capacity_batch(dataset, indices, 8)
        want = rp._pad_batch_to_capacity(stack[indices], 8)
        np.testing.assert_allclose(np.asarray(images), want, rtol=0, atol=0)
        np.testing.assert_allclose(ctf_rows, rp._pad_batch_to_capacity(ctf[indices], 8), rtol=0, atol=0)
        assert fetched.tolist() == indices.tolist()
        assert padded.tolist() == indices.tolist() + [5, 5, 5]
    assert max(calls) <= 2
    with pytest.raises(ValueError, match="cannot pad"):
        rp.fetch_capacity_batch(dataset, indices, 4)


def test_pad_batch_to_capacity_repeats_the_first_row():
    batch = np.arange(12, dtype=np.float32).reshape(3, 4)
    padded = rp._pad_batch_to_capacity(batch, 5)
    assert padded.shape == (5, 4)
    assert_matches(padded[:3], batch)
    assert_matches(padded[3], batch[0])
    assert_matches(padded[4], batch[0])
    assert_matches(rp._pad_batch_to_capacity(batch, 3), batch)
    with pytest.raises(ValueError, match="cannot pad"):
        rp._pad_batch_to_capacity(batch, 2)


def test_zero_padded_images_clears_only_the_padded_slots():
    values = np.arange(24, dtype=np.float32).reshape(4, 3, 2)
    valid = np.asarray([True, True, False, False])
    out = np.asarray(rp._zero_padded_images(jnp.asarray(values), jnp.asarray(valid)))
    assert_matches(out[:2], values[:2])
    assert_matches(out[2:], np.zeros_like(values[2:]))


def test_reorder_permutation_inverts_a_shuffled_fetch():
    requested = np.asarray([7, 3, 9, 1])
    fetched = np.asarray([1, 9, 7, 3])
    order = rp._reorder_permutation(fetched, requested, capacity=6)
    assert order.shape == (6,)
    # Position p of the table must read fetched slot order[p].
    assert_matches(fetched[order[:4]], requested)
    with pytest.raises(ValueError, match="did not return every requested image"):
        rp._reorder_permutation(np.asarray([1, 9, 7, 7]), requested, capacity=6)


def test_dispatcher_sends_every_pass_to_the_resident_engine(monkeypatch):
    """The resident driver is relax's one pass-2 engine: dispatch has no other route."""

    from relax.sparse_pass2 import dispatch as sparse_dispatch
    from relax.sparse_pass2.engine_record import take_pass_engines

    calls = []
    monkeypatch.setattr(rp, "compute_pass2_stats_resident", lambda *a, **k: calls.append(k) or "resident")
    args = _driver_fixture_args()
    args.setdefault("mean_variance", None)
    take_pass_engines()
    assert sparse_dispatch.compute_pass2_stats_sparse(**args) == "resident"
    assert len(calls) == 1
    assert take_pass_engines() == ["global:resident"]


def test_gate_still_raises_for_in_scope_mismatches():
    """Everything that is not an out-of-scope scoring mode still raises."""

    with pytest.raises(NotImplementedError, match="x-half M-step"):
        rp.require_resident_production_configuration(
            **_production_gate_kwargs(relion_x_half_mstep=False)
        )
    with pytest.raises(NotImplementedError, match="float64 scoring"):
        rp.require_resident_production_configuration(
            **_production_gate_kwargs(use_float64_scoring=True)
        )


def test_streamed_chunk_projections_gather_the_cached_arrays():
    """Chunk-local slots gather exactly what the fine-grid cache gathers by id.

    The per-iteration cache does not fit at healpix order 3 on real data
    (10097 it13: 40 GiB > 19.9 GiB); streaming projects each chunk's distinct
    rotations and re-indexes its rows, and must not change a gathered value.
    """

    n_fine, n_pix, row_capacity = 50, 6, 16
    rng = np.random.default_rng(0)
    fine_grid = jnp.asarray(rng.normal(size=(n_fine, 3, 3)), dtype=jnp.float32)
    mstep_grid = fine_grid * 2.0
    coarse_parent = jnp.asarray(np.arange(n_fine) // 8, dtype=jnp.int32)

    def project(rotations):
        flat = jnp.asarray(rotations).reshape(rotations.shape[0], -1)[:, :n_pix]
        return (flat.astype(jnp.complex64), (flat + 1.0).astype(jnp.complex64), flat * flat)

    cache = project(fine_grid)
    host_ids = np.asarray([7, 3, 7, 41, 3, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    n_valid = 6
    rows = rp._ChunkRowArrays(
        row_image_local=jnp.zeros(row_capacity, jnp.int32),
        row_fine_rot=jnp.asarray(host_ids),
        row_log_prior=jnp.zeros(row_capacity, jnp.float32),
        row_mask_bits=jnp.zeros(row_capacity, jnp.uint32),
        row_mask_mode=jnp.zeros(row_capacity, jnp.int8),
        image_ids=jnp.zeros(2, jnp.int32),
        n_valid_rows=jnp.int32(n_valid),
        n_valid_images=jnp.int32(1),
        segment_offsets=jnp.zeros(3, jnp.int32),
        image_row_start=jnp.zeros(2, jnp.int64),
        image_row_count=jnp.zeros(2, jnp.int64),
    )
    new_rows, slot_ids, caches, mstep_local, parent_local = rp._stream_chunk_projections(
        rows,
        host_ids,
        n_valid_rows=n_valid,
        row_capacity=row_capacity,
        project=lambda ids, n_rows: tuple(
            jnp.pad(values, [(0, n_rows - values.shape[0])] + [(0, 0)] * (values.ndim - 1))
            for values in project(fine_grid[ids])
        ),
        n_fine_rot=n_fine,
        mstep_grid=mstep_grid,
        coarse_parent_grid=coarse_parent,
    )
    slots = np.asarray(new_rows.row_fine_rot)[:n_valid]
    assert_matches(np.asarray(slot_ids)[slots], host_ids[:n_valid])
    for local, full in zip(caches, cache):
        assert local.shape[0] == row_capacity
        assert_matches(np.asarray(local)[slots], np.asarray(full)[host_ids[:n_valid]])
    assert_matches(np.asarray(mstep_local)[slots], np.asarray(mstep_grid)[host_ids[:n_valid]])
    assert_matches(np.asarray(parent_local)[slots], np.asarray(coarse_parent)[host_ids[:n_valid]])
    # padded rows read slot 0, a real rotation
    assert np.all(np.asarray(new_rows.row_fine_rot)[n_valid:] == 0)


def test_streamed_row_ladder_keeps_capacities_whose_cache_fits():
    assert rp._stream_row_capacity_ladder(
        (8192, 32768, 131072), bytes_per_rotation=300e3, max_projection_bytes=20 * 1024**3
    ) == (8192, 32768)
    # A configuration refusal: an error, since there is no other pass-2 engine.
    with pytest.raises(ResidentConfigurationUnsupported, match="smallest row capacity"):
        rp._stream_row_capacity_ladder(
            (8192,), bytes_per_rotation=10e6, max_projection_bytes=20 * 1024**3
        )


def test_cached_row_ladder_bounds_the_gathered_chunk():
    """10202 it3 (14434683): box 800 at current size 304 scores 36514 pixels, so a
    131072-row chunk gathered 35.70 GiB of cached projections and ran the H100 out
    of memory; capacities whose gathered block exceeds the budget are dropped."""

    row_bytes = 36514 * 8
    gib = 1024**3
    assert rp._cached_row_capacity_ladder(
        (8192, 32768, 131072), bytes_per_row=row_bytes, max_gather_bytes=30 * gib
    ) == (8192, 32768)
    # A 256-pixel-class workload keeps the whole ladder.
    assert rp._cached_row_capacity_ladder(
        (8192, 32768, 131072), bytes_per_row=6000 * 8, max_gather_bytes=30 * gib
    ) == (8192, 32768, 131072)
    # A named configuration refusal (there is no other pass-2 engine), not a crash.
    with pytest.raises(rp.ResidentConfigurationUnsupported, match="smallest row capacity"):
        rp._cached_row_capacity_ladder((8192,), bytes_per_row=row_bytes, max_gather_bytes=1 * gib)


def test_stream_slot_count_quantises_and_caps():
    assert rp._stream_slot_count(1, 131072) == rp._STREAM_SLOT_QUANTUM
    assert rp._stream_slot_count(8193, 131072) == 2 * rp._STREAM_SLOT_QUANTUM
    assert rp._stream_slot_count(200000, 32768) == 32768


@requires_resident_gpu
def test_streamed_projections_match_the_cached_pass(_resident_production_env, monkeypatch):
    """Per-chunk streamed projections change no discrete state and stay in the repeat band.

    The streamed pass gathers the same projection arrays through chunk-local
    slots, so the discrete winners are identical; float outputs are held to the
    band :func:`test_resident_driver_repeats_itself` measures.
    """

    args = _driver_fixture_args()
    cached = rp.compute_pass2_stats_resident(**args)
    monkeypatch.setattr(rp, "_projection_cache_fits_budget", lambda *a, **k: False)
    streamed = rp.compute_pass2_stats_resident(**args)

    # Discrete state: integer indices, compared exactly.
    assert_matches(cached.hard_assignment, streamed.hard_assignment)
    assert_matches(cached.best_rotation_indices, streamed.best_rotation_indices)
    # Float outputs are held to measured bands, never bitwise (user rule, 2026-09-24).
    np.testing.assert_allclose(cached.best_rotations, streamed.best_rotations, rtol=0, atol=1e-6)
    np.testing.assert_allclose(cached.best_translations, streamed.best_translations, rtol=0, atol=1e-6)

    def rel_l2(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        den = float(np.linalg.norm(a))
        return float(np.linalg.norm(a - b) / den) if den else 0.0

    for field in ("wsum_sigma2_noise", "wsum_norm_correction"):
        assert rel_l2(
            getattr(cached.noise_stats, field), getattr(streamed.noise_stats, field)
        ) < 1e-7, field
    # Derived from the float32 BPref atomics (user approval 2026-09-27): a run
    # measured 1.058e-7 against the former 1e-7 (bigbox 14558037).
    assert rel_l2(cached.Ft_y, streamed.Ft_y) < _float32_atomic_order_bound(args)
    assert rel_l2(cached.Ft_ctf, streamed.Ft_ctf) < 1e-7
    assert rel_l2(cached.noise_stats.wsum_img_power, streamed.noise_stats.wsum_img_power) < 1e-7


def test_resident_operand_reservation_and_admission_share_one_predicate(monkeypatch):
    """Operands are reserved only when they would be admitted: half of what the allocator can hand out."""

    monkeypatch.delenv("RELAX_SPARSE_PASS2_RESIDENT_OPERAND_MAX_BYTES", raising=False)
    gib = 1024**3
    assert rp._resident_operands_fit(10 * gib, 20 * gib)
    assert not rp._resident_operands_fit(10 * gib + 1, 20 * gib)
    # EMPIAR-10202 iteration 2 (box 800): 50.9 GiB of operands against a 51 GiB reading stay per chunk.
    assert not rp._resident_operands_fit(int(50.9 * gib), 51 * gib)
    # An unknown reading falls back to the fixed budget.
    assert rp._resident_operands_fit(6 * gib, None)
    assert not rp._resident_operands_fit(6 * gib + 1, None)


def test_stream_projection_budget_is_capped_by_measured_free_memory():
    gib = 1024**3
    budget = rp._stream_projection_budget_bytes
    # Unknown readings do not cap.
    assert budget(20 * gib, physical_free_bytes=None, allocator_free_bytes=None) == 20 * gib
    # Allocator headroom caps; the device bound is physical free plus the pool's unused bytes.
    assert budget(
        20 * gib, physical_free_bytes=30 * gib, allocator_free_bytes=60 * gib, pool_free_bytes=0
    ) == 15 * gib
    assert budget(
        20 * gib, physical_free_bytes=70 * gib, allocator_free_bytes=8 * gib, pool_free_bytes=0
    ) == 4 * gib
    # Without a pool reading the physical reading bounds only when the allocator reports nothing.
    assert budget(20 * gib, physical_free_bytes=30 * gib, allocator_free_bytes=None) == 15 * gib
    assert budget(20 * gib, physical_free_bytes=2 * gib, allocator_free_bytes=60 * gib) == 20 * gib
    # The half's resident operands are reserved before the fraction is taken.
    assert budget(
        20 * gib, physical_free_bytes=30 * gib, allocator_free_bytes=None, reserved_bytes=10 * gib
    ) == 10 * gib


def test_stream_projection_budget_counts_the_allocator_pool_as_free():
    """10097 it13 in a full run (14400302): the grown pool left 13.37 GiB physically
    free against 70.04 GiB of allocator headroom, and the physical reading alone
    set a zero budget. The pool's unused bytes are allocatable, so the budget is
    the cache share again."""

    gib = 1024**3
    kwargs = dict(physical_free_bytes=int(13.37 * gib), allocator_free_bytes=int(70.04 * gib))
    share = int(19.91 * gib)
    assert rp._stream_projection_budget_bytes(
        share, pool_free_bytes=int(56.0 * gib), reserved_bytes=int(14.0 * gib), **kwargs
    ) == share
    # A pool that holds little unused memory still bounds it through the device.
    assert rp._stream_projection_budget_bytes(
        share, pool_free_bytes=int(1.0 * gib), reserved_bytes=int(4.0 * gib), **kwargs
    ) == int((14.37 - 4.0) * gib * rp._STREAM_FREE_MEMORY_FRACTION)


def test_streamed_row_ladder_counts_the_padded_copy():
    """10097 it13 (14397073): capacity 131072 at 122 KiB per rotation fit the
    19.9 GiB budget once but not with its padded copy alive; it is now excluded."""

    per_rotation = 122.3 * 1024
    kept = rp._stream_row_capacity_ladder(
        (8192, 32768, 131072), bytes_per_rotation=per_rotation, max_projection_bytes=19.91 * 1024**3
    )
    assert kept == (8192, 32768)


@requires_resident_gpu
def test_streamed_projection_rows_equal_the_cached_rows_exactly(
    _resident_production_env, monkeypatch
):
    """Every on-the-fly projection row matches the fine-grid cache row (float32-tight band).

    Records each call of the projection block: the cached pass makes one call
    over the whole fine grid (its union window of the score and reconstruction
    windows), the streamed pass one per chunk over its distinct rotations (the
    three chunk-local caches). Each streamed row is matched to its fine-grid row
    by the exact rotation matrix and compared in all three outputs, the cached
    ones taken out of the union row as the chunk programs take them.
    """

    calls = []
    original = rp._compute_sparse_pass2_windowed_projections_block

    def recording(volume, rotations, *args, **kwargs):
        out = original(volume, rotations, *args, **kwargs)
        windows = tuple(
            None if kwargs.get(key) is None else np.asarray(kwargs[key])
            for key in ("score_indices", "recon_indices")
        )
        calls.append((np.asarray(rotations), windows, tuple(None if v is None else np.asarray(v) for v in out)))
        return out

    monkeypatch.setattr(rp, "_compute_sparse_pass2_windowed_projections_block", recording)
    args = _driver_fixture_args()
    rp.compute_pass2_stats_resident(**args)
    assert len(calls) == 1
    grid, (union_indices, no_recon), (union_rows, _, _) = calls.pop()
    assert no_recon is None
    index = {row.tobytes(): i for i, row in enumerate(grid.reshape(grid.shape[0], -1))}

    monkeypatch.setattr(rp, "_projection_cache_fits_budget", lambda *a, **k: False)
    rp.compute_pass2_stats_resident(**args)
    assert calls, "the streamed pass made no projection call"
    compared = 0
    for rotations, (score_indices, recon_indices), streamed in calls:
        ids = np.asarray(
            [index[row.tobytes()] for row in rotations.reshape(rotations.shape[0], -1)]
        )
        rows = union_rows[ids]
        cached_recon = rows[:, np.searchsorted(union_indices, recon_indices)]
        cached = (
            rows[:, np.searchsorted(union_indices, score_indices)],
            cached_recon,
            (np.abs(cached_recon) ** 2).astype(np.float32),
        )
        for name, full, part in zip(("score", "recon", "recon_abs2"), cached, streamed):
            # The same projection gathered by another index; measured equal, held
            # to a float32-tight band rather than bitwise (user rule, 2026-09-24).
            np.testing.assert_allclose(part, full, rtol=1e-6, atol=0, err_msg=name)
        compared += ids.size
    assert compared > 0


def test_stable_window_plan_applies_only_to_the_supported_passes(monkeypatch, caplog):
    """The flag is on by default: a sub-box model-grid pass gets a physical class above its size."""

    kwargs = dict(
        current_size=36,
        mstep_current_size=36,
        n_half=64 * 33,
        square_window=False,
        window_spec_kwargs={"window_at_box": True},
        firstiter_cc=False,
        reconstruction_image_radius=None,
        reconstruction_volume_current_size=None,
    )
    monkeypatch.setenv(rp._RESIDENT_STABLE_WINDOWS_ENV, "0")
    assert rp._resident_stable_window_plan((64, 64), **kwargs) is None
    monkeypatch.delenv(rp._RESIDENT_STABLE_WINDOWS_ENV, raising=False)
    plan = rp._resident_stable_window_plan((64, 64), **kwargs)
    assert plan.logical_current_size == 36 and plan.physical_current_size == 40
    assert plan.physical_reconstruction_pixels > plan.logical_reconstruction_pixels
    monkeypatch.setenv(rp._RESIDENT_STABLE_WINDOWS_ENV, "1")
    for override in (
        {"current_size": 64, "mstep_current_size": 64, "n_half": 64 * 33},
        {"firstiter_cc": True},
        {"mstep_current_size": 34},
        {"reconstruction_volume_current_size": 36},
    ):
        with caplog.at_level("INFO"):
            assert rp._resident_stable_window_plan((64, 64), **{**kwargs, **override}) is None
    assert "not applicable" in caplog.text


@requires_resident_gpu
def test_stable_windows_match_the_logical_window(_resident_production_env, monkeypatch):
    """Inside a physical window class the resident driver reproduces RELION's logical window.

    The fixture box is 8 pixels, so current size 4 gets the physical class 6.
    Discrete state is exact and the per-image scores are in the default band
    (the runtime scorer stops at the logical rectangle). The maps, noise and
    scale sums may change reduction order over the longer pixel axis; float
    results are never required to be bitwise (user rule), so the maps take this
    file's resident-vs-compact band and the sums the driver's repeat band.
    """

    args = {**_driver_fixture_args(), "current_size": 4}
    monkeypatch.setenv(rp._RESIDENT_STABLE_WINDOWS_ENV, "0")
    logical = rp.compute_pass2_stats_resident(**args)
    monkeypatch.setenv(rp._RESIDENT_STABLE_WINDOWS_ENV, "1")
    stable = rp.compute_pass2_stats_resident(**args)

    assert_matches(logical.hard_assignment, stable.hard_assignment)
    assert_matches(logical.best_rotation_indices, stable.best_rotation_indices)
    for field in (
        "log_evidence_per_image",
        "best_log_score_per_image",
        "max_posterior_per_image",
        "rotation_posterior_sums",
    ):
        assert_matches(
            np.asarray(getattr(logical.relion_stats, field)),
            np.asarray(getattr(stable.relion_stats, field)),
            err_msg=field,
        )

    def rel_l2(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        den = float(np.linalg.norm(a))
        return float(np.linalg.norm(a - b) / den) if den else 0.0

    # The BPref atomics add the tail's zeros in another order: measured 1.2e-7
    # (Ft_y) and 1.1e-7 (Ft_ctf) on an H100 (job 14460993), the same size as the
    # resident-vs-compact map difference this file bounds at 1e-6.
    assert rel_l2(logical.Ft_y, stable.Ft_y) < 1e-6
    assert rel_l2(logical.Ft_ctf, stable.Ft_ctf) < 1e-6
    # The noise, norm and scale sums measured 0 there; held to the repeat band.
    for field in (
        "wsum_sigma2_noise",
        "wsum_img_power",
        "wsum_norm_correction",
        "wsum_scale_correction_xa",
        "wsum_scale_correction_aa",
    ):
        assert rel_l2(getattr(logical.noise_stats, field), getattr(stable.noise_stats, field)) < 1e-7, field


def test_stream_parts_gather_rows_from_the_padded_class_calls():
    """K>1 streamed projections: each id reads its row of its class's whole padded call."""

    rng = np.random.default_rng(3)
    parts = tuple(
        tuple(jnp.asarray(rng.normal(size=(8, 5)).astype(np.float32)) for _ in range(3)) for _ in range(2)
    )
    # ids of classes [1, 0, 1, 0, 0]: class 0's rows 0-2 of call 0, class 1's rows 0-1 of call 1.
    rows = np.array([8 + 0, 0, 8 + 1, 1, 2], dtype=np.int32)
    gathered = rp._gather_stream_parts(parts, jnp.asarray(rows), n_rows=8)
    for field in range(3):
        expected = np.concatenate([np.asarray(parts[0][field]), np.asarray(parts[1][field])])[rows]
        assert gathered[field].shape == (8, 5)
        assert_matches(np.asarray(gathered[field])[: rows.size], expected)
        assert not np.any(np.asarray(gathered[field])[rows.size :])


def test_accumulators_exist_before_the_chunk_budget_is_read():
    """The x-half accumulators (15.35 + 7.7 GiB at EMPIAR-10202's full box) are allocated
    before either pass reads free device memory for its chunk budget, so the budget counts
    them; the local pass allocated them after its plan."""

    import inspect

    from relax.sparse_pass2 import resident_local_pass2 as rlp

    for source in (inspect.getsource(rp._resident_pass2), inspect.getsource(rlp.compute_local_search_resident)):
        first_budget_read = source.index("resident_chunk_budget_bytes(")
        assert source.index("Ft_y_total = ") < first_budget_read
        assert source.index("Ft_ctf_total = ") < first_budget_read


def test_pass_headroom_asks_for_the_accumulators_and_the_smallest_chunks_rows(monkeypatch):
    """Team-lead 2026-09-27 (a): before a pass allocates its accumulators, the CTF row
    caches yield unless the accumulators and the smallest row class's projections fit."""

    from relax.relion import relion_ctf
    from relax.sparse_pass2 import resident_pass2
    from relax.sparse_pass2.resident_scoring import resident_row_projection_bytes

    asked = []
    monkeypatch.setattr(relion_ctf, "ensure_device_headroom", lambda n_bytes: asked.append(n_bytes) or False)
    accumulators = resident_pass2.resident_accumulator_bytes(1000, np.complex64, np.float32, n_slots=2)
    assert accumulators == 2 * 1000 * 12
    assert not resident_pass2.ensure_pass_headroom(
        accumulators, min_row_capacity=1024, n_score_pixels=30, n_recon_pixels=20
    )
    assert asked == [accumulators + 1024 * resident_row_projection_bytes(n_score_pixels=30, n_recon_pixels=20)]


def test_plan_counts_a_lone_overflow_chunk():
    """bigbox 14607861: the EMPIAR-10202 final pass planned 1024-row chunks (16.05 of a
    16.16 GiB budget) and ran out of memory projecting a 2048-row one-image chunk next to
    the previous chunk. The chunk loops now run an overflow chunk alone, and the plan
    counts it; past three times the largest row class it is the plan's peak."""

    from types import SimpleNamespace

    row_offsets = np.array([0, 300, 2800, 3183, 3483], dtype=np.int64)
    tables = SimpleNamespace(n_images=4, row_offsets=row_offsets)
    chunks = plan_capacity_chunks(tables, row_capacity_ladder=(1024,), image_capacity_ladder=(2, 4))
    overflow = [chunk for chunk in chunks if rp.chunk_runs_alone(chunk, (1024,))]
    assert [(c.image_start, c.row_capacity, c.image_capacity) for c in overflow] == [(1, 3072, 2)]
    assert rp.max_image_rows(row_offsets) == 2500
    assert rp.overflow_row_capacity(2500, (1024,)) == 3072
    assert rp.overflow_row_capacity(1024, (1024,)) == 0

    t, p = 36, 1000
    kwargs = dict(
        row_capacity_ladder=(1024,), image_capacity_ladder=(2, 4), mstep_block_rows=32, row_bytes=10 * p,
        n_fine_trans=t, n_recon_pixels=p, budget_bytes=None, pipelined=True,
    )
    lone = rp.resident_chunk_bytes(
        row_capacity=3072, image_capacity=2, mstep_block_rows=32, row_bytes=10 * p, n_fine_trans=t,
        n_recon_pixels=p,
    )
    regular = rp.plan_resident_chunk_memory(**kwargs)
    assert regular.peak_bytes < lone
    assert rp.plan_resident_chunk_memory(**kwargs, max_image_rows=2500).peak_bytes == lone

    # Over budget, the lone chunk shrinks its M-step block; its rows cannot shrink.
    fitted = rp.plan_resident_chunk_memory(**{**kwargs, "budget_bytes": lone - 1}, max_image_rows=2500)
    assert fitted.mstep_block_rows < 32
    assert fitted.peak_bytes <= lone - 1
    with pytest.raises(ResidentConfigurationUnsupported):
        rp.plan_resident_chunk_memory(**{**kwargs, "budget_bytes": 3072 * 10 * p}, max_image_rows=2500)


def test_plan_counts_a_row_blocked_lone_chunk_by_its_posterior_cells():
    """bench 14641043: a flat-posterior image kept all 294,912 fine rotations. The global
    pass scores and reconstructs such an image in blocks of its largest row class, so the
    plan counts a largest-class chunk plus the posterior cells of every row, not every
    row's projections (147 GiB there)."""

    t, p = 84, 1000
    kwargs = dict(
        row_capacity_ladder=(8192, 32768), image_capacity_ladder=(32, 128), mstep_block_rows=512,
        row_bytes=96 * 1024, n_fine_trans=t, n_recon_pixels=p, budget_bytes=None, pipelined=True,
    )
    lone_row_bytes = rp.lone_chunk_row_bytes(t)
    plan = rp.plan_resident_chunk_memory(**kwargs, max_image_rows=294912, lone_row_bytes=lone_row_bytes)
    block = rp.resident_chunk_bytes(
        row_capacity=32768, image_capacity=32, mstep_block_rows=512, row_bytes=96 * 1024, n_fine_trans=t,
        n_recon_pixels=p,
    )
    regular = rp.plan_resident_chunk_memory(**kwargs)
    assert plan.peak_bytes == max(regular.peak_bytes, block + 294912 * lone_row_bytes)
    assert plan.peak_bytes < rp.plan_resident_chunk_memory(**kwargs, max_image_rows=294912).peak_bytes / 2


@requires_resident_gpu
@pytest.mark.parametrize("streamed", [False, True], ids=["cached", "streamed"])
def test_lone_overflow_chunks_match_the_whole_chunk_pass(_resident_production_env, monkeypatch, streamed):
    """An image past the largest row class is scored and reconstructed in row blocks.

    With a 16-row ladder most images of the fixture overflow and run alone; the scores
    are the one-call chunk's, so the discrete state is identical, and the floats differ
    only by the changed grouping of the M-step blocks and the per-chunk sums.
    """

    args = _driver_fixture_args()
    if streamed:
        monkeypatch.setattr(rp, "_projection_cache_fits_budget", lambda *a, **k: False)
    whole = rp.compute_pass2_stats_resident(**args)
    lone_calls = []
    real_lone = rp._run_lone_resident_chunk

    def spy(*a, **k):
        lone_calls.append(int(k["spec"].row_capacity))
        return real_lone(*a, **k)

    monkeypatch.setattr(rp, "_run_lone_resident_chunk", spy)
    monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_ROW_CAPACITIES", "16")
    monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_MSTEP_BLOCK_ROWS", "16")
    lone = rp.compute_pass2_stats_resident(**args)
    assert lone_calls, "the 16-row ladder made no lone chunk"

    assert_matches(whole.hard_assignment, lone.hard_assignment)
    assert_matches(whole.best_rotation_indices, lone.best_rotation_indices)
    np.testing.assert_allclose(whole.best_rotations, lone.best_rotations, rtol=0, atol=1e-6)
    np.testing.assert_allclose(whole.best_translations, lone.best_translations, rtol=0, atol=1e-6)

    def rel_l2(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        den = float(np.linalg.norm(a))
        return float(np.linalg.norm(a - b) / den) if den else 0.0

    bound = _float32_atomic_order_bound(args)
    diffs = {
        "Ft_y": rel_l2(whole.Ft_y, lone.Ft_y),
        "Ft_ctf": rel_l2(whole.Ft_ctf, lone.Ft_ctf),
        **{
            field: rel_l2(getattr(whole.noise_stats, field), getattr(lone.noise_stats, field))
            for field in ("wsum_sigma2_noise", "wsum_norm_correction", "wsum_img_power")
        },
    }
    print("lone vs whole rel_l2:", diffs, "bound", bound)
    assert all(value < bound for value in diffs.values()), diffs


def test_projection_cache_allocation_refusal_streams_and_other_errors_raise(monkeypatch):
    """The union cache is allocated before it is built: an allocator refusal returns None
    (EMPIAR-10345 it13, bench 14643272), so the pass streams; any other error is raised."""

    cache = rp._allocate_projection_cache(3, 5, np.complex64)
    assert cache.shape == (3, 5) and cache.dtype == np.complex64

    def refuse(*a, **k):
        raise RuntimeError("RESOURCE_EXHAUSTED: Out of memory while trying to allocate 20.19GiB.")

    monkeypatch.setattr(rp.jnp, "zeros", refuse)
    assert rp._allocate_projection_cache(3, 5, np.complex64) is None

    def broken(*a, **k):
        raise RuntimeError("INTERNAL: something else")

    monkeypatch.setattr(rp.jnp, "zeros", broken)
    with pytest.raises(RuntimeError, match="INTERNAL"):
        rp._allocate_projection_cache(3, 5, np.complex64)


def test_projection_cache_row_blocks_gather_the_one_array_rows(monkeypatch):
    """Class3D K4 100k from iteration 19 (bench 14719385): a 6.32 GiB cache against a 5.25 GiB
    largest block. A cache the allocator refuses whole is allocated in row blocks, built in
    place across them, and gathered row for row as the one array."""

    from relax.sparse_pass2.resident_scoring import cache_dtype, cache_rows

    n_rows, n_pixels = 10, 4
    real_allocate = rp._allocate_projection_cache
    monkeypatch.setattr(
        rp, "_allocate_projection_cache",
        lambda rows, pixels, dtype: None if rows > 3 else real_allocate(rows, pixels, dtype),
    )
    blocks = rp._allocate_projection_cache_blocks(n_rows, n_pixels, np.complex64)
    assert isinstance(blocks, tuple) and len(blocks) == 4 and all(b.shape == (3, n_pixels) for b in blocks)
    rng = np.random.default_rng(5)
    full = (rng.standard_normal((n_rows, n_pixels)) + 1j * rng.standard_normal((n_rows, n_pixels))).astype(np.complex64)

    def project_rows(class_index, start, stop):
        return jnp.asarray(full[class_index * 5 + start : class_index * 5 + stop])

    built = rp.build_projection_cache_in_place(
        project_rows, n_classes=2, n_rows_per_class=5, n_pixels=n_pixels, rows_per_call=2,
        dtype=np.complex64, cache=blocks,
    )
    ids = np.array([0, 9, 3, 4, 7, 2, 2, 5], dtype=np.int32)
    np.testing.assert_allclose(np.asarray(cache_rows(built, ids)), full[ids], rtol=0, atol=0)
    assert cache_dtype(built) == np.complex64
    whole = rp.build_projection_cache_in_place(
        project_rows, n_classes=2, n_rows_per_class=5, n_pixels=n_pixels, rows_per_call=2, dtype=np.complex64,
    )
    np.testing.assert_allclose(np.asarray(cache_rows(whole, ids)), full[ids], rtol=0, atol=0)
    monkeypatch.setattr(rp, "_allocate_projection_cache", lambda rows, pixels, dtype: None)
    assert rp._allocate_projection_cache_blocks(n_rows, n_pixels, np.complex64) is None


@requires_resident_gpu
def test_row_blocked_projection_cache_matches_the_one_array_pass(_resident_production_env, monkeypatch):
    """The global pass with its union cache in row blocks gives the one-array pass."""

    args = _driver_fixture_args()
    whole = rp.compute_pass2_stats_resident(**args)
    real_allocate = rp._allocate_projection_cache
    heights = []

    def refuse_whole(rows, pixels, dtype):
        heights.append(int(rows))
        return None if len(heights) == 1 else real_allocate(rows, pixels, dtype)

    monkeypatch.setattr(rp, "_allocate_projection_cache", refuse_whole)
    blocked = rp.compute_pass2_stats_resident(**args)
    assert len(heights) >= 3, "the pass must allocate its cache in row blocks"

    assert_matches(whole.hard_assignment, blocked.hard_assignment)
    assert_matches(whole.best_rotation_indices, blocked.best_rotation_indices)

    def rel_l2(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        den = float(np.linalg.norm(a))
        return float(np.linalg.norm(a - b) / den) if den else 0.0

    bound = _float32_atomic_order_bound(args)
    assert rel_l2(whole.Ft_y, blocked.Ft_y) < bound
    assert rel_l2(whole.Ft_ctf, blocked.Ft_ctf) < bound
    for field in ("wsum_sigma2_noise", "wsum_norm_correction", "wsum_img_power"):
        assert rel_l2(getattr(whole.noise_stats, field), getattr(blocked.noise_stats, field)) < bound, field


def test_only_host_pixel_indices_are_validated_per_call():
    """Device indices are built valid (projection_window_union); reading them back to validate
    synchronized every projector call of the resident passes (bigbox 14747301)."""

    from relax.helpers.projection import _host_pixel_indices

    assert _host_pixel_indices(np.arange(4, dtype=np.int32))
    assert not _host_pixel_indices(jnp.arange(4, dtype=jnp.int32))
