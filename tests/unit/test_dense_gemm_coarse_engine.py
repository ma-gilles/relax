"""Small-grid checks for the exact joint-class dense GEMM core."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from types import SimpleNamespace
from typing import NamedTuple

from relax.dense.gemm_experiment import (
    DenseGemmTileConfig,
    make_joint_k_batch_program,
    pad_batch,
    pad_grid,
)
from relax.dense.gemm_experiment_kernels import score_tile
from relax.dense.gemm_coarse_statistics import (
    DenseGemmStatisticsPlan,
    initial_statistics_carry,
    split_normalizer_metadata,
)
from relax.sparse_pass2.resident_statistics import ResidentStatisticsConfig
from helpers.float_compare import assert_matches


@pytest.mark.unit
@pytest.mark.parametrize("exact_bpref", [False, True])
def test_gemm_reconstruction_operands_follow_resident_preparation(exact_bpref):
    from relax.dense.gemm_coarse_engine import _dense_reconstruction_operands
    from relax.sparse_pass2.resident_operands import _BatchWindowInputs, _batch_window_operands

    raw = jnp.asarray([[1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j],
                       [2 + 1j, 4 + 3j, 6 + 5j, 8 + 7j],
                       [9 + 0j, 9 + 0j, 9 + 0j, 9 + 0j]], jnp.complex64)
    weighted = raw * jnp.asarray([[1.2, 0.8, 1.5, 0.3]], jnp.float32)
    weighted_ctf = jnp.asarray([[0.25, 0.5, 0.75, 1.25]] * 3, jnp.float32)
    ctf2 = jnp.asarray([[0.2, 0.4, 0.6, 0.8]] * 3, jnp.float32)
    recon_indices = jnp.asarray([1, 2, 3], jnp.int32)
    prepared = _batch_window_operands(
        _BatchWindowInputs(
            ctf2_over_nv_half=ctf2, sparse_score_input_half=raw,
            processed_score_half_for_noise=raw,
            recon_input_half=raw if exact_bpref else weighted,
            weighted_ctf_half=weighted_ctf if exact_bpref else None,
            score_weighted_half=weighted, ctf2_over_nv_recon_half=ctf2,
            ctf_half_rfloat=weighted_ctf.astype(jnp.float64) if exact_bpref else None,
            dc_mask=None, score_indices=jnp.asarray([0, 1], jnp.int32),
            recon_indices=recon_indices, rect_indices=jnp.arange(4, dtype=jnp.int32),
        ),
        mask_dc=False, score_real_dtype=jnp.float32,
        score_complex_dtype=jnp.complex64, acc_real_dtype=jnp.float32,
    )
    resident = SimpleNamespace(
        recon_image=prepared["recon_image"],
        recon_weight=prepared.get("recon_weight"),
        direct_ctf_rfloat_recon=prepared.get("direct_ctf_rfloat_recon"),
        ctf2_over_nv_recon=prepared["ctf2_over_nv_recon"],
    )
    numerator_image, denominator = _dense_reconstruction_operands(
        resident, exact_bpref=exact_bpref, n_images=2,
    )
    expected = ((raw[:, recon_indices] * weighted_ctf[:, recon_indices])
                if exact_bpref else weighted[:, recon_indices])[:2]
    assert_matches(numerator_image, expected)
    assert_matches(denominator, ctf2[:2, recon_indices])
    with pytest.raises(ValueError, match="presence disagrees"):
        _dense_reconstruction_operands(resident, exact_bpref=not exact_bpref, n_images=2)


@pytest.mark.unit
@pytest.mark.parametrize("relion_order", [False, True])
def test_complete_grid_local_slots_match_actual_resident_table(relion_order):
    from relax.dense.gemm_coarse_engine import _full_grid_local_rotation_slots
    from relax.sparse_pass2.resident_significance import (
        CoarseSignificanceCSR, build_resident_candidate_tables_from_csr,
    )

    n_coarse, n_trans = 18, 3  # whole psi rows at HEALPix level 0
    parent = np.repeat(np.arange(n_coarse), 1 + (np.arange(n_coarse) % 3 == 0))
    parent = parent[np.random.default_rng(7).permutation(len(parent))]
    state = SimpleNamespace(
        n_coarse_rot=n_coarse, nside_level=0, oversampling_order=1,
        random_perturbation=0.0, relion_parent_execution_order=relion_order,
        fine_rotation_parent=parent, fine_rotations=np.zeros((len(parent), 3, 3), np.float32),
        symmetry_label="C1",
    )
    inverse = _full_grid_local_rotation_slots(state)
    csr = CoarseSignificanceCSR(
        n_images=2, n_coarse_rot=n_coarse, n_coarse_trans=n_trans,
        offsets=np.zeros(3, np.int64), ids=np.zeros(0, np.int32),
        store_excluded=np.ones(2, bool), n_significant=np.full(2, n_coarse * n_trans, np.int32),
    )
    table = build_resident_candidate_tables_from_csr(
        csr, nside_level=0, oversampling_order=1, n_fine_trans=2 * n_trans,
        fine_translation_parent=np.repeat(np.arange(n_trans), 2),
        rotation_log_prior=None, random_perturbation=0.0,
        fine_rotation_parent_override=parent, relion_parent_execution_order=relion_order,
        dtype=np.float32,
    )
    for image in range(2):
        rows = table.row_fine_rot[table.row_offsets[image]:table.row_offsets[image + 1]]
        np.testing.assert_array_equal(inverse[rows], np.arange(len(parent)))


@pytest.mark.unit
@pytest.mark.parametrize("n_classes,symmetry", [(1, "C1"), (4, "C2")])
@pytest.mark.parametrize("configured_native", [False, True])
def test_public_adaptive_dense_dispatch_uses_complete_grid(n_classes, symmetry, configured_native, monkeypatch):
    from relax.classification import k_class
    from relax.sparse_pass2.engine_record import take_coarse_engine_calls

    class Returned(NamedTuple):
        significant_counts: object = None

    seen = []

    def fake_fine(*args, **kwargs):
        seen.append((args, kwargs))
        return Returned()

    monkeypatch.setattr(k_class, "_run_sparse_k_class_adaptive_pass2", fake_fine)
    take_coarse_engine_calls()
    rot = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
    fine_rot = np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0)
    trans = np.zeros((2, 2), np.float32)
    fine_trans = np.zeros((3, 2), np.float32)
    result = k_class.run_dense_k_class_em_adaptive(
        SimpleNamespace(n_units=2, image_shape=(8, 8)),
        jnp.zeros((n_classes, 8), jnp.complex64), None,
        jnp.ones(64, jnp.float32), rot, trans, fine_rot, fine_trans,
        np.asarray([0, 1, 1]), np.asarray([0, 1, 1]), "linear_interp",
        coarse_current_size=4, fine_current_size=6,
        coarse_healpix_order=1, oversampling_order=1,
        coarse_engine="gemm_dense", symmetry_label=symmetry,
        mstep_relion_x_half=configured_native,
        relion_f32_normalization_sum_weight=jnp.ones(2, jnp.float32),
    )
    assert len(seen) == 1
    args, kwargs = seen[0]
    assert args[11] == [[None, None] for _ in range(n_classes)]
    assert kwargs["engine_kwargs"]["dense_gemm_full_grid"] is True
    assert kwargs["engine_kwargs"]["mstep_relion_x_half"] is True
    assert "relion_f32_normalization_sum_weight" not in kwargs["engine_kwargs"]
    assert kwargs["engine_kwargs"]["current_size"] == 6
    assert kwargs["engine_kwargs"]["symmetry_label"] == symmetry
    assert np.array_equal(np.asarray(result.significant_counts), np.full(2, n_classes * 4))
    calls = take_coarse_engine_calls()
    assert len(calls) == 1
    assert calls[0]["resolved"] == "gemm_dense"
    assert calls[0]["evaluated_fine_candidates_total"] == 18 * n_classes
    assert calls[0]["selected_fine_candidates_total"] == 18 * n_classes


@pytest.mark.unit
def test_memory_plan_bounds_preparation_and_nondivisible_rotation_count(monkeypatch):
    from relax.dense.gemm_coarse_engine import _memory_tiles
    from relax.sparse_pass2 import sparse_pass2_budget as budget

    monkeypatch.setattr(budget, "_device_free_memory_bytes", lambda: 1 << 30)
    monkeypatch.setattr(budget, "_jax_allocator_free_memory_bytes", lambda: 1 << 30)
    monkeypatch.setattr(budget, "_jax_allocator_pool_free_bytes", lambda: 0)
    state = SimpleNamespace(
        class_volumes=(None,), class_projector_halves=(np.zeros(64, np.complex64),),
        class_rotation_priors=(np.zeros(29, np.float32),),
        fine_rotations=np.zeros((29, 3, 3), np.float32),
        fine_translations=np.zeros((5, 2), np.float32),
        dataset=SimpleNamespace(n_units=9, image_shape=(8, 8)),
    )
    b, q, u, qs = _memory_tiles(
        state, n_score=20, n_recon=25, n_rect=35, n_groups=1, bp_size=1000,
    )
    assert (b, q, u, qs) == (9, 29, 5, 29)
    monkeypatch.setattr(budget, "_device_free_memory_bytes", lambda: 1)
    monkeypatch.setattr(budget, "_jax_allocator_free_memory_bytes", lambda: 1)
    with pytest.raises(MemoryError, match="minimum B=Q=U=1"):
        _memory_tiles(state, n_score=20, n_recon=25, n_rect=35, n_groups=1, bp_size=1000)


@pytest.mark.unit
def test_memory_plan_uses_large_projection_tile_and_full_translation_when_it_fits(monkeypatch):
    from relax.dense.gemm_coarse_engine import _memory_tiles
    from relax.sparse_pass2 import sparse_pass2_budget as budget

    monkeypatch.setattr(budget, "_device_free_memory_bytes", lambda: 16 << 30)
    monkeypatch.setattr(budget, "_jax_allocator_free_memory_bytes", lambda: 16 << 30)
    monkeypatch.setattr(budget, "_jax_allocator_pool_free_bytes", lambda: 0)
    state = SimpleNamespace(
        class_volumes=(None,), class_projector_halves=(np.zeros(64, np.complex64),),
        class_rotation_priors=(np.zeros(6144, np.float32),),
        fine_rotations=np.zeros((6144, 3, 3), np.float32),
        fine_translations=np.zeros((29, 2), np.float32),
        dataset=SimpleNamespace(n_units=100, image_shape=(128, 128)),
    )
    b, q, u, qs = _memory_tiles(
        state, n_score=1624, n_recon=1227, n_rect=1624,
        n_groups=1, bp_size=767050,
    )
    assert (b, q, u, qs) == (100, 3072, 29, 128)
    state.fine_rotations = np.zeros((4608, 3, 3), np.float32)
    b, q, u, qs = _memory_tiles(
        state, n_score=1624, n_recon=1227, n_rect=1624,
        n_groups=1, bp_size=767050,
    )
    assert (b, q, u, qs) == (100, 2304, 29, 128)


@pytest.mark.unit
def test_statistics_callback_uses_resident_native_kernel_owner():
    from relax.cuda import kernels as native
    from relax.dense import gemm_coarse_statistics as dense_stats

    assert dense_stats.cuda_backproject is native
    for name in (
        "relion_translate_sum_flat_rows_f32",
        "relion_wavg_sequential_runtime_flat_rows_triplet_f32",
        "relion_wavg_exact_atomic_flat_rows_triplet_add_f32",
    ):
        assert callable(getattr(dense_stats.cuda_backproject, name))


@pytest.mark.unit
@pytest.mark.parametrize("n_classes", [1, 3, 4])
@pytest.mark.parametrize("translation_tile,translation_side", [(1, "image"), (3, "projection")])
def test_joint_exact_normalizer_and_grouped_presums(n_classes, translation_tile, translation_side):
    rng = np.random.default_rng(42 + n_classes)
    b, r, t, ps, pr, groups = 3, 5, 5, 7, 9, 2

    def complex_values(shape):
        return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)

    images = complex_values((b, ps))
    references = complex_values((n_classes, r, ps))
    score_weight = rng.uniform(0.2, 0.8, (b, ps)).astype(np.float32)
    rec_image = complex_values((b, pr))
    rec_weight = rng.uniform(0.2, 0.8, (b, pr)).astype(np.float32)
    initial = rng.uniform(0, 0.1, b).astype(np.float32)
    class_prior = rng.uniform(-0.3, 0.2, (n_classes, b, r)).astype(np.float32)
    if n_classes > 1:
        class_prior[-1] = -np.inf  # a collapsed class carries exactly zero mass
    translation_prior = rng.uniform(-0.3, 0.2, (b, t)).astype(np.float32)
    score_phase = np.exp(1j * rng.normal(size=(t, ps))).astype(np.complex64)
    rec_phase = np.exp(1j * rng.normal(size=(t, pr))).astype(np.complex64)
    rotations = np.zeros((r, 3, 3), np.float32)
    rotations[:, 0, 0] = np.arange(r)
    grid = pad_grid(
        rotations, rotations, score_phase, rec_phase,
        rotation_tile=2, translation_tile=translation_tile,
    )
    batch = pad_batch(
        images, score_weight, initial, rec_image, rec_weight,
        np.zeros((b, r), np.float32), translation_prior, [2, 0, 4],
        image_capacity=4, grid=grid, sentinel_id=5,
    )
    padded_class_prior = jnp.pad(
        jnp.asarray(class_prior), ((0, 0), (0, 1), (0, grid.score_rotations.shape[0] - r)),
        constant_values=-jnp.inf,
    )
    group_ids = jnp.asarray([1, 0, 1, 0], dtype=jnp.int32)

    def project(reference, selected_rotations):
        return reference[selected_rotations[:, 0, 0].astype(jnp.int32)]

    def backproject(y, w, y_slices, w_slices, _):
        return y + jnp.sum(y_slices, axis=0), w + jnp.sum(w_slices, axis=0)

    program = make_joint_k_batch_program(
        DenseGemmTileConfig(4, 2, translation_tile, translation_side, "exact"), project, backproject,
    )
    result = jax.tree.map(
        np.asarray,
        program(
            jnp.asarray(references),
            jnp.zeros((n_classes, groups, pr), jnp.complex64),
            jnp.zeros((n_classes, groups, pr), jnp.float32),
            batch, grid, padded_class_prior, group_ids,
        ),
    )

    scores = np.stack([
        np.asarray(score_tile(
            jnp.asarray(references[k]), jnp.asarray(images), jnp.asarray(score_weight),
            jnp.asarray(initial), jnp.asarray(score_phase), jnp.asarray(class_prior[k]),
            jnp.asarray(translation_prior), jnp.ones((b,), bool),
            jnp.ones((r,), bool), jnp.ones((t,), bool), translation_side=translation_side,
        ))
        for k in range(n_classes)
    ])
    joint = np.asarray(jax.nn.logsumexp(jnp.asarray(scores).reshape(n_classes, b, r * t).transpose(1, 0, 2).reshape(b, -1), axis=1))
    class_z = np.asarray(jax.nn.logsumexp(jnp.asarray(scores).reshape(n_classes, b, -1), axis=2))
    posterior = np.exp(scores - joint[None, :, None, None])
    expected_y = np.zeros((n_classes, groups, pr), np.complex64)
    expected_w = np.zeros((n_classes, groups, pr), np.float32)
    for k in range(n_classes):
        for g in range(groups):
            rows = np.array([1, 0, 1]) == g
            expected_y[k, g] = np.einsum(
                "brt,btp->p", posterior[k, rows],
                rec_image[rows, None, :] * rec_phase[None, :, :],
            )
            expected_w[k, g] = np.einsum("brt,bp->p", posterior[k, rows], rec_weight[rows])
    assert_matches(result.joint_normalizer[:b].sum(axis=1), joint)
    assert_matches(result.class_normalizers[:, :b].sum(axis=2), class_z)
    assert_matches(result.numerator, expected_y)
    assert_matches(result.denominator, expected_w)
    assert_matches(result.rotation_posterior_sums[:, :r], posterior.sum(axis=(1, 3)))
    assert_matches(result.class_posterior_sums, posterior.sum(axis=(1, 2, 3)))
    assert_matches(result.class_translation_marginals[:, :b, :t], posterior.sum(axis=2))
    assert_matches(result.class_translation_marginals[:, :b, t:], np.zeros_like(result.class_translation_marginals[:, :b, t:]))
    assert_matches(result.image_posterior_mass[:b], np.ones((b,), np.float32))
    assert not result.invalid_normalizer[:b].any()
    assert not result.invalid_weight[:b].any()
    expected_pose = scores.reshape(n_classes, b, -1).argmax(axis=2)
    expected_pose = np.where(np.isfinite(scores.reshape(n_classes, b, -1).max(axis=2)), expected_pose, -1)
    assert np.array_equal(result.class_best_pose_ids[:, :b], expected_pose)
    if n_classes == 3 and translation_side == "projection" and translation_tile == 3:
        # Class IDs are layout, not a hidden normalizer boundary. A stable-ID
        # permutation must permute class outputs and leave the joint E-step.
        permutation = np.array([2, 0, 1])
        permuted = jax.tree.map(np.asarray, program(
            jnp.asarray(references[permutation]),
            jnp.zeros((n_classes, groups, pr), jnp.complex64),
            jnp.zeros((n_classes, groups, pr), jnp.float32),
            batch, grid, padded_class_prior[permutation], group_ids,
        ))
        assert_matches(permuted.joint_normalizer[:b], result.joint_normalizer[:b])
        assert_matches(permuted.numerator, result.numerator[permutation])
        assert_matches(permuted.denominator, result.denominator[permutation])
        assert_matches(permuted.class_translation_marginals, result.class_translation_marginals[permutation])


@pytest.mark.unit
def test_joint_best_pose_ties_follow_canonical_rotation_translation_order():
    # The first U tile finds the maximum at (r1,t0), pose 3.  A later U tile
    # finds an equal maximum at (r0,t2), pose 2, which is first in canonical
    # R,T order.  Strict `>` alone incorrectly retains pose 3.
    n_classes, n_rot, n_trans = 3, 2, 3
    rotations = np.zeros((n_rot, 3, 3), np.float32)
    rotations[:, 0, 0] = np.arange(n_rot)
    phases = np.asarray([[-1], [-1], [1]], np.complex64)
    grid = pad_grid(rotations, rotations, phases, phases, rotation_tile=2, translation_tile=2)
    batch = pad_batch(
        np.ones((1, 1), np.complex64), np.ones((1, 1), np.float32), np.zeros(1, np.float32),
        np.zeros((1, 1), np.complex64), np.ones((1, 1), np.float32),
        np.zeros((1, n_rot), np.float32), np.zeros((1, n_trans), np.float32), [0],
        image_capacity=1, grid=grid, sentinel_id=1,
    )

    def project(reference, selected_rotations):
        return reference[selected_rotations[:, 0, 0].astype(jnp.int32)]

    def backproject(y, w, y_slices, w_slices, _):
        return y + jnp.sum(y_slices, axis=0), w + jnp.sum(w_slices, axis=0)

    result = make_joint_k_batch_program(
        DenseGemmTileConfig(1, 2, 2, "image", "exact"), project, backproject,
    )(
        jnp.tile(jnp.asarray([[[1], [-1]]], jnp.complex64), (n_classes, 1, 1)),
        jnp.zeros((n_classes, 1, 1), jnp.complex64),
        jnp.zeros((n_classes, 1, 1), jnp.float32),
        batch, grid, jnp.zeros((n_classes, 1, grid.score_rotations.shape[0]), jnp.float32),
        jnp.zeros((1,), jnp.int32),
    )
    assert np.array_equal(np.asarray(result.class_best_pose_ids)[:, 0], np.full(n_classes, 2, np.int32))


@pytest.mark.unit
def test_joint_route_rejects_lagged_mode():
    def unavailable(*_):
        raise AssertionError("callbacks must not be called")

    with pytest.raises(ValueError, match="requires exact"):
        make_joint_k_batch_program(
            DenseGemmTileConfig(1, 1, 1, "image", "lagged"), unavailable, unavailable
        )


@pytest.mark.unit
def test_split_normalizer_keeps_pmax_and_evidence_at_large_score_gauge():
    # A float32 addition loses log(2) at this gauge.  Posterior arithmetic
    # must subtract the split pair before the addition; reporting converts
    # both components to the resident metadata dtype first.
    with jax.enable_x64(True):
        pair = jnp.asarray([[1_000_000.0, np.log(2.0)]], jnp.float32)
        log_z, pmax = split_normalizer_metadata(pair, jnp.asarray([1_000_000.0], jnp.float32))
        assert_matches(np.asarray(pmax), np.asarray([0.5], np.float32))
        assert_matches(np.asarray(log_z), np.asarray([1_000_000.0 + np.float64(np.float32(np.log(2.0)))], np.float64))
        assert float(np.asarray(log_z)[0]) > 1_000_000.5


@pytest.mark.unit
def test_statistics_carry_keeps_float64_noise_norm_partials():
    with jax.enable_x64(True):
        config = ResidentStatisticsConfig(
            n_shells=2, n_fine_trans=3, image_capacity=4, n_coarse_rot=2,
            n_scale_groups=0, norm_unweighted_shell_cutoff=None,
            include_unweighted_high_shell=True, disable_cuda_binning=False,
            deterministic_norm_reduction=False, use_exact_relion_gaussian=True,
            relion_wavg_atomic_direct_noise=True, relion_wavg_atomic_scale_aa=True,
            direct_noise_exclusive_shell_stop=2, accumulate_scale=False,
        )
        plan = DenseGemmStatisticsPlan(
            image_shape=(4, 4), image_capacity=4, rotation_tile=2,
            translation_tile=1, rows_per_statistics_block=1,
            n_recon_pixels=2, n_rect_pixels=2, n_shells=2,
            n_classes=1, n_coarse_rot=2, n_fine_rotations=2, n_optics_groups=1,
            use_rfloat_ctf_wavg=True, score_mode="gaussian", stats_config=config,
        )
        carry = initial_statistics_carry(plan, noise_real_dtype=jnp.float64)
        assert carry.a2_per_image.dtype == jnp.float64
        assert carry.xa_per_image.dtype == jnp.float64


@pytest.mark.unit
@pytest.mark.parametrize("score_mode", ["gaussian", "normalized_cc"])
def test_gradient_presum_and_cc_winner_with_distinct_reconstruction_pixels(score_mode):
    # The BPref residual uses reconstruction-window projections.  Its score
    # window has two pixels and reconstruction has three, so a mistaken score
    # projection cannot fit the residual.  CC selects one class/pose globally.
    references = jnp.asarray([
        [[1 + 0j, 0 + 0j, 2 + 0j], [0 + 0j, 0 + 0j, 3 + 0j]],
        [[0 + 0j, 0 + 0j, 4 + 0j], [2 + 0j, 0 + 0j, 5 + 0j]],
    ], jnp.complex64)
    rotations = np.zeros((2, 3, 3), np.float32)
    rotations[:, 0, 0] = np.arange(2)
    mstep_rotations = rotations.copy()
    mstep_rotations[:, 0, 0] += 10  # distinct M-step matrices cannot index score projections
    grid = pad_grid(
        rotations, mstep_rotations, np.ones((3, 2), np.complex64),
        np.ones((3, 3), np.complex64), rotation_tile=2, translation_tile=2,
    )
    batch = pad_batch(
        np.array([[2, 0]], np.complex64), np.ones((1, 2), np.float32), np.zeros(1, np.float32),
        np.array([[1, 2, 3]], np.complex64), np.array([[0.5, 0.75, 1.25]], np.float32),
        np.zeros((1, 2), np.float32), np.zeros((1, 3), np.float32), [0],
        image_capacity=1, grid=grid, sentinel_id=1,
    )

    def project_score(ref, selected):
        return ref[selected[:, 0, 0].astype(jnp.int32), :2]

    def project_recon(ref, selected):
        return ref[selected[:, 0, 0].astype(jnp.int32)]

    def backproject(y, w, ys, ws, _):
        return y + jnp.sum(ys, axis=0), w + jnp.sum(ws, axis=0)

    def common():
        # The production program donates both accumulator arrays.
        return (references, jnp.zeros((2, 1, 3), jnp.complex64),
                jnp.zeros((2, 1, 3), jnp.float32), batch, grid,
                jnp.zeros((2, 1, 2), jnp.float32), jnp.zeros((1,), jnp.int32))
    cfg = DenseGemmTileConfig(1, 2, 2, "image", "exact")
    plain = make_joint_k_batch_program(cfg, project_score, backproject, score_mode=score_mode)(*common())
    residual = make_joint_k_batch_program(
        cfg, project_score, backproject, score_mode=score_mode,
        project_reconstruction=project_recon, mstep_subtract_ctf_projection=True,
    )(*common())
    plain, residual = jax.tree.map(np.asarray, (plain, residual))
    assert_matches(plain.denominator, residual.denominator)
    for k in range(2):
        expected_subtraction = np.einsum(
            "r,rp,p->p", np.asarray(residual.rotation_posterior_sums[k, :2]),
            np.asarray(references[k]), np.array([0.5, 0.75, 1.25], np.float32),
        )
        assert_matches(plain.numerator[k, 0] - residual.numerator[k, 0], expected_subtraction)
    if score_mode == "normalized_cc":
        assert_matches(residual.image_posterior_mass, np.ones(1, np.float32))
        assert_matches(residual.class_posterior_sums.sum(), 1.0)
        assert np.count_nonzero(np.asarray(residual.class_posterior_sums)) == 1
