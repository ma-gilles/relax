"""Zero oversampling (``--adaptive_oversampling 0``) on the device-resident K=1 pass 2.

RELION's fine pass then keeps the coarse pass's float32 ``sum_weight`` and max
(acc_ml_optimiser_impl.h:2868), keeps every weight (``significant_weight =
sorted[0]``, :3590) and reports Pmax as the coarse ``max_weight / sum_weight``
(:3268-3269, :4223); see docs/math/zero_oversampling.md. The compact engine
implements that as ``reuse_coarse_normalization``; the resident driver must give
the same winner, Pmax, evidence, maps and statistics.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
from helpers.float_compare import assert_matches
from test_resident_significance import _encoded_supports, _supports

from relax.scoring.compact_candidates import _candidate_mask_to_dense
from relax.scoring.sparse_bucket_arrays import (
    _prepare_per_image_pass2_inputs,
    coarse_winner_local_pose_ids,
)
from relax.sparse_pass2 import resident_pass2 as rp
from relax.sparse_pass2.resident_candidates import (
    build_resident_candidate_tables,
    coarse_winner_cells,
)

pytestmark = pytest.mark.unit

N_COARSE_ROT = 18  # level 0 whole psi rows: 3 directions x 6 psi
N_COARSE_TRANS = 5


def _z_rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _selected_winners(per_image_inputs, rng):
    """One selected (coarse rotation, coarse translation) pose id per image."""

    winners = []
    for image, mask in enumerate(per_image_inputs["candidate_mask"]):
        dense = _candidate_mask_to_dense(mask)
        rows, trans = np.nonzero(dense)
        pick = int(rng.integers(rows.size))
        parents = np.asarray(per_image_inputs["unique_rot"][image])[np.asarray(per_image_inputs["parent_map"][image])]
        winners.append(int(parents[rows[pick]]) * N_COARSE_TRANS + int(trans[pick]))
    return np.asarray(winners, dtype=np.int64)


def _os0_case(seed=11):
    """Host tables of a zero-oversampling pass: one fine child per coarse (rot, trans)."""

    n_samples = N_COARSE_ROT * N_COARSE_TRANS
    supports = [s for s in _supports(9, n_samples, seed=seed) if np.asarray(s).size]
    fine_parent = np.arange(N_COARSE_ROT, dtype=np.int64)
    fine_rotations = np.stack([_z_rotation(0.05 * k) for k in range(N_COARSE_ROT)])
    fine_translation_parent = np.arange(N_COARSE_TRANS, dtype=np.int32)
    per_image_inputs = _prepare_per_image_pass2_inputs(
        _encoded_supports(supports, n_samples),
        n_coarse_rot=N_COARSE_ROT,
        n_coarse_trans=N_COARSE_TRANS,
        nside_level=0,
        oversampling_order=0,
        n_fine_trans=N_COARSE_TRANS,
        fine_translation_parent=fine_translation_parent,
        rotation_log_prior=None,
        random_perturbation=0.0,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
        dtype=np.float32,
    )
    tables = build_resident_candidate_tables(
        per_image_inputs,
        n_coarse_trans=N_COARSE_TRANS,
        n_fine_trans=N_COARSE_TRANS,
        fine_translation_parent=fine_translation_parent,
    )
    return per_image_inputs, tables, fine_parent, fine_translation_parent


def test_coarse_winner_cells_match_the_compact_winner_ids():
    """The resident segment cell of the coarse winner is the compact engine's local pose id."""

    per_image_inputs, tables, fine_parent, fine_translation_parent = _os0_case()
    winners = _selected_winners(per_image_inputs, np.random.default_rng(3))
    compact = coarse_winner_local_pose_ids(per_image_inputs, winners, fine_translation_parent, N_COARSE_TRANS)
    resident = coarse_winner_cells(
        tables,
        winners,
        fine_rotation_parent=fine_parent,
        fine_translation_parent=fine_translation_parent,
    )
    assert_matches(resident, compact)


def test_coarse_winner_cells_refuse_an_unselected_winner():
    per_image_inputs, tables, fine_parent, fine_translation_parent = _os0_case()
    winners = _selected_winners(per_image_inputs, np.random.default_rng(3))
    for image, mask in enumerate(per_image_inputs["candidate_mask"]):
        dense = _candidate_mask_to_dense(mask)
        parents = np.asarray(per_image_inputs["unique_rot"][image])[np.asarray(per_image_inputs["parent_map"][image])]
        rows, trans = np.nonzero(~dense)
        if rows.size:
            winners[image] = int(parents[rows[0]]) * N_COARSE_TRANS + int(trans[0])
            break
    else:
        pytest.skip("the fixture has no unselected candidate")
    with pytest.raises(ValueError, match="missing from selected fine support"):
        coarse_winner_cells(
            tables,
            winners,
            fine_rotation_parent=fine_parent,
            fine_translation_parent=fine_translation_parent,
        )


# ---------------------------------------------------------------------------
# GPU: the resident driver against the compact engine at zero oversampling
# ---------------------------------------------------------------------------


def _resident_gpu_available() -> bool:
    from test_resident_pass2_driver import _gpu_available

    return _gpu_available()


requires_resident_gpu = pytest.mark.skipif(
    not _resident_gpu_available(),
    reason="the resident stages are CUDA FFI targets",
)


@pytest.fixture
def _resident_production_env(monkeypatch):
    """The production resident arm, as in ``test_resident_pass2_driver``."""

    monkeypatch.setenv("RELAX_EM_PROTOTYPE_SOFT_POSTERIOR_BLOCK_BPREF", "1")
    monkeypatch.setenv("RELAX_RELION_WAVG_ATOMIC_SCALE_AA", "1")
    monkeypatch.setenv("RELAX_RELION_WAVG_ATOMIC_DIRECT_NOISE_ONLY", "1")
    monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_ROW_CAPACITIES", "256,1024,4096")
    monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_IMAGE_CAPACITIES", "4,16,64")
    monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_MSTEP_BLOCK_ROWS", "128")


def _os0_driver_args(seed=20260925):
    """The 8x8 production-shaped K=1 fixture at zero oversampling, with a coarse state."""

    from test_resident_pass2_driver import _driver_fixture_args

    args = _driver_fixture_args(seed=seed)
    n_images = int(args["experiment_dataset"].n_units)
    n_coarse_rot = int(np.asarray(args["rotation_log_prior"]).size)
    translations = np.asarray(args["translations"])
    n_coarse_trans = int(translations.shape[0])
    rng = np.random.default_rng(seed)
    fine_parent = np.arange(n_coarse_rot, dtype=np.int32)
    fine_rotations = np.stack([_z_rotation(0.031 * k) for k in range(n_coarse_rot)]).astype(np.float32)
    args.update(
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
        fine_translations_override=translations.astype(np.float32),
        fine_translation_parent_override=np.arange(n_coarse_trans, dtype=np.int32),
    )
    winners = []
    for sample in args["significant_sample_indices"]:
        ids = np.arange(n_coarse_rot * n_coarse_trans) if sample is None else np.asarray(sample)
        winners.append(int(ids[rng.integers(ids.size)]))
    args.update(
        relion_f32_normalization_sum_weight=rng.uniform(1.0, 20.0, n_images).astype(np.float64),
        relion_coarse_hard_assignment=np.asarray(winners, dtype=np.int64),
        relion_coarse_max_posterior=rng.uniform(0.05, 0.95, n_images).astype(np.float64),
    )
    return args


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    den = float(np.linalg.norm(a))
    return float(np.linalg.norm(a - b) / den) if den else float(np.linalg.norm(a - b))


@requires_resident_gpu
def test_zero_oversampling_resident_driver_matches_the_compact_engine(_resident_production_env):
    """Winner, Pmax, evidence, maps and statistics against the compact engine's os0 arithmetic."""

    from relax.sparse_pass2.sparse_pass2_bucketed import compute_pass2_stats_sparse_bucketed

    args = _os0_driver_args()
    compact = compute_pass2_stats_sparse_bucketed(**args)
    resident = rp.compute_pass2_stats_resident(**args)

    # The coarse winner and Pmax are carried through, not recomputed.
    np.testing.assert_array_equal(compact.hard_assignment, resident.hard_assignment)
    np.testing.assert_array_equal(compact.best_rotation_indices, resident.best_rotation_indices)
    # Published in the scoring precision (float32), as RELION's max_weight / sum_weight.
    assert_matches(
        np.asarray(resident.relion_stats.max_posterior_per_image, dtype=np.float32),
        np.asarray(args["relion_coarse_max_posterior"], dtype=np.float32),
    )
    for field in (
        "log_evidence_per_image",
        "best_log_score_per_image",
        "max_posterior_per_image",
        "rotation_posterior_sums",
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(compact.relion_stats, field), dtype=np.float64),
            np.asarray(getattr(resident.relion_stats, field), dtype=np.float64),
            rtol=1e-6,
            atol=1e-9,
            err_msg=field,
        )
    assert _rel_l2(compact.Ft_y, resident.Ft_y) < 1e-6
    assert _rel_l2(compact.Ft_ctf, resident.Ft_ctf) < 1e-6
    assert _rel_l2(compact.noise_stats.wsum_sigma2_noise, resident.noise_stats.wsum_sigma2_noise) < 1e-4
    for field in ("wsum_img_power", "wsum_norm_correction", "wsum_scale_correction_xa", "wsum_scale_correction_aa"):
        assert _rel_l2(getattr(compact.noise_stats, field), getattr(resident.noise_stats, field)) < 1e-6, field


@requires_resident_gpu
def test_zero_oversampling_resident_local_matches_the_exact_engine(monkeypatch):
    """The local zero-oversampling route keeps every weight on both engines.

    RELION's symbolic second pass sets ``significant_weight`` to the minimum
    weight (acc_ml_optimiser_impl.h:3590), so the reconstruction and the
    statistics use the complete posterior; bounds as in the pruned local test.
    """

    import test_resident_local_pass2 as local_tests

    monkeypatch.setenv("RELAX_LOCAL_SEARCH_RESIDENT_ROW_CAPACITIES", "64,256,1024")
    monkeypatch.setenv("RELAX_LOCAL_SEARCH_RESIDENT_IMAGE_CAPACITIES", "2,4,8")
    case = local_tests._case()
    exact = local_tests._run(case, resident=False, monkeypatch=monkeypatch, zero_oversampling=True)
    resident = local_tests._run(case, resident=True, monkeypatch=monkeypatch, zero_oversampling=True)

    np.testing.assert_array_equal(np.asarray(exact.hard_assignment), np.asarray(resident.hard_assignment))
    assert_matches(np.asarray(exact.best_pose_eulers_deg), np.asarray(resident.best_pose_eulers_deg))
    assert _rel_l2(exact.Ft_y, resident.Ft_y) < 1e-5
    assert _rel_l2(exact.Ft_ctf, resident.Ft_ctf) < 1e-5
    np.testing.assert_allclose(
        np.asarray(exact.relion_stats.max_posterior_per_image, dtype=np.float64),
        np.asarray(resident.relion_stats.max_posterior_per_image, dtype=np.float64),
        rtol=0,
        atol=1e-5,
    )
    exact_total = np.asarray(exact.noise_stats.wsum_sigma2_noise) + np.asarray(exact.noise_stats.wsum_img_power)
    resident_total = np.asarray(resident.noise_stats.wsum_sigma2_noise) + np.asarray(
        resident.noise_stats.wsum_img_power
    )
    assert _rel_l2(exact_total, resident_total) < 1e-4
    assert abs(float(exact.noise_stats.sumw) - float(resident.noise_stats.sumw)) <= 1e-5 * abs(
        float(exact.noise_stats.sumw)
    )
