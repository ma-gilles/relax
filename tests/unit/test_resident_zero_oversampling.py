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
from test_resident_significance import _encoded_supports, _supports

from relax.scoring.compact_candidates import _candidate_mask_to_dense
from relax.scoring.sparse_bucket_arrays import _prepare_per_image_pass2_inputs
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
