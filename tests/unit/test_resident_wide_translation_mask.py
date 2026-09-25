"""The resident candidate bitset holds any number of coarse translations.

A resident candidate row carries one bit per coarse translation: bit
``k % 32`` of uint32 word ``k // 32``. Real auto-refine grids exceed one word
(EMPIAR-10081 HCN1, offset range 5 / step 2: 37 coarse translations), so the
tables, the chunk materialization and the device expansion must agree with the
compact engine's dense candidate mask at 33, 37 and 64 translations as they do
at 32 or fewer. RELION's significant-candidate choice is made upstream on
(rotation, translation) cells; the bitset only transports it.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp
from test_resident_significance import _csr_from_supports, _encoded_supports, _supports

from relax.scoring.compact_candidates import _candidate_mask_to_dense
from relax.scoring.sparse_bucket_arrays import _prepare_per_image_pass2_inputs
from relax.sparse_pass2 import resident_pass2 as rp
from relax.sparse_pass2.resident_candidates import (
    build_resident_candidate_tables,
    expand_chunk_mask_jnp,
    expand_mask_rows,
    materialize_chunk,
    n_mask_words,
    plan_capacity_chunks,
)
from relax.sparse_pass2.resident_significance import build_resident_candidate_tables_from_csr

pytestmark = pytest.mark.unit

WIDE_TRANSLATION_COUNTS = (33, 37, 64)
N_COARSE_ROT = 18  # level 0 whole psi rows: 3 directions x 6 psi
CHILDREN = 2


def _z_rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _wide_case(n_coarse_trans: int, seed: int = 3):
    n_samples = N_COARSE_ROT * n_coarse_trans
    supports = _supports(9, n_samples, seed=seed)
    fine_parent = np.repeat(np.arange(N_COARSE_ROT, dtype=np.int64), CHILDREN)
    fine_rotations = np.stack([_z_rotation(0.02 * k) for k in range(fine_parent.size)])
    fine_translation_parent = np.repeat(np.arange(n_coarse_trans, dtype=np.int32), 2)
    common = dict(
        n_coarse_trans=n_coarse_trans,
        fine_translation_parent=fine_translation_parent,
    )
    per_image_inputs = _prepare_per_image_pass2_inputs(
        _encoded_supports(supports, n_samples),
        n_coarse_rot=N_COARSE_ROT,
        nside_level=0,
        oversampling_order=0,
        n_fine_trans=fine_translation_parent.size,
        rotation_log_prior=None,
        random_perturbation=0.0,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
        dtype=np.float32,
        **common,
    )
    host_tables = build_resident_candidate_tables(per_image_inputs, n_fine_trans=fine_translation_parent.size, **common)
    csr_tables = build_resident_candidate_tables_from_csr(
        _csr_from_supports(supports, n_coarse_rot=N_COARSE_ROT, n_coarse_trans=n_coarse_trans),
        nside_level=0,
        oversampling_order=0,
        n_fine_trans=fine_translation_parent.size,
        fine_translation_parent=fine_translation_parent,
        rotation_log_prior=None,
        random_perturbation=0.0,
        fine_rotation_parent_override=fine_parent,
        dtype=np.float32,
    )
    return per_image_inputs, host_tables, csr_tables, fine_translation_parent


@pytest.mark.parametrize("n_coarse_trans", WIDE_TRANSLATION_COUNTS)
def test_wide_tables_expand_to_the_compact_dense_mask(n_coarse_trans):
    """Host and CSR tables expand, per image, to the compact engine's own dense mask."""

    per_image_inputs, host_tables, csr_tables, fine_parent = _wide_case(n_coarse_trans)
    assert host_tables.parent_trans_bits.shape[1] == n_mask_words(n_coarse_trans) > 1
    np.testing.assert_array_equal(csr_tables.parent_trans_bits, host_tables.parent_trans_bits)
    np.testing.assert_array_equal(csr_tables.mask_mode, host_tables.mask_mode)
    for image, mask in enumerate(per_image_inputs["candidate_mask"]):
        compact_dense = _candidate_mask_to_dense(mask)
        np.testing.assert_array_equal(
            expand_mask_rows(host_tables, image, fine_parent), compact_dense, err_msg=f"image {image}"
        )


@pytest.mark.parametrize("n_coarse_trans", WIDE_TRANSLATION_COUNTS)
def test_wide_chunk_masks_expand_to_the_compact_dense_mask(n_coarse_trans):
    """The device chunk expansion reproduces the compact mask row for row, padding all-false."""

    per_image_inputs, _, tables, fine_parent = _wide_case(n_coarse_trans)
    chunks = plan_capacity_chunks(tables, row_capacity_ladder=(16, 64, 256), image_capacity_ladder=(2, 4))
    for chunk in chunks:
        materialized = materialize_chunk(tables, chunk)
        assert materialized["row_mask_bits"].shape == (chunk.row_capacity, n_mask_words(n_coarse_trans))
        chunk_mask = np.asarray(
            expand_chunk_mask_jnp(
                jnp.asarray(materialized["row_mask_bits"]),
                jnp.asarray(materialized["row_mask_mode"]),
                jnp.asarray(fine_parent),
            )
        )
        assert not chunk_mask[chunk.n_valid_rows :].any()
        for image in range(chunk.image_start, chunk.image_stop):
            start = int(tables.row_offsets[image]) - chunk.row_start
            stop = int(tables.row_offsets[image + 1]) - chunk.row_start
            np.testing.assert_array_equal(
                chunk_mask[start:stop],
                _candidate_mask_to_dense(per_image_inputs["candidate_mask"][image]),
                err_msg=f"image {image}",
            )


def test_wide_translation_grids_are_in_scope():
    """The resident gate no longer caps the coarse translation count."""

    import inspect

    source = inspect.getsource(rp.require_resident_production_configuration)
    assert "n_coarse_trans" not in source


# ---------------------------------------------------------------------------
# GPU: the resident driver against the compact engine with wide grids
# ---------------------------------------------------------------------------


def _resident_gpu_available() -> bool:
    from test_resident_pass2_driver import _gpu_available

    return _gpu_available()


requires_resident_gpu = pytest.mark.skipif(
    not _resident_gpu_available(),
    reason="the resident stages are CUDA FFI targets",
)


def _wide_driver_args(n_coarse_trans: int, seed: int = 20260925):
    """The 8x8 production-shaped K=1 fixture with an n_coarse_trans coarse grid."""

    from test_resident_pass2_driver import _driver_fixture_args

    args = _driver_fixture_args(seed=seed)
    n_images = int(args["experiment_dataset"].n_units)
    n_coarse_rot = int(np.asarray(args["rotation_log_prior"]).size)
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n_coarse_trans)))
    grid = np.stack(np.meshgrid(np.arange(side), np.arange(side), indexing="ij"), -1).reshape(-1, 2)
    translations = ((grid[:n_coarse_trans] - side // 2) * 0.5).astype(np.float32)
    total = n_coarse_rot * n_coarse_trans
    samples = [None]
    for _ in range(1, n_images):
        count = int(rng.integers(1, total // 2))
        samples.append(np.sort(rng.choice(total, size=count, replace=False).astype(np.int32)))
    args.update(
        translations=translations,
        significant_sample_indices=samples,
        translation_log_prior=rng.normal(scale=0.05, size=(n_images, n_coarse_trans)).astype(np.float32),
        fine_translations_override=np.concatenate([translations, translations + 0.25]).astype(np.float32),
        fine_translation_parent_override=np.concatenate([np.arange(n_coarse_trans), np.arange(n_coarse_trans)]).astype(
            np.int32
        ),
    )
    return args


@pytest.fixture
def _resident_production_env(monkeypatch):
    """The production resident arm, as in ``test_resident_pass2_driver``."""

    monkeypatch.setenv("RELAX_EM_PROTOTYPE_SOFT_POSTERIOR_BLOCK_BPREF", "1")
    monkeypatch.setenv("RELAX_RELION_WAVG_ATOMIC_SCALE_AA", "1")
    monkeypatch.setenv("RELAX_RELION_WAVG_ATOMIC_DIRECT_NOISE_ONLY", "1")
    monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_ROW_CAPACITIES", "256,1024,4096")
    monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_IMAGE_CAPACITIES", "4,16,64")
    monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_MSTEP_BLOCK_ROWS", "128")


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    den = float(np.linalg.norm(a))
    return float(np.linalg.norm(a - b) / den) if den else float(np.linalg.norm(a - b))


@requires_resident_gpu
@pytest.mark.parametrize("n_coarse_trans", WIDE_TRANSLATION_COUNTS)
def test_wide_grid_resident_driver_matches_the_compact_engine(_resident_production_env, n_coarse_trans):
    """Resident vs compact at 33, 37 and 64 coarse translations, the C1 test's bounds."""

    from relax.sparse_pass2.sparse_pass2_bucketed import compute_pass2_stats_sparse_bucketed

    args = _wide_driver_args(n_coarse_trans)
    compact = compute_pass2_stats_sparse_bucketed(**args)
    resident = rp.compute_pass2_stats_resident(**args)

    np.testing.assert_array_equal(compact.hard_assignment, resident.hard_assignment)
    np.testing.assert_array_equal(compact.best_rotation_indices, resident.best_rotation_indices)
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
