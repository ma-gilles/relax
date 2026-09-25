"""Point-group symmetry on the device-resident K=1 pass 2 (global and local).

RELION restricts the orientation sampling to the asymmetric unit
(healpix_sampling.cpp removeSymmetryEquivalentPoints) and symmetrises each
BPref after the expectation (ml_optimiser.cpp:5541-5575
symmetriseReconstructions: enforceHermitianSymmetry, then
applyPointGroupSymmetry). The resident drivers take the reduced grid from the
caller's fine rotation override and finalize with the same
``finalize_half_volume_bpref`` the compact and exact local engines use.

The CPU tests check the grid plumbing and the wiring; the GPU tests compare
the resident drivers against the compact / exact local engines on a C4 pass
and check that the finalized weights are C4-invariant.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
from helpers.float_compare import assert_matches
from test_resident_significance import (
    FINE_TRANS_PARENT,
    N_COARSE_TRANS,
    N_FINE_TRANS,
    _assert_tables_equal,
    _csr_from_supports,
    _encoded_supports,
    _supports,
)

from relax.refinement import local_search_iteration
from relax.scoring.sparse_bucket_arrays import _prepare_per_image_pass2_inputs
from relax.sparse_pass2 import resident_local_pass2 as rlp
from relax.sparse_pass2 import resident_pass2 as rp
from relax.sparse_pass2.resident_candidates import build_resident_candidate_tables
from relax.sparse_pass2.resident_significance import build_resident_candidate_tables_from_csr

pytestmark = pytest.mark.unit

SYMMETRY = "C4"


def _relion_bind_available() -> bool:
    try:
        from relax.relion_bind import _relion_bind_core  # noqa: F401
    except ImportError:
        return False
    return True


# The reduced grid and the operators come from RELION's SymList via the binding.
requires_relion_bind = pytest.mark.skipif(
    not _relion_bind_available(),
    reason="RELION's asymmetric-unit sampling and SymList operators need the relion_bind extension",
)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    den = float(np.linalg.norm(a))
    return float(np.linalg.norm(a - b) / den) if den else float(np.linalg.norm(a - b))


def _c4_rotation_residual(ctf_public_flat) -> float:
    """Relative change of a public (x, y, z) weight volume under a 90 deg z turn.

    RELION's Cn axis is z. On an odd centred grid a quarter turn about z maps
    grid points onto grid points, so a C4-symmetrised weight is invariant to
    float32 interpolation rounding; an unsymmetrised accumulator is not.
    applyPointGroupSymmetry only symmetrises r <= ROUND(r_max * padding_factor)
    (backprojector.cpp:2651-2680), one voxel inside the accumulator's half
    width, so the outer shell the trilinear backprojection also touches is
    excluded here.
    """

    values = np.asarray(ctf_public_flat, dtype=np.float64)
    size = int(round(values.size ** (1.0 / 3.0)))
    assert size**3 == values.size, "expected the full public layout of an odd cube"
    grid = values.reshape(size, size, size)
    centre = size // 2
    axis = np.arange(size) - centre
    r2 = axis[:, None, None] ** 2 + axis[None, :, None] ** 2 + axis[None, None, :] ** 2
    inside = r2 <= (centre - 1) ** 2
    masked = np.where(inside, grid, 0.0)
    assert np.linalg.norm(masked) > 0.0, "the symmetrised support holds no weight"
    return _rel_l2(masked, np.rot90(masked, k=1, axes=(0, 1)))


def _rotation_diagnostics(ctf_public_flat) -> dict:
    """Quarter-turn residuals about each axis pair, masked and whole, for failure messages."""

    values = np.asarray(ctf_public_flat, dtype=np.float64)
    size = int(round(values.size ** (1.0 / 3.0)))
    grid = values.reshape(size, size, size)
    axis = np.arange(size) - size // 2
    r2 = axis[:, None, None] ** 2 + axis[None, :, None] ** 2 + axis[None, None, :] ** 2
    masked = np.where(r2 <= (size // 2 - 1) ** 2, grid, 0.0)
    return {
        f"{name}{axes}": _rel_l2(volume, np.rot90(volume, k=1, axes=axes))
        for name, volume in (("masked", masked), ("whole", grid))
        for axes in ((0, 1), (0, 2), (1, 2))
    }


# ---------------------------------------------------------------------------
# CPU: grid plumbing and wiring
# ---------------------------------------------------------------------------


@requires_relion_bind
@pytest.mark.parametrize("execution_order", [False, True])
def test_c4_tables_from_csr_match_the_host_path_on_the_generated_grid(execution_order):
    """The generator branch of the CSR tables follows the asymmetric-unit grid."""

    from relax.sampling import rotation_grid_size

    nside_level = 1
    n_coarse_rot = rotation_grid_size(nside_level, SYMMETRY)
    assert n_coarse_rot < rotation_grid_size(nside_level), "C4 must reduce the grid"
    n_samples = n_coarse_rot * N_COARSE_TRANS
    supports = _supports(9, n_samples, seed=5)
    rotation_log_prior = np.linspace(-1.0, 1.0, n_coarse_rot, dtype=np.float32)

    per_image_inputs = _prepare_per_image_pass2_inputs(
        _encoded_supports(supports, n_samples),
        n_coarse_rot=n_coarse_rot,
        n_coarse_trans=N_COARSE_TRANS,
        nside_level=nside_level,
        oversampling_order=1,
        n_fine_trans=N_FINE_TRANS,
        fine_translation_parent=FINE_TRANS_PARENT,
        rotation_log_prior=rotation_log_prior,
        random_perturbation=0.0,
        relion_parent_execution_order=execution_order,
        dtype=np.float32,
        symmetry_label=SYMMETRY,
    )
    expected = build_resident_candidate_tables(
        per_image_inputs,
        n_coarse_trans=N_COARSE_TRANS,
        n_fine_trans=N_FINE_TRANS,
        fine_translation_parent=FINE_TRANS_PARENT,
    )
    csr = _csr_from_supports(supports, n_coarse_rot=n_coarse_rot, n_coarse_trans=N_COARSE_TRANS)
    got = build_resident_candidate_tables_from_csr(
        csr,
        nside_level=nside_level,
        oversampling_order=1,
        n_fine_trans=N_FINE_TRANS,
        fine_translation_parent=FINE_TRANS_PARENT,
        rotation_log_prior=rotation_log_prior,
        random_perturbation=0.0,
        relion_parent_execution_order=execution_order,
        dtype=np.float32,
        symmetry_label=SYMMETRY,
    )
    _assert_tables_equal(got, expected)


def test_point_groups_are_in_scope_for_the_resident_driver():
    """A symmetric pass is covered; the dispatcher no longer routes it to compact."""

    assert "symmetry_label" not in inspect.signature(rp.resident_pass2_out_of_scope_reason).parameters
    assert "symmetry" not in inspect.getsource(rp.require_resident_production_configuration)


def test_resident_local_call_carries_the_point_group():
    """The resident local pass 2 receives the run's point group from its caller."""

    tree = ast.parse(inspect.getsource(local_search_iteration))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "compute_local_search_resident"
    ]
    assert len(calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    assert isinstance(keywords.get("symmetry_label"), ast.Name)
    assert keywords["symmetry_label"].id == "symmetry"
    assert "symmetry_label" in inspect.signature(rlp.compute_local_search_resident).parameters


@requires_relion_bind
def test_replay_reads_the_point_group_relion_sampled_with(tmp_path):
    """A replay of a symmetric RELION run takes its group from the sampling STAR."""

    from relax.relion.relion_metadata import read_relion_sampling_symmetry

    star = tmp_path / "run_it011_sampling.star"
    star.write_text("data_sampling_general\n\n_rlnHealpixOrder 3\n_rlnSymmetryGroup c4\n")
    assert read_relion_sampling_symmetry(star) == "C4"
    star.write_text("data_sampling_general\n\n_rlnHealpixOrder 3\n")
    with pytest.raises(ValueError, match="rlnSymmetryGroup"):
        read_relion_sampling_symmetry(star)
    source = (Path(__file__).resolve().parents[2] / "scripts" / "run_multi_iter_parity.py").read_text()
    assert "symmetry=SymmetryOptions(point_group=point_group)" in source


# ---------------------------------------------------------------------------
# GPU: the resident drivers against the compact / exact local engines at C4
# ---------------------------------------------------------------------------


def _resident_gpu_available() -> bool:
    from test_resident_pass2_driver import _gpu_available

    return _relion_bind_available() and _gpu_available()


requires_resident_gpu = pytest.mark.skipif(
    not _resident_gpu_available(),
    reason="the resident stages are CUDA FFI targets and C4 needs the relion_bind extension",
)


def _c4_driver_args(seed=20260924):
    """The 8x8 production-shaped K=1 fixture of the C1 test, on the C4 grid."""

    from test_resident_pass2_driver import _driver_fixture_args, _z_rotation

    from relax.sampling import rotation_grid_size

    args = _driver_fixture_args(seed=seed)
    nside_level = int(args["nside_level"])
    n_coarse_rot = rotation_grid_size(nside_level, SYMMETRY)
    n_images = int(args["experiment_dataset"].n_units)
    n_coarse_trans = int(np.asarray(args["translations"]).shape[0])
    children = 2
    rng = np.random.default_rng(seed)
    total = n_coarse_rot * n_coarse_trans
    samples = [None]
    for _ in range(1, n_images):
        count = int(rng.integers(1, total))
        samples.append(np.sort(rng.choice(total, size=count, replace=False).astype(np.int32)))
    # Off-axis fine rotations, so each projection differs from its C4 mates.
    fine_rotations = np.stack(
        [
            _z_rotation(0.031 * k)
            @ np.array(
                [[1.0, 0.0, 0.0], [0.0, np.cos(0.4), -np.sin(0.4)], [0.0, np.sin(0.4), np.cos(0.4)]],
                dtype=np.float32,
            )
            for k in range(n_coarse_rot * children)
        ]
    ).astype(np.float32)
    args.update(
        significant_sample_indices=samples,
        rotation_log_prior=rng.normal(scale=0.1, size=n_coarse_rot).astype(np.float32),
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=np.repeat(np.arange(n_coarse_rot, dtype=np.int32), children),
        symmetry_label=SYMMETRY,
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


@requires_resident_gpu
def test_c4_resident_driver_matches_the_compact_engine(_resident_production_env):
    """C4 whole-driver comparison, with the C1 test's measured bounds."""

    from relax.sparse_pass2.sparse_pass2_bucketed import compute_pass2_stats_sparse_bucketed

    args = _c4_driver_args()
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
    for field in (
        "wsum_img_power",
        "wsum_norm_correction",
        "wsum_scale_correction_xa",
        "wsum_scale_correction_aa",
    ):
        assert _rel_l2(getattr(compact.noise_stats, field), getattr(resident.noise_stats, field)) < 1e-6, field

    # The point group was applied, not only the x=0 plane.
    assert _c4_rotation_residual(resident.Ft_ctf) < 1e-5, _rotation_diagnostics(resident.Ft_ctf)
    assert _c4_rotation_residual(compact.Ft_ctf) < 1e-5, _rotation_diagnostics(compact.Ft_ctf)


def _c4_local_case(monkeypatch):
    """The resident-local fixture with the local neighbourhoods built at C4."""

    import test_resident_local_pass2 as local_tests

    from relax.local.local_layout import (
        build_local_adaptive_pass2_hypothesis_layout,
        build_local_hypothesis_layout,
    )
    from relax.sampling import build_local_search_grid_metadata

    seed = 20260919
    case = local_tests._case(seed)
    translations = case["translations"]
    parent = build_local_hypothesis_layout(
        local_tests._prior_eulers(local_tests.N_IMAGES, seed),
        None,
        0.35,
        0.35,
        local_tests.PARENT_ORDER,
        translations,
        np.zeros((local_tests.N_IMAGES, 2), dtype=np.float32),
        3.0,
        None,
        1.0,
        grid_metadata=build_local_search_grid_metadata(local_tests.PARENT_ORDER, symmetry=SYMMETRY),
        translation_prior_reference_translations=translations,
        dtype=np.float32,
    )
    n_coarse_trans = int(translations.shape[0])
    rng = np.random.default_rng(seed + 1)
    samples = []
    for image in range(parent.n_images):
        start = int(parent.rotation_offsets[image])
        stop = int(parent.rotation_offsets[image + 1])
        parent_ids = np.asarray(parent.rotation_ids_flat[start:stop], dtype=np.int64)
        pairs = (parent_ids[:, None] * n_coarse_trans + np.arange(n_coarse_trans)).reshape(-1)
        keep = rng.choice(pairs, size=max(2, pairs.size // 3), replace=False)
        samples.append(np.sort(keep.astype(np.int64)))
    case["layout"] = build_local_adaptive_pass2_hypothesis_layout(
        parent,
        samples,
        local_tests.PARENT_ORDER,
        oversampling_order=local_tests.OVERSAMPLING,
        random_perturbation=0.0,
        dtype=np.float32,
        symmetry=SYMMETRY,
    )
    # The shared runner calls the module-level entry point; give it the run's group.
    run_local = local_search_iteration._run_local_search_iteration

    def run_c4(*args, **kwargs):
        return run_local(*args, symmetry=SYMMETRY, **kwargs)

    monkeypatch.setattr(local_search_iteration, "_run_local_search_iteration", run_c4)
    return local_tests, case


@requires_resident_gpu
def test_c4_resident_local_matches_the_exact_engine(monkeypatch):
    """C4 local fine pass 2 against the exact local engine, with the C1 test's bounds."""

    monkeypatch.setenv("RELAX_LOCAL_SEARCH_RESIDENT_ROW_CAPACITIES", "64,256,1024")
    monkeypatch.setenv("RELAX_LOCAL_SEARCH_RESIDENT_IMAGE_CAPACITIES", "2,4,8")
    local_tests, case = _c4_local_case(monkeypatch)
    assert case["layout"].symmetry == SYMMETRY
    exact = local_tests._run(case, resident=False, monkeypatch=monkeypatch)
    resident = local_tests._run(case, resident=True, monkeypatch=monkeypatch)

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
    assert _c4_rotation_residual(resident.Ft_ctf) < 1e-5, _rotation_diagnostics(resident.Ft_ctf)
    assert _c4_rotation_residual(exact.Ft_ctf) < 1e-5, _rotation_diagnostics(exact.Ft_ctf)
