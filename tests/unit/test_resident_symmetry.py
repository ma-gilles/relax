"""Point-group symmetry on the device-resident K=1 local pass 2 and the replay harness.

RELION restricts the orientation sampling to the asymmetric unit
(healpix_sampling.cpp removeSymmetryEquivalentPoints) and symmetrises each
BPref after the expectation (ml_optimiser.cpp:5541-5575
symmetriseReconstructions: enforceHermitianSymmetry, then
applyPointGroupSymmetry). The resident local pass 2 finalizes with the same
``finalize_half_volume_bpref`` the exact local engine uses, and the replay
harness takes the group from RELION's sampling STAR.

The CPU tests check the wiring; the GPU test compares the resident local pass
against the exact local engine on a C4 pass and checks that the finalized
weights are C4-invariant.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

from relax.refinement import local_search_iteration
from relax.sparse_pass2 import resident_local_pass2 as rlp

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
    """

    values = np.asarray(ctf_public_flat, dtype=np.float64)
    size = int(round(values.size ** (1.0 / 3.0)))
    assert size**3 == values.size, "expected the full public layout of an odd cube"
    grid = values.reshape(size, size, size)
    return _rel_l2(grid, np.rot90(grid, k=1, axes=(0, 1)))


# ---------------------------------------------------------------------------
# CPU: grid plumbing and wiring
# ---------------------------------------------------------------------------


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
# GPU: the resident local pass 2 against the exact local engine at C4
# ---------------------------------------------------------------------------


def _resident_gpu_available() -> bool:
    from test_resident_pass2_driver import _gpu_available

    return _relion_bind_available() and _gpu_available()


requires_resident_gpu = pytest.mark.skipif(
    not _resident_gpu_available(),
    reason="the resident stages are CUDA FFI targets and C4 needs the relion_bind extension",
)


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
    np.testing.assert_allclose(
        np.asarray(exact.best_pose_eulers_deg, dtype=np.float64),
        np.asarray(resident.best_pose_eulers_deg, dtype=np.float64),
        rtol=0,
        atol=1e-9,
    )
    assert _rel_l2(exact.Ft_y, resident.Ft_y) < 1e-5
    assert _rel_l2(exact.Ft_ctf, resident.Ft_ctf) < 1e-5
    np.testing.assert_allclose(
        np.asarray(exact.relion_stats.max_posterior_per_image, dtype=np.float64),
        np.asarray(resident.relion_stats.max_posterior_per_image, dtype=np.float64),
        rtol=0,
        atol=1e-5,
    )
    assert _c4_rotation_residual(resident.Ft_ctf) < 1e-5
    assert _c4_rotation_residual(exact.Ft_ctf) < 1e-5
