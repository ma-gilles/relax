"""relax's port of relion_align_symmetry (relax.vdam.align_symmetry) against RELION 5.0.1's program.

RELION draws its orientations from the time, so its runs differ; the oracle (fixture set align_symmetry_relion_c4)
holds five reruns per input with the refined angles each run logged. At a logged orientation the search objective
and the output map are deterministic and are compared with RELION's; the seeded search is compared with the spread
of RELION's runs.
"""

from __future__ import annotations

import re

import jax.numpy as jnp
import mrcfile
import numpy as np
import pytest
import starfile
from helpers.em_fixtures import fixture_dir

from relax.healpix_sampling import euler_angles_to_matrix
from relax.symmetry import relion_symmetry_operators
from relax.vdam import align_symmetry as align

pytestmark = pytest.mark.unit

CASES = ("k1_c4_s109", "k4_c4_s53")
RUNS = range(1, 6)
# RELION logs ROT/TILT/PSI and DIFF2 to 6 significant digits; at the logged angles relax's objective agreed to
# 2.8e-6 relative and the map to 1.6e-4 relative L2 over the 10 runs (both set by the logged precision).
DIFF2_RTOL = 2e-5
MAP_REL_L2 = 5e-4
# RELION's refined DIFF2 over 5 reruns spans 5.957-5.972 (k1) and 20.179-20.328 (k4); relax's seeded searches
# (seeds 1-3, 29) gave 5.956-6.009 and 20.269-20.302.
SEARCH_DIFF2_SLACK = 0.01


def _search_diff2(volume, sym_name, eulers):
    """align_symmetry's search objective for (rot, tilt, psi) rows: re-centre, crop, rotate, symmetrise, diff2."""

    volume = align.recentre_on_centre_of_mass(volume)
    box = min(align.WORKING_BOX, volume.shape[0])
    _, right = relion_symmetry_operators(sym_name)
    matrices = jnp.asarray(euler_angles_to_matrix(np.asarray(eulers, dtype=np.float64)))
    data = align._projector_data(align.resize_map(volume, box))
    return np.asarray(align._diff2_batch(data, matrices, jnp.asarray(right[1:]), n=box))


def _rotate_and_symmetrise(volume, sym_name, eulers):
    """align_symmetry's output for a given (rot, tilt, psi): the re-centred full map rotated and symmetrised."""

    _, right = relion_symmetry_operators(sym_name)
    volume = align.recentre_on_centre_of_mass(volume)
    matrix = jnp.asarray(euler_angles_to_matrix([eulers])[0])
    rotated = align._rotate(align._projector_data(volume), matrix, n=volume.shape[0])
    return np.asarray(align._symmetrise(rotated, jnp.asarray(right[1:])))


def _largest_class_map(case_dir):
    classes = starfile.read(case_dir / "run_it200_model.star")["model_classes"]
    k = align.select_largest_class(classes["rlnClassDistribution"])
    return np.asarray(mrcfile.read(case_dir / classes["rlnReferenceImage"][k]), dtype=np.float64)


def _relion_run(case_dir, run):
    log = (case_dir / f"relion_align_symmetry_r{run}.log").read_text()
    match = re.search(r"refined solution is ROT = (\S+) TILT = (\S+) PSI = (\S+) DIFF2= (\S+)", log)
    eulers = [float(match.group(i)) for i in (1, 2, 3)]
    output = np.asarray(mrcfile.read(case_dir / f"relion_initial_model_r{run}.mrc"), dtype=np.float64)
    return eulers, float(match.group(4)), output


@pytest.mark.parametrize("case", CASES)
def test_objective_and_output_match_relion_at_its_logged_orientation(case):
    case_dir = fixture_dir("align_symmetry_relion_c4") / case
    volume = _largest_class_map(case_dir)
    runs = [_relion_run(case_dir, run) for run in RUNS]

    diff2 = _search_diff2(volume, "C4", [eulers for eulers, _, _ in runs])

    for (eulers, relion_diff2, relion_output), ours in zip(runs, diff2):
        assert abs(ours / relion_diff2 - 1.0) <= DIFF2_RTOL
        output = _rotate_and_symmetrise(volume, "C4", eulers)
        assert np.linalg.norm(output - relion_output) / np.linalg.norm(relion_output) <= MAP_REL_L2


@pytest.mark.parametrize("case", CASES)
def test_seeded_search_reaches_relion_runs_objective(case):
    case_dir = fixture_dir("align_symmetry_relion_c4") / case
    volume = _largest_class_map(case_dir)
    relion_diff2 = [_relion_run(case_dir, run)[1] for run in RUNS]

    output, report = align.align_symmetry(volume, "C4", seed=29)

    assert report["refined_diff2"] <= max(relion_diff2) * (1.0 + SEARCH_DIFF2_SLACK)
    assert report["refined_diff2"] <= report["global_diff2"]
    # The output is the map at the reported orientation.
    expected = _rotate_and_symmetrise(volume, "C4", report["refined_rot_tilt_psi"])
    assert np.linalg.norm(output - expected) / np.linalg.norm(expected) <= 1e-9


def _c4_test_map(n=48):
    c = np.arange(n) - n // 2
    z, y, x = np.meshgrid(c, c, c, indexing="ij")

    def blob(cx, cy, cz, s):
        return np.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2 * s * s))

    # Four copies of a two-blob unit at 90 degrees about the z axis through voxel n // 2, plus two axial blobs
    # that make the map C4 but not D4.
    copies = []
    for k in range(4):
        c, s = np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)
        copies.append(blob(8 * c - 2 * s, 8 * s + 2 * c, 3, 2.5) + 0.7 * blob(5 * c + 6 * s, 5 * s - 6 * c, -3, 2.0))
    return sum(copies) + blob(0, 0, -7, 3) + 0.5 * blob(0, 0, 8, 2)


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def _about_z(volume, degrees):
    phi = np.deg2rad(degrees)
    rz = np.array([[np.cos(phi), -np.sin(phi), 0.0], [np.sin(phi), np.cos(phi), 0.0], [0.0, 0.0, 1.0]])
    return np.asarray(align._apply_geometry(volume, rz, np.zeros(3)))


def _best_about_z(a, b):
    """Largest correlation of ``a`` with ``b`` rotated about z (1 degree steps) or turned upside down."""

    flipped = np.roll(b[::-1, :, ::-1], (1, 0, 1), axis=(0, 1, 2))  # 180 degrees about y on the even grid
    best = -1.0
    for candidate in (b, flipped):
        for degrees in np.arange(0.0, 90.0, 1.0):
            best = max(best, _corr(a, _about_z(candidate, degrees)))
    return best


def test_off_axis_c4_map_comes_back_on_axis_and_symmetric():
    on_axis = align.recentre_on_centre_of_mass(_c4_test_map())
    tilted = np.asarray(
        align._rotate(align._projector_data(on_axis), euler_angles_to_matrix([[33.0, 57.0, -20.0]])[0], n=48)
    )
    assert _corr(on_axis, _about_z(on_axis, 90.0)) > 0.999
    assert _corr(tilted, _about_z(tilted, 90.0)) < 0.7

    output, _ = align.align_symmetry(tilted, "C4", seed=29)

    # Measured over seeds 1, 2 and 29: self-correlation 0.9999991-0.9999994, match to the on-axis map 0.99973-0.99975.
    assert _corr(output, _about_z(output, 90.0)) > 0.99999
    assert _best_about_z(output, on_axis) > 0.999


def test_c1_writes_the_selected_map_unchanged():
    volume = _c4_test_map(16)

    output, report = align.align_symmetry(volume, "C1", seed=29)

    np.testing.assert_allclose(output, volume, rtol=1e-12, atol=0.0)
    assert report == {"sym": "C1"}


def test_recentre_moves_the_positive_centre_of_mass_to_the_centre():
    n = 32
    c = np.arange(n) - n // 2
    z, y, x = np.meshgrid(c, c, c, indexing="ij")
    volume = np.exp(-((x - 3) ** 2 + (y + 2) ** 2 + (z - 4) ** 2) / 8.0) - 0.05

    centred = align.recentre_on_centre_of_mass(volume)

    positive = np.where(centred > 0, centred, 0.0)
    for grid in (z, y, x):
        assert abs((positive * grid).sum() / positive.sum()) < 0.05


def test_select_largest_class_takes_the_first_maximum():
    assert align.select_largest_class([0.2, 0.4, 0.4]) == 1


@pytest.mark.parametrize("sym_name", ["C1", "C4"])
def test_final_output_is_the_largest_class_aligned_like_relion_gui(tmp_path, sym_name):
    from recovar.utils.helpers import recovar_volume_to_relion

    from relax.vdam import output
    from relax.vdam.bootstrap_iref import initialise_denovo_state

    n = 32
    rng = np.random.default_rng(0)
    maps = np.stack([rng.standard_normal((n,) * 3), recovar_volume_to_relion(_c4_test_map(n))]).astype(np.float32)
    state = initialise_denovo_state(ori_size=n, pixel_size=2.5, K=2, nr_iter=200, n_directions=4)
    state.iter, state.Iref, state.pdf_class = 200, maps, np.array([0.3, 0.7])

    final_mrc, class_mrcs, report = output._write_final_outputs(
        str(tmp_path / "run"), state, sym_name=sym_name, seed=29
    )

    assert final_mrc == str(tmp_path / "initial_model.mrc") and len(class_mrcs) == 2
    written = np.asarray(mrcfile.read(final_mrc), dtype=np.float64)
    relion_frame = np.asarray(mrcfile.read(class_mrcs[1]), dtype=np.float64)
    expected, _ = align.align_symmetry(relion_frame, sym_name, seed=29)
    assert report["class"] == 2
    assert np.linalg.norm(written - expected) / np.linalg.norm(expected) <= 1e-6  # float32 file
    if sym_name == "C4":
        assert _corr(written, _about_z(written, 90.0)) > 0.99999
