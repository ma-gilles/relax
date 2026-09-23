"""RELION's start-up particle table rebuilt from relion_refine's input STAR."""

import ctypes
import ctypes.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import starfile

from relax.relion.input_particle_table import (
    build_relion_start_particle_table,
    glibc_rand_sequence,
    relion_particle_order,
    relion_random_subsets,
    relion_scale_group_numbers,
)

pytestmark = pytest.mark.unit

# (relion_refine --i STAR, RELION output dir whose run_it000_data.star is the target)
_EM = Path("/scratch/gpfs/GILLES/mg6942/em_relion_proj")
_REAL = Path("/scratch/gpfs/CRYOEM/gilleslab/em_work")
RELION_START_CASES = {
    "k1_5k_128": (_EM / "data_noise1_5k_normalized/particles.star", _EM / "data_noise1_5k_normalized/relion_ref_os0"),
    "k1_50k_256": (
        _EM / "data_noise1_50k_256_normalized/particles.star",
        _EM / "data_noise1_50k_256_normalized/relion_ref_os0",
    ),
    "k1_100k_256": (
        _EM / "pdb_k1_g256_n100000_noise1_bf80_20260516/particles.star",
        _EM / "pdb_k1_g256_n100000_noise1_bf80_20260516/relion_autorefine_k1_it015_os1",
    ),
    **{
        f"real_{ds}": (root / "initialmodel/run_it200_data.star", root / "refine")
        for ds, root in (
            ("10097", _REAL / "realdata_em_10097_replacement_20260828T211155EDT/outputs/10097/relion"),
            ("10073", _REAL / "realdata_em_full_native_20260828T082038EDT/outputs/10073/relion"),
            ("10345", _REAL / "realdata_em_full_native_20260828T082038EDT/outputs/10345/relion"),
        )
    },
}


def _relion_random_seed(relion_dir):
    for line in (relion_dir / "run_it000_optimiser.star").read_text().splitlines():
        if line.strip().startswith("_rlnRandomSeed "):
            return int(line.split()[1])
    raise AssertionError(f"no _rlnRandomSeed in {relion_dir}")


@pytest.mark.parametrize("seed", [0, 1, 42, 1775735620, 2**31 - 1, 2**31, 2**32 - 1])
def test_glibc_rand_sequence_matches_libc(seed):
    name = ctypes.util.find_library("c")
    libc = ctypes.CDLL(name) if name else None
    if libc is None or not hasattr(libc, "gnu_get_libc_version"):
        pytest.skip("glibc is required as the reference")
    libc.srand(ctypes.c_uint(seed))
    expected = np.asarray([libc.rand() for _ in range(2000)], dtype=np.int64)
    np.testing.assert_array_equal(glibc_rand_sequence(seed, 2000), expected)


def test_order_split_and_groups_follow_relion_rules():
    particles = pd.DataFrame(
        {
            "rlnImageName": [f"{i}@s.mrcs" for i in range(1, 6)],
            "rlnMicrographName": ["b", "MotionCorr/job003/Movies/a.mrc", "b", "10", "1"],
        }
    )
    order = relion_particle_order(particles)
    # Byte-wise std::string order ("1" < "10" < "M..." < "b"), stable for the two "b" rows.
    np.testing.assert_array_equal(order, [4, 3, 1, 0, 2])
    table = build_relion_start_particle_table(particles, seed=2)
    np.testing.assert_array_equal(table["rlnRandomSubset"], glibc_rand_sequence(2, 5) % 2 + 1)
    # srand(7) puts all five particles in half 2; RELION stops on an empty half.
    with pytest.raises(ValueError, match="both random halves"):
        build_relion_start_particle_table(particles, seed=7)
    # The pipeline job directory is stripped from the group name, and ids follow first appearance.
    np.testing.assert_array_equal(table["rlnGroupNumber"], [1, 2, 3, 4, 4])
    assert relion_scale_group_numbers(pd.DataFrame({"rlnGroupName": ["g2", "g1", "g2"]})).tolist() == [1, 2, 1]


def test_input_random_subsets_are_kept_and_validated():
    np.testing.assert_array_equal(relion_random_subsets([2, 1, 2], seed=3, n_particles=3), [2, 1, 2])
    with pytest.raises(ValueError, match="all zero or all non-zero"):
        relion_random_subsets([1, 0, 2], seed=3, n_particles=3)
    with pytest.raises(ValueError, match="must be 1 or 2"):
        relion_random_subsets([1, 3, 2], seed=3, n_particles=3)
    np.testing.assert_array_equal(
        relion_random_subsets([0, 0, 0, 0], seed=5, n_particles=4), glibc_rand_sequence(5, 4) % 2 + 1
    )


@pytest.mark.parametrize("case", sorted(RELION_START_CASES))
def test_rebuilt_table_reproduces_relion_run_it000(case):
    from scripts.run_full_refinement import _default_refinement_subsets, _relion_fresh_initial_noise_layout

    input_star, relion_dir = RELION_START_CASES[case]
    target_star = relion_dir / "run_it000_data.star"
    if not input_star.exists() or not target_star.exists():
        pytest.skip(f"missing fixture for {case}")
    particles = starfile.read(input_star, always_dict=True)["particles"]
    target = starfile.read(target_star, always_dict=True)["particles"]
    seed = _relion_random_seed(relion_dir)

    table = build_relion_start_particle_table(particles, seed=seed)

    np.testing.assert_array_equal(table["rlnImageName"].astype(str), target["rlnImageName"].astype(str))
    np.testing.assert_array_equal(table["rlnRandomSubset"].to_numpy(int), target["rlnRandomSubset"].to_numpy(int))
    np.testing.assert_array_equal(table["rlnGroupNumber"].to_numpy(int), target["rlnGroupNumber"].to_numpy(int))
    rebuilt_noise = _relion_fresh_initial_noise_layout(particles, table)
    relion_noise = _relion_fresh_initial_noise_layout(particles, target)
    for rebuilt, relion in zip(rebuilt_noise, relion_noise, strict=True):
        np.testing.assert_array_equal(rebuilt, relion)

    if "rlnRandomSubset" not in particles.columns:
        # The seeded fallback RECOVAR used without a RELION STAR is not RELION's split.
        half1, _half2 = _default_refinement_subsets(len(particles), seed, 1)
        fallback = np.full(len(particles), 2)
        fallback[half1] = 1
        by_name = dict(zip(target["rlnImageName"].astype(str), target["rlnRandomSubset"].to_numpy(int)))
        relion_in_input_order = np.asarray([by_name[name] for name in particles["rlnImageName"].astype(str)])
        assert np.mean(fallback == relion_in_input_order) < 0.6


def test_from_input_flag_rejects_combinations_relion_would_not_build():
    from types import SimpleNamespace

    from scripts.run_full_refinement import _validate_relion_half_sets_from_input

    base = dict(relion_half_sets_from_input=True, relion_half_sets=None, n_classes=1, seed=11, frozen_boundary_dir=None)
    _validate_relion_half_sets_from_input(SimpleNamespace(**base))
    for change, message in (
        ({"relion_half_sets": "run_data.star"}, "replaces --relion_half_sets"),
        ({"n_classes": 4}, "K=1"),
        ({"seed": None}, "explicit --seed"),
        ({"frozen_boundary_dir": "boundary"}, "frozen boundary"),
    ):
        with pytest.raises(SystemExit, match=message):
            _validate_relion_half_sets_from_input(SimpleNamespace(**{**base, **change}))


def test_written_table_round_trips_input_values(tmp_path):
    from scripts.run_full_refinement import _write_relion_start_particle_table

    input_star, relion_dir = RELION_START_CASES["k1_5k_128"]
    if not input_star.exists():
        pytest.skip("missing 5k fixture")
    our_star = starfile.read(input_star, always_dict=True)
    path = _write_relion_start_particle_table(
        our_star, input_star, seed=_relion_random_seed(relion_dir), output_dir=tmp_path
    )
    written = starfile.read(path, always_dict=True)
    order = relion_particle_order(our_star["particles"])
    source = our_star["particles"].iloc[order].reset_index(drop=True)
    for column in source.columns:
        if column == "rlnRandomSubset":
            continue
        np.testing.assert_array_equal(written["particles"][column].to_numpy(), source[column].to_numpy())
    for column in our_star["optics"].columns:
        np.testing.assert_array_equal(written["optics"][column].to_numpy(), our_star["optics"][column].to_numpy())
