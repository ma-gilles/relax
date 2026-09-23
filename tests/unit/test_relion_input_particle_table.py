"""RELION's start-up particle table rebuilt from relion_refine's input STAR."""

import ctypes
import ctypes.util

import numpy as np
import pandas as pd
import pytest
import starfile
from helpers.em_fixtures import fixture_file

from relax.relion.input_particle_table import (
    build_relion_start_particle_table,
    glibc_rand_sequence,
    relion_particle_order,
    relion_random_subsets,
    relion_scale_group_numbers,
)

pytestmark = pytest.mark.unit

# case: (fixture set, relion_refine --i STAR), (fixture set, directory of the RELION run_it000 stars)
RELION_START_CASES = {
    "k1_5k_128": (("k1_5k128_data", "particles.star"), ("k1_5k128_relion_os0", "")),
    "k1_50k_256": (("k1_50k256_data", "particles.star"), ("k1_50k256_relion_os0", "")),
    "k1_100k_256": (("k1_100k256_data", "particles.star"), ("k1_100k256_relion", "")),
    **{
        f"real_{ds}": ((f"empiar_{ds}_relion_startup", "initialmodel/run_it200_data.star"), (f"empiar_{ds}_relion_startup", "refine/"))
        for ds in ("10097", "10073", "10345")
    },
}


def _case_paths(case):
    """Verified (input STAR, RELION run_it000_data.star, RELION run_it000_optimiser.star) of a case."""
    (input_set, input_rel), (relion_set, prefix) = RELION_START_CASES[case]
    return (
        fixture_file(input_set, input_rel),
        fixture_file(relion_set, prefix + "run_it000_data.star"),
        fixture_file(relion_set, prefix + "run_it000_optimiser.star"),
    )


def _relion_random_seed(optimiser_star):
    for line in optimiser_star.read_text().splitlines():
        if line.strip().startswith("_rlnRandomSeed "):
            return int(line.split()[1])
    raise AssertionError(f"no _rlnRandomSeed in {optimiser_star}")


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
    from scripts.run_full_refinement import _relion_fresh_initial_noise_layout

    input_star, target_star, optimiser_star = _case_paths(case)
    particles = starfile.read(input_star, always_dict=True)["particles"]
    target = starfile.read(target_star, always_dict=True)["particles"]
    seed = _relion_random_seed(optimiser_star)

    table = build_relion_start_particle_table(particles, seed=seed)

    np.testing.assert_array_equal(table["rlnImageName"].astype(str), target["rlnImageName"].astype(str))
    np.testing.assert_array_equal(table["rlnRandomSubset"].to_numpy(int), target["rlnRandomSubset"].to_numpy(int))
    np.testing.assert_array_equal(table["rlnGroupNumber"].to_numpy(int), target["rlnGroupNumber"].to_numpy(int))
    rebuilt_noise = _relion_fresh_initial_noise_layout(particles, table)
    relion_noise = _relion_fresh_initial_noise_layout(particles, target)
    for rebuilt, relion in zip(rebuilt_noise, relion_noise, strict=True):
        np.testing.assert_array_equal(rebuilt, relion)


def test_from_input_flag_rejects_combinations_relion_would_not_build():
    from types import SimpleNamespace

    from scripts.run_full_refinement import _validate_relion_half_sets_from_input

    base = dict(relion_half_sets_from_input=True, relion_half_sets=None, n_classes=1, seed=11, frozen_boundary_dir=None)
    _validate_relion_half_sets_from_input(SimpleNamespace(**base))
    for change, message in (
        ({"relion_half_sets": "run_data.star"}, "replaces --relion_half_sets"),
        ({"n_classes": 4}, "K=1"),
        ({"frozen_boundary_dir": "boundary"}, "frozen boundary"),
    ):
        with pytest.raises(SystemExit, match=message):
            _validate_relion_half_sets_from_input(SimpleNamespace(**{**base, **change}))


def test_fresh_k1_start_without_relion_output_resolves_to_standalone():
    """A fresh K=1 start rebuilds RELION's particle table; debug starts supply their own half sets."""
    from types import SimpleNamespace

    from scripts.run_full_refinement import _resolve_standalone_k1_start

    def resolve(**change):
        args = SimpleNamespace(
            **{
                **dict(
                    relion_half_sets_from_input=None,
                    relion_half_sets=None,
                    relion_init_dir=None,
                    perturb_replay_relion_dir=None,
                    frozen_boundary_dir=None,
                    init_relion_iteration=0,
                    n_classes=1,
                    seed=None,
                ),
                **change,
            }
        )
        _resolve_standalone_k1_start(args)
        return args.relion_half_sets_from_input

    assert resolve() is True
    assert resolve(relion_half_sets="run_it000_data.star") is False
    assert resolve(relion_init_dir="relion_ref") is False
    assert resolve(perturb_replay_relion_dir="relion_ref") is False
    assert resolve(frozen_boundary_dir="boundary") is False
    assert resolve(init_relion_iteration=3) is False
    assert resolve(n_classes=4) is False
    assert resolve(relion_half_sets_from_input=False) is False
    with pytest.raises(SystemExit, match="K=1"):
        resolve(n_classes=4, relion_half_sets_from_input=True)


def test_from_input_flag_does_not_discover_relion_optimiser_outputs(tmp_path):
    from types import SimpleNamespace

    from scripts.run_full_refinement import _find_relion_optimiser_star

    discovered = tmp_path / "relion_ref_os0" / "run_optimiser.star"
    discovered.parent.mkdir()
    discovered.write_text("data_optimiser_general\n")
    base = dict(relion_optimiser=None, relion_init_dir=None, perturb_replay_relion_dir=None, relion_half_sets=None)
    args = SimpleNamespace(data_dir=str(tmp_path), relion_half_sets_from_input=False, **base)
    assert _find_relion_optimiser_star(args) == discovered.resolve()
    args.relion_half_sets_from_input = True
    assert _find_relion_optimiser_star(args) is None
    args.relion_optimiser = str(discovered)
    assert _find_relion_optimiser_star(args) == discovered.resolve()


def test_class3d_without_relion_state_does_not_discover_relion_optimiser_outputs(tmp_path):
    from types import SimpleNamespace

    from scripts.run_full_refinement import _find_relion_optimiser_star

    discovered = tmp_path / "relion_pdb_k4_os0_ref" / "run_it015_optimiser.star"
    discovered.parent.mkdir()
    discovered.write_text("data_optimiser_general\n")
    base = dict(relion_optimiser=None, relion_init_dir=None, perturb_replay_relion_dir=None, relion_half_sets=None)
    args = SimpleNamespace(data_dir=str(tmp_path), relion_half_sets_from_input=False, n_classes=4, **base)
    assert _find_relion_optimiser_star(args) is None
    args.perturb_replay_relion_dir = str(discovered.parent)
    assert _find_relion_optimiser_star(args) == discovered.resolve()
    args.perturb_replay_relion_dir = None
    args.relion_optimiser = str(discovered)
    assert _find_relion_optimiser_star(args) == discovered.resolve()


def test_written_table_round_trips_input_values(tmp_path):
    from scripts.run_full_refinement import _write_relion_start_particle_table

    input_star, _target_star, optimiser_star = _case_paths("k1_5k_128")
    our_star = starfile.read(input_star, always_dict=True)
    path = _write_relion_start_particle_table(
        our_star, input_star, seed=_relion_random_seed(optimiser_star), output_dir=tmp_path
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
