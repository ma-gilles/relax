"""RELION's start-up particle table rebuilt from relion_refine's input STAR."""

import ctypes
import ctypes.util

import numpy as np
import pandas as pd
import pytest
import starfile
from helpers.em_fixtures import fixture_file
from helpers.float_compare import assert_matches

from relax.relion.input_particle_table import (
    GroupParticleSource,
    _read_replay_group_particles,
    build_relion_start_particle_table,
    glibc_first_rand,
    glibc_rand_sequence,
    prepare_class3d_particle_layout,
    prepare_particle_group_layout,
    prepare_relion_halfset_layout,
    relion_class3d_seed_classes,
    relion_particle_order,
    relion_random_subsets,
    relion_scale_group_numbers,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("random_seed", [None, 1711])
@pytest.mark.parametrize("first_iteration", [1, 3])
@pytest.mark.parametrize("with_optics", [False, True])
def test_particle_layout_keeps_source_rows_accuracy_ids_and_optics_paired(
    random_seed, first_iteration, with_optics,
):
    names = [f"{row // 3 + 1}@stack{row % 3}.mrcs" for row in range(18)]
    relion = pd.DataFrame({"rlnImageName": names, "rlnRandomSubset": [1, 2] * 9})
    if with_optics:
        relion["rlnOpticsGroup"] = [3, 1, 2] * 6
    order = np.random.default_rng(27).permutation(len(relion))
    input_particles = relion.iloc[order].reset_index(drop=True)
    layout = prepare_relion_halfset_layout(
        input_particles, relion, random_seed=random_seed, first_iteration=first_iteration,
    )
    assert np.array_equal(np.sort(np.r_[layout.half1_rows, layout.half2_rows]), np.arange(len(relion)))
    assert np.all(input_particles.iloc[layout.half1_rows]["rlnRandomSubset"] == 1)
    assert np.all(input_particles.iloc[layout.half2_rows]["rlnRandomSubset"] == 2)
    assert np.array_equal(
        np.asarray(input_particles.iloc[layout.half1_rows]["rlnImageName"]),
        np.asarray(relion.iloc[layout.accuracy_particle_ids]["rlnImageName"]),
    )
    expected_optics = (np.asarray(relion.iloc[layout.accuracy_particle_ids]["rlnOpticsGroup"])
                       if with_optics else np.zeros(layout.half1_rows.size, dtype=np.int64))
    assert np.array_equal(layout.accuracy_optics_group_ids, expected_optics)
    if random_seed is None:
        assert layout.accuracy_trial_order_local is None
        replay_names = np.asarray(input_particles.iloc[layout.half1_rows]["rlnImageName"])[layout.accuracy_base_order_local]
        assert np.array_equal(replay_names, np.asarray(relion.loc[relion.rlnRandomSubset == 1, "rlnImageName"]))
    else:
        assert layout.accuracy_base_order_local is None
        assert np.array_equal(layout.accuracy_trial_order_local, np.arange(layout.half1_rows.size))
        if with_optics:
            assert np.all(np.diff(layout.accuracy_optics_group_ids) >= 0)
    for field in layout:
        if field is not None:
            assert field.dtype == np.int64


@pytest.mark.parametrize("init_relion_iteration", [0, 2])
@pytest.mark.parametrize("random_seed", [0, 1711])
@pytest.mark.parametrize("with_optics", [False, True])
def test_class3d_particle_layout_keeps_all_data_and_source_trial_identity(
    init_relion_iteration, random_seed, with_optics,
):
    particles = pd.DataFrame({
        "rlnImageName": [f"{row + 1}@images.mrcs" for row in range(12)],
        "rlnMicrographName": ["z.mrc", "a.mrc", "m.mrc"] * 4,
    })
    if with_optics:
        particles["rlnOpticsGroup"] = [2, 3, 1] * 4
    layout = prepare_class3d_particle_layout(
        particles, n_particles=len(particles), random_seed=random_seed,
        init_relion_iteration=init_relion_iteration,
    )
    assert np.array_equal(layout.half1_rows, np.arange(len(particles)))
    assert layout.half2_rows.shape == (0,) and layout.half2_rows.dtype == np.int64
    assert layout.accuracy_base_order_local is layout.accuracy_optics_group_ids is None
    assert np.array_equal(np.sort(layout.accuracy_trial_order_local), np.arange(len(particles)))
    sorted_rows = np.asarray(particles.sort_values("rlnMicrographName", kind="stable").index)
    assert np.array_equal(layout.accuracy_particle_ids[sorted_rows], np.arange(len(particles)))
    if with_optics:
        optics = np.asarray(particles.rlnOpticsGroup)[layout.accuracy_trial_order_local]
        assert np.all(np.diff(optics) >= 0)


def test_fresh_group_source_has_named_absence_and_derives_input_groups():
    particles = pd.DataFrame({
        "rlnImageName": ["000001@stack.mrcs", "000002@stack.mrcs"],
        "rlnMicrographName": ["job010/a.mrc", "job010/b.mrc"],
    })
    prepared = prepare_particle_group_layout(particles, [0, 1], [])
    source, groups = prepared
    assert isinstance(source, GroupParticleSource)
    assert source.particles is source.path is None
    np.testing.assert_array_equal(groups.group_ids_per_half[0], [0, 1])
    np.testing.assert_array_equal(groups.optics_group_ids_per_half[0], [0, 0])
    assert groups.group_ids_per_half[1].shape == (0,)
    assert groups.n_groups == 2 and groups.n_optics_groups == 1

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
    assert_matches(glibc_rand_sequence(seed, 2000), expected)


def _libc():
    name = ctypes.util.find_library("c")
    libc = ctypes.CDLL(name) if name else None
    if libc is None or not hasattr(libc, "gnu_get_libc_version"):
        pytest.skip("glibc is required as the reference")
    return libc


def test_class3d_seed_classes_are_libc_rand_after_srand_of_seed_plus_sorted_position():
    """RELION's single-reference Class3D seed: srand(random_seed + j), rand() % K for sorted position j."""

    libc = _libc()
    seeds = np.concatenate([[0, 1, 2**31 - 1, 2**31, 2**32 - 1], np.random.default_rng(3).integers(0, 2**32, 64)])
    expected = []
    for seed in seeds:
        libc.srand(ctypes.c_uint(int(seed)))
        expected.append(libc.rand())
    assert_matches(glibc_first_rand(seeds), np.asarray(expected, dtype=np.int64))

    order = np.random.default_rng(5).permutation(50)
    classes = relion_class3d_seed_classes(order, 7, 3)
    for j, row in enumerate(order):
        libc.srand(ctypes.c_uint(7 + j))
        assert classes[row] == libc.rand() % 3


def test_order_split_and_groups_follow_relion_rules():
    particles = pd.DataFrame(
        {
            "rlnImageName": [f"{i}@s.mrcs" for i in range(1, 6)],
            "rlnMicrographName": ["b", "MotionCorr/job003/Movies/a.mrc", "b", "10", "1"],
        }
    )
    order = relion_particle_order(particles)
    # Byte-wise std::string order ("1" < "10" < "M..." < "b"), stable for the two "b" rows.
    assert_matches(order, [4, 3, 1, 0, 2])
    table = build_relion_start_particle_table(particles, seed=2)
    assert_matches(table["rlnRandomSubset"], glibc_rand_sequence(2, 5) % 2 + 1)
    # srand(7) puts all five particles in half 2; RELION stops on an empty half.
    with pytest.raises(ValueError, match="both random halves"):
        build_relion_start_particle_table(particles, seed=7)
    # The pipeline job directory is stripped from the group name, and ids follow first appearance.
    assert_matches(table["rlnGroupNumber"], [1, 2, 3, 4, 4])
    assert relion_scale_group_numbers(pd.DataFrame({"rlnGroupName": ["g2", "g1", "g2"]})).tolist() == [1, 2, 1]


def test_input_random_subsets_are_kept_and_validated():
    assert_matches(relion_random_subsets([2, 1, 2], seed=3, n_particles=3), [2, 1, 2])
    with pytest.raises(ValueError, match="all zero or all non-zero"):
        relion_random_subsets([1, 0, 2], seed=3, n_particles=3)
    with pytest.raises(ValueError, match="must be 1 or 2"):
        relion_random_subsets([1, 3, 2], seed=3, n_particles=3)
    assert_matches(
        relion_random_subsets([0, 0, 0, 0], seed=5, n_particles=4), glibc_rand_sequence(5, 4) % 2 + 1
    )


@pytest.mark.parametrize("case", sorted(RELION_START_CASES))
def test_rebuilt_table_reproduces_relion_run_it000(case):
    from relax.refinement.startup_noise import auto_refine_noise_order

    input_star, target_star, optimiser_star = _case_paths(case)
    particles = starfile.read(input_star, always_dict=True)["particles"]
    target = starfile.read(target_star, always_dict=True)["particles"]
    seed = _relion_random_seed(optimiser_star)

    table = build_relion_start_particle_table(particles, seed=seed)

    assert_matches(table["rlnImageName"].astype(str), target["rlnImageName"].astype(str))
    assert_matches(table["rlnRandomSubset"].to_numpy(int), target["rlnRandomSubset"].to_numpy(int))
    assert_matches(table["rlnGroupNumber"].to_numpy(int), target["rlnGroupNumber"].to_numpy(int))
    rebuilt_noise = auto_refine_noise_order(particles, table)
    relion_noise = auto_refine_noise_order(particles, target)
    for rebuilt, relion in zip(rebuilt_noise, relion_noise, strict=True):
        assert_matches(rebuilt, relion)


def test_from_input_flag_rejects_combinations_relion_would_not_build():
    from types import SimpleNamespace

    from relax.refinement.command_options import validate_input_half_sets

    base = dict(relion_half_sets_from_input=True, relion_half_sets=None, n_classes=1, seed=11, frozen_boundary_dir=None)
    validate_input_half_sets(SimpleNamespace(**base))
    for change, message in (
        ({"relion_half_sets": "run_data.star"}, "replaces --relion_half_sets"),
        ({"n_classes": 4}, "K=1"),
        ({"frozen_boundary_dir": "boundary"}, "frozen boundary"),
    ):
        with pytest.raises(SystemExit, match=message):
            validate_input_half_sets(SimpleNamespace(**{**base, **change}))


def test_fresh_k1_start_without_relion_output_resolves_to_standalone():
    """A fresh K=1 start rebuilds RELION's particle table; debug starts supply their own half sets."""
    from types import SimpleNamespace

    from relax.refinement.command_options import resolve_standalone_k1_start

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
        resolve_standalone_k1_start(args)
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

    from relax.refinement.command_options import find_relion_optimiser_star

    discovered = tmp_path / "relion_ref_os0" / "run_optimiser.star"
    discovered.parent.mkdir()
    discovered.write_text("data_optimiser_general\n")
    base = dict(relion_optimiser=None, relion_init_dir=None, perturb_replay_relion_dir=None, relion_half_sets=None)
    args = SimpleNamespace(data_dir=str(tmp_path), relion_half_sets_from_input=False, **base)
    assert find_relion_optimiser_star(args) == discovered.resolve()
    args.relion_half_sets_from_input = True
    assert find_relion_optimiser_star(args) is None
    args.relion_optimiser = str(discovered)
    assert find_relion_optimiser_star(args) == discovered.resolve()


def test_class3d_without_relion_state_does_not_discover_relion_optimiser_outputs(tmp_path):
    from types import SimpleNamespace

    from relax.refinement.command_options import find_relion_optimiser_star

    discovered = tmp_path / "relion_pdb_k4_os0_ref" / "run_it015_optimiser.star"
    discovered.parent.mkdir()
    discovered.write_text("data_optimiser_general\n")
    base = dict(relion_optimiser=None, relion_init_dir=None, perturb_replay_relion_dir=None, relion_half_sets=None)
    args = SimpleNamespace(data_dir=str(tmp_path), relion_half_sets_from_input=False, n_classes=4, **base)
    assert find_relion_optimiser_star(args) is None
    args.perturb_replay_relion_dir = str(discovered.parent)
    assert find_relion_optimiser_star(args) == discovered.resolve()
    args.perturb_replay_relion_dir = None
    args.relion_optimiser = str(discovered)
    assert find_relion_optimiser_star(args) == discovered.resolve()


def test_written_table_round_trips_input_values(tmp_path):
    from relax.refinement.full_refinement import _write_relion_start_particle_table

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
        assert_matches(written["particles"][column].to_numpy(), source[column].to_numpy())
    for column in our_star["optics"].columns:
        assert_matches(written["optics"][column].to_numpy(), our_star["optics"][column].to_numpy())


def test_native_group_layout_prefers_supplied_relion_groups_and_maps_exact_identities():
    pd = pytest.importorskip("pandas")
    our_particles = pd.DataFrame(
        {
            "rlnImageName": [
                "3@stack_a.mrcs",
                "1@stack_a.mrcs",
                "4@stack_a.mrcs",
                "2@stack_a.mrcs",
            ],
        },
    )
    relion_particles = pd.DataFrame(
        {
            "rlnImageName": [
                "1@stack_a.mrcs",
                "2@stack_a.mrcs",
                "3@stack_a.mrcs",
                "4@stack_a.mrcs",
            ],
            "rlnGroupNumber": [2, 4, 1, 7],
            "rlnOpticsGroup": [1, 2, 1, 3],
        },
    )

    layout = prepare_particle_group_layout(
        our_particles,
        half1_idx=np.asarray([0, 1], dtype=np.int64),
        half2_idx=np.asarray([2, 3], dtype=np.int64),
        halfset_particles=relion_particles,
    ).layout

    assert layout is not None
    assert layout.source == "supplied RELION data STAR"
    assert layout.n_groups == 7
    assert layout.n_optics_groups == 3
    assert_matches(layout.group_ids_per_half[0], [0, 1])
    assert_matches(layout.group_ids_per_half[1], [6, 3])
    # Internal IDs are authoritative RELION data-STAR row numbers, mapped by
    # full rlnImageName identity rather than RECOVAR row position.
    assert_matches(layout.particle_ids_per_half[0], [2, 0])
    assert_matches(layout.particle_ids_per_half[1], [3, 1])
    assert_matches(layout.optics_group_ids_per_half[0], [0, 0])
    assert_matches(layout.optics_group_ids_per_half[1], [2, 1])


def test_replay_group_loader_uses_iter0_without_relion_half_sets(tmp_path):
    pd = pytest.importorskip("pandas")
    starfile = pytest.importorskip("starfile")
    particles = pd.DataFrame(
        {
            "rlnImageName": ["2@x.mrcs", "1@x.mrcs"],
            "rlnGroupNumber": [7, 3],
        }
    )
    starfile.write({"particles": particles}, tmp_path / "run_it000_data.star")

    loaded, source = _read_replay_group_particles(tmp_path)

    assert source == tmp_path / "run_it000_data.star"
    assert_matches(loaded["rlnGroupNumber"], [7, 3])


def test_subset_only_halfset_does_not_block_permuted_replay_group_layout(tmp_path):
    pd = pytest.importorskip("pandas")
    starfile = pytest.importorskip("starfile")
    our_particles = pd.DataFrame(
        {"rlnImageName": ["3@x.mrcs", "1@x.mrcs", "2@x.mrcs"]}
    )
    subset_only = pd.DataFrame(
        {
            "rlnImageName": ["1@x.mrcs", "2@x.mrcs", "3@x.mrcs"],
            "rlnRandomSubset": [1, 2, 1],
        }
    )
    replay_particles = pd.DataFrame(
        {
            "rlnImageName": ["2@x.mrcs", "3@x.mrcs", "1@x.mrcs"],
            "rlnGroupNumber": [7, 4, 2],
        }
    )
    starfile.write({"particles": replay_particles}, tmp_path / "run_it000_data.star")

    prepared = prepare_particle_group_layout(
        our_particles,
        half1_idx=np.asarray([0, 1], dtype=np.int64),
        half2_idx=np.asarray([2], dtype=np.int64),
        halfset_particles=subset_only,
        halfset_source=tmp_path / "halfsets.star",
        replay_dirs=(tmp_path,),
    )
    source, layout = prepared.source.path, prepared.layout

    assert source == tmp_path / "run_it000_data.star"
    assert layout is not None
    assert layout.n_groups == 7
    assert_matches(layout.group_ids_per_half[0], [3, 1])
    assert_matches(layout.group_ids_per_half[1], [6])


def test_native_group_layout_preserves_full_group_axis_when_half_max_is_absent():
    pd = pytest.importorskip("pandas")
    particles = pd.DataFrame(
        {
            "rlnImageName": ["1@x.mrcs", "2@x.mrcs", "3@x.mrcs", "4@x.mrcs"],
            "rlnGroupNumber": [1, 2, 7, 4],
        },
    )

    layout = prepare_particle_group_layout(
        particles,
        half1_idx=np.asarray([0, 1], dtype=np.int64),
        half2_idx=np.asarray([2, 3], dtype=np.int64),
        halfset_particles=particles,
    ).layout

    assert layout is not None
    assert layout.n_groups == 7
    assert int(np.max(layout.group_ids_per_half[0])) == 1
    assert_matches(layout.group_ids_per_half[1], [6, 3])


def test_native_group_layout_numbers_relion_groups_when_the_star_has_none():
    """Without rlnGroupNumber anywhere, groups follow relion_refine, not a silent None.

    RELION sorts by micrograph name and numbers groups by first appearance of
    rlnGroupName, else the post-job micrograph name (exp_model.cpp:900-901, 926-965).
    """
    pd = pytest.importorskip("pandas")
    our_particles = pd.DataFrame(
        {
            "rlnImageName": ["1@x.mrcs", "2@x.mrcs", "3@x.mrcs", "4@x.mrcs"],
            "rlnMicrographName": ["Extract/job007/mic_b.mrc", "Extract/job007/mic_a.mrc",
                                  "Extract/job007/mic_b.mrc", "Extract/job007/mic_c.mrc"],
        },
    )

    layout = prepare_particle_group_layout(
        our_particles,
        half1_idx=np.asarray([0, 1], dtype=np.int64),
        half2_idx=np.asarray([2, 3], dtype=np.int64),
    ).layout

    assert layout is not None
    assert layout.n_groups == 3
    # Sorted order: mic_a (row 1), mic_b (rows 0, 2), mic_c (row 3).
    assert_matches(layout.group_ids_per_half[0], [1, 0])
    assert_matches(layout.group_ids_per_half[1], [1, 2])
    assert_matches(layout.particle_ids_per_half[0], [1, 0])
    assert_matches(layout.particle_ids_per_half[1], [2, 3])

    # An input rlnGroupNumber that disagrees with RELION's numbering is ignored,
    # as relion_refine ignores it (exp_model.cpp:963-965).
    stale = our_particles.assign(rlnGroupNumber=[9, 4, 2, 7])
    renumbered = prepare_particle_group_layout(
        stale,
        half1_idx=np.asarray([0, 1], dtype=np.int64),
        half2_idx=np.asarray([2, 3], dtype=np.int64),
    ).layout
    assert renumbered.n_groups == 3
    assert_matches(renumbered.group_ids_per_half[0], [1, 0])
    assert_matches(renumbered.group_ids_per_half[1], [1, 2])

    # No group or micrograph name: RELION reads an empty micrograph name for
    # every particle (exp_model.cpp:154-167), so all share one group.
    single = prepare_particle_group_layout(
        our_particles[["rlnImageName"]],
        half1_idx=np.asarray([0, 1], dtype=np.int64),
        half2_idx=np.asarray([2, 3], dtype=np.int64),
    ).layout
    assert single.n_groups == 1
    assert_matches(np.concatenate(single.group_ids_per_half), [0, 0, 0, 0])


@pytest.mark.parametrize(
    ("relion_names", "message"),
    [
        (["1@x.mrcs", "1@x.mrcs"], "duplicate rlnImageName/stack identities"),
        (["1@x.mrcs", "2@other.mrcs"], "do not contain the same rlnImageName/stack identities"),
    ],
)
def test_native_group_layout_rejects_duplicate_or_missing_relion_identities(relion_names, message):
    pd = pytest.importorskip("pandas")
    our_particles = pd.DataFrame({"rlnImageName": ["1@x.mrcs", "2@x.mrcs"]})
    relion_particles = pd.DataFrame(
        {"rlnImageName": relion_names, "rlnGroupNumber": [1, 2]},
    )

    with pytest.raises(ValueError, match=message):
        prepare_particle_group_layout(
            our_particles,
            half1_idx=np.asarray([0], dtype=np.int64),
            half2_idx=np.asarray([1], dtype=np.int64),
            halfset_particles=relion_particles,
        )
