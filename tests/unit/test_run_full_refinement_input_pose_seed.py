"""Production input-STAR pose initialization for full EM refinement."""

import argparse

import numpy as np
import pandas as pd
import pytest

from relax.relion.input_poses import (
    _add_initial_pose_source_argument,
    _initial_corrections_from_norm,
    _load_input_star_class3d_translations,
    _load_input_star_previous_best_poses,
    _resolve_input_star_pose_seed,
)

pytestmark = pytest.mark.unit


def _particle_tables(*, angstrom_origins=False):
    input_particles = pd.DataFrame(
        {
            "rlnImageName": ["1@stack.mrcs", "2@stack.mrcs", "3@stack.mrcs", "4@stack.mrcs"],
            "rlnAngleRot": [10.0, 20.0, 30.0, 40.0],
            "rlnAngleTilt": [11.0, 21.0, 31.0, 41.0],
            "rlnAnglePsi": [12.0, 22.0, 32.0, 42.0],
        }
    )
    origin_scale = 2.0 if angstrom_origins else 1.0
    suffix = "Angst" if angstrom_origins else ""
    input_particles[f"rlnOriginX{suffix}"] = origin_scale * np.asarray([0.5, 1.5, 2.5, 3.5])
    input_particles[f"rlnOriginY{suffix}"] = origin_scale * np.asarray([-0.5, -1.5, -2.5, -3.5])
    # Deliberately permute the half-set STAR; production mapping must use the
    # complete image identity, not the DataFrame row number.
    halfset_particles = pd.DataFrame(
        {
            "rlnImageName": ["3@stack.mrcs", "1@stack.mrcs", "4@stack.mrcs", "2@stack.mrcs"],
            "rlnRandomSubset": [1, 1, 2, 2],
        }
    )
    return input_particles, halfset_particles


@pytest.mark.parametrize("angstrom_origins", [False, True])
def test_input_star_pose_seed_follows_half_local_order_and_converts_origins(
    angstrom_origins,
):
    input_particles, halfset_particles = _particle_tables(
        angstrom_origins=angstrom_origins,
    )

    seed = _load_input_star_previous_best_poses(
        input_particles,
        halfset_particles,
        half1_idx=np.asarray([2, 0]),
        half2_idx=np.asarray([3, 1]),
        voxel_size=2.0,
    )

    assert seed["iteration"] == "input_star"
    assert seed["translation_units"] == ("angstrom" if angstrom_origins else "pixel")
    np.testing.assert_array_equal(
        seed["previous_best_rotation_eulers"][0],
        np.asarray([[30.0, 31.0, 32.0], [10.0, 11.0, 12.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        seed["previous_best_rotation_eulers"][1],
        np.asarray([[40.0, 41.0, 42.0], [20.0, 21.0, 22.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        seed["previous_best_translations"][0],
        np.asarray([[2.5, -2.5], [0.5, -0.5]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        seed["previous_best_translations"][1],
        np.asarray([[3.5, -3.5], [1.5, -1.5]], dtype=np.float32),
    )


def test_input_star_pose_seed_rejects_stale_half_layout():
    input_particles, halfset_particles = _particle_tables()

    with pytest.raises(ValueError, match="disagrees with RELION rlnRandomSubset"):
        _load_input_star_previous_best_poses(
            input_particles,
            halfset_particles,
            half1_idx=np.asarray([2, 1]),
            half2_idx=np.asarray([3, 0]),
            voxel_size=1.0,
        )


def test_input_star_pose_seed_rejects_nonpartition_and_nonfinite_pose():
    input_particles, halfset_particles = _particle_tables()
    with pytest.raises(ValueError, match="exact partition"):
        _load_input_star_previous_best_poses(
            input_particles,
            halfset_particles,
            half1_idx=np.asarray([2, 0]),
            half2_idx=np.asarray([3]),
            voxel_size=1.0,
        )

    input_particles.loc[2, "rlnAnglePsi"] = np.nan
    with pytest.raises(ValueError, match="Euler-angle values must be finite"):
        _load_input_star_previous_best_poses(
            input_particles,
            halfset_particles,
            half1_idx=np.asarray([2, 0]),
            half2_idx=np.asarray([3, 1]),
            voxel_size=1.0,
        )


def test_input_star_pose_seed_sets_absent_angles_to_zero_like_relion():
    input_particles, halfset_particles = _particle_tables()
    input_particles = input_particles.drop(columns=["rlnAngleRot", "rlnAnglePsi"])

    seed = _load_input_star_previous_best_poses(
        input_particles,
        halfset_particles,
        half1_idx=np.asarray([2, 0]),
        half2_idx=np.asarray([3, 1]),
        voxel_size=1.0,
    )

    np.testing.assert_array_equal(
        seed["previous_best_rotation_eulers"][0],
        np.asarray([[0.0, 31.0, 0.0], [0.0, 11.0, 0.0]], dtype=np.float32),
    )


def test_input_star_norm_corrections_follow_half_local_order():
    input_particles, halfset_particles = _particle_tables()
    input_particles["rlnNormCorrection"] = [0.5, 0.8, 1.25, 2.0]

    seed = _load_input_star_previous_best_poses(
        input_particles,
        halfset_particles,
        half1_idx=np.asarray([2, 0]),
        half2_idx=np.asarray([3, 1]),
        voxel_size=1.0,
    )

    np.testing.assert_array_equal(seed["norm_corrections"][0], np.asarray([1.25, 0.5]))
    np.testing.assert_array_equal(seed["norm_corrections"][1], np.asarray([2.0, 0.8]))
    image_corrections, scale_corrections = _initial_corrections_from_norm(seed["norm_corrections"])
    np.testing.assert_array_equal(image_corrections[0], np.asarray([0.8, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(image_corrections[1], np.asarray([0.5, 1.25], dtype=np.float32))
    for half in scale_corrections:
        np.testing.assert_array_equal(half, np.ones(2, dtype=np.float32))


def test_input_star_without_norm_corrections_starts_at_unit_like_relion():
    input_particles, halfset_particles = _particle_tables()

    seed = _load_input_star_previous_best_poses(
        input_particles,
        halfset_particles,
        half1_idx=np.asarray([2, 0]),
        half2_idx=np.asarray([3, 1]),
        voxel_size=1.0,
    )

    for half in seed["norm_corrections"]:
        np.testing.assert_array_equal(half, np.ones(2))
    assert _initial_corrections_from_norm(seed["norm_corrections"]) == (None, None)


@pytest.mark.parametrize("bad_value", [0.0, -1.0, np.nan])
def test_input_star_norm_corrections_reject_nonpositive_or_nonfinite(bad_value):
    input_particles, halfset_particles = _particle_tables()
    input_particles["rlnNormCorrection"] = [1.0, bad_value, 1.0, 1.0]

    with pytest.raises(ValueError, match="norm-correction|rlnNormCorrection"):
        _load_input_star_previous_best_poses(
            input_particles,
            halfset_particles,
            half1_idx=np.asarray([2, 0]),
            half2_idx=np.asarray([3, 1]),
            voxel_size=1.0,
        )


def test_initial_pose_source_cli_defaults_to_fresh_k1_halfset_auto():
    parser = argparse.ArgumentParser()
    _add_initial_pose_source_argument(parser)

    assert parser.parse_args([]).initial_pose_source == "auto"
    assert parser.parse_args(["--initial-pose-source", "input-star"]).initial_pose_source == "input-star"
    assert _resolve_input_star_pose_seed(
        "auto",
        n_classes=1,
        init_relion_iteration=0,
        has_relion_half_sets=True,
        has_competing_pose_source=False,
        diagnostic_single_half=False,
    )
    assert not _resolve_input_star_pose_seed(
        "auto",
        n_classes=4,
        init_relion_iteration=0,
        has_relion_half_sets=True,
        has_competing_pose_source=False,
        diagnostic_single_half=False,
    )


def test_explicit_input_star_pose_source_fails_closed_on_competing_state():
    with pytest.raises(ValueError, match="already owns initialization"):
        _resolve_input_star_pose_seed(
            "input-star",
            n_classes=1,
            init_relion_iteration=0,
            has_relion_half_sets=True,
            has_competing_pose_source=True,
            diagnostic_single_half=False,
        )


@pytest.mark.parametrize("angstrom_origins", [False, True])
def test_class3d_input_origins_follow_all_data_order_without_orientations(angstrom_origins):
    input_particles, _ = _particle_tables(angstrom_origins=angstrom_origins)
    seed = _load_input_star_class3d_translations(
        input_particles,
        np.asarray([2, 0, 3, 1]),
        voxel_size=2.0,
    )
    assert seed["previous_best_rotation_eulers"] == [None, None]
    assert seed["translation_units"] == ("angstrom" if angstrom_origins else "pixel")
    first, second = seed["previous_best_translations"]
    np.testing.assert_array_equal(
        first,
        np.asarray([[2.5, -2.5], [0.5, -0.5], [3.5, -3.5], [1.5, -1.5]], dtype=np.float32),
    )
    assert first.dtype == np.float32
    assert second.shape == (0, 2) and second.dtype == np.float32


def test_class3d_input_origins_absent_are_zero_and_rows_must_cover_the_input():
    input_particles, _ = _particle_tables()
    input_particles = input_particles.drop(columns=["rlnOriginX", "rlnOriginY"])
    seed = _load_input_star_class3d_translations(input_particles, np.arange(4), voxel_size=1.0)
    assert seed["translation_units"] == "implicit_zero"
    np.testing.assert_array_equal(seed["previous_best_translations"][0], np.zeros((4, 2), np.float32))
    with pytest.raises(ValueError, match="permutation"):
        _load_input_star_class3d_translations(input_particles, np.arange(3), voxel_size=1.0)


def test_class3d_input_star_origins_are_the_default_and_need_no_half_sets():
    kwargs = dict(
        n_classes=4,
        init_relion_iteration=0,
        has_competing_pose_source=False,
        diagnostic_single_half=False,
    )
    assert _resolve_input_star_pose_seed("input-star", has_relion_half_sets=False, **kwargs)
    assert _resolve_input_star_pose_seed("auto", has_relion_half_sets=False, **kwargs)
    assert not _resolve_input_star_pose_seed("none", has_relion_half_sets=False, **kwargs)
    # A replay/diagnostic pose source keeps ownership; 'auto' then stays unseeded.
    assert not _resolve_input_star_pose_seed(
        "auto", has_relion_half_sets=False, **(kwargs | {"has_competing_pose_source": True})
    )
    with pytest.raises(ValueError, match="splits no random halves"):
        _resolve_input_star_pose_seed("input-star", has_relion_half_sets=True, **kwargs)
    with pytest.raises(ValueError, match="already owns initialization"):
        _resolve_input_star_pose_seed(
            "input-star", has_relion_half_sets=False, **(kwargs | {"has_competing_pose_source": True})
        )
