"""Production input-STAR pose initialization for full EM refinement."""

import argparse
import hashlib
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import starfile
from helpers.float_compare import assert_matches

from relax.relion.input_poses import (
    _add_initial_pose_source_argument,
    _initial_corrections_from_norm,
    _load_input_star_class3d_translations,
    _load_input_star_previous_best_poses,
    _resolve_input_star_pose_seed,
    prepare_initial_poses,
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


def _pose_preparation_inputs(tmp_path, *, n_classes=1, angstrom_origins=False, norm_corrections=False):
    particles, halfset = _particle_tables(angstrom_origins=angstrom_origins)
    if norm_corrections:
        particles["rlnNormCorrection"] = [0.5, 0.8, 1.25, 2.0]
    starfile.write(particles, tmp_path / "particles.star")
    layout = SimpleNamespace(
        half1_rows=np.array([2, 0]) if n_classes == 1 else np.array([2, 0, 3, 1]),
        half2_rows=np.array([3, 1]) if n_classes == 1 else np.array([], dtype=np.int64),
    )
    return particles, dict(
        particle_layout=layout, relion_halfset_particles=halfset, pixel_size_angstrom=2.0,
        data_dir=tmp_path, requested_source="auto", n_classes=n_classes, init_relion_iteration=0,
        has_relion_half_sets=n_classes == 1, diagnostic_single_half=False, frozen_boundary=None,
        poses_npz_path=None, pose_iteration="last", has_replay_pose_source=False,
        class3d_translations=None, class3d_translation_path=None, log=logging.getLogger("test.initial_poses"),
    )


@pytest.mark.parametrize("n_classes", [1, 4])
@pytest.mark.parametrize("angstrom_origins", [False, True])
@pytest.mark.parametrize("norm_corrections", [False, True])
@pytest.mark.parametrize("requested", ["auto", "input-star", "none"])
def test_prepared_input_poses_pair_source_half_order_units_and_corrections(
    tmp_path, n_classes, angstrom_origins, norm_corrections, requested,
):
    particles, inputs = _pose_preparation_inputs(
        tmp_path, n_classes=n_classes, angstrom_origins=angstrom_origins, norm_corrections=norm_corrections,
    )
    inputs["requested_source"] = requested
    prepared = prepare_initial_poses(particles, **inputs)
    assert prepared.provenance.requested_source == requested
    if requested == "none":
        assert prepared.poses is None
        assert prepared.image_corrections is None and prepared.scale_corrections is None
        assert prepared.provenance.resolved_source == "none"
        assert prepared.provenance.path is None and prepared.provenance.sha256 is None
        return
    path = tmp_path / "particles.star"
    assert prepared.provenance.path == path.resolve()
    assert prepared.provenance.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert prepared.poses["translation_units"] == ("angstrom" if angstrom_origins else "pixel")
    if n_classes == 4:
        assert prepared.provenance.resolved_source == "input_star_translations"
        assert prepared.poses["previous_best_rotation_eulers"] == [None, None]
        assert prepared.poses["previous_best_translations"][1].shape == (0, 2)
        assert_matches(prepared.poses["previous_best_translations"][0], [[2.5, -2.5], [0.5, -0.5], [3.5, -3.5], [1.5, -1.5]])
        if norm_corrections:
            # relion_refine scales each image by 1 / rlnNormCorrection in Class3D too (all-data order 2, 0, 3, 1).
            assert_matches(prepared.image_corrections[0], np.array([0.8, 2.0, 0.5, 1.25], dtype=np.float32))
            assert prepared.image_corrections[1].shape == (0,)
            assert_matches(prepared.scale_corrections[0], np.ones(4, dtype=np.float32))
        else:
            assert prepared.image_corrections is None and prepared.scale_corrections is None
    else:
        assert prepared.provenance.resolved_source == "input_star"
        assert_matches(prepared.poses["previous_best_rotation_eulers"][0], [[30, 31, 32], [10, 11, 12]])
        assert_matches(prepared.poses["previous_best_rotation_eulers"][1], [[40, 41, 42], [20, 21, 22]])
        assert_matches(prepared.poses["previous_best_translations"][0], [[2.5, -2.5], [0.5, -0.5]])
        assert_matches(prepared.poses["previous_best_translations"][1], [[3.5, -3.5], [1.5, -1.5]])
        for norms in prepared.poses["norm_corrections"]:
            assert norms.dtype == np.float64
        if norm_corrections:
            assert_matches(prepared.image_corrections[0], np.array([0.8, 2.0], dtype=np.float32))
            assert_matches(prepared.image_corrections[1], np.array([0.5, 1.25], dtype=np.float32))
            assert all(values.dtype == np.float32 for values in prepared.image_corrections)
            for scales in prepared.scale_corrections:
                assert_matches(scales, np.ones(2, dtype=np.float32))
        else:
            assert prepared.image_corrections is None and prepared.scale_corrections is None
    assert all(values.dtype == np.float32 for values in prepared.poses["previous_best_translations"])


@pytest.mark.parametrize("requested", ["auto", "none"])
def test_frozen_pose_arrays_keep_precedence_and_aliases(tmp_path, requested):
    particles, inputs = _pose_preparation_inputs(tmp_path)
    eulers = (np.arange(6, dtype=np.float64).reshape(2, 3), np.zeros((2, 3), dtype=np.float32))
    translations = (np.zeros((2, 2), dtype=np.float64), np.ones((2, 2), dtype=np.float32))
    inputs.update(
        requested_source=requested,
        frozen_boundary=SimpleNamespace(
            completed_relion_iteration=4, previous_best_rotation_eulers=eulers, previous_best_translations=translations,
        ),
        poses_npz_path=tmp_path / "missing.npz", has_replay_pose_source=True,
    )
    (tmp_path / "particles.star").unlink()
    prepared = prepare_initial_poses(particles, **inputs)
    assert prepared.poses["iteration"] == "003"
    for half in range(2):
        assert prepared.poses["previous_best_rotation_eulers"][half] is eulers[half]
        assert prepared.poses["previous_best_translations"][half] is translations[half]
    assert prepared.image_corrections is None and prepared.scale_corrections is None
    assert prepared.provenance.resolved_source == "diagnostic_replay"
    assert prepared.provenance.path is None and prepared.provenance.sha256 is None


def test_npz_pose_source_keeps_numbered_selection_and_source_labels(tmp_path, caplog):
    particles, inputs = _pose_preparation_inputs(tmp_path)
    path = tmp_path / "poses.npz"
    arrays = {}
    for iteration in (1, 3):
        for half in range(2):
            arrays[f"best_rotation_eulers_iter_{iteration:03d}_half{half}"] = np.full((2, 3), iteration + half, np.float64)
            arrays[f"best_translations_iter_{iteration:03d}_half{half}"] = np.full((2, 2), -iteration - half, np.float64)
    np.savez(path, **arrays)
    inputs.update(poses_npz_path=path, has_replay_pose_source=True)
    (tmp_path / "particles.star").unlink()
    with caplog.at_level(logging.INFO):
        prepared = prepare_initial_poses(particles, **inputs)
    assert prepared.poses["iteration"] == "003"
    for half in range(2):
        assert_matches(prepared.poses["previous_best_rotation_eulers"][half], np.full((2, 3), 3 + half, np.float32))
        assert_matches(prepared.poses["previous_best_translations"][half], np.full((2, 2), -3 - half, np.float32))
        assert prepared.poses["previous_best_rotation_eulers"][half].dtype == np.float32
    assert prepared.provenance.resolved_source == "diagnostic_replay"
    assert prepared.provenance.path is None and prepared.provenance.sha256 is None
    assert any("Diagnostic local-search seed" in record.message and "iter=003" in record.message for record in caplog.records)


@pytest.mark.parametrize("requested", ["auto", "none"])
def test_class3d_replay_translation_seed_keeps_source_and_no_orientation_seed(tmp_path, requested):
    particles, inputs = _pose_preparation_inputs(tmp_path, n_classes=4)
    path = tmp_path / "run_it000_data.star"
    path.write_bytes((tmp_path / "particles.star").read_bytes())
    translations = [np.arange(8, dtype=np.float32).reshape(4, 2), np.empty((0, 2), np.float32)]
    inputs.update(
        requested_source=requested, class3d_translations=translations,
        class3d_translation_path=path, has_replay_pose_source=True,
    )
    prepared = prepare_initial_poses(particles, **inputs)
    assert prepared.poses["iteration"] == "000_translation_only"
    assert prepared.poses["previous_best_translations"] is translations
    assert prepared.poses["previous_best_rotation_eulers"] == [None, None]
    assert prepared.provenance.resolved_source == "relion_run_it000_translations"
    assert prepared.provenance.path is path
    assert prepared.provenance.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("override, message", [
    ({"has_replay_pose_source": True}, "already owns initialization"),
    ({"init_relion_iteration": 1}, "fresh"),
    ({"has_relion_half_sets": False}, "requires --relion_half_sets"),
    ({"diagnostic_single_half": True}, "both gold-standard halves"),
])
def test_explicit_input_pose_source_preserves_admission_errors(tmp_path, override, message):
    particles, inputs = _pose_preparation_inputs(tmp_path)
    inputs.update(requested_source="input-star", **override)
    with pytest.raises(SystemExit, match="Invalid initial pose source:.*" + message):
        prepare_initial_poses(particles, **inputs)


def test_prepared_pose_input_errors_keep_cli_context(tmp_path):
    particles, inputs = _pose_preparation_inputs(tmp_path)
    particles.loc[2, "rlnAnglePsi"] = np.nan
    with pytest.raises(SystemExit, match="Invalid input-STAR pose initialization:.*finite"):
        prepare_initial_poses(particles, **inputs)


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
    assert_matches(
        seed["previous_best_rotation_eulers"][0],
        np.asarray([[30.0, 31.0, 32.0], [10.0, 11.0, 12.0]], dtype=np.float32),
    )
    assert_matches(
        seed["previous_best_rotation_eulers"][1],
        np.asarray([[40.0, 41.0, 42.0], [20.0, 21.0, 22.0]], dtype=np.float32),
    )
    assert_matches(
        seed["previous_best_translations"][0],
        np.asarray([[2.5, -2.5], [0.5, -0.5]], dtype=np.float32),
    )
    assert_matches(
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

    assert_matches(
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

    assert_matches(seed["norm_corrections"][0], np.asarray([1.25, 0.5]))
    assert_matches(seed["norm_corrections"][1], np.asarray([2.0, 0.8]))
    image_corrections, scale_corrections = _initial_corrections_from_norm(seed["norm_corrections"])
    assert_matches(image_corrections[0], np.asarray([0.8, 2.0], dtype=np.float32))
    assert_matches(image_corrections[1], np.asarray([0.5, 1.25], dtype=np.float32))
    for half in scale_corrections:
        assert_matches(half, np.ones(2, dtype=np.float32))


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
        assert_matches(half, np.ones(2))
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
    assert_matches(
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
    assert_matches(seed["previous_best_translations"][0], np.zeros((4, 2), np.float32))
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


def test_class3d_replay_translation_seed_centres_local_searches_on_the_input_angles(tmp_path):
    # With --sigma_ang RELION searches locally from iteration 1 around each particle's input angles, 0 where
    # the STAR has none (exp_model.cpp:1104-1134, ml_optimiser.cpp:978-983); the RELION-seeded Class3D start
    # left them unset, so iteration 1 searched globally without a word.
    particles, inputs = _pose_preparation_inputs(tmp_path, n_classes=4)
    path = tmp_path / "run_it000_data.star"
    path.write_bytes((tmp_path / "particles.star").read_bytes())
    translations = [np.arange(8, dtype=np.float32).reshape(4, 2), np.empty((0, 2), np.float32)]
    inputs.update(
        requested_source="auto", class3d_translations=translations,
        class3d_translation_path=path, has_replay_pose_source=True, local_search_at_start=True,
    )
    prepared = prepare_initial_poses(particles, **inputs)
    eulers = prepared.poses["previous_best_rotation_eulers"]
    rows = inputs["particle_layout"].half1_rows
    expected = np.stack(
        [
            particles[c].to_numpy(dtype=np.float64) if c in particles else np.zeros(len(particles))
            for c in ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")
        ],
        axis=1,
    )[rows]
    assert_matches(np.asarray(eulers[0], dtype=np.float64), expected.astype(np.float32).astype(np.float64))
    assert eulers[1].shape == (0, 3)

