"""Source Euler metadata survives publication without changing compute operands."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest
from helpers.float_compare import assert_matches
from recovar import utils

from relax.dense import score_outputs

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_particle_pose_interpretation_preserves_metadata_and_offset_frame(monkeypatch, dtype):
    from relax.refinement.particle_poses import resolve_particle_poses

    eulers = _source_eulers(2)
    rotations = np.asarray(utils.R_from_relion(eulers, degrees=True), dtype=dtype)
    relative_shifts = np.array([[0.25, 1.0], [-0.5, 1.5]], dtype=dtype)

    def refuse_inverse(*args, **kwargs):
        raise AssertionError("Supplied source Euler metadata must not be re-derived")

    monkeypatch.setattr(utils, "R_to_relion", refuse_inverse)
    poses = resolve_particle_poses(
        np.array([0, 0]),
        np.zeros((1, 2)),
        previous_translations=np.array([[1.2, -3.2], [1.7, 0.2]]),
        own_pixel_factors=None,
        best_pose_rotations=rotations,
        best_pose_rotation_eulers=eulers,
        best_pose_translations=relative_shifts,
        pose_rotations=None,
        local_sampling=None,
        dtype=dtype,
    )

    assert poses.rotations is rotations
    assert poses.eulers_deg is eulers
    assert poses.relative_translations_pixels is relative_shifts
    assert poses.eulers_deg.dtype == np.float64
    assert poses.translations_pixels.dtype == dtype
    assert_matches(poses.translations_pixels, [[1.25, -2.0], [1.5, 1.5]])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_particle_pose_interpretation_decodes_dense_grid_ids(dtype):
    from relax.refinement.particle_poses import resolve_particle_poses

    rotations = np.asarray(utils.R_from_relion(_source_eulers(3), degrees=True), dtype=dtype)
    translations = np.array([[0.25, -0.5], [1.0, 0.75]], dtype=dtype)
    poses = resolve_particle_poses(
        np.array([5, 0, 3], dtype=np.int32),
        translations,
        previous_translations=None,
        own_pixel_factors=None,
        best_pose_rotations=None,
        best_pose_rotation_eulers=None,
        best_pose_translations=None,
        pose_rotations=rotations,
        local_sampling=None,
        dtype=dtype,
    )

    assert_matches(poses.rotations, rotations[[2, 0, 1]])
    assert_matches(poses.relative_translations_pixels, translations[[1, 0, 1]])
    assert poses.translations_pixels is poses.relative_translations_pixels
    assert poses.eulers_deg.dtype == dtype


def _source_eulers(n):
    return np.column_stack(
        (
            np.linspace(-143.123456789, 132.987654321, n),
            np.linspace(23.123456789, 151.987654321, n),
            np.linspace(-87.123456789, 69.987654321, n),
        )
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("second_half_count", [0, 1])
@pytest.mark.parametrize("has_previous", [False, True])
def test_pose_transition_preserves_half_identity_snapshots_and_offset_frames(
    dtype, dimension, second_half_count,
    has_previous,
):
    from relax.refinement.half_inputs import HalfSet
    from relax.refinement.particle_poses import prepare_particle_pose_update, prepare_pose_comparison

    counts = (2, second_half_count)
    scores = score_outputs.PerHalfOutputs()
    halves = []
    previous_rotations = []
    for k, n in enumerate(counts):
        scores.hard_assignments[k] = np.zeros(n, dtype=np.int32)
        scores.best_pose_rotations[k] = np.tile(np.eye(3, dtype=dtype), (n, 1, 1))
        scores.best_pose_rotation_eulers[k] = _source_eulers(n)
        scores.best_pose_translations[k] = (
            np.arange(n * dimension).reshape(n, dimension) / 4 + k
        ).astype(dtype)
        translations = (
            np.arange(n * dimension).reshape(n, dimension) / 3 + 0.123 + k
            if has_previous else None
        )
        halves.append(HalfSet(index=k, dataset=object(), translations=translations))
        previous_rotations.append(
            np.tile(np.eye(3, dtype=np.float64), (n, 1, 1)) if has_previous else None
        )

    update = prepare_particle_pose_update(
        scores, halves, np.zeros((1, dimension), dtype=dtype),
        previous_rotations=previous_rotations, local_sampling=None, dtype=dtype,
    )
    comparison = prepare_pose_comparison(
        update, translation_dimension=dimension, dtype=dtype,
        log=logging.getLogger(__name__),
    )

    for k, poses in enumerate(update.current):
        assert poses.rotations is scores.best_pose_rotations[k]
        assert poses.eulers_deg is scores.best_pose_rotation_eulers[k]
        assert poses.eulers_deg.dtype == np.float64
        assert poses.relative_translations_pixels is scores.best_pose_translations[k]
        expected = poses.relative_translations_pixels
        if has_previous:
            assert not np.shares_memory(update.previous_rotations[k], previous_rotations[k])
            assert not np.shares_memory(update.previous_translations_pixels[k], halves[k].translations)
            assert_matches(update.previous_rotations[k], previous_rotations[k])
            assert_matches(update.previous_translations_pixels[k], halves[k].translations)
            expected = expected + np.round(halves[k].translations).astype(dtype)
        else:
            assert update.previous_rotations[k] is None
            assert update.previous_translations_pixels[k] is None
        assert_matches(poses.translations_pixels, expected)
        assert poses.translations_pixels.dtype == dtype

    assert_matches(comparison.current_rotations, np.concatenate([p.rotations for p in update.current]))
    assert_matches(comparison.current_translations_pixels, np.concatenate([p.translations_pixels for p in update.current]))
    assert comparison.current_translations_pixels.shape == (sum(counts), dimension)
    if has_previous:
        assert_matches(comparison.previous_rotations, np.concatenate(previous_rotations).astype(dtype))
        assert_matches(comparison.previous_translations_pixels, np.concatenate([h.translations for h in halves]).astype(dtype))
        halves[0].translations[:] = 99
        previous_rotations[0][:] = 99
        assert np.all(update.previous_translations_pixels[0] < 99)
        assert np.all(update.previous_rotations[0] < 99)
    else:
        assert comparison.previous_rotations is None
        assert comparison.previous_translations_pixels is None


def test_pose_transition_snapshots_both_halves_before_resolving_either(monkeypatch):
    from relax.refinement import half_inputs, particle_poses

    halves = [
        half_inputs.HalfSet(index=k, dataset=object(), translations=np.full((1, 2), k + 1.0))
        for k in range(2)
    ]
    calls = []

    def resolve(assignments, grid, *, previous_translations, **kwargs):
        calls.append(previous_translations.copy())
        halves[1].translations[:] = 100
        return particle_poses.ParticlePoses(
            rotations=np.eye(3)[None, ...], eulers_deg=np.zeros((1, 3)),
            relative_translations_pixels=grid[:1], translations_pixels=previous_translations,
        )

    monkeypatch.setattr(particle_poses, "resolve_particle_poses", resolve)
    outputs = score_outputs.PerHalfOutputs(hard_assignments=[np.zeros(1, dtype=np.int32)] * 2)
    update = particle_poses.prepare_particle_pose_update(
        outputs, halves, np.zeros((1, 2)),
        previous_rotations=[None, None], local_sampling=None, dtype=np.float32,
    )

    assert_matches(calls[0], [[1.0, 1.0]])
    assert_matches(calls[1], [[2.0, 2.0]])
    assert_matches(update.previous_translations_pixels[1], [[2.0, 2.0]])


def _kclass_result():
    # Every class wins once, in an order different from the class axis.
    winners = np.array([3, 0, 2, 1], dtype=np.int32)
    source = _source_eulers(16).reshape(4, 4, 3)
    selected = source[winners, np.arange(4)]
    return SimpleNamespace(
        class_assignments=winners,
        pose_assignments=np.array([8, 2, 9, 4], dtype=np.int32),
        best_pose_eulers_deg=selected,
        per_class_best_pose_eulers_deg=source,
        best_pose_rotations=np.asarray(utils.R_from_relion(selected, degrees=True), dtype=np.float32),
        best_pose_translations=np.arange(8, dtype=np.float32).reshape(4, 2),
        per_class_stats=[SimpleNamespace(rotation_posterior_sums=np.arange(3) + i) for i in range(4)],
        class_posterior_sums=np.array([0.5, 1.5, 0.75, 1.25]),
        class_mstep_posterior_sums=np.array([0.4, 1.3, 0.7, 1.2]),
        noise_stats=object(),
        aggregate_noise_stats=object(),
        stats=SimpleNamespace(
            max_posterior_per_image=np.ones(4, dtype=np.float32),
            rotation_posterior_sums=np.ones(3, dtype=np.float32),
        ),
        Ft_y=np.arange(12).reshape(4, 3),
        Ft_ctf=np.arange(12, 24).reshape(4, 3),
    )


def _publish(result, half, dtype):
    outputs = score_outputs.PerHalfOutputs()
    returned = score_outputs.class_em_to_half_result(
        result,
        effective_rotations=np.zeros((3, 3, 3)),
        rot_pmap_for_collapse=None,
        adaptive_os_local=0,
        pose_dtype=dtype,
    )
    outputs.update_from(half, returned, dtype=dtype)
    return outputs, returned


def test_class_result_without_pose_details_preserves_existing_collector_poses():
    raw = _kclass_result()
    raw.best_pose_rotations = raw.best_pose_translations = None
    result = score_outputs.class_em_to_half_result(
        raw, effective_rotations=np.zeros((3, 3, 3)),
        rot_pmap_for_collapse=None, adaptive_os_local=0,
        require_best_pose_details=False,
    )
    outs = score_outputs.PerHalfOutputs()
    prior_pose = object()
    outs.best_pose_rotations[1] = prior_pose

    outs.update_from(1, result)

    assert result.best_pose_rotations is None
    assert result.best_pose_translations is None
    assert outs.best_pose_rotations[1] is prior_pose
    assert_matches(outs.class_assignments[1], raw.class_assignments)
    assert_matches(outs.class_posterior[1], raw.class_mstep_posterior_sums)


@pytest.mark.parametrize("half", [0, 1])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kclass_publication_preserves_selected_source_eulers(half, dtype):
    result = _kclass_result()
    originals = {name: value.copy() for name, value in vars(result).items() if isinstance(value, np.ndarray)}
    outputs, returned = _publish(result, half, dtype)
    assert outputs.best_pose_rotation_eulers[half].dtype == np.float64
    assert_matches(outputs.best_pose_rotation_eulers[half], result.best_pose_eulers_deg)
    assert_matches(outputs.best_pose_rotations[half], result.best_pose_rotations.astype(dtype))
    assert_matches(outputs.best_pose_translations[half], result.best_pose_translations.astype(dtype))
    assert outputs.best_pose_rotations[half].dtype == dtype
    assert outputs.best_pose_translations[half].dtype == dtype
    assert_matches(outputs.class_assignments[half], result.class_assignments)
    assert_matches(outputs.class_posterior[half], result.class_mstep_posterior_sums)
    assert_matches(outputs.class_full_posterior[half], result.class_posterior_sums)
    assert_matches(returned.ha, result.pose_assignments)
    assert all(
        a is b for a, b in zip(
            (returned.Ft_y, returned.Ft_ctf, returned.em_stats, returned.noise_stats),
            (result.Ft_y, result.Ft_ctf, result.stats, result.aggregate_noise_stats),
        )
    )
    assert outputs.best_pose_rotation_eulers[1 - half] is None
    for name, original in originals.items():
        assert_matches(getattr(result, name), original)


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kclass_matrix_only_publication_preserves_legacy_dtype(missing, dtype):
    result = _kclass_result()
    if missing:
        del result.best_pose_eulers_deg
    else:
        result.best_pose_eulers_deg = None
    outputs, _ = _publish(result, 0, dtype)
    expected = utils.R_to_relion(result.best_pose_rotations.astype(dtype), degrees=True).astype(dtype)
    assert outputs.best_pose_rotation_eulers[0].dtype == dtype
    assert_matches(outputs.best_pose_rotation_eulers[0], expected)


def _bucket():
    source = _source_eulers(12).reshape(3, 4, 3)
    counts = np.array([3, 1, 2], dtype=np.int32)
    return SimpleNamespace(
        image_indices=np.array([2, 0, 1]),
        actual_rotation_counts=counts,
        local_rotation_ids=np.arange(12, dtype=np.int32).reshape(3, 4),
        local_rotation_posterior_ids=None,
        local_rotations=np.asarray(utils.R_from_relion(source.reshape(-1, 3), degrees=True), dtype=np.float32).reshape(
            3, 4, 3, 3
        ),
        local_source_eulers=source,
        local_rotation_mask=np.arange(4)[None, :] < counts[:, None],
        local_rotation_log_prior=np.zeros((3, 4), dtype=np.float32),
        translation_log_prior=np.zeros((3, 3), dtype=np.float32),
    )


def _layout():
    return SimpleNamespace(
        translation_grid=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32), n_pixels=8, n_psi=2
    )
