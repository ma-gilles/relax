"""Source Euler metadata survives publication without changing compute operands."""

from types import SimpleNamespace

import numpy as np
import pytest
from helpers.float_compare import assert_matches
from recovar import utils

from relax.dense import score_outputs

pytestmark = pytest.mark.unit


def _source_eulers(n):
    return np.column_stack(
        (
            np.linspace(-143.123456789, 132.987654321, n),
            np.linspace(23.123456789, 151.987654321, n),
            np.linspace(-87.123456789, 69.987654321, n),
        )
    )


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
        stats=object(),
        Ft_y=np.arange(12).reshape(4, 3),
        Ft_ctf=np.arange(12, 24).reshape(4, 3),
    )


def _scatter(result, half, dtype):
    outputs = score_outputs.PerHalfOutputs()
    returned = score_outputs._scatter_dense_k_class_result(
        result,
        k=half,
        effective_rotations=np.zeros((3, 3, 3)),
        rot_pmap_for_collapse=None,
        adaptive_os_local=0,
        outputs=outputs,
        pose_dtype=dtype,
    )
    return outputs, returned


@pytest.mark.parametrize("half", [0, 1])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kclass_publication_preserves_selected_source_eulers(half, dtype):
    result = _kclass_result()
    originals = {name: value.copy() for name, value in vars(result).items() if isinstance(value, np.ndarray)}
    outputs, returned = _scatter(result, half, dtype)
    assert outputs.best_pose_rotation_eulers[half].dtype == np.float64
    assert_matches(outputs.best_pose_rotation_eulers[half], result.best_pose_eulers_deg)
    assert_matches(outputs.best_pose_rotations[half], result.best_pose_rotations.astype(dtype))
    assert_matches(outputs.best_pose_translations[half], result.best_pose_translations.astype(dtype))
    assert outputs.best_pose_rotations[half].dtype == dtype
    assert outputs.best_pose_translations[half].dtype == dtype
    assert_matches(outputs.class_assignments[half], result.class_assignments)
    assert_matches(outputs.class_posterior[half], result.class_mstep_posterior_sums)
    assert_matches(outputs.class_full_posterior[half], result.class_posterior_sums)
    assert_matches(returned[0], result.pose_assignments)
    assert all(
        a is b for a, b in zip(returned[1:], (result.Ft_y, result.Ft_ctf, result.stats, result.aggregate_noise_stats))
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
    outputs, _ = _scatter(result, 0, dtype)
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


