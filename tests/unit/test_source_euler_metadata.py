"""Source Euler publication preserves RFLOAT metadata without changing score inputs."""

import json
import logging

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax import sampling
from relax.classification import k_class_results
from relax.helpers.types import make_relion_stats
from relax.local.local_layout import (
    bucket_local_hypothesis_layout,
    build_pass2_hypothesis_layout,
)
from relax.scoring.sparse_bucket_arrays import _prepare_per_image_pass2_inputs

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_native_source_triplet_and_legacy_arrays(dtype, tmp_path):
    binding = pytest.importorskip("relax.relion_bind._relion_bind_core")
    # Captured native trial16 last won child4 of parent(direction37, psi3).
    kwargs = dict(
        oversampling_order=1,
        random_perturbation=0.18350759148597717,
        return_rotation_indices=True,
        return_mstep_rotations=True,
        dtype=dtype,
    )
    old = sampling.get_oversampled_rotation_grid_from_samples([37 + 3 * 48], 1, **kwargs)
    new = sampling.get_oversampled_rotation_grid_from_samples([37 + 3 * 48], 1, return_source_eulers=True, **kwargs)
    assert len(old) == 4 and len(new) == 5
    for a, b in zip(old, new[:4], strict=True):
        assert a.dtype == b.dtype
        assert_matches(b, a, strict=True)
    expected = np.array([159.3271497477632, 126.91279408422895, 85.75518260708287])
    assert new[-1].dtype == np.float64
    assert_matches(new[-1][4], expected)
    native = binding.get_oversampled_orientations(1, 1, 37, 3, kwargs["random_perturbation"])
    assert_matches(new[-1], native)
    metrics = dict(matrices_match=True, source_maxabs=float(np.max(abs(new[-1][4] - expected))))
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    logging.info("%s: %s", dtype, metrics)


@pytest.mark.parametrize("order", [False, True])
def test_sparse_override_source_follows_true_child_permutation(order):
    # IDs repeat by coarse parent; source rows must follow the chosen child, not a global nearest-grid ID.
    parent = np.array([1, 0, 1, 0])
    eulers = np.arange(12, dtype=np.float64).reshape(4, 3) + 2**-35
    matrices = np.broadcast_to(np.eye(3, dtype=np.float32), (4, 3, 3)).copy()
    kwargs = dict(
        n_coarse_rot=72,
        n_coarse_trans=1,
        nside_level=0,
        oversampling_order=1,
        n_fine_trans=1,
        fine_translation_parent=np.zeros(1, np.int32),
        rotation_log_prior=None,
        random_perturbation=0.0,
        fine_rotations_override=matrices,
        fine_rotation_parent_override=parent,
        relion_parent_execution_order=order,
    )
    inputs = _prepare_per_image_pass2_inputs(
        [np.array([0, 1]), np.array([1])], fine_source_eulers_override=eulers, **kwargs
    )
    legacy = _prepare_per_image_pass2_inputs([np.array([0, 1]), np.array([1])], **kwargs)
    for i in range(2):
        ids = inputs["oversampled_rot_indices"][i]
        assert_matches(inputs["source_eulers"][i], eulers[ids])
        for key, expected in legacy.items():
            if key == "source_eulers":
                continue
            actual = inputs[key]
            if isinstance(expected, (list, tuple)):
                assert_matches(actual[i], expected[i])
            elif expected is None:
                # Optional resident/M-step tables are absent in both layouts here.
                assert actual is None, key
            else:
                # Shared per-call tables and capacities must not depend on source metadata.
                assert_matches(actual, expected)
        assert legacy["source_eulers"][i] is None


def test_local_union_and_bucket_source_alignment():
    pytest.importorskip("relax.relion_bind._relion_bind_core")
    layout = build_pass2_hypothesis_layout(
        [np.array([7, 3]), np.array([3])],
        72,
        1,
        0,
        np.zeros((1, 2), np.float32),
        translation_step=1.0,
        oversampling_order=1,
        random_perturbation=0.13,
        rotation_index_order="relion_hidden",
    )
    for i in range(2):
        parents = np.unique([7, 3] if i == 0 else [3])
        expected = sampling.get_oversampled_rotation_grid_from_samples(
            parents, 0, random_perturbation=0.13, return_source_eulers=True, rotation_index_order="relion_hidden"
        )[-1]
        start, stop = layout.rotation_offsets[i : i + 2]
        assert_matches(layout.source_eulers_flat[start:stop], expected)
    for bucket in bucket_local_hypothesis_layout(layout, 2, 32):
        for row, idx in enumerate(bucket.image_indices):
            start, stop = layout.rotation_offsets[idx : idx + 2]
            assert_matches(
                bucket.local_source_eulers[row, : stop - start], layout.source_eulers_flat[start:stop]
            )


def _stats(n):
    return make_relion_stats(
        log_evidence_per_image=np.zeros(n),
        best_log_score_per_image=np.zeros(n),
        max_posterior_per_image=np.ones(n, np.float32),
        rotation_posterior_sums=np.ones(1),
    )


def test_inactive_class_without_metadata_does_not_erase_winner():
    eulers = np.array([[1.0 + 2**-40, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = k_class_results._assemble_result(
        class_log_evidence=np.array([[0.0, -20.0], [-20.0, 0.0], [-np.inf, -np.inf], [-np.inf, -np.inf]]),
        new_means=None,
        Ft_y=[np.zeros(1, np.complex64)] * 4,
        Ft_ctf=[np.ones(1, np.float32)] * 4,
        per_class_hard_assignments=np.zeros((4, 2), np.int32),
        per_class_stats=tuple(
            _stats(2)._replace(best_log_score_per_image=score)
            for score in np.array([[0.0, -20.0], [-20.0, 0.0], [-np.inf, -np.inf], [-np.inf, -np.inf]])
        ),
        noise_stats=None,
        per_class_best_pose_eulers_deg=[eulers, eulers + 10, None, None],
    )
    assert_matches(result.best_pose_eulers_deg, np.stack([eulers[0], eulers[1] + 10]))


