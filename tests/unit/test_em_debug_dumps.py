"""Per-iteration particle-state diagnostics (final-Q 68ac9d05ab)."""

from types import SimpleNamespace

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.diagnostics.iteration import _save_iteration_particle_states, _source_image_indices

pytestmark = pytest.mark.unit


def _poses(rotations, eulers, relative, absolute):
    return [
        SimpleNamespace(rotations=rotations[k], eulers_deg=eulers[k], relative_translations_pixels=relative[k],
                        translations_pixels=absolute[k])
        for k in range(2)
    ]


def _per_half(value1, value2):
    return [np.asarray(value1), np.asarray(value2)]


def test_save_iteration_particle_states_preserves_source_aligned_values(tmp_path):
    rotations = _per_half(np.eye(3, dtype=np.float32)[None], np.repeat(np.eye(3)[None], 2, axis=0))
    eulers = _per_half([[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    relative = _per_half([[0.25, -0.5]], [[1.0, 2.0], [3.0, 4.0]])
    absolute = _per_half([[10.25, 19.5]], [[31.0, 42.0], [53.0, 64.0]])
    pmax = _per_half([0.75], [0.5, 0.25])
    counts = _per_half([3], [4, 5])
    hard = _per_half([11], [12, 13])
    coarse = _per_half([21], [22, 23])
    original = _per_half([101], [205, 207])

    _save_iteration_particle_states(
        str(tmp_path),
        _poses(rotations, eulers, relative, absolute),
        SimpleNamespace(max_posterior=pmax, hard_assignments=hard, coarse_ha=coarse),
        SimpleNamespace(per_half=counts),
        iteration=2,
        original_image_indices_per_half=original,
    )

    half2 = np.load(tmp_path / "it002_particle_state_half2.npz")
    assert_matches(half2["half_local_indices"], [0, 1])
    assert_matches(half2["original_image_indices"], [205, 207])
    np.testing.assert_allclose(half2["rotation_eulers_deg"], eulers[1])
    np.testing.assert_allclose(half2["relative_translations_pixels"], relative[1])
    np.testing.assert_allclose(half2["absolute_translations_pixels"], absolute[1])
    np.testing.assert_allclose(half2["max_posterior"], pmax[1])
    assert_matches(half2["significant_counts"], counts[1])
    assert_matches(half2["one_based_iteration"], [3])
    assert_matches(half2["half"], [2])


def test_save_iteration_particle_states_rejects_misaligned_fields(tmp_path):
    one = _per_half([1], [2])
    misaligned = _per_half([1, 2], [3])

    with pytest.raises(ValueError, match="not source-aligned for half 1"):
        _save_iteration_particle_states(
            str(tmp_path),
            _poses(one, one, one, one),
            SimpleNamespace(max_posterior=misaligned, hard_assignments=one, coarse_ha=one),
            SimpleNamespace(per_half=one),
            iteration=0,
            original_image_indices_per_half=one,
        )


def test_source_image_indices_map_local_rows_to_the_original_stack():
    class _Layout:
        def original_image_indices_for_local(self, local):
            return np.asarray([7, 3, 9], dtype=np.int32)[np.asarray(local)]

    mapped = _source_image_indices(SimpleNamespace(n_images=3, _index_layout=_Layout()))
    assert mapped.dtype == np.int64
    assert_matches(mapped, [7, 3, 9])
    assert_matches(_source_image_indices(SimpleNamespace(n_images=2)), [0, 1])
