"""A persistent particle half is shared by numbered and final scoring."""

from dataclasses import replace

import numpy as np
import pytest

from relax.refinement.half_inputs import HalfScoringData, initialize_halfsets

pytestmark = pytest.mark.unit


def test_particle_ownership_survives_model_changes_without_repacking():
    datasets, optics = [object(), object()], [object(), object()]
    eulers = [np.zeros((2, 3), dtype=np.float64), np.zeros((3, 3), dtype=np.float64)]
    translations = [np.zeros((2, 2), dtype=np.float32), np.zeros((3, 2), dtype=np.float32)]
    corrections = [np.ones(2, dtype=np.float32), np.ones(3, dtype=np.float32)]
    halves = initialize_halfsets(
        datasets, optics_group_ids=optics,
        previous_best_rotation_eulers=eulers, previous_best_translations=translations,
        image_corrections=corrections, scale_corrections=None,
        group_ids=None, group_count=None,
    )
    for index, half in enumerate(halves):
        assert half.index == index
        assert half.dataset is datasets[index]
        assert half.optics_group_ids is optics[index]
        assert half.rotation_eulers is eulers[index]
        assert half.translations is translations[index]
        assert half.image_corrections is corrections[index]
        numbered = HalfScoringData(particles=half, reference=object(), noise_variance=object())
        next_eulers, next_translations = object(), object()
        half.rotation_eulers = next_eulers
        half.translations = next_translations
        final = replace(numbered, reference=object())
        assert final.particles is numbered.particles is half
        assert final.particles.rotation_eulers is next_eulers
        assert final.particles.translations is next_translations
        assert final.reference is not numbered.reference
        assert final.noise_variance is numbered.noise_variance
    assert halves[0].rotation_eulers is not halves[1].rotation_eulers


def test_shape_view_does_not_replace_the_resident_particle_state():
    halves = initialize_halfsets(
        [object(), object()], previous_best_rotation_eulers=None,
        previous_best_translations=None, image_corrections=None, scale_corrections=None,
        optics_group_ids=(None, None), group_ids=None, group_count=None,
    )
    resident = halves[0]
    subset = replace(resident, dataset=object(), rotation_eulers=object())
    assert subset is not resident
    assert subset.dataset is not resident.dataset
    assert resident.rotation_eulers is None
