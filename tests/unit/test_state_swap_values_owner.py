"""The state-swap probe returns one named value instead of positional 14-tuples."""

from relax.helpers.orientation_priors import DirectionPrior, HalfDirectionPriors
from types import SimpleNamespace
from relax.refinement.half_inputs import initialize_halfsets
from relax.refinement.mean_helpers import ReferenceModel
from relax.refinement.noise_updates import NoiseModel

import numpy as np
import pytest

from relax.diagnostics import state_swap_runtime


def _inputs():
    half_inputs = initialize_halfsets(
        (None, None),
        image_corrections=[np.array([1.0]), np.array([2.0])],
        scale_corrections=[np.array([3.0]), np.array([4.0])],
        previous_best_translations=[np.zeros((1, 2)), np.zeros((1, 2))],
        previous_best_rotation_eulers=[np.zeros((1, 3)), np.zeros((1, 3))],
    )
    direction_priors = [
        HalfDirectionPriors(
            classes=DirectionPrior(values, order),
            shared=DirectionPrior(shared_values, shared_order),
        )
        for values, order, shared_values, shared_order in zip(
            [np.array([0.5]), np.array([0.5])], [4, 4],
            [np.array([0.5]), np.array([0.5])], [4, 4], strict=True,
        )
    ]
    tau2 = np.array([50.0])
    return dict(
        state=SimpleNamespace(a=1), cs=52, volume_shape=(1, 1, 1),
        reference_model=ReferenceModel(
            maps=[np.array([10.0]), np.array([20.0])], tau2=tau2, tau2_per_half=[tau2, tau2],
        ),
        noise_model=NoiseModel(
            variance_per_half=[np.array([60.0]), np.array([70.0])],
            radial_per_half=[np.array([80.0]), np.array([90.0])],
            average_variance=np.array([65.0]), average_radial=np.array([85.0]),
        ),
        relion_half_inputs=half_inputs, previous_best_rotations=[np.eye(3)[None], np.eye(3)[None]], current_sigma_offset_angstrom=3.1,
        current_sigma_offset_angstrom_per_half=[2.9, 3.3], direction_priors=direction_priors,
    )


def test_state_swap_values_keep_the_controller_order():
    assert state_swap_runtime._StateSwapValues._fields == (
        "cs", "reference_model", "noise_model", "previous_best_rotations", "current_sigma_offset_angstrom",
        "current_sigma_offset_angstrom_per_half", "direction_priors",
    )


def test_unchanged_paths_return_the_input_objects():
    kw = _inputs()
    for probe, snapshot in ((None, {"x": 1}), ({"iteration": 6}, None), ({"iteration": 7, "variant": "recovar_sigma_offset"}, {"x": 1})):
        out = state_swap_runtime._apply_state_swap_probe(probe=probe, iteration=6, recovar_snapshot=snapshot, **kw)
        assert isinstance(out, state_swap_runtime._StateSwapValues)
        assert out.cs is kw["cs"] and out.reference_model is kw["reference_model"]
        assert out.current_sigma_offset_angstrom_per_half is kw["current_sigma_offset_angstrom_per_half"]
        assert out.noise_model is kw["noise_model"]
        assert out.current_sigma_offset_angstrom is kw["current_sigma_offset_angstrom"] and len(out) == 7


def test_missing_prior_keeps_its_saved_order_through_snapshot_and_restore():
    inputs = _inputs()
    inputs["direction_priors"][0].shared = DirectionPrior(None, 3)
    snapshot_inputs = dict(inputs)
    snapshot_inputs.pop("volume_shape")
    snapshot = state_swap_runtime._snapshot_state_swap_inputs(**snapshot_inputs)
    assert snapshot["global_direction_prior_per_half"][0] is None
    assert snapshot["global_direction_prior_order_per_half"][0] == 3

    restored = state_swap_runtime._apply_state_swap_probe(
        probe={"iteration": 6, "variant": "recovar_direction_prior"},
        iteration=6, recovar_snapshot=snapshot, **inputs,
    )
    assert restored.direction_priors[0].shared.values is None
    assert restored.direction_priors[0].shared.healpix_order == 3


@pytest.mark.parametrize("variant", ["recovar_noise_variance_only", "recovar_previous_noise_radial_only"])
def test_noise_probe_can_restore_one_representation_without_recomputing_the_other(variant):
    inputs = _inputs()
    snapshot_inputs = dict(inputs)
    snapshot_inputs.pop("volume_shape")
    snapshot = state_swap_runtime._snapshot_state_swap_inputs(**snapshot_inputs)
    original = inputs["noise_model"]
    snapshot["noise_variance"][:] = 111.0
    snapshot["previous_noise_radial"][:] = 222.0
    restored = state_swap_runtime._apply_state_swap_probe(
        probe={"iteration": 6, "variant": variant}, iteration=6, recovar_snapshot=snapshot, **inputs,
    ).noise_model
    if variant == "recovar_noise_variance_only":
        assert restored.radial_per_half is original.radial_per_half
        assert restored.average_radial is original.average_radial
        assert float(restored.average_variance[0]) == pytest.approx(111.0)
    else:
        assert restored.variance_per_half is original.variance_per_half
        assert restored.average_variance is original.average_variance
        assert float(restored.average_radial[0]) == pytest.approx(222.0)
        restored.average_radial[0] = 333.0
        assert float(snapshot["previous_noise_radial"][0]) == pytest.approx(222.0)
