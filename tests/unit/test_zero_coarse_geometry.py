"""Routing of the dense and local scorers' operands, on the CPU stand-in engine.

The numbered dense scoring passes its owner records down unchanged; RELION's coarse device geometry is
generated (and reaches the engine) only on the routes that use it; the diagnostic and local-adaptive
overrides are resolved before half scoring. See docs/math/zero_coarse_geometry.md.
"""

import inspect
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from helpers.tiny_refinement import CallTrace, run_tiny_refinement

from relax.dense import scoring_policy
from relax.refinement import expectation, finalization, half_scoring, iteration_loop
from relax.refinement.refinement_options import AdaptiveOptions

pytestmark = pytest.mark.unit

_DENSE_RECORDS = [
    "HalfScoringData", "DenseSamplingSpec", "DensePriorSpec", "DenseBatchPolicy", "DenseVariantPolicy",
    "DenseExecutionPolicy", "OpticsSpec",
]


def _parameters(function):
    return [name for name, parameter in inspect.signature(function).parameters.items()
            if parameter.kind is parameter.POSITIONAL_OR_KEYWORD]


def test_numbered_dense_scoring_exposes_owners_without_a_call_only_plan(monkeypatch):
    for name in ("DenseHalfScoringPlan", "_run_dense_half_scoring", "DenseHalfScoringOutputs",
                 "_dense_half_scoring_outputs"):
        assert not hasattr(iteration_loop, name) and not hasattr(expectation, name)
    trace = CallTrace(monkeypatch)
    trace.wrap(iteration_loop, "score_numbered_half", "half")
    trace.wrap(expectation, "_score_half_dense_in_bpref_scope", "dense")
    run_tiny_refinement(monkeypatch, final_after_max_iter=False)
    assert trace.labels() == ["half", "dense"] * 4
    for call in trace.calls("dense"):
        assert call.inside == ("half",)
        assert [type(argument).__name__ for argument in call.args] == _DENSE_RECORDS and not call.kwargs


@pytest.mark.parametrize("n_classes", [1, 2])
def test_adaptive_dense_route_keeps_owner_inputs_visible(monkeypatch, n_classes):
    """The dispatcher hands its mode's adaptive scorer its own seven records and the point group."""
    scorer = {1: "_score_adaptive_k1_dense", 2: "_score_adaptive_kclass_dense"}[n_classes]
    records = ["half", "sampling", "priors", "batching", "variant", "execution", "optics"]
    assert _parameters(half_scoring._score_half_dense_one_shape)[:7] == records
    if n_classes == 1:
        assert _parameters(half_scoring._score_adaptive_k1_dense) == [*records, "base_em_kwargs"]
        assert [name for name, parameter in inspect.signature(half_scoring._score_adaptive_k1_dense).parameters.items()
                if parameter.kind is parameter.KEYWORD_ONLY] == ["symmetry"]
    else:
        assert _parameters(half_scoring._score_adaptive_kclass_dense) == [*records, "em_kwargs", "symmetry"]
    trace = CallTrace(monkeypatch)
    trace.wrap(half_scoring, "_score_half_dense_one_shape", "dispatch")
    trace.wrap(half_scoring, scorer, "scorer")
    run_tiny_refinement(monkeypatch, n_classes=n_classes, final_after_max_iter=False)
    assert trace.labels() == ["dispatch", "scorer"] * 4
    for dispatch, call in zip(trace.calls("dispatch"), trace.calls("scorer"), strict=True):
        assert all(mine is theirs for mine, theirs in zip(call.args[:7], dispatch.args[:7], strict=True))
        symmetry = call.kwargs["symmetry"] if n_classes == 1 else call.args[8]
        assert symmetry == "C1"


def _device_rotations_stand_in(monkeypatch):
    """RELION's device-built coarse rotations are CUDA-only (None on a CPU); stand in a distinct host copy."""

    def coarse_rotations(rotation_grid, *args, **kwargs):
        return np.array(rotation_grid.rotations, copy=True)

    monkeypatch.setattr(iteration_loop, "coarse_pass1_rotations", coarse_rotations)


# (oversampling, classes, first-iteration CC, first-iteration hard reconstruction, float64 scoring) ->
# whether iterations 1 and 2 generate RELION's coarse device rotations.
_GATE_CASES = {
    "os0_k1": ((0, 1, False, False, False), [True, True]),
    "os0_kclass": ((0, 2, False, False, False), [False, False]),
    "os0_k1_cc": ((0, 1, True, False, False), [False, True]),
    "os0_k1_hard": ((0, 1, False, True, False), [False, True]),
    "os0_k1_float64": ((0, 1, False, False, True), [False, False]),
    "os1_k1": ((1, 1, False, False, False), [True, True]),
    "os1_kclass_cc_float64": ((1, 2, True, False, True), [True, True]),
}


def _gate_run(monkeypatch, trace, oversampling, n_classes, cc, hard, float64):
    parity = {}
    if cc:
        parity.update(emulate_relion_firstiter_cc=True, relion_firstiter_ini_high_angstrom=8.0)
    if hard:
        parity.update(first_iteration_reconstruction_mode="hard")
    if float64:
        monkeypatch.setattr(
            scoring_policy, "DENSE_PRECISION", replace(scoring_policy.DENSE_PRECISION, use_float64_scoring=True),
        )
    trace.wrap(iteration_loop, "iteration_trial_grid", "iteration")
    trace.wrap(iteration_loop, "coarse_pass1_rotations", "coarse_rotations")
    run_tiny_refinement(
        monkeypatch, n_classes=n_classes, final_after_max_iter=False, parity=parity,
        adaptive=AdaptiveOptions(adaptive_oversampling=oversampling),
    )


@pytest.mark.parametrize("case", sorted(_GATE_CASES))
def test_device_matrix_generation_gate(monkeypatch, case):
    """The local route never generates them; it has no CPU run (the GPU tiers cover it)."""
    inputs, expected = _GATE_CASES[case]
    trace = CallTrace(monkeypatch)
    _gate_run(monkeypatch, trace, *inputs)
    labels = trace.labels()
    generated = [labels[index + 1:index + 2] == ["coarse_rotations"] for index, label in enumerate(labels)
                 if label == "iteration"]
    assert generated == expected


@pytest.mark.parametrize(
    "oversampling,xhalf,score_mode,expected",
    [(0, True, "gaussian", [True] * 4), (0, False, "gaussian", [False] * 4), (1, True, "gaussian", [False] * 4),
     (0, True, "normalized_cc", [False, False, True, True])],
)
def test_only_coarse_engine_operand_changes(monkeypatch, oversampling, xhalf, score_mode, expected):
    """K=1 scores pass 1 on the generated coarse device rotations only at oversampling 0 with the x-half
    M-step and Gaussian scoring; the fine operand is always the pass-2 grid's."""
    monkeypatch.setenv("RELAX_K1_RELION_X_HALF_MSTEP", "1" if xhalf else "0")
    _device_rotations_stand_in(monkeypatch)
    trace = CallTrace(monkeypatch)
    trace.wrap(half_scoring, "_score_adaptive_k1_dense", "scorer")
    trace.wrap(half_scoring, "prepare_adaptive_pass2_grids", "grids")
    trace.wrap(half_scoring, "engine_projection_inputs", "projection")
    run_tiny_refinement(
        monkeypatch, final_after_max_iter=False, parity=dict(first_iteration_score_mode=score_mode),
        adaptive=AdaptiveOptions(adaptive_oversampling=oversampling),
    )
    scorers, grids, projections = trace.calls("scorer"), trace.calls("grids"), trace.calls("projection")
    assert len(scorers) == len(grids) == len(projections) == 4
    native = []
    for scorer, grid, projection in zip(scorers, grids, projections, strict=True):
        rotations = projection.kwargs["rotations"]
        assert rotations["fine"] is grid.result.fine_rotations
        override = scorer.args[1].coarse_scoring_rotations
        native.append(override is not None and rotations["coarse"] is override)
        assert native[-1] or rotations["coarse"] is grid.result.coarse_rotations
    assert native == expected


def test_dense_float64_diagnostic_is_resolved_by_expectation_orchestration(monkeypatch):
    """The numbered and final execution policies carry the diagnostic switch their owner resolved."""
    assert not hasattr(half_scoring, "_diagnostic_float64_pass2_matches")
    trace = CallTrace(monkeypatch)
    for module in (expectation, finalization):
        monkeypatch.setattr(module, "_diagnostic_float64_pass2_matches", lambda *args, **kwargs: True)
        trace.wrap(module, "_diagnostic_float64_pass2_matches", "resolved")
        trace.wrap(module, "DenseExecutionPolicy", "policy")
    run_tiny_refinement(monkeypatch)
    assert trace.labels() == ["resolved", "policy"] * 5
    assert all(call.kwargs["diagnostic_float64_pass2"] is True for call in trace.calls("policy"))


def test_local_adaptive_overrides_are_resolved_before_half_scoring():
    """Half scoring reads the local-adaptive overrides from its diagnostics record only.

    The numbered phase resolves them into ``LocalDiagnosticPolicy`` (test_numbered_expectation_preparation);
    the final local pass has no CPU run.
    """
    for name in ("_local_adaptive_pass2_full_parent_enabled", "_local_adaptive_pass2_rotation_only_enabled",
                 "_local_adaptive_pass2_denominator_support_mode"):
        assert not hasattr(half_scoring, name)


class _Stop(Exception):
    pass


def _local_one_shape(monkeypatch, *, parent_probe):
    """Run the local adaptive scorer of one half up to its pass-2 support; returns the trace."""
    from helpers.refinement_specs import local_half_owners
    from helpers.sparse_pass2_mock import MockDataset

    from relax.sampling import relion_angular_sampling_deg

    dataset = MockDataset(n_images=2, seed=3)
    layout = SimpleNamespace(rotation_counts=np.asarray([2]))
    trace = CallTrace(monkeypatch)
    monkeypatch.setattr(half_scoring, "_build_local_adaptive_parent_layout", lambda *args: (layout, 0))
    monkeypatch.setattr(half_scoring, "_run_local_search_iteration", parent_probe)

    def stop(*args):
        raise _Stop

    monkeypatch.setattr(half_scoring, "_prepare_local_adaptive_pass2_support", stop)
    trace.wrap(half_scoring, "_score_half_local_one_shape", "scorer")
    trace.wrap(half_scoring, "_build_local_adaptive_parent_layout", "parent_layout")
    trace.wrap(half_scoring, "_prepare_local_adaptive_pass2_support", "support")
    owners = local_half_owners(
        k=0, experiment_dataset=dataset, means_k=np.zeros(dataset.volume_size, dtype=np.complex64),
        noise_variance_k=np.ones(dataset.image_size, dtype=np.float32),
        previous_best_rotation_eulers_k=np.zeros((dataset.n_units, 3), dtype=np.float32),
        local_search_rotations=np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 2, axis=0),
        local_search_order=1, sigma_rot=np.deg2rad(1.0), sigma_psi=np.deg2rad(1.0),
        current_translations=np.zeros((1, 2), dtype=np.float32), base_translations=np.zeros((1, 2), dtype=np.float32),
        trans_prior_center=np.zeros((dataset.n_units, 2), dtype=np.float32),
        trans_prior_center_for_engine=np.zeros((dataset.n_units, 2), dtype=np.float32),
        current_sigma_offset_angstrom=1.0, disc_type="linear_interp", cs_for_engine=None,
        local_pass1_current_size=4, image_corrections_k=None, scale_corrections_k=None,
        translation_search_base=None, disable_adjoint_y=False, disable_adjoint_ctf=False, max_significants=None,
        iteration=3, local_search_random_perturbation=0.0,
        local_search_angular_sampling_deg=relion_angular_sampling_deg(1), local_parent_oversampling_order=1,
        diagnostic_score_only=False, local_search_translation_prior_mode="coarse", replay_prior_translations=None,
        collect_local_search_profile=False,
        local_profile_history=[],
    )
    with pytest.raises(_Stop):
        half_scoring._score_half_local(*owners)
    return trace, layout


def test_local_adaptive_parent_layout_exposes_five_story_inputs(monkeypatch):
    assert _parameters(half_scoring._build_local_adaptive_parent_layout) == [
        "half", "sampling", "priors", "translation_prior_reference_translations", "layout_dtype",
    ]

    def parent_probe(*args):
        raise _Stop

    trace, _ = _local_one_shape(monkeypatch, parent_probe=parent_probe)
    (scorer,), (layout,) = trace.calls("scorer"), trace.calls("parent_layout")
    assert layout.inside == ("scorer",)
    assert all(mine is theirs for mine, theirs in zip(layout.args[:3], scorer.args[:3], strict=True))


def test_local_adaptive_support_helper_exposes_six_story_inputs(monkeypatch):
    assert _parameters(half_scoring._prepare_local_adaptive_pass2_support) == [
        "parent_layout", "significant_sample_indices", "sampling", "diagnostics", "parent_order",
        "fine_layout_dtype",
    ]
    samples = object()

    def parent_probe(*args):
        return SimpleNamespace(profile_summary={"reconstruction_sample_indices_by_image": samples})

    trace, layout = _local_one_shape(monkeypatch, parent_probe=parent_probe)
    (scorer,), (support,) = trace.calls("scorer"), trace.calls("support")
    assert support.args[0] is layout and support.args[1] is samples
    assert support.args[2] is scorer.args[1] and support.args[4] == 0
    assert support.args[3] is scorer.args[5]


@pytest.mark.parametrize("oversampling", [0, 1])
def test_loop_transports_geometry_separately_from_effective_rotations(monkeypatch, oversampling):
    """The dense sampling carries the generated coarse device rotations only at oversampling 0."""
    _device_rotations_stand_in(monkeypatch)
    trace = CallTrace(monkeypatch)
    trace.wrap(iteration_loop, "coarse_pass1_rotations", "coarse_rotations")
    trace.wrap(expectation, "DenseSamplingSpec", "sampling")
    run_tiny_refinement(
        monkeypatch, final_after_max_iter=False, adaptive=AdaptiveOptions(adaptive_oversampling=oversampling),
    )
    generated, samplings = trace.calls("coarse_rotations"), trace.calls("sampling")
    assert len(generated) == len(samplings) == 2
    for rotations, sampling in zip(generated, samplings, strict=True):
        assert sampling.kwargs["coarse_scoring_rotations"] is (rotations.result if oversampling == 0 else None)
