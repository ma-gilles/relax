"""Execute production routing expressions without a full refinement fixture."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from relax.relion.optics_aberrations import projection_rotations

pytestmark = pytest.mark.unit
OWNERS = Path(__file__).resolve().parents[2] / "relax"


def tree(name):
    # d2e7e27ed moved half_scoring.py from relax/dense to relax/refinement.
    return ast.parse((OWNERS / "refinement" / name).read_text())


def evaluate(node, **scope):
    return eval(compile(ast.Expression(node), "<production-route>", "eval"), scope)


def test_numbered_dense_scoring_exposes_owners_without_a_call_only_plan():
    loop = tree("iteration_loop.py")
    assert all(
        not isinstance(node, (ast.ClassDef, ast.FunctionDef))
        or node.name
        not in {
            "DenseHalfScoringPlan",
            "_run_dense_half_scoring",
            "DenseHalfScoringOutputs",
            "_dense_half_scoring_outputs",
        }
        for node in ast.walk(loop)
    )
    direct_calls = [
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_score_half_dense_in_bpref_scope"
        and node.args
    ]
    assert len(direct_calls) == 1
    assert [argument.id for argument in direct_calls[0].args] == [
        "dense_half",
        "dense_sampling",
        "dense_priors",
        "dense_batching",
        "dense_variant",
        "dense_execution",
        "dense_optics",
    ]


def test_numbered_half_batch_policies_have_a_planning_lifecycle():
    loop = tree("iteration_loop.py")
    half_step = next(
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.FunctionDef) and node.name == "_run_half_estep"
    )
    constructors = {
        node.func.id: node
        for node in ast.walk(half_step)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"DenseBatchPolicy", "LocalBatchPolicy"}
    }
    assert set(constructors) == {"DenseBatchPolicy", "LocalBatchPolicy"}

    dense_call = next(
        node
        for node in ast.walk(half_step)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_score_half_dense_in_bpref_scope"
    )
    local_call = next(
        node
        for node in ast.walk(half_step)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_score_half_local_in_bpref_scope"
    )
    assert constructors["DenseBatchPolicy"].lineno < dense_call.lineno
    assert constructors["LocalBatchPolicy"].lineno < local_call.lineno
    assert dense_call.args[3].id == "dense_batching"
    assert next(
        keyword.value.id
        for keyword in local_call.keywords
        if keyword.arg == "batching"
    ) == "local_batching"


def test_numbered_local_scoring_uses_prepared_owners_not_call_site_constructors():
    loop = tree("iteration_loop.py")
    half_step = next(
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.FunctionDef) and node.name == "_run_half_estep"
    )
    local_call = next(
        node
        for node in ast.walk(half_step)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_score_half_local_in_bpref_scope"
    )
    assert {
        keyword.arg: keyword.value.id
        for keyword in local_call.keywords
    } == {
        "half": "local_half",
        "sampling": "local_sampling",
        "priors": "local_priors",
        "batching": "local_batching",
        "execution": "local_execution",
        "diagnostics": "local_diagnostics",
        "optics": "local_optics",
    }

    constructors = {
        node.func.id: node.lineno
        for node in ast.walk(half_step)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        in {
            "LocalHalfData",
            "LocalPriorSpec",
            "LocalBatchPolicy",
            "LocalExecutionPolicy",
            "LocalDiagnosticPolicy",
            "LocalOpticsSpec",
        }
    }
    assert set(constructors) == {
        "LocalHalfData",
        "LocalPriorSpec",
        "LocalBatchPolicy",
        "LocalExecutionPolicy",
        "LocalDiagnosticPolicy",
        "LocalOpticsSpec",
    }
    assert all(line < local_call.lineno for line in constructors.values())


def test_final_local_scoring_reuses_iteration_owners_across_halves():
    loop = tree("iteration_loop.py")
    local_calls = [
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_score_half_local_in_bpref_scope"
    ]
    assert len(local_calls) == 2
    final_call = max(local_calls, key=lambda node: node.lineno)
    assert {
        keyword.arg: keyword.value.id
        for keyword in final_call.keywords
    } == {
        "half": "final_local_half",
        "sampling": "final_local_sampling",
        "priors": "final_local_priors",
        "batching": "final_local_batching",
        "execution": "final_local_execution",
        "diagnostics": "final_local_diagnostics",
        "optics": "final_local_optics",
    }

    final_loop = next(
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "k"
        and node.lineno < final_call.lineno < node.end_lineno
    )
    iteration_owned = {
        "LocalSamplingSpec",
        "LocalBatchPolicy",
        "LocalDiagnosticPolicy",
    }
    assert all(
        not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in iteration_owned
        )
        for node in ast.walk(final_loop)
    )


def test_direct_k1_dense_route_is_an_explicit_four_input_variant():
    scorer = tree("half_scoring.py")
    helper = next(
        node
        for node in scorer.body
        if isinstance(node, ast.FunctionDef) and node.name == "_score_direct_k1_dense"
    )
    assert [argument.arg for argument in helper.args.args] == [
        "half",
        "sampling",
        "execution",
        "em_kwargs",
    ]
    dispatcher = next(
        node
        for node in scorer.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_score_half_dense_one_shape"
    )
    calls = [
        node
        for node in ast.walk(dispatcher)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_score_direct_k1_dense"
    ]
    assert len(calls) == 1
    assert [argument.id for argument in calls[0].args] == [
        "half",
        "sampling",
        "execution",
        "em_kwargs",
    ]


def test_direct_kclass_dense_route_is_an_explicit_five_input_variant():
    scorer = tree("half_scoring.py")
    helper = next(
        node
        for node in scorer.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_score_direct_kclass_dense"
    )
    assert [argument.arg for argument in helper.args.args] == [
        "half",
        "sampling",
        "priors",
        "execution",
        "em_kwargs",
    ]
    dispatcher = next(
        node
        for node in scorer.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_score_half_dense_one_shape"
    )
    calls = [
        node
        for node in ast.walk(dispatcher)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_score_direct_kclass_dense"
    ]
    assert len(calls) == 1
    assert [argument.id for argument in calls[0].args] == [
        "half",
        "sampling",
        "priors",
        "execution",
        "em_kwargs",
    ]


def test_adaptive_kclass_dense_route_keeps_owner_inputs_visible():
    scorer = tree("half_scoring.py")
    helper = next(
        node
        for node in scorer.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_score_adaptive_kclass_dense"
    )
    assert [argument.arg for argument in helper.args.args] == [
        "half",
        "sampling",
        "priors",
        "batching",
        "variant",
        "execution",
        "em_kwargs",
        "symmetry",
    ]
    dispatcher = next(
        node
        for node in scorer.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_score_half_dense_one_shape"
    )
    calls = [
        node
        for node in ast.walk(dispatcher)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_score_adaptive_kclass_dense"
    ]
    assert len(calls) == 1
    assert [argument.id for argument in calls[0].args] == [
        "half",
        "sampling",
        "priors",
        "batching",
        "variant",
        "execution",
        "em_kwargs",
        "symmetry",
    ]


def test_adaptive_k1_dense_route_keeps_owner_inputs_visible():
    scorer = tree("half_scoring.py")
    helper = next(
        node
        for node in scorer.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_score_adaptive_k1_dense"
    )
    expected_inputs = [
        "half",
        "sampling",
        "priors",
        "batching",
        "variant",
        "execution",
        "optics",
        "base_em_kwargs",
    ]
    assert [argument.arg for argument in helper.args.args] == expected_inputs
    assert [argument.arg for argument in helper.args.kwonlyargs] == ["symmetry"]
    dispatcher = next(
        node
        for node in scorer.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_score_half_dense_one_shape"
    )
    calls = [
        node
        for node in ast.walk(dispatcher)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_score_adaptive_k1_dense"
    ]
    assert len(calls) == 1
    assert [argument.id for argument in calls[0].args] == [
        *expected_inputs[:-1],
        "em_kwargs",
    ]
    assert len(calls[0].keywords) == 1
    assert calls[0].keywords[0].arg == "symmetry"
    assert calls[0].keywords[0].value.id == "symmetry"


@pytest.mark.parametrize(
    "os,local,k,mode,hard,double,expected",
    [
        (0, False, 1, "gaussian", False, False, True),
        (0, True, 1, "gaussian", False, False, False),
        (0, False, 4, "gaussian", False, False, False),
        (0, False, 1, "normalized_cc", False, False, False),
        (0, False, 1, "gaussian", True, False, False),
        (0, False, 1, "gaussian", False, True, False),
        (1, False, 1, "gaussian", False, False, True),
        (1, False, 4, "normalized_cc", True, True, True),
        (1, True, 4, "gaussian", False, False, False),
    ],
)
def test_device_matrix_generation_gate(os, local, k, mode, hard, double, expected):
    gates = [
        n
        for n in ast.walk(tree("iteration_loop.py"))
        if isinstance(n, ast.If)
        and any(
            isinstance(s, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "adaptive_pass1_rotations" for t in s.targets)
            and isinstance(s.value, ast.Call)
            and isinstance(s.value.func, ast.Name)
            and s.value.func.id == "_relion_adaptive_pass1_rotations"
            for s in n.body
        )
    ]
    assert len(gates) == 1
    got = evaluate(
        gates[0].test,
        use_local=local,
        state=SimpleNamespace(adaptive_oversampling=os),
        n_classes=k,
        firstiter_score_mode_this_iter=mode,
        firstiter_winner_take_all_this_iter=hard,
        _DENSE_EM_STATIC_KWARGS={"use_float64_scoring": double},
    )
    assert bool(got) is expected


@pytest.mark.parametrize(
    "os,sparse,xhalf,mode,double,override,expected",
    [
        (0, True, True, "gaussian", False, True, True),
        (0, True, True, "gaussian", False, False, False),
        (1, True, True, "gaussian", False, True, False),
        (0, False, True, "gaussian", False, True, False),
        (0, True, False, "gaussian", False, True, False),
        (0, True, True, "normalized_cc", False, True, False),
        (0, True, True, "gaussian", True, True, False),
    ],
)
def test_only_coarse_engine_operand_changes(os, sparse, xhalf, mode, double, override, expected):
    calls = [
        n.value
        for n in ast.walk(tree("half_scoring.py"))
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "k1_adaptive_result" for t in n.targets)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id == "run_dense_k_class_em_adaptive"
    ]
    assert len(calls) == 1
    coarse, fine, native = object(), object(), object()
    scope = dict(
        pass2_grids=SimpleNamespace(coarse_rotations=coarse, fine_rotations=fine),
        sampling=SimpleNamespace(coarse_scoring_rotations=native if override else None),
        adaptive_os=os,
        sparse_pass2=sparse,
        relion_x_half_mstep=xhalf,
        variant=SimpleNamespace(firstiter_score_mode_this_iter=mode),
        execution=SimpleNamespace(diagnostic_float64_pass2=double),
        # Images on the reference grid without magnification: applyScaleDifference and
        # applyAnisoMag are the identity.
        projection_rotations=projection_rotations,
        optics=SimpleNamespace(projection_scale=1.0),
        magnification=None,
    )
    assert evaluate(calls[0].args[4], **scope) is (native if expected else coarse)
    assert evaluate(calls[0].args[6], **scope) is fine


def test_dense_float64_diagnostic_is_resolved_by_the_iteration_controller():
    scorer_source = tree("half_scoring.py")
    assert not any(
        isinstance(node, ast.Name) and node.id == "_diagnostic_float64_pass2_matches"
        for node in ast.walk(scorer_source)
    )

    execution_policies = [
        node
        for node in ast.walk(tree("iteration_loop.py"))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "DenseExecutionPolicy"
    ]
    assert len(execution_policies) == 2
    for policy in execution_policies:
        diagnostic = next(
            keyword.value
            for keyword in policy.keywords
            if keyword.arg == "diagnostic_float64_pass2"
        )
        assert isinstance(diagnostic, ast.Call)
        assert isinstance(diagnostic.func, ast.Name)
        assert diagnostic.func.id == "_diagnostic_float64_pass2_matches"


def test_local_experimental_overrides_are_resolved_by_the_iteration_controller():
    scorer_source = tree("half_scoring.py")
    controller_helpers = {
        "_local_search_precision_flags",
        "_local_adaptive_pass2_full_parent_enabled",
        "_local_adaptive_pass2_rotation_only_enabled",
        "_local_adaptive_pass2_denominator_support_mode",
    }
    assert not any(
        isinstance(node, ast.Name) and node.id in controller_helpers
        for node in ast.walk(scorer_source)
    )

    diagnostic_policies = [
        node
        for node in ast.walk(tree("iteration_loop.py"))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "LocalDiagnosticPolicy"
    ]
    assert len(diagnostic_policies) == 2
    resolved_fields = {
        "parent_use_float64_scoring",
        "parent_use_float64_projections",
        "fine_use_float64_scoring",
        "fine_use_float64_projections",
        "adaptive_pass2_full_parent",
        "adaptive_pass2_rotation_only",
        "adaptive_pass2_denominator_mode",
    }
    for policy in diagnostic_policies:
        assert resolved_fields <= {keyword.arg for keyword in policy.keywords}


def test_local_adaptive_support_helper_exposes_six_story_inputs():
    scorer = tree("half_scoring.py")
    helper = next(
        node
        for node in scorer.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_prepare_local_adaptive_pass2_support"
    )
    expected_inputs = [
        "parent_layout",
        "significant_sample_indices",
        "sampling",
        "diagnostics",
        "parent_order",
        "fine_layout_dtype",
    ]
    assert [argument.arg for argument in helper.args.args] == expected_inputs

    local_scorer = next(
        node
        for node in scorer.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_score_half_local_one_shape"
    )
    calls = [
        node
        for node in ast.walk(local_scorer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_prepare_local_adaptive_pass2_support"
    ]
    assert len(calls) == 1
    assert [argument.id for argument in calls[0].args] == [
        "parent_layout",
        "significant_sample_indices",
        "sampling",
        "diagnostics",
        "parent_order",
        "fine_local_layout_dtype",
    ]


def test_local_adaptive_parent_layout_exposes_five_story_inputs():
    scorer = tree("half_scoring.py")
    helper = next(
        node
        for node in scorer.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_build_local_adaptive_parent_layout"
    )
    expected_inputs = [
        "half",
        "sampling",
        "priors",
        "translation_prior_reference_translations",
        "layout_dtype",
    ]
    assert [argument.arg for argument in helper.args.args] == expected_inputs

    local_scorer = next(
        node
        for node in scorer.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_score_half_local_one_shape"
    )
    calls = [
        node
        for node in ast.walk(local_scorer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_build_local_adaptive_parent_layout"
    ]
    assert len(calls) == 1
    assert [argument.id for argument in calls[0].args] == [
        "half",
        "sampling",
        "priors",
        "translation_prior_reference_translations",
        "parent_local_layout_dtype",
    ]


def test_loop_transports_geometry_separately_from_effective_rotations():
    calls = [
        n
        for n in ast.walk(tree("iteration_loop.py"))
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "DenseSamplingSpec"
        and any(keyword.arg == "coarse_scoring_rotations" for keyword in n.keywords)
    ]
    assert len(calls) == 1
    keywords = {k.arg: k.value for k in calls[0].keywords}
    marker = object()
    assert (
        evaluate(
            keywords["coarse_scoring_rotations"],
            adaptive_pass1_rotations=marker,
            state=SimpleNamespace(adaptive_oversampling=0),
        )
        is marker
    )
    assert (
        evaluate(
            keywords["coarse_scoring_rotations"],
            adaptive_pass1_rotations=marker,
            state=SimpleNamespace(adaptive_oversampling=1),
        )
        is None
    )
