"""Every VDAM E-step route scores RELION's radial window at the full box.

RELION builds ``Mresol_fine`` in updateImageSizeAndResolutionPointers for every mode
(ml_optimiser.cpp:5784-5793 at f2c1a38) and precalculateShiftedImagesCtfsAndInvSigma2s masks the
score with it without a ``do_grad`` branch (:6841-6880), so VDAM at ``current_size == box`` still
drops the corners beyond ``box / 2``. The engines' behaviour is tested in test_fourier_window.py;
these tests pin that each VDAM call site asks for it.
"""

import ast
import inspect

import pytest

pytestmark = pytest.mark.unit


def _calls(module, name):
    tree = ast.parse(inspect.getsource(module))
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, "id", None) == name]


def _passes_true(call, keyword="window_at_box"):
    value = {kw.arg: kw.value for kw in call.keywords}.get(keyword)
    return isinstance(value, ast.Constant) and value.value is True


@pytest.mark.parametrize("callee", ["run_local_k_class_em", "_compute_k_class_significance_batched"])
def test_default_vdam_route_passes_window_at_box(callee):
    from relax.vdam import sparse_pass2_estep

    calls = _calls(sparse_pass2_estep, callee)
    assert calls and all(_passes_true(call) for call in calls)


def test_adaptive_vdam_route_passes_window_at_box():
    from relax.vdam import adaptive_estep

    route = [call for call in _calls(adaptive_estep, "dict") if any(kw.arg == "sparse_pass2" for kw in call.keywords)]
    assert len(route) == 1 and _passes_true(route[0])


def test_dense_vdam_engine_kwargs_keep_window_at_box():
    from relax.vdam import dense_adapter

    source = inspect.getsource(dense_adapter._dense_engine_kwargs)
    assert '"window_at_box": True' in source
