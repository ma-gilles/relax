"""VDAM's exact-local pass scores RELION's radial window at the full box, as the local search does.

RELION builds ``Mresol_fine`` in updateImageSizeAndResolutionPointers for every mode
(ml_optimiser.cpp:5784-5793 at f2c1a38) and precalculateShiftedImagesCtfsAndInvSigma2s masks the
score with it without a ``do_grad`` branch (:6841-6880), so VDAM at ``current_size == box`` still
drops the corners beyond ``box / 2``.
"""

import ast
import inspect

import pytest

pytestmark = pytest.mark.unit


def test_vdam_exact_local_call_passes_window_at_box():
    from relax.vdam import sparse_pass2_estep

    tree = ast.parse(inspect.getsource(sparse_pass2_estep))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "run_local_k_class_em"
    ]
    assert len(calls) == 1
    keywords = {kw.arg: kw.value for kw in calls[0].keywords}
    assert isinstance(keywords.get("window_at_box"), ast.Constant) and keywords["window_at_box"].value is True
