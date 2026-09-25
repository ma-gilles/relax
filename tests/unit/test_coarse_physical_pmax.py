"""Coarse Pmax publication preserves active rows across physical batches."""

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.scoring import significance

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("active", [0, 1, 8, 24, 192, 200])
@pytest.mark.parametrize("tail", [0.0, 7.0, np.nan, np.inf, -np.inf])
def test_physical_pmax_preserves_active_bytes(active, tail):
    weights = np.random.default_rng(829).uniform(size=(200, 129)).astype(np.float32)
    weights /= weights.sum(axis=1, keepdims=True)
    weights[0] = 0
    weights[1] = -np.arange(129, dtype=np.float32)
    weights[active:] = tail
    device_weights = jnp.asarray(weights)
    # The trimmed-table reduction this publication replaced.
    control = np.asarray(jnp.max(device_weights[:active], axis=1), dtype=np.float32)
    candidate = significance._coarse_max_posterior_for_host(device_weights, active)
    assert candidate.shape == control.shape == (active,)
    assert candidate.dtype == control.dtype == np.dtype(np.float32)
    assert_matches(candidate, control, strict=True)
    assert_matches(candidate, weights[:active].max(axis=1))


@pytest.mark.parametrize("active", [0, 1, 5, 8])
def test_any_over_leading_rows_matches_the_trimmed_reduction(active):
    mask = np.random.default_rng(31).uniform(size=(8, 6, 3)) < 0.2
    expected = np.any(mask[:active], axis=0)
    actual = np.asarray(significance._any_over_leading_rows(jnp.asarray(mask), active))
    assert actual.dtype == np.bool_ and actual.shape == expected.shape
    assert_matches(actual, expected, strict=True)


@pytest.mark.gpu
@pytest.mark.parametrize("width", [37888, 50176, 29696, 21504])
def test_physical_pmax_gpu_shapes(width, monkeypatch):
    """Real fringe widths: every active count uses one reduction program and no slice."""
    import inspect
    import json
    import os
    from pathlib import Path

    import jax
    from jax._src import compiler

    assert jax.default_backend() == "gpu"
    active_sizes = (1, 8, 24, 40, 96, 192, 200)
    weights = np.random.default_rng(593).uniform(size=(200, width)).astype(np.float32)
    weights /= weights.sum(axis=1, keepdims=True)
    weights[0] = 0
    device_weights = jax.device_put(weights)
    device_weights.block_until_ready()
    expected = weights.max(axis=1)
    original = compiler.compile_or_get_cached
    signature = inspect.signature(original)
    acquisitions = []

    def observe(*args, **kwargs):
        computation = signature.bind(*args, **kwargs).arguments["computation"]
        acquisitions.append({
            "module": compiler.ir.StringAttr(computation.operation.attributes["sym_name"]).value,
            "types": [str(op.attributes["function_type"]) for op in computation.body.operations
                      if "function_type" in op.attributes],
        })
        return original(*args, **kwargs)

    monkeypatch.setattr(compiler, "compile_or_get_cached", observe)
    jax.clear_caches()
    for active in active_sizes:
        actual = significance._coarse_max_posterior_for_host(device_weights, active)
        assert_matches(actual, expected[:active], strict=True)
    reductions = [x for x in acquisitions if x["module"] == "jit__reduce_max"]
    slices = [x for x in acquisitions if x["module"] == "jit_dynamic_slice"]
    assert len(reductions) == 1
    assert not slices
    panels = [{"acquisitions": list(acquisitions)}]
    if "COARSE_GPU_ROOT" in os.environ:
        output = Path(os.environ["COARSE_GPU_ROOT"]) / f"pmax_width_{width}.json"
        with output.open("x") as stream:
            json.dump({"width": width, "physical_rows": 200, "logical_rows": active_sizes,
                       "input_bytes": weights.nbytes, "panels": panels}, stream, indent=2)
