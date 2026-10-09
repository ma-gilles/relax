"""Pass-specific staging and batch planning preserve memory and grid contracts."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from helpers.float_compare import assert_matches
from helpers.run_options import stand_in

from relax.helpers.dtype_policy import DensePrecisionPolicy
from relax.refinement import expectation_batches as batches

pytestmark = pytest.mark.unit


@pytest.fixture
def planner(monkeypatch):
    monkeypatch.delenv("RELAX_DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS", raising=False)
    return batches.BatchPlanner(
        requested=stand_in.execution(image_batch_size=200, rotation_block_size=1000),
        image_shape=(16, 16), volume_shape=(16, 16, 16), n_classes=1, precision=DensePrecisionPolicy(),
        log=SimpleNamespace(info=lambda *args: None),
    )


def prepare(planner, **overrides):
    options = dict(
        rotations=SimpleNamespace(shape=(576, 3, 3)),
        translations=SimpleNamespace(shape=(29, 2)), cs_for_engine=12, coarse_cs=8,
        model_current_size_for_engine=12, use_adaptive=False, use_local=False,
        relion_firstiter_cc_this_iter=False, firstiter_winner_take_all_this_iter=False,
        firstiter_cc_tree_rescore_max_margin=None,
        source_faithful_spectrum_norm=True, preserve_bpref_particle_order=True,
        bpref_device_signature_active=False, use_relion_x_half_mstep=False,
        multi_shape_halves=False, coarse_sizing=(30.0, 100.0),
    )
    options.update(overrides)
    return batches.prepare_half_batches(
        SimpleNamespace(image_shape=(16, 16)),
        SimpleNamespace(data=np.zeros((9, 9, 5), dtype=np.complex64), r_max=4),
        planner=planner, **options,
    )


def capture_estimates(monkeypatch):
    calls = []

    def estimate(**kwargs):
        calls.append(kwargs)
        size = kwargs["current_size"] or 16
        return SimpleNamespace(
            image_batch_size=1000 // size, rotation_block_size=2000 // size,
            log_adjustment=lambda **kw: None,
        )

    monkeypatch.setattr(batches, "estimate_relion_em_batch_sizes", estimate)
    return calls


def test_available_memory_is_read_each_time_a_compact_pass_is_planned(planner, monkeypatch):
    calls = capture_estimates(monkeypatch)
    free = iter([8_000_000_000, 5_000_000_000])
    monkeypatch.setattr(batches.sparse_pass2_budget, "_device_free_memory_bytes", lambda: next(free))
    assert calls == []
    for _ in range(2):
        planner(576, 29, compact_k1_relion_layout=True, current_size_for_batch=12)
    assert_matches([c["runtime_free_memory_gb"] for c in calls], [8.0, 5.0])
    assert all(c["volume_shape"] == (16, 16, 16) for c in calls)


@pytest.mark.parametrize(
    ("physical", "allocator", "pool", "expected_gb"),
    [
        # A real card: the pool holds what it preallocated; the physical reading is the smaller bound.
        (19 * 2**30, 55 * 2**30, 50 * 2**30, 19 * 2**30 / 1e9),
        # A pool limit (16 GB emulation on an 80 GB card): the allocator's headroom bounds the textures.
        (62 * 2**30, 10 * 2**30, 1 * 2**30, 10 * 2**30 / 1e9),
        # No allocator reading (CPU, or a backend without memory_stats): the physical reading alone.
        (8 * 2**30, None, None, 8 * 2**30 / 1e9),
    ],
)
def test_compact_texture_budget_stays_inside_the_allocator_pool(
    planner, monkeypatch, physical, allocator, pool, expected_gb
):
    """relax#44: the compact K1 budget is min(physical free, allocator available)."""
    calls = capture_estimates(monkeypatch)
    budget = batches.sparse_pass2_budget
    monkeypatch.setattr(budget, "_device_free_memory_bytes", lambda: physical)
    monkeypatch.setattr(budget, "_jax_allocator_free_memory_bytes", lambda: allocator)
    monkeypatch.setattr(budget, "_jax_allocator_pool_free_bytes", lambda: pool)
    planner(10, 3, compact_k1_relion_layout=True)
    assert_matches(calls[0]["runtime_free_memory_gb"], expected_gb)


def test_local_deferred_grids_are_not_inspected_by_batch_preparation(planner, monkeypatch):
    monkeypatch.setattr(batches, "host_relion_projector_texture_enabled", lambda *a, **k: False)
    policy = prepare(planner, use_local=True, rotations=None, translations=None)
    assert policy.safe_batch_sizes is planner
    assert policy.class_overrides is None
    assert policy.coarse_image_batch_size is None


@pytest.mark.parametrize("rescore_margin", [None, 0.0, 4e-6])
def test_firstiter_compact_coarse_staging_stays_distinct_from_fine(planner, monkeypatch, rescore_margin):
    from relax.scoring import pass1_plan

    calls = capture_estimates(monkeypatch)
    monkeypatch.setattr(batches.sparse_pass2_budget, "_device_free_memory_bytes", lambda: 8_000_000_000)
    monkeypatch.setattr(batches, "host_relion_projector_texture_enabled", lambda *a, **k: True)
    monkeypatch.setattr(pass1_plan, "global_pass1_relion_projector_texture_enabled", lambda: True)
    monkeypatch.setattr(batches.firstiter_bpref, "relion_firstiter_compact_batch_planning_decision",
                        lambda **k: SimpleNamespace(enabled=True, deferred_firstiter_bpref=True))
    policy = prepare(
        planner, use_adaptive=True, relion_firstiter_cc_this_iter=True,
        firstiter_cc_tree_rescore_max_margin=rescore_margin,
    )
    assert [c["current_size"] for c in calls] == [12, 8]
    assert calls[0]["score_projector_staging_bytes"] == 0
    expected_coarse_staging = 9 * 9 * 5 * np.dtype(np.complex64).itemsize if rescore_margin is None else 0
    assert calls[1]["score_projector_staging_bytes"] == expected_coarse_staging
    assert calls[0]["model_current_size"] == 12 and calls[0]["compact_k1_relion_layout"]
    assert calls[1]["model_current_size"] == (12 if rescore_margin is None else None)
    assert calls[1]["compact_k1_relion_layout"] == (rescore_margin is None)
    if rescore_margin is not None:
        assert policy.significance_safe_batch_sizes is planner
    assert policy.fine_image_batch_size == 83
    assert policy.coarse_image_batch_size == 125


def test_local_half_plans_no_compact_staging(planner, monkeypatch):
    """The local pass sizes its own tiles: a non-adaptive local half keeps the plain planner."""
    calls = capture_estimates(monkeypatch)
    monkeypatch.setattr(batches, "host_relion_projector_texture_enabled", lambda *a, **k: True)
    policy = prepare(planner, use_local=True, rotations=None, translations=None)
    assert calls == []
    assert policy.safe_batch_sizes is planner


@pytest.mark.parametrize("precision", ["scoring", "projection", "pass2"])
def test_diagnostic_precision_keeps_compact_planning_disabled(planner, monkeypatch, precision):
    calls = capture_estimates(monkeypatch)
    if precision == "pass2":
        monkeypatch.setenv("RELAX_DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS", "2")
    else:
        planner = replace(planner, precision=DensePrecisionPolicy(
            use_float64_scoring=precision == "scoring", use_float64_projections=precision == "projection",
        ))
    monkeypatch.setattr(batches, "host_relion_projector_texture_enabled", lambda *a, **k: pytest.fail("compact planning"))
    policy = prepare(planner, use_adaptive=True, relion_firstiter_cc_this_iter=True)
    assert policy.safe_batch_sizes is planner
    assert not any(c["compact_k1_relion_layout"] for c in calls)
    assert all(c["use_float64_scoring"] == (precision != "projection") for c in calls)
