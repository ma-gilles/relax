from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from recovar import cuda_backproject

from relax.classification import k_class
from relax.cuda import kernels as em_cuda_kernels
from relax.diagnostics import bpref_diagnostics
from relax.diagnostics import iteration as debug_dumps
from relax.refinement import half_scoring, iteration_loop

pytestmark = pytest.mark.unit


def _capture_environment() -> dict[str, str]:
    return {
        "RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR": "/tmp/device",
        "RELAX_BPREF_CONTRIBUTION_DUMP_ITERATION": "5",
        "RELAX_BPREF_CONTRIBUTION_DUMP_HALF": "1",
    }


def test_device_signature_scope_activates_only_target_numbered_half():
    env = _capture_environment()

    for iteration in range(1, 5):
        for half in (1, 2):
            assert not debug_dumps._bpref_device_signature_active_for_numbered_half(
                iteration=iteration,
                half=half,
                environ=env,
            )
    assert debug_dumps._bpref_device_signature_active_for_numbered_half(
        iteration=5,
        half=1,
        environ=env,
    )
    assert not debug_dumps._bpref_device_signature_active_for_numbered_half(
        iteration=5,
        half=2,
        environ=env,
    )
    assert not debug_dumps._bpref_device_signature_active_for_numbered_half(
        iteration=5,
        half=1,
        final_all_data=True,
        environ=env,
    )


def test_device_signature_scope_rejects_missing_or_invalid_target():
    env = _capture_environment()
    for missing in (
        "RELAX_BPREF_CONTRIBUTION_DUMP_ITERATION",
        "RELAX_BPREF_CONTRIBUTION_DUMP_HALF",
    ):
        invalid = dict(env)
        invalid.pop(missing)
        with pytest.raises(RuntimeError, match="requires explicit positive"):
            debug_dumps._bpref_device_signature_active_for_numbered_half(
                iteration=1,
                half=1,
                environ=invalid,
            )

    for name, value in (
        ("RELAX_BPREF_CONTRIBUTION_DUMP_ITERATION", "0"),
        ("RELAX_BPREF_CONTRIBUTION_DUMP_HALF", "3"),
        ("RELAX_BPREF_CONTRIBUTION_DUMP_ITERATION", "not-an-int"),
    ):
        invalid = dict(env)
        invalid[name] = value
        with pytest.raises((ValueError, RuntimeError)):
            debug_dumps._bpref_device_signature_active_for_numbered_half(
                iteration=1,
                half=1,
                environ=invalid,
            )


def test_scoped_capture_ignores_all_process_flags_off_target(monkeypatch):
    monkeypatch.setenv("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "/tmp/device")
    for name in (
        "RELAX_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH",
        "RELAX_RELION_X_HALF_BP_FUSED_ATOMICS",
        "RELAX_RELION_X_HALF_SEQUENTIAL_TRANSLATION_REDUCTION",
        "RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY",
        "RELAX_BPREF_HIGH_PRECISION_OPERAND_BUNDLE",
    ):
        monkeypatch.setenv(name, "1")

    inactive = bpref_diagnostics._scoped_bpref_diagnostic_flags(active=False)
    assert inactive == {
        "device_signature_configured": True,
        "sequential_translation_reduction": False,
        "per_particle_launches": False,
        "fused_atomics": False,
        "high_precision_operand_bundle": False,
    }
    assert not cuda_backproject.relion_x_half_bp_block_topology_enabled()
    with em_cuda_kernels.bpref_device_signature_scope(False):
        assert not cuda_backproject.relion_x_half_bp_block_topology_enabled()

    active = bpref_diagnostics._scoped_bpref_diagnostic_flags(active=True)
    assert all(value for name, value in active.items() if name != "device_signature_configured")
    with em_cuda_kernels.bpref_device_signature_scope(True):
        assert cuda_backproject.relion_x_half_bp_block_topology_enabled()


def test_device_panel_flush_writes_separate_class_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("RELAX_BPREF_CONTRIBUTION_DUMP_RUN_ID", "class-aware")
    prefix = (1, 2, "class-aware")
    common = {
        "current_size": 4,
        "max_r": 2.0,
        "image_shape": (4, 4),
        "volume_shape": (5, 5, 5),
        "reconstruction_padding_factor": 2,
        "source_stack_sha256": "a" * 64,
        "rank": 0,
        "causal_arm": "winner-take-all-per-particle-fused-xhalf",
        "winner_take_all": True,
    }
    for class_index in (0, 3):
        key = (*prefix, class_index)
        bpref_diagnostics._bpref_device_panel_accumulators[key] = (
            np.zeros(75, dtype=np.complex64),
            np.zeros(75, dtype=np.float32),
        )
        bpref_diagnostics._bpref_device_panel_launch_counters[key] = class_index + 1
        bpref_diagnostics._bpref_device_panel_metadata[key] = {
            **common,
            "class_index": class_index,
        }

    bpref_diagnostics.flush_bpref_device_panel_accumulator(iteration=1, half=2)

    outputs = sorted(Path(tmp_path).glob("recovar_device_panel_native_*.npz"))
    assert [path.name for path in outputs] == [
        "recovar_device_panel_native_it001_h2_class001_rank000.npz",
        "recovar_device_panel_native_it001_h2_class004_rank000.npz",
    ]
    assert [int(np.load(path)["class_index"]) for path in outputs] == [0, 3]


def test_target_dense_half_keeps_block_topology_inactive_for_live_work(monkeypatch):
    monkeypatch.setenv("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "/tmp/device")
    monkeypatch.setenv("RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY", "1")

    def fake_dense(half, sampling, priors, batching, variant, execution, optics):
        del half, sampling, priors, batching, variant, optics
        assert execution.bpref_device_signature_active is True
        assert not cuda_backproject.relion_x_half_bp_block_topology_enabled()
        return "ordinary-live"

    monkeypatch.setattr(half_scoring, "_score_half_dense", fake_dense)
    assert half_scoring._score_half_dense_in_bpref_scope(
        None,
        None,
        None,
        None,
        None,
        SimpleNamespace(bpref_device_signature_active=True),
        None,
    ) == "ordinary-live"
    assert not cuda_backproject.relion_x_half_bp_block_topology_enabled()


def test_standalone_diagnostics_keep_legacy_flags_without_device_capture(monkeypatch):
    monkeypatch.delenv("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", raising=False)
    for name in (
        "RELAX_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH",
        "RELAX_RELION_X_HALF_BP_FUSED_ATOMICS",
        "RELAX_RELION_X_HALF_SEQUENTIAL_TRANSLATION_REDUCTION",
        "RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY",
        "RELAX_BPREF_HIGH_PRECISION_OPERAND_BUNDLE",
    ):
        monkeypatch.setenv(name, "1")

    flags = bpref_diagnostics._scoped_bpref_diagnostic_flags(active=False)
    assert not flags["device_signature_configured"]
    assert all(value for name, value in flags.items() if name != "device_signature_configured")
    assert cuda_backproject.relion_x_half_bp_block_topology_enabled()


def test_target_half2_cannot_leak_into_final_all_data_or_local_search(monkeypatch):
    env = _capture_environment()
    env["RELAX_BPREF_CONTRIBUTION_DUMP_HALF"] = "2"
    assert debug_dumps._bpref_device_signature_active_for_numbered_half(
        iteration=5,
        half=2,
        environ=env,
    )
    assert not debug_dumps._bpref_device_signature_active_for_numbered_half(
        iteration=5,
        half=2,
        final_all_data=True,
        environ=env,
    )

    monkeypatch.setenv("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "/tmp/device")
    monkeypatch.setenv("RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY", "1")

    def fake_local(half, sampling, priors, batching, execution, diagnostics, optics):
        del half, sampling, priors, batching, execution, diagnostics, optics
        assert not cuda_backproject.relion_x_half_bp_block_topology_enabled()
        return "ordinary-local"

    monkeypatch.setattr(half_scoring, "_score_half_local", fake_local)
    assert half_scoring._score_half_local_in_bpref_scope(
        None, None, None, None, None, SimpleNamespace(bpref_device_signature_active=False), None
    ) == "ordinary-local"
    with pytest.raises(RuntimeError, match="sparse adaptive pass 2"):
        half_scoring._score_half_local_in_bpref_scope(
            None, None, None, None, None, SimpleNamespace(bpref_device_signature_active=True), None
        )


def test_clear_dump_context_marks_contribution_and_native_dumps_inactive():
    bpref_diagnostics.set_bpref_contribution_dump_context(iteration=5, half=2)
    assert bpref_diagnostics._bpref_contribution_context == {
        "iteration": 5,
        "half": 2,
    }

    bpref_diagnostics.clear_bpref_contribution_dump_context()
    assert bpref_diagnostics._bpref_contribution_context == {
        "iteration": -1,
        "half": -1,
    }


def test_iteration_loop_clears_dump_context_before_every_final_exit_or_half():
    source = inspect.getsource(iteration_loop.refine_single_volume)
    final_decision = source.index("should_run_final_iteration =")
    final_loop = source.index("for k in range(2):", source.index("final_outs = PerHalfOutputs()"))

    assert source.rfind("clear_bpref_contribution_dump_context()", 0, final_decision) >= 0
    assert "clear_bpref_contribution_dump_context()" in source[
        final_loop : source.index("final_half_t0", final_loop)
    ]


def test_active_capture_accepts_fused_kclass_route(monkeypatch):
    monkeypatch.setenv("RELAX_SPARSE_KCLASS_FUSED", "1")
    k_class._validate_bpref_device_signature_sparse_route(
        active=True,
        n_classes=4,
    )

    k_class._validate_bpref_device_signature_sparse_route(
        active=False,
        n_classes=1,
    )


def test_later_capture_support_excludes_dense_full_support_fallback():
    source = inspect.getsource(k_class.run_dense_k_class_em_adaptive)
    support_start = source.index("later_soft_particle_fused_supported =")
    support_end = source.index("fused_atomic_diagnostic_supported =", support_start)
    support_block = source[support_start:support_end]
    assert "and not skip_significance_pruning" in support_block
