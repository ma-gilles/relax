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
from relax.refinement import finalization, half_scoring, local_half
from relax.refinement.refinement_options import ExecutionOptions, bpref_device_signature_target

pytestmark = pytest.mark.unit


def _capture_environment() -> dict[str, str]:
    return {
        "RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR": "/tmp/device",
        "RELAX_BPREF_CONTRIBUTION_DUMP_ITERATION": "5",
        "RELAX_BPREF_CONTRIBUTION_DUMP_HALF": "1",
    }


def test_device_signature_target_is_the_numbered_half_the_environment_names():
    env = _capture_environment()
    assert bpref_device_signature_target(environ=env) == (5, 1)
    env["RELAX_BPREF_CONTRIBUTION_DUMP_HALF"] = "2"
    assert bpref_device_signature_target(environ=env) == (5, 2)
    # Without the device-signature directory nothing is captured, whatever the targets say.
    env.pop("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR")
    assert bpref_device_signature_target(environ=env) is None


def test_the_run_options_read_the_device_signature_target_once(monkeypatch):
    for name, value in _capture_environment().items():
        monkeypatch.setenv(name, value)
    assert ExecutionOptions.from_environ().bpref_device_signature_target == (5, 1)
    monkeypatch.delenv("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR")
    assert ExecutionOptions.from_environ().bpref_device_signature_target is None


def test_device_signature_scope_rejects_missing_or_invalid_target():
    env = _capture_environment()
    for missing in (
        "RELAX_BPREF_CONTRIBUTION_DUMP_ITERATION",
        "RELAX_BPREF_CONTRIBUTION_DUMP_HALF",
    ):
        invalid = dict(env)
        invalid.pop(missing)
        with pytest.raises(RuntimeError, match="requires explicit positive"):
            bpref_device_signature_target(environ=invalid)

    for name, value in (
        ("RELAX_BPREF_CONTRIBUTION_DUMP_ITERATION", "0"),
        ("RELAX_BPREF_CONTRIBUTION_DUMP_HALF", "3"),
        ("RELAX_BPREF_CONTRIBUTION_DUMP_ITERATION", "not-an-int"),
    ):
        invalid = dict(env)
        invalid[name] = value
        with pytest.raises((ValueError, RuntimeError)):
            bpref_device_signature_target(environ=invalid)


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


def test_target_half2_cannot_leak_into_local_search(monkeypatch):
    monkeypatch.setenv("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "/tmp/device")
    monkeypatch.setenv("RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY", "1")

    def fake_local(half, sampling, priors, batching, execution, diagnostics, optics):
        del half, sampling, priors, batching, execution, diagnostics, optics
        assert not cuda_backproject.relion_x_half_bp_block_topology_enabled()
        return "ordinary-local"

    monkeypatch.setattr(local_half, "_score_half_local", fake_local)
    assert local_half._score_half_local_in_bpref_scope(
        None, None, None, None, None, SimpleNamespace(bpref_device_signature_active=False), None
    ) == "ordinary-local"
    with pytest.raises(RuntimeError, match="sparse adaptive pass 2"):
        local_half._score_half_local_in_bpref_scope(
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


@pytest.mark.parametrize("final_pass", [False, True])
def test_iteration_loop_clears_dump_context_before_every_final_exit_or_half(monkeypatch, final_pass):
    """Numbered dump identities never reach the final decision, the return path or a final half."""
    from helpers.tiny_refinement import CallTrace, run_tiny_refinement

    cleared = {"iteration": -1, "half": -1}

    def context_cleared(call):
        assert bpref_diagnostics._bpref_contribution_context == cleared

    def half_leaves_a_context(call):
        bpref_diagnostics._bpref_contribution_context.update(iteration=99, half=call.args[0].index + 1)

    trace = CallTrace(monkeypatch)
    trace.wrap(bpref_diagnostics, "set_bpref_contribution_dump_context", "numbered_half")
    trace.wrap(finalization, "_should_run_final_all_data_iteration", "decision", before=context_cleared)
    trace.wrap(finalization, "prepare_final_half", "final_half", before=context_cleared, after=half_leaves_a_context)
    run_tiny_refinement(monkeypatch, final_after_max_iter=final_pass)

    if not final_pass:
        assert bpref_diagnostics._bpref_contribution_context == cleared
    # Two numbered iterations of two halves set the context; the final halves start from a cleared one.
    assert trace.labels("numbered_half", "decision")[:5] == 4 * ["numbered_half"] + ["decision"]
    assert len(trace.calls("final_half")) == (2 if final_pass else 0)


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


def test_later_capture_support_is_the_sparse_soft_posterior_pass():
    source = inspect.getsource(k_class.run_dense_k_class_em_adaptive)
    support_start = source.index("later_soft_particle_fused_supported =")
    support_end = source.index("fused_atomic_diagnostic_supported =", support_start)
    support_block = source[support_start:support_end]
    assert "and not firstiter_cc_pass2_only_best_coarse" in support_block
    assert "skip_significance_pruning" not in source
