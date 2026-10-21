from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
from recovar import cuda_backproject

from relax.classification import k_class
from relax.cuda import backprojection as cuda_backprojection
from relax.diagnostics import bpref_diagnostics
from relax.refinement import dense_half, local_half
from relax.refinement.refinement_options import bpref_device_signature_target

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
        "RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY",
        "RELAX_BPREF_HIGH_PRECISION_OPERAND_BUNDLE",
    ):
        monkeypatch.setenv(name, "1")

    inactive = bpref_diagnostics._scoped_bpref_diagnostic_flags(active=False)
    assert inactive == {
        "device_signature_configured": True,
        "per_particle_launches": False,
        "fused_atomics": False,
        "high_precision_operand_bundle": False,
    }
    assert not cuda_backproject.relion_x_half_bp_block_topology_enabled()
    with cuda_backprojection.bpref_device_signature_scope(False):
        assert not cuda_backproject.relion_x_half_bp_block_topology_enabled()

    active = bpref_diagnostics._scoped_bpref_diagnostic_flags(active=True)
    assert all(value for name, value in active.items() if name != "device_signature_configured")
    with cuda_backprojection.bpref_device_signature_scope(True):
        assert cuda_backproject.relion_x_half_bp_block_topology_enabled()


def test_target_dense_half_keeps_block_topology_inactive_for_live_work(monkeypatch):
    monkeypatch.setenv("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "/tmp/device")
    monkeypatch.setenv("RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY", "1")

    def fake_dense(half, sampling, priors, batching, variant, execution, optics):
        del half, sampling, priors, batching, variant, optics
        assert execution.bpref_device_signature_active is True
        assert not cuda_backproject.relion_x_half_bp_block_topology_enabled()
        return "ordinary-live"

    monkeypatch.setattr(dense_half, "_score_half_dense", fake_dense)
    assert dense_half.score_half_dense_in_bpref_scope(
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
    assert local_half.score_half_local_in_bpref_scope(
        None, None, None, None, None, SimpleNamespace(bpref_device_signature_active=False), None
    ) == "ordinary-local"
    with pytest.raises(RuntimeError, match="sparse adaptive pass 2"):
        local_half.score_half_local_in_bpref_scope(
            None, None, None, None, None, SimpleNamespace(bpref_device_signature_active=True), None
        )


def test_the_bpref_observer_arms_only_the_numbered_half_it_is_asked_about(monkeypatch, tmp_path):
    """The membership dump names a one-based iteration and half; the run asks with its zero-based indices, and
    without the observer (a run's default) nothing is armed."""
    from relax.diagnostics import observers
    from relax.refinement.ports import NoProbe, RunObserver

    for name in bpref_diagnostics.BPREF_DIAGNOSTIC_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RELAX_BPREF_MEMBERSHIP_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("RELAX_BPREF_MEMBERSHIP_DUMP_ITERATION", "2")
    monkeypatch.setenv("RELAX_BPREF_MEMBERSHIP_DUMP_HALF", "1")
    observer = observers.BpRefDiagnosticsObserver()
    armed = {
        (iteration, half): observer.bpref_diagnostics_armed(iteration, half, bpref_device_signature_active=False)
        for iteration in (0, 1) for half in (0, 1)
    }
    assert armed == {(0, 0): False, (0, 1): False, (1, 0): True, (1, 1): False}
    assert not RunObserver().bpref_diagnostics_armed(1, 0, bpref_device_signature_active=False)
    assert not NoProbe().bpref_diagnostics_armed(1, 0, bpref_device_signature_active=False)


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
