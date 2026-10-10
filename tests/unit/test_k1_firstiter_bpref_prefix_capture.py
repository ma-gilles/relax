from __future__ import annotations

import numpy as np
import pytest

from relax.refinement import firstiter_bpref as sparse
from relax.diagnostics import bpref_diagnostics


def test_deferred_firstiter_bpref_gate_uses_texture_peak_and_force_override(
    monkeypatch,
):
    box800_projector_bytes, box800_accumulator_bytes, box800_peak_bytes = (
        sparse._relion_firstiter_bpref_overlap_bytes(
            projector_shape=(1603, 1603, 802),
            projector_dtype=np.complex64,
            recon_volume_size=2_060_826_418,
            recon_y_dtype=np.complex64,
            recon_ctf_dtype=np.float32,
        )
    )
    assert box800_projector_bytes == 16_486_611_344
    assert box800_accumulator_bytes == 24_729_917_016
    assert box800_peak_bytes == 57_703_139_704

    common = dict(
        relion_firstiter_fused_bpref=True,
        use_relion_projector=True,
        projector_device_owned=True,
        projector_shape=(10, 10, 6),
        projector_dtype=np.complex64,
        recon_volume_size=100,
        recon_y_dtype=np.complex64,
        recon_ctf_dtype=np.float32,
    )
    projector_bytes, accumulator_bytes, peak_bytes = (
        sparse._relion_firstiter_bpref_overlap_bytes(
            projector_shape=common["projector_shape"],
            projector_dtype=common["projector_dtype"],
            recon_volume_size=common["recon_volume_size"],
            recon_y_dtype=common["recon_y_dtype"],
            recon_ctf_dtype=common["recon_ctf_dtype"],
        )
    )
    assert projector_bytes == 4_800
    assert accumulator_bytes == 1_200
    assert peak_bytes == 10_800

    monkeypatch.delenv(sparse._RELION_FIRSTITER_DEFERRED_BPREF_ENV, raising=False)
    assert sparse._relion_firstiter_deferred_bpref_enabled(
        **common,
        device_memory_bytes=20_000,
    )
    assert not sparse._relion_firstiter_deferred_bpref_enabled(
        **common,
        device_memory_bytes=22_000,
    )
    assert sparse._relion_firstiter_deferred_bpref_enabled(
        **common,
        device_memory_bytes=22_000,
        allocator_free_memory_bytes=1_000,
    )

    monkeypatch.setenv(sparse._RELION_FIRSTITER_DEFERRED_BPREF_ENV, "1")
    assert sparse._relion_firstiter_deferred_bpref_enabled(
        **common,
        device_memory_bytes=1_000_000,
    )
    with pytest.raises(ValueError, match="host-owned RELION projector"):
        sparse._relion_firstiter_deferred_bpref_enabled(
            **(common | {"projector_device_owned": False}),
            device_memory_bytes=20_000,
        )
    with pytest.raises(ValueError, match="host-owned RELION projector"):
        sparse._relion_firstiter_deferred_bpref_enabled(
            **common,
            device_memory_bytes=20_000,
            diagnostics_active=True,
        )

    monkeypatch.setenv(sparse._RELION_FIRSTITER_DEFERRED_BPREF_ENV, "0")
    assert not sparse._relion_firstiter_deferred_bpref_enabled(
        **common,
        device_memory_bytes=20_000,
    )


def test_scoped_bpref_configured_target_is_not_an_active_diagnostic():
    flags = {
        "device_signature_configured": True,
        "sequential_translation_reduction": False,
        "per_particle_launches": False,
        "fused_atomics": False,
        "high_precision_operand_bundle": False,
    }
    assert not bpref_diagnostics._scoped_bpref_diagnostics_active(flags)
    flags["fused_atomics"] = True
    assert bpref_diagnostics._scoped_bpref_diagnostics_active(flags)


def test_firstiter_fused_bpref_defaults_only_inside_complete_fresh_k1_guard(monkeypatch):
    monkeypatch.delenv("RELAX_K1_RELION_FIRSTITER_FUSED_BPREF", raising=False)
    monkeypatch.setitem(bpref_diagnostics._bpref_contribution_context, "iteration", 1)
    monkeypatch.setitem(bpref_diagnostics._bpref_contribution_context, "half", 2)
    kwargs = dict(
        fresh_k1_guard=True,
        winner_take_all=True,
        preserve_bpref_particle_order=True,
        relion_exact_bpref_operands=True,
        use_relion_x_half_mstep=True,
        score_only=False,
    )

    assert sparse._relion_firstiter_fused_bpref_enabled(**kwargs)
    assert not sparse._relion_firstiter_fused_bpref_enabled(
        **{**kwargs, "winner_take_all": False}
    )


def test_firstiter_fused_bpref_override_can_disable_but_not_expand_scope(monkeypatch):
    monkeypatch.setitem(bpref_diagnostics._bpref_contribution_context, "iteration", 1)
    monkeypatch.setitem(bpref_diagnostics._bpref_contribution_context, "half", 1)
    kwargs = dict(
        fresh_k1_guard=True,
        winner_take_all=True,
        preserve_bpref_particle_order=True,
        relion_exact_bpref_operands=True,
        use_relion_x_half_mstep=True,
        score_only=False,
    )
    monkeypatch.setenv("RELAX_K1_RELION_FIRSTITER_FUSED_BPREF", "0")
    assert not sparse._relion_firstiter_fused_bpref_enabled(**kwargs)

    monkeypatch.setenv("RELAX_K1_RELION_FIRSTITER_FUSED_BPREF", "1")
    with pytest.raises(ValueError, match="requires the fresh K=1"):
        sparse._relion_firstiter_fused_bpref_enabled(
            **{**kwargs, "fresh_k1_guard": False}
        )


@pytest.mark.parametrize('name', [
    'RELAX_SPARSE_PASS2_NATIVE_DUMP_DIR',
    'RELAX_BPREF_ACCUMULATOR_DELTA_DUMP_DIR',
    'RELAX_PASS2_DUMP_DIR',
])
def test_deferred_firstiter_preserves_requested_capture_route(monkeypatch, name):
    monkeypatch.setenv(name, '/requested/capture')
    active = bpref_diagnostics._relion_firstiter_bpref_diagnostics_active(
        bpref_device_signature_active=False,
    )
    assert active
    common = dict(
        relion_firstiter_fused_bpref=True, use_relion_projector=True,
        projector_device_owned=True, projector_shape=(1603, 1603, 802),
        projector_dtype=np.complex64, recon_volume_size=2_060_826_418,
        recon_y_dtype=np.complex64, recon_ctf_dtype=np.float32,
        device_memory_bytes=80 * 1024**3, diagnostics_active=active,
    )
    monkeypatch.delenv(sparse._RELION_FIRSTITER_DEFERRED_BPREF_ENV, raising=False)
    assert not sparse._relion_firstiter_deferred_bpref_enabled(**common)
    monkeypatch.setenv(sparse._RELION_FIRSTITER_DEFERRED_BPREF_ENV, '1')
    with pytest.raises(ValueError):
        sparse._relion_firstiter_deferred_bpref_enabled(**common)
