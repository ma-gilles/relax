"""Native fresh-K1 BPref operand formation, ordered accumulation and staging.

Native accumulation stays in RELION units until the half has finished; callers
own the final conversion to RECOVAR FFT normalization and CTF sign.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import NamedTuple

import jax
import numpy as np

from relax.refinement.ports import ExpectationProbe
from relax.refinement.refinement_options import FIRSTITER_FUSED_BPREF_ENV
from relax.runtime import memory_budget
from relax.runtime.engine_tuning import EngineTuning

logger = logging.getLogger(__name__)

_AUTO_DEFERRED_FIRSTITER_BPREF_DEVICE_FRACTION = 0.500


def _relion_firstiter_fused_bpref_enabled(
    *,
    requested: bool | None,
    fresh_k1_guard: bool,
    winner_take_all: bool,
    preserve_bpref_particle_order: bool,
    relion_exact_bpref_operands: bool,
    use_relion_x_half_mstep: bool,
    score_only: bool,
    first_numbered_iteration: bool,
) -> bool:
    """Route only fresh single-class firstiter BPref through its native kernel, in the run's first numbered
    iteration (``first_numbered_iteration``). ``requested`` is ``RefinementEnvironment.firstiter_fused_bpref``:
    None takes the kernel wherever it is supported, False never, True refuses where it is not supported."""

    supported = bool(
        fresh_k1_guard
        and winner_take_all
        and preserve_bpref_particle_order
        and relion_exact_bpref_operands
        and use_relion_x_half_mstep
        and not score_only
        and first_numbered_iteration
    )
    if requested is None:
        return supported
    if requested and not supported:
        raise ValueError(
            f"{FIRSTITER_FUSED_BPREF_ENV}=1 requires the fresh K=1 "
            "iteration-1 winner-take-all x-half path"
        )
    return bool(requested and supported)


def _relion_firstiter_bpref_overlap_bytes(
    *,
    projector_shape,
    projector_dtype,
    recon_volume_size: int,
    recon_y_dtype,
    recon_ctf_dtype,
) -> tuple[int, int, int]:
    """Estimate direct-path persistent bytes and its texture-time peak.

    The projection FFI creates one CUDA texture-array copy of the supplied
    Projector::data slab.  The direct route therefore owns two projector
    slabs while both native x-half accumulators are live.
    """

    projector_bytes = int(np.prod(projector_shape)) * int(
        np.dtype(projector_dtype).itemsize
    )
    accumulator_bytes = int(recon_volume_size) * (
        int(np.dtype(recon_y_dtype).itemsize)
        + int(np.dtype(recon_ctf_dtype).itemsize)
    )
    direct_peak_bytes = 2 * projector_bytes + accumulator_bytes
    return projector_bytes, accumulator_bytes, direct_peak_bytes


def _relion_firstiter_deferred_bpref_enabled(
    *,
    relion_firstiter_fused_bpref: bool,
    use_relion_projector: bool,
    projector_device_owned: bool,
    projector_shape,
    projector_dtype,
    recon_volume_size: int,
    recon_y_dtype,
    recon_ctf_dtype,
    device_memory_bytes: int | None,
    allocator_free_memory_bytes: int | None = None,
    diagnostics_active: bool = False,
) -> bool:
    """Select exact host-staged BPref when the direct GPU overlap is unsafe (from the memory readings; nothing
    forces it)."""

    supported = bool(
        relion_firstiter_fused_bpref
        and use_relion_projector
        and projector_device_owned
        and not diagnostics_active
    )
    if not supported or device_memory_bytes is None or int(device_memory_bytes) <= 0:
        return False
    _, accumulator_bytes, direct_peak_bytes = _relion_firstiter_bpref_overlap_bytes(
        projector_shape=projector_shape,
        projector_dtype=projector_dtype,
        recon_volume_size=recon_volume_size,
        recon_y_dtype=recon_y_dtype,
        recon_ctf_dtype=recon_ctf_dtype,
    )
    exceeds_physical_peak_budget = direct_peak_bytes > int(
        float(device_memory_bytes)
        * _AUTO_DEFERRED_FIRSTITER_BPREF_DEVICE_FRACTION
    )
    exceeds_allocator_budget = bool(
        allocator_free_memory_bytes is not None
        and int(allocator_free_memory_bytes) > 0
        and accumulator_bytes > int(0.8 * float(allocator_free_memory_bytes))
    )
    return bool(exceeds_physical_peak_budget or exceeds_allocator_budget)


def _relion_firstiter_compact_batch_planning_safe(
    *,
    relion_firstiter_fused_bpref: bool,
    projector_host_owned: bool,
    diagnostics_active: bool,
    deferred_firstiter_bpref: bool,
    direct_peak_bytes: int,
    fixed_base_bytes: int,
    projector_dtype,
    score_complex_dtype,
) -> bool:
    """Return whether the c64/f32 K=1 lifetime admits phase-max planning."""

    compact_dtypes = bool(
        np.dtype(projector_dtype) == np.dtype(np.complex64)
        and np.dtype(score_complex_dtype) == np.dtype(np.complex64)
    )
    return bool(
        relion_firstiter_fused_bpref
        and projector_host_owned
        and not diagnostics_active
        and compact_dtypes
        and (
            deferred_firstiter_bpref
            or int(direct_peak_bytes) <= int(fixed_base_bytes)
        )
    )

@dataclass(frozen=True)
class NumberedBPrefHalf:
    """The numbered half a first-iteration BPref plan is for: ``iteration`` and ``half_index`` (zero-based), whether
    the run's BPref device-signature target is this half (``device_signature_active``), the run's expectation
    probe, which says whether a BPref diagnostic is armed for it (asked only where a plan needs the answer), and the
    run's fused-kernel request (``fused_request``, ``RefinementEnvironment.firstiter_fused_bpref``)."""

    iteration: int
    half_index: int
    device_signature_active: bool
    probe: ExpectationProbe
    fused_request: bool | None

    def diagnostics_armed(self) -> bool:
        return self.probe.bpref_diagnostics_armed(
            self.iteration, self.half_index, bpref_device_signature_active=self.device_signature_active,
        )


class _RelionFirstiterCompactBatchPlanningDecision(NamedTuple):
    enabled: bool
    deferred_firstiter_bpref: bool
    direct_peak_bytes: int

def relion_firstiter_compact_batch_planning_decision(
    *,
    winner_take_all: bool,
    preserve_bpref_particle_order: bool,
    use_relion_x_half_mstep: bool,
    projector_half,
    score_complex_dtype,
    recon_volume_size: int,
    fixed_base_bytes: int,
    half: NumberedBPrefHalf,
    engine_tuning: EngineTuning,
) -> _RelionFirstiterCompactBatchPlanningDecision:
    """Preflight compact planning through the authoritative BPref gates, for the numbered ``half``.
    The fused-kernel request is the half's (``half.fused_request``); an armed finite check
    (``engine_tuning.finite_check``) is a diagnostic that keeps the plan off the compact route. RELION's
    source-faithful powerClass normalisation, and with it the exact BPref operands, is on wherever RELION's
    particle order is preserved (``preserve_bpref_particle_order``)."""

    if projector_half is None:
        return _RelionFirstiterCompactBatchPlanningDecision(False, False, 0)
    projector_shape = tuple(projector_half.shape)
    projector_dtype = np.dtype(projector_half.dtype)
    projector_host_owned = not isinstance(projector_half, jax.Array)
    diagnostics_active = (
        engine_tuning.finite_check != "off"
        or half.diagnostics_armed()
    )
    fresh_k1_guard = bool(preserve_bpref_particle_order)
    relion_exact_bpref_operands = fresh_k1_guard
    relion_firstiter_fused_bpref = _relion_firstiter_fused_bpref_enabled(
        requested=half.fused_request,
        fresh_k1_guard=fresh_k1_guard,
        winner_take_all=winner_take_all,
        preserve_bpref_particle_order=preserve_bpref_particle_order,
        relion_exact_bpref_operands=relion_exact_bpref_operands,
        use_relion_x_half_mstep=use_relion_x_half_mstep,
        score_only=False,
        first_numbered_iteration=half.iteration == 0,
    )
    _, _, direct_peak_bytes = _relion_firstiter_bpref_overlap_bytes(
        projector_shape=projector_shape,
        projector_dtype=projector_dtype,
        recon_volume_size=recon_volume_size,
        recon_y_dtype=np.complex64,
        recon_ctf_dtype=np.float32,
    )
    deferred_firstiter_bpref = _relion_firstiter_deferred_bpref_enabled(
        relion_firstiter_fused_bpref=relion_firstiter_fused_bpref,
        use_relion_projector=True,
        projector_device_owned=projector_host_owned,
        projector_shape=projector_shape,
        projector_dtype=projector_dtype,
        recon_volume_size=recon_volume_size,
        recon_y_dtype=np.complex64,
        recon_ctf_dtype=np.float32,
        device_memory_bytes=memory_budget._device_memory_limit_bytes(engine_tuning.device_memory_bytes),
        allocator_free_memory_bytes=memory_budget._jax_allocator_free_memory_bytes(),
        diagnostics_active=diagnostics_active,
    )
    enabled = _relion_firstiter_compact_batch_planning_safe(
        relion_firstiter_fused_bpref=relion_firstiter_fused_bpref,
        projector_host_owned=projector_host_owned,
        diagnostics_active=diagnostics_active,
        deferred_firstiter_bpref=deferred_firstiter_bpref,
        direct_peak_bytes=direct_peak_bytes,
        fixed_base_bytes=fixed_base_bytes,
        projector_dtype=projector_dtype,
        score_complex_dtype=score_complex_dtype,
    )
    return _RelionFirstiterCompactBatchPlanningDecision(
        enabled,
        deferred_firstiter_bpref,
        direct_peak_bytes,
    )
