"""BPref diagnostic capture, scoped execution checks and artifact writing.

This module owns the switches a numbered half's BPref diagnostics read
(``BpRefDiagnosticsObserver``).
Diagnostic output preserves the existing schemas, array precision and capture order. Execution-policy flags here select explicit
diagnostic overrides; callers still own the production scoring and M-step.
"""

from __future__ import annotations

import os

from relax.runtime.env_flags import parse_env_flag

_BPREF_MEMBERSHIP_DUMP_DIR_ENV = "RELAX_BPREF_MEMBERSHIP_DUMP_DIR"
_BPREF_MEMBERSHIP_DUMP_ITERATION_ENV = "RELAX_BPREF_MEMBERSHIP_DUMP_ITERATION"
_BPREF_MEMBERSHIP_DUMP_HALF_ENV = "RELAX_BPREF_MEMBERSHIP_DUMP_HALF"

_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH_ENV = "RELAX_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH"


_RELION_X_HALF_BP_FUSED_ATOMICS_ENV = "RELAX_RELION_X_HALF_BP_FUSED_ATOMICS"


# The variables the BPref observer (``relax.diagnostics.observers.BpRefDiagnosticsObserver``) reads; the command
# builds it when one of them is set. Unset or blank, its hooks do nothing and arm nothing.
BPREF_DIAGNOSTIC_ENVS = (
    "RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR",
    "RELAX_BPREF_CONTRIBUTION_DUMP_ITERATION",
    "RELAX_BPREF_CONTRIBUTION_DUMP_HALF",
    "RELAX_BPREF_CONTRIBUTION_DUMP_DIR",
    _BPREF_MEMBERSHIP_DUMP_DIR_ENV,
    _RELION_X_HALF_BP_PER_PARTICLE_LAUNCH_ENV,
    _RELION_X_HALF_BP_FUSED_ATOMICS_ENV,
    "RELAX_BPREF_HIGH_PRECISION_OPERAND_BUNDLE",
    "RELAX_BPREF_ACCUMULATOR_DELTA_DUMP_DIR",
    "RELAX_PASS2_DUMP_DIR",
    "RELAX_SPARSE_PASS2_NATIVE_DUMP_DIR",
)


def relion_x_half_bp_per_particle_launch_enabled() -> bool:
    """Return whether the diagnostic x-half path launches once per particle."""

    return parse_env_flag(_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH_ENV, default=False)


def relion_x_half_bp_fused_atomics_enabled() -> bool:
    """Return whether the diagnostic fused data/weight scatter is enabled."""

    return parse_env_flag(_RELION_X_HALF_BP_FUSED_ATOMICS_ENV, default=False)


def _scoped_bpref_diagnostic_flags(*, active: bool) -> dict[str, bool]:
    """Resolve process flags against an explicit device-capture boundary."""

    device_signature_configured = bool(os.environ.get("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "").strip())
    scope_active = bool(active or not device_signature_configured)
    return {
        "device_signature_configured": device_signature_configured,
        "per_particle_launches": bool(scope_active and relion_x_half_bp_per_particle_launch_enabled()),
        "fused_atomics": bool(scope_active and relion_x_half_bp_fused_atomics_enabled()),
        "high_precision_operand_bundle": bool(
            scope_active
            and parse_env_flag(
                "RELAX_BPREF_HIGH_PRECISION_OPERAND_BUNDLE",
                default=False,
            )
        ),
    }


def _bpref_membership_dump_requested(context_iteration: int, context_half: int):
    """Whether the membership dump targets the numbered half ``context_half`` (1 or 2) of iteration
    ``context_iteration`` (one-based; -1 for both outside a numbered half)."""
    dump_dir = os.environ.get(_BPREF_MEMBERSHIP_DUMP_DIR_ENV, "").strip()
    if not dump_dir:
        return False
    target_iteration = os.environ.get(_BPREF_MEMBERSHIP_DUMP_ITERATION_ENV)
    if target_iteration and context_iteration != int(target_iteration):
        return False
    target_half = os.environ.get(_BPREF_MEMBERSHIP_DUMP_HALF_ENV)
    if target_half:
        if int(target_half) not in {1, 2}:
            raise ValueError(f"{_BPREF_MEMBERSHIP_DUMP_HALF_ENV} must be 1 or 2")
        if context_half != int(target_half):
            return False
    return True


_BPREF_ACCUMULATOR_DELTA_DUMP_DIR_ENV = "RELAX_BPREF_ACCUMULATOR_DELTA_DUMP_DIR"


def _scoped_bpref_diagnostics_active(flags: dict[str, bool]) -> bool:
    """Return whether a scoped execution diagnostic, not just its target, is active."""

    return any(
        bool(flags[name])
        for name in (
            "per_particle_launches",
            "fused_atomics",
            "high_precision_operand_bundle",
        )
    )


def relion_firstiter_bpref_diagnostics_active(
    *,
    bpref_device_signature_active: bool,
    iteration: int,
    half: int,
) -> bool:
    """Resolve the diagnostics that make deferred firstiter BPref unsafe, for the numbered ``half`` (1 or 2) of
    ``iteration`` (one-based)."""

    device_signature_configured = bool(
        os.environ.get("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "").strip()
    )
    return bool(
        (device_signature_configured and bpref_device_signature_active)
        or (
            os.environ.get("RELAX_BPREF_CONTRIBUTION_DUMP_DIR", "").strip()
            and (bpref_device_signature_active or not device_signature_configured)
        )
        or _bpref_membership_dump_requested(iteration, half)
        or _scoped_bpref_diagnostics_active(
            _scoped_bpref_diagnostic_flags(active=bpref_device_signature_active)
        )
        or os.environ.get(_BPREF_ACCUMULATOR_DELTA_DUMP_DIR_ENV, "").strip()
        or os.environ.get("RELAX_PASS2_DUMP_DIR", "").strip()
        or os.environ.get("RELAX_SPARSE_PASS2_NATIVE_DUMP_DIR", "").strip()
    )
