"""BPref diagnostic capture, scoped execution checks and artifact writing.

This module owns the shared numbered-half context and capture counters used by
sparse and exact-local EM. Diagnostic output preserves the existing schemas,
array precision and capture order. Execution-policy flags here select explicit
diagnostic overrides; callers still own the production scoring and M-step.
"""

from __future__ import annotations

import os
from pathlib import Path

import jax
import numpy as np

from relax.helpers.env_flags import parse_env_flag
from relax.local.local_backprojection import relion_x_half_sequential_translation_reduction_enabled

_BPREF_MEMBERSHIP_DUMP_DIR_ENV = "RELAX_BPREF_MEMBERSHIP_DUMP_DIR"
_BPREF_MEMBERSHIP_DUMP_ITERATION_ENV = "RELAX_BPREF_MEMBERSHIP_DUMP_ITERATION"
_BPREF_MEMBERSHIP_DUMP_HALF_ENV = "RELAX_BPREF_MEMBERSHIP_DUMP_HALF"

_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH_ENV = "RELAX_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH"


_RELION_X_HALF_BP_FUSED_ATOMICS_ENV = "RELAX_RELION_X_HALF_BP_FUSED_ATOMICS"


_bpref_contribution_context = {"iteration": -1, "half": -1}


_bpref_image_identity_cache: dict[str, np.ndarray] = {}


_BPrefPanelKey = tuple[int, int, str, int]


_bpref_device_panel_accumulators: dict[_BPrefPanelKey, tuple[jax.Array, jax.Array]] = {}


_bpref_device_panel_launch_counters: dict[_BPrefPanelKey, int] = {}


_bpref_device_panel_metadata: dict[_BPrefPanelKey, dict[str, object]] = {}


def set_bpref_contribution_dump_context(*, iteration: int, half: int) -> None:
    """Set explicit one-based iteration/half labels for diagnostic row dumps."""

    _bpref_contribution_context["iteration"] = int(iteration)
    _bpref_contribution_context["half"] = int(half)


def clear_bpref_contribution_dump_context() -> None:
    """Mark contribution and native M-step dumps as outside a numbered half."""

    _bpref_contribution_context["iteration"] = -1
    _bpref_contribution_context["half"] = -1


class BPrefContributionDumpComplete(RuntimeError):
    """Raised after an explicitly targeted BPref diagnostic bundle is written."""

    def __init__(
        self,
        *,
        contribution_path: str | Path,
        device_signature_path: str | Path | None,
    ):
        self.contribution_path = Path(contribution_path)
        self.device_signature_path = None if device_signature_path is None else Path(device_signature_path)
        message = f"requested RECOVAR BPref contribution target was written (contribution_path={self.contribution_path}"
        if self.device_signature_path is not None:
            message += f", device_signature_path={self.device_signature_path}"
        super().__init__(message + ")")


def flush_selected_bpref_device_panel(*, iteration_index: int, half_index: int) -> None:
    """Flush the requested numbered capture, given zero-based loop indices."""

    _device_signature_target_iteration = os.environ.get(
        "RELAX_BPREF_CONTRIBUTION_DUMP_ITERATION"
    )
    _device_signature_target_half = os.environ.get(
        "RELAX_BPREF_CONTRIBUTION_DUMP_HALF"
    )
    if _device_signature_target_half and int(_device_signature_target_half) not in {1, 2}:
        raise ValueError("RELAX_BPREF_CONTRIBUTION_DUMP_HALF must be 1 or 2")
    if (
        os.environ.get("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR")
        and (
            not _device_signature_target_iteration
            or int(_device_signature_target_iteration) == iteration_index + 1
        )
        and (
            not _device_signature_target_half
            or int(_device_signature_target_half) == half_index + 1
        )
    ):
        flush_bpref_device_panel_accumulator(
            iteration=iteration_index + 1,
            half=half_index + 1,
        )


def flush_bpref_device_panel_accumulator(*, iteration: int, half: int) -> None:
    """Write and release every exact native class panel for one half."""

    dump_dir = os.environ.get("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "").strip()
    if not dump_dir:
        return
    run_id = os.environ.get("RELAX_BPREF_CONTRIBUTION_DUMP_RUN_ID", "unset")
    prefix = (int(iteration), int(half), run_id)
    keys = sorted(key for key in _bpref_device_panel_metadata if key[:3] == prefix)
    if not keys:
        raise RuntimeError(f"No RECOVAR device panel metadata exists for {prefix}")
    output = Path(dump_dir)
    output.mkdir(parents=True, exist_ok=True)
    for key in keys:
        accumulators = _bpref_device_panel_accumulators.pop(key, None)
        launch_count = _bpref_device_panel_launch_counters.pop(key, 0)
        metadata = _bpref_device_panel_metadata.pop(key)
        if accumulators is None:
            raise RuntimeError(f"No RECOVAR device panel accumulator exists for {key}")
        data_accumulator, weight_accumulator = accumulators
        class_index = int(metadata["class_index"])
        np.savez(
            output
            / (
                f"recovar_device_panel_native_it{int(iteration):03d}_h{int(half)}"
                f"_class{class_index + 1:03d}_rank{int(metadata['rank']):03d}.npz"
            ),
            magic=np.asarray("RECOVAR_DEVICE_PANEL_NATIVE"),
            schema=np.asarray("recovar-device-panel-native-v1"),
            schema_version=np.int32(1),
            run_id=np.asarray(run_id),
            iteration=np.int32(iteration),
            half=np.int32(half),
            class_index=np.int32(class_index),
            rank=np.int32(metadata["rank"]),
            launch_count=np.int64(launch_count),
            current_size=np.int32(metadata["current_size"]),
            max_r=np.float32(metadata["max_r"]),
            image_shape=np.asarray(metadata["image_shape"], dtype=np.int32),
            volume_shape=np.asarray(metadata["volume_shape"], dtype=np.int32),
            reconstruction_padding_factor=np.int32(metadata["reconstruction_padding_factor"]),
            source_stack_sha256=np.asarray(metadata["source_stack_sha256"]),
            causal_arm=np.asarray(metadata["causal_arm"]),
            winner_take_all=np.bool_(metadata["winner_take_all"]),
            topology_claim=np.asarray("causal-arm-not-relion-hypothesis-arithmetic-closure"),
            accumulator_field_legend=np.asarray("data=complex64 x-half;weight=float32 x-half;flat C order"),
            data_accumulator=np.asarray(data_accumulator),
            weight_accumulator=np.asarray(weight_accumulator),
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
        "sequential_translation_reduction": bool(
            scope_active and relion_x_half_sequential_translation_reduction_enabled()
        ),
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


def _bpref_membership_dump_requested():
    dump_dir = os.environ.get(_BPREF_MEMBERSHIP_DUMP_DIR_ENV, "").strip()
    if not dump_dir:
        return False
    context_iteration = int(_bpref_contribution_context["iteration"])
    context_half = int(_bpref_contribution_context["half"])
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
            "sequential_translation_reduction",
            "per_particle_launches",
            "fused_atomics",
            "high_precision_operand_bundle",
        )
    )


def _relion_firstiter_bpref_diagnostics_active(
    *,
    bpref_device_signature_active: bool,
) -> bool:
    """Resolve the diagnostics that make deferred firstiter BPref unsafe."""

    device_signature_configured = bool(
        os.environ.get("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "").strip()
    )
    return bool(
        (device_signature_configured and bpref_device_signature_active)
        or (
            os.environ.get("RELAX_BPREF_CONTRIBUTION_DUMP_DIR", "").strip()
            and (bpref_device_signature_active or not device_signature_configured)
        )
        or _bpref_membership_dump_requested()
        or _scoped_bpref_diagnostics_active(
            _scoped_bpref_diagnostic_flags(active=bpref_device_signature_active)
        )
        or os.environ.get(_BPREF_ACCUMULATOR_DELTA_DUMP_DIR_ENV, "").strip()
        or os.environ.get("RELAX_PASS2_DUMP_DIR", "").strip()
        or os.environ.get("RELAX_SPARSE_PASS2_NATIVE_DUMP_DIR", "").strip()
    )
