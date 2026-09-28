"""Execution-policy switches of the sparse bucketed pass 2.

The environment-gated opt-ins and per-pass policy values (BPref processing
order and execution buckets, native weighted sums, fused M-step noise, the
RELION wavg direct modes, projection cache, windowed prepare and translation
tiles, compact-pair execution and bucket coalescing). Each reader is the
single owner of its variable; the pass-2 kernels ask these once per pass.
"""

from __future__ import annotations

import os

import numpy as np

from relax.helpers.env_flags import parse_env_flag

_RELION_WAVG_ATOMIC_SCALE_AA_ENV = "RELAX_RELION_WAVG_ATOMIC_SCALE_AA"


_RELION_WAVG_ATOMIC_DIRECT_RESIDUAL_ENV = (
    "RELAX_RELION_WAVG_ATOMIC_DIRECT_RESIDUAL"
)


_RELION_WAVG_ATOMIC_DIRECT_NOISE_ONLY_ENV = (
    "RELAX_RELION_WAVG_ATOMIC_DIRECT_NOISE_ONLY"
)


_RELION_POWERCLASS_SPECTRUM_NORM_ENV = "RELAX_K1_RELION_POWERCLASS_SPECTRUM_NORM"


class ResidentConfigurationUnsupported(NotImplementedError):
    """A device-resident K=1 driver does not implement this pass's configuration.

    Raised by the resident drivers' configuration checks, which run before any
    device work. It stops the run: there is no other pass-2 engine to fall back to.
    """


_RELION_EXACT_BPREF_OPERANDS_ENV = "RELAX_K1_RELION_EXACT_BPREF_OPERANDS"


_BPREF_EXECUTION_ORDER_LOCAL_FILE_ENV = "RELAX_K1_BPREF_EXECUTION_ORDER_LOCAL_FILE"


_BPREF_REVERSE_PHYSICAL_ORDER_ENV = "RELAX_K1_BPREF_REVERSE_PHYSICAL_ORDER"


_BPREF_EXECUTION_ORDER_CHUNK_SIZE_ENV = "RELAX_K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE"


_BPREF_EXECUTION_BATCH_CONSECUTIVE_EQUAL_SUPPORT_ENV = (
    "RELAX_K1_BPREF_EXECUTION_BATCH_CONSECUTIVE_EQUAL_SUPPORT"
)


_BPREF_EXECUTION_GROUP_BY_BUCKET_SIZE_ENV = (
    "RELAX_K1_BPREF_EXECUTION_GROUP_BY_BUCKET_SIZE"
)


_SPARSE_PASS2_PROJECTION_CACHE_ENV = "RELAX_SPARSE_PASS2_PROJECTION_CACHE"


_SPARSE_PASS2_WINDOWED_PREPARE_ENV = "RELAX_SPARSE_PASS2_WINDOWED_PREPARE"


def _native_dual_weighted_sums_supported_for_operands(
    *,
    requested: bool,
    accumulate_noise: bool,
    probability_dtype,
    reconstruction_dtype,
    noise_dtype,
) -> bool:
    """Return whether operands satisfy the native F32/C64 reduction contract."""

    return bool(
        requested
        and accumulate_noise
        and np.dtype(probability_dtype) == np.dtype(np.float32)
        and np.dtype(reconstruction_dtype) == np.dtype(np.complex64)
        and np.dtype(noise_dtype) == np.dtype(np.complex64)
    )


def _relion_powerclass_spectrum_norm_enabled(
    *,
    fresh_k1_guard: bool,
) -> bool:
    """Use RELION's shell spectrum by default only in the fresh K=1 guard."""

    return parse_env_flag(
        _RELION_POWERCLASS_SPECTRUM_NORM_ENV,
        default=bool(fresh_k1_guard),
    )


def _relion_exact_bpref_operands_enabled(
    *,
    fresh_k1_guard: bool,
    source_faithful_spectrum_norm: bool,
) -> bool:
    """Pair exact BPref with the qualified fresh-K=1 spectrum path."""

    return parse_env_flag(
        _RELION_EXACT_BPREF_OPERANDS_ENV,
        default=bool(fresh_k1_guard and source_faithful_spectrum_norm),
    )


def _relion_wavg_direct_modes(
    *,
    accumulate_noise: bool,
    scale_groups_available: bool,
    scale_aa_enabled: bool,
    direct_noise_only_default: bool = False,
) -> tuple[bool, bool]:
    """Resolve the stopped direct-Wavg noise/norm factorial arms.

    ``DIRECT_RESIDUAL`` preserves the existing coupled treatment: the native
    Wavg ``diff2`` stream supplies both shell noise and per-particle norm.
    ``DIRECT_NOISE_ONLY`` supplies only shell noise, leaving normalization on
    the production algebraic path.  The latter isolates the already-localized
    radial-noise boundary without silently changing a second state variable.
    """

    direct_residual_requested = bool(
        accumulate_noise
        and parse_env_flag(
            _RELION_WAVG_ATOMIC_DIRECT_RESIDUAL_ENV,
            default=False,
        )
    )
    direct_noise_only_requested = bool(
        accumulate_noise
        and parse_env_flag(
            _RELION_WAVG_ATOMIC_DIRECT_NOISE_ONLY_ENV,
            default=direct_noise_only_default,
        )
    )
    if direct_residual_requested and direct_noise_only_requested:
        raise ValueError(
            f"{_RELION_WAVG_ATOMIC_DIRECT_RESIDUAL_ENV}=1 and "
            f"{_RELION_WAVG_ATOMIC_DIRECT_NOISE_ONLY_ENV}=1 are mutually exclusive"
        )
    direct_noise = direct_residual_requested or direct_noise_only_requested
    # Fresh iteration 1 intentionally has no scale-group accumulator.  The
    # established coupled diagnostic is dormant there and activates once the
    # scale state exists; preserve that lifecycle for the isolated arm.
    if direct_noise and not scale_groups_available:
        return False, False
    if direct_noise and not scale_aa_enabled:
        requested_name = (
            _RELION_WAVG_ATOMIC_DIRECT_RESIDUAL_ENV
            if direct_residual_requested
            else _RELION_WAVG_ATOMIC_DIRECT_NOISE_ONLY_ENV
        )
        raise ValueError(
            f"{requested_name}=1 requires "
            f"{_RELION_WAVG_ATOMIC_SCALE_AA_ENV}=1 and scale groups"
        )
    return direct_noise, direct_residual_requested


def _projection_cache_enabled_for_pass(
    *,
    fine_rotations_override,
    dump_pass2_operands: bool,
) -> bool:
    """Resolve the diagnostic projection-cache override without changing defaults."""

    if fine_rotations_override is None:
        return False
    raw = os.environ.get(_SPARSE_PASS2_PROJECTION_CACHE_ENV)
    mode = "auto" if raw is None or raw.strip() == "" else raw.strip().lower()
    if mode == "auto":
        # Preserve the currently qualified paths while cache-on/cache-off is
        # adjudicated: production uses the cache and operand dumps do not.
        return not bool(dump_pass2_operands)
    if mode in {"1", "true", "yes", "on"}:
        return True
    if mode in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{_SPARSE_PASS2_PROJECTION_CACHE_ENV} must be 'auto', 'on', or 'off', got {raw!r}",
    )


def _windowed_prepare_enabled_for_pass(use_window: bool) -> bool:
    """Return whether sparse pass-2 should materialize only active Fourier windows."""

    return bool(
        use_window
        and parse_env_flag(
            _SPARSE_PASS2_WINDOWED_PREPARE_ENV,
            default=True,
        )
    )


