"""Diagnostic scoring overrides, read once from the environment (code rule 5).

:class:`ScoringOverrides` holds the environment switches that change computed values: each one replaces a step of
the production scoring path with another for diagnosis or parity work. Every default is the production path,
RELION-faithful, and the default instance is the unset environment. Unlike ``relax.runtime.engine_tuning``
(execution settings that change no value), a run with an override set computes different numbers.
:meth:`ScoringOverrides.from_environ` reads them through ``relax.runtime.env_flags``; the commands call it once
(``RefinementEnvironment.scoring``, ``VdamEnvironment.scoring``) and the scoring engines receive the record as an
argument.
"""

from __future__ import annotations

from dataclasses import dataclass

from relax.runtime.env_flags import parse_env_binary_flag, parse_env_flag, parse_env_strict_flag, read_env_text

F32_COARSE_SUPPORT_ENV = "RECOVAR_K1_RELION_F32_COARSE_SUPPORT"
COARSE_GEMM_FLOAT64_ENV = "RELAX_COARSE_GEMM_FLOAT64"
COARSE_ROTATED_RADIUS_ENV = "RELAX_K1_COARSE_ROTATED_RADIUS"
EXACT_FLOAT32_ACCUMULATION_ENV = "RELAX_EM_EXACT_FLOAT32_ACCUMULATION"
X_HALF_MSTEP_DOUBLE_ENV = "RELAX_RELION_X_HALF_MSTEP_DOUBLE"
PROJECTOR_TEXTURE_INTERP_ENV = "RELAX_RELION_PROJECTOR_TEXTURE_INTERP"
DETERMINISTIC_REDUCTIONS_ENV = "RELAX_EM_DETERMINISTIC_REDUCTIONS"


@dataclass(frozen=True, kw_only=True)
class ScoringOverrides:
    """Diagnostic switches that change computed values; each default is the production RELION-faithful path.

    ``f32_coarse_support`` (RECOVAR_K1_RELION_F32_COARSE_SUPPORT, :func:`parse_strict_flag`; default on): the
    coarse pass's significance support follows RELION's CUDA float32 coarse weights; off selects the generic
    Gaussian support route (a diagnosis of the support, used by tests).
    ``coarse_gemm_float64`` (RELAX_COARSE_GEMM_FLOAT64, :func:`parse_coarse_gemm_float64`; default off): the coarse
    GEMMs accumulate in binary64 instead of RELION's float32 (a precision diagnosis).
    ``coarse_rotated_radius`` (RELAX_K1_COARSE_ROTATED_RADIUS, :func:`parse_coarse_rotated_radius`; default None,
    RELION's rotated-radius clipping wherever the compact RELION texture projector runs): False keeps the legacy
    source-pixel disk, True asks for the rotated radius (refused without the compact projector).
    ``exact_float32_accumulation`` (RELAX_EM_EXACT_FLOAT32_ACCUMULATION, ``env_flags.parse_env_binary_flag``; default
    off): the M-step scatters backprojected rows into a zeroed partial pair flushed into the totals every
    ``half_volume_mstep.ROW_PARTIAL_FLUSH_ROWS`` rows (``half_volume_mstep.allocate_row_partial``), for one-to-one
    comparison with RELION builds that accumulate in double; production keeps stock RELION's direct float32
    accumulation (relax#65, user 2026-10-11).
    ``x_half_mstep_double`` (RELAX_RELION_X_HALF_MSTEP_DOUBLE, ``env_flags.parse_env_flag``; default off): the RELION
    x-half M-step accumulates ``Ft_y``/``Ft_ctf`` in complex128/float64 instead of RELION's single-precision
    ``BackProjector`` (``half_volume_mstep.relion_x_half_mstep_accumulator_dtypes``), a diagnostic for isolating
    boundary-shell differences.
    ``projector_texture_interp`` (RELAX_RELION_PROJECTOR_TEXTURE_INTERP, ``env_flags.parse_env_strict_flag``; default
    on): a RELION projection that asks for no interpolator of its own (``relion_texture_interp=None``) takes RELION's
    CUDA texture interpolator where it is available; off takes the manual float interpolation, which the float64
    reference tests compare against (the texture unit weighs with 8-bit fixed-point fractions).
    ``deterministic_reductions`` (RELAX_EM_DETERMINISTIC_REDUCTIONS, ``env_flags.parse_env_binary_flag``; default off):
    the EM engines' duplicate-index sums (per-shell noise and power binning, per-image row sums, the projector and VDAM
    M-step shell sums) take a fixed-order masked reduction (``relax.numerics.deterministic_reduce``) instead of the
    scatter-add, whose GPU association order varies between runs; the same operands, another summation order.
    """

    f32_coarse_support: bool = True
    coarse_gemm_float64: bool = False
    coarse_rotated_radius: bool | None = None
    exact_float32_accumulation: bool = False
    x_half_mstep_double: bool = False
    projector_texture_interp: bool = True
    deterministic_reductions: bool = False

    @classmethod
    def from_environ(cls) -> ScoringOverrides:
        return cls(
            f32_coarse_support=parse_strict_flag(F32_COARSE_SUPPORT_ENV, read_env_text(F32_COARSE_SUPPORT_ENV), True),
            coarse_gemm_float64=parse_coarse_gemm_float64(read_env_text(COARSE_GEMM_FLOAT64_ENV, "0")),
            coarse_rotated_radius=parse_coarse_rotated_radius(read_env_text(COARSE_ROTATED_RADIUS_ENV)),
            exact_float32_accumulation=parse_env_binary_flag(EXACT_FLOAT32_ACCUMULATION_ENV),
            x_half_mstep_double=parse_env_flag(X_HALF_MSTEP_DOUBLE_ENV),
            projector_texture_interp=parse_env_strict_flag(PROJECTOR_TEXTURE_INTERP_ENV, default=True),
            deterministic_reductions=parse_env_binary_flag(DETERMINISTIC_REDUCTIONS_ENV),
        )


def parse_strict_flag(name: str, raw: str | None, default: bool) -> bool:
    """A strict boolean (``env_flags.parse_env_strict_flag``'s rule): unset is ``default``; a set value (stripped,
    any case) must be 1/true/yes/on or 0/false/no/off, anything else (blank too) is refused naming ``name``."""
    token = ("1" if default else "0") if raw is None else raw.strip().lower()
    if token in {"0", "false", "no", "off"}:
        return False
    if token in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(f"Unsupported {name}={token!r}")


def parse_coarse_gemm_float64(raw: str) -> bool:
    """RELAX_COARSE_GEMM_FLOAT64 (stripped): exactly 0 or 1; anything else (blank too) is refused."""
    token = raw.strip()
    if token not in {"0", "1"}:
        raise ValueError(f"Unsupported {COARSE_GEMM_FLOAT64_ENV}={token!r}")
    return token == "1"


def parse_coarse_rotated_radius(raw: str | None) -> bool | None:
    """RELAX_K1_COARSE_ROTATED_RADIUS (not stripped): unset is None; exactly 0 or 1; anything else (blank too) is
    refused."""
    if raw is None:
        return None
    if raw not in {"0", "1"}:
        raise ValueError(f"{COARSE_ROTATED_RADIUS_ENV} must be 0 or 1")
    return raw == "1"
