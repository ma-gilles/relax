"""Precision policy helpers for dense single-volume EM."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

_REPORTED_PRECISION_VIOLATIONS: set[tuple[str, tuple[str, ...]]] = set()


def use_float32_matmuls() -> None:
    """Run float32 and complex64 matmuls in float32 for the rest of the process.

    JAX's default matmul precision lets XLA use TF32 tensor cores on A100/H100 (about 3e-4 relative
    error, against 4e-7 in float32), while P100 runs float32. RELION computes in float32. Every
    relax EM matmul passes ``precision=HIGHEST`` (tests/unit/test_matmul_precision_lint.py); this
    default is the EM commands' backstop for ``@`` products and calls the lint cannot see. PPCA
    chooses its precision per call, which overrides this default.
    """

    jax.config.update("jax_default_matmul_precision", "highest")


def _real_component_bytes(dtype) -> int | None:
    """Return the width of one real component, or None for non-floating dtypes."""

    dtype = np.dtype(dtype)
    if dtype.kind == "c":
        return dtype.itemsize // 2
    if dtype.kind == "f":
        return dtype.itemsize
    return None


def audit_operand_precision(policy, operands: Mapping[str, object], *, where: str, mode: str):
    """Report operands carried wider than the precision policy allows.

    Production EM precision is float32 (``relax/AGENTS.md``). The ``cast_*``
    helpers above narrow the score operands unconditionally but narrow the
    reconstruction operands only when float64 scoring is *on*, so a single
    float64 factor upstream -- a binary64 sigma2 spectrum, say -- silently
    promotes reconstruction and M-step rows to complex128 under a float32
    policy. That costs bandwidth everywhere and is invisible in results.

    This audit reads dtypes only (no device synchronisation) and is meant to sit
    at operand boundaries. ``mode`` ``warn`` warns once per distinct
    violation, ``raise`` turns it into an error and ``off`` disables it.
    """

    if mode == "off":
        return ()
    limit = 8 if policy.use_float64_scoring else 4
    offenders = tuple(
        f"{name}={np.dtype(value.dtype).name}"
        for name, value in operands.items()
        if value is not None
        and getattr(value, "dtype", None) is not None
        and (_real_component_bytes(value.dtype) or 0) > limit
    )
    if not offenders:
        return ()
    message = (
        f"EM operand precision: {where} returned {len(offenders)} operand(s) wider than the "
        f"{'float64' if policy.use_float64_scoring else 'float32'} precision policy: "
        f"{', '.join(offenders)}. Production EM precision is float32; a float64 factor upstream "
        f"promotes these rows and every consumer of them. Audit with mode='off' to silence."
    )
    if mode == "raise":
        raise ValueError(message)
    key = (where, offenders)
    if key not in _REPORTED_PRECISION_VIOLATIONS:
        _REPORTED_PRECISION_VIOLATIONS.add(key)
        logger.warning("%s", message)
    return offenders


@dataclass(frozen=True)
class DensePrecisionPolicy:
    """Centralized dtype choices for dense EM scoring/projection paths."""

    use_float64_scoring: bool = False
    use_float64_projections: bool = False

    def for_local_pass(self, iteration: int | None, *, pass_index: int, float64_pass2_iterations: frozenset[int]):
        if pass_index == 2 and _diagnostic_float64_pass2_matches(iteration, float64_pass2_iterations):
            return replace(self, use_float64_scoring=True, use_float64_projections=True)
        return self

    @property
    def rotation_real_dtype(self):
        return np.float64 if (self.use_float64_scoring or self.use_float64_projections) else np.float32

    @property
    def score_complex_dtype(self):
        return jnp.complex128 if self.use_float64_scoring else jnp.complex64

    @property
    def score_real_dtype(self):
        return jnp.float64 if self.use_float64_scoring else jnp.float32

    @property
    def projection_complex_dtype(self):
        return jnp.complex128 if self.use_float64_projections else None

    def cast_local_noise_projection_scores(self, proj_for_noise, proj_abs2_for_noise):
        if not self.use_float64_scoring:
            return proj_for_noise, proj_abs2_for_noise
        if proj_abs2_for_noise is None:
            return proj_for_noise.astype(self.score_complex_dtype), None
        return (
            proj_for_noise.astype(self.score_complex_dtype),
            proj_abs2_for_noise.astype(self.score_real_dtype),
        )


def _diagnostic_float64_pass2_matches(debug_iteration: int | None, iterations: frozenset[int]) -> bool:
    """Select genuine-f64 pass 2 without perturbing an earlier f32 boundary: ``debug_iteration`` is one of
    ``iterations`` (``RefinementEnvironment.diagnostic_float64_pass2_iterations``)."""

    return debug_iteration is not None and int(debug_iteration) in iterations
