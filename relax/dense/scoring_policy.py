"""Execution defaults and existing diagnostic selectors for half-set scoring.

Global precision and fine-scoring policy retain their import-time environment
snapshot. The scoring route variants are read once per run, when the refinement's options are built
(``relax.refinement.refinement_options.ScoringVariants``); this module keeps their variable names and
K=1's x-half default. This module performs no scoring or scheduling.
"""

import logging
import os

import jax
import numpy as np

from relax.helpers.dtype_policy import DensePrecisionPolicy
from relax.helpers.env_flags import parse_env_true_flag

logger = logging.getLogger(__name__)

# RELION parses ``--adaptive_fraction 0.999`` through ``textToFloat`` and
# stores that single-precision value in its optimiser state.  Python's literal
# 0.999 is a different binary64 boundary and can change a one-sample support
# cutoff when posterior weights are accumulated in float64.
RELION_ADAPTIVE_FRACTION = float(np.float32("0.999"))

_LOCAL_ADAPTIVE_PASS2_FULL_PARENT_ENV = "RELAX_LOCAL_ADAPTIVE_PASS2_FULL_PARENT"
_LOCAL_ADAPTIVE_PASS2_ROTATION_ONLY_ENV = "RELAX_LOCAL_ADAPTIVE_PASS2_ROTATION_ONLY"
_LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT_ENV = "RELAX_LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT"
_K1_RELION_X_HALF_MSTEP_ENV = "RELAX_K1_RELION_X_HALF_MSTEP"
_K_CLASS_RELION_X_HALF_MSTEP_ENV = "RELAX_K_CLASS_RELION_X_HALF_MSTEP"
# RELAX_LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT's tokens (refinement_options.ScoringVariants reads it).
LOCAL_ADAPTIVE_PASS2_DENOMINATOR_MODES = {
    **dict.fromkeys(("0", "false", "no", "off", "none", "default", "pruned", "pruned_parent")),
    **dict.fromkeys(("rotation", "rotations", "rotation_only", "significant_rotation_full_translation"), "rotation_only"),
    **dict.fromkeys(("full", "full_parent", "all", "all_parent"), "full_parent"),
}

# RELION stores windowFourierTransform(in, out, current_size) as a rectangular
# FFTW half image, but the likelihood support is the nonzero Minvsigma2 mask:
# rounded radial shells, no DC, no redundant negative-row kx=0 entries.
RELION_FOURIER_WINDOW_SQUARE = False

# Resolve the diagnostic switches once, at the existing import boundary.
DENSE_PRECISION = DensePrecisionPolicy(
    use_float64_scoring=parse_env_true_flag("RELAX_USE_FLOAT64_SCORING"),
    use_float64_projections=parse_env_true_flag("RELAX_USE_FLOAT64_PROJECTIONS"),
)
RELION_EXACT_FINE_GAUSSIAN = not parse_env_true_flag(
    "RELAX_DISABLE_RELION_EXACT_FINE_GAUSSIAN"
)

def local_precision(iteration: int | None, *, pass_index: int) -> DensePrecisionPolicy:
    """Resolve production precision and the configured float64 diagnostic override."""
    if pass_index not in (1, 2):
        raise ValueError(f"local-search pass_index must be 1 or 2, got {pass_index}")
    return DENSE_PRECISION.for_local_pass(iteration, pass_index=pass_index)


# Off by default: reproduces RELION's GPU-accelerated projector/backprojector
# narrowing coordinates to float32 before flooring, unconditionally, even under
# ``ACC_DOUBLE_PRECISION`` (see ``recovar.em.relion.relion_project`` module
# docstring). Set ``RELAX_RELION_ACC_DOUBLE_FLOORF_QUIRK=1`` to bit-match
# that GPU-double quirk in the local-search fine-pass projector fallback (it
# only has an effect when the texture path is unavailable, e.g. under
# ``use_float64_scoring``/``use_float64_projections``, since CUDA textures
# cannot hold complex128).
RELION_ACC_DOUBLE_FLOORF_QUIRK = parse_env_true_flag(
    "RELAX_RELION_ACC_DOUBLE_FLOORF_QUIRK"
)


def _jax_cpu_forced_from_env() -> bool:
    """Return whether JAX has been forced to CPU by environment."""

    platform_name = os.environ.get("JAX_PLATFORM_NAME", "").strip().lower()
    if platform_name == "cpu":
        return True
    platforms = os.environ.get("JAX_PLATFORMS", "").strip().lower()
    if not platforms:
        return False
    requested = [token.strip() for token in platforms.split(",") if token.strip()]
    return bool(requested) and all(token == "cpu" for token in requested)


def _k1_relion_x_half_mstep_default_available() -> bool:
    """Return whether the default K=1 x-half M-step can use custom CUDA."""

    from recovar.utils.cuda_env import custom_cuda_disabled_from_env

    disabled, _ = custom_cuda_disabled_from_env()
    if disabled or _jax_cpu_forced_from_env():
        return False
    try:
        return jax.default_backend() == "gpu"
    except Exception:
        return False


def _dense_global_scoring_dtype() -> np.dtype:
    """Dtype for the dense/global (``use_local=False``) scoring path's
    float64-sensitive operands: the pass-1 rotation grid built by
    ``relion_scoring_rotation_grid``, and the offset/orientation log-prior
    arrays built by ``make_relion_translation_log_prior`` /
    ``make_relion_direction_log_prior`` and their prior-center helpers.

    None of these have a per-iteration diagnostic override (see
    ``local_precision`` for the local-search analog), so this
    collapses the global float64-scoring/-projections switches directly,
    matching RELION's ``ACC_DOUBLE_PRECISION`` build where the corresponding
    host ``RFLOAT`` values are never narrowed to float before the (no-op)
    ``XFLOAT`` cast.
    """

    return DENSE_PRECISION.rotation_real_dtype
