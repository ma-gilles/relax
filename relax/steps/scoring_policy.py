"""Execution defaults and existing diagnostic selectors for half-set scoring.

The dense precision is read by the command (``relax.numerics.precision.DensePrecisionPolicy.from_environ``)
and passed down. The scoring route variants are read once per run, when the refinement's options are built
(``relax.refinement.refinement_options.ScoringVariants``); this module keeps their variable names and
K=1's x-half default. This module performs no scoring or scheduling.
"""

import logging

import jax
import numpy as np

from relax.numerics.precision import DensePrecisionPolicy
from relax.runtime import process_platform

logger = logging.getLogger(__name__)

# RELION parses ``--adaptive_fraction 0.999`` through ``textToFloat`` and
# stores that single-precision value in its optimiser state.  Python's literal
# 0.999 is a different binary64 boundary and can change a one-sample support
# cutoff when posterior weights are accumulated in float64.
RELION_ADAPTIVE_FRACTION = float(np.float32("0.999"))

LOCAL_ADAPTIVE_PASS2_FULL_PARENT_ENV = "RELAX_LOCAL_ADAPTIVE_PASS2_FULL_PARENT"
LOCAL_ADAPTIVE_PASS2_ROTATION_ONLY_ENV = "RELAX_LOCAL_ADAPTIVE_PASS2_ROTATION_ONLY"
LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT_ENV = "RELAX_LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT"
K1_RELION_X_HALF_MSTEP_ENV = "RELAX_K1_RELION_X_HALF_MSTEP"
K_CLASS_RELION_X_HALF_MSTEP_ENV = "RELAX_K_CLASS_RELION_X_HALF_MSTEP"
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

def local_precision(
    precision: DensePrecisionPolicy, iteration: int | None, *, pass_index: int, float64_pass2_iterations: frozenset[int]
) -> DensePrecisionPolicy:
    """The run's ``precision`` for one local-search pass, with the configured float64 diagnostic override
    (``float64_pass2_iterations``, ``RefinementEnvironment.diagnostic_float64_pass2_iterations``)."""
    if pass_index not in (1, 2):
        raise ValueError(f"local-search pass_index must be 1 or 2, got {pass_index}")
    return precision.for_local_pass(iteration, pass_index=pass_index, float64_pass2_iterations=float64_pass2_iterations)


def k1_relion_x_half_mstep_default_available() -> bool:
    """Return whether the default K=1 x-half M-step can use custom CUDA."""

    if process_platform.custom_cuda_disabled() or process_platform.jax_cpu_forced():
        return False
    try:
        return jax.default_backend() == "gpu"
    except Exception:
        return False
