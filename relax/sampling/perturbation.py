"""RELION's sampling perturbation.

The per-iteration random perturbation (its advance from RELION's generator), its application to rotation rows,
Euler rows and translations, and the perturbed trial grids built from it (the coarse trial grid, the exact local
fine grid and the local-search M-step rotations).
"""

import jax.numpy as jnp
import numpy as np

from relax.numerics import relion_random
from relax.sampling import grids, healpix
from relax.sampling.symmetry import canonicalize_rotational_symmetry


def _wrap_relion_perturbation(value, perturbation_factor):
    """RELION ``realWRAP(value, -pf, +pf)`` for SamplingPerturbation."""
    pf = float(perturbation_factor)
    wrapped = float(value)
    while wrapped > pf:
        wrapped -= 2 * pf
    while wrapped < -pf:
        wrapped += 2 * pf
    return float(wrapped)


def _relion_rnd_unif_scaled_first_draw(seed, low, high):
    """Return RELION ``rnd_unif(low, high)`` after seeding, bit-faithfully.

    RELION's ``rnd_unif`` accepts float arguments and evaluates the range
    scaling inside the C++ function.  Scaling a separately rounded
    ``rnd_unif(0, 1)`` result in Python is not equivalent: the difference is
    observable in SamplingPerturbation Euler matrices at the outer M-step
    radius. :func:`relax.numerics.relion_random.rnd_unif` keeps each float
    operation of funcs.cpp::rnd_unif.
    """
    return float(relion_random.rnd_unif_sequence(int(seed), 1, float(low), float(high))[0])


def advance_relion_perturbation(prev_random_perturbation, perturbation_factor, rng):
    """Update the RELION per-iteration perturbation state.

    Ports `HealpixSampling::resetRandomlyPerturbedSampling` (healpix_sampling.cpp:167-174):
        random_perturbation += rnd_unif(0.5 * perturbation_factor, perturbation_factor)
        random_perturbation = realWRAP(random_perturbation, -pf, +pf)

    Parameters
    ----------
    prev_random_perturbation : float
        Previous iteration's random_perturbation (initialize to 0.0 before iter 1).
    perturbation_factor : float
        Typically 0.5 (RELION default `--perturb 0.5`).
    rng : np.random.Generator

    Returns
    -------
    float : new random_perturbation, wrapped to [-pf, +pf].
    """
    pf = float(perturbation_factor)
    new = prev_random_perturbation + rng.uniform(0.5 * pf, pf)
    return _wrap_relion_perturbation(new, pf)


def advance_relion_perturbation_from_seed(prev_random_perturbation, perturbation_factor, seed):
    """Advance SamplingPerturbation using RELION's per-iteration RNG seed."""
    pf = float(perturbation_factor)
    increment = _relion_rnd_unif_scaled_first_draw(int(seed), 0.5 * pf, pf)
    new = float(prev_random_perturbation) + increment
    return _wrap_relion_perturbation(new, pf)


def advance_relion_perturbation_for_iteration(random_perturbation, *, perturb_factor, perturb_seed, relion_iteration, rng):
    """Advance RELION's SamplingPerturbation to ``relion_iteration``.

    With an explicit seed RELION draws the iteration's perturbation from
    ``random_seed + iteration``; without one the run's generator draws it.
    Returns ``(random_perturbation, seed)`` with ``seed`` ``None`` on the
    generator path. The regular iterations and the final all-data pass share
    this rule.
    """

    if perturb_seed is not None:
        seed = int(perturb_seed) + int(relion_iteration)
        return advance_relion_perturbation_from_seed(random_perturbation, perturb_factor, seed=seed), seed
    return advance_relion_perturbation(random_perturbation, perturb_factor, rng), None


def relion_sampling_perturbation_for_iteration(
    perturbation_factor,
    random_seed,
    relion_iteration,
    *,
    restart_state_iteration: int | None = None,
):
    """Return RELION's stored SamplingPerturbation at ``run_itNNN``.

    ``run_it000_sampling.star`` is written after the initial sampling object has
    already advanced once from the C RNG default state. Later expectation
    iterations re-seed with ``random_seed + iter`` before advancing.

    ``restart_state_iteration`` records an explicit RELION continuation
    boundary. ``HealpixSampling::read`` restores the perturbation factor but
    not ``random_perturbation``; the later ``initialise`` call therefore
    advances a clear sampling object with RELION's seed-1 stream. The saved
    state at that boundary is that seed-1 initial value rather than the value
    obtained by advancing one uninterrupted process from iteration zero. The
    next expectation still uses ``random_seed + iter``. This option is only
    for replaying a provenance-qualified, stitched RELION trajectory.
    """
    if relion_iteration < 0:
        raise ValueError("relion_iteration must be non-negative")
    if restart_state_iteration is not None:
        if restart_state_iteration < 0:
            raise ValueError("restart_state_iteration must be non-negative")
        if restart_state_iteration >= int(relion_iteration):
            raise ValueError(
                "restart_state_iteration must precede relion_iteration "
                f"({restart_state_iteration} >= {int(relion_iteration)})"
            )
    current = advance_relion_perturbation_from_seed(0.0, perturbation_factor, seed=1)
    first_advance = 1 if restart_state_iteration is None else restart_state_iteration + 1
    for iter_idx in range(first_advance, int(relion_iteration) + 1):
        current = advance_relion_perturbation_from_seed(
            current,
            perturbation_factor,
            seed=int(random_seed) + iter_idx,
        )
    return float(current)


def apply_relion_rotation_perturbation_to_eulers(
    eulers_deg,
    random_perturbation,
    angular_sampling_deg,
    *,
    dtype: np.dtype = np.float32,
):
    """Apply RELION's SamplingPerturbation and return eulers plus matrices.

    RELION first converts the perturbed matrix back to RFLOAT Euler angles.
    Its fine-score and weighted-sum paths then call host
    ``generateEulerMatrices(..., inverse=true)`` and cast those matrices to
    XFLOAT before copying them to the device. Use the same host-double
    reconstruction here. Adaptive coarse scoring has a distinct matrix
    path exposed by :func:`relion_adaptive_pass1_rotations`.

    ``dtype`` controls the returned rotation-matrix and Euler precision (default
    float32, matching RELION's single-precision ACC build's XFLOAT cast).
    Pass ``np.float64`` to match ``ACC_DOUBLE_PRECISION``, where that cast is
    a no-op. RELION stores these working Euler angles as RFLOAT, so retaining
    float64 here is also required for a double-precision build.

    Scoring and M-step matrices share this host-generated path; callers supply
    the appropriate source Euler precision before calling.
    """
    eulers = np.asarray(eulers_deg, dtype=np.float64).reshape(-1, 3)
    if abs(float(random_perturbation)) < 1e-12:
        rotations = grids.relion_mstep_rotations_from_eulers(eulers, dtype=dtype)
        return rotations, eulers.astype(dtype)

    myperturb = float(random_perturbation) * float(angular_sampling_deg)
    A = healpix.euler_angles_to_matrix(eulers)
    R_perturb = healpix.euler_angles_to_matrix(np.array([[myperturb, myperturb, myperturb]], dtype=np.float64))[0]
    perturbed_A = healpix.matmul3(A, R_perturb)
    perturbed_eulers = healpix.euler_matrix_to_angles(perturbed_A)
    perturbed_rotations = grids.relion_mstep_rotations_from_eulers(perturbed_eulers, dtype=dtype)
    return perturbed_rotations, perturbed_eulers.astype(dtype)


def apply_relion_translation_perturbation(translations, random_perturbation, offset_step_pixels):
    """Port of RELION's translation perturbation (healpix_sampling.cpp:1810-1820).

    Adds ``myperturb = random_perturbation * offset_step_pixels`` to both axes
    of each translation vector.
    """
    if abs(random_perturbation) < 1e-12:
        return translations
    myperturb = float(random_perturbation) * float(offset_step_pixels)
    return translations + np.asarray(myperturb, dtype=translations.dtype)


def perturbed_trial_grid(
    *,
    rotation_eulers,
    mstep_source_eulers,
    base_translations,
    translation_step: float,
    random_perturbation: float,
    angular_sampling_deg: float,
    dtype,
) -> grids.TrialGrid:
    """Apply RELION's SamplingPerturbation to a trial grid.

    ``healpix_sampling.cpp:1909-1934`` rotates every trial orientation by the
    same rigid SO(3) perturbation after oversampling and ``1810-1820`` shifts the
    translation grid; the exact M-step rotations are rebuilt from
    ``mstep_source_eulers`` with the same perturbation. The regular iterations
    and the final all-data pass share this rule.
    """

    rotations, rotation_eulers = apply_relion_rotation_perturbation_to_eulers(
        rotation_eulers,
        random_perturbation,
        angular_sampling_deg,
        dtype=dtype,
    )
    mstep_rotations, _ = apply_relion_rotation_perturbation_to_eulers(
        mstep_source_eulers,
        random_perturbation,
        angular_sampling_deg,
        dtype=dtype,
    )
    translations = jnp.asarray(
        apply_relion_translation_perturbation(
            np.asarray(base_translations),
            random_perturbation,
            translation_step,
        ),
        dtype=dtype,
    )
    return grids.TrialGrid(rotations, rotation_eulers, mstep_rotations, translations)


def exact_local_fine_grid(*, healpix_order, angular_sampling_deg, random_perturbation, dtype=np.float32, symmetry="C1"):
    """Materialize RELION's fine local-search grid once, with its SamplingPerturbation.

    RELION rotates every fine orientation by the iteration's perturbation
    (``healpix_sampling.cpp:1909-1934``) and rebuilds the exact M-step matrices
    from the canonical RFLOAT angles with the same perturbation.  ``None`` keeps
    the unperturbed grid matrices for a pass that drew no perturbation.  A
    point group ``symmetry`` materializes its reduced grid and source angles.
    Returns ``(rotations, rotation_eulers, mstep_rotations)``: the scoring rotations and angles in ``dtype``,
    perturbed or not, and the M-step matrices in float32.
    """

    symmetry = canonicalize_rotational_symmetry(symmetry)
    rotation_grid = grids.relion_scoring_rotation_grid(healpix_order, dtype=dtype, symmetry=symmetry)
    rotations = rotation_grid.rotations
    rotation_eulers = rotation_grid.rotation_eulers
    del rotation_grid
    if random_perturbation is not None:
        rotations, rotation_eulers = apply_relion_rotation_perturbation_to_eulers(
            rotation_eulers,
            float(random_perturbation),
            angular_sampling_deg,
            dtype=dtype,
        )
    mstep_rotations, _ = apply_relion_rotation_perturbation_to_eulers(
        grids.get_relion_rotation_grid_eulers_float64(healpix_order, symmetry=symmetry),
        0.0 if random_perturbation is None else float(random_perturbation),
        angular_sampling_deg,
    )
    return rotations, rotation_eulers, mstep_rotations


def local_search_mstep_rotations(effective_mstep_rotations, rotation_eulers, healpix_order, *, symmetry="C1"):
    """Exact M-step rotations of a local search that reuses the scoring grid.

    The perturbed trial grid already carries its M-step matrices; a grid
    without them rebuilds RELION's host-inverse matrices from the M-step
    source angles at ``healpix_order`` (no further perturbation).
    """

    if effective_mstep_rotations is not None:
        return effective_mstep_rotations
    mstep_rotations, _ = apply_relion_rotation_perturbation_to_eulers(
        grids.relion_mstep_source_eulers(rotation_eulers, healpix_order, symmetry=symmetry),
        0.0,
        grids.relion_angular_sampling_deg(healpix_order, adaptive_oversampling=0),
    )
    return mstep_rotations
