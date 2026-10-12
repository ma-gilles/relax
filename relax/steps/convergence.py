"""RELION's convergence measures over the particles' poses, shared by Refine3D/Class3D and VDAM.

The optimiser's average Pmax (``MlOptimiser::maximization``), the changes in the optimal orientations and offsets
between iterations (``monitorHiddenVariableChanges``, ``updateOverallChangesInHiddenVariables``) and the MPI
leader's test that the hidden-variable changes have stalled. Pure functions of arrays and scalars; the refinement
state that carries their results is ``relax.refinement.refinement_state``.
"""

from typing import Optional

import numpy as np


def relion_optimizer_average_pmax(max_posterior_per_half, normalization_mass_per_half=None):
    """Return RELION's optimizer Pmax scalar and its normalization mass.

    In split-half refinement, RELION computes this independently per half,
    then broadcasts half 1's scalar for shared scheduling. Class3D divides
    half 1's Pmax sum by that half's retained M-step posterior mass
    (``sum(wsum_model.pdf_class)``). The concatenated array is still returned
    for per-particle diagnostics.
    """

    per_half = [np.asarray(pmax).reshape(-1) for pmax in max_posterior_per_half]
    if not per_half:
        raise ValueError("RELION average Pmax requires at least one half-set")
    combined = np.concatenate(per_half, axis=0)
    numerator = float(np.sum(per_half[0], dtype=np.float64))
    if normalization_mass_per_half is None:
        denominator = float(per_half[0].size)
    else:
        half1_mass = normalization_mass_per_half[0]
        if half1_mass is None:
            raise ValueError("RELION average Pmax requires half-1 M-step posterior mass")
        denominator = float(np.asarray(half1_mass, dtype=np.float64))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError(f"RELION average-Pmax denominator must be finite and positive, got {denominator}")
    average = numerator / denominator
    if not np.isfinite(average) or average < 0.0 or average > 1.0 + 1e-6:
        raise ValueError(f"RELION average Pmax must be a probability, got {average}")
    return combined, float(average), denominator


def relion_mpi_hidden_variable_change_is_small(
    *,
    current_classes: float,
    current_offsets_angstrom: float,
    current_orientations_deg: float,
    smallest_classes: float,
    smallest_offsets_angstrom: float,
    smallest_orientations_deg: float,
    mpi_leader_angular_step_deg: float,
    mpi_leader_translation_step_angstrom: float,
) -> bool:
    """Evaluate RELION MPI's hidden-variable stall predicate.

    This deliberately accepts the sampling steps held by the MPI leader, not
    the current follower sampling.  In ``MlOptimiserMpi::expectation`` RELION
    calls ``updateAngularSampling`` only on followers.  For non-helical runs it
    broadcasts the updated counters and smallest-change trackers, but not the
    sampling object.  Later, rank 0 calls
    ``updateOverallChangesInHiddenVariables`` and therefore divides by the
    effective sampling steps that the leader had when the process (or restart)
    was initialized.  See ``ml_optimiser_mpi.cpp:1094-1125,2880-2903`` and
    ``ml_optimiser.cpp:9263-9273`` in RELION d476e6f.

    The caller is responsible for applying adaptive oversampling when it
    initializes the two fixed leader-step arguments.
    """
    angular_step = float(mpi_leader_angular_step_deg)
    translation_step = float(mpi_leader_translation_step_angstrom)
    ratio_orient = (
        float(current_orientations_deg) / angular_step
        if angular_step > 0.0
        else float("inf")
    )
    ratio_trans = (
        float(current_offsets_angstrom) / translation_step
        if translation_step > 0.0
        else float("inf")
    )
    return bool(
        1.03 * float(current_classes) >= float(smallest_classes)
        and (
            ratio_trans < 0.40
            or 1.03 * float(current_offsets_angstrom) >= float(smallest_offsets_angstrom)
        )
        and (
            ratio_orient < 0.40
            or 1.03 * float(current_orientations_deg) >= float(smallest_orientations_deg)
        )
    )


def _relion_angular_distance_per_particle(
    M_current: np.ndarray,
    M_previous: np.ndarray,
    *,
    symmetry_label: str = "C1",
) -> np.ndarray:
    """Per-particle RELION-style angular distance between two rotation matrices.

    Implements ``HealpixSampling::calculateAngularDistance`` (see
    ``relion/src/healpix_sampling.cpp:1969-2013``), including RELION's
    minimization over point-group symmetry mates.
    The distance for one matrix pair is::

        axes_dist = (1/3) * sum_{i=0..2} ACOSD(dot(E1[i,:], E2[i,:]))

    i.e., the mean of the angles between corresponding ROWS of the two
    Euler matrices, in degrees. For a nontrivial group, the second Euler
    matrix is transformed as ``L @ E2 @ R`` for every identity-inclusive
    RELION ``SymList`` operator pair and the smallest axes distance is used.

    Parameters
    ----------
    M_current, M_previous : np.ndarray, shape ``(N, 3, 3)``
        Per-particle rotation matrices for the current and previous
        iterations. Both must have the same shape.

    Returns
    -------
    np.ndarray, shape ``(N,)``
        Per-particle angular distance in degrees.
    """
    M_current = np.asarray(M_current, dtype=np.float64)
    M_previous = np.asarray(M_previous, dtype=np.float64)
    if M_current.shape != M_previous.shape:
        raise ValueError(
            f"M_current and M_previous must have the same shape, got {M_current.shape} vs {M_previous.shape}"
        )
    if M_current.ndim != 3 or M_current.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (N, 3, 3) rotation matrices, got {M_current.shape}")

    from relax.sampling.symmetry import canonicalize_rotational_symmetry

    symmetry_label = canonicalize_rotational_symmetry(symmetry_label)

    def _axes_distance(second: np.ndarray) -> np.ndarray:
        # einsum 'nij,nij->ni' gives shape (N, 3), with the dot product
        # between corresponding Euler-matrix rows in the final dimension.
        cos_per_row = np.einsum("nij,nij->ni", M_current, second)
        cos_per_row = np.clip(cos_per_row, -1.0, 1.0)
        return np.rad2deg(np.arccos(cos_per_row)).mean(axis=-1)

    # Preserve the historical C1 operation order exactly.
    if symmetry_label == "C1":
        return _axes_distance(M_previous)

    from relax.sampling.symmetry import relion_symmetry_operators

    left_operators, right_operators = relion_symmetry_operators(symmetry_label)
    minimum = np.full(M_current.shape[0], np.inf, dtype=np.float64)
    # Stream over operators: materialising N x |G| transformed matrices is
    # unnecessarily expensive for icosahedral real-data refinements.
    for left, right in zip(left_operators, right_operators, strict=True):
        transformed = np.matmul(np.matmul(left[None, :, :], M_previous), right[None, :, :])
        minimum = np.minimum(minimum, _axes_distance(transformed))
    return minimum


def compute_relion_orientation_changes(
    current_rotations: Optional[np.ndarray],
    previous_rotations: Optional[np.ndarray],
    *,
    symmetry_label: str = "C1",
) -> float:
    """Mean RELION-style angular distance across all particles, in degrees.

    Returns ``+inf`` when either input is None or shapes don't match,
    matching the convention used elsewhere in this module for "no prior
    iteration to compare against".

    Implements the aggregate part of ``monitorHiddenVariableChanges`` /
    ``updateOverallChangesInHiddenVariables`` (ml_optimiser.cpp:9230 and
    9251)::

        sum_changes_optimal_orientations += angular_distance(...)
        ...
        current_changes_optimal_orientations =
            sum_changes_optimal_orientations / sum_changes_count
    """
    if current_rotations is None or previous_rotations is None:
        return float("inf")
    current_rotations = np.asarray(current_rotations)
    previous_rotations = np.asarray(previous_rotations)
    if current_rotations.shape != previous_rotations.shape:
        return float("inf")
    if current_rotations.size == 0:
        return 0.0
    per_particle = _relion_angular_distance_per_particle(
        current_rotations,
        previous_rotations,
        symmetry_label=symmetry_label,
    )
    return float(np.mean(per_particle))


def compute_relion_offset_changes_angstrom(
    current_translations_pixel: Optional[np.ndarray],
    previous_translations_pixel: Optional[np.ndarray],
    voxel_size: float,
) -> float:
    """RELION-style per-particle offset change in **angstroms**.

    Implements ml_optimiser.cpp:9232 + 9252::

        sum_changes_optimal_offsets +=
            (xoff-old_xoff)^2 + (yoff-old_yoff)^2 + (zoff-old_zoff)^2
        ...
        current_changes_optimal_offsets =
            sqrt(sum_changes_optimal_offsets / (2 * sum_changes_count))

    where xoff/yoff are in **angstroms** (RELION multiplies the metadata
    pixel offsets by ``my_pixel_size`` at line 9222-9225). The factor 2 in
    the denominator is RELION's hard-coded per-dim averaging convention,
    NOT the data dimensionality (it stays 2 even with z != 0). The result
    has units of **angstroms** because the inputs are in angstroms.

    Parameters
    ----------
    current_translations_pixel, previous_translations_pixel : np.ndarray
        Per-particle translations in PIXEL units, shape ``(N, 2)``.
    voxel_size : float
        Pixel size in angstroms (e.g. 4.25 for the 5k benchmark).
    """
    if current_translations_pixel is None or previous_translations_pixel is None:
        return float("inf")
    current_translations_pixel = np.asarray(current_translations_pixel)
    previous_translations_pixel = np.asarray(previous_translations_pixel)
    if current_translations_pixel.shape != previous_translations_pixel.shape:
        return float("inf")
    if current_translations_pixel.size == 0:
        return 0.0
    diffs_pixel = current_translations_pixel - previous_translations_pixel
    diffs_ang = diffs_pixel * float(voxel_size)
    sum_sq_per_particle = np.sum(diffs_ang**2, axis=-1)  # (N,)
    n = sum_sq_per_particle.shape[0]
    return float(np.sqrt(sum_sq_per_particle.sum() / (2.0 * n)))

