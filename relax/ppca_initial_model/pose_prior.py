"""The input poses that centre the per-particle pose prior (``Config.pose_prior_sigma_deg``).

The prior itself is :mod:`relax.ppca_refinement.pose_prior`; section 19 of ``docs/math/vdam_ppca_algorithm.md``.
"""

import numpy as np

ANGLE_LABELS = ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")


def read_input_rotations(particles_star) -> np.ndarray:
    """Each particle's input rotation ``(N, 3, 3)`` float32, in the STAR's row order (the PPCA's particle ids).

    The matrix is RELION's ``Euler_angles2matrix`` of ``rlnAngleRot/Tilt/Psi``
    (:func:`relax.sampling.healpix.euler_angles_to_matrix`), the matrix of the PPCA's rotation grid
    (:func:`relax.sampling.grids.get_relion_hidden_rotation_grid` builds its matrices from its Euler angles with it).
    For subtomogram particles it is the pose in the particle's subtomogram frame: the tilt reader composes each
    tilt's ``Aproj`` with the subtomogram matrix ``rlnTomoSubtomogramRot/Tilt/Psi`` and then with the candidate
    rotation, RELION's ``Aproj A_subtomogram A_particle`` (:func:`relax.star.tomo_input.relion_image_geometry`).
    A STAR without the three columns, or with a non-finite angle, is refused.
    """
    from recovar.data_io.starfile import read_star, star_column

    from relax.sampling.healpix import euler_angles_to_matrix

    particles, _ = read_star(str(particles_star))
    missing = [label for label in ANGLE_LABELS if star_column(particles, label) is None]
    if missing:
        raise ValueError(f"The pose prior needs the input angles {', '.join(missing)} in {particles_star}")
    eulers = np.stack([np.asarray(star_column(particles, label), np.float64) for label in ANGLE_LABELS], axis=1)
    if not np.all(np.isfinite(eulers)):
        raise ValueError(f"{particles_star} has non-finite input angles")
    return euler_angles_to_matrix(eulers).astype(np.float32)
