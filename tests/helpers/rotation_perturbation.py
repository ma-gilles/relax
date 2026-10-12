"""RELION's sampling perturbation applied to rotation matrices, for building perturbed test grids.

Moved from relax/sampling/perturbation.py, where no relax module used it: relax perturbs its grids in Euler
space (``apply_relion_rotation_perturbation_to_eulers``). Tests build perturbed matrix grids with it, and
``tests/unit/test_relion_bind/test_s1_orientations.py`` checks it against RELION.
"""

import numpy as np
from recovar import utils


def apply_relion_rotation_perturbation(rotations, random_perturbation, angular_sampling_deg):
    """Port of RELION's 3D grid perturbation (healpix_sampling.cpp:1909-1934).

    RELION computes ``A_perturbed = A @ Euler_angles2matrix(p, p, p)`` where
    ``p = random_perturbation * angular_sampling``. Since recovar's grid uses
    ``R_from_relion(theta, phi, psi) = Euler_angles2matrix(phi, theta, psi)``
    and ``R_from_relion(p, p, p) = Euler_angles2matrix(p, p, p)`` (symmetric),
    the correct implementation is right-multiply by ``R_from_relion(p, p, p)``.

    Parameters
    ----------
    rotations : np.ndarray, shape (N, 3, 3)
    random_perturbation : float
        Current iteration's value in [-pf, +pf].
    angular_sampling_deg : float
        The nominal step at the coarse healpix order (pre-oversampling) in degrees.
        Use ``relion_angular_sampling_deg(healpix_order, adaptive_oversampling=0)``.

    Returns
    -------
    np.ndarray, shape (N, 3, 3)
    """
    if abs(random_perturbation) < 1e-12:
        return rotations
    myperturb = float(random_perturbation) * float(angular_sampling_deg)
    R_perturb = utils.R_from_relion(np.array([[myperturb, myperturb, myperturb]], dtype=np.float64))[0]
    R_perturb = R_perturb.astype(rotations.dtype)
    # RELION: A = A * R  (right multiply, each matrix independently)
    return np.einsum("nij,jk->nik", rotations, R_perturb)
