"""Test reference: tilt-image geometry and per-particle score/weight maps of RELION tomography.

Moved from relax/refinement/tomo_particles.py (PLAN e1): no relax module uses them, only tests.
"""

import numpy as np


def tilt_projection_matrices(image_matrices, particle_matrices, image_particle):
    """Each tilt image's projection ``Aproj_i`` from its full matrix and its particle's pose.

    The flattened per-tilt STAR (recovar's RELION 5 converter) carries each image's matrix
    ``A_i = Aproj_i A_p`` (RELION convention, ``Euler_angles2matrix``) for the particle pose
    ``A_p`` it was written with, so ``Aproj_i = A_i A_p^T``. RELION builds ``Aproj_i`` from the
    tilt series' projection matrix times the subtomogram orientation
    (``Experiment::read``, exp_model.cpp:1003-1026).
    """

    image_matrices = np.asarray(image_matrices, dtype=np.float64)
    particle_matrices = np.asarray(particle_matrices, dtype=np.float64)
    return np.einsum("iab,icb->iac", image_matrices, particle_matrices[np.asarray(image_particle, dtype=np.int64)])


def tilt_image_rotations(pose_rotations, image_projections):
    """``Aproj_i R_h`` for every image ``i`` and pose hypothesis ``h``: ``[I, H, 3, 3]``.

    RELION multiplies the pose matrix on the left by the image's projection matrix
    before anisotropic magnification and the scale difference
    (``A = getRotationMatrix(part_id, img_id) * A``, ml_optimiser.cpp:7266).
    """

    pose_rotations = np.asarray(pose_rotations)
    image_projections = np.asarray(image_projections)
    return np.einsum("iab,hbc->ihac", image_projections, pose_rotations)


def particle_scores(image_scores, image_particle, n_particles: int):
    """Per-particle scores: each hypothesis's image scores summed over the particle's images.

    ``image_scores`` is ``[I, H]`` (diff2 or log-likelihood, one row per image); the
    result is ``[P, H]`` (``exp_Mweight[ihidden] += diff2`` over ``img_id``,
    ml_optimiser.cpp:7650-7661).
    """

    image_scores = np.asarray(image_scores)
    out = np.zeros((int(n_particles),) + image_scores.shape[1:], dtype=image_scores.dtype)
    np.add.at(out, np.asarray(image_particle, dtype=np.int64), image_scores)
    return out


def image_weights(particle_weights, image_particle):
    """Each image's M-step weights: its particle's posterior weights, ``[I, H]``."""

    return np.asarray(particle_weights)[np.asarray(image_particle, dtype=np.int64)]
