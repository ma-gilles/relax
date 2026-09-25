"""Subtomogram particles as groups of tilt images (RELION 5 2D-stack tomography, S4).

A particle is its visible tilt images. They share one pose hypothesis (rotation ``R``,
3D shift ``t``), one class and one half set; every image has its own projection matrix
``Aproj_i`` (tilt-series projection times the subtomogram orientation), CTF with dose
and optics group. Line numbers refer to RELION's upstream ``src/ml_optimiser.cpp``,
``src/exp_model.cpp`` and ``src/acc/acc_ml_optimiser_impl.h`` (the GPU path, which
``relion_refine --gpu`` runs and relax follows where the two paths differ).

- Image matrix: ``Aproj_i R`` (ml_optimiser.cpp:7266, acc_ml_optimiser_impl.h:1037).
- Image shift: ``Aproj_i[:2] (t + old_offset)``; the old offset is not applied to tomo
  images and is added to the trial shift instead (ml_optimiser.cpp:6115, 7361-7366;
  ``Experiment::getTranslationInTiltSeries``, exp_model.cpp:106-114).
- Particle score: the images' diff2 summed per hypothesis (ml_optimiser.cpp:7650-7661).
- M-step (GPU path, acc_ml_optimiser_impl.h storeWeightedSums): pose, class and offset
  sums once per particle (:4139-4145); noise, scale and norm sums per image divided by
  the particle's image count (:4903-4905, :4923-4924, :4945-4949); every image
  backprojected with the particle's weights (image loop from :4268).

See PLAN.md "S4 design" in the cryo-ET coordination directory.
"""

from __future__ import annotations

import numpy as np


def image_particle_counts(image_particle, n_particles: int) -> np.ndarray:
    """Number of images of each particle."""

    return np.bincount(np.asarray(image_particle, dtype=np.int64), minlength=int(n_particles))


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


def tilt_image_shifts(shifts_3d, old_offsets_3d, image_projections, image_particle):
    """2D shift of every image for every 3D trial shift: ``[I, T, 2]``.

    ``Aproj_i[:2] (t + old_offset_p)`` with ``p`` the image's particle
    (``getTranslationInTiltSeries``, exp_model.cpp:106-114; the old offset joins the
    trial shift for tomo images, ml_optimiser.cpp:7361-7366).
    """

    shifts_3d = np.asarray(shifts_3d, dtype=np.float64)
    old = np.asarray(old_offsets_3d, dtype=np.float64)[np.asarray(image_particle, dtype=np.int64)]
    total = shifts_3d[None, :, :] + old[:, None, :]
    aproj = np.asarray(image_projections, dtype=np.float64)[:, None, :, :]
    # RELION's summation order, a(r,0) x + a(r,1) y + a(r,2) z (exp_model.cpp:111-112).
    return np.stack(
        [
            aproj[..., r, 0] * total[..., 0] + aproj[..., r, 1] * total[..., 1] + aproj[..., r, 2] * total[..., 2]
            for r in (0, 1)
        ],
        axis=-1,
    )


def tilt_translation_angles(shifts_3d, old_offsets_3d, image_projections, image_particle, image_size):
    """Each image's scoring phase operand for every 3D trial shift: float32 ``[I, T, 2]`` radians.

    RELION's GPU path adds the particle's old offset to the trial shift, projects it with
    the image's ``Aproj`` and stores ``-2 pi shift / image_full_size`` as float
    (acc_ml_optimiser_impl.h:1761-1787; :func:`tilt_image_shifts`). Shifts and offsets
    are in pixels of the image's optics group; ``image_size`` is its full image size
    (a scalar, or one value per image).
    """

    shifts = tilt_image_shifts(shifts_3d, old_offsets_3d, image_projections, image_particle)
    size = np.broadcast_to(np.asarray(image_size, dtype=np.float64), (shifts.shape[0],))
    return np.asarray(-2.0 * np.pi * shifts / size[:, None, None], dtype=np.float32)


def image_slot_ids(row_unit, unit_image_offsets, slot: int) -> np.ndarray:
    """Image of each hypothesis row for one image slot, ``-1`` where the row's particle has fewer images.

    The scorer visits a particle's images in slot order ``0..n_images - 1`` (RELION's
    ``img_id`` loop, acc_ml_optimiser_impl.h:1737 and :2282) and adds each image's diff2
    to the row's running sum. ``unit_image_offsets`` is the CSR ``[n_units + 1]`` of the
    units' image rows (:class:`relax.relion.tomo_input.TomoParticleIndex.image_offsets`).
    """

    row_unit = np.asarray(row_unit, dtype=np.int64)
    offsets = np.asarray(unit_image_offsets, dtype=np.int64)
    start = offsets[row_unit]
    count = offsets[row_unit + 1] - start
    return np.where(int(slot) < count, start + int(slot), -1).astype(np.int32)


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


def image_noise_scale(image_particle, n_particles: int) -> np.ndarray:
    """Factor on each image's noise, scale and norm sums: ``1 / n_images`` of its particle.

    The GPU path averages these sums over the particle's images
    (acc_ml_optimiser_impl.h:4903-4905, 4923-4924, 4945-4949); pose, class and offset
    sums stay once per particle.
    """

    image_particle = np.asarray(image_particle, dtype=np.int64)
    counts = image_particle_counts(image_particle, n_particles)
    return 1.0 / counts[image_particle].astype(np.float64)
