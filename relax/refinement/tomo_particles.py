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


def relion_gpu_old_offsets(offsets_px):
    """The old offsets RELION's GPU path adds to a subtomogram's trial shifts: rounded to whole pixels.

    ``my_old_offset.selfROUND()`` (acc_ml_optimiser_impl.h:659, stored as ``op.old_offset`` at :675)
    applies to tomo particles too, although their images are not pre-shifted (:873), so the
    fractional part of the offset is dropped from the E-step (the CPU path keeps it,
    ml_optimiser.cpp:7259). ``ROUND`` rounds half away from zero (macros.h:197).
    """

    offsets_px = np.asarray(offsets_px, dtype=np.float64)
    return np.where(offsets_px > 0, np.trunc(offsets_px + 0.5), np.trunc(offsets_px - 0.5))


def relion_offset_log_prior_3d(translations_angst, old_offsets_px, *, pixel_size, sigma_offset_angst):
    """Each particle's log offset prior over the coarse 3D grid, ``[P, T]`` float32 (RELION's ``pdf_offset``).

    The GPU coarse pass adds the rounded old offset in pixels (:func:`relion_gpu_old_offsets`) to
    the sampling translation in Angstrom, subtracts the prior (zero in auto-refine), and scales the
    squared distance by ``pixel_size**2 / (-2 sigma2_offset)`` (acc_ml_optimiser_impl.h:3059-3094).
    The SPA prior has the same unit mix, which :func:`make_relion_translation_log_prior` encodes
    with pixel-unit translations and the centre ``-rounded_old / pixel_size``. Checked against a
    RELION f2c1a3 dump of a tilt-series particle to 2e-6 (em_work/cryoet_s42_20260925).
    """

    from relax.helpers.orientation_priors import make_relion_translation_log_prior

    pixel_size = float(pixel_size)
    centers = -relion_gpu_old_offsets(np.asarray(old_offsets_px, dtype=np.float64).reshape(-1, 3)) / pixel_size
    return make_relion_translation_log_prior(
        np.asarray(translations_angst, dtype=np.float64) / pixel_size,
        pixel_size,
        float(sigma_offset_angst),
        centers,
        dtype=np.float32,
    ).reshape(centers.shape[0], -1)


def tilt_translation_angles(shifts_3d, old_offsets_3d, image_projections, image_particle, image_size):
    """Each image's scoring phase operand for every 3D trial shift: float32 ``[I, T, 2]`` radians.

    RELION's GPU path adds the particle's old offset (:func:`relion_gpu_old_offsets`, rounded)
    to the trial shift, projects it with the image's ``Aproj`` and stores
    ``-2 pi shift / image_full_size`` as float (acc_ml_optimiser_impl.h:1761-1787;
    :func:`tilt_image_shifts`). Shifts and offsets are in pixels of the image's optics group;
    ``image_size`` is its full image size (a scalar, or one value per image).
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


def relion_left_matrices(image_left, *, accuracy: float = 1e-6):
    """Each image's generateEulerMatrices ``L``, and whether RELION applies it.

    RELION passes ``L`` (the image's ``Aproj`` times the optics scale) only when it is not the
    identity to within ``XMIPP_EQUAL_ACCURACY`` (``Matrix2D::isIdentity``, matrix2d.h:1191-1206;
    acc_ml_optimiser_impl.h:1614-1618); an identity image keeps the SPA matrices, whose inverse
    is the transpose. Returns ``(left [I, 3, 3] float64, applies [I] bool)``.
    """

    image_left = np.asarray(image_left, dtype=np.float64)
    applies = np.any(np.abs(image_left - np.eye(3)) > accuracy, axis=(1, 2))
    return image_left, applies
