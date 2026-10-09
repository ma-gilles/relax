"""Subtomogram particles as groups of tilt images (RELION 5 2D-stack tomography).

A particle is its visible tilt images. They share one pose hypothesis (rotation ``R``,
3D shift ``t``), one class and one half set; every image has its own projection matrix
``Aproj_i`` (tilt-series projection times the subtomogram orientation), CTF with dose
and optics group. Line numbers refer to RELION f2c1a38 (``git show HEAD:src/<file>``) ``src/ml_optimiser.cpp``,
``src/exp_model.cpp`` and ``src/acc/acc_ml_optimiser_impl.h`` (the GPU path, which
``relion_refine --gpu`` runs and relax follows where the two paths differ).

- Image matrix: ``Aproj_i R`` (ml_optimiser.cpp:7266, acc_ml_optimiser_impl.h:573).
- Image shift: ``Aproj_i[:2] (t + old_offset)``; the old offset is not applied to tomo
  images and is added to the trial shift instead (ml_optimiser.cpp:6115, 7361-7366;
  ``Experiment::getTranslationInTiltSeries``, exp_model.cpp:106-114).
- Particle score: the images' diff2 summed per hypothesis (ml_optimiser.cpp:7650-7661).
- M-step (GPU path, acc_ml_optimiser_impl.h storeWeightedSums): pose, class and offset
  sums once per particle (:2840-2846); noise and norm sums per image divided by the
  particle's image count (:3490-3491, :3512-3516), scale sums added as they are (:3474-3479);
  every image backprojected with the particle's weights (image loop from :2959).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Phase values (images x trial shifts) above which the per-image phases are computed in blocks on host threads.
_THREADED_PHASE_VALUES = 1 << 20


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

    ``my_old_offset.selfROUND()`` (acc_ml_optimiser_impl.h:216, stored as ``op.old_offset`` at :232)
    applies to tomo particles too, although their images are not pre-shifted (:430), so the
    fractional part of the offset is dropped from the E-step (the CPU path keeps it,
    ml_optimiser.cpp:6085). ``ROUND`` rounds half away from zero (macros.h:197).
    """

    offsets_px = np.asarray(offsets_px, dtype=np.float64)
    return np.where(offsets_px > 0, np.trunc(offsets_px + 0.5), np.trunc(offsets_px - 0.5))


def relion_offset_log_prior_3d(translations_angst, old_offsets_px, *, pixel_size, sigma_offset_angst):
    """Each particle's log offset prior over the coarse 3D grid, ``[P, T]`` float32 (RELION's ``pdf_offset``).

    The GPU coarse pass adds the rounded old offset in pixels (:func:`relion_gpu_old_offsets`) to
    the sampling translation in Angstrom, subtracts the prior (zero in auto-refine), and scales the
    squared distance by ``pixel_size**2 / (-2 sigma2_offset)`` (acc_ml_optimiser_impl.h:2135-2170).
    The SPA prior has the same unit mix, which :func:`make_relion_translation_log_prior` encodes
    with pixel-unit translations and the centre ``-rounded_old / pixel_size``. Checked against a
    RELION f2c1a3 dump of a tilt-series particle to 2e-6 (em_work/cryoet_s42_20260925).
    """

    from relax.helpers.orientation_priors import make_relion_translation_log_prior

    centers = -relion_gpu_old_offsets(np.asarray(old_offsets_px, dtype=np.float64).reshape(-1, 3)) / pixel_size
    return make_relion_translation_log_prior(
        np.asarray(translations_angst, dtype=np.float64) / pixel_size,
        pixel_size,
        float(sigma_offset_angst),
        centers,
        dtype=np.float32,
    ).reshape(centers.shape[0], -1)


def tilt_translation_angles(shifts_3d, old_offsets_3d, image_projections, image_particle, box_size):
    """Each image's scoring phase operand for every 3D trial shift: float32 ``[I, T, 2]`` radians.

    RELION's GPU path adds the particle's old offset (:func:`relion_gpu_old_offsets`, rounded)
    to the trial shift, projects it with the image's ``Aproj`` and stores
    ``-2 pi shift / image_full_size`` as float (acc_ml_optimiser_impl.h:1214-1240;
    :func:`tilt_image_shifts`). Shifts and offsets are in pixels of the image's optics group;
    ``box_size`` is its image box (a scalar, or one value per image).
    """

    image_particle = np.asarray(image_particle, dtype=np.int64)
    image_projections = np.asarray(image_projections, dtype=np.float64)
    n_images = int(image_particle.shape[0])
    size = np.broadcast_to(np.asarray(box_size, dtype=np.float64), (n_images,))

    def phases(rows):
        shifts = tilt_image_shifts(shifts_3d, old_offsets_3d, image_projections[rows], image_particle[rows])
        return np.asarray(-2.0 * np.pi * shifts / size[rows, None, None], dtype=np.float32)

    n_shifts = int(np.shape(shifts_3d)[0])
    if n_images * n_shifts < _THREADED_PHASE_VALUES:
        return phases(slice(0, n_images))
    # Every image's phases are its own arithmetic, so blocks of images run on host threads (NumPy
    # releases the GIL inside each operation) and fill one float32 array: the same values as one
    # call, without its float64 [I, T, 2] temporaries. One fine-grid call was 1.8 s of each late
    # et09 subtomogram VDAM iteration (39,000 tilt images; py-spy, job 14993731).
    out = np.empty((n_images, n_shifts, 2), dtype=np.float32)
    block = max(1, _THREADED_PHASE_VALUES // max(n_shifts, 1))

    def fill(start):
        rows = slice(start, min(start + block, n_images))
        out[rows] = phases(rows)

    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        list(pool.map(fill, range(0, n_images, block)))
    return out


def image_slot_ids(row_unit, unit_image_offsets, slot: int) -> np.ndarray:
    """Image of each hypothesis row for one image slot, ``-1`` where the row's particle has fewer images.

    The scorer visits a particle's images in slot order ``0..n_images - 1`` (RELION's
    ``img_id`` loop, acc_ml_optimiser_impl.h:1190 and :1490) and adds each image's diff2
    to the row's running sum. ``unit_image_offsets`` is the CSR ``[n_units + 1]`` of the
    units' image rows (:class:`relax.relion.tomo_input.TomoParticleIndex.image_offsets`).
    """

    row_unit = np.asarray(row_unit, dtype=np.int64)
    offsets = np.asarray(unit_image_offsets, dtype=np.int64)
    start = offsets[row_unit]
    count = offsets[row_unit + 1] - start
    return np.where(int(slot) < count, start + int(slot), -1).astype(np.int32)


def image_noise_scale(image_particle, n_particles: int) -> np.ndarray:
    """Factor on each image's noise and norm sums: ``1 / n_images`` of its particle.

    The GPU path averages these sums over the particle's images
    (acc_ml_optimiser_impl.h:3490-3491, 3512-3516); the scale sums (:3474-3479) and the pose,
    class and offset sums are not divided.
    """

    image_particle = np.asarray(image_particle, dtype=np.int64)
    counts = np.bincount(image_particle, minlength=n_particles)
    return 1.0 / counts[image_particle].astype(np.float64)


def relion_left_matrices(image_left, *, accuracy: float = 1e-6):
    """Each image's generateEulerMatrices ``L``, and whether RELION applies it.

    RELION passes ``L`` (the image's ``Aproj`` times the optics scale) only when it is not the
    identity to within ``XMIPP_EQUAL_ACCURACY`` (``Matrix2D::isIdentity``, matrix2d.h:1191-1206;
    acc_ml_optimiser_impl.h:1100-1104); an identity image keeps the SPA matrices, whose inverse
    is the transpose. Returns ``(left [I, 3, 3] float64, applies [I] bool)``.
    """

    image_left = np.asarray(image_left, dtype=np.float64)
    applies = np.any(np.abs(image_left - np.eye(3)) > accuracy, axis=(1, 2))
    return image_left, applies
