"""Optics groups on another pixel size and box than the reference (RELION S3b rules).

The reference lives on optics group 0's grid (``ori_size`` pixels at ``angpix_ref``).
A group with ``box_g`` pixels at ``angpix_g`` has the scale
``s_g = box_g angpix_g / (ori_size angpix_ref)``:

- projection and backprojection matrices are multiplied by ``s_g``
  (``ObservationModel::applyScaleDifference``, ``obs_model.cpp:1332-1339``); the
  projector uses the inverse, so a relax rotation is divided by ``s_g``;
- the group's image sizes follow ``updateImageSizeAndResolutionPointers``
  (``ml_optimiser.cpp:5741-5777``); a group with ``s_g > 1`` gets a Fourier window
  wider than the model sphere, whose outer rows RELION's accelerated kernels project
  as zero (:func:`relax.helpers.projection.relion_kernel_zero_rows`);
- ``sigma2_noise`` stays on the reference shells: the E-step reads reference shell
  ``round(ires / s_g)`` for group shell ``ires`` (``:6840``), the M-step adds group shell
  ``i`` into reference shell ``round(i / s_g)`` (``:9100-9107``) and an empty shell takes
  the previous shell's value (``:5282-5285``).
"""

from __future__ import annotations

import math

import numpy as np


def scale_difference(box_size, pixel_size, ori_size, ref_pixel_size) -> float:
    """``s_g``: the factor RELION's applyScaleDifference multiplies a matrix by."""

    return float(box_size) * float(pixel_size) / (float(ori_size) * float(ref_pixel_size))


def group_current_size(current_size, box_size, scale) -> int:
    """``image_current_size[g] = min(box_g, 2 ceil(0.5 s_g current_size))``."""

    return int(min(int(box_size), 2 * math.ceil(0.5 * float(scale) * int(current_size))))


def coarse_rows_wrap_inside(window: int, r_max: int, scale: float) -> bool:
    """Whether RELION's coarse kernel projects rows beyond ``maxR`` inside the model sphere.

    The coarse diff2 kernel relabels FFTW row ``i > maxR`` as ``i - imgY`` for the projection and
    the image shift alike (acc/cuda/cuda_kernels/diff2.cuh:86-90, 163-164). The relabelled pixel
    lies outside the sphere of radius ``r_max`` exactly when ``window / 2`` exceeds
    ``s * r_max``, allowing for the integer-truncated radius test. A coarse window strictly
    between ``2 r_max`` and about ``2 s r_max`` makes RELION score those rows with nonzero
    references and relabelled phases. The fused coarse CUDA scorer reproduces that; the other
    coarse paths zero the rows (:func:`relax.helpers.projection.relion_kernel_zero_rows`).
    """

    half = int(window) // 2
    return half > int(r_max) and half < float(scale) * math.sqrt(int(r_max) ** 2 + 1)


def group_coarse_size(coarse_resolution_pixels, current_size_g, box_size, scale, max_coarse_size=None) -> int:
    """``image_coarse_size[g]`` for adaptive oversampling (``ml_optimiser.cpp:5761-5777``).

    ``coarse_resolution_pixels`` is ``pixel_size_ref * ori_size / coarse_resolution``, the
    reference-grid Fourier radius RELION's coarse pass needs; the group's size is that
    radius scaled by ``s_g``, capped by the scaled ``max_coarse_size`` and by the group's
    current size.
    """

    size = 2 * math.ceil(float(scale) * float(coarse_resolution_pixels))
    cap = int(box_size) if max_coarse_size is None or max_coarse_size <= 0 else int(float(scale) * max_coarse_size)
    return int(min(size, cap, int(current_size_g)))


def reference_shell_of_group_shell(n_group_shells, scale) -> np.ndarray:
    """Reference shell RELION reads for each group shell: ``round(i / s_g)`` (RELION ROUND)."""

    shells = np.arange(int(n_group_shells), dtype=np.float64) / float(scale)
    return np.floor(shells + 0.5).astype(np.int64)


def group_noise_from_reference(reference_sigma2, n_group_shells, scale) -> np.ndarray:
    """A group's shell spectrum read from the reference-grid ``sigma2_noise``.

    Shells whose remapped index falls past the reference spectrum are 0, as RELION
    skips them (``ires_remapped < XSIZE(sigma2_noise)``).
    """

    reference_sigma2 = np.asarray(reference_sigma2, dtype=np.float64)
    index = reference_shell_of_group_shell(n_group_shells, scale)
    out = np.zeros(int(n_group_shells), dtype=np.float64)
    inside = index < reference_sigma2.shape[-1]
    out[inside] = reference_sigma2[index[inside]]
    return out


def add_group_shells_to_reference(reference_sums, group_sums, scale) -> np.ndarray:
    """Add a group's per-shell sums into the reference shells, as the M-step does."""

    reference_sums = np.array(reference_sums, dtype=np.float64, copy=True)
    group_sums = np.asarray(group_sums, dtype=np.float64)
    index = reference_shell_of_group_shell(group_sums.shape[-1], scale)
    inside = index < reference_sums.shape[-1]
    np.add.at(reference_sums, index[inside], group_sums[inside])
    return reference_sums
