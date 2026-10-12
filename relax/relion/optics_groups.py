"""RELION optics groups: a group on another pixel size and box than the reference, and per-group noise spectra.

**Another grid (RELION S3b rules).**

The reference lives on optics group 0's grid (``ori_size`` pixels at ``angpix_ref``).
A group with ``box_g`` pixels at ``angpix_g`` has the scale
``s_g = box_g angpix_g / (ori_size angpix_ref)``:

- projection and backprojection matrices are multiplied by ``s_g``
  (``ObservationModel::applyScaleDifference``, ``obs_model.cpp:1332-1339``); the
  projector uses the inverse, so a relax rotation is divided by ``s_g``;
- the group's image sizes follow ``updateImageSizeAndResolutionPointers``
  (``ml_optimiser.cpp:5741-5777``); a group with ``s_g > 1`` gets a Fourier window
  wider than the model sphere, whose outer rows RELION's accelerated kernels project
  as zero (:func:`relax.projection.projection.relion_kernel_zero_rows`);
- ``sigma2_noise`` stays on the reference shells: the E-step reads reference shell
  ``round(ires / s_g)`` for group shell ``ires`` (``:6840``), the M-step adds group shell
  ``i`` into reference shell ``round(i / s_g)`` (``:9100-9107``) and an empty shell takes
  the previous shell's value (``:5282-5285``).

**Per-group noise spectra: table rows per image and sums per group.** RELION keeps one ``sigma2_noise``
spectrum per optics group (``MlModel::sigma2_noise[optics_group]``). Every image scores and backprojects
with its own group's spectrum, and adds its noise terms to that group's sums
(``wsum_model.sigma2_noise[optics_group]``, ``sumw_group[optics_group]``). A noise operand is either one
shared spectrum ``[P]`` (one optics group) or a table ``[G, P]``; ``image_optics_groups`` holds each
dataset-local image's row (``0 .. G-1``). With one group every helper here returns its input unchanged, so
single-group callers run exactly the arithmetic they ran before.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np


def scale_difference(box_size, pixel_size, model_box_size, ref_pixel_size) -> float:
    """``s_g``: the factor RELION's applyScaleDifference multiplies a matrix by."""

    return float(box_size) * float(pixel_size) / (float(model_box_size) * float(ref_pixel_size))


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
    coarse paths zero the rows (:func:`relax.projection.projection.relion_kernel_zero_rows`).
    """

    half = int(window) // 2
    return half > int(r_max) and half < float(scale) * math.sqrt(int(r_max) ** 2 + 1)


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


def noise_rows(noise, image_optics_groups, image_indices):
    """The noise spectrum of each image in ``image_indices``.

    ``noise`` is ``[P]`` (returned unchanged) or a ``[G, P]`` table, gathered to
    ``[B, P]`` with ``image_optics_groups[image_indices]``.
    """

    if jnp.ndim(noise) == 1:
        return noise
    if image_optics_groups is None:
        raise ValueError("a per-optics-group noise table needs each image's optics group")
    groups = np.asarray(image_optics_groups, dtype=np.int32)[np.asarray(image_indices, dtype=np.int64)]
    return jnp.asarray(noise)[jnp.asarray(groups)]


def pixel_rows(values):
    """``[P] -> [1, P]`` so it broadcasts over images; ``[B, P]`` rows unchanged."""

    return values[None, :] if values.ndim == 1 else values


def dense_optics_groups(optics_group_labels):
    """RELION optics-group labels to dense row indices ``0 .. G-1`` and ``G``."""

    labels = np.asarray(optics_group_labels, dtype=np.int64).reshape(-1)
    unique, dense = np.unique(labels, return_inverse=True)
    return dense.astype(np.int32), int(unique.size)
