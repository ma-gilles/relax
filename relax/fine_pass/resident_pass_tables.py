"""Host tables both resident pass-2 drivers derive from the same inputs.

The global driver (:mod:`relax.fine_pass.resident_pass2`) and the local-search driver
(:mod:`relax.local_search.resident_pass2`) build these the same way; each derivation has its owner here.
"""

from __future__ import annotations

import numpy as np

from relax.fine_pass import resident_stable_windows
from relax.fine_pass.wavg import make_relion_wavg_rectangle
from relax.fourier.half_spectrum import (
    make_relion_noise_shell_indices_half,
    mask_relion_noise_shell_indices_to_current_window,
)
from relax.projection.adjoint import mstep_adjoint_max_r
from relax.projection.projection import relion_scale_correction_pixel_mask
from relax.reconstruction.half_volume_mstep import relion_backprojector_volume_shape


def resident_scale_and_triplet(*, group_ids, accumulate_noise: bool) -> tuple[bool, bool]:
    """``(accumulate_scale, atomic_triplet)`` of a resident pass, for the global and the local driver.

    RELION runs the Wavg triplet whether or not it corrects scales: without --scale every particle's
    scale is 1 and only the XA/AA sums are skipped (acc_ml_optimiser_impl.h:4367-4372, 4907, 4980).
    The atomic triplet, and with it the direct low-shell residual, therefore runs whenever the pass
    accumulates noise, groups or not (scale 1, group -1, so no scale sums accumulate); the scale
    statistics themselves need real groups.
    """

    return group_ids is not None, bool(accumulate_noise)


def optics_group_rows(noise_variance_half, optics_group_ids, *, n_images: int) -> tuple[int, np.ndarray | None]:
    """``(n_optics_groups, optics_groups_np)``: the noise table's group count and each image's row of it.

    A ``[G, P]`` noise table (G > 1 optics groups) needs ``optics_group_ids``, one row in ``[0, G)`` per
    image; a single noise spectrum has no rows (None).
    """

    n_optics_groups = 1 if noise_variance_half.ndim == 1 else int(noise_variance_half.shape[0])
    optics_groups_np = None
    if n_optics_groups > 1:
        if optics_group_ids is None:
            raise ValueError("a per-optics-group noise table needs optics_group_ids")
        optics_groups_np = np.asarray(optics_group_ids, dtype=np.int32).reshape(-1)
        if optics_groups_np.shape != (n_images,) or np.any(optics_groups_np < 0) or np.any(
            optics_groups_np >= n_optics_groups
        ):
            raise ValueError(
                f"optics_group_ids must give each of {n_images} images a row of the "
                f"{n_optics_groups}-group noise table"
            )
    return n_optics_groups, optics_groups_np


def resident_noise_tables(
    noise_variance_half,
    *,
    window_spec,
    window_indices,
    window_indices_np,
    recon_window_indices,
    n_recon_windowed,
    stable_window_plan,
    image_shape,
    current_size,
    wsum_current_size,
    mstep_current_size,
) -> tuple:
    """``(shell_indices_half, shell_indices_noise, noise_variance_for_noise, relion_wavg_rectangle,
    logical_rect_pixels, logical_recon_pixels)``: the noise tables of a resident pass.

    ``shell_indices_half`` is RELION's noise shell of each half-spectrum pixel, masked to the noise crop; the
    rest are :func:`resident_stable_windows.stable_noise_tables` over the reconstruction window.

    The crop mask sees RELION's logical window, not the physical class; with a weighted-sum size
    above the scoring size it is that size's crop rectangle (the Wavg rectangle's).
    """

    noise_crop_indices = resident_stable_windows.logical_score_indices(
        window_indices, window_indices_np, stable_window_plan
    )
    if wsum_current_size != current_size:
        noise_crop_indices = make_relion_wavg_rectangle(
            image_shape,
            wsum_current_size,
            recon_window_indices,
            reconstruction_current_size=mstep_current_size,
        ).centered_indices
    shell_indices_half = mask_relion_noise_shell_indices_to_current_window(
        make_relion_noise_shell_indices_half(image_shape),
        image_shape,
        wsum_current_size,
        noise_crop_indices,
    )
    return (
        shell_indices_half,
        *resident_stable_windows.stable_noise_tables(
            window_spec.recon_values(shell_indices_half),
            window_spec.recon_values(noise_variance_half),
            plan=stable_window_plan,
            image_shape=image_shape,
            wsum_current_size=wsum_current_size,
            recon_window_indices=recon_window_indices,
            mstep_current_size=mstep_current_size,
            n_recon_windowed=n_recon_windowed,
        ),
    )


def scale_pixel_masks_rect(
    scale_correction_data_vs_prior, *, n_classes: int, shell_indices_noise, n_shells: int, relion_wavg_rectangle
) -> np.ndarray:
    """Each class's scale-sum pixel mask over the Wavg rectangle, ``[n_classes, n_rect]`` bool.

    RELION masks each class's scale sums with its own data_vs_prior_class[iclass] > 3
    (acc_ml_optimiser_impl.h:4893-4912): ``[K, n_shells]`` curves, or one shell vector for every class.
    """

    n_rect = int(relion_wavg_rectangle.centered_indices.size)
    scale_dvp_by_class = [scale_correction_data_vs_prior] * n_classes
    if scale_correction_data_vs_prior is not None and n_classes > 1:
        scale_dvp_array = np.asarray(scale_correction_data_vs_prior)
        if scale_dvp_array.ndim == 2:
            if scale_dvp_array.shape[0] != n_classes:
                raise ValueError(
                    "scale_correction_data_vs_prior must be one shell vector or have "
                    f"shape ({n_classes}, n_shells), got {scale_dvp_array.shape}"
                )
            scale_dvp_by_class = [scale_dvp_array[k] for k in range(n_classes)]
    masks = np.zeros((n_classes, n_rect), dtype=bool)
    for class_index, class_dvp in enumerate(scale_dvp_by_class):
        masks[class_index, relion_wavg_rectangle.exact_positions] = np.asarray(
            relion_scale_correction_pixel_mask(class_dvp, shell_indices_noise, n_shells=n_shells),
            dtype=bool,
        )
    return masks


def reconstruction_volume(
    volume_shape,
    *,
    mstep_current_size,
    reconstruction_volume_current_size,
    reconstruction_padding_factor,
    reconstruction_image_radius,
    anisotropic_magnification: bool,
) -> tuple:
    """``(volume_current_size, recon_volume_shape, mstep_max_r)``: RELION's x-half BPref of the pass.

    The backprojector's model size (``reconstruction_volume_current_size``, else the M-step's), its
    padded volume shape and the M-step adjoint radius; the stable-window physical cube derives from these
    (:func:`resident_stable_windows.physical_reconstruction_volume`).
    """

    volume_current_size = (
        mstep_current_size
        if reconstruction_volume_current_size is None
        else int(reconstruction_volume_current_size)
    )
    recon_volume_shape = relion_backprojector_volume_shape(
        volume_shape,
        reconstruction_padding_factor,
        current_size=volume_current_size,
    )
    mstep_max_r = mstep_adjoint_max_r(
        volume_current_size,
        reconstruction_image_radius,
        reconstruction_padding_factor,
        anisotropic_magnification=anisotropic_magnification,
    )
    return volume_current_size, recon_volume_shape, mstep_max_r
