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
