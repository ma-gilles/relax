"""Set-up checks of a refinement run: what refine_single_volume refuses or resolves from its inputs before the
first iteration (code rule 12: check at the edge, then trust).
"""

import numpy as np


def _relion_k1_translation_angle_scale(
    *,
    n_classes: int,
    model_pixel_size: float,
    optics_pixel_sizes,
) -> float:
    """Convert K=1 model-pixel translations to the shared optics pixel size.

    RELION keeps sampling translations in Angstrom and converts them with the
    particle's optics pixel size (HealpixSampling::getTranslationsInPixel),
    while RECOVAR's translation grid is in model pixels. The scale
    model_pixel_size / optics_pixel_size is exactly 1.0 when the serialized
    sizes agree; it multiplies only the RELION translation-phase operand.
    """

    if int(n_classes) != 1 or optics_pixel_sizes is None:
        return 1.0
    optics = np.asarray(optics_pixel_sizes, dtype=np.float64).reshape(-1)
    if optics.size == 0 or not np.all(np.isfinite(optics)) or np.any(optics <= 0.0):
        raise ValueError("RELION optics pixel sizes must be non-empty, positive, and finite")
    unique_optics = np.unique(optics)
    if unique_optics.size != 1:
        raise NotImplementedError(
            "K=1 exact RELION translation phases currently require one shared optics pixel size; "
            "per-particle optics scaling is not yet implemented"
        )
    # A Python float: the model pixel size may arrive as the input's float32, and the scale is float64 arithmetic.
    return float(model_pixel_size) / float(unique_optics[0])


def _optics_group_ids_per_half(optics_group_ids_per_half, noise_variance_per_half, experiment_datasets):
    """Each half's per-image optics-group rows, or ``[None, None]`` for one group.

    A half's noise is a flat vector (one optics group) or ``[G, P]`` rows
    (:mod:`relax.helpers.optics_noise`); with rows every image needs its group.
    """

    n_groups = 1 if noise_variance_per_half[0].ndim == 1 else int(noise_variance_per_half[0].shape[0])
    if n_groups == 1:
        return [None, None]
    if optics_group_ids_per_half is None or len(optics_group_ids_per_half) != 2:
        raise ValueError(
            f"a {n_groups}-optics-group noise table needs parity.optics_group_ids_per_half for both halves"
        )
    ids = []
    for half, (values, dataset) in enumerate(zip(optics_group_ids_per_half, experiment_datasets)):
        values = np.asarray(values, dtype=np.int32).reshape(-1)
        if values.shape != (int(dataset.n_units),) or np.any(values < 0) or np.any(values >= n_groups):
            raise ValueError(
                f"half {half + 1} optics-group ids must give each of {int(dataset.n_units)} images "
                f"a row 0..{n_groups - 1}"
            )
        ids.append(values)
    return ids


def _internal_solvent_mask(path, box_size, pixel_size_angstrom):
    """RELION --solvent_mask on the model grid in relax's internal (z, y, x) frame, or None.

    Map files hold RELION's axis order, the transpose of the internal frame (relax.helpers.map_io);
    a mask has no sign to undo.
    """
    if path is None:
        return None
    from relax.reconstruction.solvent_mask import read_solvent_mask

    mask = read_solvent_mask(path, box=int(box_size), pixel_size=float(pixel_size_angstrom))
    return np.ascontiguousarray(np.transpose(mask, (2, 1, 0)))
