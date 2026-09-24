"""Per-optics-group noise spectra: table rows per image and sums per group.

RELION keeps one ``sigma2_noise`` spectrum per optics group
(``MlModel::sigma2_noise[optics_group]``). Every image scores and backprojects
with its own group's spectrum, and adds its noise terms to that group's sums
(``wsum_model.sigma2_noise[optics_group]``, ``sumw_group[optics_group]``).

A noise operand is either one shared spectrum ``[P]`` (one optics group) or a
table ``[G, P]``; ``image_optics_groups`` holds each dataset-local image's row
(``0 .. G-1``). With one group every helper here returns its input unchanged, so
single-group callers run exactly the arithmetic they ran before.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


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
