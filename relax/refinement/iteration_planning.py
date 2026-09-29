"""Host-side plans for one standard refinement iteration."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from relax import sampling
from relax.dense.scoring_policy import _dense_global_scoring_dtype
from relax.diagnostics.relion_replay import _sealed_sampling_base_grids


@dataclass(frozen=True)
class InitialCoarseGrids:
    """Exhaustive coarse trial grid selected for refinement startup."""

    rotations: np.ndarray
    rotation_eulers: np.ndarray
    base_translations: np.ndarray
    translations: jnp.ndarray
    healpix_order: int


def build_initial_coarse_grids(
    healpix_order,
    translations,
    *,
    translation_range,
    translation_step,
    n_classes,
    voxel_size,
    symmetry="C1",
) -> InitialCoarseGrids:
    """Pair RELION's canonical initial rotations with its translation grid."""

    dtype = _dense_global_scoring_dtype()
    rotations, rotation_eulers = sampling._relion_rotation_grid_float32(
        healpix_order,
        dtype=dtype,
        **({"symmetry": symmetry} if symmetry != "C1" else {}),
    )
    if translations is None:
        translations = sampling._relion_base_translation_grid(
            translation_range,
            translation_step,
            n_classes=n_classes,
            voxel_size=voxel_size,
        )
    base_translations = np.asarray(translations, dtype=np.float64)
    return InitialCoarseGrids(
        rotations=rotations,
        rotation_eulers=rotation_eulers,
        base_translations=base_translations,
        translations=jnp.asarray(translations, dtype=dtype),
        healpix_order=int(healpix_order),
    )


def build_sealed_initial_coarse_grids(
    sealed_sampling_state,
    *,
    initialized_healpix_order,
    voxel_size,
    log: logging.Logger,
) -> InitialCoarseGrids:
    """Materialize and validate a schema-v3 sealed initial sampling grid."""

    rotations, rotation_eulers, current_translations = _sealed_sampling_base_grids(
        sealed_sampling_state,
        voxel_size_angstrom=voxel_size,
        dtype=_dense_global_scoring_dtype(),
    )
    healpix_order = int(sealed_sampling_state["healpix_order_original"])
    if healpix_order != int(initialized_healpix_order):
        raise ValueError(
            "sealed sampling HEALPix order does not match initialized boundary: "
            f"sealed={healpix_order} init={initialized_healpix_order}"
        )
    log.info(
        "Frozen-boundary v3 directly materialized %d Euler rows and %d translations",
        int(rotation_eulers.shape[0]),
        int(current_translations.shape[0]),
    )
    return InitialCoarseGrids(
        rotations=rotations,
        rotation_eulers=rotation_eulers,
        base_translations=np.asarray(current_translations, dtype=np.float64),
        translations=current_translations,
        healpix_order=healpix_order,
    )
