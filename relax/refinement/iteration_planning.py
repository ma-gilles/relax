"""Host-side plans for one standard refinement iteration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from relax import sampling
from relax.dense.scoring_policy import _dense_global_scoring_dtype
from relax.diagnostics.relion_replay import _sealed_sampling_base_grids


@dataclass(frozen=True)
class InitialCoarseGridSpec:
    """Specification selecting the first exhaustive RELION trial grid.

    Array payloads are retained by reference. Constructing this specification does
    not cast, copy or transfer them.
    """

    healpix_order: int
    sealed_sampling_state: Any | None
    translations: Any | None
    init_healpix_order: int
    init_translation_range: float
    init_translation_step: float
    n_classes: int
    voxel_size: float
    log: logging.Logger
    symmetry: str = "C1"


@dataclass(frozen=True)
class InitialCoarseGrids:
    """Exhaustive coarse trial grid selected for refinement startup."""

    rotations: np.ndarray
    rotation_eulers: np.ndarray
    base_translations: np.ndarray
    translations: jnp.ndarray
    healpix_order: int


def build_initial_coarse_grids(spec: InitialCoarseGridSpec) -> InitialCoarseGrids:
    """Materialize the first exhaustive coarse grid of a RELION refinement.

    A schema-v3 sealed sampling state supplies its own restricted Euler rows
    and translations and must sit at the initialized HEALPix order. Otherwise
    RELION's canonical grid at ``healpix_order`` is paired with the caller's
    translation table or, when none is given, with the RELION translation grid
    of the initial offset range and step.
    """

    dtype = _dense_global_scoring_dtype()
    healpix_order = spec.healpix_order
    if spec.sealed_sampling_state is not None:
        rotations, rotation_eulers, current_translations = _sealed_sampling_base_grids(
            spec.sealed_sampling_state,
            voxel_size_angstrom=spec.voxel_size,
            dtype=dtype,
        )
        base_translations = np.asarray(current_translations, dtype=np.float64)
        healpix_order = int(spec.sealed_sampling_state["healpix_order_original"])
        if healpix_order != int(spec.init_healpix_order):
            raise ValueError(
                "sealed sampling HEALPix order does not match initialized boundary: "
                f"sealed={healpix_order} init={spec.init_healpix_order}"
            )
        spec.log.info(
            "Frozen-boundary v3 directly materialized %d Euler rows and %d translations",
            int(rotation_eulers.shape[0]),
            int(current_translations.shape[0]),
        )
    else:
        rotations, rotation_eulers = sampling._relion_rotation_grid_float32(
            healpix_order,
            dtype=dtype,
            **({"symmetry": spec.symmetry} if spec.symmetry != "C1" else {}),
        )
        translations = spec.translations
        if translations is None:
            translations = sampling._relion_base_translation_grid(
                spec.init_translation_range,
                spec.init_translation_step,
                n_classes=spec.n_classes,
                voxel_size=spec.voxel_size,
            )
        base_translations = np.asarray(translations, dtype=np.float64)
        current_translations = jnp.asarray(translations, dtype=dtype)
    return InitialCoarseGrids(
        rotations=rotations,
        rotation_eulers=rotation_eulers,
        base_translations=base_translations,
        translations=current_translations,
        healpix_order=int(healpix_order),
    )
