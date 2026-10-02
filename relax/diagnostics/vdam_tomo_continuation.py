"""Subtomogram particle state from a RELION VDAM checkpoint, for the one-iteration diagnostic continuation.

A fresh run starts every particle in class 0 with Pmax 0 (``initial_model_io._tomo_particle_state_from_star``);
a checkpoint's data STAR also carries each particle's class (0 = not yet visited) and Pmax, which the
change monitor and the subset bookkeeping read.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from recovar.data_io.starfile import star_column

from relax.relion.initial_model_io import _tomo_particle_state_from_star


def tomo_checkpoint_particle_state(main_star, *, pixel_size: float):
    state = _tomo_particle_state_from_star(main_star, pixel_size=pixel_size)
    numbers = np.asarray(star_column(main_star, "_rlnClassNumber", required=True).astype(int), dtype=np.int32)
    pmax = star_column(main_star, "_rlnMaxValueProbDistribution", required=True).astype(float)
    return replace(
        state,
        class_assignments=np.maximum(numbers - 1, 0).astype(np.int32),
        max_posterior=np.asarray(pmax, dtype=np.float32),
        visited=numbers > 0,
    )
