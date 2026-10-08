"""VDAM particle subset selection: random shuffle, prefix, stable-sort, halfset id.

Mirrors RELION's ``randomiseParticlesOrder`` → first ``subset_size`` → stable-sort by
optics group (ml_optimiser.cpp:4907) → ``part_id % 2`` pseudo-halfset assignment
(:10349). The scheduler calls the native shuffle binding; this module owns
prefix selection, optics ordering and halfset assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SubsetPlan:
    """Selected input rows and RELION part IDs (int64), with halfset IDs (int8)."""

    particle_ids: np.ndarray
    part_ids: np.ndarray
    halfset_ids: np.ndarray


def select_vdam_subset(
    shuffled_particle_ids: np.ndarray,
    subset_size: int,
    optics_group_by_particle: Sequence[int],
    pseudo_halfsets: bool,
    halfset_particle_ids: np.ndarray,
) -> SubsetPlan:
    """Per-iteration plan: prefix-N of shuffle, stable-sort by optics group, BPref halfset assignment.

    ``halfset_particle_ids`` are RELION's part ids of the shuffled rows (same order); a pseudo-halfset
    is ``part_id % 2``. Caller must resolve ``subset_size=-1`` to ``nr_particles`` first.
    """
    if subset_size < 0 or subset_size > shuffled_particle_ids.size:
        raise ValueError(
            f"subset_size={subset_size} out of range for nr_particles={shuffled_particle_ids.size}; resolve -1 first"
        )
    prefix = np.asarray(shuffled_particle_ids[:subset_size], dtype=np.int64)
    keys = np.asarray([optics_group_by_particle[int(p)] for p in prefix], dtype=np.int64)
    stable_order = np.argsort(keys, kind="stable")
    sorted_prefix = prefix[stable_order]
    sorted_halfset_source = np.asarray(halfset_particle_ids, dtype=np.int64)[:subset_size][stable_order]
    halfsets = (
        (sorted_halfset_source % 2).astype(np.int8, copy=False)
        if pseudo_halfsets
        else np.zeros(sorted_prefix.size, dtype=np.int8)
    )
    return SubsetPlan(particle_ids=sorted_prefix, part_ids=sorted_halfset_source, halfset_ids=halfsets)
