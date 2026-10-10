"""Local supports are selected from the prior angles in double, as RELION does."""

from __future__ import annotations

import numpy as np
import pytest

from relax.local_search import layout
from relax.sampling import build_local_search_grid_metadata

pytestmark = pytest.mark.unit


def test_direction_support_at_the_cone_edge_uses_double_prior_angles():
    # RELION's selectOrientationsWithNonZeroPriorProbability (healpix_sampling.cpp:710-790) keeps direction idir
    # when ACOSD(prior_direction . my_direction) < 3 sigma, with the prior angles in RFLOAT (double) and
    # Euler_angles2direction; the GPU build calls it with the same doubles (acc_ml_optimiser_impl.h:611).
    # Priors 1e-9 deg inside the cone around one grid direction must keep it; float32 prior angles move them
    # by ~1e-6 deg and drop about half.
    order, sigma_deg, target = 3, 2.5, 100
    meta = build_local_search_grid_metadata(order)
    n_pixels = int(meta["n_pixels"])
    d0 = np.asarray(meta["dir_vecs"], dtype=np.float64)[target]
    theta, phi = np.arccos(d0[2]), np.arctan2(d0[1], d0[0])
    e1 = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    e2 = np.cross(d0, e1)
    angle = np.deg2rad(3 * sigma_deg - 1e-9)
    around = np.linspace(0.0, 2 * np.pi, 64, endpoint=False)
    v = np.cos(angle) * d0[None] + np.sin(angle) * (np.cos(around)[:, None] * e1 + np.sin(around)[:, None] * e2)
    prior = np.column_stack(
        [np.degrees(np.arctan2(v[:, 1], v[:, 0])), np.degrees(np.arccos(v[:, 2])), np.zeros(len(v))]
    )
    rot, tilt = np.deg2rad(prior[:, 0]), np.deg2rad(prior[:, 1])
    direction = np.column_stack([np.sin(tilt) * np.cos(rot), np.sin(tilt) * np.sin(rot), np.cos(tilt)])
    assert np.all(np.degrees(np.arccos(direction @ d0)) < 3 * sigma_deg)

    offsets, _, ids, _ = layout._build_factorized_local_entries(prior, order, np.deg2rad(sigma_deg), 0.0, meta)
    kept = [target in set((ids[offsets[i] : offsets[i + 1]] % n_pixels).tolist()) for i in range(len(prior))]
    assert all(kept)
