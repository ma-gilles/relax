"""The local direction cone searched in the grid's latitude band keeps the dense search's supports."""

from __future__ import annotations

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.local_search import layout
from relax.sampling import build_local_search_grid_metadata

pytestmark = pytest.mark.unit


def _dense_candidates(prior_dir_vecs, dir_vecs, dir_z_order, dir_z_sorted, cone_rad, cos_prefilter):
    """The dense prefilter the band replaces: every grid direction's dot product."""

    dots = prior_dir_vecs @ dir_vecs.T
    image, point = np.nonzero(dots >= cos_prefilter)
    return layout._SparseCandidates(int(prior_dir_vecs.shape[0]), image, point, dots[image, point])


@pytest.mark.parametrize(
    "order,sigma_rot_deg,sigma_psi_deg",
    [(3, 1.87, 1.87), (4, 3.75, 0.0), (3, 7.5, 3.75), (2, 20.0, 10.0)],
)
def test_banded_cone_keeps_the_dense_supports(monkeypatch, order, sigma_rot_deg, sigma_psi_deg):
    rng = np.random.default_rng(order * 101 + int(sigma_rot_deg * 10))
    n = 700
    eulers = np.stack(
        [rng.uniform(-180, 180, n), np.degrees(np.arccos(rng.uniform(-1, 1, n))), rng.uniform(-180, 180, n)], axis=1
    )
    # Poles and the equator: a band clipped at z = +-1 and a band crossing z = 0.
    eulers[:3, 1] = [0.0, 180.0, 90.0]
    meta = build_local_search_grid_metadata(order)
    args = (eulers.astype(np.float64), order, np.deg2rad(sigma_rot_deg), np.deg2rad(sigma_psi_deg), meta)

    banded = layout._build_factorized_local_entries(*args, dtype=np.float32)
    monkeypatch.setattr(layout, "_direction_cone_candidates", _dense_candidates)
    dense = layout._build_factorized_local_entries(*args, dtype=np.float32)

    offsets, counts, ids, log_priors = banded
    np.testing.assert_array_equal(offsets, dense[0])
    np.testing.assert_array_equal(counts, dense[1])
    np.testing.assert_array_equal(ids, dense[2])
    assert_matches(log_priors, dense[3])
