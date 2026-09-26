"""RELION firstiter-CC float32 reduction parity tests."""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp
from helpers.sparse_pass2_test_support import (
    _score_pass2_pairs_relion_gpu_normalized_cc,
)


pytestmark = pytest.mark.unit


def _inputs(n_pix=17):
    rng = np.random.default_rng(161552)
    batch, n_rot, n_trans = 1, 3, 4
    shifted = (
        rng.normal(size=(batch, n_trans, n_pix))
        + 1j * rng.normal(size=(batch, n_trans, n_pix))
    ).astype(np.complex64)
    projections = (
        rng.normal(size=(batch, n_rot, n_pix))
        + 1j * rng.normal(size=(batch, n_rot, n_pix))
    ).astype(np.complex64)
    score_weight = rng.uniform(0.1, 1.5, size=(batch, n_pix)).astype(np.float32)
    half_weight = rng.choice(np.asarray([1.0, 2.0], dtype=np.float32), size=n_pix)
    mask = np.ones((batch, n_rot, n_trans), dtype=bool)
    return shifted, score_weight, projections, half_weight, mask


def test_normalized_cc_pair_lowers_to_relion_lane_scan_not_generic_reduction():
    shifted, score_weight, projections, half_weight, _mask = map(jnp.asarray, _inputs())
    rotation_rows, translation_rows = np.meshgrid(
        np.arange(projections.shape[1], dtype=np.int32),
        np.arange(shifted.shape[1], dtype=np.int32),
        indexing="ij",
    )
    pair_mask = jnp.ones((1, rotation_rows.size), dtype=bool)
    jaxpr = str(
        jax.make_jaxpr(_score_pass2_pairs_relion_gpu_normalized_cc)(
            shifted,
            score_weight,
            projections,
            half_weight,
            jnp.asarray(rotation_rows.reshape(1, -1)),
            jnp.asarray(translation_rows.reshape(1, -1)),
            pair_mask,
        )
    )

    assert "scan" in jaxpr
    assert "reduce_sum" not in jaxpr
    assert "dot_general" not in jaxpr


