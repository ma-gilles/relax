"""Per-optics-group noise in the device-resident pass-2 statistics (CPU, pure JAX).

Each image adds its noise sums to its own optics group (RELION's
``wsum_model.sigma2_noise[optics_group]`` and ``sumw_group[optics_group]``). The
per-group path must give, for group g, what the one-group path gives on group g's
images and noise spectrum alone.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.helpers.half_spectrum import make_relion_noise_shell_indices_half
from relax.helpers.optics_noise import dense_optics_groups, noise_rows, pixel_rows
from relax.helpers.projection import compute_noise_block, compute_noise_block_per_optics_group
from relax.sparse_pass2 import resident_pass2 as rp
from relax.sparse_pass2.resident_statistics import make_resident_statistics, resolve_statistics_config
from relax.sparse_pass2.sparse_pass2_wavg import image_power_shells

IMAGE_SHAPE = (16, 16)
N_SHELLS = 9
P_HALF = 16 * 9


def _complex(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


@pytest.mark.unit
def test_noise_rows_and_dense_groups():
    table = jnp.arange(12.0).reshape(3, 4)
    groups, n_groups = dense_optics_groups([7, 3, 7, 9])
    assert n_groups == 3 and groups.tolist() == [1, 0, 1, 2]
    assert_matches(np.asarray(noise_rows(table, groups, [3, 0])), np.asarray(table)[[2, 1]])
    shared = jnp.ones(4)
    assert noise_rows(shared, None, [0, 1]) is shared
    assert pixel_rows(shared).shape == (1, 4) and pixel_rows(table).shape == (3, 4)


@pytest.mark.unit
def test_noise_block_per_group_equals_one_group_blocks():
    rng = np.random.default_rng(0)
    rows, n_groups = 12, 3
    shells = make_relion_noise_shell_indices_half(IMAGE_SHAPE)
    proj = _complex(rng, (rows, P_HALF))
    proj_abs2 = np.abs(proj) ** 2
    summed = _complex(rng, (rows, P_HALF))
    ctf_probs = rng.uniform(0.0, 1.0, (rows, P_HALF)) * (rng.uniform(size=(rows, P_HALF)) > 0.2)
    table = rng.uniform(0.5, 2.0, (n_groups, P_HALF))
    row_groups = np.array([0, 1, 2, 1, 0, 0, 2, 2, 1, 0, 1, 2], dtype=np.int32)

    per_group = compute_noise_block_per_optics_group(
        proj, proj_abs2, summed, ctf_probs, jnp.asarray(table), jnp.asarray(row_groups), shells, N_SHELLS
    )
    assert per_group.shape == (n_groups, N_SHELLS)
    for g in range(n_groups):
        mine = row_groups == g
        expected, _, _ = compute_noise_block(
            proj[mine], proj_abs2[mine], summed[mine], ctf_probs[mine], table[g], shells, N_SHELLS
        )
        np.testing.assert_allclose(np.asarray(per_group[g]), np.asarray(expected), rtol=1e-12, atol=1e-12)


@pytest.mark.unit
def test_block_norm_terms_use_each_rows_group_noise():
    rng = np.random.default_rng(1)
    rows, capacity = 8, 4
    shells = make_relion_noise_shell_indices_half(IMAGE_SHAPE)
    proj = _complex(rng, (rows, P_HALF))
    summed = _complex(rng, (rows, P_HALF))
    ctf_probs = rng.uniform(0.0, 1.0, (rows, P_HALF))
    table = rng.uniform(0.5, 2.0, (2, P_HALF))
    row_image = np.array([0, 0, 1, 1, 2, 3, 3, 3], dtype=np.int32)
    image_groups = np.array([1, 0, 1, 0], dtype=np.int32)
    row_groups = image_groups[row_image]

    shells_g, a2_g, xa_g = rp._resident_block_noise_and_norm(
        proj,
        np.abs(proj) ** 2,
        summed,
        ctf_probs,
        jnp.asarray(table),
        shells,
        jnp.asarray(row_image),
        jnp.asarray(row_groups),
        n_shells=N_SHELLS,
        image_capacity=capacity,
    )
    assert shells_g.shape == (2, N_SHELLS)
    for g in range(2):
        mine = row_groups == g
        one_shells, one_a2, one_xa = rp._resident_block_noise_and_norm(
            proj[mine],
            np.abs(proj[mine]) ** 2,
            summed[mine],
            ctf_probs[mine],
            jnp.asarray(table[g]),
            shells,
            jnp.asarray(row_image[mine]),
            n_shells=N_SHELLS,
            image_capacity=capacity,
        )
        np.testing.assert_allclose(np.asarray(shells_g[g]), np.asarray(one_shells), rtol=1e-12, atol=1e-12)
        images = image_groups == g
        np.testing.assert_allclose(np.asarray(a2_g)[images], np.asarray(one_a2)[images], rtol=1e-12)
        np.testing.assert_allclose(np.asarray(xa_g)[images], np.asarray(one_xa)[images], rtol=1e-12)


def _chunk(rng, *, image_ids, optics_groups, block_noise_shells, wavg_diff2, capacity, n_rect):
    rows = 10
    wavg = np.zeros((capacity, n_rect, 3), dtype=np.float32)
    wavg[:, :, 2] = wavg_diff2
    return rp._ChunkImageOperands(
        row_posterior=jnp.asarray(rng.uniform(0.0, 0.3, (rows, 3)), dtype=jnp.float32),
        row_image_local=jnp.asarray(np.arange(rows) % capacity, dtype=jnp.int32),
        row_coarse_rot=jnp.zeros(rows, dtype=jnp.int32),
        image_ids=jnp.asarray(image_ids, dtype=jnp.int32),
        group_ids=jnp.full(capacity, -1, dtype=jnp.int32),
        image_power_shells=image_power_shells(
            jnp.asarray(_complex(rng, (capacity, P_HALF)), dtype=jnp.complex64),
            jnp.asarray(make_relion_noise_shell_indices_half(IMAGE_SHAPE), dtype=jnp.int32),
            shell_count=N_SHELLS,
        ),
        relion_norm_high_shell=None,
        wavg_triplet_pixels=jnp.asarray(wavg),
        block_noise_shells=jnp.asarray(block_noise_shells),
        a2_per_image=jnp.zeros(capacity),
        xa_per_image=jnp.zeros(capacity),
        class_log_z=jnp.zeros(capacity, dtype=jnp.float64),
        min_diff2=jnp.zeros(capacity),
        best_log_score=jnp.zeros(capacity, dtype=jnp.float32),
        max_posterior=jnp.zeros(capacity, dtype=jnp.float32),
        best_cell_index=jnp.zeros(capacity, dtype=jnp.int64),
        best_fine_rot=jnp.zeros(capacity, dtype=jnp.int64),
        optics_groups=None if optics_groups is None else jnp.asarray(optics_groups, dtype=jnp.int32),
    )


@pytest.mark.unit
def test_chunk_statistics_per_group_equal_one_group_runs():
    capacity, n_rect, n_groups = 6, 20, 2
    groups = np.array([0, 1, 1, 0, 1, 0], dtype=np.int32)
    image_ids = np.array([0, 1, 2, 3, 4, -1], dtype=np.int32)
    base = np.random.default_rng(2)
    block_noise_shells = base.normal(size=(n_groups, N_SHELLS))
    wavg_diff2 = base.uniform(size=(capacity, n_rect)).astype(np.float32)
    tables = rp._ChunkImageTables(
        shell_indices_half=make_relion_noise_shell_indices_half(IMAGE_SHAPE),
        wavg_shell_indices=jnp.asarray(np.arange(n_rect) % N_SHELLS, dtype=jnp.int32),
        wavg_scale_pixel_mask=jnp.ones(n_rect, dtype=bool),
        translation_sqdist_ang=None,
    )

    def config(g):
        return resolve_statistics_config(
            n_shells=N_SHELLS,
            n_fine_trans=3,
            n_images=5,
            n_coarse_rot=4,
            n_scale_groups=1,
            current_size=8,
            accumulate_scale=False,
            n_optics_groups=g,
        )

    stats = rp._accumulate_chunk_image_terms(
        make_resident_statistics(config(n_groups)),
        _chunk(
            np.random.default_rng(3),
            image_ids=image_ids,
            optics_groups=groups,
            block_noise_shells=block_noise_shells,
            wavg_diff2=wavg_diff2,
            capacity=capacity,
            n_rect=n_rect,
        ),
        tables,
        config=config(n_groups),
    )
    assert stats.wsum_sigma2_noise.shape == (n_groups, N_SHELLS) and stats.sumw.shape == (n_groups,)
    for g in range(n_groups):
        mine = (groups == g) & (image_ids >= 0)
        one = rp._accumulate_chunk_image_terms(
            make_resident_statistics(config(1)),
            _chunk(
                np.random.default_rng(3),
                image_ids=np.where(mine, image_ids, -1),
                optics_groups=None,
                block_noise_shells=block_noise_shells[g],
                wavg_diff2=np.where(mine[:, None], wavg_diff2, 0.0),
                capacity=capacity,
                n_rect=n_rect,
            ),
            tables,
            config=config(1),
        )
        np.testing.assert_allclose(
            np.asarray(stats.wsum_sigma2_noise[g]), np.asarray(one.wsum_sigma2_noise), rtol=1e-12
        )
        np.testing.assert_allclose(np.asarray(stats.wsum_img_power[g]), np.asarray(one.wsum_img_power), rtol=1e-6)
        np.testing.assert_allclose(float(stats.sumw[g]), float(one.sumw), rtol=1e-12)
