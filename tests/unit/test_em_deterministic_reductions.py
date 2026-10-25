"""Fixed-order segment sums behind RELAX_EM_DETERMINISTIC_REDUCTIONS.

The opt-in replaces duplicate-index scatter-adds (shell binning, per-group
scale terms, per-image residual totals) with masked reductions.  These checks
pin the helper's semantics against the host reference and prove the flagged
``bin_shell_values_jax`` path matches the scatter path on the same operands.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers.float_compare import assert_matches

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from relax.helpers import deterministic_reduce as dr
from relax.helpers.half_spectrum import bin_shell_values_jax, bin_shell_values_np


@pytest.fixture(autouse=True, scope="module")
def _on_cpu():
    """Run this module on CPU without changing the process-wide platform for later test files."""
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _shell_case(seed: int = 0):
    rng = np.random.default_rng(seed)
    n_pix, n_shells = 517, 9
    # Include sentinel (== n_shells) and out-of-range ids, as the engines do.
    ids = rng.integers(-1, n_shells + 2, size=n_pix).astype(np.int32)
    values = rng.standard_normal(n_pix)
    return values, ids, n_shells


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv(dr.DETERMINISTIC_REDUCTIONS_ENV, raising=False)
    assert dr.deterministic_reductions_enabled() is False
    monkeypatch.setenv(dr.DETERMINISTIC_REDUCTIONS_ENV, "1")
    assert dr.deterministic_reductions_enabled() is True


def test_fixed_order_segment_sum_matches_host_bincount_and_drops_out_of_range():
    values, ids, n = _shell_case()
    got = np.asarray(dr.fixed_order_segment_sum(jnp.asarray(values), ids, n))
    expected = bin_shell_values_np(values, ids, n)
    assert got.shape == (n,)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
    # Integer-valued float32 inputs reduce exactly in any order.
    ints = np.round(values * 4).astype(np.float32)
    got32 = np.asarray(dr.fixed_order_segment_sum(jnp.asarray(ints), ids, n))
    assert got32.dtype == np.float32
    assert_matches(got32, bin_shell_values_np(ints, ids, n).astype(np.float32))


def test_fixed_order_segment_sum_leading_axes_and_shape_check():
    values, ids, n = _shell_case(1)
    stacked = np.stack([values, 2.0 * values, -values], axis=0)
    got = np.asarray(dr.fixed_order_segment_sum(jnp.asarray(stacked), ids, n))
    assert got.shape == (3, n)
    for row, scale in zip(got, (1.0, 2.0, -1.0)):
        np.testing.assert_allclose(row, scale * bin_shell_values_np(values, ids, n), atol=1e-12)
    with pytest.raises(ValueError, match="segment ids"):
        dr.fixed_order_segment_sum(jnp.asarray(values[:-1]), ids, n)


def test_add_segment_sum_matches_scatter_under_both_flag_states(monkeypatch):
    rng = np.random.default_rng(3)
    n_groups, batch = 4, 37
    group_ids = rng.integers(0, n_groups, size=batch).astype(np.int32)
    per_image = rng.standard_normal(batch).astype(np.float32)
    acc0 = jnp.asarray(rng.standard_normal(n_groups).astype(np.float32))
    monkeypatch.delenv(dr.DETERMINISTIC_REDUCTIONS_ENV, raising=False)
    scatter = np.asarray(dr.add_segment_sum(acc0, group_ids, per_image))
    monkeypatch.setenv(dr.DETERMINISTIC_REDUCTIONS_ENV, "1")
    fixed = np.asarray(dr.add_segment_sum(acc0, group_ids, per_image))
    assert fixed.dtype == scatter.dtype == np.float32
    reference = np.asarray(acc0, dtype=np.float64) + np.bincount(
        group_ids, weights=per_image.astype(np.float64), minlength=n_groups
    )
    np.testing.assert_allclose(scatter, reference, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(fixed, reference, rtol=1e-5, atol=1e-6)


def test_bin_shell_values_jax_flagged_path_matches_scatter_path(monkeypatch):
    values, ids, n = _shell_case(5)
    monkeypatch.delenv(dr.DETERMINISTIC_REDUCTIONS_ENV, raising=False)
    scatter = np.asarray(bin_shell_values_jax(jnp.asarray(values), ids, n))
    monkeypatch.setenv(dr.DETERMINISTIC_REDUCTIONS_ENV, "1")
    fixed = np.asarray(bin_shell_values_jax(jnp.asarray(values), ids, n))
    assert fixed.shape == scatter.shape == (n,)
    np.testing.assert_allclose(fixed, scatter, rtol=0, atol=1e-12)
    np.testing.assert_allclose(fixed, bin_shell_values_np(values, ids, n), atol=1e-12)


def test_fixed_order_segment_sum_is_bitwise_repeatable_under_jit():
    values, ids, n = _shell_case(7)
    fn = jax.jit(lambda v: dr.fixed_order_segment_sum(v, ids, n))
    first = np.asarray(fn(jnp.asarray(values, dtype=jnp.float32)))
    for _ in range(3):
        assert_matches(np.asarray(fn(jnp.asarray(values, dtype=jnp.float32))), first)


def test_mstep_fixed_order_shell_sums_match_scatter_rule():
    from relax.helpers.deterministic_reduce import fixed_order_shell_sums, static_shell_voxel_lists

    capacity, pf, n_shells = 19, 1, 9  # ori_size 16, padding 1 -> capacity 19
    lists = static_shell_voxel_lists(capacity, pf, n_shells)
    clamped = static_shell_voxel_lists(capacity, pf, n_shells, clamp_to_last=True)
    assert sum(len(l) for l in clamped) == capacity * capacity * (capacity // 2 + 1)
    coord = np.arange(capacity) - capacity // 2; x = np.arange(capacity // 2 + 1)
    r2 = coord[:, None, None] ** 2 + coord[None, :, None] ** 2 + x[None, None, :] ** 2
    shells = np.floor(np.sqrt(r2.astype(np.float64)) / pf + 0.5).astype(int).reshape(-1)
    assert sum(len(l) for l in lists) == int((shells < n_shells).sum())
    rng = np.random.default_rng(4); flat = rng.random(shells.size).astype(np.float32)
    got = np.asarray(fixed_order_shell_sums(jnp.asarray(flat), lists, jnp.float32))
    ref = np.bincount(np.where(shells < n_shells, shells, n_shells), weights=flat.astype(np.float64), minlength=n_shells + 1)[:n_shells]
    np.testing.assert_allclose(got, ref, rtol=2e-6)




def _per_shell_scan_lists(capacity, padding_factor, n_shells, *, clamp_to_last=False, rows=None):
    """The lists by one grid scan per shell (relax ca30e2bb), the reference for the sorted construction."""
    coord = np.arange(capacity, dtype=np.int64) - capacity // 2
    y = coord if rows is None else coord[rows[0] : rows[1]]
    x = np.arange(capacity // 2 + 1, dtype=np.int64)
    r2 = coord[:, None, None] ** 2 + y[None, :, None] ** 2 + x[None, None, :] ** 2
    shells = np.floor(np.sqrt(r2.astype(np.float64)) / float(padding_factor) + 0.5).astype(np.int64)
    if clamp_to_last:
        shells = np.minimum(shells, n_shells - 1)
    shells = shells.reshape(-1)
    return tuple(np.flatnonzero(shells == s).astype(np.int32) for s in range(n_shells))


@pytest.mark.parametrize("clamp_to_last", [False, True])
@pytest.mark.parametrize(
    "capacity, padding_factor, n_shells, rows",
    [
        (11, 1, 6, None),
        (19, 1, 9, None),  # M-step capacity of ori_size 16, padding 1
        (20, 1, 9, None),  # even capacities
        (64, 2, 17, None),
        (35, 2, 9, (5, 30)),
        (67, 2, 17, (0, 12)),
        (131, 2, 33, (120, 131)),  # last, short chunk
        (131, 2, 33, (40, 40)),  # empty chunk
        (387, 2, 97, None),  # host window of box 192 at full radius
        (1603, 2, 401, (800, 806)),  # a y-chunk of box 800's host build (padding 2, full radius)
    ],
)
def test_static_shell_voxel_lists_equal_the_per_shell_scan(capacity, padding_factor, n_shells, rows, clamp_to_last):
    lists = dr.static_shell_voxel_lists(capacity, padding_factor, n_shells, clamp_to_last=clamp_to_last, rows=rows)
    reference = _per_shell_scan_lists(capacity, padding_factor, n_shells, clamp_to_last=clamp_to_last, rows=rows)
    assert len(lists) == len(reference) == n_shells
    for got, expected in zip(lists, reference):
        assert got.dtype == np.int32
        np.testing.assert_array_equal(got, expected)
    assert dr.static_shell_voxel_lists(capacity, padding_factor, 0, clamp_to_last=clamp_to_last, rows=rows) == ()


@pytest.mark.parametrize(
    "ori_size, padding_factor, window_radius, cy",
    [(16, 1, 8, None), (24, 2, 7, None), (32, 2, 16, 9)],  # the last one in y-chunks of 9 rows
)
def test_mask_and_shell_power_program_is_unchanged_by_the_sorted_lists(
    monkeypatch, ori_size, padding_factor, window_radius, cy
):
    """The lists are trace-time constants: the lowered program must be the per-shell scan's, op for op."""
    from relax.relion import relion_projector_setup as setup

    monkeypatch.setenv(dr.DETERMINISTIC_REDUCTIONS_ENV, "1")
    size = 2 * (padding_factor * window_radius + 1) + 1
    rng = np.random.default_rng(11)
    slab = rng.standard_normal((size, size, size // 2 + 1)) + 1j * rng.standard_normal((size, size, size // 2 + 1))
    cy = size if cy is None else cy

    def lower_and_run(lists_fn):
        monkeypatch.setattr(setup, "static_shell_voxel_lists", lists_fn)
        jax.clear_caches()
        programs, outputs = [], []
        for y0 in range(0, size, cy):
            block = jnp.asarray(slab[:, y0 : y0 + cy])
            kwargs = dict(ori_size=ori_size, padding_factor=padding_factor, size=size, y_start=y0)
            programs.append(setup._mask_and_shell_power.lower(block, window_radius - 1, **kwargs).as_text())
            outputs.append(setup._mask_and_shell_power(block, window_radius - 1, **kwargs))
        return programs, outputs

    sorted_programs, sorted_outputs = lower_and_run(dr.static_shell_voxel_lists)
    scan_programs, scan_outputs = lower_and_run(_per_shell_scan_lists)
    jax.clear_caches()
    assert sorted_programs == scan_programs
    for got, expected in zip(sorted_outputs, scan_outputs):
        for got_array, expected_array in zip(got, expected):
            assert_matches(np.asarray(got_array), np.asarray(expected_array))
