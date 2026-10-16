"""The M-step's partial accumulator pair (relax#65), the diagnostic behind ``RELAX_EM_EXACT_FLOAT32_ACCUMULATION=1``.

Float32 rows are scattered into a zeroed pair that is flushed into the totals; without the switch (the default)
the pass allocates no partial and every loop accumulates directly, as before.

A float32 accumulator drops an addend below half an ulp of its running total, so a pass that scatters
millions of rows into one pair loses weight where every row lands (the low shells). The references here are
NumPy float64 sums and NumPy float32 sums in the stated order; none of them calls the code under test.

The row-block bodies are stand-ins that add each row's weight to voxel 0 one row at a time, the order a
scatter into one voxel has on a single thread. The loops around them are the production drivers.
"""

import dataclasses
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from relax.fine_pass import (
    resident_block_programs,
    resident_chunk_records,
    resident_chunk_stages,
    resident_mstep_blocks,
    resident_tilt_mstep,
)
from relax.reconstruction import half_volume_mstep as hvm
from relax.runtime.scoring_overrides import ScoringOverrides

pytestmark = pytest.mark.unit

BLOCK_ROWS = 2048
N_ROWS = 1 << 22
VOLUME = 4


def _weights(n_rows=N_ROWS):
    """Heavy-tailed row weights (log-normal, sigma 2): most addends are far below the mean."""

    return np.exp(2.0 * np.random.default_rng(65).standard_normal(n_rows)).astype(np.float32)


def _float32_references(weights, rows_per_partial):
    """NumPy float32 sums in row order: one accumulator, and partials of ``rows_per_partial`` rows added in order."""

    single = np.cumsum(weights, dtype=np.float32)[-1]
    partials = np.cumsum(weights.reshape(-1, rows_per_partial), axis=1, dtype=np.float32)[:, -1]
    return single, np.cumsum(partials, dtype=np.float32)[-1]


def _add_rows(carry, block_weights):
    """Add a block's weights to voxel 0 of both accumulators, one row at a time."""

    def add(acc, weight):
        return acc + weight, None

    ctf0, _ = jax.lax.scan(add, carry.Ft_ctf[0], block_weights)
    y0, _ = jax.lax.scan(add, carry.Ft_y[0], block_weights.astype(carry.Ft_y.dtype))
    return carry._replace(Ft_y=carry.Ft_y.at[0].set(y0), Ft_ctf=carry.Ft_ctf.at[0].set(ctf0))


def _carry(Ft_y, Ft_ctf):
    return resident_chunk_records.ChunkMstepCarry(
        Ft_y=Ft_y,
        Ft_ctf=Ft_ctf,
        wavg_triplet_pixels=jnp.zeros((1, 1, 3), jnp.float32),
        noise_shells=jnp.zeros(2, jnp.float64),
        a2_per_image=jnp.zeros(1, jnp.float32),
        xa_per_image=jnp.zeros(1, jnp.float32),
    )


def _totals():
    return jnp.zeros(VOLUME, jnp.complex64), jnp.zeros(VOLUME, jnp.float32)


@dataclasses.dataclass(frozen=True)
class _Spec:
    mstep_block_rows: int = BLOCK_ROWS
    n_slots: int = 1
    n_classes: int = 1
    row_capacity: int = N_ROWS
    presum_adjoint: bool = False
    glue_jit: bool = False  # the eager stage path, the oracle


def _stage_driver(monkeypatch, weights, row_partial):
    """The default global driver (per-stage block loop) over one slot holding every row."""

    add_block = jax.jit(lambda carry, operands, start: _add_rows(carry, jax.lax.dynamic_slice_in_dim(operands, start, BLOCK_ROWS)))

    def block_at(start, blocks, operands, tables, carry, *, spec):
        return add_block(carry, operands, start)

    monkeypatch.setattr(resident_mstep_blocks, "resident_mstep_block_at", block_at)
    monkeypatch.setattr(resident_mstep_blocks, "initial_mstep_carry", lambda Ft_y, Ft_ctf, *a, **k: _carry(Ft_y, Ft_ctf))
    monkeypatch.setattr(resident_block_programs, "add_chunk_wavg_image_power", lambda mstep, *a, **k: mstep)
    monkeypatch.setattr(resident_chunk_stages, "_resident_chunk_statistics", lambda stats, *a, **k: stats)
    n_rows = int(weights.shape[0])
    blocks = resident_chunk_records.MstepBlockInputs(
        row_image_local=None,
        kernel_row_image_ids=None,
        row_posterior=None,
        row_fine_rot=None,
        projections=None,
        slot_offsets=jnp.asarray([0, n_rows], jnp.int32),
    )
    posterior = types.SimpleNamespace(row_posterior=None, kernel_row_image_ids=None)
    Ft_y, Ft_ctf = _totals()
    Ft_y_out, Ft_ctf_out, _ = resident_chunk_stages.resident_chunk_stages_finish(
        (posterior, blocks),
        None,
        jnp.asarray(weights),
        None,
        ((Ft_y,), (Ft_ctf,), None),
        spec=_Spec(row_capacity=n_rows),
        row_partial=row_partial,
    )
    return Ft_y_out[0], Ft_ctf_out[0]


def _local_driver(monkeypatch, weights, row_partial):
    """The local-search and final-pass driver (``run_resident_mstep_blocks``)."""

    add_rows = jax.jit(_add_rows)

    def block(*, block_row_image, carry, **kwargs):
        return add_rows(carry, block_row_image)

    monkeypatch.setattr(resident_mstep_blocks, "_resident_mstep_block", block)
    monkeypatch.setattr(resident_mstep_blocks, "initial_mstep_carry", lambda Ft_y, Ft_ctf, *a, **k: _carry(Ft_y, Ft_ctf))
    monkeypatch.setattr(resident_block_programs, "add_chunk_wavg_image_power", lambda mstep, *a, **k: mstep)
    n_rows = int(weights.shape[0])
    recon = dict.fromkeys(
        (
            "shifted_recon",
            "shifted_noise",
            "ctf2_over_nv_recon",
            "direct_ctf_rfloat_recon",
            "raw_translated_wavg_rectangle",
            "raw_translated_wavg_for_atomic",
            "scale",
        )
    )
    recon.update(shifted_recon=object(), shifted_noise=object())
    Ft_y, Ft_ctf = _totals()
    out = resident_mstep_blocks.run_resident_mstep_blocks(
        lambda start, stop: (jnp.zeros(1, jnp.complex64), jnp.zeros(1, jnp.float32), None),
        row_capacity=n_rows,
        n_valid_rows=n_rows,
        mstep_block_rows=BLOCK_ROWS,
        glue_jit=False,
        image_capacity=1,
        row_image_local=jnp.asarray(weights),
        kernel_row_image_ids=np.zeros(n_rows, np.int32),
        row_posterior=np.zeros((n_rows, 1), np.float32),
        recon=recon,
        n_rect=1,
        n_shells=2,
        n_recon_windowed=1,
        noise_variance_for_noise=None,
        shell_indices_noise=None,
        exact_positions_device=None,
        Ft_y_total=Ft_y,
        Ft_ctf_total=Ft_ctf,
        image_shape=(8, 8),
        recon_volume_shape=(8, 8, 8),
        mstep_current_size=8,
        relion_x_half_recon_indices=None,
        max_adjoint_block_bytes=1 << 20,
        row_partial=row_partial,
    )
    return out[0], out[1]


def _partial():
    return hvm.RowPartial(VOLUME, jnp.complex64, jnp.float32)


def test_flush_size_is_a_static_block_count():
    assert hvm.ROW_PARTIAL_FLUSH_ROWS == 32768
    assert hvm.blocks_per_flush(2048) == 16
    assert hvm.blocks_per_flush(32) == 1024
    assert hvm.blocks_per_flush(32768) == 1
    assert hvm.blocks_per_flush(1 << 20) == 1  # a block larger than the flush size is flushed on its own


def test_flushed_float32_total_matches_the_float64_sum_where_one_accumulator_does_not(monkeypatch):
    """4.2M heavy-tailed rows: one float32 accumulator is off by more than 1e-3, the flushed one by less than 1e-6.

    The per-stage and local drivers are run on the same rows, with and without the pass's partial pair.
    """

    weights = _weights()
    reference = float(weights.astype(np.float64).sum())
    single32, flushed32 = _float32_references(weights, hvm.ROW_PARTIAL_FLUSH_ROWS)
    # The data separate the two accumulations, so the assertions below can fail.
    assert abs(1.0 - float(single32) / reference) > 1e-3
    assert abs(1.0 - float(flushed32) / reference) < 1e-6

    drivers = {
        "per-stage": lambda partial: _stage_driver(monkeypatch, weights, partial),
        "local": lambda partial: _local_driver(monkeypatch, weights, partial),
    }
    for name, run in drivers.items():
        direct_y, direct_ctf = run(None)
        # Without a partial every driver is the single accumulator, bit for bit: the path is unchanged.
        assert np.asarray(direct_ctf)[0] == single32, name
        flushed_y, flushed_ctf = run(_partial())
        # With it, the partials of 32,768 rows added in order, bit for bit, so the flush points are
        # every blocks_per_flush blocks in every driver.
        assert np.asarray(flushed_ctf)[0] == flushed32, name
        assert abs(1.0 - float(np.asarray(flushed_ctf)[0]) / reference) < 1e-6, name
        assert abs(1.0 - float(np.asarray(flushed_y)[0].real) / reference) < 1e-6, name
        assert abs(1.0 - float(np.asarray(direct_y)[0].real) / reference) > 1e-3, name


@pytest.mark.parametrize("n_blocks", [1, 5, 16])
def test_a_chunk_smaller_than_the_flush_size_is_the_direct_accumulation(monkeypatch, n_blocks):
    """Up to blocks_per_flush blocks from zeroed totals: zero plus the partial is the direct sum, bit for bit."""

    weights = _weights(n_blocks * BLOCK_ROWS)
    for name, run in (
        ("per-stage", lambda partial: _stage_driver(monkeypatch, weights, partial)),
        ("local", lambda partial: _local_driver(monkeypatch, weights, partial)),
    ):
        direct = run(None)
        flushed = run(_partial())
        np.testing.assert_array_equal(np.asarray(direct[0]), np.asarray(flushed[0]), err_msg=name)
        np.testing.assert_array_equal(np.asarray(direct[1]), np.asarray(flushed[1]), err_msg=name)


@pytest.mark.parametrize("n_blocks, expected", [(0, 0), (1, 1), (16, 1), (17, 2), (40, 3)])
def test_flush_count_is_ceil_blocks_over_blocks_per_flush(monkeypatch, n_blocks, expected):
    """One flush per blocks_per_flush blocks and one for the remainder; a slot without blocks flushes nothing."""

    flushes = []
    real_flush = hvm.flush_partial
    monkeypatch.setattr(hvm, "flush_partial", lambda *pair: flushes.append(1) or real_flush(*pair))
    row_partial = _partial()
    carry = row_partial.bind(_carry(*_totals()))
    total_y, total_ctf = _totals()
    for _ in range(n_blocks):
        carry, total_y, total_ctf = row_partial.block_done(carry, total_y, total_ctf, block_rows=BLOCK_ROWS)
    carry, total_y, total_ctf = row_partial.flush(carry, total_y, total_ctf)
    assert len(flushes) == expected == -(-n_blocks // hvm.blocks_per_flush(BLOCK_ROWS))


def test_the_switch_is_off_by_default_and_then_nothing_is_allocated(monkeypatch):
    """Unset or 0: the pass gets no partial, so the row-block loops take the direct path with no extra pair."""

    def no_allocation(*args, **kwargs):
        raise AssertionError("a partial pair was allocated with the switch off")

    monkeypatch.setattr(hvm, "RowPartial", no_allocation)
    assert not ScoringOverrides().exact_float32_accumulation
    assert hvm.allocate_row_partial(1000, np.complex64, np.float32, exact_float32_accumulation=False) is None


def test_the_switch_changes_the_accumulators_where_the_loss_shows(monkeypatch):
    """Off and on must differ on rows built to show the loss; identical results would mean the switch does nothing."""

    weights = _weights()
    reference = float(weights.astype(np.float64).sum())
    results = {}
    for value in ("0", "1"):
        row_partial = hvm.allocate_row_partial(  # what a pass asks, once
            VOLUME, jnp.complex64, jnp.float32, exact_float32_accumulation=value == "1"
        )
        assert (row_partial is None) == (value == "0")
        results[value] = float(np.asarray(_stage_driver(monkeypatch, weights, row_partial)[1])[0])
    assert results["0"] != results["1"]
    assert abs(1.0 - results["0"] / reference) > 1e-3  # off: one accumulator, the production path
    assert abs(1.0 - results["1"] / reference) < 1e-6  # on: the exact float32 path


def test_pairs_above_the_size_rule_get_no_partial(caplog, monkeypatch):
    """The rule reads the pair's size alone: 4 GiB and below get a partial, above it the pass accumulates directly."""

    import logging

    pair_itemsize = np.dtype(np.complex64).itemsize + np.dtype(np.float32).itemsize
    assert pair_itemsize == 12
    assert isinstance(
        hvm.allocate_row_partial(1000, np.complex64, np.float32, exact_float32_accumulation=True), hvm.RowPartial
    )
    too_large = hvm.ROW_PARTIAL_MAX_PAIR_BYTES // pair_itemsize + 1
    with caplog.at_level(logging.INFO, logger="relax.reconstruction.half_volume_mstep"):
        assert hvm.allocate_row_partial(too_large, np.complex64, np.float32, exact_float32_accumulation=True) is None
    assert "accumulated directly" in caplog.text
    # Box 256 at padding 2 (515 x 515 x 258 half grid) is 0.76 GiB a pair; box 800 (1603 x 1603 x 802) is 23 GiB.
    assert 515 * 515 * 258 * pair_itemsize < hvm.ROW_PARTIAL_MAX_PAIR_BYTES < 1603 * 1603 * 802 * pair_itemsize


def test_the_traced_flush_loop_flushes_at_the_same_blocks_as_the_host_one():
    """``flushed_block_loop`` (the tilt program's form) gives the NumPy partials-of-32,768-rows sum, bit for bit."""

    weights = _weights(1 << 20)
    _, flushed32 = _float32_references(weights, hvm.ROW_PARTIAL_FLUSH_ROWS)
    rows = jnp.asarray(weights)

    def block(index, carry):
        return _add_rows(carry, jax.lax.dynamic_slice_in_dim(rows, index * BLOCK_ROWS, BLOCK_ROWS))

    carry, total_y, total_ctf = jax.jit(
        lambda carry, total_y, total_ctf: hvm.flushed_block_loop(
            jnp.int32(weights.size // BLOCK_ROWS),
            block,
            carry,
            total_y,
            total_ctf,
            flush_blocks=hvm.blocks_per_flush(BLOCK_ROWS),
            get_pair=lambda c: (c.Ft_y, c.Ft_ctf),
            set_pair=lambda c, y, ctf: c._replace(Ft_y=y, Ft_ctf=ctf),
        )
    )(_carry(*_totals()), *_totals())
    assert np.asarray(total_ctf)[0] == flushed32
    assert not np.any(np.asarray(carry.Ft_ctf)) and not np.any(np.asarray(carry.Ft_y)), "the partial is left zeroed"


@pytest.mark.parametrize("merge_row_partials", [False, True])
def test_tilt_mstep_program_flushes_its_slots_through_the_partial(monkeypatch, merge_row_partials):
    """The tilt M-step with the partial pair adds the same rows as without it, and returns the pair zeroed.

    Row weights are small integers, so every order of addition is exact and the two results are equal bit
    for bit; ``mstep_block_rows`` is the flush size, so each block is flushed on its own.
    """

    units, rows, slots_n, block_rows = 4, 2 * hvm.ROW_PARTIAL_FLUSH_ROWS, 3, hvm.ROW_PARTIAL_FLUSH_ROWS
    row_weight = jnp.asarray(np.random.default_rng(3).integers(1, 5, size=rows), jnp.float32)

    def translate(window_rows, valid, angles, rect_indices, exact_positions, *, image_shape):
        return window_rows, window_rows

    def block_at(start, blocks, operands, tables, carry, *, spec):
        block = jax.lax.dynamic_slice_in_dim(row_weight, start, block_rows)
        if spec.presum_adjoint:  # the row-sum merge: each row's partial lands on its own row of [C_R, recon]
            ids = start + jnp.arange(block_rows)
            return carry._replace(
                Ft_y=carry.Ft_y.at[ids, 0].add(block.astype(carry.Ft_y.dtype)),
                Ft_ctf=carry.Ft_ctf.at[ids, 0].add(block),
            )
        return carry._replace(Ft_y=carry.Ft_y.at[0].add(jnp.sum(block)), Ft_ctf=carry.Ft_ctf.at[0].add(jnp.sum(block)))

    def backproject_block_rows(summed, ctf_probs, rotations, Ft_y, Ft_ctf, *, tables, spec):
        return Ft_y.at[0].add(jnp.sum(summed)), Ft_ctf.at[0].add(jnp.sum(ctf_probs))

    monkeypatch.setattr(resident_tilt_mstep, "_slot_view_translate", translate)
    monkeypatch.setattr(resident_mstep_blocks, "resident_mstep_block_at", block_at)
    monkeypatch.setattr(resident_mstep_blocks, "backproject_block_rows", backproject_block_rows)
    monkeypatch.setattr(resident_mstep_blocks, "cached_block_projections", lambda tables, ids: (None, None, ids))
    monkeypatch.setattr(resident_block_programs, "add_chunk_wavg_image_power", lambda carry, *a, **k: carry)
    monkeypatch.setattr(
        resident_mstep_blocks,
        "initial_mstep_carry",
        lambda Ft_y, Ft_ctf, operands, tables, *, spec: resident_chunk_records.ChunkMstepCarry(
            Ft_y=Ft_y,
            Ft_ctf=Ft_ctf,
            wavg_triplet_pixels=jnp.zeros((units, 1, 3), jnp.float32),
            noise_shells=jnp.zeros(4, jnp.float64),
            a2_per_image=jnp.zeros(units, jnp.float32),
            xa_per_image=jnp.zeros(units, jnp.float32),
        ),
    )
    operands = resident_chunk_records.ChunkStageOperands(
        **{
            name: (jnp.ones((units, 4), jnp.float32) if name in resident_tilt_mstep._SLOT_VIEW_FIELDS else None)
            for name in resident_chunk_records.ChunkStageOperands._fields
        }
    )
    tables = resident_chunk_records.ChunkStageTables(**{name: None for name in resident_chunk_records.ChunkStageTables._fields})

    @dataclasses.dataclass(frozen=True)
    class Spec:
        mstep_block_rows: int = block_rows
        presum_adjoint: bool = merge_row_partials
        n_recon_pixels: int = 1

    slots = resident_tilt_mstep.SlotMstepTables(
        slot=jnp.arange(slots_n, dtype=jnp.int32),
        safe_images=jnp.zeros((slots_n, units), jnp.int32),
        valid_images=jnp.ones((slots_n, units), bool),
        targets=jnp.zeros((slots_n, units), jnp.int32),
        order=jnp.tile(jnp.arange(rows, dtype=jnp.int32), (slots_n, 1)),
        active=jnp.ones((slots_n, rows), bool),
        units=jnp.zeros((slots_n, rows), jnp.int32),
        n_blocks=jnp.full(slots_n, rows // block_rows, jnp.int32),
        angles=jnp.ones((slots_n, 1, units, 1, 2), jnp.float32),
    )

    def run(with_partial):
        mstep = resident_mstep_blocks.initial_mstep_carry(*_totals(), None, None, spec=None)
        pair = ((jnp.zeros(VOLUME, jnp.complex64), jnp.zeros(VOLUME, jnp.float32)),) if with_partial else ()
        return resident_tilt_mstep.tilt_mstep_program(
            (mstep, *pair),
            slots,
            operands,
            tables,
            jnp.ones((units, 4), jnp.complex64),
            jnp.ones((rows, 1), jnp.float32),
            jnp.zeros((1, 1), jnp.int32),
            jnp.ones((1, 1), bool),
            jnp.arange(4, dtype=jnp.int32),
            jnp.arange(2, dtype=jnp.int32),
            slot_spec=Spec(),
            image_shape=(8, 8),
        )

    (direct,) = run(False)
    flushed, pair = run(True)
    expected = slots_n * float(np.asarray(row_weight, np.float64).sum())
    assert float(np.asarray(direct.Ft_ctf)[0]) == expected
    np.testing.assert_array_equal(np.asarray(direct.Ft_ctf), np.asarray(flushed.Ft_ctf))
    np.testing.assert_array_equal(np.asarray(direct.Ft_y), np.asarray(flushed.Ft_y))
    assert not np.any(np.asarray(pair[0])) and not np.any(np.asarray(pair[1]))
