"""P3-A: the resident chunk loop's glue, as programs instead of eager ops.

Ticket: ``em_parity_tickets_20260918/P3A_resident_driver_glue.md``.

The chunk loop used to issue about 780 eager JAX dispatches per chunk at the
early state and 324 at hp3, almost all of them the M-step block's own slices,
gathers and scalar operands, plus three ``jax.eval_shape`` probes per chunk
that existed only to learn the carry's dtypes. What is CPU-testable, and is
tested here, is the part that does not need the custom CUDA library:

* the carry dtypes are host arithmetic on the operand dtypes, and they agree
  with the ``jax.eval_shape`` probe of the real block stages;
* the probe check fails when the arithmetic is wrong, so it is a real guard;
* the block program's ``dynamic_slice`` picks the rows the Python slice picked,
  for every row capacity on the ladder;
* the device row offsets are made once per value, not once per block.

The bitwise comparison of the whole block body against the loose dispatch
needs a GPU and the custom library; it is the matched hp3/early pair in the
ticket's report, and the driver's own GPU test exercises the default path.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers.float_compare import assert_matches

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from relax.fine_pass import resident_chunk_records, resident_mstep_blocks
from relax.fine_pass import resident_pass2 as rp

pytestmark = pytest.mark.unit


def _spec(**overrides):
    kwargs = dict(
        row_capacity=8192,
        image_capacity=32,
        n_fine_trans=21,
        n_score_pixels=64,
        n_recon_pixels=97,
        n_rect=64,
        mstep_block_rows=4096,
        adaptive_fraction=0.999,
        current_size=16,
        mstep_current_size=16,
        mstep_max_r=8.0,
        image_shape=(16, 16),
        recon_volume_shape=(16, 16, 16),
        max_adjoint_block_bytes=1 << 30,
        stats_config=resident_mstep_blocks._MstepOnlyStatsConfig(n_shells=9),
        use_rfloat_ctf_wavg=False,
        use_translate_sum_kernel=False,
        bpref_recon_operand=False,
        glue_jit=True,
    )
    kwargs.update(overrides)
    return resident_chunk_records.ChunkProgramSpec(**kwargs)


def _operands(spec, *, tile_dtype=jnp.complex64, ctf_dtype=jnp.float32):
    """The three operand fields the carry dtypes are derived from."""

    shape = (spec.image_capacity, spec.n_fine_trans, spec.n_recon_pixels)
    tiles = jnp.zeros(shape, dtype=tile_dtype)
    return resident_chunk_records.ChunkStageOperands(
        score_input=None,
        corr_img_score=None,
        highres_xi2_half=None,
        translation_prior=None,
        shifted_recon=tiles,
        shifted_noise=tiles,
        recon_image=None,
        recon_weight=None,
        noise_image=None,
        ctf2_over_nv_recon=jnp.zeros(
            (spec.image_capacity, spec.n_recon_pixels), dtype=ctf_dtype
        ),
        direct_ctf_rfloat_recon=None,
        image_power_shells=None,
        relion_norm_high_shell=None,
        raw_translated_wavg_rectangle=None,
        raw_translated_wavg_for_atomic=None,
        scale=None,
        group_ids=None,
        translation_sqdist_ang=None,
    )


def _tables(spec, *, noise_dtype=jnp.float32):
    fields = {name: None for name in resident_chunk_records.ChunkStageTables._fields}
    fields["noise_variance_for_noise"] = jnp.ones(
        (spec.n_recon_pixels,), dtype=noise_dtype
    )
    fields["shell_indices_noise"] = jnp.zeros((spec.n_recon_pixels,), dtype=jnp.int32)
    return resident_chunk_records.ChunkStageTables(**fields)


@pytest.mark.parametrize("proj_dtype", [jnp.complex64, jnp.complex128])
@pytest.mark.parametrize("noise_dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("use_kernel", [False, True])
def test_static_carry_dtypes_match_the_traced_block_stages(
    proj_dtype, noise_dtype, use_kernel
):
    """The promotion arithmetic answers what ``jax.eval_shape`` answered."""

    spec = _spec(use_translate_sum_kernel=use_kernel)
    proj_abs2_dtype = jnp.zeros((), dtype=proj_dtype).real.dtype
    operands = _operands(spec)
    tables = _tables(spec, noise_dtype=noise_dtype)
    dtypes = resident_mstep_blocks._mstep_block_operand_dtypes(
        operands,
        tables,
        spec=spec,
        projection_dtypes=(proj_dtype, proj_abs2_dtype),
    )
    # Raises if the arithmetic and the probe disagree.
    resident_mstep_blocks._check_mstep_carry_avals(tables, spec=spec, dtypes=dtypes)

    shells, a2, xa = resident_mstep_blocks._probe_mstep_block_output_avals(
        tables, spec=spec, dtypes=dtypes
    )
    assert shells.shape == (spec.stats_config.n_shells,)
    assert a2.shape == xa.shape == (spec.image_capacity,)
    assert jnp.dtype(shells.dtype) == jnp.dtype(jnp.float64)
    assert jnp.dtype(a2.dtype) == dtypes["a2"]
    assert jnp.dtype(xa.dtype) == dtypes["xa"]


def test_initial_mstep_carry_has_the_probed_shapes_without_a_probe(monkeypatch):
    """The carry the loop starts from, built without tracing anything."""

    calls = []
    original = resident_mstep_blocks._probe_mstep_block_output_avals
    monkeypatch.setattr(
        resident_mstep_blocks,
        "_probe_mstep_block_output_avals",
        lambda *a, **k: (calls.append(1), original(*a, **k))[1],
    )

    spec = _spec()
    operands = _operands(spec)
    tables = _tables(spec)
    Ft_y = jnp.zeros((4, 4, 4), dtype=jnp.complex64)
    Ft_ctf = jnp.zeros((4, 4, 4), dtype=jnp.float32)
    carry = resident_mstep_blocks.initial_mstep_carry(
        Ft_y,
        Ft_ctf,
        operands,
        tables,
        spec=spec,
        projection_dtypes=(jnp.complex64, jnp.float32),
    )
    assert calls == []
    assert carry.Ft_y is Ft_y and carry.Ft_ctf is Ft_ctf
    assert carry.wavg_triplet_pixels.shape == (spec.image_capacity, spec.n_rect, 3)
    assert carry.noise_shells.shape == (spec.stats_config.n_shells,)
    assert carry.noise_shells.dtype == jnp.float64
    assert carry.a2_per_image.shape == carry.xa_per_image.shape == (spec.image_capacity,)
    assert not np.any(np.asarray(carry.noise_shells))
    assert not np.any(np.asarray(carry.a2_per_image))
    assert not np.any(np.asarray(carry.wavg_triplet_pixels))


def test_the_carry_aval_check_rejects_a_wrong_dtype():
    """The probe check is a real guard, not a formality."""

    spec = _spec()
    operands = _operands(spec)
    tables = _tables(spec)
    dtypes = resident_mstep_blocks._mstep_block_operand_dtypes(
        operands, tables, spec=spec, projection_dtypes=(jnp.complex64, jnp.float32)
    )
    wrong = dict(dtypes)
    wrong["a2"] = jnp.dtype(jnp.float64)
    with pytest.raises(AssertionError, match="disagree with the traced block stages"):
        resident_mstep_blocks._check_mstep_carry_avals(tables, spec=spec, dtypes=wrong)


@pytest.mark.parametrize("row_capacity", rp._DEFAULT_ROW_CAPACITY_LADDER)
def test_block_dynamic_slice_picks_the_python_slice_rows(row_capacity):
    """Every capacity is a whole number of blocks, so the slice never clamps."""

    block_rows = 4096
    assert row_capacity % block_rows == 0
    values = jnp.arange(row_capacity, dtype=jnp.int32)
    for start in range(0, row_capacity, block_rows):
        taken = jax.lax.dynamic_slice_in_dim(
            values, resident_chunk_records.device_int32(start), block_rows, axis=0
        )
        expected = values[start : start + block_rows]
        assert_matches(np.asarray(taken), np.asarray(expected))


def test_device_row_offsets_are_made_once_per_value():
    """The block offsets must not put an eager dispatch back per block."""

    resident_chunk_records._DEVICE_INT32_CACHE.clear()
    first = resident_chunk_records.device_int32(4096)
    second = resident_chunk_records.device_int32(4096)
    assert first is second
    assert int(first) == 4096
    assert first.dtype == jnp.int32
    assert resident_chunk_records.device_int32(8192) is not first


def test_the_block_program_is_keyed_on_the_capacity_class_not_the_offset():
    """One program serves every block: the offset is an operand, not a key.

    Counted by tracing, not by a private cache attribute: the body runs once
    per retrace, so a second offset that traces again would show up here.
    """

    spec = _spec()
    traces = []

    @jax.jit
    def slice_block(start, values):
        traces.append(1)
        return jax.lax.dynamic_slice_in_dim(
            values, start, spec.mstep_block_rows, axis=0
        )

    values = jnp.arange(spec.row_capacity, dtype=jnp.int32)
    first = slice_block(resident_chunk_records.device_int32(0), values)
    assert len(traces) == 1
    second = slice_block(resident_chunk_records.device_int32(spec.mstep_block_rows), values)
    assert len(traces) == 1, "a second block offset must not retrace the program"
    assert_matches(
        np.asarray(first), np.asarray(values[: spec.mstep_block_rows])
    )
    assert_matches(
        np.asarray(second),
        np.asarray(values[spec.mstep_block_rows : 2 * spec.mstep_block_rows]),
    )
