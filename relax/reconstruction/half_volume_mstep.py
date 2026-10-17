"""Shared helpers for RELION-style half-volume M-step accumulation."""

from __future__ import annotations

import logging
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import recovar.core.fourier_transform_utils as fourier_transform_utils

from relax.runtime.engine_tuning import XHalfLargeGrid


def relion_x_half_mstep_accumulator_dtypes(dataset_dtype, *, use_relion_x_half_mstep: bool, mstep_double: bool):
    """Return ``(Ft_y dtype, Ft_ctf dtype)`` for an M-step accumulator.

    RELION stores its ``BackProjector`` accumulators in single precision. ``mstep_double``
    (``ScoringOverrides.x_half_mstep_double``) widens the RELION x-half accumulators to double, a diagnostic
    toggle for isolating boundary-shell differences, not the default production path.
    """

    base_dtype = np.dtype(dataset_dtype)
    if use_relion_x_half_mstep:
        if mstep_double:
            return np.dtype(np.complex128), np.dtype(np.float64)
        return np.dtype(np.complex64), np.dtype(np.float32)
    return base_dtype, base_dtype


# Exact float32 accumulation of the per-row backprojection, the production path since relax#92 (user 2026-10-16;
# until then an opt-in diagnostic, relax#65). Stock GPU RELION 5.0.1 scatters every row into one float32 pair and
# loses low-resolution weight; RELION builds whose device backprojection accumulates in double do not.
# A float32 accumulator loses an addend below half an ulp of the running total. With every row of a pass
# scattered into one pair, the low shells of a soft pass lost 1e-2 of their weight (relax#65: 9.9e-3 and 1.5e-2
# at shell 0 of a tilt-series VDAM iteration, 1.6e-2 at shell 0 of a Refine3D iteration 1 with 11.2M rows).
# Rows are therefore scattered into a zeroed partial pair that is added into the totals every
# ``ROW_PARTIAL_FLUSH_ROWS`` backprojected rows (a tilt row counts once per tilt image): at 32,768 rows the
# weight missing at shell 0 was <= 2e-7 on the tilt case and <= 1.1e-5 on the Refine3D one, and the float32
# total of the partials equals their float64 total to 5e-8 per shell.
# Formulation and measurements: docs/math/relion_refinement_algorithm.md, "Exact float32 accumulation".
ROW_PARTIAL_FLUSH_ROWS = 32768
# The partial is a second accumulator pair. Above this size the pass accumulates directly, as stock GPU RELION does
# (a current size above about 440 at padding 2; 22.9 GiB at box 800, full size). A shape rule, so that numerics never
# depend on the card.
ROW_PARTIAL_MAX_PAIR_BYTES = 4 * 1024**3


def blocks_per_flush(block_rows: int) -> int:
    """How many M-step row blocks fill one partial: a static count, so flush points follow the row order alone."""

    return max(1, ROW_PARTIAL_FLUSH_ROWS // max(int(block_rows), 1))


@partial(jax.jit, donate_argnums=(0, 1, 2, 3))
def flush_partial(total_y, total_ctf, partial_y, partial_ctf):
    """Add a partial pair into its totals and zero it: ``(total_y, total_ctf, partial_y, partial_ctf)``."""

    return total_y + partial_y, total_ctf + partial_ctf, jnp.zeros_like(partial_y), jnp.zeros_like(partial_ctf)


def flushed_block_loop(n_blocks, block, state, total_y, total_ctf, *, flush_blocks: int, get_pair, set_pair):
    """``lax.fori_loop`` over ``n_blocks`` row blocks whose pair is a partial, flushed every ``flush_blocks``.

    ``block(index, state)`` scatters into the pair ``get_pair(state)``, which enters zeroed. After every
    ``flush_blocks`` blocks, and after the last one, the pair is added into ``(total_y, total_ctf)`` and
    zeroed (``set_pair(state, y, ctf)``). Returns ``(state, total_y, total_ctf)`` with the pair zero again.
    The traced counterpart of :class:`RowPartial`; both flush at the same blocks.
    """

    def group(index, carry):
        state, total_y, total_ctf = carry
        start = index * flush_blocks
        state = jax.lax.fori_loop(start, jnp.minimum(start + flush_blocks, n_blocks), block, state)
        partial_y, partial_ctf = get_pair(state)
        state = set_pair(state, jnp.zeros_like(partial_y), jnp.zeros_like(partial_ctf))
        return state, total_y + partial_y, total_ctf + partial_ctf

    return jax.lax.fori_loop(0, (n_blocks + (flush_blocks - 1)) // flush_blocks, group, (state, total_y, total_ctf))


class RowPartial:
    """The partial accumulator pair of one pass: zero between flushes, shared by the pass's slots.

    The pass allocates it with its totals (before it reads free device memory for the chunk budget) and hands
    it to the row-block loops. A loop scatters its blocks into ``y``/``ctf`` in place of the totals, calls
    :meth:`block_done` after each block and :meth:`flush` when it leaves a slot, so a partial never outlives
    a slot's blocks of a chunk.
    """

    def __init__(self, volume_size: int, y_dtype, ctf_dtype):
        self.y = jnp.zeros(int(volume_size), dtype=y_dtype)
        self.ctf = jnp.zeros(int(volume_size), dtype=ctf_dtype)
        self._blocks = 0

    def bind(self, carry):
        """``carry`` with its pair replaced by the (zero) partial."""

        return carry._replace(Ft_y=self.y, Ft_ctf=self.ctf)

    def block_done(self, carry, total_y, total_ctf, *, block_rows: int):
        """After one block of ``block_rows`` rows: flush when the partial is full. Returns ``(carry, totals)``."""

        self._blocks += 1
        if self._blocks < blocks_per_flush(block_rows):
            return carry, total_y, total_ctf
        return self.flush(carry, total_y, total_ctf)

    def flush(self, carry, total_y, total_ctf):
        """Add the carry's pair into the totals; the carry gets the zeroed partial back."""

        if self._blocks == 0:
            return carry, total_y, total_ctf
        self._blocks = 0
        total_y, total_ctf, self.y, self.ctf = flush_partial(total_y, total_ctf, carry.Ft_y, carry.Ft_ctf)
        return self.bind(carry), total_y, total_ctf


def allocate_row_partial(
    volume_size: int, y_dtype, ctf_dtype, *, stock_relion_direct_accumulation: bool
) -> RowPartial | None:
    """The pass's :class:`RowPartial`, or ``None`` (direct accumulation) for a pair above ``ROW_PARTIAL_MAX_PAIR_BYTES``
    or under ``stock_relion_direct_accumulation`` (``ScoringOverrides``; the stock-fixture parity gates only).

    A pass whose rows are backprojected one by one asks once, before it sizes its chunks, which then count the pair.
    Above the size rule (a current size over about 440 at padding 2) the pass accumulates directly, as stock GPU
    RELION does, and what it loses there has not been measured. With direct accumulation, one pair per pass lost
    8.5e-3 of the weight at shell 0 in the final pass of a 100k-image box-256 refinement. See "Exact float32
    accumulation" in ``docs/math/relion_refinement_algorithm.md``.
    """

    if stock_relion_direct_accumulation:
        return None
    pair_bytes = int(volume_size) * (np.dtype(y_dtype).itemsize + np.dtype(ctf_dtype).itemsize)
    if pair_bytes > ROW_PARTIAL_MAX_PAIR_BYTES:
        logging.getLogger(__name__).info(
            "M-step accumulators of %.2f GiB a pair exceed %.0f GiB, rows are accumulated directly, without partials",
            pair_bytes / float(1024**3),
            ROW_PARTIAL_MAX_PAIR_BYTES / float(1024**3),
        )
        return None
    return RowPartial(volume_size, y_dtype, ctf_dtype)


def half_volume_accumulator_shape(recon_volume_shape):
    """Return the packed half-volume accumulator shape."""

    return fourier_transform_utils.volume_shape_to_half_volume_shape(recon_volume_shape)


def _relion_backprojector_r_max(volume_shape, current_size=None):
    """Return RELION ``BackProjector`` support radius for ``initZeros``."""

    volume_shape = tuple(int(v) for v in volume_shape)
    if len(volume_shape) != 3 or len(set(volume_shape)) != 1:
        raise ValueError(f"RELION BackProjector requires a cubic 3-D volume_shape, got {volume_shape}")
    ori_half = volume_shape[0] // 2
    if current_size is None or int(current_size) < 0:
        r_max = ori_half
    else:
        r_max = int(current_size) // 2
    return min(int(r_max), int(ori_half))


def relion_backprojector_volume_shape(volume_shape, padding_factor, current_size=None):
    """Return RELION's odd BPref accumulator grid for ``BackProjector::initZeros``."""

    if padding_factor <= 0:
        raise ValueError(f"padding_factor must be positive, got {padding_factor!r}")
    r_max = _relion_backprojector_r_max(volume_shape, current_size=current_size)
    pad_size = 2 * (int(padding_factor * float(r_max) + 0.5) + 1) + 1
    return (int(pad_size), int(pad_size), int(pad_size))


def enforce_half_volume_x0(
    Ft_y,
    Ft_ctf,
    recon_volume_shape,
    *,
    logger: logging.Logger,
    label: str,
    large_grid: XHalfLargeGrid,
    force_host: bool = False,
):
    """Apply RELION x=0 Hermitian-plane enforcement to half-volume accumulators (on the host when ``force_host``
    or ``large_grid.host_x0`` is on)."""

    logger.info("%s M-step: enforcing RELION half-volume x=0 Hermitian plane", label)
    full_voxels = int(np.prod(recon_volume_shape))
    if force_host or large_grid.host_x0.on(full_voxels):
        logger.info(
            "%s M-step: using host x=0 Hermitian enforcement for large RELION half-volume "
            "accumulators (shape=%s, full_voxels=%d)",
            label,
            tuple(recon_volume_shape),
            full_voxels,
        )
        return (
            enforce_relion_half_volume_x0_hermitian_host(Ft_y, recon_volume_shape),
            enforce_relion_half_volume_x0_hermitian_host(Ft_ctf, recon_volume_shape),
        )
    return _enforce_relion_half_volume_x0_hermitian_pair_jit(
        Ft_y, Ft_ctf, tuple(int(value) for value in recon_volume_shape)
    )


def _validated_relion_right_operators(symmetry_label: str, symmetry_operators):
    """Return identity-inclusive RELION right operators in source order."""

    from relax.sampling.symmetry import parse_rotational_symmetry, rotational_operators

    parsed = parse_rotational_symmetry(symmetry_label)
    if symmetry_operators is None:
        operators = rotational_operators(parsed.label, dtype=np.float64)
    else:
        operators = np.asarray(symmetry_operators, dtype=np.float64)
    expected_shape = (parsed.operator_count, 3, 3)
    if operators.shape != expected_shape:
        raise ValueError(
            f"RELION {parsed.label} requires right operators with shape {expected_shape}, "
            f"got {operators.shape}"
        )
    if not np.all(np.isfinite(operators)):
        raise ValueError(f"RELION {parsed.label} right operators must be finite")
    if not np.array_equal(operators[0], np.eye(3, dtype=np.float64)):
        raise ValueError(f"RELION {parsed.label} right operators must contain exact identity first")
    if not np.allclose(
        operators @ np.swapaxes(operators, -1, -2),
        np.eye(3, dtype=np.float64)[None, :, :],
        rtol=0.0,
        atol=2e-7,
    ):
        raise ValueError(f"RELION {parsed.label} right operators must be orthogonal")
    if not np.allclose(np.linalg.det(operators), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError(f"RELION {parsed.label} right operators must be proper rotations")
    return parsed.label, np.ascontiguousarray(operators)


def finalize_half_volume_bpref(
    Ft_y,
    Ft_ctf,
    recon_volume_shape,
    *,
    logger: logging.Logger,
    label: str,
    symmetry_label: str = "C1",
    symmetry_operators=None,
    relion_x_half: bool,
    force_host: bool | None = None,
    symmetry_chunk_voxels: int,
    large_grid: XHalfLargeGrid,
):
    """Finalize half-volume BPref accumulators before layout conversion (``symmetry_chunk_voxels``: the host
    point-group finalizer's chunk, ``EngineTuning.bpref_symmetry_chunk_voxels``; ``large_grid``:
    ``EngineTuning.x_half_large_grid``).

    C1 deliberately calls the historical x=0 helper verbatim.  This keeps
    existing C1 output bitwise stable.  Non-C1 currently requires RELION's
    odd ``(z, y, xhalf)`` BPref layout and uses a streamed CUDA finalizer that
    fuses x=0 Hermitian enforcement with ordered point-group accumulation.
    Large float32 accumulators use bounded output ranges copied to host;
    the complex input stays in place and is never split into device copies.
    """

    if not isinstance(symmetry_label, str) or not symmetry_label.strip():
        raise ValueError("symmetry_label must be a nonempty RELION point-group label")
    normalized_label = symmetry_label.strip().upper()
    if normalized_label == "C1":
        if symmetry_operators is not None:
            identity = np.asarray(symmetry_operators)
            if identity.shape != (1, 3, 3) or not np.array_equal(identity[0], np.eye(3)):
                raise ValueError("C1 symmetry_operators must contain exact identity only")
        return enforce_half_volume_x0(
            Ft_y,
            Ft_ctf,
            recon_volume_shape,
            logger=logger,
            label=label,
            large_grid=large_grid,
            **({"force_host": force_host} if force_host is not None else {}),
        )

    canonical_label, right_operators = _validated_relion_right_operators(
        normalized_label,
        symmetry_operators,
    )
    if not relion_x_half:
        raise NotImplementedError(
            f"{label} requested {canonical_label} point-group symmetry for a native half-volume "
            "accumulator; non-C1 reconstruction symmetry requires RELION x-half BPref storage"
        )

    recon_volume_shape = tuple(int(value) for value in recon_volume_shape)
    if len(recon_volume_shape) != 3 or len(set(recon_volume_shape)) != 1:
        raise ValueError(
            "RELION point-group BPref symmetry requires a cubic accumulator, "
            f"got {recon_volume_shape}"
        )
    if any(value <= 0 or value % 2 == 0 for value in recon_volume_shape):
        raise ValueError(
            "RELION point-group BPref symmetry requires an odd positive accumulator grid, "
            f"got {recon_volume_shape}"
        )

    data = jnp.asarray(Ft_y).reshape(-1)
    weight = jnp.asarray(Ft_ctf).reshape(-1)
    if data.dtype == jnp.dtype(jnp.complex64):
        operator_dtype = np.float32
        expected_weight_dtype = jnp.dtype(jnp.float32)
    elif data.dtype == jnp.dtype(jnp.complex128):
        operator_dtype = np.float64
        expected_weight_dtype = jnp.dtype(jnp.float64)
    else:
        raise TypeError(
            "RELION point-group BPref data must be complex64 or complex128, "
            f"got {data.dtype}"
        )
    if weight.dtype != expected_weight_dtype:
        raise TypeError(
            "RELION point-group BPref weight precision must match the data component, "
            f"got data={data.dtype}, weight={weight.dtype}"
        )

    support_radius = recon_volume_shape[0] // 2 - 1
    logger.info(
        "%s M-step: enforcing RELION x=0 and %s point-group symmetry on CUDA "
        "(operators=%d, support_radius=%d)",
        label,
        canonical_label,
        right_operators.shape[0],
        support_radius,
    )
    from relax.cuda import backprojection as cuda_backprojection

    use_host = (
        (data.dtype == jnp.dtype(jnp.complex64)
         and large_grid.host_x0.on(int(np.prod(recon_volume_shape))))
        if force_host is None else bool(force_host)
    )
    finalize = (
        cuda_backprojection.relion_point_group_symmetrise_bpref_host
        if use_host else cuda_backprojection.relion_point_group_symmetrise_bpref
    )
    return finalize(
        data,
        weight,
        jnp.asarray(right_operators, dtype=operator_dtype),
        recon_volume_shape,
        support_radius,
        **({"chunk_voxels": symmetry_chunk_voxels} if use_host else {}),
    )


@partial(jax.jit, static_argnames=("recon_volume_shape",))
def _relion_x_half_volume_to_full_device(volume_flat, recon_volume_shape):
    relion_full = fourier_transform_utils.half_volume_to_full_volume(
        volume_flat,
        recon_volume_shape,
    ).reshape(recon_volume_shape)
    return relion_full.transpose(2, 1, 0).reshape(-1)


@partial(jax.jit, static_argnames=("recon_volume_shape",))
def _relion_x_half_volume_pair_to_full_device(Ft_y, Ft_ctf, recon_volume_shape):
    return (
        _relion_x_half_volume_to_full_device(Ft_y, recon_volume_shape),
        _relion_x_half_volume_to_full_device(Ft_ctf, recon_volume_shape),
    )


def relion_x_half_volume_to_full(volume_flat, recon_volume_shape, *, large_grid: XHalfLargeGrid, force_host: bool = False):
    """Expand a RELION-layout ``(z, y, xhalf)`` accumulator to RECOVAR full layout.

    RELION's BackProjector packs the Fourier x-axis and stores arrays in
    public order ``(z, y, xhalf)``. RECOVAR's public full accumulator order is
    ``(x, y, z)``, so expansion is a last-axis Hermitian unpack in RELION
    layout followed by a ``(2, 1, 0)`` transpose.
    """

    if force_host or large_grid.full_host.on(int(np.prod(recon_volume_shape))):
        logging.getLogger(__name__).info(
            "RELION x-half M-step: expanding large accumulator to public full layout on host "
            "(shape=%s, full_voxels=%d)",
            tuple(recon_volume_shape),
            int(np.prod(recon_volume_shape)),
        )
        return _relion_x_half_volume_to_full_host(volume_flat, recon_volume_shape).reshape(-1)

    return _relion_x_half_volume_to_full_device(volume_flat, tuple(int(v) for v in recon_volume_shape))


def relion_x_half_volume_to_native_half(volume_flat, recon_volume_shape):
    """Repack RELION ``(z, y, xhalf)`` to RECOVAR native ``(x, y, zhalf)``."""

    return _relion_x_half_volume_to_native_half_host(volume_flat, recon_volume_shape).reshape(-1)


def relion_x_half_volume_to_public_layout(
    volume_flat, recon_volume_shape, *, large_grid: XHalfLargeGrid, force_host: bool = False,
):
    """Convert RELION x-half to the downstream public accumulator layout.

    Normal grids keep the historical full-volume contract.  Very large grids
    are repacked into RECOVAR's native packed half-volume layout, which the
    RELION FSC/tau2/reconstruction path already accepts by shape inference.
    """

    if large_grid.to_native_half.on(int(np.prod(recon_volume_shape))):
        logging.getLogger(__name__).info(
            "RELION x-half M-step: repacking large accumulator to RECOVAR native half layout "
            "(shape=%s, full_voxels=%d)",
            tuple(recon_volume_shape),
            int(np.prod(recon_volume_shape)),
        )
        return relion_x_half_volume_to_native_half(volume_flat, recon_volume_shape)
    return relion_x_half_volume_to_full(volume_flat, recon_volume_shape, large_grid=large_grid, force_host=force_host)


def relion_x_half_accumulators_to_public_layout(
    Ft_y,
    Ft_ctf,
    recon_volume_shape,
    *,
    large_grid: XHalfLargeGrid,
    force_host: bool = False,
):
    """Convert RELION ``(z, y, xhalf)`` accumulators for downstream consumers."""

    full_voxels = int(np.prod(recon_volume_shape))
    if (
        force_host
        or large_grid.to_native_half.on(full_voxels)
        or large_grid.full_host.on(full_voxels)
    ):
        return (
            relion_x_half_volume_to_public_layout(Ft_y, recon_volume_shape, large_grid=large_grid, force_host=force_host),
            relion_x_half_volume_to_public_layout(
                Ft_ctf, recon_volume_shape, large_grid=large_grid, force_host=force_host,
            ),
        )
    # Both accumulators in one program: two per shape compiled at every new size.
    return _relion_x_half_volume_pair_to_full_device(Ft_y, Ft_ctf, tuple(int(v) for v in recon_volume_shape))


def crop_relion_x_half_accumulator(
    values,
    physical_volume_shape,
    logical_volume_shape,
):
    """Crop a centered physical BPref capacity to its logical odd cube.

    RELION's x-half layout is ``(z, y, x>=0)``.  The signed z/y axes are
    centered and therefore crop symmetrically; the packed x axis starts at
    zero and keeps its logical prefix.  No arithmetic is performed here.
    """

    physical_volume_shape = tuple(int(value) for value in physical_volume_shape)
    logical_volume_shape = tuple(int(value) for value in logical_volume_shape)
    if (
        len(physical_volume_shape) != 3
        or len(logical_volume_shape) != 3
        or len(set(physical_volume_shape)) != 1
        or len(set(logical_volume_shape)) != 1
        or logical_volume_shape[0] > physical_volume_shape[0]
        or physical_volume_shape[0] % 2 == 0
        or logical_volume_shape[0] % 2 == 0
    ):
        raise ValueError(
            "stable RELION BPref cropping requires nested odd cubic shapes, "
            f"got physical={physical_volume_shape}, logical={logical_volume_shape}"
        )
    physical_size = physical_volume_shape[0]
    logical_size = logical_volume_shape[0]
    physical_half_width = physical_size // 2 + 1
    logical_half_width = logical_size // 2 + 1
    expected_size = physical_size * physical_size * physical_half_width
    if int(values.size) != expected_size:
        raise ValueError(
            f"RELION x-half accumulator has {values.size} entries, expected {expected_size}"
        )
    start = (physical_size - logical_size) // 2
    grid = values.reshape((physical_size, physical_size, physical_half_width))
    return grid[
        start : start + logical_size,
        start : start + logical_size,
        :logical_half_width,
    ].reshape(-1)


def physical_bpref_finalize_applies(
    physical_volume_shape, logical_volume_shape, symmetry_label, *, large_grid: XHalfLargeGrid,
) -> bool:
    """Whether a stable-window pass can finalize its BPref at the physical cube and crop afterwards.

    C1 only (the x=0 enforcement is then the whole finalize), and only while
    neither cube takes the large-grid host or native-half branches, so the
    physical and logical paths run the same device programs.
    """

    if str(symmetry_label).strip().upper() != "C1":
        return False
    for shape in (physical_volume_shape, logical_volume_shape):
        voxels = int(np.prod(tuple(int(v) for v in shape)))
        if large_grid.host_x0.on(voxels) or large_grid.to_native_half.on(voxels):
            return False
    return True


def crop_public_full_volume(values, physical_volume_shape, logical_volume_shape):
    """Crop a centered public full ``(x, y, z)`` accumulator to its logical odd cube.

    The full-layout counterpart of :func:`crop_relion_x_half_accumulator`: the
    x=0 enforcement and the x-half expansion only pair voxels with their
    centered mirrors, so finalizing the physical cube and cropping it leaves
    the logical voxels as finalizing the cropped cube would. No arithmetic.
    """

    physical_size = int(tuple(physical_volume_shape)[0])
    logical_size = int(tuple(logical_volume_shape)[0])
    if (
        len(set(int(v) for v in physical_volume_shape)) != 1
        or len(set(int(v) for v in logical_volume_shape)) != 1
        or logical_size > physical_size
        or physical_size % 2 == 0
        or logical_size % 2 == 0
    ):
        raise ValueError(
            "stable RELION BPref cropping requires nested odd cubic shapes, "
            f"got physical={tuple(physical_volume_shape)}, logical={tuple(logical_volume_shape)}"
        )
    if int(values.size) != physical_size**3:
        raise ValueError(f"full accumulator has {values.size} entries, expected {physical_size**3}")
    return _crop_centered_cube(values, physical_size=physical_size, logical_size=logical_size)


@partial(jax.jit, static_argnames=("physical_size", "logical_size"))
def _crop_centered_cube(values, *, physical_size: int, logical_size: int):
    # One program per (physical, logical) pair; the eager slices compiled three.
    start = (physical_size - logical_size) // 2
    grid = values.reshape((physical_size, physical_size, physical_size))
    return grid[start : start + logical_size, start : start + logical_size, start : start + logical_size].reshape(-1)


# Elements of one repack slab (complex64: 0.25 GiB).
_REPACK_SLAB_ELEMENTS = 1 << 25


def _slabs(n_columns: int, column_elements: int):
    """Slices over ``n_columns`` whose slabs hold at most :data:`_REPACK_SLAB_ELEMENTS` elements (one column at least)."""

    step = max(1, _REPACK_SLAB_ELEMENTS // max(1, int(column_elements)))
    return [slice(start, start + step) for start in range(0, int(n_columns), step)]


def _relion_x_half_volume_to_native_half_host(volume_flat, recon_volume_shape):
    """Host implementation of the RELION x-half to RECOVAR native-half repack."""

    recon_volume_shape = tuple(int(v) for v in recon_volume_shape)
    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(recon_volume_shape)
    half_grid = np.asarray(jax.device_get(volume_flat)).reshape(half_shape)

    n0, n1, n2 = recon_volume_shape
    ic2 = n2 // 2
    packed_idx = np.asarray(
        fourier_transform_utils.get_real_fft_packed_last_axis_indices(n2),
        dtype=np.intp,
    )

    # Filled in slabs, straight into the output: the whole-array form held a gathered copy of the half and its
    # conjugate next to the output (+13 GiB per half at EMPIAR-10202's padded box, relax#39). Copies and
    # conjugations only, so the values are the whole-array form's.
    native_half = np.empty(half_shape, dtype=half_grid.dtype)
    for columns in _slabs(packed_idx.size, n0 * n1):
        native_half[packed_idx[columns], :, :] = half_grid[packed_idx, :, columns].transpose(2, 1, 0)

    if n2 % 2 == 0:
        redundant = np.arange(1, ic2, dtype=np.intp)
    else:
        redundant = np.arange(0, ic2, dtype=np.intp)
    if redundant.size:
        partner_i0 = ((n0 - (n0 % 2) - np.arange(n0)) % n0).astype(np.intp, copy=False)
        partner_i1 = ((n1 - (n1 % 2) - np.arange(n1)) % n1).astype(np.intp, copy=False)
        source_cols = ic2 - redundant
        for columns in _slabs(redundant.size, n0 * n1):
            conjugate_source = np.conj(half_grid[np.ix_(partner_i0[packed_idx], partner_i1, source_cols[columns])])
            native_half[redundant[columns], :, :] = conjugate_source.transpose(2, 1, 0)

    return np.ascontiguousarray(native_half)


def _relion_x_half_volume_to_full_host(volume_flat, recon_volume_shape):
    """Host implementation of RELION x-half to public full-layout expansion."""

    recon_volume_shape = tuple(int(v) for v in recon_volume_shape)
    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(recon_volume_shape)
    half_grid = np.asarray(jax.device_get(volume_flat)).reshape(half_shape)

    n0, n1, n2 = recon_volume_shape
    ic2 = n2 // 2
    packed_idx = np.asarray(
        fourier_transform_utils.get_real_fft_packed_last_axis_indices(n2),
        dtype=np.intp,
    )
    # Written in slabs through a transposed view of the output: the whole-array form held the RELION-layout
    # volume, a gathered copy of the half, its conjugate and the transposed copy at once (relax#39).
    public_full = np.zeros(recon_volume_shape[::-1], dtype=half_grid.dtype)
    relion_full = public_full.transpose(2, 1, 0)
    for columns in _slabs(packed_idx.size, n0 * n1):
        relion_full[:, :, packed_idx[columns]] = half_grid[:, :, columns]

    if n2 % 2 == 0:
        redundant = np.arange(1, ic2, dtype=np.intp)
    else:
        redundant = np.arange(0, ic2, dtype=np.intp)
    if redundant.size:
        partner_i0 = ((n0 - (n0 % 2) - np.arange(n0)) % n0).astype(np.intp, copy=False)
        partner_i1 = ((n1 - (n1 % 2) - np.arange(n1)) % n1).astype(np.intp, copy=False)
        source_cols = ic2 - redundant
        for columns in _slabs(redundant.size, n0 * n1):
            relion_full[:, :, redundant[columns]] = np.conj(
                half_grid[np.ix_(partner_i0, partner_i1, source_cols[columns])]
            )

    return public_full


def enforce_relion_half_volume_x0_hermitian(volume_flat, full_volume_shape):
    """Match RELION BackProjector::enforceHermitianSymmetry on x=0 plane."""

    return _enforce_relion_half_volume_x0_hermitian_jit(
        volume_flat,
        tuple(int(value) for value in full_volume_shape),
    )


@partial(jax.jit, static_argnames=("full_volume_shape",))
def _enforce_relion_half_volume_x0_hermitian_jit(volume_flat, full_volume_shape):
    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(full_volume_shape)
    vol = jnp.asarray(volume_flat).reshape(half_shape)
    n0, n1, _ = half_shape
    i0 = jnp.arange(n0, dtype=jnp.int32)
    i1 = jnp.arange(n1, dtype=jnp.int32)
    # RELION pairs logical Xmipp-origin coordinates (z, y) with (-z, -y).
    # In RECOVAR's centered array convention this is (N - (N % 2) - i) % N;
    # odd RELION BPref grids therefore use N-1-i, not the unshifted -i.
    p0 = (n0 - (n0 % 2) - i0) % n0
    p1 = (n1 - (n1 % 2) - i1) % n1
    plane = vol[:, :, 0]
    partner = jnp.conj(plane[p0[:, None], p1[None, :]])
    summed = plane + partner
    self_partner = (p0[:, None] == i0[:, None]) & (p1[None, :] == i1[None, :])
    plane = jnp.where(self_partner, plane, summed)
    return vol.at[:, :, 0].set(plane).reshape(-1)


@partial(jax.jit, static_argnames=("full_volume_shape",))
def _enforce_relion_half_volume_x0_hermitian_pair_jit(Ft_y, Ft_ctf, full_volume_shape):
    # Both accumulators in one program: two per shape compiled at every new size.
    return (
        _enforce_relion_half_volume_x0_hermitian_jit(Ft_y, full_volume_shape),
        _enforce_relion_half_volume_x0_hermitian_jit(Ft_ctf, full_volume_shape),
    )


def enforce_relion_half_volume_x0_hermitian_host(volume_flat, full_volume_shape):
    """Host implementation of RELION x=0 Hermitian-plane enforcement.

    The device path updates a single plane with ``.at[..., 0].set(...)`` but may
    still allocate another full packed half-volume.  Large RELION BPref grids
    already repack through host memory downstream, so handling the plane update
    here avoids a transient GPU allocation without changing the arithmetic.
    """

    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(full_volume_shape)
    host = np.asarray(jax.device_get(volume_flat))
    if isinstance(volume_flat, np.ndarray) or not host.flags.writeable:
        host = host.copy()
    vol = host.reshape(half_shape)
    n0, n1, _ = half_shape
    i0 = np.arange(n0, dtype=np.int32)
    i1 = np.arange(n1, dtype=np.int32)
    p0 = (n0 - (n0 % 2) - i0) % n0
    p1 = (n1 - (n1 % 2) - i1) % n1
    plane = vol[:, :, 0]
    partner = np.conj(plane[np.ix_(p0, p1)])
    summed = plane + partner
    self_partner = (p0[:, None] == i0[:, None]) & (p1[None, :] == i1[None, :])
    vol[:, :, 0] = np.where(self_partner, plane, summed)
    return vol.reshape(-1)
