"""RELION's per-row float32 posterior kernels and its batched CUB sort-scan on the GPU.

Test oracle: moved from relax/cuda/posterior.py, where only tests called them. The per-row exponentiate, divide
and CUB sort-scan are the references the live batched kernels (``relion_exponentiate_batched_f32``,
``relion_divide_batched_f32``) are compared with, and the batched CUB sort-scan is RELION's sort-and-scan cut that
the live ``relion_coarse_cut_f32`` is compared with. Their native handlers stay in the relax library.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from recovar.cuda_backproject import custom_cuda_requested

from relax.cuda import library


def _relion_posterior_rows_f32(target: str, operation: str, operand_name: str, values, operand):
    """One RELION float32 posterior kernel on a 1-D ``values`` and a float32 scalar ``operand`` (traced)."""

    if values.dtype != jnp.float32:
        raise TypeError(f"values must be float32, got {values.dtype}")
    if values.ndim != 1 or values.shape[0] < 1:
        raise ValueError(f"values must be a nonempty 1-D array, got {values.shape}")
    if operand.dtype != jnp.float32 or operand.ndim != 0:
        raise TypeError(f"{operand_name} must be a float32 scalar, got {operand.dtype} {operand.shape}")
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"RELION float32 {operation} requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(f"RELION float32 {operation} was requested but custom CUDA is disabled")
    library.ensure_ffi()

    output_type = jax.ShapeDtypeStruct(values.shape, jnp.float32)
    return jax.ffi.ffi_call(
        target,
        output_type,
        vmap_method="sequential",
    )(values, operand)


@jax.jit
def relion_exponentiate_f32(values: jax.Array, add: jax.Array) -> jax.Array:
    """Apply RELION's fine-weight ``expf(value + add)`` CUDA kernel."""

    return _relion_posterior_rows_f32(library.TARGET_RELION_EXPONENTIATE_F32, "exponentiation", "add", values, add)


@jax.jit
def relion_divide_f32(values: jax.Array, divisor: jax.Array) -> jax.Array:
    """Apply RELION's CUDA ``float / float`` posterior normalization."""

    return _relion_posterior_rows_f32(library.TARGET_RELION_DIVIDE_F32, "division", "divisor", values, divisor)


@jax.jit
def relion_cub_sort_scan_f32(values: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Sort and inclusively scan one float32 vector with RELION's CUB calls.

    A test reference: relax's cuts use :func:`relion_coarse_cut_f32`.

    This is a strict diagnostic primitive for the coarse-significance boundary.
    RELION invokes ``cub::DeviceRadixSort::SortKeys`` followed by
    ``cub::DeviceScan::InclusiveSum`` on each particle's positive weights.
    Keeping both intermediate arrays observable lets parity tests identify a
    sort discrepancy separately from a scan discrepancy.
    """

    if values.dtype != jnp.float32:
        raise TypeError(f"values must be float32, got {values.dtype}")
    if values.ndim != 1 or values.shape[0] < 1:
        raise ValueError(f"values must be a nonempty 1-D array, got {values.shape}")
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION CUB sort/scan requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError("RELION CUB sort/scan was requested but custom CUDA is disabled")
    library.ensure_ffi()

    output_type = jax.ShapeDtypeStruct(values.shape, jnp.float32)
    return jax.ffi.ffi_call(
        library.TARGET_RELION_CUB_SORT_SCAN_F32,
        (output_type, output_type),
        vmap_method="sequential",
    )(values)


@jax.jit
def relion_cub_sort_scan_batched_f32(
    values: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Run the exact RELION CUB sort/scan row order in one FFI call.

    A test reference: relax's cuts use :func:`relion_coarse_cut_f32`.

    Rows remain serialized on the caller's XLA stream. The implementation
    reuses one stream-ordered scratch allocation, so this changes dispatch and
    allocation topology without changing any row's CUB arithmetic. Rows must be
    posterior weights (zero or positive, no NaN): only the positive weights are
    radix-sorted and the zeros are written in front of them.
    """

    if values.dtype != jnp.float32 or values.ndim != 2:
        raise TypeError(f"values must be a float32 matrix, got {values.dtype} {values.shape}")
    if values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError(f"values must be nonempty, got {values.shape}")
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION batched CUB sort/scan requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION batched CUB sort/scan was requested but custom CUDA is disabled"
        )
    library.ensure_ffi()

    output_type = jax.ShapeDtypeStruct(values.shape, jnp.float32)
    return jax.ffi.ffi_call(
        library.TARGET_RELION_CUB_SORT_SCAN_BATCHED_F32,
        (output_type, output_type),
    )(values)
