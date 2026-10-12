"""RELION scoring rotations on the GPU: the f32, left-multiplied f32 and f64 rotation builds of librelax_cuda.so.

Each wrapper registers the library targets through ``library.ensure_ffi()`` before its first FFI call.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from recovar.cuda_backproject import (
    custom_cuda_requested,
)

from relax.cuda import library


@functools.partial(jax.jit, static_argnums=(2,))
def relion_make_scoring_rotations_f32(
    eulers_deg: jax.Array,
    right_matrix: jax.Array,
    do_right: bool = True,
) -> jax.Array:
    """Build scorer rotations with RELION's accelerated float32 arithmetic.

    This reproduces ``cuda_kernel_make_eulers_3D<true,false,do_right>``
    through its Euler construction and optional ``B = A @ R`` product.  The
    returned matrices use RECOVAR's scorer convention ``B``.  RELION stores
    the inverse-projector convention ``B.T``, so a frozen RELION device dump
    must be transposed before a bitwise comparison with this result.

    This is a strict CUDA primitive: it has no CPU or JAX arithmetic fallback.
    """

    if eulers_deg.dtype != jnp.float32:
        raise TypeError(f"eulers_deg must be float32, got {eulers_deg.dtype}")
    if right_matrix.dtype != jnp.float32:
        raise TypeError(f"right_matrix must be float32, got {right_matrix.dtype}")
    if eulers_deg.ndim != 2 or eulers_deg.shape[1:] != (3,):
        raise ValueError(f"eulers_deg must have shape (N, 3), got {eulers_deg.shape}")
    if right_matrix.shape != (3, 3):
        raise ValueError(f"right_matrix must have shape (3, 3), got {right_matrix.shape}")
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION scorer-rotation construction requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError("RELION scorer-rotation construction was explicitly requested but custom CUDA is disabled")
    library.ensure_ffi()

    out_type = jax.ShapeDtypeStruct((eulers_deg.shape[0], 3, 3), jnp.float32)
    return jax.ffi.ffi_call(
        library.TARGET_RELION_MAKE_SCORING_ROTATIONS_F32,
        out_type,
        vmap_method="sequential",
    )(
        eulers_deg,
        right_matrix,
        do_right=np.int64(int(do_right)),
    )


@functools.partial(jax.jit, static_argnums=(3,))
def relion_make_scoring_rotations_left_f32(
    eulers_deg: jax.Array,
    right_matrix: jax.Array,
    left_matrices: jax.Array,
    do_right: bool = True,
) -> jax.Array:
    """Scorer rotations of tilt images with RELION's float32 ``make_eulers_3D<true, true, do_right>``.

    Each image ``b`` has its own left matrix ``L_b`` (its ``Aproj`` times the optics scale):
    ``B = L_b (A R)``, inverted with RELION's float32 adjugate and determinant, since with a
    left matrix the inverse is not the transpose (acc/cuda/cuda_kernels/helper.cuh:777-811).
    Returns ``[B, N, 3, 3]`` in RECOVAR's scorer frame, the transpose of RELION's stored
    matrices, like :func:`relion_make_scoring_rotations_f32`. CUDA only.
    """

    for name, value in (("eulers_deg", eulers_deg), ("right_matrix", right_matrix), ("left_matrices", left_matrices)):
        if value.dtype != jnp.float32:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
    if eulers_deg.ndim != 2 or eulers_deg.shape[1:] != (3,):
        raise ValueError(f"eulers_deg must have shape (N, 3), got {eulers_deg.shape}")
    if right_matrix.shape != (3, 3):
        raise ValueError(f"right_matrix must have shape (3, 3), got {right_matrix.shape}")
    if left_matrices.ndim != 3 or left_matrices.shape[1:] != (3, 3):
        raise ValueError(f"left_matrices must have shape (B, 3, 3), got {left_matrices.shape}")
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION scorer-rotation construction requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError("RELION scorer-rotation construction was explicitly requested but custom CUDA is disabled")
    library.ensure_ffi()

    out_type = jax.ShapeDtypeStruct((left_matrices.shape[0], eulers_deg.shape[0], 3, 3), jnp.float32)
    return jax.ffi.ffi_call(
        library.TARGET_RELION_MAKE_SCORING_ROTATIONS_LEFT_F32,
        out_type,
        vmap_method="sequential",
    )(
        eulers_deg,
        right_matrix,
        left_matrices,
        do_right=np.int64(int(do_right)),
    )


@functools.partial(jax.jit, static_argnums=(2,))
def relion_make_scoring_rotations_f64(
    eulers_deg: jax.Array,
    right_matrix: jax.Array,
    do_right: bool = True,
) -> jax.Array:
    """Build scorer rotations with RELION's accelerated float64 arithmetic.

    This is the ``ACC_DOUBLE_PRECISION`` specialization of RELION's CUDA
    ``make_eulers_3D`` operation. It deliberately executes trigonometry and
    the optional right-matrix product on the device, matching coarse scoring.
    """

    if eulers_deg.dtype != jnp.float64:
        raise TypeError(f"eulers_deg must be float64, got {eulers_deg.dtype}")
    if right_matrix.dtype != jnp.float64:
        raise TypeError(f"right_matrix must be float64, got {right_matrix.dtype}")
    if eulers_deg.ndim != 2 or eulers_deg.shape[1:] != (3,):
        raise ValueError(f"eulers_deg must have shape (N, 3), got {eulers_deg.shape}")
    if right_matrix.shape != (3, 3):
        raise ValueError(f"right_matrix must have shape (3, 3), got {right_matrix.shape}")
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION scorer-rotation construction requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError("RELION scorer-rotation construction was explicitly requested but custom CUDA is disabled")
    library.ensure_ffi()

    out_type = jax.ShapeDtypeStruct((eulers_deg.shape[0], 3, 3), jnp.float64)
    return jax.ffi.ffi_call(
        library.TARGET_RELION_MAKE_SCORING_ROTATIONS_F64,
        out_type,
        vmap_method="sequential",
    )(
        eulers_deg,
        right_matrix,
        do_right=np.int64(int(do_right)),
    )
