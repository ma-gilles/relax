"""RELION backprojection kernels on the GPU: the fused x-half backprojections (indexed and particle-grid), the
first-iteration fused BPref build, the point-group symmetrisation of BPref and the device-signature scope of
the backprojection diagnostics.

Each wrapper registers the library targets through ``library.ensure_ffi()`` before its first FFI call.
"""

from __future__ import annotations

import functools
import os
from contextlib import contextmanager
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from recovar.cuda_backproject import (
    bpref_device_signature_scope_var as _bpref_device_signature_scope,
)
from recovar.cuda_backproject import (
    custom_cuda_requested,
    logger,
)
from recovar.cuda_backproject import (
    ffi_kwargs as _ffi_kwargs,
)
from recovar.cuda_backproject import (
    prepare_relion_x_half_block_topology_operands as _prepare_relion_x_half_block_topology_operands,
)
from recovar.cuda_backproject import (
    relion_x_half_backproject_rotation_to_kernel as _relion_x_half_backproject_rotation_to_kernel,
)
from recovar.cuda_backproject import (
    rot_to_compact as _rot_to_compact,
)
from recovar.cuda_backproject import (
    validate_inputs as _validate_inputs,
)

from relax.cuda import library


@contextmanager
def bpref_device_signature_scope(active: bool):
    """Scope trace-time CUDA diagnostic flags to one explicit score boundary."""

    token = _bpref_device_signature_scope.set(bool(active))
    try:
        yield
    finally:
        _bpref_device_signature_scope.reset(token)


@functools.partial(jax.jit, static_argnums=(3, 4))
def relion_point_group_symmetrise_bpref(
    data_volume: jax.Array,
    weight_volume: jax.Array,
    right_operators: jax.Array,
    volume_shape: Tuple[int, int, int],
    support_radius: int,
) -> tuple[jax.Array, jax.Array]:
    """Apply RELION's x=0 and point-group BPref finalisation on CUDA.

    ``data_volume`` and ``weight_volume`` are flat RELION BackProjector
    accumulators whose logical storage order is ``(z, y, xhalf)``.  The
    identity-inclusive ``right_operators`` must be in the order returned by
    RELION's ``SymList``.  Each CUDA thread owns one output voxel and streams
    over the operators, so this does not materialise one rotated accumulator
    per symmetry mate.

    The kernel reads the input through RELION's summed (not averaged) x=0
    Hermitian-plane rule before applying point-group interpolation.  This
    fuses the two source-ordered operations without a full-size intermediate.
    Callers implementing C1 compatibility should bypass this primitive and
    use their pre-existing x=0 path so the historical result remains bitwise
    unchanged.
    """

    volume_shape = tuple(int(value) for value in volume_shape)
    if len(volume_shape) != 3 or len(set(volume_shape)) != 1:
        raise ValueError(
            "RELION point-group BPref symmetry requires a cubic 3-D grid, "
            f"got {volume_shape}"
        )
    if any(value <= 0 or value % 2 == 0 for value in volume_shape):
        raise ValueError(
            "RELION point-group BPref symmetry requires an odd positive grid, "
            f"got {volume_shape}"
        )

    maximum_supported_radius = volume_shape[0] // 2 - 1
    if support_radius < 0 or support_radius > maximum_supported_radius:
        raise ValueError(
            "RELION point-group BPref support radius must leave its one-voxel "
            f"interpolation margin: got {support_radius} for {volume_shape} "
            f"(maximum {maximum_supported_radius})"
        )

    expected_size = int(volume_shape[0] * volume_shape[1] * (volume_shape[2] // 2 + 1))
    if data_volume.ndim != 1 or data_volume.shape != (expected_size,):
        raise ValueError(
            "RELION point-group data accumulator must be flat with shape "
            f"{(expected_size,)}, got {data_volume.shape}"
        )
    if weight_volume.ndim != 1 or weight_volume.shape != (expected_size,):
        raise ValueError(
            "RELION point-group weight accumulator must be flat with shape "
            f"{(expected_size,)}, got {weight_volume.shape}"
        )

    if data_volume.dtype == jnp.dtype(jnp.complex64):
        real_dtype = jnp.dtype(jnp.float32)
    elif data_volume.dtype == jnp.dtype(jnp.complex128):
        real_dtype = jnp.dtype(jnp.float64)
    else:
        raise TypeError(
            "RELION point-group data accumulator must be complex64 or complex128, "
            f"got {data_volume.dtype}"
        )
    if weight_volume.dtype != real_dtype:
        raise TypeError(
            "RELION point-group weight dtype must match the data component dtype, "
            f"got data={data_volume.dtype}, weight={weight_volume.dtype}"
        )
    if right_operators.ndim != 3 or right_operators.shape[1:] != (3, 3):
        raise ValueError(
            "RELION point-group operators must have shape (n, 3, 3), "
            f"got {right_operators.shape}"
        )
    if right_operators.shape[0] < 1:
        raise ValueError("RELION point-group operators must include identity first")
    if right_operators.dtype != real_dtype:
        raise TypeError(
            "RELION point-group operator dtype must match the accumulator component dtype, "
            f"got operators={right_operators.dtype}, data={data_volume.dtype}"
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION point-group BPref symmetry requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION point-group BPref symmetry was requested but custom CUDA is disabled"
        )

    library.ensure_ffi()
    output_types = (
        jax.ShapeDtypeStruct(data_volume.shape, data_volume.dtype),
        jax.ShapeDtypeStruct(weight_volume.shape, weight_volume.dtype),
    )
    return jax.ffi.ffi_call(
        library.TARGET_RELION_POINT_GROUP_SYMMETRISE_BPREF,
        output_types,
        vmap_method="sequential",
    )(
        data_volume,
        weight_volume,
        right_operators,
        full_z=np.int64(volume_shape[0]),
        full_y=np.int64(volume_shape[1]),
        full_x=np.int64(volume_shape[2]),
        support_radius=np.int64(support_radius),
    )


def relion_point_group_symmetrise_bpref_host(
    data_volume: jax.Array,
    weight_volume: jax.Array,
    right_operators: jax.Array,
    volume_shape: Tuple[int, int, int],
    support_radius: int,
    *,
    chunk_voxels: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Finalize complex64 BPref into bounded host chunks without splitting inputs.

    Keeps the caller's complex accumulator on device and allocates only a
    bounded complex/weight output range, including the final partial chunk.
    """

    volume_shape = tuple(int(value) for value in volume_shape)
    if len(volume_shape) != 3 or len(set(volume_shape)) != 1:
        raise ValueError(
            "RELION point-group BPref symmetry requires a cubic 3-D grid, "
            f"got {volume_shape}"
        )
    if any(value <= 0 or value % 2 == 0 for value in volume_shape):
        raise ValueError(
            "RELION point-group BPref symmetry requires an odd positive grid, "
            f"got {volume_shape}"
        )
    maximum_supported_radius = volume_shape[0] // 2 - 1
    if support_radius < 0 or support_radius > maximum_supported_radius:
        raise ValueError(
            "RELION point-group BPref support radius must leave its one-voxel "
            f"interpolation margin: got {support_radius} for {volume_shape} "
            f"(maximum {maximum_supported_radius})"
        )

    expected_size = int(volume_shape[0] * volume_shape[1] * (volume_shape[2] // 2 + 1))
    for label, value in (
        ("complex data", data_volume),
        ("weight", weight_volume),
    ):
        if value.ndim != 1 or value.shape != (expected_size,):
            raise ValueError(
                f"RELION complex point-group {label} accumulator must be flat with shape "
                f"{(expected_size,)}, got {value.shape}"
            )
        expected_dtype = jnp.complex64 if label == "complex data" else jnp.float32
        if value.dtype != jnp.dtype(expected_dtype):
            raise TypeError(
                f"RELION complex point-group {label} accumulator must be {expected_dtype}, "
                f"got {value.dtype}"
            )
    if right_operators.ndim != 3 or right_operators.shape[1:] != (3, 3):
        raise ValueError(
            "RELION point-group operators must have shape (n, 3, 3), "
            f"got {right_operators.shape}"
        )
    if right_operators.shape[0] < 1:
        raise ValueError("RELION point-group operators must include identity first")
    if right_operators.dtype != jnp.dtype(jnp.float32):
        raise TypeError(
            "RELION complex point-group operators must be float32, "
            f"got {right_operators.dtype}"
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION point-group BPref symmetry requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION point-group BPref symmetry was requested but custom CUDA is disabled"
        )

    if chunk_voxels is None:
        raw_chunk_voxels = os.environ.get(
            "RELAX_RELION_BPREF_SYMMETRY_CHUNK_VOXELS",
        )
        chunk_voxels = 16 * 1024 * 1024 if raw_chunk_voxels is None else int(raw_chunk_voxels)
    if chunk_voxels <= 0:
        raise ValueError(f"RELION BPref symmetry chunk_voxels must be positive, got {chunk_voxels}")
    chunk_voxels = min(chunk_voxels, expected_size)

    data_host = np.empty(expected_size, dtype=np.complex64)
    weight_host = np.empty(expected_size, dtype=np.float32)
    chunk_count = (expected_size + chunk_voxels - 1) // chunk_voxels
    logger.info(
        "RELION complex BPref symmetry: voxels=%d chunk_voxels=%d chunks=%d host=%.2f GiB",
        expected_size,
        chunk_voxels,
        chunk_count,
        (data_host.nbytes + weight_host.nbytes) / float(1024**3),
    )
    for chunk_index, start in enumerate(range(0, expected_size, chunk_voxels)):
        data_chunk, weight_chunk = _relion_point_group_symmetrise_bpref_complex_range_static(
            data_volume,
            weight_volume,
            right_operators,
            jnp.asarray([start], dtype=jnp.int64),
            volume_shape,
            support_radius,
            chunk_voxels,
        )
        data_chunk_host, weight_chunk_host = jax.device_get((data_chunk, weight_chunk))
        stop = min(start + chunk_voxels, expected_size)
        valid_count = stop - start
        np.copyto(data_host[start:stop], np.asarray(data_chunk_host)[:valid_count])
        np.copyto(weight_host[start:stop], np.asarray(weight_chunk_host)[:valid_count])
        del data_chunk, weight_chunk, data_chunk_host, weight_chunk_host
        if chunk_index + 1 == chunk_count or (chunk_index + 1) % 16 == 0:
            logger.info(
                "RELION complex BPref symmetry progress: chunk=%d/%d voxels=%d/%d",
                chunk_index + 1,
                chunk_count,
                stop,
                expected_size,
            )
    return data_host, weight_host


@functools.partial(jax.jit, static_argnums=(4, 5, 6))
def _relion_point_group_symmetrise_bpref_complex_range_static(
    data_volume: jax.Array,
    weight_volume: jax.Array,
    right_operators: jax.Array,
    range_start: jax.Array,
    volume_shape: Tuple[int, int, int],
    support_radius: int,
    range_voxels: int,
) -> tuple[jax.Array, jax.Array]:
    """Emit one bounded complex/weight range from full complex input."""

    library.ensure_ffi()
    output_types = (
        jax.ShapeDtypeStruct((range_voxels,), jnp.complex64),
        jax.ShapeDtypeStruct((range_voxels,), jnp.float32),
    )
    return jax.ffi.ffi_call(
        library.TARGET_RELION_POINT_GROUP_SYMMETRISE_BPREF_COMPLEX_RANGE,
        output_types,
        vmap_method="sequential",
    )(
        data_volume,
        weight_volume,
        right_operators,
        range_start,
        full_z=np.int64(volume_shape[0]),
        full_y=np.int64(volume_shape[1]),
        full_x=np.int64(volume_shape[2]),
        support_radius=np.int64(support_radius),
    )


@functools.partial(jax.jit, static_argnums=(6, 7, 8))
def relion_fused_x_half_backproject_indexed(
    data_volume: jax.Array,
    weight_volume: jax.Array,
    data_rows: jax.Array,
    weight_rows: jax.Array,
    pixel_indices: jax.Array,
    rotation_matrices: jax.Array,
    image_shape: Tuple[int, int],
    volume_shape: Tuple[int, int, int],
    max_r: float | None,
) -> tuple[jax.Array, jax.Array]:
    """Fused RELION x-half data/weight backprojection diagnostic.

    This target is intentionally narrower than :func:`backproject_indexed`:
    it accepts matching complex64/float32 or complex128/float64 rows, expands
    both to RELION's native current-size FFTW square, and updates the matching
    data and weight accumulators in one 128-thread CUDA grid.  The two
    output buffers alias their corresponding input accumulators.
    """

    library.ensure_ffi()
    _validate_inputs(volume_shape, image_shape, 1, True, True, max_r=max_r)
    if max_r is None:
        raise ValueError("RELION fused x-half backprojection requires an explicit support radius")
    if int(volume_shape[2]) % 2 == 0:
        raise ValueError(f"RELION fused x-half backprojection requires an odd BPref grid, got {volume_shape}")

    if data_volume.dtype not in {jnp.dtype(jnp.complex64), jnp.dtype(jnp.complex128)}:
        raise TypeError(f"RELION fused x-half data volume must be complex64 or complex128, got {data_volume.dtype}")
    real_dtype = jnp.dtype(jnp.float64 if data_volume.dtype == jnp.dtype(jnp.complex128) else jnp.float32)
    if weight_volume.dtype != real_dtype:
        raise TypeError(f"RELION fused x-half weight volume must be {real_dtype}, got {weight_volume.dtype}")
    if data_rows.dtype != data_volume.dtype:
        raise TypeError(f"RELION fused x-half data rows must match the volume dtype, got {data_rows.dtype}")
    if weight_rows.dtype != real_dtype:
        raise TypeError(f"RELION fused x-half weight rows must be {real_dtype}, got {weight_rows.dtype}")
    if pixel_indices.dtype != jnp.dtype(jnp.int32):
        raise TypeError(f"RELION fused x-half pixel indices must be int32, got {pixel_indices.dtype}")
    if not jnp.issubdtype(rotation_matrices.dtype, jnp.floating):
        raise TypeError(f"RELION fused x-half rotations must be floating point, got {rotation_matrices.dtype}")

    if data_rows.ndim != 2 or weight_rows.ndim != 2:
        raise ValueError(
            "RELION fused x-half rows must both have shape (n_rotations, n_pixels), "
            f"got {data_rows.shape} and {weight_rows.shape}"
        )
    if data_rows.shape != weight_rows.shape:
        raise ValueError(
            f"RELION fused x-half data/weight row shape mismatch: {data_rows.shape} vs {weight_rows.shape}"
        )
    if data_rows.shape[0] <= 0 or data_rows.shape[1] <= 0:
        raise ValueError(f"RELION fused x-half rows must be nonempty, got {data_rows.shape}")
    if pixel_indices.ndim != 1 or pixel_indices.shape[0] != data_rows.shape[1]:
        raise ValueError(
            "RELION fused x-half pixel indices must match the row pixel axis, "
            f"got {pixel_indices.shape} for rows {data_rows.shape}"
        )
    if rotation_matrices.shape != (data_rows.shape[0], 3, 3):
        raise ValueError(
            f"RELION fused x-half rotations must have shape ({data_rows.shape[0]}, 3, 3), got {rotation_matrices.shape}"
        )
    if data_volume.ndim != 1 or weight_volume.ndim != 1:
        raise ValueError(
            f"RELION fused x-half accumulators must be flat, got {data_volume.shape} and {weight_volume.shape}"
        )
    expected_volume_size = int(volume_shape[0] * volume_shape[1] * (volume_shape[2] // 2 + 1))
    if data_volume.shape != (expected_volume_size,) or weight_volume.shape != (expected_volume_size,):
        raise ValueError(
            "RELION fused x-half accumulator shape mismatch: expected "
            f"{(expected_volume_size,)}, got {data_volume.shape} and {weight_volume.shape}"
        )

    dense_data_rows, dense_indices, current_height, current_half_width = _prepare_relion_x_half_block_topology_operands(
        data_rows, pixel_indices, image_shape, max_r
    )
    dense_weight_rows, weight_dense_indices, weight_height, weight_half_width = (
        _prepare_relion_x_half_block_topology_operands(weight_rows, pixel_indices, image_shape, max_r)
    )
    if (weight_height, weight_half_width) != (current_height, current_half_width):
        raise ValueError("RELION fused x-half data/weight topology metadata mismatch")
    # Both preparations use the same original indices, shape, and radius, so
    # their dense index vectors are identical. Keep a shape assertion here and
    # pass only one vector to the FFI target.
    if weight_dense_indices.shape != dense_indices.shape:
        raise ValueError("RELION fused x-half data/weight dense index shape mismatch")

    kw, _, _ = _ffi_kwargs(image_shape, volume_shape, 1, True, True, max_r)
    kw["image_h"] = np.int64(current_height)
    kw["image_w"] = np.int64(current_half_width)
    kw["full_image_w"] = np.int64(current_height)
    # Sparse pass-2 supplies the dedicated M-step matrices generated through
    # RELION's host ``generateEulerMatrices(..., inverse=true)`` emulation.
    # They are already in RECOVAR's transposed convention; applying the
    # generic scorer-to-backprojector numerical inverse here would invert them
    # a second time.  The packed half-axis permutation is still required.
    rotation_matrices = rotation_matrices.astype(real_dtype)[..., [2, 1, 0]]
    rot6 = _rot_to_compact(rotation_matrices, real_dtype)

    out_types = (
        jax.ShapeDtypeStruct(data_volume.shape, data_volume.dtype),
        jax.ShapeDtypeStruct(weight_volume.shape, weight_volume.dtype),
    )
    return jax.ffi.ffi_call(
        library.TARGET_RELION_FUSED_X_HALF_BP,
        out_types,
        input_output_aliases={4: 0, 5: 1},
        vmap_method="sequential",
    )(
        dense_data_rows,
        dense_weight_rows,
        dense_indices,
        rot6,
        data_volume,
        weight_volume,
        **kw,
    )


@functools.partial(jax.jit, static_argnums=(6, 7, 8))
def relion_fused_x_half_backproject_particle_grid_indexed(
    data_volume: jax.Array,
    weight_volume: jax.Array,
    data_rows: jax.Array,
    weight_rows: jax.Array,
    pixel_indices: jax.Array,
    rotation_matrices: jax.Array,
    image_shape: Tuple[int, int],
    volume_shape: Tuple[int, int, int],
    max_r: float | None,
) -> tuple[jax.Array, jax.Array]:
    """Launch one ordered native fused grid for every particle in one FFI call.

    Inputs retain explicit ``(particle, rotation, pixel)`` ownership.  The
    native handler loops over particles on the caller's CUDA stream and
    launches the same RELION-shaped grid used by the single-particle target,
    eliminating framework dispatch between particle-owned launches without
    combining their atomic streams.
    """

    library.ensure_ffi()
    _validate_inputs(volume_shape, image_shape, 1, True, True, max_r=max_r)
    if max_r is None:
        raise ValueError("RELION fused x-half particle grid requires an explicit support radius")
    if int(volume_shape[2]) % 2 == 0:
        raise ValueError(f"RELION fused x-half particle grid requires an odd BPref grid, got {volume_shape}")
    if data_volume.dtype != jnp.dtype(jnp.complex64) or data_rows.dtype != jnp.dtype(jnp.complex64):
        raise TypeError("RELION fused x-half particle-grid data must be complex64")
    if weight_volume.dtype != jnp.dtype(jnp.float32) or weight_rows.dtype != jnp.dtype(jnp.float32):
        raise TypeError("RELION fused x-half particle-grid weights must be float32")
    if pixel_indices.dtype != jnp.dtype(jnp.int32):
        raise TypeError("RELION fused x-half particle-grid pixel indices must be int32")
    if data_rows.ndim != 3 or weight_rows.shape != data_rows.shape:
        raise ValueError(
            "RELION fused x-half particle-grid rows must have matching "
            f"(particle, rotation, pixel) shapes, got {data_rows.shape} and {weight_rows.shape}"
        )
    n_particles, rows_per_particle, n_pixels = map(int, data_rows.shape)
    if n_particles <= 0 or rows_per_particle <= 0 or n_pixels <= 0:
        raise ValueError(f"RELION fused x-half particle-grid rows must be nonempty, got {data_rows.shape}")
    if pixel_indices.shape != (n_pixels,):
        raise ValueError(
            f"RELION fused x-half particle-grid pixel indices must have shape {(n_pixels,)}, "
            f"got {pixel_indices.shape}"
        )
    if rotation_matrices.shape != (n_particles, rows_per_particle, 3, 3):
        raise ValueError(
            "RELION fused x-half particle-grid rotations must have shape "
            f"{(n_particles, rows_per_particle, 3, 3)}, got {rotation_matrices.shape}"
        )

    flat_data = data_rows.reshape(n_particles * rows_per_particle, n_pixels)
    flat_weight = weight_rows.reshape(n_particles * rows_per_particle, n_pixels)
    dense_data, dense_indices, current_height, current_half_width = (
        _prepare_relion_x_half_block_topology_operands(
            flat_data, pixel_indices, image_shape, max_r
        )
    )
    dense_weight, weight_dense_indices, weight_height, weight_half_width = (
        _prepare_relion_x_half_block_topology_operands(
            flat_weight, pixel_indices, image_shape, max_r
        )
    )
    if (weight_height, weight_half_width) != (current_height, current_half_width):
        raise ValueError("RELION fused x-half particle-grid topology metadata mismatch")
    if weight_dense_indices.shape != dense_indices.shape:
        raise ValueError("RELION fused x-half particle-grid dense index shape mismatch")

    kw, _, _ = _ffi_kwargs(image_shape, volume_shape, 1, True, True, max_r)
    kw.update(
        image_h=np.int64(current_height),
        image_w=np.int64(current_half_width),
        full_image_w=np.int64(current_height),
        n_particles=np.int64(n_particles),
        rows_per_particle=np.int64(rows_per_particle),
    )
    kernel_rotations = _relion_x_half_backproject_rotation_to_kernel(
        rotation_matrices.reshape(n_particles * rows_per_particle, 3, 3),
        jnp.float32,
    )
    rot6 = _rot_to_compact(kernel_rotations, jnp.float32)
    out_types = (
        jax.ShapeDtypeStruct(data_volume.shape, data_volume.dtype),
        jax.ShapeDtypeStruct(weight_volume.shape, weight_volume.dtype),
    )
    return jax.ffi.ffi_call(
        library.TARGET_RELION_FUSED_X_HALF_BP_PARTICLE_GRID,
        out_types,
        input_output_aliases={4: 0, 5: 1},
        vmap_method="sequential",
    )(
        dense_data,
        dense_weight,
        dense_indices,
        rot6,
        data_volume,
        weight_volume,
        **kw,
    )


def relion_firstiter_bpref_fused_x_half(
    data_volume: jax.Array,
    weight_volume: jax.Array,
    image: jax.Array,
    ctf: jax.Array,
    minvsigma2: jax.Array,
    posterior: jax.Array,
    translation_angles: jax.Array,
    native_euler_matrices: jax.Array,
    significant_weight: jax.Array,
    weight_norm: jax.Array,
    image_shape: Tuple[int, int],
    volume_shape: Tuple[int, int, int],
    max_r: float,
) -> tuple[jax.Array, jax.Array]:
    """Dispatch the exact-source kernel with host-known scalar attributes."""

    significant_weight_scalar = float(np.asarray(significant_weight).reshape(()))
    weight_norm_scalar = float(np.asarray(weight_norm).reshape(()))
    return _relion_firstiter_bpref_fused_x_half_static(
        data_volume,
        weight_volume,
        image,
        ctf,
        minvsigma2,
        posterior,
        translation_angles,
        native_euler_matrices,
        significant_weight_scalar,
        weight_norm_scalar,
        image_shape,
        volume_shape,
        max_r,
    )


@functools.partial(jax.jit, static_argnums=(8, 9, 10, 11, 12))
def _relion_firstiter_bpref_fused_x_half_static(
    data_volume: jax.Array,
    weight_volume: jax.Array,
    image: jax.Array,
    ctf: jax.Array,
    minvsigma2: jax.Array,
    posterior: jax.Array,
    translation_angles: jax.Array,
    native_euler_matrices: jax.Array,
    significant_weight: float,
    weight_norm: float,
    image_shape: Tuple[int, int],
    volume_shape: Tuple[int, int, int],
    max_r: float,
) -> tuple[jax.Array, jax.Array]:
    """Form native-unit firstiter BPref operands and scatter in one CUDA block.

    Inputs are the exact arrays consumed by RELION ``BP.cuh`` after its image
    crop: one dense current-size FFTW half image, native CTF and Minvsigma2,
    the rotation/translation posterior matrix, native translation angles, and
    native Euler matrices.  The output remains in RELION accumulator units;
    callers must not fold RECOVAR's N²/N⁴ normalization into these operands.
    """

    expected_volume_size = int(
        volume_shape[0] * volume_shape[1] * (volume_shape[2] // 2 + 1)
    )
    if data_volume.dtype != jnp.dtype(jnp.complex64) or data_volume.shape != (
        expected_volume_size,
    ):
        raise TypeError(
            "RELION firstiter fused BPref data accumulator must be flat complex64 shape "
            f"{(expected_volume_size,)}, got {data_volume.shape}/{data_volume.dtype}"
        )
    data_volume_real = jnp.real(data_volume).astype(jnp.float32)
    data_volume_imag = jnp.imag(data_volume).astype(jnp.float32)
    data_real_out, data_imag_out, weight_out = (
        _relion_firstiter_bpref_fused_x_half_split_impl(
            data_volume_real,
            data_volume_imag,
            weight_volume,
            image,
            ctf,
            minvsigma2,
            posterior,
            translation_angles,
            native_euler_matrices,
            significant_weight,
            weight_norm,
            image_shape,
            volume_shape,
            max_r,
        )
    )
    return jax.lax.complex(data_real_out, data_imag_out), weight_out


def _relion_firstiter_bpref_fused_x_half_split_impl(
    data_volume_real: jax.Array,
    data_volume_imag: jax.Array,
    weight_volume: jax.Array,
    image: jax.Array,
    ctf: jax.Array,
    minvsigma2: jax.Array,
    posterior: jax.Array,
    translation_angles: jax.Array,
    native_euler_matrices: jax.Array,
    significant_weight: float,
    weight_norm: float,
    image_shape: Tuple[int, int],
    volume_shape: Tuple[int, int, int],
    max_r: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Validate and issue the shared three-accumulator firstiter FFI call."""

    library.ensure_ffi()
    _validate_inputs(volume_shape, image_shape, 1, True, True, max_r=max_r)
    if int(volume_shape[2]) % 2 == 0:
        raise ValueError(
            f"RELION firstiter fused BPref requires an odd BPref grid, got {volume_shape}"
        )
    expected_pixels = int(image_shape[0] * (image_shape[1] // 2 + 1))
    expected_volume_size = int(
        volume_shape[0] * volume_shape[1] * (volume_shape[2] // 2 + 1)
    )
    if image.dtype != jnp.dtype(jnp.complex64) or image.shape != (expected_pixels,):
        raise TypeError(
            "RELION firstiter fused BPref image must be a dense complex64 FFTW half row, "
            f"got {image.shape}/{image.dtype}"
        )
    for label, value in (("ctf", ctf), ("minvsigma2", minvsigma2)):
        if value.dtype != jnp.dtype(jnp.float32) or value.shape != (expected_pixels,):
            raise TypeError(
                f"RELION firstiter fused BPref {label} must be float32 shape "
                f"{(expected_pixels,)}, got {value.shape}/{value.dtype}"
            )
    if posterior.dtype != jnp.dtype(jnp.float32) or posterior.ndim != 2:
        raise TypeError(
            "RELION firstiter fused BPref posterior must be a float32 rotation/translation matrix"
        )
    if posterior.shape[0] <= 0 or posterior.shape[1] <= 0:
        raise ValueError("RELION firstiter fused BPref posterior must be nonempty")
    if (
        translation_angles.dtype != jnp.dtype(jnp.float32)
        or translation_angles.shape != (posterior.shape[1], 2)
    ):
        raise TypeError(
            "RELION firstiter fused BPref translation angles must be float32 shape "
            f"{(posterior.shape[1], 2)}, got {translation_angles.shape}/{translation_angles.dtype}"
        )
    if (
        native_euler_matrices.dtype != jnp.dtype(jnp.float32)
        or native_euler_matrices.shape != (posterior.shape[0], 3, 3)
    ):
        raise TypeError(
            "RELION firstiter fused BPref Euler matrices must be native-convention float32 "
            f"shape {(posterior.shape[0], 3, 3)}, got "
            f"{native_euler_matrices.shape}/{native_euler_matrices.dtype}"
        )
    if not np.isfinite(significant_weight) or not np.isfinite(weight_norm):
        raise ValueError("RELION firstiter fused BPref scalar attributes must be finite")
    if weight_norm <= 0.0:
        raise ValueError("RELION firstiter fused BPref weight norm must be positive")
    for label, value in (
        ("real data", data_volume_real),
        ("imaginary data", data_volume_imag),
        ("weight", weight_volume),
    ):
        if value.dtype != jnp.dtype(jnp.float32) or value.shape != (
            expected_volume_size,
        ):
            raise TypeError(
                f"RELION firstiter fused BPref {label} accumulator must be flat "
                f"float32 shape {(expected_volume_size,)}, got {value.shape}/{value.dtype}"
            )

    kw, _, _ = _ffi_kwargs(image_shape, volume_shape, 1, True, True, max_r)
    kw["significant_weight"] = np.float32(significant_weight)
    kw["weight_norm"] = np.float32(weight_norm)
    out_types = (
        jax.ShapeDtypeStruct(data_volume_real.shape, data_volume_real.dtype),
        jax.ShapeDtypeStruct(data_volume_imag.shape, data_volume_imag.dtype),
        jax.ShapeDtypeStruct(weight_volume.shape, weight_volume.dtype),
    )
    data_real_out, data_imag_out, weight_out = jax.ffi.ffi_call(
        library.TARGET_RELION_FIRSTITER_BPREF_FUSED_X_HALF,
        out_types,
        input_output_aliases={8: 0, 9: 1, 10: 2},
        vmap_method="sequential",
    )(
        jnp.real(image).astype(jnp.float32),
        jnp.imag(image).astype(jnp.float32),
        ctf,
        minvsigma2,
        posterior,
        translation_angles[:, 0],
        translation_angles[:, 1],
        native_euler_matrices.reshape(posterior.shape[0], 9),
        data_volume_real,
        data_volume_imag,
        weight_volume,
        **kw,
    )
    return data_real_out, data_imag_out, weight_out
