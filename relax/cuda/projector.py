"""RELION projector kernels on the GPU: half-volume projection at a runtime radius, and the capacity and persistent
half-volume textures with their native handle lifetimes.

The texture classes own their librelax_cuda.so handles (created and destroyed through ``library.get_lib()``);
each wrapper registers the library targets through ``library.ensure_ffi()`` or
``library.ensure_optional_ffi(target)`` before its first FFI call.
"""

from __future__ import annotations

import ctypes
import functools
import threading
import time
import weakref
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from recovar.cuda_backproject import (
    custom_cuda_requested,
)
from recovar.cuda_backproject import (
    rot_to_compact as _rot_to_compact,
)

from relax.cuda import library


@functools.partial(jax.jit, static_argnames=("image_shape", "padding_factor"))
def project_relion_half_capacity(
    projector_half: jax.Array,
    rotation_matrices: jax.Array,
    logical_r_max: jax.Array,
    *,
    image_shape: Tuple[int, int],
    padding_factor: int = 1,
    image_r_max: jax.Array | None = None,
) -> jax.Array:
    """Opt-in RELION texture projection with physical storage and runtime radius.

    ``projector_half`` is C64 [z,y,x>=0], centered on y/z, with physical
    shape (pf*Q+3, pf*Q+3, pf*Q//2+2). Center-pad the *logical* projector,
    including its ghost planes; rebuilding at the physical radius is not
    equivalent. Rotations use the same F32 matrix convention as ``project``.
    Radius is an S32 scalar in unpadded units, with 0 <= r <= Q//2. Invalid
    runtime values return NaNs without a host readback; producers must enforce
    this range. Physical geometry, rotations and output shape control caching.

    Returns C64 [rotation, H*(W//2+1)] with the existing texture projector's
    centered rows/positive Nyquist convention. Compact pixel gathering and
    current-image masks remain the existing caller's responsibility unless
    ``image_r_max`` is supplied. That S32 scalar applies RELION's rotated
    float32 radius test at min(logical_r_max, image_r_max), preserving model
    texture staging. Do not additionally apply an exact source-pixel disk:
    it rejects valid boundary contributions. Image radius must be between
    zero and H//2; invalid runtime values return NaNs. This
    transaction includes texture allocation, staging, synchronization and
    cleanup; there is no persistent resource cache or automatic fallback.
    """
    projector_half = jnp.asarray(projector_half)
    rotation_matrices = jnp.asarray(rotation_matrices)
    logical_r_max = jnp.asarray(logical_r_max)
    shape = projector_half.shape
    if padding_factor not in (1, 2):
        raise ValueError("padding_factor must be 1 or 2")
    if (
        projector_half.dtype != jnp.complex64
        or len(shape) != 3
        or shape[0] < 5
        or shape[0] != shape[1]
        or shape[0] % 2 != 1
        or shape[2] != shape[0] // 2 + 1
        or (shape[0] - 3) % (2 * padding_factor) != 0
    ):
        raise ValueError("projector_half must be C64 [pf*Q+3,pf*Q+3,pf*Q//2+2] for even Q")
    if int(np.prod(shape)) > np.iinfo(np.int32).max:
        raise ValueError("projector_half has more texels than the staging kernel's int32 indexing")
    if (
        rotation_matrices.dtype != jnp.float32
        or rotation_matrices.ndim != 3
        or rotation_matrices.shape[1:] != (3, 3)
        or not 0 < rotation_matrices.shape[0] <= 65535
    ):
        raise ValueError("rotation_matrices must be nonempty F32 [rotation,3,3]")
    if logical_r_max.dtype != jnp.int32 or logical_r_max.shape != ():
        raise ValueError("logical_r_max must be an S32 scalar")
    if image_r_max is not None:
        image_r_max = jnp.asarray(image_r_max)
        if image_r_max.dtype != jnp.int32 or image_r_max.shape != ():
            raise ValueError("image_r_max must be an S32 scalar")
    if (
        len(image_shape) != 2
        or image_shape[0] != image_shape[1]
        or not 0 < image_shape[0] <= 4096
        or image_shape[0] % 2 != 0
    ):
        raise ValueError("image_shape must be a positive even square <=4096")
    n_pixels = image_shape[0] * (image_shape[1] // 2 + 1)
    if rotation_matrices.shape[0] * n_pixels > np.iinfo(np.int32).max:
        raise ValueError("projection output exceeds int32 kernel indexing")
    if image_r_max is None:
        library.ensure_optional_ffi(library.TARGET_PROJECT_RELION_HALF_RUNTIME)
        target = library.TARGET_PROJECT_RELION_HALF_RUNTIME
    else:
        library.ensure_optional_ffi(library.TARGET_PROJECT_RELION_HALF_IMAGE_RADIUS)
        target = library.TARGET_PROJECT_RELION_HALF_IMAGE_RADIUS
    rot6 = _rot_to_compact(rotation_matrices, jnp.float32)
    output = jax.ShapeDtypeStruct((rotation_matrices.shape[0], n_pixels), jnp.complex64)
    operands = (projector_half, rot6, logical_r_max)
    if image_r_max is not None:
        operands += (image_r_max,)
    return jax.ffi.ffi_call(
        target, output, vmap_method="sequential"
    )(
        *operands,
        image_h=np.int64(image_shape[0]),
        image_w=np.int64(image_shape[1]),
        padding_factor=np.int64(padding_factor),
    )


class RelionCapacityHalfTextureF32:
    """:func:`project_relion_half_capacity` with the texture staged once.

    ``project_relion_half_capacity`` stages the projector into a CUDA texture,
    projects, synchronizes the stream and frees the texture on every call. This
    owner stages ``projector_half`` at ``logical_r_max`` once (the same fill
    kernel and texture arrays) and :meth:`project` launches the same projection
    kernel on it with no synchronization, so it returns what
    ``project_relion_half_capacity(projector_half, rotations, logical_r_max,
    ...)`` returns. :meth:`close` waits for the last eager projection before
    the texture is destroyed. For a compiled caller, keep this owner alive
    across every invocation and call :meth:`close_after` with a concrete
    completed result from the final invocation. A trace-time projection is
    not evidence that the compiled GPU work has completed.
    """

    def __init__(
        self, projector_half: jax.Array, logical_r_max: int, *, padding_factor: int,
        reusable_staging: bool = False,
    ) -> None:
        projector_half = jnp.asarray(projector_half)
        shape = tuple(int(v) for v in projector_half.shape)
        if projector_half.dtype != jnp.complex64 or len(shape) != 3:
            raise ValueError("projector_half must be C64 [pf*Q+3,pf*Q+3,pf*Q//2+2]")
        if jax.default_backend() != "gpu" or not custom_cuda_requested():
            raise RuntimeError("a persistent capacity projector texture requires the custom CUDA GPU backend")
        library.ensure_optional_ffi(library.TARGET_PROJECT_RELION_HALF_CAPACITY_TEXTURE)
        (device,) = projector_half.devices()
        device_ordinal = int(getattr(device, "local_hardware_id", getattr(device, "id", -1)))
        # The C API reads the device buffer directly on its own stream.
        projector_half = jax.block_until_ready(projector_half)
        lib = library.get_lib()
        create = (
            lib.relax_relion_capacity_half_texture_f32_create_reusable
            if reusable_staging else lib.relax_relion_capacity_half_texture_f32_create
        )
        create.argtypes = (ctypes.c_void_p,) + (ctypes.c_int,) * 6 + (ctypes.POINTER(ctypes.c_uint64),)
        create.restype = ctypes.c_int
        handle = ctypes.c_uint64()
        status = int(
            create(
                ctypes.c_void_p(projector_half.unsafe_buffer_pointer()),
                *(ctypes.c_int(v) for v in shape),
                ctypes.c_int(logical_r_max),
                ctypes.c_int(padding_factor),
                ctypes.c_int(device_ordinal),
                ctypes.byref(handle),
            )
        )
        if status != 0 or int(handle.value) == 0:
            raise RuntimeError(f"failed to create a persistent capacity projector texture: CUDA error code {status}")
        self.shape = shape
        self.logical_r_max = logical_r_max
        self.padding_factor = padding_factor
        self.reusable_staging = bool(reusable_staging)
        self.device = device
        self._handle = int(handle.value)
        self._last_output = None
        self._finalizer = weakref.finalize(self, _destroy_capacity_half_texture, self._handle)

    @property
    def closed(self) -> bool:
        return self._handle == 0

    def project(
        self,
        rotation_matrices: jax.Array,
        *,
        image_shape: Tuple[int, int],
        image_r_max: jax.Array | None = None,
    ) -> jax.Array:
        """C64 [rotation, H*(W//2+1)], as :func:`project_relion_half_capacity` returns it."""

        if self.closed:
            raise RuntimeError("the persistent capacity projector texture is closed")
        rotation_matrices = jnp.asarray(rotation_matrices)
        if rotation_matrices.dtype != jnp.float32 or rotation_matrices.ndim != 3 or rotation_matrices.shape[1:] != (3, 3):
            raise ValueError("rotation_matrices must be F32 [rotation,3,3]")
        radius = (
            jnp.zeros((), dtype=jnp.int32) if image_r_max is None else jnp.asarray(image_r_max, dtype=jnp.int32)
        )
        output = _project_relion_half_capacity_texture(
            rotation_matrices,
            radius,
            owner_handle=self._handle,
            image_shape=(int(image_shape[0]), int(image_shape[1])),
            has_image_radius=image_r_max is not None,
        )
        self._last_output = output
        return output

    def project_compact_packed(
        self,
        rotation_matrices: jax.Array,
        crop_index: jax.Array,
        *,
        image_shape: Tuple[int, int],
        image_r_max: jax.Array,
        scale: float,
    ) -> jax.Array:
        """F32 ``[rotation, 2 P]``: :meth:`project` at the crop pixels ``crop_index`` ``[P]`` (-1 for a zero entry),
        times ``scale``, real parts then imaginary parts (the coarse GEMM scorer's packed operand)."""

        if self.closed:
            raise RuntimeError("the persistent capacity projector texture is closed")
        rotation_matrices = jnp.asarray(rotation_matrices)
        if rotation_matrices.dtype != jnp.float32 or rotation_matrices.ndim != 3 or rotation_matrices.shape[1:] != (3, 3):
            raise ValueError("rotation_matrices must be F32 [rotation,3,3]")
        library.ensure_optional_ffi(library.TARGET_PROJECT_RELION_HALF_CAPACITY_TEXTURE_COMPACT_PACKED)
        output = _project_relion_half_capacity_texture_compact_packed(
            rotation_matrices,
            jnp.asarray(image_r_max, dtype=jnp.int32),
            jnp.asarray(crop_index, dtype=jnp.int32),
            owner_handle=self._handle,
            image_shape=(int(image_shape[0]), int(image_shape[1])),
            scale=float(scale),
        )
        self._last_output = output
        return output

    def close_after(self, completion) -> None:
        """Destroy after a concrete result depending on the last use is ready.

        A compiled executable captures this owner's native handle. The caller
        must not invoke that executable again after closing the owner.
        """
        if self.closed:
            return
        leaves = jax.tree_util.tree_leaves(completion)
        if not leaves or any(isinstance(leaf, jax.core.Tracer) for leaf in leaves):
            raise ValueError("texture completion must be a concrete result, not a tracer")
        jax.block_until_ready(completion)
        handle = self._handle
        _destroy_capacity_half_texture(handle, raise_on_error=True)
        self._handle = 0
        self._last_output = None
        self._finalizer.detach()

    def refresh_after(self, projector_half: jax.Array, completion, *, logical_r_max: int | None = None) -> float:
        """Refill the same texture at an iteration boundary and return elapsed seconds.

        ``completion`` must depend on the last compiled use of this owner;
        the caller must not launch another use until this synchronous refill
        finishes. A compiled executable may then be invoked again unchanged.
        Only owners created with ``reusable_staging=True`` support refresh.
        """
        if self.closed or not self.reusable_staging:
            raise RuntimeError("texture refresh requires a live reusable-staging owner")
        leaves = jax.tree_util.tree_leaves(completion)
        if not leaves or not any(isinstance(leaf, jax.Array) for leaf in leaves) or any(
            isinstance(leaf, jax.core.Tracer) for leaf in leaves
        ):
            raise ValueError("refresh completion must contain a concrete result of the final invocation")
        if isinstance(projector_half, jax.core.Tracer):
            raise ValueError("refresh projector must be concrete")
        projector_half = jnp.asarray(projector_half)
        if projector_half.dtype != jnp.complex64 or tuple(projector_half.shape) != self.shape:
            raise ValueError("refresh projector must be C64 with the owner's capacity shape")
        if projector_half.devices() != {self.device}:
            raise ValueError("refresh projector must be on the owner's CUDA device")
        radius = self.logical_r_max if logical_r_max is None else int(logical_r_max)
        if radius < 0 or radius > (self.shape[1] // 2 - 1) // self.padding_factor:
            raise ValueError("refresh logical radius is outside the owner's capacity")
        start = time.perf_counter()
        jax.block_until_ready(completion)
        projector_half = jax.block_until_ready(projector_half)
        refresh = library.get_lib().relax_relion_capacity_half_texture_f32_refresh
        refresh.argtypes = (ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
        refresh.restype = ctypes.c_int
        status = int(refresh(
            ctypes.c_uint64(self._handle),
            ctypes.c_void_p(projector_half.unsafe_buffer_pointer()),
            ctypes.c_int(radius),
            ctypes.c_int(int(getattr(self.device, "local_hardware_id", getattr(self.device, "id", -1)))),
        ))
        if status != 0:
            raise RuntimeError(f"failed to refresh capacity projector texture: CUDA error code {status}")
        self.logical_r_max = radius
        self._last_output = None
        return time.perf_counter() - start

    def close(self) -> None:
        if self.closed:
            return
        if self._last_output is not None:
            self.close_after(self._last_output)
            return
        handle = self._handle
        _destroy_capacity_half_texture(handle, raise_on_error=True)
        self._handle = 0
        self._finalizer.detach()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _destroy_capacity_half_texture(handle: int, *, raise_on_error: bool = False) -> None:
    destroy = library.get_lib().relax_relion_capacity_half_texture_f32_destroy
    destroy.argtypes = (ctypes.c_uint64,)
    destroy.restype = ctypes.c_int
    status = int(destroy(ctypes.c_uint64(int(handle))))
    if status != 0 and raise_on_error:
        raise RuntimeError(f"failed to destroy a persistent capacity projector texture: CUDA error code {status}")


@functools.partial(jax.jit, static_argnames=("owner_handle", "image_shape", "has_image_radius"))
def _project_relion_half_capacity_texture(rotation_matrices, image_r_max, *, owner_handle, image_shape, has_image_radius):
    n_pixels = image_shape[0] * (image_shape[1] // 2 + 1)
    output = jax.ShapeDtypeStruct((rotation_matrices.shape[0], n_pixels), jnp.complex64)
    return jax.ffi.ffi_call(library.TARGET_PROJECT_RELION_HALF_CAPACITY_TEXTURE, output, vmap_method="sequential")(
        _rot_to_compact(rotation_matrices, jnp.float32),
        image_r_max,
        owner_handle=np.int64(owner_handle),
        image_h=np.int64(image_shape[0]),
        image_w=np.int64(image_shape[1]),
        has_image_radius=np.int64(bool(has_image_radius)),
    )


@functools.partial(jax.jit, static_argnames=("owner_handle", "image_shape", "scale"))
def _project_relion_half_capacity_texture_compact_packed(
    rotation_matrices, image_r_max, crop_index, *, owner_handle, image_shape, scale
):
    output = jax.ShapeDtypeStruct((rotation_matrices.shape[0], 2 * crop_index.shape[0]), jnp.float32)
    return jax.ffi.ffi_call(library.TARGET_PROJECT_RELION_HALF_CAPACITY_TEXTURE_COMPACT_PACKED, output, vmap_method="sequential")(
        _rot_to_compact(rotation_matrices, jnp.float32),
        image_r_max,
        crop_index,
        owner_handle=np.int64(owner_handle),
        image_h=np.int64(image_shape[0]),
        image_w=np.int64(image_shape[1]),
        scale=np.float32(scale),
    )


def _persistent_relion_half_texture_c_api():
    lib = library.get_lib()
    create = lib.recovar_relion_persistent_half_texture_f32_create
    create.argtypes = (
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_uint64),
    )
    create.restype = ctypes.c_int
    destroy = lib.recovar_relion_persistent_half_texture_f32_destroy
    destroy.argtypes = (ctypes.c_uint64,)
    destroy.restype = ctypes.c_int
    return create, destroy


def _destroy_persistent_relion_half_texture_handle(
    owner_handle: int,
    *,
    raise_on_error: bool,
) -> None:
    if not owner_handle:
        return
    try:
        _, destroy = _persistent_relion_half_texture_c_api()
        status = int(destroy(ctypes.c_uint64(int(owner_handle))))
    except Exception:
        if raise_on_error:
            raise
        return
    if status != 0 and raise_on_error:
        raise RuntimeError(
            "failed to destroy persistent RELION half texture: "
            f"CUDA error code {status}"
        )


class RelionPersistentHalfTextureF32:
    """One host-uploaded RELION ``Projector::data`` texture.

    The input must already be a C-contiguous ``complex64`` host slab.  It is
    copied synchronously into a CUDA array exactly once; projection calls pass
    only a dynamic opaque handle plus their rotation rows to XLA FFI.  Handles
    are never reused by the CUDA registry, so an executable retained in JAX's
    compilation cache cannot alias a later texture after :meth:`close`.
    """

    def __init__(
        self,
        projector_half: np.ndarray,
        *,
        padding_factor: int,
        projector_max_r: int,
        projector_scale: float = 1.0,
        device=None,
    ) -> None:
        if not isinstance(projector_half, np.ndarray):
            raise TypeError(
                "persistent RELION half texture requires a NumPy host array"
            )
        if projector_half.dtype != np.dtype(np.complex64):
            raise TypeError(
                "persistent RELION half texture requires complex64 host data, "
                f"got {projector_half.dtype}"
            )
        if projector_half.ndim != 3 or not projector_half.flags.c_contiguous:
            raise ValueError(
                "persistent RELION half texture requires a C-contiguous "
                f"(z, y, x-half) slab, got shape={projector_half.shape} "
                f"c_contiguous={projector_half.flags.c_contiguous}"
            )
        if padding_factor <= 0 or projector_max_r <= 0:
            raise ValueError("padding_factor and projector_max_r must be positive")
        if projector_scale != 1.0:
            raise ValueError(
                "persistent host-uploaded RELION texture is exact only for "
                f"projector_scale=1.0, got {projector_scale}"
            )
        padded_max_r = projector_max_r * padding_factor
        expected_shape = (
            2 * padded_max_r + 3,
            2 * padded_max_r + 3,
            padded_max_r + 2,
        )
        if projector_half.shape != expected_shape:
            raise ValueError(
                "persistent RELION half texture geometry mismatch: "
                f"got {projector_half.shape}, expected {expected_shape}"
            )
        if jax.default_backend() != "gpu":
            raise RuntimeError("persistent RELION half texture requires a JAX GPU backend")
        if not custom_cuda_requested():
            raise RuntimeError(
                "persistent RELION half texture requires the custom CUDA extension"
            )

        gpu_devices = jax.devices("gpu")
        if not gpu_devices:
            raise RuntimeError("persistent RELION half texture requires a JAX GPU device")
        if device is None:
            device = gpu_devices[0]
        if device not in gpu_devices:
            raise ValueError("persistent RELION half texture device is not a local JAX GPU")
        device_ordinal = int(
            getattr(device, "local_hardware_id", getattr(device, "id", -1))
        )
        if device_ordinal < 0:
            raise ValueError("could not resolve the CUDA ordinal for the JAX GPU device")

        library.ensure_ffi()
        create, _ = _persistent_relion_half_texture_c_api()
        owner_handle = ctypes.c_uint64()
        status = int(
            create(
                projector_half.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(expected_shape[2]),
                ctypes.c_int(expected_shape[1]),
                ctypes.c_int(expected_shape[0]),
                ctypes.c_int(device_ordinal),
                ctypes.c_float(projector_scale),
                ctypes.byref(owner_handle),
            )
        )
        if status != 0 or int(owner_handle.value) == 0:
            raise RuntimeError(
                "failed to create persistent RELION half texture: "
                f"CUDA error code {status}"
            )

        self.shape = tuple(int(value) for value in expected_shape)
        self.dtype = np.dtype(np.complex64)
        self.padding_factor = padding_factor
        self.projector_max_r = projector_max_r
        self.projector_scale = projector_scale
        self.device = device
        self.device_ordinal = device_ordinal
        self._owner_handle = int(owner_handle.value)
        self._handle_array = None
        self._last_output = None
        self._lock = threading.RLock()
        try:
            handle_array = jax.device_put(
                np.asarray(self._owner_handle, dtype=np.uint64),
                device,
            )
            if handle_array.dtype != jnp.uint64 or handle_array.shape != ():
                raise RuntimeError(
                    "persistent RELION texture handles require scalar JAX uint64"
                )
            self._handle_array = jax.block_until_ready(handle_array)
        except Exception:
            _destroy_persistent_relion_half_texture_handle(
                self._owner_handle,
                raise_on_error=False,
            )
            self._owner_handle = 0
            raise
        self._finalizer = weakref.finalize(
            self,
            _destroy_persistent_relion_half_texture_handle,
            self._owner_handle,
            raise_on_error=False,
        )

    @property
    def closed(self) -> bool:
        return self._owner_handle == 0


    def _require_live_geometry(
        self,
        *,
        padding_factor: int,
        projector_max_r: int,
    ) -> jax.Array:
        if self.closed or self._handle_array is None:
            raise RuntimeError("persistent RELION half texture is closed")
        if (
            int(padding_factor) != self.padding_factor
            or int(projector_max_r) != self.projector_max_r
        ):
            raise ValueError(
                "persistent RELION half texture geometry does not match the "
                "projection request"
            )
        return self._handle_array

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            readiness_error = None
            if self._last_output is not None:
                try:
                    jax.block_until_ready(self._last_output)
                except Exception as exc:  # still destroy after a failed launch
                    readiness_error = exc
            owner_handle = self._owner_handle
            self._owner_handle = 0
            self._handle_array = None
            self._last_output = None
            self._finalizer.detach()
            _destroy_persistent_relion_half_texture_handle(
                owner_handle,
                raise_on_error=True,
            )
            if readiness_error is not None:
                raise readiness_error

    def __enter__(self):
        if self.closed:
            raise RuntimeError("persistent RELION half texture is closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


@functools.partial(
    jax.jit,
    static_argnames=("current_size", "padding_factor", "projector_max_r"),
)
def _relion_projector_persistent_half_texture_f32(
    owner_handle: jax.Array,
    rotation_matrices: jax.Array,
    *,
    current_size: int,
    padding_factor: int,
    projector_max_r: int,
) -> jax.Array:
    compact_rotations = _rot_to_compact(rotation_matrices, jnp.float32)
    output_type = jax.ShapeDtypeStruct(
        (
            rotation_matrices.shape[0],
            current_size * (current_size // 2 + 1),
        ),
        jnp.complex64,
    )
    return jax.ffi.ffi_call(
        library.TARGET_RELION_PROJECTOR_PERSISTENT_HALF_TEXTURE_F32,
        output_type,
        vmap_method="sequential",
    )(
        owner_handle,
        compact_rotations,
        current_size=np.int64(current_size),
        padding_factor=np.int64(padding_factor),
        projector_max_r=np.int64(projector_max_r),
    )


def relion_projector_persistent_half_texture_f32(
    texture: RelionPersistentHalfTextureF32,
    rotation_matrices: jax.Array,
    *,
    current_size: int,
    padding_factor: int,
    projector_max_r: int,
) -> jax.Array:
    """Project through one explicitly owned host-uploaded RELION texture."""

    if not isinstance(texture, RelionPersistentHalfTextureF32):
        raise TypeError("texture must be a RelionPersistentHalfTextureF32")
    rotation_matrices = jnp.asarray(rotation_matrices)
    if rotation_matrices.dtype != jnp.float32:
        raise TypeError(
            "RELION half-texture rotations must be float32, got "
            f"{rotation_matrices.dtype}"
        )
    if (
        int(current_size) <= 0
        or rotation_matrices.ndim != 3
        or rotation_matrices.shape[0] <= 0
        or rotation_matrices.shape[1:] != (3, 3)
    ):
        raise ValueError(
            "persistent RELION half-texture projection operands have "
            f"inconsistent shapes: rotations={rotation_matrices.shape}, "
            f"current_size={current_size}"
        )
    with texture._lock:
        owner_handle = texture._require_live_geometry(
            padding_factor=padding_factor,
            projector_max_r=projector_max_r,
        )
        output = _relion_projector_persistent_half_texture_f32(
            owner_handle,
            rotation_matrices,
            current_size=int(current_size),
            padding_factor=int(padding_factor),
            projector_max_r=int(projector_max_r),
        )
        texture._last_output = output
        output = jax.block_until_ready(output)
        texture._last_output = None
        return output


@functools.partial(
    jax.jit,
    static_argnames=(
        "current_size",
        "padding_factor",
        "projector_max_r",
        "projector_scale",
    ),
)
def relion_projector_half_texture_f32(
    projector_half: jax.Array,
    rotation_matrices: jax.Array,
    *,
    current_size: int,
    padding_factor: int,
    projector_max_r: int,
    projector_scale: float = 1.0,
) -> jax.Array:
    """Project native RELION half storage through one interleaved texture.

    The output is a centered-row, rfft-packed square image for every input
    rotation. Keeping ``Projector::data`` in ``(z, y, x-half)`` form avoids a
    full cubic JAX expansion and the split real/imaginary texture staging used
    by the legacy generic projector.
    """

    projector_half = jnp.asarray(projector_half)
    rotation_matrices = jnp.asarray(rotation_matrices)
    expected_pad = 2 * projector_max_r * padding_factor + 3
    expected_shape = (
        expected_pad,
        expected_pad,
        projector_max_r * padding_factor + 2,
    )
    if projector_half.dtype != jnp.complex64:
        raise TypeError(
            f"RELION half-texture projector must be complex64, got {projector_half.dtype}"
        )
    if rotation_matrices.dtype != jnp.float32:
        raise TypeError(
            "RELION half-texture rotations must be float32, got "
            f"{rotation_matrices.dtype}"
        )
    if (
        current_size <= 0
        or padding_factor <= 0
        or projector_max_r <= 0
        or projector_half.shape != expected_shape
        or rotation_matrices.ndim != 3
        or rotation_matrices.shape[0] <= 0
        or rotation_matrices.shape[1:] != (3, 3)
    ):
        raise ValueError(
            "RELION half-texture projection operands have inconsistent shapes: "
            f"projector={projector_half.shape}, expected={expected_shape}, "
            f"rotations={rotation_matrices.shape}, current_size={current_size}"
        )
    if not np.isfinite(projector_scale):
        raise ValueError(f"projector_scale must be finite, got {projector_scale}")
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION half-texture projection requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION half-texture projection requires the custom CUDA extension"
        )
    library.ensure_ffi()
    compact_rotations = _rot_to_compact(rotation_matrices, jnp.float32)
    output_type = jax.ShapeDtypeStruct(
        (
            rotation_matrices.shape[0],
            current_size * (current_size // 2 + 1),
        ),
        jnp.complex64,
    )
    return jax.ffi.ffi_call(
        library.TARGET_RELION_PROJECTOR_HALF_TEXTURE_F32,
        output_type,
        vmap_method="sequential",
    )(
        projector_half,
        compact_rotations,
        current_size=np.int64(current_size),
        padding_factor=np.int64(padding_factor),
        projector_max_r=np.int64(projector_max_r),
        projector_scale=np.float32(projector_scale),
    )
