"""RELION projector construction shared by EM and VDAM.

Matches Projector::computeFourierTransformMap for a 3-D RELION-frame real
reference, data_dim=2 and trilinear gridding correction. No axis/contrast
conversion is performed by the device kernel. Host wrappers below convert
RECOVAR references and select the native or JAX implementation.

There is one device build (``_build_projector_window``): the padded transform
taken one axis at a time inside a static window of radius w, giving a
``(L, L, L // 2 + 1)`` slab with ``L = 2 (pf w + 1) + 1``. Jitted callers keep
the fixed capacity w = N / 2, ``(M + 3, M + 3, M // 2 + 2)`` for
``M = padding_factor * ori_size``, so the logical radius remains a device scalar;
the host wrapper uses w = r_max and runs large boxes in chunks. The native binding
is the test reference and the default of the host wrapper.
"""

from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from relax.helpers.deterministic_reduce import (
    deterministic_reductions_enabled,
    fixed_order_shell_sums,
    static_shell_voxel_lists,
)
from relax.helpers.fourier_window import stable_fourier_window_current_size, stable_fourier_window_quantum
from relax.relion.relion_project import gridding_correct_volume_real


@partial(jax.jit, static_argnames=("dtype",))
def swap_relion_volume_layout(volume, dtype=jnp.float64):
    """``recovar_volume_to_relion`` on the device: ``-transpose(2, 1, 0)``, its own inverse, cast to ``dtype``.

    Exact like the host helper; the host's transposed copy of a 256^3 map takes about 130 ms.
    """
    return -jnp.transpose(jnp.asarray(volume, dtype), (2, 1, 0))


@partial(jax.jit, static_argnames=("ori_size", "padding_factor", "compute_dtype", "window_radius"))
def setup_relion_projector(
    reference_relion,
    r_max,
    *,
    ori_size: int,
    padding_factor: int = 1,
    do_gridding=True,
    compute_dtype=jnp.float64,
    window_radius: int | None = None,
):
    """Return corrected projector data and power at ``compute_dtype``.

    ``r_max`` is the unpadded logical radius; negative means original Nyquist,
    zero means DC only, and larger radii are clamped like native initialiseData.
    The slab covers the static window of ``window_radius`` (default ``ori_size // 2``,
    the fixed capacity), so the radius may change without recompiling; ``r_max``
    must not exceed the window. For native comparison, crop the centered y/z
    region of width ``2*(padding_factor*r_max+1)+1`` and the corresponding
    positive-x prefix. The default is the existing float64 native-oracle route.
    Opt-in float32 performs gridding correction, FFT and power calculation in float32.
    See ``docs/math/vdam_ppca_algorithm.md`` for the production precision path.

    The global RECOVAR x64 policy is required. No Python callbacks, host
    materialization, persistent mutable cache, or per-class batching is used.
    """
    reference = _checked_reference(reference_relion, ori_size, padding_factor, compute_dtype)
    reference = jax.lax.cond(
        jnp.asarray(do_gridding, dtype=jnp.bool_),
        lambda volume: gridding_correct_volume_real(volume, ori_size, padding_factor),
        lambda volume: volume,
        reference,
    )
    window = ori_size // 2 if window_radius is None else window_radius
    return _build_projector_window(reference, r_max, ori_size, padding_factor, window)


@partial(jax.jit, static_argnames=("ori_size", "padding_factor", "compute_dtype"))
def setup_relion_projector_uncorrected(
    reference_relion,
    r_max,
    *,
    ori_size: int,
    padding_factor: int = 1,
    compute_dtype=jnp.float64,
):
    """Prepare an uncorrected M-step projector at the requested precision.

    The FFT, power reductions and shell geometry use ``compute_dtype``;
    outputs are complex64/float32 or complex128/float64. This static path
    does not trace the double-precision gridding-correction branch used by
    the existing E-step wrapper. Radius and Nyquist ownership are shared.
    """
    reference = _checked_reference(reference_relion, ori_size, padding_factor, compute_dtype)
    return _build_projector_window(reference, r_max, ori_size, padding_factor, ori_size // 2)


def setup_relion_projector_on_host(
    reference_relion,
    r_max: int,
    *,
    ori_size: int,
    padding_factor: int = 1,
    compute_dtype=jnp.float64,
    chunk_bytes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`setup_relion_projector` for one ``r_max``, dispatched chunk by chunk to host arrays.

    Returns the slab ``[L, L, L // 2 + 1]`` for ``L = 2 (pf r + 1) + 1``, the
    shape the native binding returns, and the shell power. The transform runs in
    the stable Fourier-window class of ``2 r`` (``stable_fourier_window_current_size``,
    the quantum the pass-2 engines use), so consecutive iterations whose current
    size stays in one class reuse the compiled programs; each chunk is cropped to
    ``L`` on its way to the host. The chunks are separate device programs, so a
    large box never holds the whole transform: at EMPIAR-10202 (box 800, padding
    2) the full-capacity transform needed about 100 GB and the build fell back to
    two single-threaded host builds (117 s per iteration, bigbox py-spy 14561585).
    """
    reference = _checked_reference(reference_relion, ori_size, padding_factor, compute_dtype)
    radius = ori_size // 2 if int(r_max) < 0 else min(int(r_max), ori_size // 2)
    window = radius
    if 0 < radius < ori_size // 2:
        quantum = stable_fourier_window_quantum()
        window = stable_fourier_window_current_size(2 * radius, ori_size, quantum=quantum) // 2
    reference = _gridding_corrected(reference, ori_size=ori_size, padding_factor=padding_factor)
    return _build_projector_window(
        reference, radius, ori_size, padding_factor, window,
        chunk_bytes=chunk_bytes, to_host=True, output_radius=radius,
    )


@partial(jax.jit, static_argnames=("ori_size", "padding_factor"))
def _gridding_corrected(reference, *, ori_size: int, padding_factor: int):
    """:func:`gridding_correct_volume_real` as one fused program.

    Called eagerly it materializes its coordinate grids, radius and sinc as
    separate volumes: about 31 GiB of float64 temporaries at box 800, the peak of
    the whole host build (bigbox 14575379).
    """

    return gridding_correct_volume_real(reference, ori_size, padding_factor)


def _checked_reference(reference_relion, ori_size, padding_factor, compute_dtype):
    if ori_size <= 0 or ori_size % 2:
        raise ValueError("ori_size must be positive and even")
    if padding_factor not in (1, 2):
        raise ValueError("projector setup supports padding_factor 1 or 2")
    if reference_relion.shape != (ori_size,) * 3:
        raise ValueError("reference_relion must have shape (ori_size,)*3")
    dtype = jnp.dtype(compute_dtype)
    if dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise ValueError("Projector computation dtype must be float32 or float64")
    if dtype == jnp.dtype(jnp.float64) and not jax.config.x64_enabled:
        raise ValueError("Float64 projector setup requires JAX float64 support")
    return jnp.asarray(reference_relion, dtype=dtype)


# Complex working set of one chunk of the transform. Boxes up to padded 512 are one chunk.
_CHUNK_BYTES = 2 * 1024**3


def _build_projector_window(
    reference, r_max, ori_size, padding_factor, window_radius, *, chunk_bytes=None, to_host=False, output_radius=None
):
    """RELION's computeFourierTransformMap evaluated inside a static window, one axis at a time.

    Native computes ``rfftn(fftshift(padded), norm="forward")`` of the padded
    reference, scales by ``pf^3 N`` and keeps the half grid inside the sphere of
    ``pf r_max``. This takes the x rfft, the y FFT (over z-slabs) and the z FFT
    (over y-slabs) in turn, keeping after each axis only the frequencies inside
    the window ``L = 2 (pf w + 1) + 1`` of ``window_radius`` w. The unpadded half
    of every padded axis is zero, so only the reference's own voxels are
    transformed. With w = N / 2 the window is the fixed capacity
    ``(M + 3, M + 3, M // 2 + 2)``; that costs 0.99-1.04x the whole-volume rfftn at
    padded 128-512 and agrees with it to 4.8e-16 (bigbox 14563130), and a smaller
    window is proportionally cheaper. Only the order of the three one-dimensional
    passes differs from native; tests/unit/test_relion_projector_setup.py holds
    the float64 bound. ``to_host`` collects each chunk on the host (eager calls
    only), cropped to the window of ``output_radius`` when that is smaller.
    """
    n, pf = int(ori_size), int(padding_factor)
    m = pf * n
    size = 2 * (pf * int(window_radius) + 1) + 1
    n_x = size // 2 + 1
    # ftu centers y/z with -Nyquist at index zero. RELION's FFTW traversal
    # instead gives that same coefficient +Nyquist; never duplicate it at -N/2.
    yz_index = jnp.asarray((np.arange(size) - size // 2) % m, dtype=jnp.int32)
    chunk_bytes = _CHUNK_BYTES if chunk_bytes is None else int(chunk_bytes)

    cz = max(1, chunk_bytes // (16 * m * m))
    if to_host and cz < n:
        # Eager chunks are written into one preallocated array in place, so the
        # xy stage never holds its chunks and their concatenation together (2 x
        # 15.3 GiB at EMPIAR-10202's full box, current size 800).
        xy = None
        for z0 in range(0, n, cz):
            block = _transform_xy(reference[z0 : z0 + cz], yz_index, fft_size=m, n_x=n_x)
            if xy is None:
                xy = jnp.zeros((n,) + block.shape[1:], dtype=block.dtype)
            xy = _place_z_block(xy, block, z0)
            del block
    else:
        xy = [_transform_xy(reference[z0 : z0 + cz], yz_index, fft_size=m, n_x=n_x) for z0 in range(0, n, cz)]
        xy = xy[0] if len(xy) == 1 else jnp.concatenate(xy, axis=0)
    cy = max(1, chunk_bytes // (16 * m * n_x))
    out_size = size if output_radius is None else 2 * (pf * int(output_radius) + 1) + 1
    off = (size - out_size) // 2
    if not to_host and out_size != size:
        raise ValueError("a device slab keeps its window; crop on the host path only")
    blocks, slab, sums, counts = [], None, None, None
    for y0 in range(0, size, cy):
        block = _transform_z(xy[:, y0 : y0 + cy], yz_index, fft_size=m)
        block, block_sums, block_counts = _mask_and_shell_power(
            block, r_max, ori_size=n, padding_factor=pf, size=size, y_start=y0
        )
        if to_host:
            if slab is None:
                slab = np.empty((out_size, out_size, out_size // 2 + 1), dtype=block.dtype)
            lo, hi = max(y0, off), min(y0 + block.shape[1], off + out_size)
            if lo < hi:
                slab[:, lo - off : hi - off] = np.asarray(jax.device_get(block))[
                    off : off + out_size, lo - y0 : hi - y0, : out_size // 2 + 1
                ]
        else:
            blocks.append(block)
        sums = block_sums if sums is None else sums + block_sums
        counts = block_counts if counts is None else counts + block_counts
    del xy
    if to_host:
        projector = slab
    else:
        projector = blocks[0] if len(blocks) == 1 else jnp.concatenate(blocks, axis=1)
    spectrum = jnp.where(counts >= 1, sums / jnp.maximum(counts, 1), jnp.zeros((), dtype=sums.dtype))
    return (projector, np.asarray(jax.device_get(spectrum))) if to_host else (projector, spectrum)


def _wrap_pad(values, axis: int, fft_size: int):
    """Zero-pad a centered axis of length N to ``fft_size`` and apply ``fftshift``.

    That is what ``get_dft3_real`` does to the centred padded volume before its
    FFT: sample ``i`` of the unpadded axis lands at ``(i - N // 2) mod fft_size``.
    """

    n = values.shape[axis]
    head = jax.lax.slice_in_dim(values, n // 2, n, axis=axis)
    tail = jax.lax.slice_in_dim(values, 0, n // 2, axis=axis)
    shape = list(values.shape)
    shape[axis] = fft_size - n
    return jnp.concatenate([head, jnp.zeros(shape, values.dtype), tail], axis=axis)


@partial(jax.jit, static_argnames=("fft_size", "n_x"))
def _transform_xy(block, yz_index, *, fft_size: int, n_x: int):
    """x rfft then y FFT of a z-slab ``[cz, N, N]``, kept inside the window: ``[cz, L, n_x]``."""

    fx = jnp.fft.rfft(_wrap_pad(block, 2, fft_size), axis=2, norm="forward")
    # At the full radius the window reaches one column past Nyquist; like
    # native, read Nyquist there and let the validity mask zero it.
    fx = jnp.take(fx, jnp.minimum(jnp.arange(n_x), fft_size // 2), axis=2)
    fy = jnp.fft.fft(_wrap_pad(fx, 1, fft_size), axis=1, norm="forward")
    return jnp.take(fy, yz_index, axis=1)


@partial(jax.jit, donate_argnums=0)
def _place_z_block(xy, block, z0):
    """Write a z-slab of the xy stage into ``xy`` at ``z0``, in place (``xy`` is donated)."""

    return jax.lax.dynamic_update_slice_in_dim(xy, block, z0, axis=0)


@partial(jax.jit, static_argnames=("fft_size",))
def _transform_z(block, yz_index, *, fft_size: int):
    """z FFT of a y-slab ``[N, cy, n_x]`` of the xy stage, kept inside the window: ``[L, cy, n_x]``."""

    fz = jnp.fft.fft(_wrap_pad(block, 0, fft_size), axis=0, norm="forward")
    return jnp.take(fz, yz_index, axis=0)


@partial(jax.jit, static_argnames=("ori_size", "padding_factor", "size", "y_start"))
def _mask_and_shell_power(block, r_max, *, ori_size: int, padding_factor: int, size: int, y_start: int):
    """Scale, mask and shell-sum window rows ``y_start:`` (``block`` is ``[L, cy, n_x]``)."""

    fft_size = padding_factor * ori_size
    real_dtype = block.real.dtype
    # Native FourierTransformer divides by M^3, then projector.cpp multiplies
    # by pf^3*N for 3-D references projected into 2-D images. Keep that order.
    block = block * float(padding_factor**3 * ori_size)
    rows = block.shape[1]
    coord = jnp.arange(size, dtype=jnp.int32) - size // 2
    z = coord[:, None, None]
    y = coord[y_start : y_start + rows][None, :, None]
    x = jnp.arange(block.shape[2], dtype=jnp.int32)[None, None, :]
    r2 = z * z + y * y + x * x
    radius = jnp.asarray(r_max, dtype=jnp.int32)
    radius = jnp.where(radius < 0, ori_size // 2, jnp.minimum(radius, ori_size // 2))
    valid = (
        (z > -fft_size // 2)
        & (z <= fft_size // 2)
        & (y > -fft_size // 2)
        & (y <= fft_size // 2)
        & (x <= fft_size // 2)
        & (r2 <= (padding_factor * radius) ** 2)
    )
    projector = jnp.where(valid, block, jnp.asarray(0, dtype=block.dtype))
    # Native uses norm(complex)/2 rather than abs(complex)**2/2.
    power = (projector.real * projector.real + projector.imag * projector.imag) / 2.0
    n_shells = ori_size // 2 + 1
    if deterministic_reductions_enabled():
        # ``bincount`` lowers to a scatter-add with duplicate shells (float
        # atomics); use static per-shell gathers with fixed-order reductions.
        shell_lists = static_shell_voxel_lists(
            size, padding_factor, n_shells, clamp_to_last=True, rows=(y_start, y_start + rows)
        )
        sums = fixed_order_shell_sums(power.reshape(-1), shell_lists, real_dtype)
        counts = fixed_order_shell_sums(valid.reshape(-1).astype(real_dtype), shell_lists, real_dtype)
    else:
        shells = jnp.floor(jnp.sqrt(r2.astype(real_dtype)) / padding_factor + 0.5).astype(jnp.int32)
        shells = jnp.minimum(shells, ori_size // 2).reshape(-1)
        sums = jnp.bincount(shells, weights=power.reshape(-1), length=n_shells)
        counts = jnp.bincount(shells, weights=valid.reshape(-1).astype(real_dtype), length=n_shells)
    return projector, sums, counts


ProjectorSetupBackend = Literal["native", "jax"]


def reference_to_relion_projector_half_maps(
    references: np.ndarray,
    *,
    current_size: int,
    padding_factor: int = 1,
    interpolator: int = 1,
    projector_setup_backend: ProjectorSetupBackend = "jax",
    projector_data_dtype=None,
    compute_dtype=np.float64,
) -> tuple[np.ndarray, int]:
    """Convert references to RELION half maps without retaining their spectrum."""
    half_maps, _power, r_max = reference_to_relion_projector_half_maps_and_power(
        references,
        current_size=current_size,
        padding_factor=padding_factor,
        interpolator=interpolator,
        projector_data_dtype=projector_data_dtype,
        projector_setup_backend=projector_setup_backend,
        compute_dtype=compute_dtype,
    )
    return half_maps, r_max


def reference_to_relion_projector_half_maps_and_power(
    references: np.ndarray,
    *,
    current_size: int,
    padding_factor: int = 1,
    interpolator: int = 1,
    projector_setup_backend: ProjectorSetupBackend = "jax",
    projector_data_dtype=None,
    compute_dtype=np.float64,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Convert references to native-layout half maps and their corrected spectrum.

    The JAX backend (the default) is the device build of
    :func:`setup_relion_projector_on_host` and uses ``compute_dtype`` for
    gridding correction, FFT and power. The native binding is the test
    reference; unsupported geometry (odd box, padding other than 1 or 2, an
    interpolator other than trilinear) also takes it, for the float64 route only.

    ``projector_data_dtype`` is what the caller wants the slab in. ``None``
    keeps each backend's own output: complex64 from the JAX path, whose
    consumer is the InitialModel engine, and complex128 from the native
    binding, which is what refinement consumes. A caller that needs one
    precision from either backend must say so, because the two backends do not
    agree by default and switching backend would otherwise change precision
    silently.
    """
    from recovar.utils.helpers import recovar_volume_to_relion

    if projector_setup_backend not in {"native", "jax"}:
        raise ValueError(f"Unknown projector_setup_backend: {projector_setup_backend!r}")
    compute_dtype = np.dtype(compute_dtype)
    if compute_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("Projector computation dtype must be float32 or float64")
    # VDAM's references are device arrays; reading them back here is not needed on the JAX backend.
    refs = references if isinstance(references, jax.Array) else np.asarray(references)
    if refs.ndim != 4:
        raise ValueError(f"references must have shape (K, N, N, N), got {refs.shape}")
    n = int(refs.shape[-1])
    use_jax = (
        projector_setup_backend == "jax"
        and n > 0 and n % 2 == 0
        and refs.shape[1:] == (n, n, n)
        and int(padding_factor) in {1, 2}
        and int(interpolator) == 1
    )
    if compute_dtype == np.dtype(np.float32) and not use_jax:
        raise ValueError("Float32 projector setup requires supported JAX geometry and backend")
    if not use_jax:
        from relax.relion_bind import _relion_bind_core as bind

    halves = []
    power_spectra = []
    r_max_values = []
    for ref in refs:
        if use_jax:
            # Projector::initialiseData uses a negative size for full resolution;
            # zero means radius zero here (state wrappers retain their defaults).
            r_max = n // 2 if int(current_size) < 0 else min(int(current_size) // 2, n // 2)
            projector_data, power = setup_relion_projector_on_host(
                swap_relion_volume_layout(ref), r_max, ori_size=n,
                padding_factor=int(padding_factor), compute_dtype=compute_dtype.type,
            )
            if projector_data_dtype is None:
                projector_data = projector_data.astype(np.complex64)
        else:
            (
                projector_data, power, _ori_size, _padding_factor_out,
                r_max, _r_min_nn, _interpolator_out,
            ) = bind.compute_fourier_transform_map(
                np.asarray(recovar_volume_to_relion(ref), dtype=np.float64),
                n,
                int(padding_factor),
                int(interpolator),
                int(current_size),
                True,
                2,
            )
        if projector_data_dtype is not None:
            projector_data = np.asarray(projector_data).astype(
                np.dtype(projector_data_dtype), copy=False
            )
        halves.append(np.asarray(projector_data))
        power_spectra.append(np.asarray(power, dtype=np.float64))
        r_max_values.append(int(r_max))
    if len(set(r_max_values)) != 1:
        raise ValueError(f"RELION projector maps disagree on r_max: {r_max_values}")
    return (
        np.asarray(halves),
        np.asarray(power_spectra, dtype=np.float64),
        int(r_max_values[0]),
    )


def cast_relion_projector_for_execution(projector_half, *, use_float64_projections=False):
    """Select EM projection precision without changing the setup array.

    Cast host slabs before upload, including their optional class axis. Double
    projection is an explicit diagnostic; production projection uses complex64.
    """
    if projector_half is None:
        return None
    dtype = np.complex128 if use_float64_projections else np.complex64
    if isinstance(projector_half, np.ndarray):
        return projector_half.astype(dtype, copy=False)
    return jnp.asarray(projector_half, dtype=dtype)


def prepare_local_projector_slab(projector_half, *, path_label="local RELION projector path"):
    """Return one (z, y, x_half) slab, preserving the input's JAX dtype.

    Local scoring accepts a slab or a singleton class axis. This is shape
    normalization only: no Fourier conversion, interpolation or precision policy.
    A host singleton drops its class axis as a NumPy view before the upload, so
    only the 3-D slab reaches the device (no transient 4-D copy to slice).
    """
    if isinstance(projector_half, np.ndarray) and projector_half.ndim == 4 and int(projector_half.shape[0]) == 1:
        projector_half = np.reshape(projector_half, projector_half.shape[1:])
    slab = jnp.asarray(projector_half)
    if slab.ndim == 4:
        if int(slab.shape[0]) != 1:
            raise ValueError(
                f"{path_label} expected a single-class projector slab, got {slab.shape}",
            )
        slab = slab[0]
    if slab.ndim != 3:
        raise ValueError(
            f"{path_label} expected Projector::data shape (z, y, x_half), got {slab.shape}",
        )
    return slab
