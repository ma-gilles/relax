"""RELION projector construction shared by EM and VDAM.

Matches Projector::computeFourierTransformMap for a 3-D RELION-frame real
reference, data_dim=2 and trilinear gridding correction. No axis/contrast
conversion is performed by the device kernel. Host wrappers below convert
RECOVAR references.

There is one device build (``_build_projector_window``): the padded transform
taken one axis at a time inside a static window of radius w, giving a
``(L, L, L // 2 + 1)`` slab with ``L = 2 (pf w + 1) + 1``. Jitted callers keep
the fixed capacity w = N / 2, ``(M + 3, M + 3, M // 2 + 2)`` for
``M = padding_factor * ori_size``, so the logical radius remains a device scalar;
the host wrapper uses w = r_max and runs large boxes in chunks. RELION's own
transform (through the binding) is the test oracle, in
``scripts.lib.native_projector_setup``.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from relax.helpers.deterministic_reduce import (
    deterministic_reductions_enabled,
    fixed_order_shell_sums,
    static_shell_voxel_lists,
)
from relax.helpers.fourier_window import stable_fourier_window_current_size, stable_fourier_window_quantum
from relax.relion.relion_project import gridding_correct_volume_real, gridding_correct_volume_real_separable

# Real-space gridding-correction windows of the projector setup: RELION's radial
# sinc²(|x| / (pf N)) and the per-axis product, the trilinear kernel's exact transform.
GRIDDING_KERNELS = ("radial", "separable")


@partial(jax.jit, static_argnames=("dtype",))
def swap_relion_volume_layout(volume, dtype=jnp.float64):
    """``recovar_volume_to_relion`` on the device: ``-transpose(2, 1, 0)``, its own inverse, cast to ``dtype``.

    Exact like the host helper; the host's transposed copy of a 256^3 map takes about 130 ms.
    """
    return -jnp.transpose(jnp.asarray(volume, dtype), (2, 1, 0))


@partial(jax.jit, static_argnames=("box_size", "padding_factor", "compute_dtype", "window_radius"))
def setup_relion_projector(
    reference_relion,
    r_max,
    *,
    box_size: int,
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
    reference = _checked_reference(reference_relion, box_size, padding_factor, compute_dtype)
    reference = jax.lax.cond(
        jnp.asarray(do_gridding, dtype=jnp.bool_),
        lambda volume: gridding_correct_volume_real(volume, box_size, padding_factor),
        lambda volume: volume,
        reference,
    )
    window = box_size // 2 if window_radius is None else window_radius
    return _build_projector_window(reference, r_max, box_size, padding_factor, window)


@partial(jax.jit, static_argnames=("box_size", "padding_factor", "compute_dtype"))
def setup_relion_projector_uncorrected(
    reference_relion,
    r_max,
    *,
    box_size: int,
    padding_factor: int = 1,
    compute_dtype=jnp.float64,
):
    """Prepare an uncorrected M-step projector at the requested precision.

    The FFT, power reductions and shell geometry use ``compute_dtype``;
    outputs are complex64/float32 or complex128/float64. This static path
    does not trace the double-precision gridding-correction branch used by
    the existing E-step wrapper. Radius and Nyquist ownership are shared.
    """
    reference = _checked_reference(reference_relion, box_size, padding_factor, compute_dtype)
    return _build_projector_window(reference, r_max, box_size, padding_factor, box_size // 2)


def setup_relion_projector_on_host(
    reference_relion,
    r_max: int,
    *,
    box_size: int,
    padding_factor: int = 1,
    compute_dtype=jnp.float64,
    chunk_bytes: int | None = None,
    gridding_kernel: str = "radial",
    shell_pair_counting: str = "relion",
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

    ``gridding_kernel`` selects the real-space correction window: RELION's
    ``"radial"`` one or the ``"separable"`` per-axis product.
    ``shell_pair_counting`` is how the shell power counts the Hermitian pairs of
    the stored half: ``"relion"`` counts every stored coefficient once, so the
    pairs of the ``x = 0`` plane twice; ``"once"`` counts every pair once.
    """
    if gridding_kernel not in GRIDDING_KERNELS:
        raise ValueError(f"gridding_kernel must be one of {GRIDDING_KERNELS}, got {gridding_kernel!r}")
    if shell_pair_counting not in ("relion", "once"):
        raise ValueError(f"shell_pair_counting must be 'relion' or 'once', got {shell_pair_counting!r}")
    reference = _checked_reference(reference_relion, box_size, padding_factor, compute_dtype)
    radius = box_size // 2 if int(r_max) < 0 else min(int(r_max), box_size // 2)
    window = radius
    if 0 < radius < box_size // 2:
        quantum = stable_fourier_window_quantum()
        window = stable_fourier_window_current_size(2 * radius, box_size, quantum=quantum) // 2
    if gridding_kernel == "radial":
        reference = _gridding_corrected(reference, box_size=box_size, padding_factor=padding_factor)
    else:
        reference = _gridding_corrected_separable(reference, box_size=box_size, padding_factor=padding_factor)
    return _build_projector_window(
        reference, radius, box_size, padding_factor, window,
        chunk_bytes=chunk_bytes, to_host=True, output_radius=radius, pair_once=shell_pair_counting == "once",
    )


@partial(jax.jit, static_argnames=("box_size", "padding_factor"))
def _gridding_corrected(reference, *, box_size: int, padding_factor: int):
    """:func:`gridding_correct_volume_real` as one fused program.

    Called eagerly it materializes its coordinate grids, radius and sinc as
    separate volumes: about 31 GiB of float64 temporaries at box 800, the peak of
    the whole host build (bigbox 14575379).
    """

    return gridding_correct_volume_real(reference, box_size, padding_factor)


@partial(jax.jit, static_argnames=("box_size", "padding_factor"))
def _gridding_corrected_separable(reference, *, box_size: int, padding_factor: int):
    """:func:`gridding_correct_volume_real_separable` as one fused program."""

    return gridding_correct_volume_real_separable(reference, box_size, padding_factor)


def _checked_reference(reference_relion, box_size, padding_factor, compute_dtype):
    if box_size <= 0 or box_size % 2:
        raise ValueError("box_size must be positive and even")
    if padding_factor not in (1, 2):
        raise ValueError("projector setup supports padding_factor 1 or 2")
    if reference_relion.shape != (box_size,) * 3:
        raise ValueError("reference_relion must have shape (box_size,)*3")
    dtype = jnp.dtype(compute_dtype)
    if dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise ValueError("Projector computation dtype must be float32 or float64")
    if dtype == jnp.dtype(jnp.float64) and not jax.config.x64_enabled:
        raise ValueError("Float64 projector setup requires JAX float64 support")
    return jnp.asarray(reference_relion, dtype=dtype)


# Complex working set of one chunk of the transform. Boxes up to padded 512 are one chunk.
_CHUNK_BYTES = 2 * 1024**3


def _build_projector_window(
    reference, r_max, box_size, padding_factor, window_radius, *, chunk_bytes=None, to_host=False, output_radius=None,
    pair_once=False,
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
    ``pair_once`` counts each Hermitian pair once in the shell power
    (:func:`_mask_and_shell_power`); the projector data does not depend on it.
    """
    n, pf = int(box_size), int(padding_factor)
    m = pf * n
    size = 2 * (pf * int(window_radius) + 1) + 1
    n_x = size // 2 + 1
    # ftu centers y/z with -Nyquist at index zero. RELION's FFTW traversal
    # instead gives that same coefficient +Nyquist; never duplicate it at -N/2.
    yz_index = jnp.asarray((np.arange(size) - size // 2) % m, dtype=jnp.int32)
    chunk_bytes = _CHUNK_BYTES if chunk_bytes is None else int(chunk_bytes)

    cz = max(1, chunk_bytes // (16 * m * m))
    cy = max(1, chunk_bytes // (16 * m * n_x))
    out_size = size if output_radius is None else 2 * (pf * int(output_radius) + 1) + 1
    off = (size - out_size) // 2
    if not to_host and out_size != size:
        raise ValueError("a device slab keeps its window; crop on the host path only")
    if cz >= n and cy >= size:
        # One chunk on both axes (boxes up to padded 512): the three stages are one
        # program, which compiles once per window instead of three times.
        stages = [
            (0,)
            + _window_in_one_program(
                reference, yz_index, r_max, box_size=n, padding_factor=pf, size=size, n_x=n_x, pair_once=pair_once
            )
        ]
    else:
        stages = _window_in_chunks(
            reference, yz_index, r_max, box_size=n, padding_factor=pf, size=size, n_x=n_x,
            cz=cz, cy=cy, to_host=to_host, pair_once=pair_once,
        )
    blocks, slab, sums, counts = [], None, None, None
    for y0, block, block_sums, block_counts in stages:
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
    del stages, block
    if to_host:
        projector = slab
    else:
        projector = blocks[0] if len(blocks) == 1 else jnp.concatenate(blocks, axis=1)
    if pair_once:
        # A shell that holds only self-mated coefficients (the origin) has a count of 1/2.
        spectrum = jnp.where(counts > 0, sums / jnp.where(counts > 0, counts, 1), jnp.zeros((), dtype=sums.dtype))
    else:
        spectrum = jnp.where(counts >= 1, sums / jnp.maximum(counts, 1), jnp.zeros((), dtype=sums.dtype))
    return (projector, np.asarray(jax.device_get(spectrum))) if to_host else (projector, spectrum)


@partial(jax.jit, static_argnames=("box_size", "padding_factor", "size", "n_x", "pair_once"))
def _window_in_one_program(reference, yz_index, r_max, *, box_size, padding_factor, size, n_x, pair_once):
    """The window of a reference that fits one chunk on both axes: ``(block, sums, counts)``."""

    fft_size = padding_factor * box_size
    xy = _transform_xy(reference, yz_index, fft_size=fft_size, n_x=n_x)
    block = _transform_z(xy, yz_index, fft_size=fft_size)
    return _mask_and_shell_power(
        block, r_max, box_size=box_size, padding_factor=padding_factor, size=size, y_start=0, pair_once=pair_once
    )


def _window_in_chunks(reference, yz_index, r_max, *, box_size, padding_factor, size, n_x, cz, cy, to_host, pair_once):
    """The window in z-slabs, then y-slabs: yields ``(y_start, block, sums, counts)`` per y-slab."""

    n, m = int(box_size), int(padding_factor) * int(box_size)
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
    for y0 in range(0, size, cy):
        block = _transform_z(xy[:, y0 : y0 + cy], yz_index, fft_size=m)
        yield (y0,) + _mask_and_shell_power(
            block, r_max, box_size=n, padding_factor=padding_factor, size=size, y_start=y0, pair_once=pair_once
        )
        del block


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


@partial(jax.jit, static_argnames=("box_size", "padding_factor", "size", "y_start", "pair_once"))
def _mask_and_shell_power(
    block, r_max, *, box_size: int, padding_factor: int, size: int, y_start: int, pair_once: bool = False
):
    """Scale, mask and shell-sum window rows ``y_start:`` (``block`` is ``[L, cy, n_x]``).

    The shell sums visit every stored coefficient once, as ``computeFourierTransformMap`` does
    (projector.cpp:509-530), which counts the Hermitian pairs of the ``x = 0`` plane twice.
    ``pair_once`` weights the coefficients whose mate is stored too (``x = 0`` and the Nyquist
    ``x``) by 1/2 in the sums and the counts, so every pair counts once.
    """

    fft_size = padding_factor * box_size
    real_dtype = block.real.dtype
    # Native FourierTransformer divides by M^3, then projector.cpp multiplies
    # by pf^3*N for 3-D references projected into 2-D images. Keep that order.
    block = block * float(padding_factor**3 * box_size)
    rows = block.shape[1]
    coord = jnp.arange(size, dtype=jnp.int32) - size // 2
    z = coord[:, None, None]
    y = coord[y_start : y_start + rows][None, :, None]
    x = jnp.arange(block.shape[2], dtype=jnp.int32)[None, None, :]
    r2 = z * z + y * y + x * x
    radius = jnp.asarray(r_max, dtype=jnp.int32)
    radius = jnp.where(radius < 0, box_size // 2, jnp.minimum(radius, box_size // 2))
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
    counted = valid.astype(real_dtype)
    if pair_once:
        pair_weight = jnp.where((x == 0) | (x == fft_size // 2), 0.5, 1.0).astype(real_dtype)
        power = power * pair_weight
        counted = counted * pair_weight
    n_shells = box_size // 2 + 1
    if deterministic_reductions_enabled():
        # ``bincount`` lowers to a scatter-add with duplicate shells (float
        # atomics); use static per-shell gathers with fixed-order reductions.
        shell_lists = static_shell_voxel_lists(
            size, padding_factor, n_shells, clamp_to_last=True, rows=(y_start, y_start + rows)
        )
        sums = fixed_order_shell_sums(power.reshape(-1), shell_lists, real_dtype)
        counts = fixed_order_shell_sums(counted.reshape(-1), shell_lists, real_dtype)
    else:
        shells = jnp.floor(jnp.sqrt(r2.astype(real_dtype)) / padding_factor + 0.5).astype(jnp.int32)
        shells = jnp.minimum(shells, box_size // 2).reshape(-1)
        sums = jnp.bincount(shells, weights=power.reshape(-1), length=n_shells)
        counts = jnp.bincount(shells, weights=counted.reshape(-1), length=n_shells)
    return projector, sums, counts


def reference_to_relion_projector_half_maps(
    references: np.ndarray,
    *,
    current_size: int,
    padding_factor: int = 1,
    interpolator: int = 1,
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
        compute_dtype=compute_dtype,
    )
    return half_maps, r_max


def reference_to_relion_projector_half_maps_and_power(
    references: np.ndarray,
    *,
    current_size: int,
    padding_factor: int = 1,
    interpolator: int = 1,
    projector_data_dtype=None,
    compute_dtype=np.float64,
    gridding_kernel: str = "radial",
    shell_pair_counting: str = "relion",
) -> tuple[np.ndarray, np.ndarray, int]:
    """Convert references to native-layout half maps and their corrected spectrum.

    The device build of :func:`setup_relion_projector_on_host`, which uses
    ``compute_dtype`` for gridding correction, FFT and power. It reproduces
    ``Projector::computeFourierTransformMap`` for even boxes, padding 1 or 2 and
    trilinear interpolation, and refuses other geometry; RELION's own transform is
    the oracle in :mod:`scripts.lib.native_projector_setup`.

    ``projector_data_dtype`` is what the caller wants the slab in; ``None`` keeps
    complex64, whose consumer is the InitialModel engine. Refinement asks for
    complex128 explicitly. ``gridding_kernel`` and ``shell_pair_counting`` are the
    correction window and the spectrum's pair counting of :func:`setup_relion_projector_on_host`.
    """
    compute_dtype = np.dtype(compute_dtype)
    if compute_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("Projector computation dtype must be float32 or float64")
    # VDAM's references are device arrays; reading them back here is not needed.
    refs = references if isinstance(references, jax.Array) else np.asarray(references)
    if refs.ndim != 4:
        raise ValueError(f"references must have shape (K, N, N, N), got {refs.shape}")
    n = int(refs.shape[-1])
    if not (n > 0 and n % 2 == 0 and refs.shape[1:] == (n, n, n)):
        raise ValueError(f"projector setup needs cubic references of an even box, got {refs.shape[1:]}")
    if int(padding_factor) not in {1, 2}:
        raise ValueError(f"projector setup supports padding factor 1 or 2, got {padding_factor}")
    if int(interpolator) != 1:
        raise ValueError("projector setup supports trilinear interpolation only (interpolator 1)")

    halves = []
    power_spectra = []
    # Projector::initialiseData uses a negative size for full resolution;
    # zero means radius zero here (state wrappers retain their defaults).
    r_max = n // 2 if int(current_size) < 0 else min(int(current_size) // 2, n // 2)
    for ref in refs:
        projector_data, power = setup_relion_projector_on_host(
            swap_relion_volume_layout(ref), r_max, box_size=n,
            padding_factor=int(padding_factor), compute_dtype=compute_dtype.type,
            gridding_kernel=gridding_kernel, shell_pair_counting=shell_pair_counting,
        )
        dtype = np.complex64 if projector_data_dtype is None else np.dtype(projector_data_dtype)
        halves.append(np.asarray(projector_data).astype(dtype, copy=False))
        power_spectra.append(np.asarray(power, dtype=np.float64))
    return np.asarray(halves), np.asarray(power_spectra, dtype=np.float64), int(r_max)


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


def prepare_local_class_projector_slabs(projector_half, n_classes: int, *, path_label="local RELION projector path"):
    """One (z, y, x_half) slab per class of a ``[K, z, y, x_half]`` projector (Class3D local search).

    Shape normalization only, as :func:`prepare_local_projector_slab`; a host array is sliced before its
    upload, so each class's slab reaches the device on its own.
    """
    n_classes = int(n_classes)
    if np.ndim(projector_half) != 4 or int(np.shape(projector_half)[0]) != n_classes:
        raise ValueError(
            f"{path_label} expected {n_classes} class slabs (K, z, y, x_half), got {np.shape(projector_half)}",
        )
    return [prepare_local_projector_slab(projector_half[k], path_label=path_label) for k in range(n_classes)]
