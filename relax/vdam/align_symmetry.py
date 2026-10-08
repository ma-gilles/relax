"""RELION's ``relion_align_symmetry`` for the InitialModel output (RELION 5.0.1 apps/align_symmetry.cpp).

RELION's InitialModel GUI job refines in C1 and then runs (pipeline_jobs.cpp:3572-3587)::

    relion_align_symmetry --i run_itNNN_model.star --o initial_model.mrc --sym S --apply_sym --select_largest_class

with the program's defaults: re-centre on the centre of mass, Fourier-crop to a 64-pixel working box, score 400
random orientations and then a 5x5x5 local grid by the real-space squared difference between the rotated map and
its symmetrised copy, rotate the full map by the best orientation and symmetrise it. Every array here is a map in
RELION's MRC layout ``[z, y, x]`` (what relax writes to disk), in float64 as RELION's CPU build computes.

RELION draws the random orientations from a time seed; relax draws them from the run's seed, so the orientation
search is reproducible but the output matches RELION's only within the spread of RELION's own reruns.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from relax.healpix_sampling import euler_angles_to_matrix
from relax.relion.macros import relion_round
from relax.symmetry import is_identity_symmetry, relion_symmetry_operators

# align_symmetry.cpp defaults: --box_size 64, --nr_uniform 400, --local_search_range 2, --local_search_step 2,
# --pad 2, trilinear interpolation, --r_min_nn 10 (unused by trilinear rotation).
WORKING_BOX = 64
NR_UNIFORM = 400
LOCAL_SEARCH_RANGE = 2
LOCAL_SEARCH_STEP_DEG = 2.0
PADDING_FACTOR = 2
# XMIPP_EQUAL_ACCURACY of RELION's double-precision CPU build (macros.h:115).
_EQUAL_ACCURACY = 1e-6
_SEARCH_CHUNK = 25


def select_largest_class(pdf_class) -> int:
    """``--select_largest_class``: the first class of largest rlnClassDistribution (strict ``>``)."""

    return int(np.argmax(np.asarray(pdf_class, dtype=np.float64)))


def recentre_on_centre_of_mass(volume: np.ndarray) -> np.ndarray:
    """``selfTranslateCenterOfMassToCenter(V, DONT_WRAP)``: shift the positive-voxel centre of mass to the centre.

    ``MultidimArray::centerOfMass`` weighs only voxels above zero, in logical (centred) indices.
    """

    volume = np.asarray(volume, dtype=np.float64)
    coords = np.arange(volume.shape[0], dtype=np.float64) - volume.shape[0] // 2
    positive = np.where(volume > 0, volume, 0.0)
    mass = positive.sum()
    if mass == 0:
        return volume.copy()
    centre = (
        np.array(
            [
                (positive.sum(axis=(1, 2)) * coords).sum(),
                (positive.sum(axis=(0, 2)) * coords).sum(),
                (positive.sum(axis=(0, 1)) * coords).sum(),
            ]
        )
        / mass
    )  # (z, y, x)
    # translate(V, -centre) is applyGeometry with the inverse matrix: input = output + centre.
    return np.asarray(_apply_geometry(jnp.asarray(volume), jnp.eye(3), jnp.asarray(centre[::-1].copy())))


def resize_map(volume: np.ndarray, new_size: int) -> np.ndarray:
    """``resizeMap``: normalised forward FFT, ``windowFourierTransform`` to ``new_size``, unnormalised inverse."""

    n = volume.shape[0]
    if new_size == n:
        return np.asarray(volume, dtype=np.float64).copy()
    ft = np.fft.rfftn(np.asarray(volume, dtype=np.float64)) / n**3
    half = new_size // 2 + 1
    # FFTW index k of the new array is frequency k for k < half (k == new_size / 2 is +Nyquist), else k - new_size.
    freq = np.array([k if k < half else k - new_size for k in range(new_size)])
    rows = np.where(freq >= 0, freq, freq + n)
    window = ft[np.ix_(rows, rows, np.arange(half))]
    return np.fft.irfftn(window, s=(new_size,) * 3, axes=(0, 1, 2)) * new_size**3


def _projector_data(volume: np.ndarray) -> jnp.ndarray:
    """``Projector::computeFourierTransformMap`` with ``data_dim = 3`` at Nyquist, padding 2, trilinear gridding.

    Returns RELION's ``data`` array ``[pad_size, pad_size, pad_size // 2 + 1]`` with logical y/z origin at
    ``pad_size // 2`` (``setXmippOrigin``) and x from 0.
    """

    n = volume.shape[0]
    pad = PADDING_FACTOR
    padded_n = pad * n
    r = np.sqrt(sum(c**2 for c in np.meshgrid(*(np.arange(n) - n // 2,) * 3, indexing="ij")))
    rval = r / padded_n
    with np.errstate(invalid="ignore", divide="ignore"):
        sinc = np.where(r > 0, np.sin(np.pi * rval) / (np.pi * rval), 1.0)
    corrected = np.asarray(volume, dtype=np.float64) / (sinc * sinc)
    mpad = np.zeros((padded_n,) * 3)
    lo = padded_n // 2 - n // 2
    mpad[lo : lo + n, lo : lo + n, lo : lo + n] = corrected
    # Normalised forward transform of the centred padded map; CenterFFTbySign is the origin shift (ifftshift).
    faux = np.fft.rfftn(np.fft.ifftshift(mpad)) / padded_n**3
    r_max = n // 2
    max_r = relion_round(r_max * pad)
    pad_size = 2 * (max_r + 1) + 1
    origin = pad_size // 2
    data = np.zeros((pad_size, pad_size, pad_size // 2 + 1), dtype=np.complex128)
    # FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM: index k is frequency k up to padded_n / 2 (+Nyquist), else k - padded_n.
    freq = np.array([k if k <= padded_n // 2 else k - padded_n for k in range(padded_n)])
    kz, ky, kx = np.meshgrid(freq, freq, np.arange(padded_n // 2 + 1), indexing="ij")
    inside = kz * kz + ky * ky + kx * kx <= max_r * max_r
    data[kz[inside] + origin, ky[inside] + origin, kx[inside]] = faux[inside] * pad**3
    return jnp.asarray(data)


@partial(jax.jit, static_argnames=("n",))
def _rotate(data, matrix, *, n: int):
    """``Projector::get2DFourierTransform`` of a 3-D map (``rotate3D``), inverse FFT and ``CenterFFT``.

    ``matrix`` is ``Euler_rotation3DMatrix``; the output is the real ``[n, n, n]`` rotated map at the input scale.
    """

    pad = PADDING_FACTOR
    r_out = n // 2
    r_ref = (n // 2) * pad
    origin = data.shape[0] // 2
    ainv = jnp.linalg.inv(matrix) * pad
    freq = jnp.where(jnp.arange(n) <= r_out, jnp.arange(n), jnp.arange(n) - n)
    z, y, x = jnp.meshgrid(freq, freq, jnp.arange(n // 2 + 1), indexing="ij")
    out_mask = (y * y + z * z <= r_out * r_out) & (x * x + y * y + z * z <= r_out * r_out)
    xyz = jnp.stack([x, y, z], axis=-1).astype(jnp.float64)
    p = xyz @ ainv.T
    # RELION truncates the squared radius to int before the r_max_ref test.
    ref_mask = jnp.trunc(jnp.sum(p * p, axis=-1)) <= r_ref * r_ref
    neg = p[..., 0] < 0
    p = jnp.where(neg[..., None], -p, p)
    x0 = jnp.floor(p[..., 0]).astype(jnp.int32)
    y0 = jnp.floor(p[..., 1]).astype(jnp.int32)
    z0 = jnp.floor(p[..., 2]).astype(jnp.int32)
    fx, fy, fz = p[..., 0] - x0, p[..., 1] - y0, p[..., 2] - z0
    mask = out_mask & ref_mask
    y0 = jnp.where(mask, y0 + origin, 0)
    z0 = jnp.where(mask, z0 + origin, 0)
    x0 = jnp.where(mask, x0, 0)

    def at(dz, dy, dx):
        return data[z0 + dz, y0 + dy, x0 + dx]

    def lerp(a, lo, hi):
        return lo + (hi - lo) * a

    dx00 = lerp(fx, at(0, 0, 0), at(0, 0, 1))
    dx01 = lerp(fx, at(1, 0, 0), at(1, 0, 1))
    dx10 = lerp(fx, at(0, 1, 0), at(0, 1, 1))
    dx11 = lerp(fx, at(1, 1, 0), at(1, 1, 1))
    value = lerp(fz, lerp(fy, dx00, dx10), lerp(fy, dx01, dx11))
    value = jnp.where(neg, jnp.conj(value), value)
    f3d = jnp.where(mask, value, 0.0)
    rotated = jnp.fft.irfftn(f3d, s=(n, n, n)) * n**3
    return jnp.fft.fftshift(rotated)


@jax.jit
def _apply_geometry(volume, matrix, shift):
    """``applyGeometry(V1, V2, A, IS_INV, DONT_WRAP)`` for 3-D: trilinear, zero outside.

    ``matrix``/``shift`` are ``A``'s rotation and translation in (x, y, z): input = A @ output + shift, in logical
    (centred) voxel coordinates.
    """

    n = volume.shape[0]
    centre = n // 2
    lo, hi = -centre, n - centre - 1
    c = jnp.arange(n, dtype=jnp.float64) - centre
    z, y, x = jnp.meshgrid(c, c, c, indexing="ij")
    p = jnp.stack([x, y, z], axis=-1) @ matrix.T + shift
    inside = jnp.all((p >= lo - _EQUAL_ACCURACY) & (p <= hi + _EQUAL_ACCURACY), axis=-1)
    w = p + centre
    i0 = jnp.trunc(w).astype(jnp.int32)
    f = w - i0
    i0 = jnp.where(inside[..., None], i0, 0)
    m1, n1, o1 = i0[..., 0], i0[..., 1], i0[..., 2]
    wx, wy, wz = f[..., 0], f[..., 1], f[..., 2]

    def at(do, dn, dm):
        # RELION skips a neighbour past the last index; its weight is then (numerically) zero.
        o, nn, m = o1 + do, n1 + dn, m1 + dm
        valid = (o < n) & (nn < n) & (m < n)
        return jnp.where(valid, volume[jnp.minimum(o, n - 1), jnp.minimum(nn, n - 1), jnp.minimum(m, n - 1)], 0.0)

    value = (
        (1 - wz) * (1 - wy) * (1 - wx) * at(0, 0, 0)
        + (1 - wz) * (1 - wy) * wx * at(0, 0, 1)
        + (1 - wz) * wy * (1 - wx) * at(0, 1, 0)
        + (1 - wz) * wy * wx * at(0, 1, 1)
        + wz * (1 - wy) * (1 - wx) * at(1, 0, 0)
        + wz * (1 - wy) * wx * at(1, 0, 1)
        + wz * wy * (1 - wx) * at(1, 1, 0)
        + wz * wy * wx * at(1, 1, 1)
    )
    return jnp.where(inside, value, 0.0)


@jax.jit
def _symmetrise(volume, right_operators):
    """``symmetriseMap``: the mean of the map and ``applyGeometry(map, R, IS_INV, DONT_WRAP)`` over SymList's R."""

    zero = jnp.zeros(3)
    total = volume + jnp.sum(jax.vmap(lambda r: _apply_geometry(volume, r, zero))(right_operators), axis=0)
    return total / (right_operators.shape[0] + 1)


@partial(jax.jit, static_argnames=("n",))
def _diff2_batch(data, matrices, right_operators, *, n: int):
    def one(matrix):
        rotated = _rotate(data, matrix, n=n)
        return jnp.sum((rotated - _symmetrise(rotated, right_operators)) ** 2)

    return jax.lax.map(one, matrices)


def _search(data, eulers: np.ndarray, right_operators, n: int) -> tuple[int, float]:
    """``align_symmetry::search``: the first orientation of least squared difference."""

    matrices = euler_angles_to_matrix(eulers)
    diff2 = []
    for start in range(0, len(matrices), _SEARCH_CHUNK):
        diff2.append(
            np.asarray(_diff2_batch(data, jnp.asarray(matrices[start : start + _SEARCH_CHUNK]), right_operators, n=n))
        )
    diff2 = np.concatenate(diff2)
    best = int(np.argmin(diff2))
    return best, float(diff2[best])


def _uniform_orientations(rng: np.random.Generator) -> np.ndarray:
    """``--nr_uniform`` orientations as align_symmetry draws them: rot and psi uniform, tilt by sin(tilt) rejection."""

    eulers = np.empty((NR_UNIFORM, 3))
    for i in range(NR_UNIFORM):
        rot = rng.random() * 360.0
        while True:
            tilt = rng.random() * 180.0
            if rng.random() < abs(np.sin(np.deg2rad(tilt))):
                break
        eulers[i] = rot, tilt, rng.random() * 360.0
    return eulers


def _local_grid(rot: float, tilt: float, psi: float) -> np.ndarray:
    # align_symmetry.cpp steps psi by k * search_range, not k * search_step; kept for parity.
    steps = range(-LOCAL_SEARCH_RANGE, LOCAL_SEARCH_RANGE + 1)
    return np.array(
        [
            (rot + i * LOCAL_SEARCH_STEP_DEG, tilt + j * LOCAL_SEARCH_STEP_DEG, psi + k * LOCAL_SEARCH_RANGE)
            for i in steps
            for j in steps
            for k in steps
        ]
    )


def align_symmetry(volume: np.ndarray, sym_name: str, *, seed: int) -> tuple[np.ndarray, dict]:
    """``relion_align_symmetry --sym S --apply_sym`` on one map in RELION's MRC layout.

    Returns the aligned and symmetrised map and a report with the global and refined (rot, tilt, psi) and diff2.
    C1 returns the map unchanged, as RELION writes the selected map again.
    """

    volume = np.asarray(volume, dtype=np.float64)
    if is_identity_symmetry(sym_name):
        return volume.copy(), {"sym": "C1"}
    n = volume.shape[0]
    if n % 2:
        raise ValueError("The input box size must be an even number.")
    box = min(WORKING_BOX, n)
    _, right = relion_symmetry_operators(sym_name)
    right = jnp.asarray(right[1:])
    volume = recentre_on_centre_of_mass(volume)
    work_data = _projector_data(resize_map(volume, box))
    uniform = _uniform_orientations(np.random.default_rng(int(seed)))
    best, global_diff2 = _search(work_data, uniform, right, box)
    global_eulers = uniform[best]
    local = _local_grid(*global_eulers)
    best, diff2 = _search(work_data, local, right, box)
    rot, tilt, psi = local[best]
    # Rotate the original-size (re-centred) map and apply the symmetry.
    matrix = jnp.asarray(euler_angles_to_matrix([[rot, tilt, psi]])[0])
    aligned = np.asarray(_symmetrise(_rotate(_projector_data(volume), matrix, n=n), right))
    report = {
        "sym": sym_name,
        "seed": int(seed),
        "global_rot_tilt_psi": [float(v) for v in global_eulers],
        "refined_rot_tilt_psi": [float(rot), float(tilt), float(psi)],
        "global_diff2": global_diff2,
        "refined_diff2": diff2,
    }
    return aligned, report
