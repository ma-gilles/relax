"""RELION's HEALPix orientation sampling in NumPy.

Ports the parts of RELION 5.0.1 ``HealpixSampling`` (healpix_sampling.cpp) and
its bundled ``Healpix_Base`` (Healpix_2.15a/healpix_base.cc) that refinement
uses for 3-D sampling: the NEST pixel geometry, ``setOrientations`` with the
asymmetric-unit pruning of ``removeSymmetryEquivalentPoints`` for the proper
point groups, and ``getOrientations`` with its random perturbation.

The arithmetic follows RELION's operation order (``DEG2RAD(d) = d * PI / 180``,
``ACOSD``, ``Matrix2D`` products summed from zero over ``k``), and ``acos`` and
``atan2`` come from the C library (:mod:`relax.helpers.libm`), so the angles equal
RELION's. ``relax.relion_bind`` is the unit-test oracle
(``tests/unit/test_healpix_sampling_vs_relion_bind.py``); nothing here calls it.
"""

from __future__ import annotations

import functools
import math

import numpy as np

from relax.helpers import libm
from relax.relion.macros import PI, relion_ceil, relion_round
from relax.symmetry import canonicalize_rotational_symmetry, parse_rotational_symmetry, relion_symmetry_operators

# Healpix lsconstants.h halfpi, as a double.
_HALFPI = 1.570796326794896619231321691639751442099
_FLT_EPSILON = float(np.finfo(np.float32).eps)
# Healpix_Base Tablefiller (healpix_base.cc:45-56).
_M = np.arange(0x100, dtype=np.int64)
_CTAB = (
    (_M & 0x1)
    | ((_M & 0x2) << 7)
    | ((_M & 0x4) >> 1)
    | ((_M & 0x8) << 6)
    | ((_M & 0x10) >> 2)
    | ((_M & 0x20) << 5)
    | ((_M & 0x40) >> 3)
    | ((_M & 0x80) << 4)
)
_UTAB = (
    (_M & 0x1)
    | ((_M & 0x2) << 1)
    | ((_M & 0x4) << 2)
    | ((_M & 0x8) << 3)
    | ((_M & 0x10) << 4)
    | ((_M & 0x20) << 5)
    | ((_M & 0x40) << 6)
    | ((_M & 0x80) << 7)
)
_JRLL = np.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], dtype=np.int64)
_JPLL = np.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7], dtype=np.int64)


def _deg2rad(degrees):
    return degrees * PI / 180


def _rad2deg(radians):
    return radians * 180 / PI


def nside(order: int) -> int:
    return 1 << int(order)


def n_pixels(order: int) -> int:
    return 12 * nside(order) * nside(order)


def nest_to_xyf(order: int, pix):
    """``Healpix_Base::nest2xyf`` for NEST pixel indices."""

    pix = np.asarray(pix, dtype=np.int64)
    face = pix >> (2 * order)
    pix = pix & ((nside(order) << order) - 1)
    raw = (pix & 0x5555) | ((pix & 0x55550000) >> 15)
    ix = _CTAB[raw & 0xFF] | (_CTAB[raw >> 8] << 4)
    pix = pix >> 1
    raw = (pix & 0x5555) | ((pix & 0x55550000) >> 15)
    iy = _CTAB[raw & 0xFF] | (_CTAB[raw >> 8] << 4)
    return ix, iy, face


def xyf_to_nest(order: int, ix, iy, face):
    """``Healpix_Base::xyf2nest``."""

    ix = np.asarray(ix, dtype=np.int64)
    iy = np.asarray(iy, dtype=np.int64)
    face = np.asarray(face, dtype=np.int64)
    return (face << (2 * int(order))) + (
        _UTAB[ix & 0xFF] | (_UTAB[ix >> 8] << 16) | (_UTAB[iy & 0xFF] << 1) | (_UTAB[iy >> 8] << 17)
    )


def pix_to_z_phi(order: int, pix):
    """``Healpix_Base::pix2ang_z_phi`` in the NEST scheme (healpix_base.cc:494-527)."""

    ns = nside(order)
    npix = n_pixels(order)
    fact2 = 4.0 / npix
    fact1 = (ns << 1) * fact2
    nl4 = 4 * ns
    ix, iy, face = nest_to_xyf(order, pix)
    jr = (_JRLL[face] << order) - ix - iy - 1
    north = jr < ns
    south = jr > 3 * ns
    equator = ~(north | south)
    nr = np.where(north, jr, np.where(south, nl4 - jr, ns))
    nr_sq = (nr * nr).astype(np.float64)
    z = np.where(
        north,
        1 - nr_sq * fact2,
        np.where(south, nr_sq * fact2 - 1, (2 * ns - jr).astype(np.float64) * fact1),
    )
    kshift = np.where(equator, (jr - ns) & 1, 0)
    numerator = _JPLL[face] * nr + ix - iy + 1 + kshift
    # C integer division truncates toward zero.
    jp = np.sign(numerator) * (np.abs(numerator) // 2)
    jp = np.where(jp > nl4, jp - nl4, jp)
    jp = np.where(jp < 1, jp + nl4, jp)
    phi = (jp - (kshift + 1) * 0.5) * (_HALFPI / nr)
    return z, phi


def max_pixrad(order: int) -> float:
    """``Healpix_Base::max_pixrad`` (healpix_base.cc:781-789)."""

    ns = nside(order)

    def set_z_phi(z, phi):
        sintheta = math.sqrt((1.0 - z) * (1.0 + z))
        return (sintheta * math.cos(phi), sintheta * math.sin(phi), z)

    va = set_z_phi(2.0 / 3.0, math.pi / (4 * ns))
    t1 = 1.0 - 1.0 / ns
    t1 *= t1
    vb = set_z_phi(1 - t1 / 3, 0)
    cross = (
        va[1] * vb[2] - va[2] * vb[1],
        va[2] * vb[0] - va[0] * vb[2],
        va[0] * vb[1] - va[1] * vb[0],
    )
    length = math.sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2])
    return math.atan2(length, va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2])


def pixel_rot_tilt(order: int, pix):
    """``getDirectionFromHealPix``: RELION ``(rot, tilt)`` degrees, rot in [-180, 180]."""

    z, phi = pix_to_z_phi(order, pix)
    rot = _rad2deg(phi)
    tilt = libm.acos(z) * 180.0 / PI
    # checkDirection: acos keeps tilt >= 0 and phi >= 0 keeps rot >= 0.
    rot = np.where(rot > 180.0, rot - 360.0, rot)
    return rot, tilt


def default_psi_step(order: int) -> float:
    return 360.0 / (6 * relion_round(2.0 ** int(order)))


def psi_angles(order: int, psi_step: float = -1.0) -> np.ndarray:
    """``setOrientations`` in-plane angles: ``ipsi * 360 / CEIL(360 / psi_step)``."""

    step = default_psi_step(order) if psi_step <= 0.0 else float(psi_step)
    nr_psi = relion_ceil(360.0 / step)
    step = 360.0 / float(nr_psi)
    return np.arange(nr_psi, dtype=np.float64) * step


def angular_sampling_deg(order: int) -> float:
    """``getAngularSampling()`` for 3-D sampling at ``adaptive_oversampling = 0``."""

    return 360.0 / (6 * relion_round(2.0 ** int(order)))


def euler_angles_to_matrix(eulers_deg) -> np.ndarray:
    """``Euler_angles2matrix`` (euler.cpp) for ``[N, 3]`` ``(rot, tilt, psi)`` rows."""

    eulers = np.asarray(eulers_deg, dtype=np.float64).reshape(-1, 3)
    alpha = _deg2rad(eulers[:, 0])
    beta = _deg2rad(eulers[:, 1])
    gamma = _deg2rad(eulers[:, 2])
    ca, cb, cg = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sa, sb, sg = np.sin(alpha), np.sin(beta), np.sin(gamma)
    cc = cb * ca
    cs = cb * sa
    sc = sb * ca
    ss = sb * sa
    matrix = np.empty((eulers.shape[0], 3, 3), dtype=np.float64)
    matrix[:, 0, 0] = cg * cc - sg * sa
    matrix[:, 0, 1] = cg * cs + sg * ca
    matrix[:, 0, 2] = -cg * sb
    matrix[:, 1, 0] = -sg * cc - cg * sa
    matrix[:, 1, 1] = -sg * cs + cg * ca
    matrix[:, 1, 2] = sg * sb
    matrix[:, 2, 0] = sc
    matrix[:, 2, 1] = ss
    matrix[:, 2, 2] = cb
    return matrix


def euler_matrix_to_angles(matrices) -> np.ndarray:
    """``Euler_matrix2angles`` (euler.cpp:148-188) for ``[N, 3, 3]`` matrices."""

    a = np.asarray(matrices, dtype=np.float64).reshape(-1, 3, 3)
    abs_sb = np.sqrt(a[:, 0, 2] * a[:, 0, 2] + a[:, 1, 2] * a[:, 1, 2])
    regular = abs_sb > 16 * _FLT_EPSILON
    gamma = libm.atan2(a[:, 1, 2], -a[:, 0, 2])
    alpha = libm.atan2(a[:, 2, 1], a[:, 2, 0])
    sin_gamma = np.sin(gamma)
    with np.errstate(divide="ignore", invalid="ignore"):
        sign_small = np.where(-a[:, 0, 2] / np.cos(gamma) >= 0, 1.0, -1.0)
    sgn_12 = np.where(a[:, 1, 2] >= 0, 1.0, -1.0)
    sign_sb = np.where(np.abs(sin_gamma) < _FLT_EPSILON, sign_small, np.where(sin_gamma > 0, sgn_12, -sgn_12))
    beta = libm.atan2(sign_sb * abs_sb, a[:, 2, 2])
    upright = a[:, 2, 2] >= 0
    # The singular branch's atan2 only for the rows that take it (the C-library atan2 is slow).
    singular = np.flatnonzero(~regular)
    singular_gamma = np.zeros_like(gamma)
    singular_gamma[singular] = np.where(
        upright[singular],
        libm.atan2(-a[singular, 1, 0], a[singular, 0, 0]),
        libm.atan2(a[singular, 1, 0], -a[singular, 0, 0]),
    )
    alpha = np.where(regular, alpha, 0.0)
    beta = np.where(regular, beta, np.where(upright, 0.0, PI))
    gamma = np.where(regular, gamma, singular_gamma)
    return np.stack([_rad2deg(alpha), _rad2deg(beta), _rad2deg(gamma)], axis=-1)


def matmul3(left, right):
    """``Matrix2D::operator*`` for ``[..., 3, 3]``: each entry summed from zero over k."""

    return (
        left[..., :, 0, None] * right[..., None, 0, :]
        + left[..., :, 1, None] * right[..., None, 1, :]
        + left[..., :, 2, None] * right[..., None, 2, :]
    )


def perturb_orientations(eulers_deg, random_perturbation: float, order: int) -> np.ndarray:
    """``getOrientations`` random perturbation: ``A * Euler(p, p, p)`` back to angles."""

    eulers = np.asarray(eulers_deg, dtype=np.float64).reshape(-1, 3)
    if not abs(float(random_perturbation)) > 0.0:
        return eulers
    perturbation = float(random_perturbation) * angular_sampling_deg(order)
    rotation = euler_angles_to_matrix([[perturbation, perturbation, perturbation]])[0]
    return euler_matrix_to_angles(matmul3(euler_angles_to_matrix(eulers), rotation))


def _direction(rot_deg, tilt_deg):
    """``Euler_angles2direction`` as ``[N, 3]``."""

    alpha = _deg2rad(np.asarray(rot_deg, dtype=np.float64))
    beta = _deg2rad(np.asarray(tilt_deg, dtype=np.float64))
    sb = np.sin(beta)
    return np.stack([sb * np.cos(alpha), sb * np.sin(alpha), np.cos(beta)], axis=-1)


def _dot(vectors, axis):
    """``dotProduct`` of ``[N, 3]`` rows with one vector, summed from zero in order."""

    return vectors[:, 0] * axis[0] + vectors[:, 1] * axis[1] + vectors[:, 2] * axis[2]


def _normalized(vector):
    """``Matrix1D::selfNormalize``: multiply by ``1 / module``."""

    x, y, z = (float(value) for value in vector)
    module = math.sqrt(0.0 + x * x + y * y + z * z)
    if not abs(module) > 1e-6:
        return (0.0, 0.0, 0.0)
    inverse = 1.0 / module
    return (x * inverse, y * inverse, z * inverse)


def _rotated(matrix, vector):
    """``Matrix2D * Matrix1D`` summed from zero in order."""

    return tuple(0.0 + matrix[i][0] * vector[0] + matrix[i][1] * vector[1] + matrix[i][2] * vector[2] for i in range(3))


_ICOSAHEDRAL_FACE_EDGES = (
    (0.0, 1.0, 0.0),
    (-0.4999999839058737, -0.8090170074556163, 0.3090169861701543),
    (0.4999999839058737, -0.8090170074556163, 0.3090169861701543),
)


def _geometric_keep(parsed, rot, tilt, directions) -> np.ndarray:
    """``removeSymmetryEquivalentPointsGeometric`` for the proper groups."""

    def inside(axes, *, below=False):
        keep = np.ones(rot.shape, dtype=bool)
        for axis in axes:
            dots = _dot(directions, axis)
            keep &= dots <= 0 if below else dots >= 0
        return keep

    if parsed.family == "cyclic":
        return (rot >= (-180.0 / parsed.order)) & (rot <= (180.0 / parsed.order))
    if parsed.family == "dihedral":
        if parsed.order == 1:
            return tilt <= 90.0
        return (rot >= -180.0 / parsed.order + 90.0) & (rot <= 180.0 / parsed.order + 90.0) & (tilt <= 90.0)
    if parsed.family == "tetrahedral":
        axes = [
            _normalized(v) for v in ((-0.942809, 0.0, 0.0), (0.471405, 0.272165, 0.7698), (0.471404, 0.816497, 0.0))
        ]
        return (((rot >= 90.0) & (rot <= 150.0)) | (rot == 0)) & inside(axes)
    if parsed.family == "octahedral":
        axes = [_normalized(v) for v in ((0.0, -1.0, 1.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0))]
        return (((rot >= 45.0) & (rot <= 135.0) & (tilt <= 90.0)) | (rot == 0.0)) & inside(axes)
    if parsed.label == "I2":
        return inside([_normalized(v) for v in _ICOSAHEDRAL_FACE_EDGES])
    if parsed.label in {"I1", "I3"}:
        tilt_deg = 90.0 if parsed.label == "I1" else 31.7174745559
        matrix = euler_angles_to_matrix([[0.0, tilt_deg, 0.0]])[0]
        return inside([_normalized(_rotated(matrix, v)) for v in _ICOSAHEDRAL_FACE_EDGES])
    if parsed.label == "I4":
        matrix = euler_angles_to_matrix([[0.0, -31.7174745559, 0.0]])[0]
        edges = (
            (0.0, 0.0, 1.0),
            (0.187592467856686, -0.303530987314591, -0.491123477863004),
            (0.187592467856686, 0.303530987314591, -0.491123477863004),
        )
        return inside([_normalized(_rotated(matrix, v)) for v in edges], below=True)
    raise ValueError(f"no RELION asymmetric-unit rule for {parsed.label}")


def _remove_seam_duplicates(directions, left, right, cos_max_ang) -> np.ndarray:
    """The seam pass of ``removeSymmetryEquivalentPoints`` (healpix_sampling.cpp:2230-2270).

    A direction is dropped when any symmetry image ``L (v^T R)^T`` of an already
    kept direction lies within ``max_ang`` of it; the result does not depend on
    RELION's loop order because it only asks whether such an image exists.
    """

    # (v^T R)_j and then L w, each summed from zero over k in order.
    kept_images = []
    keep = np.zeros(directions.shape[0], dtype=bool)
    for i, direction in enumerate(directions):
        if kept_images:
            images = np.concatenate(kept_images, axis=0)
            if np.any(_dot(images, direction) > cos_max_ang):
                continue
        keep[i] = True
        w = direction[0] * right[:, 0, :] + direction[1] * right[:, 1, :] + direction[2] * right[:, 2, :]
        kept_images.append(
            left[:, :, 0] * w[:, 0, None] + left[:, :, 1] * w[:, 1, None] + left[:, :, 2] * w[:, 2, None]
        )
    return keep


@functools.lru_cache(maxsize=None)
def _sampling(order: int, symmetry: str):
    parsed = parse_rotational_symmetry(symmetry)
    pixels = np.arange(n_pixels(order), dtype=np.int64)
    rot, tilt = pixel_rot_tilt(order, pixels)
    if parsed.label != "C1":
        # setOrientations -> removeSymmetryEquivalentPoints(0.5 * RAD2DEG(max_pixrad)).
        directions = _direction(rot, tilt)
        keep = _geometric_keep(parsed, rot, tilt, directions)
        pixels, rot, tilt, directions = pixels[keep], rot[keep], tilt[keep], directions[keep]
        if rot.size < 4000:
            left, right = relion_symmetry_operators(parsed.label, dtype=np.float64)
            cos_max_ang = math.cos(_deg2rad(0.5 * _rad2deg(max_pixrad(order))))
            keep = _remove_seam_duplicates(directions, left, right, cos_max_ang)
            pixels, rot, tilt = pixels[keep], rot[keep], tilt[keep]
    arrays = {
        "directions_ipix": pixels,
        "rot": rot,
        "tilt": tilt,
        "psi": psi_angles(order),
    }
    for value in arrays.values():
        value.setflags(write=False)
    return arrays


def healpix_sampling(order: int, symmetry: str = "C1") -> dict[str, np.ndarray]:
    """``HealpixSampling::setOrientations`` for 3-D, no tilt limit, default psi step.

    Returns read-only ``directions_ipix`` (the NEST pixel of each retained
    direction), ``rot``, ``tilt`` and ``psi`` in degrees.
    """

    return _sampling(int(order), canonicalize_rotational_symmetry(symmetry))


def coarse_orientations(order: int, symmetry: str = "C1") -> np.ndarray:
    """RELION's coarse ``(rot, tilt, psi)`` grid, direction-major, psi fastest."""

    sampling = healpix_sampling(order, symmetry)
    n_dir = sampling["rot"].size
    n_psi = sampling["psi"].size
    return np.stack(
        [
            np.repeat(sampling["rot"], n_psi),
            np.repeat(sampling["tilt"], n_psi),
            np.tile(sampling["psi"], n_dir),
        ],
        axis=-1,
    )


def oversampled_orientations(
    order: int,
    oversampling_order: int,
    idirs,
    ipsis,
    random_perturbation: float = 0.0,
    symmetry: str = "C1",
) -> np.ndarray:
    """``HealpixSampling::getOrientations`` rows for each ``(idir, ipsi)`` in input order.

    ``idirs`` index the retained directions of :func:`healpix_sampling`. Each
    parent yields ``4**oversampling_order`` fine pixels (``j`` slow, ``i`` fast)
    times ``2**oversampling_order`` psi samples, then the perturbation.
    """

    sampling = healpix_sampling(order, symmetry)
    idirs = np.asarray(idirs, dtype=np.int64).reshape(-1)
    ipsis = np.asarray(ipsis, dtype=np.int64).reshape(-1)
    if idirs.shape != ipsis.shape:
        raise ValueError("idirs and ipsis must have the same length")
    if np.any((idirs < 0) | (idirs >= sampling["rot"].size)):
        raise ValueError("idir out of range")
    if np.any((ipsis < 0) | (ipsis >= sampling["psi"].size)):
        raise ValueError("ipsi out of range")
    psi_center = sampling["psi"][ipsis]
    if oversampling_order == 0:
        eulers = np.stack([sampling["rot"][idirs], sampling["tilt"][idirs], psi_center], axis=-1)
    else:
        fact = nside(order + oversampling_order) // nside(order)
        x, y, face = nest_to_xyf(order, sampling["directions_ipix"][idirs])
        dj, di = np.meshgrid(np.arange(fact), np.arange(fact), indexing="ij")
        fine_x = fact * x[:, None] + di.reshape(-1)[None, :]
        fine_y = fact * y[:, None] + dj.reshape(-1)[None, :]
        fine_pix = xyf_to_nest(order + oversampling_order, fine_x, fine_y, face[:, None])
        rot, tilt = pixel_rot_tilt(order + oversampling_order, fine_pix)
        nr_psi_over = relion_round(2.0**oversampling_order)
        psi_step = 360.0 / float(sampling["psi"].size)
        over = np.arange(nr_psi_over, dtype=np.float64)
        psi = psi_center[:, None] - 0.5 * psi_step + (0.5 + over)[None, :] * psi_step / nr_psi_over
        n_rows = idirs.size * fact * fact * nr_psi_over
        eulers = np.stack(
            [
                np.repeat(rot.reshape(-1), nr_psi_over),
                np.repeat(tilt.reshape(-1), nr_psi_over),
                np.broadcast_to(psi[:, None, :], (idirs.size, fact * fact, nr_psi_over)).reshape(-1),
            ],
            axis=-1,
        ).reshape(n_rows, 3)
    return perturb_orientations(eulers, random_perturbation, order)


__all__ = [
    "angular_sampling_deg",
    "coarse_orientations",
    "euler_angles_to_matrix",
    "euler_matrix_to_angles",
    "healpix_sampling",
    "matmul3",
    "max_pixrad",
    "nest_to_xyf",
    "oversampled_orientations",
    "perturb_orientations",
    "pix_to_z_phi",
    "pixel_rot_tilt",
    "psi_angles",
    "xyf_to_nest",
]
