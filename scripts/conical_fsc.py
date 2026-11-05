"""Conical FSC (cFSC) of two maps and a conical FSC area ratio (cFAR) over the cones.

A cFSC is the FSC of two maps restricted to a double cone in Fourier space: the voxels within ``half_angle_deg`` of
an axis or of its opposite. Conical FSC was introduced for electron tomograms (Diebolder et al. 2015, J Struct Biol
190:215-223), used to diagnose a preferred orientation as the 3DFSC (Tan et al. 2017, Nat Methods 14:793-796) and as
the directional FSC of the Fourier shell occupancy, whose cone half-angle sets its angular resolution (Vilas and
Tagare 2023, Nat Methods 20:1021-1024). The threshold 0.143 is the gold-standard criterion of Rosenthal and Henderson
2003, J Mol Biol 333:721-745. Shells are those of ``scripts.fsc_metrics.shell_fsc`` (``rint(|k|)``, Nyquist edges
excluded), so a conical curve is directly comparable to the global one. The axes are spread uniformly over the
hemisphere (``cone_axes``); an axis is a direction in Fourier space, not a viewing direction, and the maps' array
axes are ``[x, y, z]`` of RELION's frame (``recovar.utils.helpers.load_relion_volume``), so an axis' ``tilt`` is its
angle from RELION's z and ``rot`` its azimuth.

The cFAR is defined here, not taken from a publication: ``wAuC`` of a cone is the sum of its cFSC over the shells
below the first 0.143 crossing, each shell weighted by its surface ``4 pi r^2``, so a cone's wAuC grows with how
much of Fourier space it correlates; ``cFAR`` is the smallest wAuC over the largest, 1 when every cone correlates
equally and 0 when some cone has no correlated shell. A cone holds few voxels in the lowest shells (about 2 in shell 1 at 20 degrees), where an FSC is noise: a shell with
fewer than ``MIN_CONE_VOXELS`` voxels in a cone is unmeasured, so it neither crosses the threshold nor adds area.

Use::

    from scripts.conical_fsc import conical_fsc, conical_fsc_summary, cone_axes

    cfsc = conical_fsc(half1, half2, cone_axes(100), half_angle_deg=20.0)
    summary = conical_fsc_summary(cfsc, voxel=1.06, box=256)  # summary["cfar"]
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import numpy as np

from scripts.fsc_metrics import first_shell_below

HALF_ANGLE_DEG = 20.0  # a default: the cones of neighbouring axes (about 14 degrees apart for N_AXES) overlap
N_AXES = 100
FSC_THRESHOLD = 0.143
# An FSC of m voxels has a standard error of about 1 / sqrt(m / 2) (the voxels of k and -k are one); it falls below
# the 0.143 threshold from m = 98 voxels.
MIN_CONE_VOXELS = 100


class ConicalFsc(NamedTuple):
    """The cFSC curves of one pair of maps.

    ``curves[a, s]`` is the FSC of shell ``s`` inside the cone of ``axes[a]``, NaN where the cone holds no voxel of
    the shell (shell 0, and the lowest shells of a narrow cone); ``voxels[a, s]`` counts the voxels it holds.
    """

    axes: np.ndarray  # (n_axes, 3) unit vectors in array-axis order [x, y, z], z > 0
    half_angle_deg: float
    curves: np.ndarray  # (n_axes, n_shells)
    voxels: np.ndarray  # (n_axes, n_shells), int


def cone_axes(n_axes: int = N_AXES) -> np.ndarray:
    """``n_axes`` unit vectors spread uniformly over the upper hemisphere (z > 0), a Fibonacci lattice.

    Fourier transforms of real maps are Hermitian, so the cone of an axis and of its opposite are one: the
    hemisphere holds every distinct cone. Returns an array of shape ``(n_axes, 3)``, columns x, y, z.
    """
    if n_axes < 1:
        raise ValueError(f"n_axes must be positive, got {n_axes}")
    index = np.arange(n_axes)
    z = (index + 0.5) / n_axes  # cos(tilt): uniform in z is uniform in area
    azimuth = index * math.pi * (3.0 - math.sqrt(5.0))
    radius = np.sqrt(1.0 - z * z)
    return np.stack([radius * np.cos(azimuth), radius * np.sin(azimuth), z], axis=1)


def axes_rot_tilt_deg(axes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(rot, tilt)`` in degrees of unit vectors ``[x, y, z]``: rot = atan2(y, x), tilt = angle from +z."""
    axes = np.asarray(axes, dtype=np.float64)
    rot = np.degrees(np.arctan2(axes[:, 1], axes[:, 0]))
    tilt = np.degrees(np.arccos(np.clip(axes[:, 2], -1.0, 1.0)))
    return rot, tilt


def conical_fsc(
    map_a: np.ndarray, map_b: np.ndarray, axes: np.ndarray, half_angle_deg: float = HALF_ANGLE_DEG
) -> ConicalFsc:
    """The cFSC of two cubic real maps inside the double cone of each of ``axes``.

    A Fourier voxel ``k`` of shell ``1 .. n // 2 - 2`` is in the cone of unit axis ``v`` when
    ``|k . v| >= |k| cos(half_angle)``. Each cone's curve is ``sum Re(A conj(B)) / sqrt(sum |A|^2 sum |B|^2)``
    over its voxels of the shell. Returns the curves and the voxel counts.
    """
    a = np.asarray(map_a, dtype=np.float64)
    b = np.asarray(map_b, dtype=np.float64)
    axes = np.asarray(axes, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 3 or len(set(a.shape)) != 1:
        raise ValueError(f"cFSC needs two cubic maps of one shape, got {a.shape} and {b.shape}")
    if axes.ndim != 2 or axes.shape[1] != 3 or not np.allclose(np.linalg.norm(axes, axis=1), 1.0):
        raise ValueError("axes must be an (n, 3) array of unit vectors")
    if not 0.0 < half_angle_deg < 90.0:
        raise ValueError(f"half_angle_deg must be in (0, 90), got {half_angle_deg}")

    n = a.shape[0]
    n_shells = n // 2 - 1
    freqs = np.fft.fftfreq(n) * n
    k2 = freqs[:, None, None] ** 2 + freqs[None, :, None] ** 2 + freqs[None, None, :] ** 2
    shell_of = np.rint(np.sqrt(k2)).astype(np.int64)
    used = (shell_of >= 1) & (shell_of < n_shells)
    fa = np.fft.fftn(a)[used]
    fb = np.fft.fftn(b)[used]
    ix, iy, iz = np.nonzero(used)
    k = np.stack([freqs[ix], freqs[iy], freqs[iz]], axis=1)
    shell = shell_of[used]
    min_projection = np.sqrt(k2[used]) * math.cos(math.radians(half_angle_deg))
    cross = (fa * np.conj(fb)).real
    power_a = np.abs(fa) ** 2
    power_b = np.abs(fb) ** 2

    n_axes = len(axes)
    curves = np.full((n_axes, n_shells), np.nan)
    voxels = np.zeros((n_axes, n_shells), dtype=np.int64)
    for j, axis in enumerate(axes):
        inside = np.abs(k @ axis) >= min_projection
        s = shell[inside]
        numerator = np.bincount(s, weights=cross[inside], minlength=n_shells)
        denominator = np.sqrt(
            np.bincount(s, weights=power_a[inside], minlength=n_shells)
            * np.bincount(s, weights=power_b[inside], minlength=n_shells)
        )
        np.divide(numerator, denominator, out=curves[j], where=denominator > 0.0)
        voxels[j] = np.bincount(s, minlength=n_shells)
    return ConicalFsc(axes=axes, half_angle_deg=float(half_angle_deg), curves=curves, voxels=voxels)


def measured_curves(cfsc: ConicalFsc, min_voxels: int = MIN_CONE_VOXELS) -> np.ndarray:
    """The curves of ``cfsc`` with NaN in every shell where the cone holds fewer than ``min_voxels`` voxels."""
    return np.where(cfsc.voxels >= min_voxels, cfsc.curves, np.nan)


def weighted_auc(curves: np.ndarray, threshold: float = FSC_THRESHOLD) -> tuple[np.ndarray, np.ndarray]:
    """wAuC and the first-crossing shell of each curve of ``curves`` (shape ``(n_curves, n_shells)``).

    wAuC is ``sum 4 pi s^2 C_s`` over the shells ``s = 1 ..`` before the first one below ``threshold``, or over
    every shell when the curve never crosses; a shell with no value (NaN) adds nothing and cannot cross. The
    crossing shell is ``n_shells`` for a curve that never crosses, so a caller can tell the two cases apart.
    """
    curves = np.asarray(curves, dtype=np.float64)
    n_shells = curves.shape[1]
    weight = 4.0 * math.pi * np.arange(n_shells, dtype=np.float64) ** 2
    wauc = np.zeros(len(curves))
    crossing = np.zeros(len(curves), dtype=np.int64)
    for j, curve in enumerate(curves):
        first = first_shell_below(curve, threshold)
        crossing[j] = n_shells if first is None else first
        upto = np.nan_to_num(curve[1 : crossing[j]], nan=0.0)
        wauc[j] = float(np.sum(weight[1 : crossing[j]] * upto))
    return wauc, crossing


def cone_areas(
    cfsc: ConicalFsc, threshold: float = FSC_THRESHOLD, min_voxels: int = MIN_CONE_VOXELS
) -> tuple[np.ndarray, np.ndarray]:
    """The wAuC and the first-crossing shell of every cone of ``cfsc``, over its measured shells."""
    return weighted_auc(measured_curves(cfsc, min_voxels), threshold)


def conical_fsc_summary(
    cfsc: ConicalFsc,
    voxel: float,
    box: int,
    threshold: float = FSC_THRESHOLD,
    min_voxels: int = MIN_CONE_VOXELS,
) -> dict[str, Any]:
    """The scalars of a cFSC: cFAR, the wAuC extremes with their axes, and the crossing resolutions (A).

    Shells where a cone holds fewer than ``min_voxels`` voxels are unmeasured (``measured_curves``).
    ``first_shell_measured_by_all_cones`` is the lowest shell from which every cone is measured in every shell.
    A cone whose curve never crosses ``threshold`` correlates up to the last measured shell: it is at least as good
    as ``last_shell_resolution_A`` and has no crossing resolution; ``n_axes_without_crossing`` counts such cones.
    ``worst_resolution_A`` is the largest crossing resolution (None when no cone crosses, every cone being at least
    as good as the last shell) and ``best_resolution_A`` the smallest, defined only when every cone crosses (a cone
    that does not cross is better than any that does). Raises when no cone has any area (cFAR undefined).
    """
    wauc, crossing = cone_areas(cfsc, threshold, min_voxels)
    if not np.max(wauc) > 0.0:
        raise ValueError("no cone has a positive wAuC: the maps share no signal above the threshold")
    rot, tilt = axes_rot_tilt_deg(cfsc.axes)
    lowest, highest = int(np.argmin(wauc)), int(np.argmax(wauc))
    n_shells = cfsc.curves.shape[1]
    crossed = crossing < n_shells
    resolution = box * voxel / crossing[crossed] if crossed.any() else np.array([])
    unmeasured = np.flatnonzero((cfsc.voxels[:, 1:] < min_voxels).any(axis=0))
    return {
        "n_axes": len(cfsc.axes),
        "half_angle_deg": cfsc.half_angle_deg,
        "threshold": threshold,
        "min_cone_voxels": min_voxels,
        "cfar": float(wauc[lowest] / wauc[highest]),
        "wauc_min": float(wauc[lowest]),
        "wauc_max": float(wauc[highest]),
        "min_axis": {"rot_deg": float(rot[lowest]), "tilt_deg": float(tilt[lowest])},
        "max_axis": {"rot_deg": float(rot[highest]), "tilt_deg": float(tilt[highest])},
        "worst_resolution_A": float(resolution.max()) if resolution.size else None,
        "best_resolution_A": float(resolution.min()) if crossed.all() else None,
        "last_shell_resolution_A": box * voxel / (n_shells - 1),
        "n_axes_without_crossing": int(np.sum(~crossed)),
        "first_shell_measured_by_all_cones": int(unmeasured.max()) + 2 if unmeasured.size else 1,
    }
