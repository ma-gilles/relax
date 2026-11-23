"""Conical-FSC regularization of a K=1 reconstruction: RELION's FSC-to-tau2 rule applied in cones of Fourier space.

RELION converts the half-set FSC of each shell into a signal power ``tau2`` (``BackProjector::updateSSNRarrays``,
here ``regularization.compute_relion_tau2_from_weights``) and adds ``1 / (p^3 T tau2)`` to every padded Fourier
weight of the shell before the Wiener division (``BackProjector::reconstruct``). Under a preferred orientation the
half-set correlation and the Fourier weight vary strongly between directions of one shell, and one value per shell
fits neither the well-sampled nor the poorly sampled directions. This module runs the same rule in each double cone
of a set of axes and applies it to every voxel as a factor on RELION's shell value:

    tau2(k) = tau2_RELION(ires(k)) * rho(round(k / p)),

where ``rho`` averages, over the cones that contain the voxel, the ratio of the cone's RELION estimate to the
shell's (equal statistics in every cone give ``rho = 1``, RELION's prior). The derivation, the choices and their
limits are in docs/math/conical_fsc_regularization.md.

The statistics are those of RELION's FSC: the two halves' unregularized averages on the native grid
(``BackProjector::getDownsampledAverage``), binned by ``ROUND(|nu|)`` up to ``r_max``. They are computed in the
accumulators' own axis frame on the full Fourier grid, so the cones and the voxels the prior is added to share one
frame whatever the accumulator layout. The prior is added to the weight before the solve
(:meth:`ConePrior.regularized_weight`), on the support of RELION's MAP term (``|k| < p r_max`` and
``ires >= minres_map``), so the solver's denominator floor and windowing stay RELION's.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from relax.relion.macros import relion_round_array

logger = logging.getLogger(__name__)

# A cone-shell bin with fewer native voxels keeps RELION's shell value: an FSC of m voxels has a standard error of
# about 1 / sqrt(m / 2) (k and -k are one), as for scripts/conical_fsc.py's MIN_CONE_VOXELS.
MIN_CONE_VOXELS = 100
# RELION's clamp of the FSC before SSNR = FSC / (1 - FSC) (BackProjector::updateSSNRarrays).
FSC_CLAMP = 0.001
# RELION's test for a zero tau2 in BackProjector::reconstruct (RECOVAR's adjust_regularization_relion_style),
# whose fallback this module does not reproduce.
ZERO_TAU2 = 1e-20
# Directions of the sphere that probe whether the cones cover it.
_COVERAGE_PROBES = 20_000
# Padded-grid planes per block of the prior term.
_PLANES_PER_BLOCK = 16


def cone_axes(n_axes: int) -> np.ndarray:
    """``n_axes`` unit vectors spread uniformly over the hemisphere ``z > 0`` (a Fibonacci lattice), shape (n_axes, 3).

    A cone of an axis and of its opposite are one double cone (Fourier transforms of real maps are Hermitian), so the
    hemisphere holds every distinct cone. The components are in the order of the grid's axes.
    """
    if n_axes < 1:
        raise ValueError(f"n_axes must be positive, got {n_axes}")
    index = np.arange(n_axes)
    z = (index + 0.5) / n_axes  # cos(tilt): uniform in z is uniform in area
    azimuth = index * math.pi * (3.0 - math.sqrt(5.0))
    radius = np.sqrt(1.0 - z * z)
    return np.stack([radius * np.cos(azimuth), radius * np.sin(azimuth), z], axis=1)


def largest_axis_distance_deg(axes: np.ndarray) -> float:
    """The largest angle (degrees) from a direction of the sphere to its nearest axis or opposite axis.

    A double cone of half-angle ``a`` contains a direction within ``a`` of its axis; the cones of ``axes`` cover the
    sphere when this angle is at most ``a``. Probed on a Fibonacci lattice of the whole sphere.
    """
    probes = cone_axes(_COVERAGE_PROBES)
    probes = np.concatenate([probes, -probes])
    nearest = np.max(np.abs(probes @ np.asarray(axes, dtype=np.float64).T), axis=1)
    return float(np.degrees(np.arccos(np.clip(nearest.min(), -1.0, 1.0))))


@dataclass(frozen=True)
class ConicalPriorSettings:
    """The cones of the conical-FSC regularization: ``n_axes`` axes over the hemisphere, each the axis of a double
    cone of half-angle ``half_angle_deg``. The cones must cover the sphere."""

    half_angle_deg: float = 20.0
    n_axes: int = 100

    def __post_init__(self):
        if not 0.0 < float(self.half_angle_deg) < 90.0:
            raise ValueError(f"the cone half-angle must be in (0, 90) degrees, got {self.half_angle_deg}")
        if int(self.n_axes) < 1:
            raise ValueError(f"the number of cone axes must be positive, got {self.n_axes}")
        gap = largest_axis_distance_deg(self.axes())
        if gap > float(self.half_angle_deg):
            raise ValueError(
                f"{self.n_axes} cones of {self.half_angle_deg} deg leave directions {gap:.1f} deg from every axis, "
                "outside every cone: use more axes or a wider cone"
            )

    def axes(self) -> np.ndarray:
        """The cone axes, shape ``(n_axes, 3)``."""
        return cone_axes(int(self.n_axes))

    @property
    def cos_half_angle(self) -> float:
        """The cosine of the half-angle: ``nu`` is in the cone of ``a`` when ``|nu . a| >= |nu| cos(half_angle)``."""
        return math.cos(math.radians(float(self.half_angle_deg)))


def _full_padded(values, accumulator_shape: tuple) -> np.ndarray:
    """The centered full padded grid of an accumulator stored full or as a packed half, on the host."""
    from relax.reconstruction.regularization import packed_half_to_full_numpy

    shape = tuple(int(s) for s in accumulator_shape)
    array = np.asarray(values)
    full_size = int(np.prod(shape))
    half_size = shape[0] * shape[1] * (shape[2] // 2 + 1)
    if array.size == full_size:
        return array.reshape(shape)
    if array.size == half_size:
        return packed_half_to_full_numpy(array, shape)
    raise ValueError(f"an accumulator of grid {shape} has {full_size} or {half_size} entries, got {array.size}")


@dataclass(frozen=True)
class NativeGrid:
    """The two halves' unregularized averages on the native grid of radius ``r_max + 1`` (RELION's
    ``getDownsampledAverage``), for the voxels ``0 < |nu| <= r_max + 1``.

    ``coords`` is ``(n, 3)`` integer frequencies in the accumulators' axis order, ``radius`` their length,
    ``averages`` the ``(2, n)`` complex averages and ``weights`` the ``(2, n)`` summed padded weights.
    """

    r_max: int
    coords: np.ndarray
    radius: np.ndarray
    averages: np.ndarray
    weights: np.ndarray

    @property
    def cube_radius(self) -> int:
        """The half-width of the native cube the voxels index: ``r_max + 1``."""
        return self.r_max + 1


def native_half_averages(numerators, denominators, *, accumulator_shape, padding_factor: int, r_max: int) -> NativeGrid:
    """Both halves' padded accumulators averaged onto the native grid, as RELION's FSC does.

    A padded voxel ``k`` goes to native voxel ``ROUND(k / p)``; a native average is its summed data over its summed
    weight (0 without weight). The voxels out to ``r_max + 1`` are kept: the statistics read those within ``r_max``,
    the prior term looks up ``ROUND(k / p)`` for ``|k| < p r_max``, which lies within ``r_max + 1``.
    """
    shape = tuple(int(s) for s in accumulator_shape)
    p = int(padding_factor)
    cube = int(r_max) + 1
    side = 2 * cube + 1
    bins = []
    for n in shape:
        native = relion_round_array((np.arange(n, dtype=np.float64) - n // 2) / p).astype(np.int64)
        keep = np.flatnonzero(np.abs(native) <= cube)
        bins.append((slice(int(keep[0]), int(keep[-1]) + 1), native[keep] + cube))
    (s0, b0), (s1, b1), (s2, b2) = bins
    plane_labels = (b1[:, None] * side + b2[None, :]).reshape(-1)

    def downsample(values, *, complex_values):
        # One padded plane at a time into its native plane: bins of one plane, not of the cube.
        grid = _full_padded(values, shape)
        sums = np.zeros((side, side * side), dtype=np.complex128 if complex_values else np.float64)
        for local, i0 in enumerate(range(s0.start, s0.stop)):
            plane = grid[i0, s1, s2].reshape(-1)
            target = sums[int(b0[local])]
            target += np.bincount(plane_labels, weights=plane.real, minlength=side * side)
            if complex_values:
                target += 1j * np.bincount(plane_labels, weights=plane.imag, minlength=side * side)
        return sums.reshape(-1)

    axis = np.arange(-cube, cube + 1, dtype=np.int64)
    c0, c1, c2 = np.meshgrid(axis, axis, axis, indexing="ij")
    radius_cube = np.sqrt((c0 * c0 + c1 * c1 + c2 * c2).astype(np.float64)).reshape(-1)
    kept = np.flatnonzero((radius_cube > 0.0) & (radius_cube <= float(cube)))
    coords = np.stack([c0.reshape(-1)[kept], c1.reshape(-1)[kept], c2.reshape(-1)[kept]], axis=1)
    averages = np.zeros((2, kept.size), dtype=np.complex128)
    weights = np.zeros((2, kept.size), dtype=np.float64)
    for half, (numerator, denominator) in enumerate(zip(numerators, denominators, strict=True)):
        weight = downsample(denominator, complex_values=False)[kept]
        data = downsample(numerator, complex_values=True)[kept]
        has_weight = weight > 0.0
        averages[half, has_weight] = data[has_weight] / weight[has_weight]
        weights[half] = weight
    return NativeGrid(r_max=int(r_max), coords=coords, radius=radius_cube[kept], averages=averages, weights=weights)


def relion_ssnr(fsc, *, whole: bool) -> np.ndarray:
    """RELION's SSNR of an FSC without the fudge factor: clamp to [0.001, 0.999], ``sqrt(2 F / (F + 1))`` first for
    a whole-data map (``is_whole_instead_of_half``), then ``F / (1 - F)``."""
    fsc = np.maximum(np.asarray(fsc, dtype=np.float64), FSC_CLAMP)
    if whole:
        fsc = np.sqrt(2.0 * fsc / (fsc + 1.0))
    fsc = np.minimum(fsc, 1.0 - FSC_CLAMP)
    return fsc / (1.0 - fsc)


@dataclass(frozen=True)
class ConeShellSums:
    """Sums over the native voxels of each cone-shell bin (shape ``(n_axes, r_max + 1)``) and of each shell.

    ``power`` holds each half's ``|F|^2`` sum (leading axis 2); ``weight`` each weight set's summed weight (leading
    axis: the two halves for a split-half prior, their sum for a whole-data one). Shell 0 is empty.
    """

    counts: np.ndarray
    cross: np.ndarray
    power: np.ndarray
    weight: np.ndarray
    shell_counts: np.ndarray
    shell_cross: np.ndarray
    shell_power: np.ndarray
    shell_weight: np.ndarray

    def cone_fsc(self) -> np.ndarray:
        """Each cone's FSC per shell; 0 where a bin has no power."""
        return _fsc(self.cross, self.power)

    def shell_fsc(self) -> np.ndarray:
        """The shell FSC of the same voxels; 0 where a shell has no power."""
        return _fsc(self.shell_cross, self.shell_power)


def _fsc(cross, power) -> np.ndarray:
    """``cross / sqrt(power_1 power_2)``, 0 where the product is 0."""
    product = power[0] * power[1]
    out = np.zeros_like(cross, dtype=np.float64)
    np.divide(cross, np.sqrt(product), out=out, where=product > 0.0)
    return out


def cone_shell_sums(native: NativeGrid, weight_sets: np.ndarray, axes: np.ndarray, cos_half_angle: float) -> ConeShellSums:
    """The FSC and weight sums of every cone-shell bin over the native voxels ``0 < |nu| <= r_max``.

    The shell of a voxel is ``ROUND(|nu|)`` (RELION's FSC binning); ``weight_sets`` is ``(n_sets, n)``.
    """
    r_max = native.r_max
    n_shells = r_max + 1
    n_axes = len(axes)
    n_sets = weight_sets.shape[0]
    in_support = native.radius <= float(r_max)
    shells = np.floor(native.radius + 0.5).astype(np.int64)
    cross_v = (native.averages[0] * np.conj(native.averages[1])).real
    power_v = np.abs(native.averages) ** 2
    counts = np.zeros((n_axes, n_shells), dtype=np.int64)
    cross = np.zeros((n_axes, n_shells))
    power = np.zeros((2, n_axes, n_shells))
    weight = np.zeros((n_sets, n_axes, n_shells))
    by_shell = _voxels_by_shell(np.where(in_support, shells, n_shells), n_shells + 1)
    for s in range(1, n_shells):
        idx = by_shell[s]
        if idx.size == 0:
            continue
        inside = _cone_membership(native.coords[idx], native.radius[idx], axes, cos_half_angle).astype(np.float64)
        counts[:, s] = np.rint(inside.sum(axis=0)).astype(np.int64)
        cross[:, s] = inside.T @ cross_v[idx]
        power[:, :, s] = (inside.T @ power_v[:, idx].T).T
        weight[:, :, s] = (inside.T @ weight_sets[:, idx].T).T
    shell_index = np.where(in_support, shells, 0)
    shell_counts = np.bincount(shell_index[in_support], minlength=n_shells)
    shell_cross = np.bincount(shell_index[in_support], weights=cross_v[in_support], minlength=n_shells)
    shell_power = np.stack(
        [np.bincount(shell_index[in_support], weights=power_v[h, in_support], minlength=n_shells) for h in range(2)]
    )
    shell_weight = np.stack(
        [np.bincount(shell_index[in_support], weights=weight_sets[w, in_support], minlength=n_shells) for w in range(n_sets)]
    )
    return ConeShellSums(counts, cross, power, weight, shell_counts, shell_cross, shell_power, shell_weight)


def _voxels_by_shell(shells: np.ndarray, n_shells: int) -> list:
    """The indices of the voxels of each shell ``0 .. n_shells - 1`` (one sort instead of a scan per shell)."""
    order = np.argsort(shells, kind="stable")
    edges = np.searchsorted(shells[order], np.arange(n_shells + 1))
    return [order[edges[s] : edges[s + 1]] for s in range(n_shells)]


def _cone_membership(coords, radius, axes, cos_half_angle) -> np.ndarray:
    """``(n, n_axes)`` booleans: whether each voxel lies in each double cone."""
    return np.abs(coords.astype(np.float64) @ axes.T) >= radius[:, None] * cos_half_angle


def cone_to_shell_ratios(sums: ConeShellSums, *, whole: bool) -> tuple[np.ndarray, np.ndarray]:
    """RELION's estimate of each cone-shell bin over the shell's: ``(ratios, measured)``, shapes
    ``(n_sets, n_axes, r_max + 1)`` and ``(n_axes, r_max + 1)``.

    ``ratio = [SSNR_cone / SSNR_shell] * [mean weight of the shell / mean weight of the cone]``: the cone's
    ``SSNR sigma2`` over the shell's, ``sigma2`` being RELION's inverse mean weight. A bin with fewer than
    ``MIN_CONE_VOXELS`` voxels, no power or no weight keeps the shell's value (ratio 1); so does shell 0.
    """
    ssnr_ratio = relion_ssnr(sums.cone_fsc(), whole=whole) / relion_ssnr(sums.shell_fsc(), whole=whole)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        cone_mean = sums.weight / sums.counts[None, :, :]
        shell_mean = sums.shell_weight / sums.shell_counts[None, :]
        ratios = ssnr_ratio[None, :, :] * shell_mean[:, None, :] / cone_mean
    measured = (
        (sums.counts >= MIN_CONE_VOXELS)
        & (sums.power[0] * sums.power[1] > 0.0)
        & (sums.shell_power[0] * sums.shell_power[1] > 0.0)[None, :]
        & np.all(cone_mean > 0.0, axis=0)
    )
    measured[:, 0] = False
    ratios = np.where(measured[None, :, :], ratios, 1.0)
    return ratios, measured


def voxel_ratio_cubes(native: NativeGrid, ratios: np.ndarray, axes: np.ndarray, cos_half_angle: float) -> np.ndarray:
    """Each weight set's ratio at every native voxel: the mean of its cone-shell ratios over the cones that contain
    the voxel, at the voxel's shell (clipped to ``1 .. r_max``). Shape ``(n_sets, side, side, side)`` over the cube
    of half-width ``r_max + 1`` in the accumulators' axis order; 1 at the origin and outside the kept sphere."""
    n_sets, _, n_shells = ratios.shape
    cube = native.cube_radius
    side = 2 * cube + 1
    shells = np.clip(np.floor(native.radius + 0.5).astype(np.int64), 1, n_shells - 1)
    values = np.ones((n_sets, native.radius.size))
    by_shell = _voxels_by_shell(shells, n_shells)
    for s in range(1, n_shells):
        idx = by_shell[s]
        if idx.size == 0:
            continue
        inside = _cone_membership(native.coords[idx], native.radius[idx], axes, cos_half_angle).astype(np.float64)
        n_cones = inside.sum(axis=1)
        if np.any(n_cones == 0):
            raise RuntimeError(f"a native voxel of shell {s} lies in no cone; the cone settings should cover the sphere")
        values[:, idx] = (inside @ ratios[:, :, s].T).T / n_cones[None, :]
    cubes = np.ones((n_sets, side**3))
    flat = ((native.coords[:, 0] + cube) * side + (native.coords[:, 1] + cube)) * side + (native.coords[:, 2] + cube)
    cubes[:, flat] = values
    return cubes.reshape(n_sets, side, side, side)


@dataclass(frozen=True)
class ConePrior:
    """The conical-FSC prior of one M-step: RELION's shell tau2 of each weight set times its per-voxel ratio.

    ``tau2_shells`` holds RELION's shell curve of each weight set (each half of a split-half M-step, the joined
    halves of the final one); ``ratio_cubes`` the per-voxel ratios (:func:`voxel_ratio_cubes`); ``sums``,
    ``ratios`` and ``measured`` the cone statistics they come from. ``solve_dtype`` is the dtype of RELION's
    regularized weight on this path (float64 when the shell prior is promoted, as the numbered solves do).
    """

    settings: ConicalPriorSettings
    whole: bool
    volume_shape: tuple
    accumulator_shape: tuple
    padding_factor: int
    current_size: int
    tau2_fudge: float
    minres_map: int
    tau2_shells: tuple
    sums: ConeShellSums
    ratios: np.ndarray
    measured: np.ndarray
    ratio_cubes: np.ndarray
    solve_dtype: np.dtype

    @property
    def r_max(self) -> int:
        """RELION's support radius of this M-step, ``current_size // 2``."""
        return int(self.current_size) // 2

    def weight_term(self, index: int, *, packed: bool) -> np.ndarray:
        """``1 / (p^3 T tau2(k))`` of weight set ``index`` at every voxel of the accumulator (flat, float64), 0
        outside RELION's MAP support (``|k| >= p r_max`` or ``ires < minres_map``).

        ``tau2(k) = tau2_shells[index][ires(k)] * rho(ROUND(k / p))`` with ``ires(k) = ROUND(|k| / p)`` clipped to
        the curve, as RECOVAR maps a shell curve onto the accumulator. ``packed`` selects the packed-half layout
        (last axis ``0 .. n // 2``) over the full one.
        """
        shape = tuple(int(s) for s in self.accumulator_shape)
        p = int(self.padding_factor)
        tau2 = np.asarray(self.tau2_shells[index], dtype=np.float64)
        cube = self.r_max + 1
        freqs = [np.arange(n, dtype=np.int64) - n // 2 for n in shape]
        if packed:
            freqs[2] = np.arange(shape[2] // 2 + 1, dtype=np.int64)
        native = [np.clip(relion_round_array(f / p).astype(np.int64), -cube, cube) + cube for f in freqs]
        support_r2 = (p * self.r_max) ** 2
        rho = self.ratio_cubes[index]
        scale = float(p**3) * float(self.tau2_fudge)
        k1_sq = freqs[1][:, None] ** 2 + freqs[2][None, :] ** 2
        out = np.zeros((len(freqs[0]), len(freqs[1]), len(freqs[2])), dtype=np.float64)
        for start in range(0, len(freqs[0]), _PLANES_PER_BLOCK):
            block = slice(start, min(start + _PLANES_PER_BLOCK, len(freqs[0])))
            r2 = freqs[0][block, None, None] ** 2 + k1_sq[None, :, :]
            ires = np.minimum(np.floor(np.sqrt(r2.astype(np.float64)) / p + 0.5).astype(np.int64), tau2.size - 1)
            support = (r2 < support_r2) & (ires >= int(self.minres_map))
            shell_tau2 = tau2[ires]
            if np.any(support & (shell_tau2 < ZERO_TAU2)):
                raise ValueError("a shell tau2 inside the MAP support is zero; RELION's zero-tau2 fallback is not reproduced")
            local_rho = rho[native[0][block, None, None], native[1][None, :, None], native[2][None, None, :]]
            with np.errstate(divide="ignore"):
                out[block] = np.where(support, 1.0 / (scale * shell_tau2 * local_rho), 0.0)
        return out.reshape(-1)

    def regularized_weight(self, index: int, weight):
        """The accumulator weight ``weight`` plus weight set ``index``'s prior term, in ``solve_dtype``, in the
        layout and shape of ``weight`` (a host array stays on the host, a device array on the device). The solve
        then runs without a shell prior (``tau=None``)."""
        from recovar.reconstruction import relion_functions

        reconstruction_shape = relion_functions._relion_reconstruction_padded_shape(self.volume_shape, self.padding_factor)
        if relion_functions.large_grid_postprocess_single_precision_enabled(int(np.prod(reconstruction_shape))):
            raise ValueError("the conical-FSC prior does not take RECOVAR's large-grid single-precision reconstruction")
        shape = tuple(int(s) for s in self.accumulator_shape)
        full_size = int(np.prod(shape))
        packed = int(np.size(weight)) != full_size
        term = self.weight_term(index, packed=packed)
        if term.size != int(np.size(weight)):
            raise ValueError(f"an accumulator weight of {np.size(weight)} entries does not fit grid {shape}")
        dtype = np.dtype(self.solve_dtype)
        if isinstance(weight, np.ndarray):
            return (weight.real.reshape(-1).astype(dtype) + term.astype(dtype)).reshape(weight.shape)
        import jax.numpy as jnp

        device_weight = jnp.asarray(weight)
        return (device_weight.real.reshape(-1).astype(dtype) + jnp.asarray(term, dtype=dtype)).reshape(device_weight.shape)

    def star_tables(self, index: int) -> tuple[list, dict]:
        """Weight set ``index``'s statistics as a model STAR's extra blocks: the settings (list items) and one row
        per cone-shell bin from shell 1 (axis index from 1, axis components, shell, voxels, cone FSC, ratio,
        measured flag)."""
        n_axes, n_shells = self.ratios.shape[1:]
        axes = self.settings.axes()
        axis_index = np.repeat(np.arange(n_axes), n_shells - 1)
        shell = np.tile(np.arange(1, n_shells), n_axes)
        general = [
            ("rlnRelaxCfscHalfAngle", float(self.settings.half_angle_deg)),
            ("rlnRelaxCfscAxes", int(n_axes)),
            ("rlnRelaxCfscMinVoxels", int(MIN_CONE_VOXELS)),
            ("rlnRelaxCfscWholeData", bool(self.whole)),
        ]
        bins = {
            "rlnRelaxCfscAxis": axis_index + 1,
            "rlnRelaxCfscAxisX": axes[axis_index, 0],
            "rlnRelaxCfscAxisY": axes[axis_index, 1],
            "rlnRelaxCfscAxisZ": axes[axis_index, 2],
            "rlnSpectralIndex": shell,
            "rlnRelaxCfscVoxels": self.sums.counts[axis_index, shell],
            "rlnRelaxCfscConeFsc": self.sums.cone_fsc()[axis_index, shell],
            "rlnRelaxCfscTau2Ratio": self.ratios[index][axis_index, shell],
            "rlnRelaxCfscMeasured": self.measured[axis_index, shell].astype(np.int64),
        }
        return general, bins

    def log_summary(self, label: str) -> None:
        """One log line per weight set: the spread of the measured cone-shell ratios and of the cone FSCs at the shell
        where the shell FSC first falls below 0.5."""
        fsc = self.sums.cone_fsc()
        shell_fsc = self.sums.shell_fsc()
        below = np.flatnonzero(shell_fsc[1:] < 0.5)
        shell = int(below[0]) + 1 if below.size else self.r_max
        for index in range(self.ratios.shape[0]):
            values = self.ratios[index][self.measured]
            quantiles = np.quantile(values, [0.0, 0.1, 0.5, 0.9, 1.0]) if values.size else [float("nan")] * 5
            logger.info(
                "%s conical-FSC prior set %d: %d of %d cone-shell bins measured; tau2 ratio min/p10/median/p90/max "
                "%.3g/%.3g/%.3g/%.3g/%.3g; shell %d (shell FSC %.3f): cone FSC %.3f..%.3f",
                label, index + 1, int(self.measured.sum()), self.measured.size, *quantiles, shell,
                float(shell_fsc[shell]), float(fsc[:, shell].min()), float(fsc[:, shell].max()),
            )


def estimate_cone_prior(
    numerators,
    denominators,
    tau2_shells,
    *,
    settings: ConicalPriorSettings,
    volume_shape,
    accumulator_shape,
    padding_factor: int,
    current_size: int,
    tau2_fudge: float,
    minres_map: int,
    whole: bool,
    solve_dtype,
) -> ConePrior:
    """The conical-FSC prior of an M-step from its two half accumulators and RELION's shell tau2.

    A split-half M-step (``whole=False``) passes each half's shell curve in ``tau2_shells`` and gets a ratio per half
    from that half's weight; the final M-step (``whole=True``) passes the joined halves' one curve and gets one ratio
    from the summed weights, with RELION's whole-map SSNR.
    """
    if len(numerators) != 2 or len(denominators) != 2:
        raise ValueError("the conical-FSC prior needs the two half accumulators")
    tau2_shells = tuple(np.asarray(t, dtype=np.float64) for t in tau2_shells)
    if len(tau2_shells) != (1 if whole else 2):
        raise ValueError(f"a {'whole-data' if whole else 'split-half'} prior needs {1 if whole else 2} shell curves")
    r_max = int(current_size) // 2
    native = native_half_averages(
        numerators, denominators, accumulator_shape=accumulator_shape, padding_factor=padding_factor, r_max=r_max,
    )
    weight_sets = native.weights.sum(axis=0, keepdims=True) if whole else native.weights
    axes = settings.axes()
    sums = cone_shell_sums(native, weight_sets, axes, settings.cos_half_angle)
    ratios, measured = cone_to_shell_ratios(sums, whole=whole)
    cubes = voxel_ratio_cubes(native, ratios, axes, settings.cos_half_angle)
    return ConePrior(
        settings=settings,
        whole=bool(whole),
        volume_shape=tuple(int(s) for s in volume_shape),
        accumulator_shape=tuple(int(s) for s in accumulator_shape),
        padding_factor=int(padding_factor),
        current_size=int(current_size),
        tau2_fudge=float(tau2_fudge),
        minres_map=int(minres_map),
        tau2_shells=tau2_shells,
        sums=sums,
        ratios=ratios,
        measured=measured,
        ratio_cubes=cubes,
        solve_dtype=np.dtype(solve_dtype),
    )
