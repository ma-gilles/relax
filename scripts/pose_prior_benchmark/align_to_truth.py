"""Rigidly align a refined map to the ground-truth map: the frame correction of the pose-prior benchmark.

A refinement started from a map in a slightly rotated or shifted frame (an ab-initio map registered to the ground truth
only as precisely as that fit) returns its maps and poses in that frame, so its FSC to the ground truth and its pose
errors include the offset. ``fit_rigid_to_reference`` finds the rotation and shift that bring a map onto the reference: a
Gauss-Newton fit of the squared difference of the two maps, both low-passed at a given Fourier shell, with a free
overall scale (the refined map and the truth differ in amplitude). The maps' spectral gradient gives the linearization,
so a fit takes a few iterations and is exact for a band-limited map; the transform is applied to the original map with
cubic-spline interpolation. ``evaluate_run.py`` fits the final merged map and applies the transform to it and to both
half maps.

The geometry is ``relax.diagnostics.gt_registration.apply_rigid_volume_transform``: y = c + R (x - c) + t with
c = (N - 1) / 2 and t in voxels, on the arrays ``recovar.utils.helpers.load_relion_volume`` returns. This module is a
library for ``evaluate_run.py``; it has no command line.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation

from relax.diagnostics.gt_metrics import lowpass_volume_by_shell
from relax.diagnostics.gt_registration import apply_rigid_volume_transform

INTERPOLATION_ORDER = 3
MAX_ITERATIONS = 20
# The fit stops when an update turns the map by less than this angle and shifts it by less than this distance. The
# updates of a converged fit do not shrink further: the merged final maps of two runs end on 0.0009 to 0.0011 degrees
# and 0.0002 voxels, their half maps on 0.0015 to 0.0018 degrees and 0.0005 voxels (2026-11-03, box 256 at 2.125 A,
# shell 54).
ROTATION_TOLERANCE_DEG = 0.002
SHIFT_TOLERANCE_VOXELS = 0.002


class RigidFit(NamedTuple):
    """A rigid transform y = c + R (x - c) + t and how the fit that found it ended.

    ``scale`` is the factor by which the transformed map matches the reference; ``iterations`` the Gauss-Newton updates
    taken and ``last_rotation_step_deg`` / ``last_shift_step_voxels`` the size of the final one.
    """

    rotation: np.ndarray
    translation_voxels: np.ndarray
    scale: float
    iterations: int
    last_rotation_step_deg: float
    last_shift_step_voxels: float


def spectral_gradient(volume: np.ndarray) -> list[np.ndarray]:
    """The derivative of ``volume`` along each array axis (per voxel), from its Fourier transform."""
    n = volume.shape[0]
    spectrum = np.fft.fftn(volume)
    derivative = np.fft.fftfreq(n) * 2j * np.pi
    gradient = []
    for axis in range(3):
        shape = [1, 1, 1]
        shape[axis] = n
        gradient.append(np.fft.ifftn(spectrum * derivative.reshape(shape)).real)
    return gradient


def fit_rigid_to_reference(moving: np.ndarray, reference: np.ndarray, shell: int) -> RigidFit:
    """The rotation and shift that bring ``moving`` onto ``reference`` (cubic maps, same shape).

    Both maps are low-passed at Fourier shell ``shell`` and fitted over the sphere inscribed in the box. Each iteration
    transforms the low-passed ``moving`` by the current estimate and solves, by linear least squares, for the small
    rotation, shift and scale that explain the remaining difference to the reference. Raises ``ValueError`` when the
    maps do not correlate positively or the fit has not converged after ``MAX_ITERATIONS`` updates.
    """
    if moving.shape != reference.shape or moving.ndim != 3 or len(set(moving.shape)) != 1:
        raise ValueError(f"the maps must be cubes of one shape, got {moving.shape} and {reference.shape}")
    n = moving.shape[0]
    target = lowpass_volume_by_shell(reference - reference.mean(), shell)
    source = lowpass_volume_by_shell(moving - moving.mean(), shell)
    offset = np.arange(n) - (n - 1) / 2
    position = np.meshgrid(offset, offset, offset, indexing="ij")
    inside = sum(p**2 for p in position) < (n / 2) ** 2
    rotation, shift = np.eye(3), np.zeros(3)
    for iteration in range(1, MAX_ITERATIONS + 1):
        current = apply_rigid_volume_transform(source, rotation, shift, order=INTERPOLATION_ORDER)
        gradient = spectral_gradient(current)
        # A step (omega, dt) moves a map by omega x r + dt, which changes it by -(r x grad) . omega - grad . dt.
        torque = [
            position[1] * gradient[2] - position[2] * gradient[1],
            position[2] * gradient[0] - position[0] * gradient[2],
            position[0] * gradient[1] - position[1] * gradient[0],
        ]
        design = np.stack([current, *(-t for t in torque), *(-g for g in gradient)], axis=-1)[inside]
        solution = np.linalg.lstsq(design, target[inside], rcond=None)[0]
        scale = float(solution[0])
        if not scale > 0:
            raise ValueError(
                f"the map correlates with the reference at scale {scale:.3g}: the frames or the hands differ"
            )
        omega, step = solution[1:4] / scale, solution[4:7] / scale
        turn = Rotation.from_rotvec(omega).as_matrix()
        rotation, shift = turn @ rotation, turn @ shift + step
        last_turn_deg, last_step = float(np.degrees(np.linalg.norm(omega))), float(np.linalg.norm(step))
        if last_turn_deg < ROTATION_TOLERANCE_DEG and last_step < SHIFT_TOLERANCE_VOXELS:
            return RigidFit(rotation, shift, scale, iteration, last_turn_deg, last_step)
    raise ValueError(
        f"the fit did not converge in {MAX_ITERATIONS} iterations (last update {last_turn_deg:.4f} degrees, "
        f"{last_step:.4f} voxels)"
    )


def apply_fit(volume: np.ndarray, fit: RigidFit) -> np.ndarray:
    """``volume`` transformed by ``fit`` (cubic-spline interpolation, zeros outside the box)."""
    return apply_rigid_volume_transform(volume, fit.rotation, fit.translation_voxels, order=INTERPOLATION_ORDER)


def inverse_fit(fit: RigidFit) -> RigidFit:
    """The transform that undoes ``fit`` (the fit's diagnostics are carried over)."""
    return fit._replace(rotation=fit.rotation.T, translation_voxels=-fit.rotation.T @ fit.translation_voxels)


def rotation_angle_deg(rotation: np.ndarray) -> float:
    """The angle (degrees) of a rotation matrix."""
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(rotation).as_rotvec())))
