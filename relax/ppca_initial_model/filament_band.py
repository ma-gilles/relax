"""The filament band: a fixed-frame search for one attachment around an aligned filament.

Each particle's frame is its filament's (``--particle-frame``: RELION's ``A_subtomogram * A_particle`` composed into
every tilt's projection), with the filament axis along +z. An attachment is a copy of the model placed by a
rotation about the axis and a shift along it,

    T(a, phi) f (r) = f(R_z(phi)^T (r - a e_z)),

with its canonical support near ``support_center_ang`` on +x. The pose grid is the Cartesian product of
``phi`` in ``[0, 360)`` every ``phi_step_deg`` and ``a`` in ``[-length/2, length/2]`` every ``axial_step_ang``, with
uniform normalized priors: a refinement of the grid then leaves the occupied evidence of the empty-component model
(:mod:`relax.ppca_refinement.occupancy`) unchanged up to the quadrature.

The rotation candidates are RELION Euler angles ``(phi, 0, 0)`` (``Euler_angles2matrix``: a rotation about z),
kept as canonical metadata; the translations are 3D shifts ``(0, 0, a)`` in pixels in the particle frame, which the
tilt reader projects with each frame's ``Aproj`` (:func:`relax.ppca_initial_model.tomo.tilt_shifts`).
"""

import dataclasses

import jax.numpy as jnp
import numpy as np

from relax.sampling.healpix import euler_angles_to_matrix


@dataclasses.dataclass(frozen=True)
class FilamentBand:
    length_ang: float
    axial_step_ang: float = 15.0
    phi_step_deg: float = 5.0
    # The attachment's canonical support: a soft sphere at (support_center_ang, 0, 0) in the particle frame.
    support_center_ang: float = 180.0
    support_radius_ang: float = 60.0
    # The support excludes the cylinder of this radius about the axis (None: nothing excluded): what a fixed core
    # models there (its leftover after subtraction is shared by every window) is not the attachment's.
    core_exclusion_radius_ang: float | None = None

    def __post_init__(self):
        if not self.length_ang >= 0 or not self.axial_step_ang > 0 or not 0 < self.phi_step_deg <= 360:
            raise ValueError(f"Invalid filament band {self}")
        if not 360.0 / self.phi_step_deg == round(360.0 / self.phi_step_deg):
            raise ValueError(f"phi_step_deg must divide 360, got {self.phi_step_deg}")
        if not self.support_radius_ang > 0 or self.support_center_ang < 0:
            raise ValueError(f"Invalid attachment support {self}")
        if self.core_exclusion_radius_ang is not None and not (
            0 < self.core_exclusion_radius_ang < self.support_center_ang + self.support_radius_ang
        ):
            raise ValueError(f"The core exclusion leaves no attachment support: {self}")

    def phis_deg(self) -> np.ndarray:
        return np.arange(round(360.0 / self.phi_step_deg)) * np.float64(self.phi_step_deg)

    def axial_offsets_ang(self) -> np.ndarray:
        """Shifts along the axis, symmetric about 0 and including it; the outermost lie within ``length / 2``."""
        half = int(np.floor(self.length_ang / 2 / self.axial_step_ang + 1e-9))
        return np.arange(-half, half + 1) * np.float64(self.axial_step_ang)

    def eulers(self) -> np.ndarray:
        """Canonical ``(R, 3)`` RELION Euler angles of the rotation candidates."""
        phis = self.phis_deg()
        return np.stack([phis, np.zeros_like(phis), np.zeros_like(phis)], axis=1)

    def rotations(self) -> np.ndarray:
        return euler_angles_to_matrix(self.eulers()).astype(np.float32)

    def translations_px(self, voxel_size: float) -> np.ndarray:
        """``(T, 3)`` float32 particle-frame shifts ``(0, 0, a)`` in pixels."""
        a = self.axial_offsets_ang() / voxel_size
        return np.stack([np.zeros_like(a), np.zeros_like(a), a], axis=1).astype(np.float32)

    def support_center_px(self, n: int, voxel_size: float) -> np.ndarray:
        """The support centre in the internal volume array's index space (axis 0 = RELION x, relax/io/map_io.py)."""
        return np.array([n // 2 + self.support_center_ang / voxel_size, n // 2, n // 2], np.float64)


def uniform_log_prior(n: int) -> np.ndarray:
    return np.full(n, -np.log(n), np.float32)


def attachment_support_mask(
    n: int, voxel_size: float, band: FilamentBand, edge: int = 5, exclusion_edge: int = 2
) -> jnp.ndarray:
    """Soft sphere of ``band.support_radius_ang`` at the attachment's canonical centre on an ``n``-cube, float32: 1
    inside, a raised-cosine fall to 0 over ``edge`` voxels, as :func:`initialization.support_mask` about its centre.
    With ``band.core_exclusion_radius_ang`` it is 0 within that distance of the axis (axis 2 through the box centre)
    and rises to 1 over ``exclusion_edge`` voxels outside it; the cylinder is invariant under the band's poses, so
    the placed support never covers the core."""
    center = band.support_center_px(n, voxel_size)
    radius = band.support_radius_ang / voxel_size
    if center[0] + radius + edge > n:
        raise ValueError(f"The attachment support ({band}) leaves the {n}-voxel box at {voxel_size} A/px")
    grid = np.indices((n, n, n), dtype=np.float64)
    distance = np.sqrt(sum((grid[i] - center[i]) ** 2 for i in range(3)))
    weight = np.where(
        distance <= radius,
        1.0,
        np.where(distance >= radius + edge, 0.0, 0.5 + 0.5 * np.cos(np.pi * (distance - radius) / edge)),
    )
    if band.core_exclusion_radius_ang is not None:
        inner = band.core_exclusion_radius_ang / voxel_size
        rho = np.sqrt((grid[0] - n // 2) ** 2 + (grid[1] - n // 2) ** 2)
        weight = weight * np.where(
            rho <= inner,
            0.0,
            np.where(rho >= inner + exclusion_edge, 1.0, 0.5 - 0.5 * np.cos(np.pi * (rho - inner) / exclusion_edge)),
        )
    return jnp.asarray(weight, jnp.float32)
