"""The bodies of a multi-body refinement: masks, centres, orientations and priors, as RELION sets them up.

RELION reads a body STAR file (``--multibody_masks``) in ``MlModel::initialiseBodies`` (ml_model.cpp:1168-1340 at
f2c1a38). Every quantity here is in RELION's map frame: a map array ``m[k, i, j]`` read from an MRC file, with
Xmipp logical coordinates ``(x, y, z) = (j, i, k) - D // 2``. A volume in the recovar frame is
``-transpose(relion_volume, (2, 1, 0))`` for densities and ``transpose(relion_mask, (2, 1, 0))`` for masks.

``RESIDUAL_IDENTITY_EULER`` is the residual pose of a body that has not moved: RELION composes a body's
projection matrix as ``A_consensus O^T A_rot90 R(residual) O`` (acc_ml_optimiser_impl.h:1084-1085), where
``A_rot90 R(0, 90, 0)`` is the identity, and starts every body there (``Experiment::initialiseBodies``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mrcfile
import numpy as np

from relax.relion.macros import relion_round
from relax.star.blocks import read_star_blocks

RESIDUAL_IDENTITY_EULER = (0.0, 90.0, 0.0)
# rotation3DMatrix(90., 'Y', A_rot90) (ml_optimiser.cpp:2186, transformations.cpp:118-124 at f2c1a38).
A_ROT90 = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
# A mask voxel above this value counts towards a body's radius (ml_model.cpp:1217).
MASK_RADIUS_THRESHOLD = 0.05


@dataclass(frozen=True)
class Body:
    """One body as ``MlModel::initialiseBodies`` leaves it (RELION map frame, pixels unless stated)."""

    mask: np.ndarray  # [D, D, D] soft mask in [0, 1], RELION frame
    com: np.ndarray  # [3] rounded centre of mass (x, y, z), pixels: com_bodies
    orientation: np.ndarray  # [3, 3] orient_bodies
    max_radius_angstrom: float  # max_radius_mask_bodies
    sigma_tilt_deg: float
    sigma_psi_deg: float
    sigma_offset_angstrom: float
    keep_fixed: bool
    reference_path: Path | None = None  # rlnBodyReferenceName: the body's start map; None keeps the consensus half maps


def relion_center_of_mass(volume: np.ndarray) -> np.ndarray:
    """``MultidimArray::centerOfMass`` of a 3D array: the positive voxels' weighted mean logical (x, y, z)."""

    volume = np.asarray(volume, dtype=np.float64)
    weights = np.where(volume > 0.0, volume, 0.0)
    mass = weights.sum()
    if mass == 0.0:
        return np.zeros(3)
    k, i, j = (np.arange(n) - n // 2 for n in volume.shape)
    return np.array([
        (weights.sum(axis=(0, 1)) * j).sum() / mass,
        (weights.sum(axis=(0, 2)) * i).sum() / mass,
        (weights.sum(axis=(1, 2)) * k).sum() / mass,
    ])


def align_with_z(axis: np.ndarray) -> np.ndarray:
    """``alignWithZ(axis, result, false)`` (transformations.cpp:138-181): the rotation taking ``axis`` to +z."""

    axis = np.asarray(axis, dtype=np.float64)
    x, y, z = axis / np.sqrt(np.sum(axis * axis))
    proj_mod = np.sqrt(y * y + z * z)
    if proj_mod > 1e-6:  # XMIPP_EQUAL_ACCURACY
        return np.array([
            [proj_mod, -x * y / proj_mod, -x * z / proj_mod],
            [0.0, z / proj_mod, -y / proj_mod],
            [x, y, z],
        ])
    sign = -1.0 if x > 0 else 1.0
    return np.array([[0.0, 0.0, sign], [0.0, 1.0, 0.0], [-sign, 0.0, 0.0]])


def _mask_max_radius_angstrom(mask: np.ndarray, com_unrounded: np.ndarray, pixel_size: float) -> float:
    """``max_radius_mask_bodies``: CEIL(pixel_size sqrt(max d2)), d2 the int-truncated squared distance of every
    voxel above 0.05 from the unrounded centre of mass (ml_model.cpp:1214-1223)."""

    k, i, j = np.nonzero(mask > MASK_RADIUS_THRESHOLD)
    n = mask.shape[0] // 2
    d2 = ((k - n) - com_unrounded[2]) ** 2 + ((i - n) - com_unrounded[1]) ** 2 + ((j - n) - com_unrounded[0]) ** 2
    max_d2 = int(np.max(np.trunc(d2))) if d2.size else 0
    return float(np.ceil(pixel_size * np.sqrt(float(max_d2))))


def _sigma_offset_angstrom(row: dict) -> float:
    """The body's offset prior width as ``MlModel::initialiseBodies`` stores it (ml_model.cpp:1262-1273).

    RELION 5.0.1 reads the deprecated ``rlnBodySigmaOffset`` (pixels) into a local, converts it to Angstroms and
    never stores it, so ``sigma_offset_bodies`` stays 0 (ml_model.cpp:49, :1182). A width of 0 flattens the offset
    prior over the translation grid (acc_ml_optimiser_impl.h:3067-3085, relax
    ``make_relion_translation_log_prior``). Reproduced: the default path follows RELION 5.0.1.
    """

    if "rlnBodySigmaOffsetAngst" in row:
        return float(row["rlnBodySigmaOffsetAngst"])
    if "rlnBodySigmaOffset" in row:
        return 0.0
    raise ValueError("the body STAR file needs a rlnBodySigmaOffsetAngst column for each body's offset prior")


def read_bodies(
    body_star: str | Path, *, pixel_size: float, norm_overlap: bool = False, paths_from: str | Path | None = None
) -> list[Body]:
    """The bodies of a RELION body STAR file, set up as ``MlModel::initialiseBodies`` does.

    Supported columns: ``rlnBodyMaskName``, ``rlnBodyRotateRelativeTo``, ``rlnBodySigmaAngles`` (or
    ``rlnBodySigmaTilt`` and ``rlnBodySigmaPsi``), ``rlnBodySigmaOffsetAngst`` or the deprecated
    ``rlnBodySigmaOffset`` (see :func:`_sigma_offset_angstrom`) and ``rlnBodyReferenceName``. Mask and reference paths
    that are not absolute are relative to ``paths_from``, relion_refine's working directory; by default the STAR
    file's directory. A body is kept fixed when all three prior widths are below 0.001.
    """

    body_star = Path(body_star)
    loops = [block for block in read_star_blocks(body_star).values() if "rlnBodyMaskName" in block]
    if len(loops) != 1 or not isinstance(loops[0]["rlnBodyMaskName"], list):
        raise ValueError(f"{body_star}: expected one loop with rlnBodyMaskName")
    rows = [dict(zip(loops[0], values)) for values in zip(*loops[0].values())]
    unsupported = {"rlnBodyRotateDirectionX"} & set(rows[0])
    if unsupported:
        raise NotImplementedError(f"{body_star}: body STAR columns {sorted(unsupported)} are not supported")
    root = body_star.parent if paths_from is None else Path(paths_from)
    masks, coms, coms_unrounded = [], [], []
    for row in rows:
        path = root / row["rlnBodyMaskName"]  # an absolute name replaces root
        mask = np.asarray(mrcfile.read(str(path)), dtype=np.float64)
        if mask.ndim != 3 or mask.min() < 0.0 or mask.max() > 1.0:
            raise ValueError(f"{path}: a body mask is a 3D map with values in [0, 1]")
        com = relion_center_of_mass(mask)
        masks.append(mask)
        coms_unrounded.append(com)
        coms.append(np.array([relion_round(c) for c in com], dtype=np.float64))
    radii = [_mask_max_radius_angstrom(m, c, pixel_size) for m, c in zip(masks, coms_unrounded)]
    if norm_overlap:
        # --multibody_norm_overlap: where the masks sum above one, each is divided by the sum (ml_model.cpp:1368-1378),
        # after the centres and radii were taken from the masks as read.
        total = np.sum(masks, axis=0)
        masks = [np.where(total > 1.0, m / np.where(total > 1.0, total, 1.0), m) for m in masks]
    bodies = []
    for b, row in enumerate(rows):
        relative_to = int(row.get("rlnBodyRotateRelativeTo", 0)) - 1
        # Rotate about the other body's centre, or about the box centre when none is given (ml_model.cpp:1286-1307).
        direction = (coms[relative_to] if relative_to >= 0 else np.zeros(3)) - coms[b]
        if "rlnBodySigmaAngles" in row:
            sigma_tilt = sigma_psi = float(row["rlnBodySigmaAngles"])
        else:
            sigma_tilt, sigma_psi = float(row["rlnBodySigmaTilt"]), float(row["rlnBodySigmaPsi"])
        sigma_offset = _sigma_offset_angstrom(row)
        reference = row.get("rlnBodyReferenceName", "None")
        bodies.append(Body(
            mask=masks[b],
            com=coms[b],
            orientation=align_with_z(-direction / np.sqrt(np.sum(direction * direction))),
            max_radius_angstrom=radii[b],
            sigma_tilt_deg=sigma_tilt,
            sigma_psi_deg=sigma_psi,
            sigma_offset_angstrom=sigma_offset,
            keep_fixed=sigma_tilt < 0.001 and sigma_psi < 0.001 and sigma_offset < 0.001,
            reference_path=None if reference == "None" else root / reference,
        ))
    return bodies


def subtraction_masks(bodies: list[Body], *, norm_overlap: bool = False) -> list[list]:
    """The mask each other body is projected with when it is subtracted from a body's images: ``masks[i][o]``.

    Without ``--multibody_norm_overlap`` RELION removes from body ``o``'s mask its overlap with body ``i`` and with
    the masks already chosen for the bodies before ``o`` (``pointer_body_overlap``, ml_model.cpp:1400-1456): the
    overlap mask ``(m_i + sum of those masks) * m_o``, when it is not empty, gives ``o`` the extra projector
    ``m_o - overlap`` for ``i``'s subtraction only. A pair without overlap, and every pair with
    ``--multibody_norm_overlap`` (masks already normalised), uses the body's own mask, the same array.
    """

    n = len(bodies)
    masks = [[None] * n for _ in range(n)]
    for i in range(n):
        for o in range(n):
            if o == i:
                continue
            if norm_overlap:
                masks[i][o] = bodies[o].mask
                continue
            overlap = bodies[i].mask.copy()
            for old in range(o):
                if old != i:
                    overlap = overlap + masks[i][old]
            overlap = overlap * bodies[o].mask
            masks[i][o] = bodies[o].mask - overlap if overlap.sum() > 0.0 else bodies[o].mask
    return masks
