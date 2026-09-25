"""relax map files carry RELION's convention.

relax holds real-space volumes in RECOVAR's internal frame, which is the negated
(2, 1, 0) transpose of the array RELION writes (``recovar.utils.helpers.relion_volume_to_recovar``).
The negation comes with RECOVAR's projector and Euler conventions: the RELION projector
setup already converts every reference with ``recovar_volume_to_relion`` before projecting.
Map files are always in RELION's convention, so a relax map and the RELION map of the same
run agree voxel for voxel and in sign, and a RELION map can be read back as a reference:

* :func:`write_map` converts an internal-frame volume on the way out with
  ``recovar_volume_to_relion`` (what ``write_relion_mrc`` writes) and labels the file with
  :data:`RELAX_MAP_LABEL`.
* Reference and RELION maps are read with ``recovar.utils.helpers.load_relion_volume``.
* :func:`load_relax_map` reads a map relax wrote and checks the label.

The header matches RELION's writer (``rwMRC.h``: float32 mode 2, origin 0, start 0, axis
order 1 2 3, voxel size, space group 0, one label).

relax maps written before this convention hold the negated array (RECOVAR ``write_mrc``) and
have no label. ``load_relax_map(path, legacy_recovar_sign=True)`` reads such a map; without
the option an unlabeled map is refused, so an old map is never silently read with the wrong
sign.
"""

from __future__ import annotations

from pathlib import Path

import mrcfile
import numpy as np
from recovar.utils.helpers import load_mrc, load_relion_volume, recovar_volume_to_relion

RELAX_MAP_LABEL = "relax map, RELION sign and axis convention"


def write_map(path, volume, voxel_size=None) -> None:
    """Write an internal-frame real volume as a RELION-convention MRC map."""
    volume = np.asarray(volume)
    if volume.ndim != 3 or len(set(volume.shape)) != 1:
        raise ValueError(f"write_map expects a cubic volume, got shape {volume.shape}")
    with mrcfile.new(str(path), overwrite=True) as handle:
        handle.set_data(np.asarray(recovar_volume_to_relion(volume.real), dtype=np.float32))
        if voxel_size is not None:
            handle.voxel_size = voxel_size
        handle.header.ispg = 0  # rwMRC.h leaves the space group at 0
        handle.header.label[0] = RELAX_MAP_LABEL.encode("ascii")
        handle.header.nlabl = 1


def write_map_from_ft(path, volume_ft, volume_shape, voxel_size=None) -> None:
    """Write a centered internal-frame Fourier volume (flat or shaped) as a map."""
    import jax.numpy as jnp
    from recovar.core import fourier_transform_utils as ftu

    shape = tuple(int(n) for n in volume_shape)
    real = np.real(np.asarray(ftu.get_idft3(jnp.asarray(np.asarray(volume_ft).reshape(shape)))))
    write_map(path, real.astype(np.float32), voxel_size=voxel_size)


def map_labels(path) -> list[str]:
    """The text labels of an MRC header."""
    with mrcfile.open(str(path), header_only=True, permissive=True) as handle:
        count = int(handle.header.nlabl)
        return [bytes(label).decode("ascii", "replace").strip() for label in handle.header.label[:count]]


def is_relax_map(path) -> bool:
    """True when ``path`` was written by :func:`write_map`."""
    return RELAX_MAP_LABEL in map_labels(path)


def load_relax_map(path, *, legacy_recovar_sign: bool = False) -> np.ndarray:
    """Read a map relax wrote, in the internal frame.

    ``legacy_recovar_sign`` reads an unlabeled map as one written before the RELION
    convention (RECOVAR ``load_mrc``); a labeled map is always read in RELION's convention.
    """
    if is_relax_map(path):
        return load_relion_volume(str(path))
    labels = map_labels(path)
    if any(label.startswith("Relion") for label in labels):
        raise ValueError(f"{path} was written by RELION; read it with load_relion_volume")
    if not legacy_recovar_sign:
        raise ValueError(
            f"{Path(path).name} has no relax map label: it predates the RELION map convention "
            "and holds the negated array; pass legacy_recovar_sign=True to read it"
        )
    return load_mrc(str(path))
