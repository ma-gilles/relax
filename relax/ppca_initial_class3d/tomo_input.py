"""Multi-class-only input adapter for particles packed into a merged tilt stack.

All geometry, CTF metadata and particle indexing stay owned by the upstream
RELION reader. This adapter changes only its derived per-image stack addresses:
tilt ``t`` of a particle at ``first@file`` is image ``first + t - 1`` of ``file``.
Neither the input STARs nor the shared reader are modified.
"""

import re

import numpy as np

_MERGED_STACK_NAME = re.compile(r"^(\d+)@(\d+)@(.+)$")


def merged_stack_image_names(names) -> list[str] | None:
    """Resolve RECOVAR's ``t@first@file`` addresses, or return None for ordinary stacks.

    Indices are one-based and visible tilts occupy consecutive slices. Refuse a
    mix of merged and ordinary stack layouts rather than resolving only part
    of the dataset.
    """
    matches = [_MERGED_STACK_NAME.match(str(name)) for name in names]
    if not any(matches):
        return None
    if not all(matches):
        raise ValueError("rlnImageName mixes merged-stack slices (first@file) with per-particle stacks")
    resolved = []
    for match in matches:
        tilt, first, path = int(match.group(1)), int(match.group(2)), match.group(3)
        if tilt < 1 or first < 1:
            raise ValueError(f"Stack slices are 1-based: {match.group(0)}")
        resolved.append(f"{first + tilt - 1:06d}@{path}")
    return resolved


def load_tomo_dataset(particles_star, tomograms_star, flat_star, *, datadir, lazy):
    """Load upstream tomography geometry after resolving merged-stack image names.

    The shared ``load_tomo_dataset`` always regenerates the flat STAR itself;
    compose its public dataset constructor and RECOVAR loader here so that the
    address-only correction is applied before any image data are read.
    """
    from recovar.data_io import starfile
    from recovar.data_io.cryoem_dataset import load_dataset
    from recovar.data_io.starfile import read_star, star_column

    from relax.refinement.tomo_half import TomoDataset
    from relax.relion.tomo_input import flatten_relion5_tomo

    flat_star = flatten_relion5_tomo(particles_star, tomograms_star, flat_star)
    rows, optics = read_star(str(flat_star))
    merged = merged_stack_image_names(star_column(rows, "rlnImageName", required=True))
    if merged is not None:
        rows["_rlnImageName"] = merged
        starfile.write_star(str(flat_star), rows, data_optics=optics)
    images = load_dataset(str(flat_star), datadir=datadir, lazy=lazy, dtype=np.complex64, absent_angles_zero=True)
    return TomoDataset(images, rows, particles_star, tomograms_star)
