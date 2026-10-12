"""Pose grids for an E-step at each particle's stored pose (RELION ``--skip_align``).

RELION skips the pose search by giving its sampling object one orientation and one
translation per particle of the pool (``ml_optimiser.cpp:4660-4720``,
``sampling.addOneOrientation`` / ``addOneTranslation``): particle ``i`` evaluates
sample ``i`` only, for every class, with no orientational prior, no oversampling
and no perturbation (``:2619-2660``). The orientation is the particle's stored
Euler angles; the translation is what is left of its stored offset after the
rounded offset has been applied to the image at reading
(``old_offset - ROUND(old_offset)``; zero for 2D-stack subtomograms, whose
offsets are not rounded).

relax's pass 2 takes a coarse HEALPix grid, a fine grid with a parent map, and
each image's significant coarse samples. :func:`given_pose_grids` presents the
particles' own poses in that form without touching the grids' meaning: the coarse
and the fine rotation grid are the same list, one rotation per particle, padded
with identities to the size of the smallest HEALPix grid that holds it; the
translation grid is the one zero translation, and each particle's own translation
goes to pass 2 as an image translation (``image_translations`` of the resident
pass), which translates the particle's images as they are prepared. Image ``i``'s
only sample is (rotation ``i``, the zero translation), sample id ``i``.

Subtomograms (docs/development/relion_feature_gaps.md, row 16) are not covered yet; the resident
pass refuses ``image_translations`` with tilt images. RELION gives a 2D-stack subtomogram the zero
translation sample without rounding (ml_optimiser.cpp:4688-4700) and translates each tilt image by
the particle's 3D offset projected into that tilt (``getTranslationInTiltSeries``,
acc_ml_optimiser_impl.h:1524-1546). The same keyword is per image, so it extends as one phase per
tilt image: ``image_translations`` holds each tilt-image row's projected 2D offset
(``tomo_particles.tilt_translation_angles`` already forms those projections), the tilt pass scores
its one zero translation, and the posterior unit stays the particle.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from relax.sampling import rotation_grid_size

# HEALPix orders a list of given poses may be padded to (order 6 holds 18.9 million rotations in C1).
_MAX_ORDER = 6


class GivenPoseGrids(NamedTuple):
    """Pass-2 grids and supports at the particles' own poses; the coarse and the fine grid are the same."""

    healpix_order: int
    rotations: np.ndarray  # [G, 3, 3]: the particles' rotations, then identities up to the grid size
    translations: np.ndarray  # [1, 2]: the one zero translation
    image_translations: np.ndarray  # [N, 2]: each particle's translation sample, in pixels
    rotation_parent_map: np.ndarray  # [G], the identity map
    translation_parent_map: np.ndarray  # [1]
    supports: list  # [N] arrays holding image i's one coarse sample, i


def given_pose_grids(rotations, translations, *, symmetry: str = "C1") -> GivenPoseGrids:
    """Pass-2 grids whose only sample for image ``i`` is its own ``rotations[i]`` and ``translations[i]``.

    ``rotations`` is ``[N, 3, 3]`` and ``translations`` ``[N, 2]`` (pixels, relative to the image
    pre-shift). A sample's id is ``rotation * n_translations + translation``, as pass 1 writes it.
    """

    rotations = np.asarray(rotations)
    translations = np.asarray(translations)
    n = int(rotations.shape[0])
    if rotations.shape != (n, 3, 3) or translations.shape != (n, 2) or n == 0:
        raise ValueError(
            f"given poses are [N, 3, 3] rotations and [N, 2] translations, got {rotations.shape}, {translations.shape}"
        )
    order = next((o for o in range(_MAX_ORDER + 1) if rotation_grid_size(o, symmetry) >= n), None)
    if order is None:
        raise ValueError(f"{n} given poses exceed the HEALPix order-{_MAX_ORDER} grid")
    grid_size = int(rotation_grid_size(order, symmetry))
    grid = np.broadcast_to(np.eye(3, dtype=rotations.dtype), (grid_size, 3, 3)).copy()
    grid[:n] = rotations
    return GivenPoseGrids(
        healpix_order=int(order),
        rotations=grid,
        translations=np.zeros((1, 2), dtype=translations.dtype),
        image_translations=translations,
        rotation_parent_map=np.arange(grid_size, dtype=np.int64),
        translation_parent_map=np.zeros(1, dtype=np.int64),
        supports=[np.asarray([i], dtype=np.int32) for i in range(n)],
    )
