"""The numpy Projector::project reference (tests/helpers) against RELION's own, through relion_bind."""

import numpy as np
from helpers.float_compare import assert_matches
from helpers.relion_projector_reference import make_projector, project
from scipy.spatial.transform import Rotation

from relax.relion_bind import _relion_bind_core as bind


def test_reference_projection_is_relions_for_rotations_and_a_scaled_matrix():
    rng = np.random.default_rng(7)
    n, current_size = 24, 18
    volume = rng.standard_normal((n, n, n))
    rotations = Rotation.random(3, random_state=11).as_matrix()
    # A matrix scaled like applyScaleDifference, as for images on another grid.
    rotations = np.concatenate([rotations, rotations[:1] / 1.12])
    projector = make_projector(bind, volume, n, 2, current_size)
    ours = project(projector, rotations, n)
    rows = np.arange(n)
    y = np.where(rows <= n // 2, rows, rows - n)
    radius = np.hypot(*np.meshgrid(y, np.arange(n // 2 + 1), indexing="ij"))
    for rotation, image in zip(rotations, ours):
        relion = bind.project_volume(
            volume, rotation, ori_size=n, padding_factor=2, current_size=current_size, data_dim=2
        )
        # A pixel whose rotated radius equals r_max exactly is a float tie: RELION's own matrix inverse
        # and numpy's round differently in the last bit, so either side may keep it.
        scale = float(np.linalg.norm(rotation[0]))
        tie = np.isclose(radius / scale, current_size // 2)
        # Both sides run the same double-precision arithmetic in a different order.
        assert_matches(image[~tie], relion[~tie], rtol=1e-12)
