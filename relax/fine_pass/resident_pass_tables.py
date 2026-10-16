"""Host tables both resident pass-2 drivers derive from the same inputs.

The global driver (:mod:`relax.fine_pass.resident_pass2`) and the local-search driver
(:mod:`relax.local_search.resident_pass2`) build these the same way; each derivation has its owner here.
"""

from __future__ import annotations

import numpy as np


def optics_group_rows(noise_variance_half, optics_group_ids, *, n_images: int) -> tuple[int, np.ndarray | None]:
    """``(n_optics_groups, optics_groups_np)``: the noise table's group count and each image's row of it.

    A ``[G, P]`` noise table (G > 1 optics groups) needs ``optics_group_ids``, one row in ``[0, G)`` per
    image; a single noise spectrum has no rows (None).
    """

    n_optics_groups = 1 if noise_variance_half.ndim == 1 else int(noise_variance_half.shape[0])
    optics_groups_np = None
    if n_optics_groups > 1:
        if optics_group_ids is None:
            raise ValueError("a per-optics-group noise table needs optics_group_ids")
        optics_groups_np = np.asarray(optics_group_ids, dtype=np.int32).reshape(-1)
        if optics_groups_np.shape != (n_images,) or np.any(optics_groups_np < 0) or np.any(
            optics_groups_np >= n_optics_groups
        ):
            raise ValueError(
                f"optics_group_ids must give each of {n_images} images a row of the "
                f"{n_optics_groups}-group noise table"
            )
    return n_optics_groups, optics_groups_np
