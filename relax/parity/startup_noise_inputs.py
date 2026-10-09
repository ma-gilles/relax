"""Start-up noise taken from elsewhere instead of estimated by the run (code rule 15).

A frozen boundary's noise (``--frozen-boundary-dir``), an earlier run's archived noise (``--init_noise_from_npz``,
diagnostic only) and RELION's live K=1 estimate (``RELAX_K1_RELION_LIVE_INITIAL_NOISE``, a STRICT-PARITY
diagnostic). The command chooses one; the run's own start-up noise is ``relax.refinement.startup_noise``'s.
"""

from __future__ import annotations

import numpy as np

from relax.refinement.half_inputs import HalfPair
from relax.refinement.startup_noise import StartupNoise, estimate_startup_sigma2, scoring_noise_from_sigma2


def frozen_boundary_noise(frozen_boundary, image_shape) -> StartupNoise:
    """A frozen boundary's noise: each half's scoring pixel variance, their mean radial curve."""
    from relax.parity import frozen_boundary_cli

    return StartupNoise(
        radial=np.mean(np.stack(frozen_boundary.noise_radial_per_half, axis=0), axis=0),
        pixel_variance=HalfPair(
            *frozen_boundary_cli.expand_boundary_noise(frozen_boundary.noise_radial_per_half, image_shape)
        ),
    )


def archived_noise(path, iteration, image_shape, *, log) -> StartupNoise:
    """The radial noise an earlier run's archive recorded at ``iteration`` (``--init_noise_from_npz``)."""
    from recovar.reconstruction import noise as recon_noise

    from relax.helpers import iteration_history

    init_noise = iteration_history._load_init_noise_radial_npz(path, iteration)
    radial = init_noise["noise_radial"]
    log.info(
        "Diagnostic init: loaded sigma2_noise from %s iter=%s: min=%.3e median=%.3e max=%.3e",
        path,
        init_noise["iteration"],
        float(np.min(np.asarray(radial))),
        float(np.median(np.asarray(radial))),
        float(np.max(np.asarray(radial))),
    )
    return StartupNoise(
        radial=radial, pixel_variance=HalfPair.shared(recon_noise.make_radial_noise(radial, image_shape)),
    )


def live_initial_noise(dataset, half_sets, *, mask_params, log):
    """The fresh K=1 noise RELION estimates live from the images (one optics group): its sigma2 and the
    float64 scoring variance, the pass's exact BPref operands. ``half_sets`` lists the particles it reads."""
    sigma2_per_group = estimate_startup_sigma2(
        dataset,
        source_rows=half_sets.noise_source_rows,
        optics_group_ids=half_sets.noise_optics_group_ids,
        image_pixel_size=float(half_sets.optics_pixel_sizes[0]),
        particle_diameter_ang=float(mask_params[0]),
        width_mask_edge_px=int(mask_params[1]),
    )
    if sigma2_per_group.shape[0] != 1:
        raise NotImplementedError(
            "fresh K=1 live-noise scoring currently requires one optics group",
        )
    sigma2 = sigma2_per_group[0]
    variance = scoring_noise_from_sigma2(sigma2, box_size=int(dataset.grid_size), output_dtype=np.float64)
    log.warning(
        "STRICT-PARITY: fresh K=1 RELION live initial noise enabled: particles=%d "
        "source_rows_head=%s sigma2_head=%s",
        min(1000, int(np.asarray(half_sets.noise_source_rows).size)),
        np.asarray(half_sets.noise_source_rows, dtype=np.int64)[:5].tolist(),
        np.asarray(sigma2[:5]),
    )
    return sigma2, variance
