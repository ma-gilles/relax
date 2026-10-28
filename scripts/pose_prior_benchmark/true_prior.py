"""The true direction prior of a pose-prior benchmark dataset, in closed form.

The probability per HEALPix direction (NEST order) of the viewing directions the simulator draws, from the
simulator's pose-distribution record (``simulate_dataset.py``'s ``pose_distribution.json``). The evaluator scores
learned priors against it, and ``write_fixed_prior.py`` writes it for ``relax refine --fixed-direction-prior``.
"""

from __future__ import annotations

from typing import Any

import healpy as hp
import numpy as np

# Quadrature of the population prior: HEALPix orders below each prior order that integrate a pixel, and the
# tilt and azimuth samples of the azimuthal average of the Kent density.
POPULATION_FINE_ORDERS = 3
POPULATION_TILT_SAMPLES = 4096
POPULATION_AZIMUTH_SAMPLES = 360


def kent_population_prior(kappa: float, beta: float, mu: list[float], mu0: list[float], order: int) -> np.ndarray:
    """Probability per HEALPix direction of the viewing directions that RECOVAR's Kent option produces.

    The simulator draws the third column ``u`` of each rotation matrix from the Kent distribution and the
    in-plane angle uniformly. RELION's viewing direction is the third row ``(a_z, b_z, u_z)``: its tilt is the
    angle between ``u`` and z and its azimuth is uniform. The density of a viewing direction at tilt ``t`` is
    therefore the Kent density averaged over the azimuth of ``u`` at polar angle ``t``, integrated here over
    each pixel from the pixels of a finer HEALPix order (equal areas).
    """
    from recovar.simulation.simulator import _kent_frame  # the simulator's own Kent parameterization

    gamma1, gamma2, gamma3 = _kent_frame(mu, mu0)
    tilt = (np.arange(POPULATION_TILT_SAMPLES) + 0.5) * np.pi / POPULATION_TILT_SAMPLES
    azimuth = (np.arange(POPULATION_AZIMUTH_SAMPLES) + 0.5) * 2 * np.pi / POPULATION_AZIMUTH_SAMPLES
    sin_tilt = np.sin(tilt)[:, None]
    points = np.stack(
        [sin_tilt * np.cos(azimuth)[None], sin_tilt * np.sin(azimuth)[None], np.cos(tilt)[:, None] + 0 * azimuth],
        axis=-1,
    )
    log_density = kappa * (points @ gamma1) + beta * ((points @ gamma2) ** 2 - (points @ gamma3) ** 2)
    zonal = np.exp(log_density - log_density.max()).mean(axis=1)
    fine_nside = 1 << (order + POPULATION_FINE_ORDERS)
    fine_tilt, _ = hp.pix2ang(fine_nside, np.arange(hp.nside2npix(fine_nside)), nest=True)
    # NEST children of one pixel are consecutive, so each pixel's children sum by reshaping.
    prior = np.interp(fine_tilt, tilt, zonal).reshape(-1, 4**POPULATION_FINE_ORDERS).sum(axis=1)
    return prior / prior.sum()


def population_prior(poses: dict[str, Any], order: int) -> np.ndarray:
    """Probability per HEALPix direction of the simulated viewing directions, from the pose-distribution record."""
    if poses["poses"] == "uniform":
        n = hp.nside2npix(1 << int(order))
        return np.full(n, 1.0 / n)
    kappa, beta, mu, mu0 = poses["dataset_params_option"]
    return kent_population_prior(kappa, beta, mu, mu0, order)
