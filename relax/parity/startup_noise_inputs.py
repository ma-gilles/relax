"""Start-up noise taken from elsewhere instead of estimated by the run (code rule 15).

A frozen boundary's noise (``--frozen-boundary-dir``), an earlier run's archived noise (``--init_noise_from_npz``,
diagnostic only) and RELION's live K=1 estimate (``RELAX_K1_RELION_LIVE_INITIAL_NOISE``, a STRICT-PARITY
diagnostic). The command chooses one; the run's own start-up noise is ``relax.refinement.startup_noise``'s.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from relax.refinement.startup_noise import StartupNoise, estimate_startup_sigma2, scoring_noise_from_sigma2
from relax.steps.half_inputs import HalfPair


def _load_init_noise_radial_npz(path, noise_iter="last"):
    """Load a diagnostic initial noise spectrum from refinement_results.npz."""

    noise_path = Path(path)
    with np.load(noise_path, allow_pickle=False) as npz:
        if str(noise_iter).lower() in {"last", "latest"}:
            pattern = re.compile(r"^noise_radial_iter_(\d{3})$")
            available = sorted(
                int(match.group(1)) for key in npz.files if (match := pattern.match(key)) is not None
            )
            if not available:
                raise ValueError(f"No numbered noise_radial_iter arrays found in {noise_path}")
            iter_label = f"{available[-1]:03d}"
        else:
            iter_label = f"{int(noise_iter):03d}"
        key = f"noise_radial_iter_{iter_label}"
        if key not in npz.files:
            raise ValueError(f"Missing {key} in {noise_path}")
        noise_radial = np.asarray(npz[key], dtype=np.float64)

    if noise_radial.ndim != 1:
        raise ValueError(f"{key} must be a 1D radial spectrum, got shape {noise_radial.shape}")
    if not np.all(np.isfinite(noise_radial)):
        raise ValueError(f"{key} contains non-finite values")
    if np.any(noise_radial <= 0.0):
        raise ValueError(f"{key} must be strictly positive")
    return {
        "iteration": iter_label,
        "noise_radial": noise_radial,
    }


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


    init_noise = _load_init_noise_radial_npz(path, iteration)
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
