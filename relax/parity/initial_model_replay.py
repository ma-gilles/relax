"""Iteration-zero model replay: file identity, noise, priors and optimiser controls.

The command selects replay and installs each result at its existing boundary.
See ``docs/math/relion_refinement_algorithm.md#iteration-zero-model-replay``.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from recovar import utils

from relax.refinement.half_inputs import HalfPair
from relax.relion import metadata
from relax.relion.initial_noise import (
    read_relion_single_optics_sigma2_noise,
    relion_mpi_process_start_scoring_noise_pair,
)


class ModelStar(NamedTuple):
    path: Path
    tables: dict


@dataclass(frozen=True)
class InitialModelReplay:
    models: list[ModelStar]
    source: str

    @property
    def reference(self) -> ModelStar:
        return self.models[0]


class NoiseReplay(NamedTuple):
    """Each half's scoring variance and the RELION-frame spectra used to report its source."""

    variance: HalfPair
    sigma2_per_model: list[np.ndarray]
    frame_scale: int


class InitialModelControls(NamedTuple):
    tau2_fudge: float | None
    sigma_offset_angstrom: float | None


def read_initial_model(directory, *, n_classes: int) -> InitialModelReplay:
    """Prefer the shared model; only K1 can replay a half-specific model pair."""
    import starfile

    directory = Path(directory)
    shared_path = directory / "run_it000_model.star"
    if shared_path.exists():
        return InitialModelReplay(
            models=[ModelStar(shared_path, starfile.read(str(shared_path)))],
            source="shared",
        )

    half_paths = [directory / "run_it000_half1_model.star", directory / "run_it000_half2_model.star"]
    if n_classes == 1 and all(path.exists() for path in half_paths):
        return InitialModelReplay(
            models=[ModelStar(path, starfile.read(str(path))) for path in half_paths],
            source="half-specific",
        )

    expected = [shared_path, *half_paths]
    missing = [str(path) for path in expected if not path.exists()]
    raise SystemExit(
        "--relion_init_dir given but no compatible iter-0 model STAR was found; "
        f"missing candidates: {', '.join(missing)}",
    )


def prepare_noise(
    model,
    *,
    box_size,
    image_shape,
    explicit_noise_radial,
    live_sigma2,
    log,
) -> NoiseReplay:
    """Resolve NPZ/live/STAR precedence and the MPI follower-1 broadcast.

    Explicit radial noise is in RECOVAR's image frame; live sigma2 is in
    RELION's frame. Missing model spectra remain an error even with an override.
    """
    from recovar.reconstruction import noise as recon_noise

    frame_scale = box_size**4
    sigma2_per_model = [
        read_relion_single_optics_sigma2_noise(
            source.tables,
            context=f"RELION iteration-0 model {index + 1}",
        )
        for index, source in enumerate(model.models)
    ]
    if any(sigma2 is None for sigma2 in sigma2_per_model):
        raise ValueError("RELION iteration-0 model is missing rlnSigma2Noise")
    if explicit_noise_radial is not None:
        explicit_sigma2 = np.asarray(explicit_noise_radial, dtype=np.float64) / float(frame_scale)
        sigma2_per_model = [explicit_sigma2.copy() for sigma2 in sigma2_per_model]
        log.info(
            "STRICT-PARITY: explicit --init-noise-from-npz overrides rounded "
            "RELION iteration-0 rlnSigma2Noise values",
        )
    elif live_sigma2 is not None:
        sigma2_per_model = [np.asarray(live_sigma2, dtype=np.float64).copy() for sigma2 in sigma2_per_model]
    if model.source == "half-specific":
        sigma2_per_model = relion_mpi_process_start_scoring_noise_pair(
            sigma2_per_model[0], sigma2_per_model[1], split_random_halves=True,
        )
        log.info(
            "STRICT-PARITY: emulating RELION MPI rank-1 sigma2_noise broadcast "
            "for both AutoRefine half-sets",
        )
    if len(sigma2_per_model) == 1:
        sigma2 = sigma2_per_model[0]
        noise_radial = jnp.asarray(sigma2 * frame_scale)
        variance = HalfPair.shared(recon_noise.make_radial_noise(noise_radial, image_shape))
    else:
        variance = HalfPair(*(
            recon_noise.make_radial_noise(jnp.asarray(sigma2 * frame_scale), image_shape)
            for sigma2 in sigma2_per_model
        ))
    return NoiseReplay(variance, sigma2_per_model, frame_scale)


def log_noise_source(noise, *, log):
    """Report the installed noise without retaining its scoring arrays."""
    if len(noise.sigma2_per_model) == 1:
        sigma2 = noise.sigma2_per_model[0]
        log.info(
            "STRICT-PARITY: replaced bootstrapped sigma2_noise with RELION it000 "
            "shared spectrum (× N^4=%.3e). RELION shape=%s, head=%s",
            float(noise.frame_scale),
            sigma2.shape,
            np.asarray(sigma2[:5]),
        )
    else:
        log.info(
            "STRICT-PARITY: replaced bootstrapped sigma2_noise with RELION it000 "
            "per-half spectra (× N^4=%.3e). half1 shape=%s head=%s half2 shape=%s head=%s",
            float(noise.frame_scale),
            noise.sigma2_per_model[0].shape,
            np.asarray(noise.sigma2_per_model[0][:5]),
            noise.sigma2_per_model[1].shape,
            np.asarray(noise.sigma2_per_model[1][:5]),
        )


def prepare_prior(model, *, n_classes: int, box_size, volume_shape):
    """Expand the reference model's class tau2 shells in RECOVAR's image frame."""
    frame_scale = box_size**4
    tables = model.reference.tables
    if n_classes > 1:
        per_class_tau2 = []
        for class_index in range(n_classes):
            table = tables[f"model_class_{class_index + 1}"]
            column = "rlnReferenceTau2" if "rlnReferenceTau2" in table.columns else "rlnReferenceSigma2"
            per_class_tau2.append(np.asarray(table[column], dtype=np.float64) * frame_scale)
        return jnp.stack(
            [
                jnp.asarray(utils.make_radial_image(shells, volume_shape, extend_last_frequency=True)).reshape(-1)
                for shells in per_class_tau2
            ],
            axis=0,
        )
    table = tables["model_class_1"]
    column = "rlnReferenceTau2" if "rlnReferenceTau2" in table.columns else "rlnReferenceSigma2"
    tau2 = np.asarray(table[column], dtype=np.float64) * frame_scale
    return jnp.asarray(utils.make_radial_image(tau2, volume_shape, extend_last_frequency=True)).reshape(-1)


def read_controls(model, *, log) -> InitialModelControls:
    """Prefer model tau2 fudge; read offset sigma only from the optimiser."""
    sigma_offset_angstrom = None
    model_text = model.reference.path.read_text()
    tau2_fudge = metadata._parse_relion_tau2_fudge(model_text)
    if tau2_fudge is not None:
        log.info("STRICT-PARITY: tau2_fudge from RELION it000 model.star: %.3f", tau2_fudge)
    optimiser_path = model.reference.path.parent / "run_it000_optimiser.star"
    if optimiser_path.exists():
        optimiser_text = optimiser_path.read_text()
        if tau2_fudge is None:
            tau2_fudge = metadata._parse_relion_tau2_fudge(optimiser_text)
            if tau2_fudge is not None:
                log.info("STRICT-PARITY: --tau2_fudge override from RELION it000 optimiser: %.3f", tau2_fudge)
        match = re.search(r"_rlnSigmaOffsetsAngst\s+(\S+)", optimiser_text)
        if match is not None:
            sigma_offset_angstrom = float(match.group(1))
            log.info("STRICT-PARITY: --offset_sigma_angstrom override from RELION it000: %.3f Å", sigma_offset_angstrom)
    return InitialModelControls(tau2_fudge, sigma_offset_angstrom)
