"""One body's maximisation: its two half maps from its own accumulators, prior and masked FSC.

RELION reconstructs each body like a K=1 split-half auto-refine reference (ml_optimiser_mpi.cpp at f2c1a38): the
two half accumulators are joined below ``--low_resol_join_halves`` (``joinTwoHalvesAtLowResolution``,
:3112-3220), the half-set FSC is the solvent-corrected FSC of the unregularised half maps inside the body mask
(``reconstructUnregularisedMapAndCalculateSolventCorrectedFSC``, :3221-3403; multi-body always corrects), each
half's prior follows from that FSC and its own weights (``updateSSNRarrays``) and each half is solved
(``reconstruct``). The reconstruction is in the body frame, centred on the body's centre of mass; RELION moves it back
to the stored frame (``selfTranslate(Iref, com, DONT_WRAP)``, :2490-2493) without masking it and without solvent
flattening (ml_optimiser.cpp:5475-5477).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from recovar.core import fourier_transform_utils as ftu
from recovar.utils.helpers import recovar_volume_to_relion

from relax.multibody.bodies import Body
from relax.multibody.subtraction import stored_body_map
from relax.reconstruction import regularization
from relax.reconstruction.solvent_mask import solvent_corrected_fsc
from relax.refinement import numbered_reconstruction
from relax.refinement.priors import join_half_accumulators_at_low_resolution, split_half_prior_from_fsc


def _real(map_ft, volume_shape) -> np.ndarray:
    return np.asarray(ftu.get_idft3(jnp.asarray(map_ft).reshape(volume_shape)).real, dtype=np.float64)


def _fourier(real) -> np.ndarray:
    return np.asarray(ftu.get_dft3(jnp.asarray(real)).reshape(-1))


def _joined(numerators, denominators, settings, *, accumulator_shape, low_resolution_join_angstrom, current_resolution_angstrom):
    """The half accumulators joined below the low-resolution limit (RELION's ``--low_resol_join_halves``)."""

    numerators, denominators = list(numerators), list(denominators)
    if low_resolution_join_angstrom is not None and low_resolution_join_angstrom > 0:
        numerators[0], numerators[1], denominators[0], denominators[1] = join_half_accumulators_at_low_resolution(
            numerators, denominators,
            accumulator_volume_shape=accumulator_shape,
            box_size=settings.box_size,
            voxel_size=settings.voxel_size,
            padding_factor=settings.padding_factor,
            low_resolution_angstrom=low_resolution_join_angstrom,
            pixel_resolutions=[],
            current_resolution=current_resolution_angstrom,
            preserve_inputs=False,
            route=settings.programs.lowres_join,
        )
    return numerators, denominators


def body_solvent_corrected_fsc(body: Body, numerators, denominators, settings, *, current_size, accumulator_shape, phase_stream):
    """The solvent-corrected FSC of the body's unregularised half maps in its mask, in the stored frame; also returns
    those maps (RELION's ``_unfil.mrc``, recovar frame) and the correction's details."""

    unregularised = numbered_reconstruction.reconstruct_unregularized_k1_halfmaps(
        numerators, denominators, settings, accumulator_volume_shape=accumulator_shape, current_size=current_size
    )
    unfiltered = [stored_body_map(_real(m, settings.volume_shape), body) for m in unregularised]
    # RELION's FSC reads the maps as its files hold them (relax.io.map_io: transposed and negated), where the body
    # mask is as read. The sign matters: the masked FSC of the phase-randomised maps mixes the kept low shells
    # with the randomised ones.
    fsc, details = solvent_corrected_fsc(
        *(recovar_volume_to_relion(m) for m in unfiltered),
        body.mask,
        current_size=current_size,
        stream=phase_stream,
    )
    return fsc, unfiltered, details


@dataclass(frozen=True)
class BodyMaximization:
    """One body's new state: stored-frame half maps (recovar Fourier, flat), the FSC and each half's prior."""

    maps: list
    unfiltered_maps: list  # stored-frame real maps (recovar frame), RELION's _unfil.mrc
    fsc: np.ndarray
    prior: object  # relax.refinement.priors.SplitHalfPrior
    fsc_details: dict
    fourier_coverage: np.ndarray | None = None  # half 1, per shell (rlnFourierCompleteness)


def maximize_body(
    body: Body,
    numerators,
    denominators,
    settings,
    *,
    current_size: int,
    accumulator_shape,
    full_half_axes,
    low_resolution_join_angstrom: float | None,
    current_resolution_angstrom: float,
    phase_stream,
    this_iteration,
    scoring_dtype,
) -> BodyMaximization:
    """RELION's maximisation of one body from its two half accumulators (body frame).

    ``settings`` is the run's ``ReconstructionSettings`` without a solvent flatten; ``phase_stream`` the run's
    RELION leader ``rand()`` stream, which the FSC's phase randomisation advances.
    """

    numerators, denominators = _joined(
        numerators, denominators, settings, accumulator_shape=accumulator_shape,
        low_resolution_join_angstrom=low_resolution_join_angstrom, current_resolution_angstrom=current_resolution_angstrom,
    )
    fsc, unfiltered, details = body_solvent_corrected_fsc(
        body, numerators, denominators, settings, current_size=current_size, accumulator_shape=accumulator_shape,
        phase_stream=phase_stream,
    )
    fsc = jnp.asarray(fsc, dtype=scoring_dtype)
    prior = split_half_prior_from_fsc(
        denominators, fsc, settings,
        current_size=current_size, accumulator_shape=accumulator_shape, full_half_axes=full_half_axes,
        scoring_dtype=scoring_dtype,
    )
    body_frame_maps = numbered_reconstruction.reconstruct_numbered_k1_halfmaps(
        numerators, denominators, prior.shells_per_half, settings,
        this_iteration=this_iteration, accumulator_volume_shape=accumulator_shape,
    )
    maps = [_fourier(stored_body_map(_real(m, settings.volume_shape), body)) for m in body_frame_maps]
    coverage = fourier_coverage(
        denominators[0], prior.shells_per_half[0], settings,
        current_size=current_size, accumulator_shape=accumulator_shape, full_half_axis=full_half_axes[0],
    )
    return BodyMaximization(
        maps=maps, unfiltered_maps=unfiltered, fsc=np.asarray(fsc), prior=prior, fsc_details=details,
        fourier_coverage=coverage,
    )


@dataclass(frozen=True)
class BodyFinalMaps:
    """A body's last, halves-joined iteration: the joined map, the unfiltered half maps written before the
    low-resolution join, and the FSC and prior of the joined reconstruction (stored frame)."""

    map: np.ndarray  # recovar Fourier, flat
    unfiltered_maps: list
    fsc: np.ndarray
    details: dict
    fourier_coverage: np.ndarray | None = None  # the joined weights, per shell


def fourier_coverage(weight, tau2_shells, settings, *, current_size: int, accumulator_shape, full_half_axis) -> np.ndarray:
    """Per shell, the fraction of a reconstruction's Fourier voxels whose data-to-prior ratio
    ``weight * pad^3 * tau2_fudge * tau2`` is at least 1 (``BackProjector::updateSSNRarrays``,
    backprojector.cpp:1263-1331, with the FSC-based tau2 it has just set). The voxels and their shells are those
    :func:`relax.reconstruction.regularization.compute_relion_weight_shell_stats` averages the weight over."""

    pad = int(settings.padding_factor)
    volume_shape = tuple(int(v) for v in settings.volume_shape)
    grid = volume_shape if accumulator_shape is None else tuple(int(v) for v in accumulator_shape)
    if pad > 1 and accumulator_shape is None:
        grid = tuple(pad * v for v in volume_shape)
    values = np.asarray(weight).real.reshape(-1).astype(np.float64)
    half = values.size != int(np.prod(grid))
    radial = (ftu.get_grid_of_radial_distances_real if half else ftu.get_grid_of_radial_distances)(
        grid, scaled=False, frequency_shift=0, rounded=False
    )
    shells = np.minimum(np.floor(np.asarray(radial, dtype=np.float64).reshape(-1) / pad + 0.5).astype(np.int64),
                        volume_shape[0] // 2)
    tau2 = np.asarray(tau2_shells, dtype=np.float64).reshape(-1)
    tau2 = np.pad(tau2, (0, max(0, volume_shape[0] // 2 + 1 - tau2.size)))[shells]
    covered = ((tau2 > 0) & (values * float(pad) ** 3 * float(settings.tau2_fudge) * tau2 >= 1.0)).astype(np.float64)
    stats = regularization.compute_relion_weight_shell_stats(
        covered, volume_shape, padding_factor=pad, r_max=current_size // 2,
        full_half_axis=-1 if full_half_axis is None else int(full_half_axis), accumulator_volume_shape=accumulator_shape,
        shell_pair_counting=settings.shell_pair_counting,
    )
    return np.asarray(stats["avg_weight_shells"], dtype=np.float64)


def maximize_body_final(
    body: Body,
    numerators,
    denominators,
    settings,
    *,
    current_size: int,
    accumulator_shape,
    full_half_axis: int,
    low_resolution_join_angstrom: float | None,
    current_resolution_angstrom: float,
    phase_stream,
    scoring_dtype,
) -> BodyFinalMaps:
    """RELION's last iteration of one body (``do_join_random_halves``): the unfiltered half maps from the halves as
    scored, the low-resolution join, the masked solvent-corrected FSC, the whole-data prior
    (``updateSSNRarrays(..., is_whole_instead_of_half)``) and one map from the summed accumulators, as the K=1 final
    pass does (``relax.refinement.finalization``)."""

    unfiltered = [
        stored_body_map(_real(m, settings.volume_shape), body)
        for m in numbered_reconstruction.reconstruct_unregularized_k1_halfmaps(
            numerators, denominators, settings, accumulator_volume_shape=accumulator_shape, current_size=current_size
        )
    ]
    numerators, denominators = _joined(
        numerators, denominators, settings, accumulator_shape=accumulator_shape,
        low_resolution_join_angstrom=low_resolution_join_angstrom, current_resolution_angstrom=current_resolution_angstrom,
    )
    fsc, _, _ = body_solvent_corrected_fsc(
        body, numerators, denominators, settings, current_size=current_size, accumulator_shape=accumulator_shape,
        phase_stream=phase_stream,
    )
    variance, _, details = regularization.compute_relion_tau2_from_weights(
        denominators[0], denominators[1], jnp.asarray(fsc, dtype=scoring_dtype), settings.volume_shape,
        tau2_fudge=settings.tau2_fudge, padding_factor=settings.padding_factor, r_max=current_size // 2,
        is_whole_instead_of_half=True, return_details=True, full_half_axis=full_half_axis,
        accumulator_volume_shape=accumulator_shape, weight_combination="sum", output_dtype=scoring_dtype,
        shell_pair_counting=settings.shell_pair_counting,
    )
    merged = settings.reconstruct(
        denominators[0] + denominators[1], numerators[0] + numerators[1], tau=variance, current_size=current_size,
        accumulator_volume_shape=accumulator_shape,
    ).reshape(-1)
    return BodyFinalMaps(
        map=_fourier(stored_body_map(_real(merged, settings.volume_shape), body)),
        unfiltered_maps=unfiltered,
        fsc=np.asarray(fsc),
        details=details,
        fourier_coverage=fourier_coverage(
            denominators[0] + denominators[1], details["prior_shells"], settings,
            current_size=current_size, accumulator_shape=accumulator_shape, full_half_axis=full_half_axis,
        ),
    )
