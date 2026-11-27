"""Prior estimation and map reconstruction for the final all-data expectation.

The controller applies resolution and class-weight updates between these steps.
See ``docs/math/relion_refinement_algorithm.md`` for the reconstruction order.
"""

import logging
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from relax.reconstruction import conical_regularization, regularization
from relax.refinement import numbered_reconstruction, priors
from relax.refinement.numbered_reconstruction import ReconstructionSettings
from relax.runtime.timing import Stopwatch

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class HalfmapPrior:
    variance: object
    fsc: object
    details: dict
    # The conical-FSC prior of the joined halves (relax.reconstruction.conical_regularization.ConePrior); None: RELION's.
    cone_prior: object = None


@dataclass(frozen=True)
class ClassPriors:
    variance: object
    shells: object
    data_vs_prior: np.ndarray
    details: dict


@dataclass(frozen=True)
class FinalMaps:
    merged: object
    halves: list


def reconstruct_unfiltered_halfmaps(
    numerators,
    denominators,
    *,
    settings: ReconstructionSettings,
    current_size: int,
    accumulator_shape: tuple,
) -> list:
    """Reconstruct host maps before low-frequency joining modifies the halves."""
    return [
        np.asarray(
            settings.reconstruct(
                half_ctf, half_y, tau=None, current_size=current_size, accumulator_volume_shape=accumulator_shape
            ).reshape(-1)
        )
        for half_ctf, half_y in zip(denominators, numerators, strict=True)
    ]


def compute_final_halfmap_prior(
    numerators,
    denominators,
    *,
    settings: ReconstructionSettings,
    current_size: int,
    accumulator_shape: tuple,
    full_half_axis: int,
    scoring_dtype,
) -> HalfmapPrior:
    """Estimate the whole-data prior from the joined halves' backprojector FSC (and its conical-FSC form with
    ``settings.conical_prior``)."""
    fsc = regularization.compute_relion_fsc_from_backprojector(
        numerators[0],
        numerators[1],
        denominators[0],
        denominators[1],
        settings.volume_shape,
        padding_factor=settings.padding_factor,
        r_max=current_size // 2,
        accumulator_volume_shape=accumulator_shape,
        output_dtype=scoring_dtype,
        full_is_hermitian=True,
        shell_pair_counting=settings.shell_pair_counting,
        packed_stream_min_elements=settings.programs.fsc_packed_stream_min_elements,
        fsc_dump=settings.programs.fsc_dump,
    )
    if settings.solvent_correct_fsc:
        # RELION corrects the joined iteration's FSC too (ml_optimiser_mpi.cpp:4028, iter -1); its
        # final resolution is then "already with masking".
        fsc = priors.solvent_corrected_fsc(
            numerators,
            denominators,
            settings,
            current_size=current_size,
            accumulator_shape=accumulator_shape,
            label="final iteration",
            like=fsc,
            log=logger,
        )
    variance, _, details = regularization.compute_relion_tau2_from_weights(
        denominators[0],
        denominators[1],
        fsc,
        settings.volume_shape,
        tau2_fudge=settings.tau2_fudge,
        padding_factor=settings.padding_factor,
        r_max=current_size // 2,
        is_whole_instead_of_half=True,
        return_details=True,
        full_half_axis=full_half_axis,
        accumulator_volume_shape=accumulator_shape,
        weight_combination="sum",
        output_dtype=scoring_dtype,
        shell_pair_counting=settings.shell_pair_counting,
    )
    cone_prior = None
    if settings.conical_prior is not None:
        clock = Stopwatch()
        # RECOVAR adds the shell prior in the dtype of the weight and the prior volume; the term keeps it (float32 on
        # RECOVAR's large-grid single-precision path: estimate_cone_prior).
        solve_dtype = np.result_type(np.finfo(np.dtype(denominators[0].dtype)).dtype, np.dtype(variance.dtype))
        cone_prior = conical_regularization.estimate_cone_prior(
            numerators,
            denominators,
            [details["prior_shells"]],
            settings=settings.conical_prior,
            volume_shape=settings.volume_shape,
            accumulator_shape=accumulator_shape,
            padding_factor=settings.padding_factor,
            current_size=current_size,
            tau2_fudge=settings.tau2_fudge,
            minres_map=settings.minres_map,
            whole=True,
            solve_dtype=solve_dtype,
        )
        logger.info("Computed final-iteration conical-FSC prior: %.1fs", clock.seconds)
        cone_prior.log_summary("final iteration")
    return HalfmapPrior(variance=variance, fsc=fsc, details=details, cone_prior=cone_prior)


def compute_final_class_priors(
    denominator,
    references,
    *,
    projector_power_spectrum,
    n_classes: int,
    settings: ReconstructionSettings,
    current_size: int,
    accumulator_shape: tuple,
    full_half_axis: int,
    stable_window_quantum: int,
) -> ClassPriors:
    """Estimate each class prior from its reference and merged backprojection (``stable_window_quantum``:
    ``EngineTuning.stable_window_quantum``)."""
    frame_scale = float(settings.box_size) ** 4
    variances = []
    shells = []
    data_vs_prior = []
    details = []
    for class_idx in range(n_classes):
        prior = priors.estimate_class_prior(
            references,
            denominator,
            class_index=class_idx,
            settings=settings,
            current_size=current_size,
            accumulator_shape=accumulator_shape,
            full_half_axis=full_half_axis,
            frame_scale=frame_scale,
            projector_power_spectrum=projector_power_spectrum,
            stable_window_quantum=stable_window_quantum,
        )
        variances.append(prior.variance)
        shells.append(prior.shells)
        data_vs_prior.append(prior.data_vs_prior)
        details.append(prior.details)
    return ClassPriors(
        variance=jnp.stack(variances, axis=0),
        shells=jnp.stack(shells, axis=0),
        data_vs_prior=np.stack(
            [np.asarray(value, dtype=np.float32) for value in data_vs_prior], axis=0,
        ),
        details=priors.stack_class_tau2_update_details(details),
    )


def reconstruct_final_class_maps(
    numerator,
    denominator,
    prior_shells,
    *,
    class_weights,
    n_classes: int,
    settings: ReconstructionSettings,
    current_size: int,
    accumulator_shape: tuple,
) -> FinalMaps:
    """Reconstruct class maps and their posterior-weighted merged map."""
    class_means = jnp.stack(
        [
            settings.reconstruct(
                denominator[class_idx],
                numerator[class_idx],
                tau=prior_shells[class_idx],
                current_size=current_size,
                accumulator_volume_shape=accumulator_shape,
                tau_is_1d=True,
            ).reshape(-1)
            for class_idx in range(n_classes)
        ],
        axis=0,
    )
    return FinalMaps(
        merged=numbered_reconstruction.weighted_class_merge(class_means, class_weights),
        halves=[class_means, class_means],
    )


def _add_into_first(first, second):
    """``first + second``; written into ``first`` when both are host arrays of the sum's dtype, so the sum of two
    box-scale accumulators allocates nothing."""

    if (
        isinstance(first, np.ndarray)
        and isinstance(second, np.ndarray)
        and first.flags.writeable
        and np.result_type(first, second) == first.dtype
    ):
        np.add(first, second, out=first)
        return first
    return first + second


def reconstruct_final_halfmaps(
    half_backprojections: list,
    prior,
    *,
    settings: ReconstructionSettings,
    current_size: int,
    accumulator_shape: tuple,
    cone_prior=None,
) -> FinalMaps:
    """Each half's map from its own backprojection, then the merged map from their sum, keeping maps on the host.

    With ``cone_prior`` (the joined halves' conical-FSC prior) every weight carries its prior term and the solves add
    no shell prior.

    ``half_backprojections`` holds the two halves' (denominator, numerator) pairs and is emptied here: this
    function is their last reader. The halves are solved first, so the merged accumulators can then be formed
    in the first half's arrays instead of beside both pairs (a third pair is 24.7 GB of host memory at
    EMPIAR-10202's box 800, and sat in the final pass's peak; relax#39). A solve is a function of its own
    accumulators and the prior (``settings.reconstruct``, which changes neither), so the three maps do not
    depend on the order they are solved in, and ``a + b`` written into ``a`` is the same sum.
    """

    def solve(denominator, numerator):
        if cone_prior is not None:
            denominator = cone_prior.regularized_weight(0, denominator)
        return np.asarray(
            settings.reconstruct(
                denominator,
                numerator,
                tau=prior if cone_prior is None else None,
                current_size=current_size,
                accumulator_volume_shape=accumulator_shape,
            ).reshape(-1)
        )

    halves = [solve(denominator, numerator) for denominator, numerator in half_backprojections]
    (denominator, numerator), (other_denominator, other_numerator) = half_backprojections
    half_backprojections.clear()
    denominator = _add_into_first(denominator, other_denominator)
    numerator = _add_into_first(numerator, other_numerator)
    del other_denominator, other_numerator
    merged = solve(denominator, numerator)
    del denominator, numerator
    return FinalMaps(merged=merged, halves=halves)
