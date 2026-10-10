"""Prior estimation and map reconstruction for the final all-data expectation.

The controller applies resolution and class-weight updates between these steps.
See ``docs/math/relion_refinement_algorithm.md`` for the reconstruction order.
"""

import logging
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from relax.reconstruction import regularization_relion
from relax.refinement import mean_helpers
from relax.refinement.mean_helpers import ReconstructionSettings

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class HalfmapPrior:
    variance: object
    fsc: object
    details: dict


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
    """Estimate the whole-data prior from the joined halves' backprojector FSC."""
    fsc = regularization_relion.compute_relion_fsc_from_backprojector(
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
    )
    if settings.solvent_correct_fsc:
        # RELION corrects the joined iteration's FSC too (ml_optimiser_mpi.cpp:4028, iter -1); its
        # final resolution is then "already with masking".
        fsc = mean_helpers.solvent_corrected_fsc(
            numerators,
            denominators,
            settings,
            current_size=current_size,
            accumulator_shape=accumulator_shape,
            label="final iteration",
            like=fsc,
            log=logger,
        )
    variance, _, details = regularization_relion.compute_relion_tau2_from_weights(
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
    return HalfmapPrior(variance=variance, fsc=fsc, details=details)


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
) -> ClassPriors:
    """Estimate each class prior from its reference and merged backprojection."""
    frame_scale = float(settings.box_size) ** 4
    variances = []
    shells = []
    data_vs_prior = []
    details = []
    for class_idx in range(n_classes):
        prior = mean_helpers.estimate_class_prior(
            references,
            denominator,
            class_index=class_idx,
            settings=settings,
            current_size=current_size,
            accumulator_shape=accumulator_shape,
            full_half_axis=full_half_axis,
            frame_scale=frame_scale,
            projector_power_spectrum=projector_power_spectrum,
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
        details=mean_helpers._stack_class_tau2_update_details(details),
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
        merged=mean_helpers.weighted_class_merge(class_means, class_weights),
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
) -> FinalMaps:
    """Each half's map from its own backprojection, then the merged map from their sum, keeping maps on the host.

    ``half_backprojections`` holds the two halves' (denominator, numerator) pairs and is emptied here: this
    function is their last reader. The halves are solved first, so the merged accumulators can then be formed
    in the first half's arrays instead of beside both pairs (a third pair is 24.7 GB of host memory at
    EMPIAR-10202's box 800, and sat in the final pass's peak; relax#39). A solve is a function of its own
    accumulators and the prior (``settings.reconstruct``, which changes neither), so the three maps do not
    depend on the order they are solved in, and ``a + b`` written into ``a`` is the same sum.
    """

    def solve(denominator, numerator):
        return np.asarray(
            settings.reconstruct(
                denominator, numerator, tau=prior, current_size=current_size, accumulator_volume_shape=accumulator_shape
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
