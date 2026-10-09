"""The numbered iteration's M-steps: RELION's K=1 split-half solve and its Class3D solve, their results, and
the one-reference Class3D start's copy of class 1 to every class.

The controller (``iteration_loop.refine_single_volume``) chooses the mode's M-step, records its results in the
history and installs them; each M-step writes the reference model in place, as its docstring says.
"""

import logging
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
import numpy as np

from relax.dense.score_outputs import _combine_optional_half_accumulators
from relax.helpers.orientation_priors import DirectionPrior, learn_class_direction_priors, learn_k1_direction_priors
from relax.helpers.resolution import _firstiter_cc_ini_high_tapered
from relax.helpers.timing import Stopwatch
from relax.refinement.iteration_planning import NumberedIteration
from relax.refinement.mean_helpers import (
    ClassMixture,
    _host_tau2_volumes,
    _snapshot_and_release_previous_k1_means,
    _stack_class_tau2_update_details,
    _updated_mean_variance_per_half,
    class_mixture_from_weights,
    estimate_class_priors,
    estimate_split_half_prior,
    join_half_accumulators_at_low_resolution,
    reconstruct_numbered_class_maps,
    reconstruct_numbered_k1_halfmaps,
    shared_tau2_per_half,
    taper_first_cc_class_prior,
    taper_first_cc_k1_prior,
)
from relax.refinement.ports import ClassTau2
from relax.refinement.refinement_options import RefinementOptions
from relax.relion.geometry import RECONSTRUCTION_PADDING_FACTOR, REFERENCE_FILTER_EDGE_SHELLS
from relax.sampling import rotation_grid_size

if TYPE_CHECKING:
    from relax.refinement.setup_checks import RunContext

# The M-steps log as part of the controller's iteration.
logger = logging.getLogger("relax.refinement.iteration_loop")


def _copy_first_class(stacked):
    """``stacked`` (a leading class axis) with every class set to class 0, as the input's array type."""

    if isinstance(stacked, np.ndarray):
        return np.broadcast_to(stacked[:1], stacked.shape).copy()
    return jnp.broadcast_to(stacked[:1], stacked.shape)


class ClassMaximization(NamedTuple):
    """What a Class3D M-step leaves for the rest of its iteration (the model is written in place)."""

    Ft_y_combined: object
    Ft_ctf_combined: object
    # The references the M-step replaced, on the device; the iteration releases them at its end.
    previous_means: list
    tau2_shells: object
    data_vs_prior: object
    tau2_update_details: dict


def copy_first_class_to_every_class(
    reference_model,
    direction_priors,
    mstep: ClassMaximization,
    class_mixture,
    *,
    n_classes: int,
) -> tuple[ClassMaximization, ClassMixture]:
    """Give every class the first class's model after the CC iteration of a one-reference Class3D start.

    Writes ``reference_model``'s maps and tau2 and the entries of ``direction_priors`` in place; returns
    ``mstep`` with its tau2 shells, data-vs-prior curve and tau2 details copied from the first class, and the
    class mixture with the first class's weight shared equally. The caller records the returned curve as the
    iteration's.
    """
    # After the CC iteration RELION copies class 0's model to every class for the seed iteration:
    # Iref, tau2_class, data_vs_prior_class and pdf_direction, each class taking pdf_class[0] / K
    # (maximizationOtherParameters, ml_optimiser.cpp:6423-6437).
    # One copy serves both halves: a Class3D M-step leaves the two halves one shared class stack.
    first_class_maps = None if reference_model.maps[0] is None else _copy_first_class(reference_model.maps[0])
    reference_model.maps = [first_class_maps, first_class_maps]
    reference_model.tau2 = _copy_first_class(reference_model.tau2)
    reference_model.tau2_per_half = shared_tau2_per_half(reference_model.tau2)
    copied = mstep._replace(
        tau2_shells=_copy_first_class(mstep.tau2_shells),
        data_vs_prior=_copy_first_class(mstep.data_vs_prior),
        tau2_update_details={
            key: None if value is None else _copy_first_class(value)
            for key, value in mstep.tau2_update_details.items()
        },
    )
    for half_index, prior in enumerate(direction_priors):
        if prior.values is not None:
            direction_priors[half_index] = DirectionPrior(
                _copy_first_class(prior.values), prior.healpix_order,
            )
    class_mixture = class_mixture_from_weights(
        np.full(n_classes, float(class_mixture.weights[0]) / n_classes, dtype=np.float64)
    )
    logger.info("Class3D one-reference start: copied class 1 to every class after the CC iteration")
    return copied, class_mixture


def class_maximization(
    reference_model,
    Ft_y_per_half,
    Ft_ctf_per_half,
    ctx: "RunContext",
    options: RefinementOptions,
    this_iteration: NumberedIteration,
    *,
    halves,
    current_size,
    image_current_size,
    mstep_accumulator_shape,
    mstep_full_half_axis,
    projector_power_spectrum,
    class_tau2: ClassTau2,
) -> ClassMaximization:
    """RELION's Class3D M-step: one prior and one Wiener solve per class from the combined halves.

    In order: combine the half accumulators, take each class's tau2 from the previous reference's power
    spectrum, replace ``reference_model``'s tau2, release its maps and replace them with the
    reconstruction; after a first-iteration CC pass, taper the reported curves. The caller records the
    returned data-vs-prior curve in the history and installs it as the next iteration's scheduling curve.
    ``class_tau2`` is the prior an input source supplies (None shells: the previous references').
    Reads ``reference_model.maps``; from ``options``: ``k_class.n_classes`` and
    ``parity.relion_firstiter_ini_high_angstrom``; from ``ctx``: the reconstruction settings, scoring dtype,
    pixel size and M-step probe; from ``this_iteration``: its index and first-iteration CC.
    """
    parity = options.parity
    Ft_y_0, Ft_y_1 = Ft_y_per_half
    Ft_ctf_0, Ft_ctf_1 = Ft_ctf_per_half
    Ft_y_combined = _combine_optional_half_accumulators(Ft_y_0, Ft_y_1, label="Ft_y")
    Ft_ctf_combined = _combine_optional_half_accumulators(Ft_ctf_0, Ft_ctf_1, label="Ft_ctf")
    # K-class 256px maps are large enough that materializing both
    # previous class stacks on the host immediately after pass 2 can
    # SIGBUS under Slurm/tmp quota pressure.  JAX arrays are immutable;
    # keep device references here and let the later per-class tau2/sign
    # code transfer only the slices it actually needs.
    previous_means = [jnp.asarray(mean) if mean is not None else None for mean in reference_model.maps]
    tau2_clock = Stopwatch()
    class_priors = estimate_class_priors(
        previous_means,
        Ft_y_combined,
        Ft_ctf_combined,
        ctx.reconstruction_settings,
        half_denominators=(Ft_ctf_0, Ft_ctf_1),
        halves=halves,
        n_classes=options.k_class.n_classes,
        iteration=this_iteration.iteration,
        current_size=current_size,
        image_current_size=image_current_size,
        accumulator_shape=mstep_accumulator_shape,
        full_half_axis=mstep_full_half_axis,
        projector_power_spectrum=projector_power_spectrum,
        class_tau2=class_tau2,
        scoring_dtype=ctx.scoring_dtype,
        log=logger,
        probe=ctx.maximization_probe,
    )
    tau2_update_details = _stack_class_tau2_update_details(class_priors.details_per_class)
    logger.info(
        "Computed iter-%d Class3D tau2 from %s: %.1fs",
        this_iteration.iteration + 1,
        class_tau2.source,
        tau2_clock.seconds,
    )
    reference_model.tau2 = class_priors.variance
    reference_model.tau2_per_half = shared_tau2_per_half(reference_model.tau2)

    # --- Free previous-iteration means to reclaim GPU memory ---
    # (previous_means already snapshotted earlier for FSC sign alignment)
    for k in range(2):
        reference_model.maps[k] = None

    # --- Now reconstruct the regularized means ---
    recon_clock = Stopwatch()
    reference_model.maps[:] = reconstruct_numbered_class_maps(
        Ft_y_combined,
        Ft_ctf_combined,
        class_priors.shells,
        ctx.reconstruction_settings,
        n_classes=options.k_class.n_classes,
        iteration=this_iteration.iteration,
        current_size=current_size,
        accumulator_volume_shape=mstep_accumulator_shape,
        relion_firstiter_cc_this_iter=this_iteration.first_iteration.relion_firstiter_cc,
        probe=ctx.maximization_probe,
    )
    logger.info(
        "Regularized reconstruction (2 halves + flatten): %.1fs",
        recon_clock.seconds,
    )
    if this_iteration.first_iteration.relion_firstiter_cc and parity.relion_firstiter_ini_high_angstrom is not None:
        # Class3D tapers each class's tau2_class and data_vs_prior_class the
        # same way (ml_optimiser.cpp:6389-6420). RELION's comment calls this
        # output only, but the next E-step gates each class's scale sums on
        # data_vs_prior_class > 3 (:10473), so the untapered curve let
        # shells past ini_high into iteration 2's scale correction. The
        # class tau2 volumes are recomputed from the Iref power next
        # iteration, so only the shell curves carry the taper.
        tapered_data_vs_prior = _firstiter_cc_ini_high_tapered(
            class_priors.data_vs_prior,
            ctx.reconstruction_settings.box_size,
            ctx.source_pixel_size_angstrom,
            parity.relion_firstiter_ini_high_angstrom,
            filter_edgewidth=REFERENCE_FILTER_EDGE_SHELLS,
        )
        tapered_prior = taper_first_cc_class_prior(
            class_priors.shells,
            tau2_update_details,
            ctx.reconstruction_settings,
            pixel_size_angstrom=ctx.source_pixel_size_angstrom,
        )
        logger.info(
            "RELION iter-1 CC emulation: tapered Class3D tau2/data-vs-prior with ini_high=%.2f A",
            float(parity.relion_firstiter_ini_high_angstrom),
        )
        return ClassMaximization(
            Ft_y_combined, Ft_ctf_combined, previous_means, tapered_prior.shells, tapered_data_vs_prior,
            tapered_prior.details,
        )
    return ClassMaximization(
        Ft_y_combined, Ft_ctf_combined, previous_means, class_priors.shells, class_priors.data_vs_prior,
        tau2_update_details,
    )


class K1Maximization(NamedTuple):
    """What a K=1 M-step leaves for the rest of its iteration (the model is written in place)."""

    # The half accumulators the solve used: joined at low resolution when that is on.
    Ft_y_per_half: tuple
    Ft_ctf_per_half: tuple
    # Host copies of the references the M-step replaced, for sign alignment.
    previous_means: list
    fsc: object
    tau2_update_details: dict
    tau2_update_details_per_half: list


def k1_maximization(
    reference_model,
    Ft_y_per_half,
    Ft_ctf_per_half,
    ctx: "RunContext",
    this_iteration: NumberedIteration,
    *,
    parity,
    pixel_resolutions,
    current_resolution,
    current_size,
    mstep_accumulator_shape,
    mstep_full_half_axes,
) -> K1Maximization:
    """RELION's split-half auto-refine M-step (compareTwoHalves -> updateSSNRarrays -> reconstruct).

    In order: join the half accumulators at low resolution when requested, copy the previous references
    to host and release them, estimate the split-half prior from this iteration's FSC, replace
    ``reference_model``'s tau2 and maps; after a first-iteration CC pass, taper the reported tau2; then
    park the tau2 volumes on the host. ``ctx.maximization_probe`` sees the joined accumulators before the prior
    reads them. Reads from ``ctx``: the reconstruction settings, scoring dtype and pixel size; from
    ``this_iteration``: its index and first-iteration CC.
    """
    Ft_y_0, Ft_y_1 = Ft_y_per_half
    Ft_ctf_0, Ft_ctf_1 = Ft_ctf_per_half
    retained_Ft_y_0_device = None
    # RELION's --low_resol_join_halves averages the low-resolution shells of
    # the K=1 half accumulators before the Wiener solve; see
    # join_half_accumulators_at_low_resolution for the rationale and cap.
    if parity.low_resol_join_halves_angstrom is not None and parity.low_resol_join_halves_angstrom > 0:
        Ft_y_0, Ft_y_1, Ft_ctf_0, Ft_ctf_1, retained_Ft_y_0_device = join_half_accumulators_at_low_resolution(
            (Ft_y_0, Ft_y_1),
            (Ft_ctf_0, Ft_ctf_1),
            accumulator_volume_shape=mstep_accumulator_shape,
            box_size=ctx.reconstruction_settings.box_size,
            voxel_size=ctx.source_pixel_size_angstrom,
            padding_factor=RECONSTRUCTION_PADDING_FACTOR,
            low_resolution_angstrom=parity.low_resol_join_halves_angstrom,
            pixel_resolutions=pixel_resolutions,
            current_resolution=current_resolution,
            preserve_inputs=False,
            return_retained_first_numerator=True,
        )
    previous_means = _snapshot_and_release_previous_k1_means(reference_model.maps)
    ctx.maximization_probe.k1_accumulators_joined(
        this_iteration.iteration, numerators=(Ft_y_0, Ft_y_1), denominators=(Ft_ctf_0, Ft_ctf_1), settings=ctx.reconstruction_settings,
        current_size=current_size, accumulator_shape=mstep_accumulator_shape,
        pixel_size_angstrom=ctx.source_pixel_size_angstrom,
    )
    split_prior = estimate_split_half_prior(
        (Ft_y_0, Ft_y_1),
        (Ft_ctf_0, Ft_ctf_1),
        ctx.reconstruction_settings,
        current_size=current_size,
        accumulator_shape=mstep_accumulator_shape,
        full_half_axes=mstep_full_half_axes,
        iteration=this_iteration.iteration,
        scoring_dtype=ctx.scoring_dtype,
        log=logger,
    )
    logger.info("tau2 updated from this iteration's FSC")
    reference_model.tau2 = split_prior.variance
    reference_model.tau2_per_half = _updated_mean_variance_per_half(
        reference_model.tau2,
        split_prior.variance_per_half,
        use_per_half_mean_variance=parity.use_per_half_mean_variance,
    )

    # --- Now reconstruct the regularized means ---
    # (the previous K=1 references were released by the snapshot above)
    recon_clock = Stopwatch()
    reference_model.maps[:] = reconstruct_numbered_k1_halfmaps(
        (Ft_y_0, Ft_y_1),
        (Ft_ctf_0, Ft_ctf_1),
        split_prior.shells_per_half,
        ctx.reconstruction_settings,
        iteration=this_iteration.iteration,
        current_size=current_size,
        accumulator_volume_shape=mstep_accumulator_shape,
        relion_firstiter_cc_this_iter=this_iteration.first_iteration.relion_firstiter_cc,
        retained_first_numerator=retained_Ft_y_0_device,
        probe=ctx.maximization_probe,
    )
    logger.info(
        "Regularized reconstruction (2 halves + flatten): %.1fs",
        recon_clock.seconds,
    )
    retained_Ft_y_0_device = None

    # RELION reconstructs the first-iteration CC maps with the untapered
    # updateSSNRarrays tau2.  Only afterwards does
    # initialLowPassFilterReferences taper tau2/data_vs_prior for the
    # model state and reporting; that tapered spectrum is explicitly not
    # used in the reconstruction calculation (ml_optimiser.cpp:5296-5328).
    # The taper rewrites split_prior's per-half volumes and details in place; its shared volume is new.
    if this_iteration.first_iteration.relion_firstiter_cc and parity.relion_firstiter_ini_high_angstrom is not None:
        reference_model.tau2 = taper_first_cc_k1_prior(
            split_prior.variance_per_half,
            split_prior.details_per_half,
            ctx.reconstruction_settings,
            pixel_size_angstrom=ctx.source_pixel_size_angstrom,
            scoring_dtype=ctx.scoring_dtype,
        ).variance
        reference_model.tau2_per_half = _updated_mean_variance_per_half(
            reference_model.tau2,
            split_prior.variance_per_half,
            use_per_half_mean_variance=parity.use_per_half_mean_variance,
        )
        logger.info(
            "RELION iter-1 CC emulation: tapered post-reconstruction tau2/data-vs-prior "
            "with ini_high=%.2f A",
            float(parity.relion_firstiter_ini_high_angstrom),
        )
    # The K=1 tau2 volumes are read again only by the next M-step (the
    # resident E-step does not use them). Keep them on the host between
    # uses, as RELION keeps tau2 as a host spectrum: at box 800 the four
    # float32 volumes are 8 GB of the device floor (GPU census, bigbox
    # 14480607). The per-half reconstruction volumes are not read again:
    # they are released with split_prior when this function returns.
    reference_model.tau2, reference_model.tau2_per_half = _host_tau2_volumes(
        reference_model.tau2,
        reference_model.tau2_per_half,
    )
    return K1Maximization(
        (Ft_y_0, Ft_y_1),
        (Ft_ctf_0, Ft_ctf_1),
        previous_means,
        split_prior.fsc,
        # Diagnostics follow the half-1 model.star, matching the parity report.
        split_prior.details_per_half[0],
        split_prior.details_per_half,
    )


def k1_learned_direction_priors(
    rotation_posterior_per_half, *, direction_prior_order: int, symmetry: str, dtype, log,
) -> tuple[DirectionPrior | None, DirectionPrior | None]:
    """Each K=1 half's next direction prior, collapsed from its rotation posterior sums at
    ``direction_prior_order``; None for a half that keeps its prior (a rejected half, or no half when either
    half has no posterior sums)."""
    if any(rot_sum is None for rot_sum in rotation_posterior_per_half):
        return None, None
    return learn_k1_direction_priors(
        rotation_posterior_per_half, direction_prior_order=direction_prior_order,
        expected_rotation_count=rotation_grid_size(direction_prior_order, symmetry=symmetry),
        dtype=dtype, log=log, symmetry=symmetry,
    )


def class_learned_direction_priors(
    rotation_posterior_per_half,
    class_rotation_posterior_per_half,
    *,
    n_classes: int,
    direction_prior_order: int,
    symmetry: str,
    use_local: bool,
    n_trial_rotations: int,
    dtype,
) -> tuple[DirectionPrior | None, DirectionPrior | None]:
    """Both Class3D halves' next direction prior, one shared prior from the two halves' class rotation
    posteriors at ``direction_prior_order``; (None, None) when the halves keep theirs: a half without posterior
    sums, or a global search whose trial grid is not the exhaustive grid of that order (a local search's
    posterior grid is the direction-prior grid, where pdf_direction still accumulates)."""
    if any(rot_sum is None for rot_sum in rotation_posterior_per_half):
        return None, None
    exhaustive_grid_size = rotation_grid_size(direction_prior_order, symmetry=symmetry)
    if not (use_local or n_trial_rotations == exhaustive_grid_size) or any(
        rot_sum is None for rot_sum in class_rotation_posterior_per_half
    ):
        return None, None
    return learn_class_direction_priors(
        class_rotation_posterior_per_half, n_classes=n_classes, healpix_order=direction_prior_order, dtype=dtype,
        symmetry=symmetry,
    )
