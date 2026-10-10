"""Numbered prior estimation and map reconstruction.

Reconstruction, first-CC reporting tapers and their private numerical helpers
share this owner. Controllers install the returned maps and priors explicitly;
final all-data orchestration shares numerical primitives with its own policies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
from recovar.core import fourier_transform_utils

from relax.helpers.resolution import (
    _firstiter_cc_ini_high_tapered,
    firstiter_cc_ini_high_tau2_taper,
    shell_index_to_resolution_angstrom,
)
from relax.helpers.timing import Stopwatch
from relax.helpers.types import total_sumw
from relax.reconstruction import regularization_relion
from relax.reconstruction.volume_solver import _finish_host_staged_reconstruction, _reconstruct_volume_eager
from relax.refinement import map_postprocess
from relax.refinement.optics_shapes import average_ctf2_parts
from relax.refinement.ports import ClassPriorEstimated, MaximizationProbe, NoProbe
from relax.refinement.refinement_options import ReconstructionPrograms
from relax.refinement.tomo_half import TomoHalf
from relax.relion import relion_ctf

logger = logging.getLogger(__name__)


def _previous_resolution_angstrom_for_half_join(
    pixel_resolutions, current_resolution, *, box_size: int, voxel_size: float
):
    """Previous-iteration resolution in Å that caps the low-resolution half join.

    The last recorded shell resolution wins when the history has one; a
    non-positive recorded shell leaves the join uncapped. Without history, a
    finite state resolution is used. Mirrors RELION's
    ``XMIPP_MAX(low_resol_join_halves, 1./mymodel.current_resolution)``.
    """

    if pixel_resolutions:
        previous_shell = pixel_resolutions[-1]
        if previous_shell > 0:
            return shell_index_to_resolution_angstrom(previous_shell, box_size, voxel_size)
        return None
    if np.isfinite(float(current_resolution)):
        return float(current_resolution)
    return None


def join_half_accumulators_at_low_resolution(
    numerators_by_half,
    denominators_by_half,
    *,
    accumulator_volume_shape,
    box_size,
    voxel_size,
    padding_factor,
    low_resolution_angstrom,
    pixel_resolutions,
    current_resolution,
    preserve_inputs: bool,
    return_retained_first_numerator=False,
):
    """Apply RELION's ``--low_resol_join_halves`` to K=1 half accumulators before the Wiener solve.

    Averaging the low-resolution shells of the two half accumulators forces
    both half-maps to share their low-frequency content, as in
    ``MlOptimiserMpi::joinTwoHalvesAtLowResolution``; without it the half-map
    FSC drifts below 1 at SNR-poor low shells and current-size growth lags.
    The join radius is capped by the previous iteration's resolution so shells
    beyond the map's actual resolution are never joined. The regular and final
    all-data passes call this with their own accumulators. Returns the four
    joined arrays in input order (``preserve_inputs`` False: host inputs may be updated in place), and a
    fifth value with ``return_retained_first_numerator``, as
    :func:`regularization_relion.join_halves_at_low_resolution` does; refinement state is not mutated.
    """

    previous_resolution_angstrom = _previous_resolution_angstrom_for_half_join(
        pixel_resolutions,
        current_resolution,
        box_size=box_size,
        voxel_size=voxel_size,
    )
    return regularization_relion.join_halves_at_low_resolution(
        numerators_by_half[0],
        numerators_by_half[1],
        denominators_by_half[0],
        denominators_by_half[1],
        accumulator_volume_shape,
        voxel_size,
        box_size,
        low_resolution_angstrom,
        current_resolution_angstrom=previous_resolution_angstrom,
        padding_factor=padding_factor,
        preserve_inputs=preserve_inputs,
        return_retained_first_numerator=return_retained_first_numerator,
    )


_CLASS_TAU2_DETAIL_KEYS = (
    "prior_shells",
    "sigma2_shells",
    "avg_weight_shells",
    "shell_sum",
    "shell_count",
    "fsc_shells",
    "ssnr_shells",
)


def _class_tau2_from_iref_power_spectrum(
    iref_fourier,
    volume_shape,
    *,
    padding_factor,
    current_size: int,
    frame_scale: float,
    projector_power_spectrum=None,
    shell_pair_counting="relion",
):
    """Class3D tau2 for one class from its previous ``Iref`` power spectrum.

    Dense RECOVAR accumulators live in the historical unnormalised image frame
    (RELION BPref weight = ``Ft_ctf * N^4``), so the RELION-frame tau2 is
    scaled by ``frame_scale`` before the Wiener solve. Returns the
    RECOVAR-frame tau2 volume, the RELION-frame radial shells and the
    RECOVAR-frame radial shells, all in the RELION result dtype.
    ``projector_power_spectrum`` is the class's spectrum from this iteration's
    scoring projector setup, when it has one (see
    ``compute_relion_tau2_from_iref_power_spectrum``).
    """

    mean_signal_variance_relion, details = regularization_relion.compute_relion_tau2_from_iref_power_spectrum(
        iref_fourier,
        volume_shape,
        padding_factor=padding_factor,
        current_size=current_size,
        return_details=True,
        projector_power_spectrum=projector_power_spectrum,
        shell_pair_counting=shell_pair_counting,
    )
    mean_signal_variance = mean_signal_variance_relion * jnp.asarray(
        frame_scale,
        dtype=mean_signal_variance_relion.dtype,
    )
    tau2_shells_relion_frame = jnp.asarray(
        details["tau2_shells"],
        dtype=mean_signal_variance.dtype,
    )
    tau2_shells_recovar_frame = tau2_shells_relion_frame * jnp.asarray(
        frame_scale,
        dtype=mean_signal_variance.dtype,
    )
    return mean_signal_variance, tau2_shells_relion_frame, tau2_shells_recovar_frame


def _class_tau2_update_details(
    Ft_ctf_class,
    tau2_shells_recovar_frame,
    shell_stats,
    settings: ReconstructionSettings,
    *,
    current_size: int,
    full_half_axis,
    accumulator_volume_shape,
    average_ctf2=None,
    shell_pair_counting="relion",
):
    """Data-vs-prior and the host tau2 detail record for one class.

    Reads from ``settings``: ``volume_shape``, ``padding_factor`` and ``tau2_fudge``.

    ``shell_stats`` are the round-shell weight statistics of ``Ft_ctf_class``.
    ``average_ctf2`` is RELION's average CTF^2 of CTF-premultiplied images
    (:func:`relax.relion.relion_ctf.premultiplied_average_ctf2`), or None: RELION
    divides ``invtau2`` by it where it is positive (backprojector.cpp:1277-1279), which
    multiplies ``data_vs_prior`` by it. The record uses the K=1 per-half key layout with
    ``fsc_shells`` set to ``None``. Returns ``(data_vs_prior, details)``.
    """

    data_vs_prior = regularization_relion.compute_data_vs_prior(
        Ft_ctf_class,
        tau2_shells_recovar_frame,
        settings.volume_shape,
        padding_factor=settings.padding_factor,
        tau2_fudge=settings.tau2_fudge,
        current_size=current_size,
        full_half_axis=full_half_axis,
        accumulator_volume_shape=accumulator_volume_shape,
        shell_pair_counting=shell_pair_counting,
    )
    if average_ctf2 is not None:
        count = min(int(data_vs_prior.shape[0]), int(np.asarray(average_ctf2).shape[0]))
        factor = np.ones(int(data_vs_prior.shape[0]))
        factor[:count] = np.where(np.asarray(average_ctf2[:count]) > 0.0, average_ctf2[:count], 1.0)
        data_vs_prior = data_vs_prior * jnp.asarray(factor, dtype=data_vs_prior.dtype)
    details = {
        "prior_shells": np.asarray(tau2_shells_recovar_frame, dtype=np.float64),
        "sigma2_shells": np.asarray(
            jnp.where(
                shell_stats["avg_weight_shells"] > 0,
                1.0 / (settings.padding_factor**3 * shell_stats["avg_weight_shells"]),
                0.0,
            ),
            dtype=np.float64,
        ),
        "avg_weight_shells": np.asarray(shell_stats["avg_weight_shells"], dtype=np.float64),
        "shell_sum": np.asarray(shell_stats["shell_sum"], dtype=np.float64),
        "shell_count": np.asarray(shell_stats["shell_count"], dtype=np.float64),
        "fsc_shells": None,
        "ssnr_shells": np.asarray(data_vs_prior, dtype=np.float64),
    }
    return data_vs_prior, details


def _stack_class_tau2_update_details(details_per_class):
    """Stack per-class tau2 detail records along a leading class axis.

    ``fsc_shells`` stays ``None``: Class3D has no half-set FSC.
    """

    return {
        key: None if key == "fsc_shells" else np.stack([details[key] for details in details_per_class], axis=0)
        for key in _CLASS_TAU2_DETAIL_KEYS
    }


@dataclass(frozen=True)
class ClassPriorEstimate:
    """One class's prior, weight statistics and data-vs-prior estimate.

    ``variance`` and ``shells`` use the RECOVAR frame. ``relion_shells`` keeps
    the unscaled diagnostic spectrum. ``details`` holds the weight statistics on round shells.
    """

    variance: object
    shells: object
    relion_shells: object
    data_vs_prior: object
    details: dict


def estimate_class_prior(
    references,
    denominators,
    *,
    class_index,
    settings: ReconstructionSettings,
    current_size,
    accumulator_shape,
    full_half_axis,
    frame_scale,
    projector_power_spectrum=None,
    replay_tau2_shells=None,
    average_ctf2=None,
) -> ClassPriorEstimate:
    """Estimate a class prior from Iref power or diagnostic replay, then its weights.

    Index inside the operation so reference/projector views are created only
    on the native path and weight views follow prior estimation.
    """
    if replay_tau2_shells is not None:
        shells = jnp.asarray(replay_tau2_shells[class_index], dtype=jnp.float32)
        from recovar import utils

        variance = jnp.asarray(
            utils.make_radial_image(shells, settings.volume_shape, extend_last_frequency=True),
            dtype=jnp.float32,
        ).reshape(-1)
        relion_shells = shells / jnp.asarray(frame_scale, dtype=shells.dtype)
    else:
        variance, relion_shells, shells = _class_tau2_from_iref_power_spectrum(
            references[class_index],
            settings.volume_shape,
            padding_factor=settings.padding_factor,
            current_size=current_size,
            frame_scale=frame_scale,
            # The scoring projector's spectrum counts as RELION does; another counting builds its own.
            projector_power_spectrum=(
                None
                if projector_power_spectrum is None or settings.shell_pair_counting != "relion"
                else projector_power_spectrum[class_index]
            ),
            shell_pair_counting=settings.shell_pair_counting,
        )
    weight_shells = regularization_relion.compute_relion_weight_shell_stats(
        denominators[class_index],
        settings.volume_shape,
        padding_factor=settings.padding_factor,
        r_max=current_size // 2,
        shell_rounding="round",
        full_half_axis=full_half_axis,
        accumulator_volume_shape=accumulator_shape,
        shell_pair_counting=settings.shell_pair_counting,
    )
    data_vs_prior, details = _class_tau2_update_details(
        denominators[class_index],
        shells,
        weight_shells,
        settings,
        current_size=current_size,
        full_half_axis=full_half_axis,
        accumulator_volume_shape=accumulator_shape,
        average_ctf2=average_ctf2,
        shell_pair_counting=settings.shell_pair_counting,
    )
    return ClassPriorEstimate(
        variance=variance,
        shells=shells,
        relion_shells=relion_shells,
        data_vs_prior=data_vs_prior,
        details=details,
    )


@dataclass(frozen=True)
class ClassPriorAggregation:
    """Class-axis priors, scheduling curves and detail rows for ordered publication.

    Variance and shell stacks use the RECOVAR frame. The controller publishes
    the data-vs-prior curve before aggregating detail rows, preserving RELION's
    numbered M-step order. No reference maps or unused class scratch are held.
    """

    variance: object
    shells: object
    data_vs_prior: np.ndarray
    details_per_class: list[dict]


def estimate_class_priors(
    previous_half_maps,
    combined_numerators,
    combined_denominators,
    settings: ReconstructionSettings,
    *,
    half_denominators,
    halves,
    noise_stats_per_half,
    n_classes,
    iteration,
    current_size,
    image_current_size,
    accumulator_shape,
    full_half_axis,
    projector_power_spectrum,
    class_tau2,
    scoring_dtype,
    log,
    probe: MaximizationProbe | None = None,
) -> ClassPriorAggregation:
    """Prepare and aggregate numbered Class3D priors from the previous Iref.

    Own diagnostic replay admission, premultiplied-CTF adaptation, ordered class
    computation/capture and class-axis reductions. The controller installs the
    products and applies first-CC taper only after regularized reconstruction.
    See ``docs/math/relion_refinement_algorithm.md`` for the M-step ordering.
    """
    clock = Stopwatch()
    tau2_update_details_per_class = []
    mean_signal_variance_per_class = []
    mean_signal_variance_shells_per_class = []
    data_vs_prior_per_class = []
    # Dense RECOVAR accumulators live in the historical unnormalised
    # image frame: RELION BPref weight = Ft_ctf * N^4. Equivalently,
    # keep Ft_y/Ft_ctf in RECOVAR frame and scale RELION tau2 by N^4
    # before the Wiener solve. The same frame conversion is documented
    # in docs/math/ab_initio_initial_model_algorithm.md.
    kclass_tau2_frame_scale = float(settings.box_size) ** 4
    # The prior shells an input source supplies (``ports.ClassTau2``), or None: the previous references'.
    kclass_tau2_source = class_tau2.source
    probe = NoProbe() if probe is None else probe
    # CTF-premultiplied images: RELION's average CTF^2 correction of data_vs_prior
    # (setAverageCTF2; Class3D has no split halves and does not fix tau2). Its numerator sums
    # over images, so a subtomogram half counts its tilt images, each with its particle's scale,
    # and a half with several image shapes counts each shape class on its own grid. Its
    # denominator is the E-step's significant weight (sumw_group), the noise update's sumw. A
    # subtomogram adds it once per particle (acc_ml_optimiser_impl.h:2842), not per tilt image
    # (relax#63), so there the average is the per-particle sum of CTF^2 over its tilts.
    ctf2_parts = []
    for half in halves:
        dataset, scales = half.dataset, half.scale_corrections
        if isinstance(dataset, TomoHalf):
            scales = None if scales is None else np.repeat(np.asarray(scales), np.diff(dataset.unit_image_offsets))
            dataset = dataset.images
        ctf2_parts += average_ctf2_parts(
            dataset, scales, current_size=current_size, image_current_size=image_current_size
        )
    sumw = sum(total_sumw(stats.sumw) for stats in noise_stats_per_half if stats is not None)
    average_ctf2 = relion_ctf.premultiplied_average_ctf2(ctf2_parts, settings.box_size, sumw)
    for class_idx in range(n_classes):
        log.info(
            "Class3D tau2 update start: iter=%d class=%d/%d current_size=%d source=%s spectrum=%s",
            iteration + 1,
            class_idx + 1,
            n_classes,
            int(current_size),
            kclass_tau2_source,
            (
                "scoring projector"
                if projector_power_spectrum is not None and settings.shell_pair_counting == "relion"
                else "host transform"
            ),
        )
        class_prior = estimate_class_prior(
            previous_half_maps[0],
            combined_denominators,
            class_index=class_idx,
            settings=settings,
            current_size=current_size,
            accumulator_shape=accumulator_shape,
            full_half_axis=full_half_axis,
            frame_scale=kclass_tau2_frame_scale,
            projector_power_spectrum=projector_power_spectrum,
            replay_tau2_shells=class_tau2.shells,
            average_ctf2=average_ctf2,
        )
        mean_signal_variance_per_class.append(class_prior.variance)
        mean_signal_variance_shells_per_class.append(class_prior.shells)
        data_vs_prior_per_class.append(class_prior.data_vs_prior)
        tau2_update_details_per_class.append(class_prior.details)
        probe.class_prior_estimated(ClassPriorEstimated(
            iteration, class_idx, class_prior, numerators=combined_numerators, denominators=combined_denominators,
            half_denominators=half_denominators, references=previous_half_maps, settings=settings,
            current_size=current_size, source=kclass_tau2_source, accumulator_shape=accumulator_shape,
            full_half_axis=full_half_axis, frame_scale=kclass_tau2_frame_scale,
        ))
        log.info(
            "Class3D tau2 update done: iter=%d class=%d/%d elapsed=%.1fs",
            iteration + 1,
            class_idx + 1,
            n_classes,
            clock.seconds,
        )
    mean_signal_variance = jnp.stack(mean_signal_variance_per_class, axis=0)
    mean_signal_variance_shells = jnp.stack(mean_signal_variance_shells_per_class, axis=0)
    data_vs_prior_iter = np.stack(
        [np.asarray(dvp, dtype=scoring_dtype) for dvp in data_vs_prior_per_class],
        axis=0,
    )
    return ClassPriorAggregation(
        variance=mean_signal_variance,
        shells=mean_signal_variance_shells,
        data_vs_prior=data_vs_prior_iter,
        details_per_class=tau2_update_details_per_class,
    )


def merged_half_map(means):
    """The mean of the two half maps (or half class stacks).

    Class3D's two slots hold one stack, and (x + x) / 2 is x: no second box-scale stack (relax#49: 1.23 GiB at the
    end of a box-380 K3 run on a 16 GB card).
    """
    return means[0] if means[0] is means[1] else (means[0] + means[1]) / 2


def weighted_class_merge(class_means, class_weights):
    """One map from a ``(K, V)`` class stack, weighted by the ``(K,)`` class weights."""
    class_weights_jax = jnp.asarray(class_weights, dtype=class_means.real.dtype)
    return jnp.sum(class_weights_jax[:, None] * class_means, axis=0)


@dataclass(frozen=True, kw_only=True)
class ReconstructionSettings:
    """Run-level geometry, regularization, mask and initial-filter settings."""

    box_size: int
    voxel_size: float
    volume_shape: tuple
    padding_factor: int
    projection_padding_factor: int
    minres_map: int
    width_mask_edge: int
    fmask_edge: int
    tau2_fudge: float
    particle_diameter_angstrom: float | None
    first_iteration_lowpass_angstrom: float | None
    # Real-space gridding-correction window of every reconstruction ("radial" is RELION's);
    # "separable" is K=1 only (RelionConsistencyOptions; require_consistency_route refuses it for Class3D).
    gridding_kernel: str
    # How the 3-D shell statistics behind tau2, data-vs-prior and the half-map FSC count Hermitian
    # pairs: "relion" counts those of the stored half's zero plane twice, "once" every pair once.
    # The 1/1000 weight floor inside the reconstruction (RECOVAR) keeps RELION's counting.
    shell_pair_counting: str
    # RELION --solvent_mask: the user reference mask on the model grid in the internal (z, y, x)
    # frame (relax.reconstruction.solvent_mask.read_solvent_mask, transposed); it replaces the
    # particle-diameter sphere of the solvent flatten. None keeps the sphere.
    solvent_mask: object = field(compare=False, repr=False)
    # RELION --solvent_correct_fsc (MPI relion_refine): the K=1 half-set FSC is the masked,
    # phase-randomisation corrected FSC of the unregularised half maps; needs solvent_mask.
    solvent_correct_fsc: bool
    # The corrected FSC's random-phase source: the run's one glibc rand() stream (RELION's MPI leader
    # stream, seeded once with --random_seed; relax.reconstruction.solvent_mask), advanced by every
    # iteration that randomises phases. Mutable run state, so outside comparison; None without
    # solvent_correct_fsc.
    solvent_phase_stream: object = field(compare=False, repr=False)
    # The run's reconstruction programs (ScoringVariants.reconstruction): no effect on values.
    programs: ReconstructionPrograms

    def __post_init__(self):
        # Python floats, so the solvent-mask radius is the same double arithmetic for every caller.
        object.__setattr__(self, "voxel_size", float(self.voxel_size))
        if self.particle_diameter_angstrom is not None:
            object.__setattr__(self, "particle_diameter_angstrom", float(self.particle_diameter_angstrom))
        if self.solvent_correct_fsc and self.solvent_mask is None:
            raise ValueError("--solvent_correct_fsc needs --solvent_mask (RELION corrects only with a user mask)")
        if self.solvent_correct_fsc and self.solvent_phase_stream is None:
            raise ValueError("--solvent_correct_fsc needs the run's random-phase stream")
        if self.solvent_mask is not None and tuple(np.shape(self.solvent_mask)) != tuple(self.volume_shape):
            raise ValueError(
                f"solvent mask shape {np.shape(self.solvent_mask)} is not the model's {tuple(self.volume_shape)}"
            )

    def reconstruct(self, Ft_ctf, Ft_y, *, tau, current_size, accumulator_volume_shape, **solve_options):
        """One map from one accumulator pair with this run's reconstruction settings.

        Forwards ``volume_shape``, ``padding_factor``, ``tau2_fudge``, ``projection_padding_factor``,
        ``minres_map`` and ``gridding_kernel`` to ``_reconstruct_volume_eager``; ``solve_options`` are its
        per-call options (``tau_is_1d``, ``preserve_output_precision``, ...).
        """
        return _reconstruct_volume_eager(
            Ft_ctf,
            Ft_y,
            self.volume_shape,
            self.padding_factor,
            tau=tau,
            tau2_fudge=self.tau2_fudge,
            projection_padding_factor=self.projection_padding_factor,
            minres_map=self.minres_map,
            current_size=current_size,
            accumulator_volume_shape=accumulator_volume_shape,
            gridding_kernel=self.gridding_kernel,
            programs=self.programs,
            **solve_options,
        )


@dataclass(frozen=True)
class SplitHalfPrior:
    """K1 scoring/reconstruction priors and the shared FSC, which also drives tau2 and size growth."""

    variance: object
    variance_per_half: list
    shells_per_half: list
    fsc: object
    details_per_half: list[dict]


def estimate_split_half_prior(
    numerators,
    denominators,
    settings: ReconstructionSettings,
    *,
    current_size,
    accumulator_shape,
    full_half_axes,
    iteration,
    scoring_dtype,
    log,
) -> SplitHalfPrior:
    """Estimate independent half priors from this expectation's shared FSC.

    The raw backprojector FSC is reported and drives tau2 and size growth.
    The two halves retain their own
    Fourier weights.
    See ``docs/math/relion_refinement_algorithm.md`` for the M-step ordering.
    """

    fsc_clock = Stopwatch()
    current_iter_fsc = regularization_relion.compute_relion_fsc_from_backprojector(
        numerators[0],
        numerators[1],
        denominators[0],
        denominators[1],
        settings.volume_shape,
        padding_factor=settings.padding_factor,
        r_max=current_size // 2,
        accumulator_volume_shape=accumulator_shape,
        output_dtype=scoring_dtype,
        shell_pair_counting=settings.shell_pair_counting,
    )
    log.info(
        "Computed iter-%d FSC for tau2 (RELION backprojector path): %.1fs",
        iteration + 1,
        fsc_clock.seconds,
    )
    if settings.solvent_correct_fsc:
        current_iter_fsc = solvent_corrected_fsc(
            numerators,
            denominators,
            settings,
            current_size=current_size,
            accumulator_shape=accumulator_shape,
            label=f"iter-{iteration + 1}",
            like=current_iter_fsc,
            log=log,
        )

    # RELION calls BackProjector::updateSSNRarrays independently for each
    # half-map BPref.  The gold-standard FSC is shared, but sigma2/tau2
    # come from each half's own Fourier weight outside the joined shells.
    tau2_update_details_per_half = []
    mean_signal_variance_per_half = []
    for half_idx, Ft_ctf_half in enumerate((denominators[0], denominators[1])):
        full_half_axis = full_half_axes[half_idx]
        mean_signal_variance_k, _, tau2_update_details_k = regularization_relion.compute_relion_tau2_from_weights(
            Ft_ctf_half,
            Ft_ctf_half,
            current_iter_fsc,
            settings.volume_shape,
            tau2_fudge=settings.tau2_fudge,
            padding_factor=settings.padding_factor,
            r_max=current_size // 2,
            return_details=True,
            full_half_axis=-1 if full_half_axis is None else int(full_half_axis),
            accumulator_volume_shape=accumulator_shape,
            output_dtype=scoring_dtype,
            shell_pair_counting=settings.shell_pair_counting,
        )
        mean_signal_variance_per_half.append(mean_signal_variance_k)
        tau2_update_details_per_half.append(tau2_update_details_k)
    mean_signal_variance_shells_per_half = [details["prior_shells"] for details in tau2_update_details_per_half]
    mean_signal_variance = 0.5 * (mean_signal_variance_per_half[0] + mean_signal_variance_per_half[1])
    return SplitHalfPrior(
        variance=mean_signal_variance,
        variance_per_half=mean_signal_variance_per_half,
        shells_per_half=mean_signal_variance_shells_per_half,
        fsc=current_iter_fsc,
        details_per_half=tau2_update_details_per_half,
    )


def solvent_corrected_fsc(numerators, denominators, settings, *, current_size, accumulator_shape, label, like, log):
    """RELION's --solvent_correct_fsc curve in place of the backprojector FSC ``like``.

    The unregularised half maps are those relax writes as run_itNNN_half*_unfil.mrc
    (``reconstruct_unregularized_k1_halfmaps``), made from this iteration's accumulators after the
    low-resolution join, as RELION reconstructs its BPref copies (ml_optimiser_mpi.cpp:3221-3300).
    The phases come from ``settings.solvent_phase_stream``, which this call advances; ``label`` names
    the iteration in the log.
    """

    from relax.reconstruction.solvent_mask import solvent_corrected_fsc

    clock = Stopwatch()
    unregularised = reconstruct_unregularized_k1_halfmaps(
        numerators,
        denominators,
        settings,
        accumulator_volume_shape=accumulator_shape,
    )
    # RELION's getFSC sums the FFTW half spectrum halved along the map file's x axis, and the
    # half-spectrum sum depends on which axis is halved; the maps and the mask return to the file's
    # axis order (the transpose of the internal frame; relax.helpers.map_io).
    half1, half2 = (
        np.transpose(
            np.real(np.asarray(fourier_transform_utils.get_idft3(jnp.asarray(m).reshape(settings.volume_shape)))),
            (2, 1, 0),
        )
        for m in unregularised
    )
    fsc, details = solvent_corrected_fsc(
        half1,
        half2,
        np.transpose(np.asarray(settings.solvent_mask, dtype=np.float64), (2, 1, 0)),
        current_size=current_size,
        stream=settings.solvent_phase_stream,
    )
    like = np.asarray(like)
    if fsc.shape != like.shape:
        raise ValueError(f"the corrected FSC has {fsc.shape[0]} shells, the backprojector FSC {like.shape[0]}")
    log.info(
        "%s solvent-corrected FSC: randomize phases beyond shell %d (%.2f A), %d phase draws (%d in the run): %.1fs",
        label,
        details["randomize_at"],
        settings.box_size * settings.voxel_size / max(details["randomize_at"], 1),
        details["draws"],
        settings.solvent_phase_stream.draws,
        clock.seconds,
    )
    return jnp.asarray(fsc, dtype=like.dtype)


def _reconstruct_k1_maps(
    numerators_by_half,
    denominators_by_half,
    tau_by_half,
    settings: ReconstructionSettings,
    *,
    current_size,
    accumulator_volume_shape,
    retained_first_numerator=None,
) -> list:
    """Reconstruct both K=1 halves while preserving RELION buffer lifetime."""

    cs_int = int(current_size) if current_size is not None else None
    reconstructed_means = []
    retained_device_numerator = retained_first_numerator
    for k, (Ft_y_half, Ft_ctf_half, tau_half) in enumerate(zip(numerators_by_half, denominators_by_half, tau_by_half)):
        # This RELION build uses double RFLOAT in BackProjector::reconstruct.
        # Keep the stored/controller tau2 state compact, but promote the
        # reconstruction operand so 1 / (padding_factor**3 * tau2) is not
        # rounded in float32 before it enters the Wiener denominator.
        reconstruction_tau = jnp.asarray(tau_half, dtype=jnp.float64)
        reconstructed = settings.reconstruct(
            Ft_ctf_half,
            Ft_y_half,
            tau=reconstruction_tau,
            current_size=cs_int,
            accumulator_volume_shape=accumulator_volume_shape,
            tau_is_1d=True,
            preserve_output_precision=True,
            relion_filter_scale=float(settings.volume_shape[0] ** 4),
            **(
                {"retained_device_numerator": retained_device_numerator}
                if k == 0 and retained_device_numerator is not None
                else {}
            ),
        ).reshape(-1)
        reconstructed_means.append(_finish_host_staged_reconstruction(reconstructed, Ft_ctf_half, Ft_y_half))
        if k == 0 and retained_device_numerator is not None:
            retained_device_numerator = None
    return reconstructed_means


def _reconstruct_class_maps(
    combined_numerators,
    combined_denominators,
    tau_by_class,
    settings: ReconstructionSettings,
    *,
    n_classes,
    iteration,
    current_size,
    accumulator_volume_shape,
    unsolved=None,
):
    """Reconstruct the shared Class3D stack from combined accumulators.

    ``unsolved`` maps a class that received no weight to the reference it keeps (None: zero), as RELION's
    maximization does not reconstruct it (ml_optimiser.cpp:4958-5026).
    """

    clock = Stopwatch()
    cs_int = int(current_size) if current_size is not None else None
    shared_class_maps = []
    for class_idx in range(n_classes):
        if unsolved and class_idx in unsolved:
            kept = unsolved[class_idx]
            shared_class_maps.append(None if kept is None else jnp.asarray(kept).reshape(-1))
            logger.info(
                "Class3D reconstruction skipped: iter=%d class=%d/%d received no weight (%s)",
                iteration + 1, class_idx + 1, n_classes, "zero reference" if kept is None else "previous reference kept",
            )
            continue
        logger.info(
            "Class3D reconstruction start: iter=%d class=%d/%d current_size=%s",
            iteration + 1,
            class_idx + 1,
            n_classes,
            cs_int,
        )
        class_map = settings.reconstruct(
            combined_denominators[class_idx],
            combined_numerators[class_idx],
            tau=tau_by_class[class_idx],
            current_size=cs_int,
            accumulator_volume_shape=accumulator_volume_shape,
            tau_is_1d=True,
        ).reshape(-1)
        shared_class_maps.append(class_map)
        logger.info(
            "Class3D reconstruction done: iter=%d class=%d/%d elapsed=%.1fs",
            iteration + 1,
            class_idx + 1,
            n_classes,
            clock.seconds,
        )
    solved = next((class_map for class_map in shared_class_maps if class_map is not None), None)
    if solved is None:
        raise RuntimeError("every class is empty: no class received any particle weight")
    shared_classes = jnp.stack(
        [jnp.zeros_like(solved) if class_map is None else class_map.astype(solved.dtype) for class_map in shared_class_maps],
        axis=0,
    )
    logger.info(
        "Class3D reconstruction stack complete: iter=%d classes=%d elapsed=%.1fs",
        iteration + 1,
        n_classes,
        clock.seconds,
    )
    return shared_classes


def reconstruct_numbered_k1_halfmaps(
    numerators_by_half,
    denominators_by_half,
    tau_by_half,
    settings: ReconstructionSettings,
    *,
    iteration,
    current_size,
    accumulator_volume_shape,
    relion_firstiter_cc_this_iter,
    retained_first_numerator=None,
    probe: MaximizationProbe | None = None,
) -> list:
    """Solve two independent numbered K1 maps, then postprocess each half.

    Numerators, denominators and priors are ordered half pairs; each prior is
    a shell curve. The private solve frame owns promoted priors, host
    completion and the retained half-0 numerator boundary. Both solves finish
    before each half is captured, first-CC filtered and solvent-flattened in
    turn. The flatten host-stages box-scale results and consumes its mask.
    Return ready maps for installation.
    """
    probe = NoProbe() if probe is None else probe
    means = _reconstruct_k1_maps(
        numerators_by_half,
        denominators_by_half,
        tau_by_half,
        settings,
        current_size=current_size,
        accumulator_volume_shape=accumulator_volume_shape,
        retained_first_numerator=retained_first_numerator,
    )
    for k in range(2):
        probe.map_solved(iteration, k, means[k], settings=settings, current_size=current_size, n_classes=1)
        # RELION filters Iref inside maximizationOtherParameters, then calls
        # solventFlatten from the outer iteration loop.  These operations do
        # not commute: masking in real space after the Fourier low-pass adds a
        # small, deterministic high-shell tail.
        if relion_firstiter_cc_this_iter:
            means[k] = map_postprocess._apply_relion_initial_lowpass_filter(
                means[k],
                settings.volume_shape,
                settings.voxel_size,
                settings.first_iteration_lowpass_angstrom,
                filter_edgewidth=settings.fmask_edge,
            )
        if map_postprocess._solvent_flatten_requested(settings):
            solvent_mask = map_postprocess._numbered_solvent_mask(settings, dtype=means[k].real.dtype)
            means[k] = map_postprocess._apply_relion_solvent_flatten_k1(
                means[k],
                solvent_mask,
                settings.volume_shape,
            )
    if relion_firstiter_cc_this_iter:
        map_postprocess._log_first_cc_lowpass(settings)
    return means


def reconstruct_numbered_class_maps(
    combined_numerators,
    combined_denominators,
    tau_by_class,
    settings: ReconstructionSettings,
    *,
    n_classes,
    iteration,
    current_size,
    accumulator_volume_shape,
    relion_firstiter_cc_this_iter,
    probe: MaximizationProbe | None = None,
    unsolved=None,
) -> list:
    """Solve one numbered Class3D reference stack from combined partitions.

    Accumulators and priors have a leading class axis; each prior is a shell
    curve. All class solves finish before premask capture, initial filtering
    and solvent flattening.
    Return a two-entry list of particle-execution slots, not scientific
    halves; both entries alias one stack. Each slot is captured, then every
    class is first-CC filtered, then every class is flattened on the device
    with one mask. Both slots hold the same stack and the steps are the same,
    so they run once, in place.
    """
    probe = NoProbe() if probe is None else probe
    shared_classes = _reconstruct_class_maps(
        combined_numerators,
        combined_denominators,
        tau_by_class,
        settings,
        n_classes=n_classes,
        iteration=iteration,
        current_size=current_size,
        accumulator_volume_shape=accumulator_volume_shape,
        unsolved=unsolved,
    )
    for k in range(2):
        probe.map_solved(
            iteration, k, shared_classes, settings=settings, current_size=current_size, n_classes=n_classes
        )
    # As for K1, the low-pass precedes the solvent flatten and does not commute with it.
    if relion_firstiter_cc_this_iter:
        shared_classes = map_postprocess._lowpass_class_stack(shared_classes, settings, n_classes)
    if map_postprocess._solvent_flatten_requested(settings):
        solvent_mask = map_postprocess._numbered_solvent_mask(settings, dtype=jnp.finfo(shared_classes.dtype).dtype)
        shared_classes = map_postprocess._flatten_class_stack(shared_classes, solvent_mask, settings.volume_shape, n_classes)
    if relion_firstiter_cc_this_iter:
        map_postprocess._log_first_cc_lowpass(settings)
    return [shared_classes, shared_classes]


@dataclass(frozen=True)
class K1ReportingPrior:
    """Tapered K1 shared/half volumes and their per-half reporting details."""

    variance: object
    variance_per_half: list
    details_per_half: list[dict]


def taper_first_cc_k1_prior(
    variance_per_half,
    details_per_half,
    settings: ReconstructionSettings,
    *,
    pixel_size_angstrom,
    scoring_dtype,
) -> K1ReportingPrior:
    """Taper K1 reporting priors after the untapered regularized reconstruction.

    Updates ``variance_per_half`` and ``details_per_half`` in place, preserving their identity and the
    half -> prior/SSNR update order. The expanded taper and radial grid are temporary implementation
    arrays; model installation and shared/per-half policy stay with the caller.
    """
    tau2_taper = firstiter_cc_ini_high_tau2_taper(
        len(details_per_half[0]["prior_shells"]),
        settings.box_size,
        pixel_size_angstrom,
        settings.first_iteration_lowpass_angstrom,
        filter_edgewidth=settings.fmask_edge,
    )
    radial_shells = np.asarray(
        fourier_transform_utils.get_grid_of_radial_distances(
            settings.volume_shape,
            scaled=False,
            frequency_shift=0,
        ),
        dtype=np.int32,
    ).reshape(-1)
    radial_shells = np.minimum(radial_shells, len(tau2_taper) - 1)
    tau2_taper_volume = jnp.asarray(
        tau2_taper[radial_shells],
        dtype=scoring_dtype,
    )
    for half_idx in range(2):
        variance_per_half[half_idx] = variance_per_half[half_idx] * tau2_taper_volume
        for key in ("prior_shells", "ssnr_shells"):
            field_values = details_per_half[half_idx][key]
            details_per_half[half_idx][key] = field_values * jnp.asarray(
                tau2_taper,
                dtype=field_values.dtype,
            )
    variance = 0.5 * (variance_per_half[0] + variance_per_half[1])
    return K1ReportingPrior(variance, variance_per_half, details_per_half)


@dataclass(frozen=True)
class ClassReportingPrior:
    """Tapered Class3D shell stack and its aggregate prior/SSNR details."""

    shells: object
    details: dict


def taper_first_cc_class_prior(
    shells,
    details,
    settings: ReconstructionSettings,
    *,
    pixel_size_angstrom,
) -> ClassReportingPrior:
    """Taper class reporting shells after the caller publishes the tapered curve.

    The controller first tapers/publishes data-vs-prior for scheduling. This
    operation then adapts the shell stack and aggregate detail arrays in their
    existing order, in place, preserving the detail mapping's identity.
    """

    def taper_shells(values):
        return _firstiter_cc_ini_high_tapered(
            values,
            settings.box_size,
            pixel_size_angstrom,
            settings.first_iteration_lowpass_angstrom,
            filter_edgewidth=settings.fmask_edge,
        )

    shells = jnp.asarray(taper_shells(np.asarray(shells)))
    for key in ("prior_shells", "ssnr_shells"):
        details[key] = taper_shells(np.asarray(details[key]))
    return ClassReportingPrior(shells, details)


# ---------------------------------------------------------------------------
# Unregularized half-map reconstruction + sign alignment
# ---------------------------------------------------------------------------


def reconstruct_unregularized_k1_halfmaps(
    Ft_y_per_half,
    Ft_ctf_per_half,
    settings: ReconstructionSettings,
    *,
    accumulator_volume_shape=None,
) -> list:
    """Reconstruct each K=1 half from its own unregularized accumulator."""

    return [
        settings.reconstruct(
            Ft_ctf_half, Ft_y_half, tau=None, current_size=None, accumulator_volume_shape=accumulator_volume_shape
        )
        for Ft_ctf_half, Ft_y_half in zip(Ft_ctf_per_half, Ft_y_per_half)
    ]


def reconstruct_unregularized_class_means(
    Ft_y_combined,
    Ft_ctf_combined,
    settings: ReconstructionSettings,
    n_classes,
    *,
    accumulator_volume_shape=None,
) -> list:
    """Reconstruct the shared K-class stack from combined accumulators."""

    unreg_shared = jnp.stack(
        [
            settings.reconstruct(
                Ft_ctf_combined[class_idx],
                Ft_y_combined[class_idx],
                tau=None,
                current_size=None,
                accumulator_volume_shape=accumulator_volume_shape,
            ).reshape(-1)
            for class_idx in range(n_classes)
        ],
        axis=0,
    )
    return [unreg_shared, unreg_shared]
