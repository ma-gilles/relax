"""The numbered reconstructions: their settings (``ReconstructionSettings``), the regularized K=1 and class
maps of an M-step with their post-processing, and the unregularized maps an FSC or a dump reads."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from relax.helpers.timing import Stopwatch
from relax.reconstruction.volume_solver import _finish_host_staged_reconstruction, _reconstruct_volume_eager
from relax.refinement import map_postprocess
from relax.refinement.ports import MaximizationProbe, NoProbe
from relax.refinement.refinement_options import ReconstructionPrograms

logger = logging.getLogger(__name__)


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
            means[k] = map_postprocess.apply_relion_initial_lowpass_filter(
                means[k],
                settings.volume_shape,
                settings.voxel_size,
                settings.first_iteration_lowpass_angstrom,
                filter_edgewidth=settings.fmask_edge,
            )
        if map_postprocess.solvent_flatten_requested(settings):
            solvent_mask = map_postprocess.numbered_solvent_mask(settings, dtype=means[k].real.dtype)
            means[k] = map_postprocess.apply_relion_solvent_flatten_k1(
                means[k],
                solvent_mask,
                settings.volume_shape,
            )
    if relion_firstiter_cc_this_iter:
        map_postprocess.log_first_cc_lowpass(settings)
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
        shared_classes = map_postprocess.lowpass_class_stack(shared_classes, settings, n_classes)
    if map_postprocess.solvent_flatten_requested(settings):
        solvent_mask = map_postprocess.numbered_solvent_mask(settings, dtype=jnp.finfo(shared_classes.dtype).dtype)
        shared_classes = map_postprocess.flatten_class_stack(shared_classes, solvent_mask, settings.volume_shape, n_classes)
    if relion_firstiter_cc_this_iter:
        map_postprocess.log_first_cc_lowpass(settings)
    return [shared_classes, shared_classes]


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
