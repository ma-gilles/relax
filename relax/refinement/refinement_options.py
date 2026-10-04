"""Grouped settings accepted by ``refine_single_volume(options=...)``.

The option groups configure scheduling, scoring, replay and diagnostics.

Frozen dataclasses prevent field rebinding. Array and mapping payloads are
still shared objects; freezing does not make them immutable or hashable.
See ``docs/math/relion_refinement_algorithm.md`` for the execution map.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from relax.helpers.convergence import LOCAL_SEARCH_HEALPIX_ORDER
from relax.symmetry import canonicalize_rotational_symmetry


@dataclass(frozen=True)
class RefinementSchedule:
    """How long the refinement runs and what grid it starts at."""

    max_iter: int = 10
    init_current_size: int = 32
    init_healpix_order: int = 2
    # None: RELION's auto-refine has no HEALPix cap; an int is an explicit opt-in cap.
    max_healpix_order: int | None = None
    init_translation_range: float = 10.0
    init_translation_step: float = 2.0
    init_translation_sigma_angstrom: float = 10.0
    particle_diameter_ang: float | None = None
    init_relion_iteration: int = 0
    init_fsc: Any | None = None
    # RELION's --ini_high of a fresh run, which seeds the iteration-0 current resolution
    # (ml_optimiser.cpp:6768); None leaves it unset, as RELION without --ini_high.
    ini_high_angstrom: float | None = None
    # RELION-unit data_vs_prior spectrum of the start-up model (initialiseDataVersusPrior);
    # it selects the iteration-1 scale-correction shells (ml_optimiser.cpp:10473).
    init_data_vs_prior: Any | None = None
    init_ave_Pmax: float | None = None
    init_has_high_fsc_at_limit: bool | None = None
    force_max_iter_after_convergence: bool = False
    skip_final_iteration: bool = False
    init_relion_incr_size: int = 10


@dataclass(frozen=True)
class AdaptiveOptions:
    """Pose-search resolution + adaptive-oversampling knobs."""

    adaptive_oversampling: int = 0
    max_significants: int = 500
    # gemm_dense is an experimental, no-pruning global route; auto remains the production default.
    coarse_engine: Literal["auto", "gemm_hybrid", "gemm_dense"] = "auto"
    relion_current_sizes: tuple[int, ...] | None = None
    relion_healpix_orders: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.coarse_engine not in {"auto", "gemm_hybrid", "gemm_dense"}:
            raise ValueError(f"unknown coarse engine {self.coarse_engine!r}")


@dataclass(frozen=True)
class RelionParityOptions:
    """Knobs that pin RELION numerical behavior."""

    low_resol_join_halves_angstrom: float = 40.0
    tau2_fudge: float = 1.0
    perturb_factor: float = 0.0
    perturb_seed: int | None = None
    relion_optics_image_sizes: Any | None = None
    relion_optics_pixel_sizes: Any | None = None
    # Each half's per-image optics-group row (0 .. G-1) of a per-group noise table
    # (relax.helpers.optics_noise); required when the initial noise has G > 1 rows.
    optics_group_ids_per_half: Any | None = None
    relion_model_pixel_size: float | None = None
    perturb_replay_relion_dir: str | None = None
    perturb_replay_relion_prefix: str = "run"
    # Diagnostic-only cutoff (number of physical recovar iterations): after
    # this many iterations, `refine_single_volume` stops reading
    # RELION's per-iteration sampling/model/optimiser STAR files entirely and
    # resumes native sampling/convergence from recovar's carried state. Zero
    # disables numbered replay immediately after the initial snapshot. `None`
    # (default) reads RELION's STAR files every iteration, matching prior
    # behavior. See scripts/run_multi_iter_parity.py's
    # --replay-override-max-iter, which sets this alongside
    # replay.replay_iteration_overrides.
    perturb_replay_max_iter: int | None = None
    perturb_replay_precision: Literal["auto", "seed_exact", "star"] = "auto"
    perturb_replay_restart_state_iterations: tuple[int, ...] = ()
    final_sampling_replay_relion_dir: str | None = None
    emulate_relion_firstiter_cc: bool = False
    relion_firstiter_ini_high_angstrom: float | None = None
    first_iteration_score_mode: str = "gaussian"
    first_iteration_reconstruction_mode: str = "soft"
    image_fourier_backend: Literal["host_numpy", "jax_gpu", "relion_cuda"] = "host_numpy"
    optimizer_random_seed: int | None = None
    use_per_half_mean_variance: bool = False
    preserve_bpref_particle_order: bool = False
    allow_replayed_bpref_particle_order: bool = False
    # K=1 --firstiter_cc: rescore an image's top two coarse CC poses on RELION's
    # coarse tree when their margin is at most this (the CLI default is 4e-6); None is off.
    firstiter_cc_tree_rescore_max_margin: float | None = None

    def __post_init__(self):
        if self.image_fourier_backend not in {
            "host_numpy",
            "jax_gpu",
            "relion_cuda",
        }:
            raise ValueError(
                "image_fourier_backend must be "
                "'host_numpy', 'jax_gpu', or 'relion_cuda', "
                f"got {self.image_fourier_backend!r}"
            )

        margin = self.firstiter_cc_tree_rescore_max_margin
        if margin is not None and not (math.isfinite(float(margin)) and float(margin) >= 0.0):
            raise ValueError(f"firstiter_cc_tree_rescore_max_margin must be a finite non-negative float, got {margin!r}")

        if self.perturb_replay_max_iter is not None and self.perturb_replay_max_iter < 0:
            raise ValueError(
                "perturb_replay_max_iter must be non-negative, got "
                f"{self.perturb_replay_max_iter!r}"
            )

        iterations = tuple(
            sorted({int(value) for value in self.perturb_replay_restart_state_iterations})
        )

        if any(value < 0 for value in iterations):
            raise ValueError(
                "perturbation replay restart-state iterations must be non-negative"
            )

        if iterations and self.perturb_replay_relion_dir is None:
            raise ValueError(
                "perturbation replay restart-state iterations require "
                "perturb_replay_relion_dir"
            )

        object.__setattr__(
            self,
            "perturb_replay_restart_state_iterations",
            iterations,
        )


_CONSISTENCY_CHOICES = {
    "gridding_kernel": ("radial", "separable"),
    "shell_pair_counting": ("relion", "once"),
    "noise_shell_count": ("relion", "summed"),
    "initial_noise_pair_counting": ("relion", "once"),
    "nyquist_column_counting": ("relion", "once"),
    "firstiter_cc_support": ("relion", "gaussian"),
}


@dataclass(frozen=True)
class RelionConsistencyOptions:
    """Opt-in corrections of RELION's own mathematical inconsistencies, for studying their effect.

    The first value of each option is RELION's rule and the default; relax reproduces RELION
    with every option at its default. A route that cannot honour a non-default value refuses it.
    See ``docs/math/relion_consistency_options.md``.
    """

    # Real-space gridding-correction window of the scoring projector and the reconstructions:
    # RELION's radial sinc²(|x| / (pf N)) or the per-axis product (the trilinear kernel's exact
    # transform). "separable" is K=1 single-particle refinement only.
    gridding_kernel: Literal["radial", "separable"] = "radial"
    # How the 3-D shell statistics behind tau2, data-vs-prior, the half-map FSC and the scheduling
    # count Hermitian pairs: RELION's loops over the stored x-half count the pairs of the kx = 0
    # plane twice; "once" counts every pair once. The reconstruction's 1/1000 weight floor (RECOVAR)
    # keeps RELION's counting.
    shell_pair_counting: Literal["relion", "once"] = "relion"
    # The pixel count that turns the noise sums of an expectation into sigma2: RELION counts every
    # shell on the full image, but below the box it sums shell current_size / 2 on a crop that lacks
    # one row, so that shell's sigma2 is 3.6-7.1% low; "summed" counts the pixels that were summed.
    noise_shell_count: Literal["relion", "summed"] = "relion"
    # The start-up noise spectrum averages each image's power over every stored pixel of its FFTW
    # half, which counts the Hermitian pairs of the kx = 0 column twice; every later sigma2 update
    # counts them once. "once" counts every pair once at start-up too. The command applies it (the
    # start-up noise is estimated before the refinement loop).
    initial_noise_pair_counting: Literal["relion", "once"] = "relion"
    # On the Nyquist column of a full-size image both members of each Hermitian pair are stored
    # and RELION's per-image sums (Gaussian score, image power, noise, norm and scale sums,
    # Npix_per_shell) drop neither, unlike the kx = 0 column; "once" counts each pair once.
    nyquist_column_counting: Literal["relion", "once"] = "relion"
    # RELION's first-iteration normalized-CC kernels apply no mask: both kx = 0 copies, the DC
    # pixel and the corners of the cropped rectangle are summed. "gaussian" scores the CC
    # iteration on the support and weights of the Gaussian iterations.
    firstiter_cc_support: Literal["relion", "gaussian"] = "relion"

    def __post_init__(self):
        for name, choices in _CONSISTENCY_CHOICES.items():
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} must be one of {choices}, got {getattr(self, name)!r}")

    def non_default(self) -> dict[str, str]:
        """The options that depart from RELION's rule, by name."""

        return {
            name: getattr(self, name) for name, choices in _CONSISTENCY_CHOICES.items() if getattr(self, name) != choices[0]
        }


# What ``--mode relax`` sets: the consistent value of each option it includes. Add or remove an
# option here; ``relax_mode_consistency`` names the routes on which an entry keeps RELION's value.
RELAX_MODE_CONSISTENCY = {
    "gridding_kernel": "separable",
    "shell_pair_counting": "once",
    "noise_shell_count": "summed",
    "initial_noise_pair_counting": "once",
    "nyquist_column_counting": "once",
    "firstiter_cc_support": "gaussian",
}


def relax_mode_consistency(
    explicit: dict[str, str],
    *,
    n_classes: int,
    has_cc_iteration: bool,
    coarse_engine: str,
    ctf_premultiplied: bool,
) -> tuple[RelionConsistencyOptions, dict[str, str]]:
    """The options of ``--mode relax`` on one route, and why each skipped one keeps RELION's rule.

    ``explicit`` holds the options the user set by name; they override the mode and are not
    skipped (the route guards judge them as they judge any explicit option). Every other entry of
    ``RELAX_MODE_CONSISTENCY`` is applied unless this route does not honour it: the separable
    gridding window in Class3D, the CC support without a CC iteration, the per-image options with
    the ``gemm_dense`` coarse engine, and the Nyquist-column counting on CTF-premultiplied images.
    Routes that honour no option at all (``require_consistency_route``) are the caller's to refuse.
    """

    not_honoured = {}
    if int(n_classes) != 1:
        not_honoured["gridding_kernel"] = "Class3D's tau2 is the power of the radially corrected reference"
    if not has_cc_iteration:
        not_honoured["firstiter_cc_support"] = "this run has no CC iteration (--no-firstiter_cc)"
    if coarse_engine == "gemm_dense":
        for name in ("nyquist_column_counting", "firstiter_cc_support"):
            not_honoured.setdefault(name, "not implemented for the experimental gemm_dense coarse engine")
    if ctf_premultiplied:
        not_honoured["nyquist_column_counting"] = (
            "CTF-premultiplied images: RELION's average CTF^2 divides by the full-image Npix_per_shell"
        )
    skipped = {
        name: reason for name, reason in not_honoured.items() if name in RELAX_MODE_CONSISTENCY and name not in explicit
    }
    chosen = {name: value for name, value in RELAX_MODE_CONSISTENCY.items() if name not in skipped}
    return RelionConsistencyOptions(**{**chosen, **explicit}), skipped


def require_consistency_route(
    options: RefinementOptions, *, subtomograms: bool, several_image_shapes: bool
) -> RelionConsistencyOptions:
    """The run's consistency options, refused on the routes that keep RELION's rules.

    Subtomogram particles and optics groups on several image shapes run their own scorers,
    which no option reaches. Replayed, frozen or swapped RELION state (statistics, references,
    captured projectors) was computed with RELION's rules. The separable gridding window is
    threaded through K=1 only: Class3D's tau2 is the power of the radially corrected reference.
    """

    consistency = options.consistency
    chosen = consistency.non_default()
    if not chosen:
        return consistency
    replay, debug = options.replay, options.debug
    relion_state = (
        options.parity.perturb_replay_relion_dir is not None
        or replay.replay_iteration_overrides is not None
        or replay.final_replay_override is not None
        or replay.final_replay_reference_maps is not None
        or replay.init_refinement_state_fields is not None
        or debug.sealed_sampling_state is not None
        or debug.state_swap_probe is not None
    )
    reasons = [
        reason
        for reason, present in (
            ("subtomogram particles", subtomograms),
            ("optics groups on several image shapes", several_image_shapes),
            ("replayed or frozen RELION state", relion_state),
        )
        if present
    ]
    if reasons:
        raise NotImplementedError(
            f"RELION-consistency options {chosen} are implemented for single-particle refinement of one "
            f"image shape from relax's own state; not with {', '.join(reasons)}"
        )
    if (
        consistency.nyquist_column_counting != "relion" or consistency.firstiter_cc_support != "relion"
    ) and options.adaptive.coarse_engine == "gemm_dense":
        raise NotImplementedError(
            "nyquist_column_counting / firstiter_cc_support are not implemented for the experimental "
            "gemm_dense coarse engine"
        )
    if consistency.firstiter_cc_support != "relion" and not (
        options.parity.emulate_relion_firstiter_cc or options.parity.first_iteration_score_mode == "normalized_cc"
    ):
        raise NotImplementedError(
            f"firstiter_cc_support={consistency.firstiter_cc_support!r} needs the first-iteration "
            "cross-correlation (--firstiter_cc); this run has no CC iteration"
        )
    if consistency.gridding_kernel != "radial" and options.k_class.n_classes != 1:
        raise NotImplementedError(
            f"gridding_kernel={consistency.gridding_kernel!r} is implemented for K=1 single-particle refinement "
            f"only (n_classes={options.k_class.n_classes})"
        )
    return consistency


@dataclass(frozen=True)
class LocalSearchOptions:
    """Local angular-search controls."""

    auto_local_healpix_order: int = LOCAL_SEARCH_HEALPIX_ORDER
    local_search_profile_mode: Literal["auto", "on", "off"] = "auto"
    local_search_translation_prior_mode: str = "coarse"

    def __post_init__(self):
        if self.local_search_profile_mode not in {
            "auto",
            "on",
            "off"
        }:
            raise ValueError(
                "local_search_profile_mode must be "
                "'auto', 'on', or 'off', "
                f"got {self.local_search_profile_mode!r}"
            )


@dataclass(frozen=True)
class ExpectedAccuracyOptions:
    """Half1 oracle inputs for RELION's expected-accuracy calculation."""

    half1_base_order_local: Any | None = None
    half1_trial_order_local: Any | None = None
    half1_optics_group_ids: Any | None = None
    half1_particle_ids: Any | None = None
    half1_ctf_params: Any | None = None
    do_ctf_correction: bool | None = None


@dataclass(frozen=True)
class EngineDebugOptions:
    """Adjoint ablation, intermediate-dump, and test-harness controls."""

    disable_adjoint_y: bool = False
    disable_adjoint_ctf: bool = False
    save_intermediates_dir: str | None = None
    save_intermediates_skip_unregularized: bool = False
    state_swap_probe: str | None = None
    assert_initial_scoring_state_immutable: bool = False
    stop_after_local_search_profile: bool = False
    stop_after_local_search: bool = False
    stop_after_local_search_score_only: bool = False
    sealed_sampling_state: Any | None = None
    sealed_scoring_context: Any | None = None
    expected_accuracy: ExpectedAccuracyOptions = field(default_factory=ExpectedAccuracyOptions)


@dataclass(frozen=True)
class KClassOptions:
    """K-class refinement controls."""

    n_classes: int = 1
    init_class_log_priors: Any | None = None
    # RELION's Class3D from one reference (do_generate_seeds): each particle's class in the first
    # iteration, 0-based, per input particle row (input_particle_table.relion_class3d_seed_classes).
    first_iteration_seed_classes: Any | None = None

    def __post_init__(self):
        if int(self.n_classes) == 1 and self.first_iteration_seed_classes is not None:
            raise ValueError("first_iteration_seed_classes seeds a Class3D start; K=1 has no classes to seed")


@dataclass(frozen=True)
class SymmetryOptions:
    """RELION-compatible proper rotational point group."""

    point_group: str = "C1"

    def __post_init__(self):
        object.__setattr__(
            self,
            "point_group",
            canonicalize_rotational_symmetry(self.point_group),
        )


@dataclass(frozen=True)
class ReplayState:
    """Per-iteration RELION-replay seed state.

    Mirrors what ``refine_single_volume`` takes as ``init_*`` and
    ``replay_iteration_overrides`` so a downstream replay harness can build
    one struct instead of passing many kwargs.
    """

    init_image_corrections: Any | None = None
    init_scale_corrections: Any | None = None
    init_group_ids: Any | None = None
    init_group_count: Any | None = None
    init_direction_prior: Any | None = None
    init_previous_best_translations: Any | None = None
    init_previous_best_rotation_eulers: Any | None = None
    preserve_initial_direction_prior: bool = False
    replay_iteration_overrides: Any | None = None
    final_replay_override: Any | None = None
    final_replay_reference_maps: Any | None = None
    final_replay_source_iteration: int | None = None
    init_reference_real: Any | None = None
    init_refinement_state_fields: Any | None = None
    init_relion_optics_group_count: Any | None = None
    relion_scale_follower_count: int = 0
    relion_scale_follower_owners_by_iteration: Any | None = None
    relion_scale_reduction_mode: str | None = None
    relion_follower_scale_replay: Any | None = None


@dataclass(frozen=True)
class RefinementBatching:
    """Batch sizes the iteration loop hands down to the engines."""

    image_batch_size: int = 500
    rotation_block_size: int = 5000


@dataclass(frozen=True)
class HalfOverlapOptions:
    """Whether the two half-sets' E-steps may run concurrently.

    The halves are independent inside the E-step, so one half's host work can
    be issued while the other's kernels run. Kernels still serialise on JAX's
    single compute stream, so this trades no device order; it only stops the
    host from idling. Off by default: it is a performance experiment, not a
    scientific choice, and it is only engaged when every guard in
    ``iteration_loop`` holds.
    """

    overlap_halves: bool = False


@dataclass(frozen=True)
class CheckpointOptions:
    """RELION's per-iteration run files and ``--continue``.

    ``writer`` is called with an ``IterationSnapshot`` at the end of every numbered
    iteration it reports as due (``relax.refinement.run_files.RunFileWriter``).
    ``resume`` is the snapshot of the last completed iteration of an earlier run;
    ``schedule.init_relion_iteration`` must equal its ``relion_iteration``. See
    ``relax/refinement/iteration_snapshot.py``.
    """

    writer: Any | None = None
    resume: Any | None = None


@dataclass(frozen=True)
class RefinementOptions:
    """Configuration groups consumed by ``refine_single_volume``.

    Passed as the ``options`` argument of ``refine_single_volume``.
    """

    schedule: RefinementSchedule = field(default_factory=RefinementSchedule)
    adaptive: AdaptiveOptions = field(default_factory=AdaptiveOptions)
    parity: RelionParityOptions = field(default_factory=RelionParityOptions)
    local_search: LocalSearchOptions = field(default_factory=LocalSearchOptions)
    k_class: KClassOptions = field(default_factory=KClassOptions)
    replay: ReplayState = field(default_factory=ReplayState)
    debug: EngineDebugOptions = field(default_factory=EngineDebugOptions)
    batching: RefinementBatching = field(default_factory=RefinementBatching)
    overlap: HalfOverlapOptions = field(default_factory=HalfOverlapOptions)
    disc_type: str = "linear_interp"
    # Keep new option groups after the historical positional fields so
    # external positional construction retains its pre-symmetry meaning.
    symmetry: SymmetryOptions = field(default_factory=SymmetryOptions)
    checkpoint: CheckpointOptions = field(default_factory=CheckpointOptions)
    consistency: RelionConsistencyOptions = field(default_factory=RelionConsistencyOptions)


def _validate_relion_healpix_orders(orders, *, max_iter, init_healpix_order, max_healpix_order):
    if orders is None:
        return None
    orders = tuple(int(order) for order in orders)
    if len(orders) < int(max_iter):
        raise ValueError(
            "relion_healpix_orders must provide at least max_iter entries "
            f"({len(orders)} < {int(max_iter)})"
        )
    if any(right < left for left, right in zip(orders, orders[1:])):
        raise ValueError("relion_healpix_orders must be monotone nondecreasing")
    if orders[0] < int(init_healpix_order):
        raise ValueError(
            "relion_healpix_orders cannot coarsen below init_healpix_order "
            f"({orders[0]} < {int(init_healpix_order)})"
        )
    if max_healpix_order is not None and orders[-1] > int(max_healpix_order):
        raise ValueError(
            "relion_healpix_orders exceeds max_healpix_order "
            f"({orders[-1]} > {int(max_healpix_order)})"
        )
    return orders


def with_validated_sampling_schedule(options: RefinementOptions) -> RefinementOptions:
    """Validate explicit sampling schedules when refinement starts.

    Return shallow option/adaptive copies; retain all other payload identities.
    Construction itself deliberately does not run these entry-point checks.
    """
    adaptive = options.adaptive
    if adaptive.relion_current_sizes is not None and len(adaptive.relion_current_sizes) == 0:
        raise ValueError("relion_current_sizes must be non-empty when provided")
    validated_orders = _validate_relion_healpix_orders(
        adaptive.relion_healpix_orders,
        max_iter=options.schedule.max_iter,
        init_healpix_order=options.schedule.init_healpix_order,
        max_healpix_order=options.schedule.max_healpix_order,
    )
    return replace(
        options, adaptive=replace(adaptive, relion_healpix_orders=validated_orders)
    )


# Program-count settings that K=1 Refine3D turns on by default since the resident flip. They
# are exact (the same primitives in the same order), but the stage glue and the local ladder
# are shared with VDAM, whose defaults its own qualification changes, so they are set at the
# K=1 entry points rather than in the shared library defaults.
K1_REFINE3D_ENV_DEFAULTS = {
    "RELAX_EM_JIT_STAGE_GLUE": "1",
    "RELAX_LOCAL_IMAGE_CAPACITY_LADDER": "1",
}


def apply_k1_refine3d_env_defaults() -> None:
    """Set the K=1 Refine3D program defaults; an explicit environment value still wins."""

    for name, value in K1_REFINE3D_ENV_DEFAULTS.items():
        os.environ.setdefault(name, value)


__all__ = [
    "K1_REFINE3D_ENV_DEFAULTS",
    "apply_k1_refine3d_env_defaults",
    "RefinementSchedule",
    "AdaptiveOptions",
    "RelionParityOptions",
    "RelionConsistencyOptions",
    "require_consistency_route",
    "LocalSearchOptions",
    "ExpectedAccuracyOptions",
    "EngineDebugOptions",
    "KClassOptions",
    "SymmetryOptions",
    "ReplayState",
    "RefinementBatching",
    "HalfOverlapOptions",
    "CheckpointOptions",
    "RefinementOptions",
    "with_validated_sampling_schedule",
]
