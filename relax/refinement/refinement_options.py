"""Grouped settings accepted by ``refine_single_volume(options=...)``.

The option groups configure scheduling, scoring, replay and diagnostics.

Frozen dataclasses prevent field rebinding. Array and mapping payloads are
still shared objects; freezing does not make them immutable or hashable.
See ``docs/math/relion_refinement_algorithm.md`` for the execution map.
"""

from __future__ import annotations

import logging
import math
import operator
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from relax.reconstruction.accumulator_guard import HALF_ACCUMULATOR_GUARD_ENV, half_accumulator_guard_mode
from relax.reconstruction.volume_solver import ReconstructionPrograms
from relax.refinement.refinement_state import (
    APPROX_ACC_ROT_CONVERGENCE_ENV,
    LOCAL_SEARCH_HEALPIX_ORDER,
    LowPmaxRefineGuard,
)
from relax.relion.geometry import IMAGE_MASK_EDGE_PIXELS
from relax.runtime.engine_tuning import EngineTuning
from relax.runtime.env_flags import (
    parse_env_choice,
    parse_env_flag,
    parse_env_flag_or_false,
    parse_env_float_or_default,
    parse_env_int_or_default,
    parse_env_optional_flag,
    parse_env_true_flag,
    read_env_text,
)
from relax.runtime.scoring_overrides import ScoringOverrides
from relax.sampling.symmetry import canonicalize_rotational_symmetry
from relax.steps.projector_preparation import ProjectorFiles

if TYPE_CHECKING:
    from relax.numerics.precision import DensePrecisionPolicy

# Every option field's default is the value ``relax refine`` (K=1) uses when its flag is not given; a field
# whose value the command computes from the data (init_current_size) has none (code rule 5).

# RELION's --incr_size (ml_optimiser.cpp:1001): Fourier shells beyond the current resolution in the first iteration.
RELION_INCR_SIZE = 10
# The --low_resol_join_halves the RELION GUI passes to auto-refine (pipeline_jobs.cpp:4509); the binary's is -1.
RELION_GUI_LOW_RESOL_JOIN_HALVES_ANGSTROM = 40.0
# The --firstiter_cc_tree_rescore_max_margin "auto" margin of K=1 with --firstiter_cc.
FIRSTITER_CC_TREE_RESCORE_DEFAULT_MAX_MARGIN = 4e-6
# RELION's --maskedge (ml_optimiser.cpp:996): the soft edge of the particle mask, in pixels.
RELION_WIDTH_MASK_EDGE_PX = 5.0


@dataclass(frozen=True, kw_only=True)
class RefinementSchedule:
    """How long the refinement runs and what grid it starts at."""

    # --max_iter: relax refine runs to convergence, nr_iter = RELION's --auto_iter_max (relax class3d: 25).
    max_iter: int = 999
    # The command computes it from the data (the start-up resolution and the box).
    init_current_size: int
    init_healpix_order: int = 2
    # None: RELION's auto-refine has no HEALPix cap; an int is an explicit opt-in cap.
    max_healpix_order: int | None = None
    # --offset_range and --offset_step, in pixels.
    init_translation_range: float = 5.0
    init_translation_step: float = 2.0
    particle_diameter_ang: float | None = None
    # RELION's width_mask_edge (--maskedge, ml_optimiser.cpp:1235-1237): the soft edge, in pixels, of the
    # particle-diameter image mask (ml_optimiser.cpp:3193) and of the reference's solvent flattening.
    width_mask_edge_px: float = IMAGE_MASK_EDGE_PIXELS
    init_relion_iteration: int = 0
    force_max_iter_after_convergence: bool = False
    skip_final_iteration: bool = False

    def numbered_relion_iteration(self, iteration: int) -> int:
        """RELION's number of the run's zero-based ``iteration``: a continuation counts on from its start."""
        return int(self.init_relion_iteration) + int(iteration) + 1


@dataclass(frozen=True, kw_only=True)
class AdaptiveOptions:
    """Pose-search resolution + adaptive-oversampling knobs."""

    adaptive_oversampling: int = 1
    # The active cap: RELION's --maxsig -1 (uncapped) without gradient refinement (relion_active_max_significants).
    max_significants: int = -1
    # gemm_dense is an experimental, no-pruning global route; auto remains the production default.
    coarse_engine: Literal["auto", "gemm_hybrid", "gemm_dense"] = "auto"
    relion_current_sizes: tuple[int, ...] | None = None
    relion_healpix_orders: tuple[int, ...] | None = None
    # RELION --strict_highres_exp in Angstrom: both E-step passes score at the size of this resolution
    # (capped at the current size); the M-step keeps the current size. None: off.
    strict_highres_exp_angstrom: float | None = None

    def __post_init__(self):
        if self.coarse_engine not in {"auto", "gemm_hybrid", "gemm_dense"}:
            raise ValueError(f"unknown coarse engine {self.coarse_engine!r}")
        if self.strict_highres_exp_angstrom is not None and not self.strict_highres_exp_angstrom > 0:
            raise ValueError(f"strict_highres_exp_angstrom must be positive, got {self.strict_highres_exp_angstrom}")


@dataclass(frozen=True, kw_only=True)
class SolventOptions:
    """RELION's --solvent_mask and --solvent_correct_fsc (relax.reconstruction.solvent_mask)."""

    # The user reference mask; None flattens with the particle-diameter sphere.
    mask_path: str | None = None
    # The K=1 half-set FSC is the masked, phase-randomisation corrected FSC; needs mask_path.
    correct_fsc: bool = False
    # rand() values RELION's leader stream gave to the random-halves split before its first phase
    # randomisation (relion_split_draw_count): the input's particle count when relax drew the halves
    # from the input STAR, else 0. A --continue run reseeds and reads its halves, so 0 there too.
    split_draws: int = 0


@dataclass(frozen=True, kw_only=True)
class OpticsGeometry:
    """The data's optics geometry, read from the input particles and the reference: an input of the algorithm,
    not a knob."""

    # Each optics group's image size and pixel size (None: one group at the datasets' own).
    relion_optics_image_sizes: Any | None = None
    relion_optics_pixel_sizes: Any | None = None
    # Each half's per-image optics-group row (0 .. G-1) of a per-group noise table
    # (relax.relion.optics_groups); required when the initial noise has G > 1 rows.
    optics_group_ids_per_half: Any | None = None
    # The reference maps' pixel size (None: the image pixel size).
    relion_model_pixel_size: float | None = None


@dataclass(frozen=True, kw_only=True)
class RelionParityOptions:
    """Knobs that pin RELION numerical behavior."""

    low_resol_join_halves_angstrom: float = RELION_GUI_LOW_RESOL_JOIN_HALVES_ANGSTROM
    # RELION auto-refine's (relax class3d: 4.0).
    tau2_fudge: float = 1.0
    perturb_factor: float = 0.5
    perturb_seed: int | None = None
    # --firstiter_cc (on by default, as the RELION GUI).
    emulate_relion_firstiter_cc: bool = True
    relion_firstiter_ini_high_angstrom: float | None = None
    # --image-fourier-backend auto: RELION's CUDA image preprocessing.
    image_fourier_backend: Literal["host_numpy", "jax_gpu", "relion_cuda"] = "relion_cuda"
    optimizer_random_seed: int | None = None
    use_per_half_mean_variance: bool = False
    preserve_bpref_particle_order: bool = False
    allow_replayed_bpref_particle_order: bool = False
    # K=1 --firstiter_cc: rescore an image's top two coarse CC poses on RELION's coarse tree when their margin is
    # at most this; None is off (the CLI's auto: this margin for K=1 with --firstiter_cc, else off).
    firstiter_cc_tree_rescore_max_margin: float | None = FIRSTITER_CC_TREE_RESCORE_DEFAULT_MAX_MARGIN

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


_CONSISTENCY_CHOICES = {
    "gridding_kernel": ("radial", "separable"),
    "shell_pair_counting": ("relion", "once"),
    "noise_shell_count": ("relion", "summed"),
    "initial_noise_pair_counting": ("relion", "once"),
    "nyquist_column_counting": ("relion", "once"),
    "firstiter_cc_support": ("relion", "gaussian"),
}


@dataclass(frozen=True, kw_only=True)
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
    options: RefinementOptions, *, subtomograms: bool, several_image_shapes: bool, replays_relion_state: bool = False,
) -> RelionConsistencyOptions:
    """The run's consistency options, refused on the routes that keep RELION's rules.

    Subtomogram particles and optics groups on several image shapes run their own scorers,
    which no option reaches. Replayed, frozen or swapped RELION state (statistics, references,
    frozen sampling) was computed with RELION's rules. The separable gridding window is
    threaded through K=1 only: Class3D's tau2 is the power of the radially corrected reference.
    """

    consistency = options.consistency
    chosen = consistency.non_default()
    if not chosen:
        return consistency
    reasons = [
        reason
        for reason, present in (
            ("subtomogram particles", subtomograms),
            ("optics groups on several image shapes", several_image_shapes),
            ("replayed or frozen RELION state", replays_relion_state),
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
    if consistency.firstiter_cc_support != "relion" and not options.parity.emulate_relion_firstiter_cc:
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


@dataclass(frozen=True, kw_only=True)
class LocalSearchOptions:
    """Local angular-search controls."""

    auto_local_healpix_order: int = LOCAL_SEARCH_HEALPIX_ORDER
    # RELION --sigma_ang in degrees: local angular searches from iteration 1 (None: global until the order above).
    sigma_ang_deg: float | None = None
    # RELION --sigma_tilt in degrees: a prior on the tilt angle alone until the order above (None: no prior).
    # Not with --sigma_ang, which sets all three widths and wins, as in relion_refine.
    sigma_tilt_deg: float | None = None
    local_search_profile_mode: Literal["auto", "on", "off"] = "auto"
    # A probe run of the first local-search iteration: it returns after that iteration's expectation, with no
    # reconstruction (RefinementResult.profile_stop). ``..._profile`` also collects the search's profile;
    # ``..._score_only`` scores without accumulating. They change what the run does, so they are run options,
    # not diagnostics (code rule 9).
    stop_after_local_search_profile: bool = False
    stop_after_local_search: bool = False
    stop_after_local_search_score_only: bool = False

    def tilt_prior_deg(self, do_local_search: bool) -> float:
        """Width in degrees of RELION's ``--sigma_tilt`` prior on the tilt angle alone this iteration; 0: none.

        relion_refine holds ``sigma2_rot == sigma2_psi == 0 < sigma2_tilt`` from parseInitial until
        updateAngularSampling reaches ``--auto_local_healpix_order``, where all three widths become twice
        the angular step (ml_optimiser.cpp:9923-9931): the local search (``do_local_search``, the run
        state's) then replaces the tilt prior for good. ``--sigma_ang`` sets all three widths and the tilt
        width is not read (ml_optimiser.cpp:739-752). The width is run-constant, so it is an option and not
        run state; degrees as given, since RELION compares angular distances in degrees with three widths
        (docs/math/relion_refinement_algorithm.md, "A prior on the tilt angle alone").
        """
        if self.sigma_tilt_deg is None or self.sigma_ang_deg is not None or do_local_search:
            return 0.0
        return self.sigma_tilt_deg

    @property
    def stops_after_local_search(self) -> bool:
        return self.stop_after_local_search or self.stop_after_local_search_profile or self.stop_after_local_search_score_only

    def collects_profile(self, observer_collects: bool) -> bool:
        """Whether local searches collect their profile: always for a profile probe, else by the mode, where
        "auto" follows the run's observer (``observer_collects``)."""
        collect = observer_collects if self.local_search_profile_mode == "auto" else self.local_search_profile_mode == "on"
        if self.stop_after_local_search_profile:
            collect = True
        return collect

    def __post_init__(self):
        if self.sigma_ang_deg is not None and not self.sigma_ang_deg > 0:
            raise ValueError(f"sigma_ang_deg must be positive, got {self.sigma_ang_deg}")
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


class InitialSampling(NamedTuple):
    """Startup pass orders and the resolved refinement limit with its source."""

    coarse_order: int
    fine_order: int
    max_order: int | None
    max_order_source: str


class RestartProvenance(NamedTuple):
    """The numbered iterations whose sampling-perturbation state a replay restarts from (sorted, unique),
    with the provenance file that justifies them and its SHA-256; empty and None without a restart."""

    iterations: tuple[int, ...]
    path: Path | None
    sha256: str | None


@dataclass(frozen=True)
class JobDefaults:
    """The arguments whose default depends on the job type, resolved (``--max_iter``,
    ``--image-fourier-backend``, ``--apply-initial-lowpass``)."""

    max_iter: int
    image_fourier_backend: str
    apply_initial_lowpass: bool


@dataclass(frozen=True)
class RandomSeed:
    """The run's random seed (RELION's ``--random_seed``) and where it came from (stored in the archive)."""

    value: int
    source: str


@dataclass(frozen=True, kw_only=True)
class ExpectedAccuracyOptions:
    """Half 1's particle order, optics groups, ids and CTFs for RELION's expected-accuracy estimate.

    The estimate sets the angular and translational sampling, so these are algorithm inputs (code rule 9).
    """

    half1_base_order_local: Any | None = None
    half1_trial_order_local: Any | None = None
    half1_optics_group_ids: Any | None = None
    half1_particle_ids: Any | None = None
    half1_ctf_params: Any | None = None
    do_ctf_correction: bool | None = None


FINAL_ALL_DATA_AFTER_MAX_ITER_ENV = "RELAX_FINAL_ALL_DATA_AFTER_MAX_ITER"
FINAL_ALL_DATA_USE_MERGED_REFERENCE_ENV = "RELAX_FINAL_ALL_DATA_USE_MERGED_REFERENCE"


@dataclass(frozen=True, kw_only=True)
class FinalPassOptions:
    """Two departures of the final all-data pass from RELION, set by environment variable and read once, by
    ``from_environ``, which the command calls. They change what the run does, so they are run options (code rule 9).

    ``after_max_iter`` (``RELAX_FINAL_ALL_DATA_AFTER_MAX_ITER``): K=1 runs the final pass after the last
    numbered iteration even without convergence; Class3D refuses it at set-up
    (``require_final_pass_route``). ``merged_reference``
    (``RELAX_FINAL_ALL_DATA_USE_MERGED_REFERENCE``): its K=1 E-step scores both halves against the merged map.
    """

    after_max_iter: bool = False
    merged_reference: bool = False

    @classmethod
    def from_environ(cls) -> FinalPassOptions:
        return cls(
            after_max_iter=parse_env_flag_or_false(FINAL_ALL_DATA_AFTER_MAX_ITER_ENV, logger=logging.getLogger(__name__)),
            merged_reference=parse_env_true_flag(FINAL_ALL_DATA_USE_MERGED_REFERENCE_ENV),
        )


def require_final_pass_route(options: RefinementOptions) -> None:
    """Refuse ``final_pass.after_max_iter`` for Class3D, which runs the final pass only after convergence."""
    if options.final_pass.after_max_iter and options.k_class.n_classes > 1:
        raise NotImplementedError(
            f"{FINAL_ALL_DATA_AFTER_MAX_ITER_ENV} (the final all-data pass after the iteration cap, without "
            f"convergence) is implemented for K=1 only; this run has n_classes={options.k_class.n_classes}, and "
            f"Class3D runs the final pass only after convergence. Unset {FINAL_ALL_DATA_AFTER_MAX_ITER_ENV}."
        )


def bpref_device_signature_target(environ=None) -> tuple[int, int] | None:
    """The numbered (iteration, half), both one-based, whose BPref device signature is captured, or None.

    Armed by ``RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR``; the capture then needs an explicit
    ``RELAX_BPREF_CONTRIBUTION_DUMP_ITERATION`` (> 0) and ``RELAX_BPREF_CONTRIBUTION_DUMP_HALF`` (1 or 2). The
    final all-data pass is never a target. Read once, by ``ExecutionOptions.from_environ``, which the command calls.
    """

    env = os.environ if environ is None else environ
    if not str(env.get("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "")).strip():
        return None
    raw_iteration = str(env.get("RELAX_BPREF_CONTRIBUTION_DUMP_ITERATION", "")).strip()
    raw_half = str(env.get("RELAX_BPREF_CONTRIBUTION_DUMP_HALF", "")).strip()
    if not raw_iteration or not raw_half:
        raise RuntimeError(
            "Scoped BPref device capture requires explicit positive "
            "RELAX_BPREF_CONTRIBUTION_DUMP_ITERATION and half 1 or 2"
        )
    try:
        target_iteration = int(raw_iteration)
        target_half = int(raw_half)
    except ValueError as exc:
        raise ValueError("BPref device capture iteration/half targets must be integers") from exc
    if target_iteration <= 0 or target_half not in {1, 2}:
        raise ValueError("BPref device capture requires target iteration > 0 and half 1 or 2")
    return target_iteration, target_half


@dataclass(frozen=True, kw_only=True)
class LocalAdaptivePass2Support:
    """The support of adaptive local pass 2: RELION's pruned parent, all parent samples (``full_parent``), or
    the significant parent rotations with all their translations (``rotation_only``); and a broader support of
    its denominator (``denominator_mode``), or None."""

    full_parent: bool
    rotation_only: bool
    denominator_mode: Literal["full_parent", "rotation_only"] | None

    def at(self, oversampling_order: int) -> LocalAdaptivePass2Support:
        """This support where adaptive pass 2 runs (oversampling above 0); RELION's pruned parent elsewhere."""
        if oversampling_order > 0:
            return self
        return LocalAdaptivePass2Support(full_parent=False, rotation_only=False, denominator_mode=None)

    @property
    def parent_mode(self) -> str:
        """The profile's name of this support: ``full_parent``, ``significant_rotation_full_translation``
        or ``pruned_parent``."""
        if self.full_parent:
            return "full_parent"
        if self.rotation_only:
            return "significant_rotation_full_translation"
        return "pruned_parent"


@dataclass(frozen=True, kw_only=True)
class ScoringVariants:
    """Scoring and M-step route variants chosen by environment variable, read once by ``from_environ``, which the
    command calls (code rule 5); nothing below reads the variables again. They change what the run
    computes, so they are run options, not diagnostics (code rule 9).

    ``k1_relion_x_half_mstep`` / ``k_class_relion_x_half_mstep`` (``RELAX_K1_RELION_X_HALF_MSTEP``,
    ``RELAX_K_CLASS_RELION_X_HALF_MSTEP``): RELION's x-half BPref M-step accumulators; K=1's default is on where
    custom CUDA runs on a GPU, K-class's is on. ``local_adaptive_pass2`` (``RELAX_LOCAL_ADAPTIVE_PASS2_FULL_PARENT``,
    ``_ROTATION_ONLY``, ``_DENOMINATOR_SUPPORT``; default RELION's pruned parent). ``approx_acc_rot_for_convergence``
    (``RELAX_EM_USE_APPROX_ACC_ROT_FOR_CONVERGENCE``): the support-width angular accuracy gates convergence.
    ``reconstruction``: the M-step reconstruction's programs (ReconstructionPrograms).
    """

    k1_relion_x_half_mstep: bool
    k_class_relion_x_half_mstep: bool
    local_adaptive_pass2: LocalAdaptivePass2Support
    approx_acc_rot_for_convergence: bool
    reconstruction: ReconstructionPrograms

    def relion_x_half_mstep(self, *, k_class: bool) -> bool:
        """Whether this run's K=1 (or, with ``k_class``, K-class) M-step uses RELION's x-half accumulators."""
        return self.k_class_relion_x_half_mstep if k_class else self.k1_relion_x_half_mstep

    @classmethod
    def from_environ(cls, *, reconstruction: ReconstructionPrograms) -> ScoringVariants:
        from relax.steps import scoring_policy as sp

        log = logging.getLogger(__name__)
        k1 = parse_env_optional_flag(sp.K1_RELION_X_HALF_MSTEP_ENV, logger=log, fallback="K=1 RELION x-half M-step default")
        k_class = parse_env_optional_flag(
            sp.K_CLASS_RELION_X_HALF_MSTEP_ENV, logger=log, fallback="K-class RELION x-half M-step default"
        )
        full_parent = parse_env_optional_flag(
            sp.LOCAL_ADAPTIVE_PASS2_FULL_PARENT_ENV, logger=log, fallback="RELION pruned-parent local pass-2 default"
        )
        return cls(
            k1_relion_x_half_mstep=sp.k1_relion_x_half_mstep_default_available() if k1 is None else k1,
            k_class_relion_x_half_mstep=True if k_class is None else k_class,
            local_adaptive_pass2=LocalAdaptivePass2Support(
                full_parent=bool(full_parent),
                rotation_only=parse_env_flag_or_false(sp.LOCAL_ADAPTIVE_PASS2_ROTATION_ONLY_ENV, logger=log),
                denominator_mode=parse_env_choice(
                    sp.LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT_ENV,
                    sp.LOCAL_ADAPTIVE_PASS2_DENOMINATOR_MODES,
                    logger=log,
                    expected="rotation_only or full_parent",
                ),
            ),
            approx_acc_rot_for_convergence=parse_env_true_flag(APPROX_ACC_ROT_CONVERGENCE_ENV),
            reconstruction=reconstruction,
        )


@dataclass(frozen=True, kw_only=True)
class KClassOptions:
    """K-class refinement controls."""

    n_classes: int = 1
    # RELION's Class3D from one reference (do_generate_seeds): each particle's class in the first
    # iteration, 0-based, per input particle row (input_particle_table.relion_class3d_seed_classes).
    first_iteration_seed_classes: Any | None = None
    # RELION --skip_align: classify at each particle's stored pose, with no pose search.
    skip_align: bool = False

    def __post_init__(self):
        # The class count enters relax here (the command, scripts, a resumed snapshot): an int from here on.
        object.__setattr__(self, "n_classes", operator.index(self.n_classes))
        if self.n_classes == 1 and self.skip_align:
            raise ValueError("--skip_align classifies; K=1 has nothing to classify")
        if self.n_classes == 1 and self.first_iteration_seed_classes is not None:
            raise ValueError("first_iteration_seed_classes seeds a Class3D start; K=1 has no classes to seed")


@dataclass(frozen=True, kw_only=True)
class SymmetryOptions:
    """RELION-compatible proper rotational point group."""

    point_group: str = "C1"

    def __post_init__(self):
        object.__setattr__(
            self,
            "point_group",
            canonicalize_rotational_symmetry(self.point_group),
        )


@dataclass(frozen=True, kw_only=True)
class StartState:
    """The state a run starts from: the particles' poses, corrections, groups and priors, and the
    follower-scale and frozen-boundary seed state.

    What a run replays from RELION per iteration is the replay input source's
    (``relax.parity.relion_replay_source.RelionReplay``, code rule 15).
    """

    init_image_corrections: Any | None = None
    init_scale_corrections: Any | None = None
    init_group_ids: Any | None = None
    init_group_count: Any | None = None
    init_direction_prior: Any | None = None
    init_previous_best_translations: Any | None = None
    init_previous_best_rotation_eulers: Any | None = None
    # Per half, [N, 3] degrees: the input STAR's rlnAngle{Rot,Tilt,Psi}Prior, NaN where absent (None: no column).
    init_angle_priors: Any | None = None
    init_relion_optics_group_count: Any | None = None
    # The translation prior width the run starts with (--offset_sigma_angstrom, a RELION-seeded start's or a
    # frozen boundary's per-half pair).
    init_translation_sigma_angstrom: Any = 10.0
    # RELION's --ini_high of a fresh run, which seeds the iteration-0 current resolution
    # (ml_optimiser.cpp:6768); None leaves it unset, as RELION without --ini_high.
    ini_high_angstrom: float | None = None
    # RELION-unit data_vs_prior spectrum of the start-up model (initialiseDataVersusPrior);
    # it selects the iteration-1 scale-correction shells (ml_optimiser.cpp:10473). It is half set 1's; half set 2's
    # own (its reference and particle count, as its RELION rank computes it) gates half set 2's iteration-1 shells.
    init_data_vs_prior: Any | None = None
    init_data_vs_prior_half2: Any | None = None


CLEAR_JAX_CACHES_BETWEEN_ITERATIONS_ENV = "RELAX_RELION_CLEAR_JAX_CACHES_BETWEEN_ITERS"


@dataclass(frozen=True, kw_only=True)
class ExecutionOptions:
    """How the run executes, not what it computes: the batch sizes the iteration loop hands down to the
    engines, whether JAX's caches are cleared after every
    numbered iteration, and which numbered half a BPref device-signature dump captures.

    ``clear_jax_caches_between_iterations``
    (``RELAX_RELION_CLEAR_JAX_CACHES_BETWEEN_ITERS``) and ``bpref_device_signature_target`` (the numbered
    (iteration, half), one-based, whose BPref device signature a dump captures, or None; that half plans its
    compact first-iteration batches for the capture; see ``bpref_device_signature_target``) come from the
    environment, read by ``from_environ``, which the command calls. (The other dumps of the run are observers,
    ``relax.diagnostics.observers``; code rule 15.)
    """

    image_batch_size: int = 500
    rotation_block_size: int = 40000
    clear_jax_caches_between_iterations: bool = False
    bpref_device_signature_target: tuple[int, int] | None = None

    @classmethod
    def from_environ(cls, **fields) -> ExecutionOptions:
        """``fields`` with the cache switch and the BPref capture target read from the environment."""
        return cls(
            **fields,
            clear_jax_caches_between_iterations=parse_env_true_flag(CLEAR_JAX_CACHES_BETWEEN_ITERATIONS_ENV),
            bpref_device_signature_target=bpref_device_signature_target(),
        )


FIRSTITER_FUSED_BPREF_ENV = "RELAX_K1_RELION_FIRSTITER_FUSED_BPREF"
DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS_ENV = "RELAX_DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS"


def _diagnostic_float64_pass2_iterations() -> frozenset[int]:
    """The RELION iterations whose pass 2 scores in float64: RELAX_DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS,
    comma-separated integers (blank tokens skipped); unset or blank is none; anything else is refused."""
    raw = read_env_text(DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS_ENV, "")
    if not raw.strip():
        return frozenset()
    try:
        return frozenset(int(token.strip()) for token in raw.split(",") if token.strip())
    except ValueError as exc:
        raise ValueError(f"{DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS_ENV} must be comma-separated integers") from exc


def _optional_env_flag(name: str) -> bool | None:
    """None when ``name`` is unset or blank (the route decides), else ``env_flags.parse_env_flag``: 0/false/no/off
    off, any other text on."""
    return None if not read_env_text(name, "").strip() else parse_env_flag(name)


LOW_PMAX_REFINE_GUARD_ENV = "RELAX_EM_LOW_PMAX_REFINE_GUARD"


def _low_pmax_refine_guard() -> LowPmaxRefineGuard | None:
    """The diffuse-posterior guard: None unless RELAX_EM_LOW_PMAX_REFINE_GUARD is a true token
    (``parse_env_true_flag``); then RELAX_EM_LOW_PMAX_REFINE_MAX_AVE_PMAX (0.15), RELAX_EM_LOW_PMAX_REFINE_MIN_RES_STALL
    (3; an invalid number warns and is the default) and RELAX_EM_LOW_PMAX_REFINE_REQUIRE_LOCAL (true; unset, blank
    or invalid is true, an invalid token warning: ``parse_env_optional_flag``)."""
    if not parse_env_true_flag(LOW_PMAX_REFINE_GUARD_ENV):
        return None
    log = logging.getLogger(__name__)
    return LowPmaxRefineGuard(
        require_local_search=parse_env_optional_flag("RELAX_EM_LOW_PMAX_REFINE_REQUIRE_LOCAL", logger=log, fallback="True")
        is not False,
        max_ave_pmax=parse_env_float_or_default("RELAX_EM_LOW_PMAX_REFINE_MAX_AVE_PMAX", 0.15, logger=log),
        min_res_stall=parse_env_int_or_default("RELAX_EM_LOW_PMAX_REFINE_MIN_RES_STALL", 3, logger=log),
    )


@dataclass(frozen=True, kw_only=True)
class RefinementEnvironment:
    """The controller's switches of the process environment, read once by :meth:`from_environ`, which the
    command calls when it builds the run's options (code rule 5); nothing below reads these variables again.
    The default instance is the unset environment.

    ``half_accumulator_guard``: the mode of the finite guard of the half accumulators before their join, "raise"
    (the default), "warn" or "off", from RELAX_EM_BPREF_FINITE_GUARD (1/true/yes/on/raise or unset/blank:
    raise; 0/false/no/off: off; warn; anything else refused) demoted to "warn" by RELAX_EM_FINITE_CHECK_WARN
    (``env_flags.parse_env_true_flag``); see ``accumulator_guard.half_accumulator_guard_mode``.
    ``firstiter_fused_bpref`` (RELAX_K1_RELION_FIRSTITER_FUSED_BPREF, :func:`_optional_env_flag`; None: the
    first-iteration BPref takes its fused kernel wherever it is supported; False: never; True: required there,
    refused elsewhere; ``firstiter_bpref._relion_firstiter_fused_bpref_enabled``). ``diagnostic_float64_pass2_iterations``
    (RELAX_DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS, :func:`_diagnostic_float64_pass2_iterations`): the RELION
    iterations whose pass 2 scores and is planned in float64. ``low_pmax_refine_guard``
    (:func:`_low_pmax_refine_guard`; None: off) holds the angular refinement while the posterior is diffuse.
    ``projector_files`` (``steps.projector_preparation.ProjectorFiles``): the projector slab cache
    (RELAX_RELION_PROJECTOR_CACHE_DIR) and dump (RELAX_RELION_PROJECTOR_DUMP_DIR).
    ``engine`` (``relax.runtime.engine_tuning.EngineTuning``): how the scoring and reconstruction engines execute.
    ``scoring`` (``relax.runtime.scoring_overrides.ScoringOverrides``): the diagnostic overrides that change values.
    """

    half_accumulator_guard: str = "raise"
    firstiter_fused_bpref: bool | None = None
    diagnostic_float64_pass2_iterations: frozenset[int] = frozenset()
    low_pmax_refine_guard: LowPmaxRefineGuard | None = None
    projector_files: ProjectorFiles = ProjectorFiles()
    engine: EngineTuning = EngineTuning()
    scoring: ScoringOverrides = ScoringOverrides()

    @classmethod
    def from_environ(cls, *, engine: EngineTuning, scoring: ScoringOverrides) -> RefinementEnvironment:
        return cls(
            half_accumulator_guard=half_accumulator_guard_mode(
                read_env_text(HALF_ACCUMULATOR_GUARD_ENV),
                warn_only=parse_env_true_flag("RELAX_EM_FINITE_CHECK_WARN"),  # engine_tuning.FINITE_CHECK_WARN_ENV
            ),
            firstiter_fused_bpref=_optional_env_flag(FIRSTITER_FUSED_BPREF_ENV),
            diagnostic_float64_pass2_iterations=_diagnostic_float64_pass2_iterations(),
            low_pmax_refine_guard=_low_pmax_refine_guard(),
            projector_files=ProjectorFiles(
                cache_dir=read_env_text("RELAX_RELION_PROJECTOR_CACHE_DIR", "").strip() or None,
                dump_dir=read_env_text("RELAX_RELION_PROJECTOR_DUMP_DIR") or None,
            ),
            engine=engine,
            scoring=scoring,
        )


@dataclass(frozen=True, kw_only=True)
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


@dataclass(frozen=True, kw_only=True)
class RefinementOptions:
    """Configuration groups consumed by ``refine_single_volume``.

    Passed as the ``options`` argument of ``refine_single_volume``.
    """

    schedule: RefinementSchedule
    adaptive: AdaptiveOptions = field(default_factory=AdaptiveOptions)
    parity: RelionParityOptions = field(default_factory=RelionParityOptions)
    optics_geometry: OpticsGeometry = field(default_factory=OpticsGeometry)
    local_search: LocalSearchOptions = field(default_factory=LocalSearchOptions)
    k_class: KClassOptions = field(default_factory=KClassOptions)
    start: StartState = field(default_factory=StartState)
    execution: ExecutionOptions = field(default_factory=ExecutionOptions)
    symmetry: SymmetryOptions = field(default_factory=SymmetryOptions)
    checkpoint: CheckpointOptions = field(default_factory=CheckpointOptions)
    consistency: RelionConsistencyOptions = field(default_factory=RelionConsistencyOptions)
    # The command reads it from the environment (FinalPassOptions.from_environ); nothing below reads it again.
    final_pass: FinalPassOptions = field(default_factory=FinalPassOptions)
    # The command reads it from the environment (RefinementEnvironment.from_environ); nothing below reads it again.
    environment: RefinementEnvironment = field(default_factory=RefinementEnvironment)
    solvent: SolventOptions = field(default_factory=SolventOptions)
    expected_accuracy: ExpectedAccuracyOptions = field(default_factory=ExpectedAccuracyOptions)
    # The dense scoring precision: the command reads it (DensePrecisionPolicy.from_environ) and passes it.
    precision: DensePrecisionPolicy
    # The command reads it from the environment (ScoringVariants.from_environ); nothing below reads it again. No
    # default: K=1's x-half M-step default depends on the device, so no plain record equals the unset environment.
    variants: ScoringVariants


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
