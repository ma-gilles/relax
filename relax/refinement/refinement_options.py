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

from relax.helpers.convergence import _APPROX_ACC_ROT_CONVERGENCE_ENV, LOCAL_SEARCH_HEALPIX_ORDER
from relax.helpers.env_flags import (
    parse_env_auto_flag,
    parse_env_choice,
    parse_env_flag,
    parse_env_flag_or_false,
    parse_env_optional_flag,
    parse_env_true_flag,
    parse_env_worker_count,
)
from relax.relion.geometry import IMAGE_MASK_EDGE_PIXELS
from relax.symmetry import canonicalize_rotational_symmetry

if TYPE_CHECKING:
    from relax.helpers.dtype_policy import DensePrecisionPolicy


@dataclass(frozen=True, kw_only=True)
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
    # RELION's width_mask_edge (--maskedge, ml_optimiser.cpp:1235-1237): the soft edge, in pixels, of the
    # particle-diameter image mask (ml_optimiser.cpp:3193) and of the reference's solvent flattening.
    width_mask_edge_px: float = IMAGE_MASK_EDGE_PIXELS
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


@dataclass(frozen=True, kw_only=True)
class AdaptiveOptions:
    """Pose-search resolution + adaptive-oversampling knobs."""

    adaptive_oversampling: int = 0
    max_significants: int = 500
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


@dataclass(frozen=True, kw_only=True)
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


@dataclass(frozen=True, kw_only=True)
class LocalSearchOptions:
    """Local angular-search controls."""

    auto_local_healpix_order: int = LOCAL_SEARCH_HEALPIX_ORDER
    # RELION --sigma_ang in degrees: local angular searches from iteration 1 (None: global until the order above).
    sigma_ang_deg: float | None = None
    local_search_profile_mode: Literal["auto", "on", "off"] = "auto"
    local_search_translation_prior_mode: str = "coarse"
    # A probe run of the first local-search iteration: it returns after that iteration's expectation, with no
    # reconstruction (RefinementResult.profile_stop). ``..._profile`` also collects the search's profile;
    # ``..._score_only`` scores without accumulating. They change what the run does, so they are run options,
    # not diagnostics (code rule 9).
    stop_after_local_search_profile: bool = False
    stop_after_local_search: bool = False
    stop_after_local_search_score_only: bool = False

    @property
    def stops_after_local_search(self) -> bool:
        return self.stop_after_local_search or self.stop_after_local_search_profile or self.stop_after_local_search_score_only

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
FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE_ENV = "RELAX_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE"
FINAL_ALL_DATA_DISABLE_REPLAY_LAST_NUMBERED_STATE_ENV = "RELAX_FINAL_ALL_DATA_DISABLE_REPLAY_LAST_NUMBERED_STATE"


@dataclass(frozen=True, kw_only=True)
class FinalPassOptions:
    """Two departures of the final all-data pass from RELION, set by environment variable and read once, when
    the run's options are built. They change what the run does, so they are run options (code rule 9).

    ``after_max_iter`` (``RELAX_FINAL_ALL_DATA_AFTER_MAX_ITER``): K=1 runs the final pass after the last
    numbered iteration even without convergence. ``merged_reference``
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


def bpref_device_signature_target(environ=None) -> tuple[int, int] | None:
    """The numbered (iteration, half), both one-based, whose BPref device signature is captured, or None.

    Armed by ``RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR``; the capture then needs an explicit
    ``RELAX_BPREF_CONTRIBUTION_DUMP_ITERATION`` (> 0) and ``RELAX_BPREF_CONTRIBUTION_DUMP_HALF`` (1 or 2). The
    final all-data pass is never a target. Read once, with the run's options (``DiagnosticEnvironment``).
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
class DiagnosticEnvironment:
    """The refinement's execution switches from the environment, read once, when the run's options are built.

    ``clear_jax_caches_between_iterations``: clear JAX's caches after every numbered iteration.
    ``bpref_device_signature_target``: the numbered (iteration, half), one-based, whose BPref device signature
    a dump captures (``bpref_device_signature_target``), or None; that half plans
    its compact first-iteration batches for the capture. (The other dumps of the run are observers,
    ``relax.diagnostics.observers``; code rule 15.)
    """

    clear_jax_caches_between_iterations: bool = False
    bpref_device_signature_target: tuple[int, int] | None = None

    @classmethod
    def from_environ(cls) -> DiagnosticEnvironment:
        return cls(
            clear_jax_caches_between_iterations=parse_env_true_flag("RELAX_RELION_CLEAR_JAX_CACHES_BETWEEN_ITERS"),
            bpref_device_signature_target=bpref_device_signature_target(),
        )


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


@dataclass(frozen=True, kw_only=True)
class ReconstructionPrograms:
    """Which programs run a RELION reconstruction; no effect on its values.

    ``stable_windows`` (``RELAX_SPARSE_PASS2_RESIDENT_STABLE_WINDOWS``, default on, the resident engine's switch
    too): reconstruct in the full-box class so one compiled program serves every size. ``host_irfft``
    (``RELAX_RELION_HOST_IRFFT``): the padded inverse FFT on the host when True, on the device when False, by
    size and allocator limit when None (``auto``). ``host_fft_workers`` (``RELAX_RELION_HOST_FFT_WORKERS``, else
    ``SLURM_CPUS_PER_TASK``, else 1): that transform's threads.
    """

    stable_windows: bool
    host_irfft: bool | None
    host_fft_workers: int

    @classmethod
    def from_environ(cls) -> ReconstructionPrograms:
        from relax.sparse_pass2.resident_pass2 import _RESIDENT_STABLE_WINDOWS_ENV

        log = logging.getLogger(__name__)
        return cls(
            stable_windows=parse_env_flag(_RESIDENT_STABLE_WINDOWS_ENV, default=True),
            host_irfft=parse_env_auto_flag("RELAX_RELION_HOST_IRFFT", logger=log),
            host_fft_workers=parse_env_worker_count(
                "RELAX_RELION_HOST_FFT_WORKERS", "SLURM_CPUS_PER_TASK", logger=log
            ),
        )


@dataclass(frozen=True, kw_only=True)
class ScoringVariants:
    """Scoring and M-step route variants chosen by environment variable, read once when the run's options are
    built (``from_environ``, code rule 5); nothing below reads the variables again. They change what the run
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
    def from_environ(cls) -> ScoringVariants:
        from relax.dense import scoring_policy as sp

        log = logging.getLogger(__name__)
        k1 = parse_env_optional_flag(sp._K1_RELION_X_HALF_MSTEP_ENV, logger=log, fallback="K=1 RELION x-half M-step default")
        k_class = parse_env_optional_flag(
            sp._K_CLASS_RELION_X_HALF_MSTEP_ENV, logger=log, fallback="K-class RELION x-half M-step default"
        )
        full_parent = parse_env_optional_flag(
            sp._LOCAL_ADAPTIVE_PASS2_FULL_PARENT_ENV, logger=log, fallback="RELION pruned-parent local pass-2 default"
        )
        return cls(
            k1_relion_x_half_mstep=sp._k1_relion_x_half_mstep_default_available() if k1 is None else k1,
            k_class_relion_x_half_mstep=True if k_class is None else k_class,
            local_adaptive_pass2=LocalAdaptivePass2Support(
                full_parent=bool(full_parent),
                rotation_only=parse_env_flag_or_false(sp._LOCAL_ADAPTIVE_PASS2_ROTATION_ONLY_ENV, logger=log),
                denominator_mode=parse_env_choice(
                    sp._LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT_ENV,
                    sp.LOCAL_ADAPTIVE_PASS2_DENOMINATOR_MODES,
                    logger=log,
                    expected="rotation_only or full_parent",
                ),
            ),
            approx_acc_rot_for_convergence=parse_env_true_flag(_APPROX_ACC_ROT_CONVERGENCE_ENV),
            reconstruction=ReconstructionPrograms.from_environ(),
        )


@dataclass(frozen=True, kw_only=True)
class EngineDebugOptions:
    """Adjoint ablation, intermediate-dump, and test-harness controls."""

    disable_adjoint_y: bool = False
    disable_adjoint_ctf: bool = False
    # Read from the environment when the options are built; nothing below reads it again.
    environment: DiagnosticEnvironment = field(default_factory=DiagnosticEnvironment.from_environ)


@dataclass(frozen=True, kw_only=True)
class KClassOptions:
    """K-class refinement controls."""

    n_classes: int = 1
    init_class_log_priors: Any | None = None
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
class ReplayState:
    """The run's initial particle state (poses, corrections, groups, priors, the start-up reference) and the
    follower-scale and frozen-boundary seed state.

    What a run replays from RELION per iteration is not here: it is the replay input source's
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
    init_reference_real: Any | None = None
    init_relion_optics_group_count: Any | None = None


@dataclass(frozen=True, kw_only=True)
class RefinementBatching:
    """Batch sizes the iteration loop hands down to the engines."""

    image_batch_size: int = 500
    rotation_block_size: int = 5000


@dataclass(frozen=True, kw_only=True)
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

    schedule: RefinementSchedule = field(default_factory=RefinementSchedule)
    adaptive: AdaptiveOptions = field(default_factory=AdaptiveOptions)
    parity: RelionParityOptions = field(default_factory=RelionParityOptions)
    local_search: LocalSearchOptions = field(default_factory=LocalSearchOptions)
    k_class: KClassOptions = field(default_factory=KClassOptions)
    replay: ReplayState = field(default_factory=ReplayState)
    debug: EngineDebugOptions = field(default_factory=EngineDebugOptions)
    batching: RefinementBatching = field(default_factory=RefinementBatching)
    overlap: HalfOverlapOptions = field(default_factory=HalfOverlapOptions)
    symmetry: SymmetryOptions = field(default_factory=SymmetryOptions)
    checkpoint: CheckpointOptions = field(default_factory=CheckpointOptions)
    consistency: RelionConsistencyOptions = field(default_factory=RelionConsistencyOptions)
    # Read from the environment when the options are built; nothing below reads it again.
    final_pass: FinalPassOptions = field(default_factory=FinalPassOptions.from_environ)
    solvent: SolventOptions = field(default_factory=SolventOptions)
    expected_accuracy: ExpectedAccuracyOptions = field(default_factory=ExpectedAccuracyOptions)
    # The dense scoring precision, taken when the options are built (code rule 5); see _process_dense_precision.
    precision: DensePrecisionPolicy = field(default_factory=lambda: _process_dense_precision())
    # Read from the environment when the options are built; nothing below reads it again.
    variants: ScoringVariants = field(default_factory=ScoringVariants.from_environ)


def _process_dense_precision() -> DensePrecisionPolicy:
    """The process's dense precision, ``relax.dense.scoring_policy.DENSE_PRECISION``.

    That policy owns the default: ``RELAX_USE_FLOAT64_SCORING`` and ``RELAX_USE_FLOAT64_PROJECTIONS``, read once
    when ``relax.dense.scoring_policy`` is imported. The scoring engines still read it there, so the
    refinement refuses options whose precision differs (``require_process_precision``).
    """
    from relax.dense import scoring_policy

    return scoring_policy.DENSE_PRECISION


def require_process_precision(options: RefinementOptions) -> None:
    """Refuse options whose ``precision`` is not the process's dense precision, which the engines read."""
    process = _process_dense_precision()
    if options.precision != process:
        raise ValueError(
            f"options.precision={options.precision!r} differs from the process's dense precision {process!r} "
            "(relax.dense.scoring_policy.DENSE_PRECISION), which the scoring engines read; build the options "
            "after setting RELAX_USE_FLOAT64_SCORING / RELAX_USE_FLOAT64_PROJECTIONS"
        )


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

