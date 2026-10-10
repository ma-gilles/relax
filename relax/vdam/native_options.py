"""Shared run defaults and native InitialModel options.

Sampling and continuation read these records without importing the driver.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from relax.io.particle_io import DEFAULT_KEEP_FREE_SCRATCH_GB
from relax.sampling.symmetry import is_identity_symmetry
from relax.vdam.schedules import (
    DEFAULT_GRAD_EM_ITERS,
    DEFAULT_GRAD_FIN_FRAC,
    DEFAULT_GRAD_INI_FRAC,
    DEFAULT_GRAD_MU,
    DEFAULT_STEPSIZE_3D_INITIAL_MODEL,
    GUI_DEFAULT_NR_CLASSES,
    GUI_DEFAULT_NR_ITER,
    GUI_DEFAULT_TAU2_FUDGE,
    compute_phase_lengths,
    compute_tau2_fudge,
    default_subset_sizes_for_3d_initial_model,
)


@dataclass(frozen=True)
class VdamPilotControls:
    """Opt-in caps for the native VDAM comparison arm of the PPCA pilots; ``None`` is native VDAM, unchanged.

    The matched VDAM runs of ``docs/math/vdam_ppca_algorithm.md`` fix the non-final subset size and cap the
    Fourier radius and HEALPix order. Native frequency and angular adaptation stays active within the caps.
    """

    stochastic_batch_size: int | None = None
    max_fourier_radius: int | None = None
    max_healpix_order: int | None = None
    stop_file: str | None = None

    def __post_init__(self):
        if self.stochastic_batch_size is not None and self.stochastic_batch_size < 1:
            raise ValueError("stochastic_batch_size must be positive")
        if self.max_fourier_radius is not None and self.max_fourier_radius < 1:
            raise ValueError("max_fourier_radius must be positive")

    @classmethod
    def from_values(cls, stochastic_batch_size=None, max_fourier_radius=None, max_healpix_order=None, stop_file=None):
        """The controls, or ``None`` (native VDAM) when every cap is unset."""
        values = (stochastic_batch_size, max_fourier_radius, max_healpix_order, stop_file)
        return None if all(value is None for value in values) else cls(*values)

    def validate(self, *, initial_healpix_order: int) -> None:
        if self.max_healpix_order is not None and self.max_healpix_order < initial_healpix_order:
            raise ValueError("max_healpix_order must not be below the initial order")

    def subset_sizes(self, n_images: int) -> tuple[int, int]:
        """InitialModel's gradient subset sizes; a fixed non-final subset replaces both."""
        if self.stochastic_batch_size is None:
            return default_subset_sizes_for_3d_initial_model(int(n_images))
        size = min(int(self.stochastic_batch_size), int(n_images))
        return size, size

    def cap_current_size(self, current_size: int) -> int:
        if self.max_fourier_radius is None:
            return current_size
        return min(current_size, 2 * int(self.max_fourier_radius))

    def stop_requested(self) -> bool:
        return self.stop_file is not None and Path(self.stop_file).exists()

    def check_completed(self, final_iteration: int, nr_iter: int) -> None:
        """Refuse final outputs when the stop file ended the run early."""
        if self.stop_requested() and final_iteration < nr_iter:
            raise RuntimeError(f"VDAM stopped at saved iteration {final_iteration} before final output")


@dataclass(frozen=True)
class VdamEnvironment:
    """The InitialModel's switches of the process environment, read once when the options are built.

    :meth:`from_environ` reads them; the default instance has every switch off.
    ``clear_jax_caches_per_iteration`` (RELAX_CLEAR_JAX_CACHES_PER_ITER: 1, true or TRUE) releases JAX's
    buffers after each iteration (CUFFT_ALLOC_FAILED at 50k x 256^2); ``skip_expected_accuracy`` and
    ``isolate_expected_accuracy`` (RELAX_INITIALMODEL_SKIP_EXPECTED_ACCURACY and
    RELAX_INITIALMODEL_EXPECTED_ACCURACY_SUBPROCESS: 0 or 1) skip the estimate or run it in a spawned process.
    Two engine diagnostics: ``adaptive_fraction`` (RELAX_ADAPTIVE_FRACTION, a float; ``None`` when unset or
    empty keeps the engine's RELION 0.999) and ``subtract_projected_reference`` (``False`` when
    RELAX_DISABLE_SUBTRACT_PROJECTED_REFERENCE has any value: back-project the images, not the residuals).
    """

    clear_jax_caches_per_iteration: bool = False
    skip_expected_accuracy: bool = False
    isolate_expected_accuracy: bool = False
    adaptive_fraction: float | None = None
    subtract_projected_reference: bool = True

    @classmethod
    def from_environ(cls, environ=None) -> VdamEnvironment:
        env = os.environ if environ is None else environ
        if env.get("RELAX_USE_FLOAT64_SCORING", "").strip().lower() in {"1", "true", "yes", "on"}:
            raise NotImplementedError("RELAX_USE_FLOAT64_SCORING: InitialModel's resident E-step scores in float32 only")

        def strict(name):
            if (value := env.get(name, "").strip()) not in {"", "0", "1"}:
                raise ValueError(f"{name} must be 0 or 1")
            return value == "1"

        return cls(env.get("RELAX_CLEAR_JAX_CACHES_PER_ITER", "") in ("1", "true", "TRUE"),
                   strict("RELAX_INITIALMODEL_SKIP_EXPECTED_ACCURACY"),
                   strict("RELAX_INITIALMODEL_EXPECTED_ACCURACY_SUBPROCESS"),
                   float(adaptive_fraction) if (adaptive_fraction := env.get("RELAX_ADAPTIVE_FRACTION")) else None,
                   not env.get("RELAX_DISABLE_SUBTRACT_PROJECTED_REFERENCE"))


@dataclass(frozen=True, kw_only=True)
class InitialModelDefaults:
    """Run settings shared by the native driver and the command/GUI defaults."""

    nr_iter: int = GUI_DEFAULT_NR_ITER
    grad_write_iter: int = 10
    nr_classes: int = GUI_DEFAULT_NR_CLASSES
    tau2_fudge: float = GUI_DEFAULT_TAU2_FUDGE
    sym_name: str = "C1"
    do_run_C1: bool = True
    particle_diameter: float = 200.0
    do_solvent: bool = True
    do_zero_mask: bool = True
    do_ctf_correction: bool = True
    # relion_refine's --random_seed default: -1 takes the time (the GUI passes none).
    random_seed: int = -1
    healpix_order: int = 1
    oversampling: int = 1
    offset_range_px: float = 6.0
    offset_step_px: float = 2.0
    perturbation_factor: float = 0.5
    image_batch_size: int = 500
    rotation_block_size: int = 5000
    pilot_controls: VdamPilotControls | None = None
    pass2_engine: str = "auto"
    coarse_engine: Literal["auto", "gemm_hybrid", "gemm_dense"] = "auto"
    bootstrap_min_particles: int = 1000
    sigma2_min_particles: int = 1000
    padding_factor: int = 1
    # RELION GUI defaults: no pre-read, no scratch copy (relax.io.particle_io).
    preread_images: bool = False
    scratch_dir: str = ""
    keep_free_scratch_gb: float = DEFAULT_KEEP_FREE_SCRATCH_GB
    write_iter_artifacts: bool = True
    random_perturbation: float | None = None
    translation_sigma_angstrom: float | None = None
    grad_ini_frac: float = DEFAULT_GRAD_INI_FRAC
    grad_fin_frac: float = DEFAULT_GRAD_FIN_FRAC
    grad_em_iters: int = DEFAULT_GRAD_EM_ITERS
    stepsize: float = DEFAULT_STEPSIZE_3D_INITIAL_MODEL
    mu: float = DEFAULT_GRAD_MU
    optimizer: Literal["vdam", "momentum_sgd"] = "vdam"
    sgd_learning_rate: float = 1.0
    fourier_radius_schedule: tuple[int, ...] | None = None
    fixed_healpix_order: int | None = None
    stochastic_all_iterations: bool = False
    uniform_class_direction_prior: bool = False


@dataclass(frozen=True, kw_only=True)
class NativeInitialModelOptions(InitialModelDefaults):
    """Options for a native InitialModel run; defaults mirror the GUI command."""

    fn_img: str
    # RELION 5 subtomogram 2D stacks (``--ios``): fn_img is the particles STAR, this the tomograms STAR.
    fn_tomograms: str | None = None
    outputname: str = "ab_initio/run"
    width_mask_edge_px: float = 5.0
    image_fourier_backend: str = "host_numpy"
    # Production EM is float32; float64 is the diagnostic reference (native replays and dumps).
    mstep_compute_dtype: Literal["float32", "float64"] = "float32"
    datadir: str | None = None
    strip_prefix: str | None = None
    # Diagnostic-only, one-next-iteration restart from a native RELION VDAM
    # optimiser.  This is deliberately not a general production continuation
    # surface: the caller must also stop at checkpoint_iteration + 1.
    diagnostic_continue_optimiser: str | None = None
    diagnostic_stop_after_iteration: int | None = None
    diagnostic_continue_input_order: bool = False
    # Read from the process environment when the options are built; not part of the saved options file.
    environment: VdamEnvironment = field(default_factory=VdamEnvironment.from_environ)

    def validate_run(self) -> None:
        """Check supported settings before loading particles or creating run state."""
        if self.mstep_compute_dtype not in {"float32", "float64"}:
            raise ValueError(f"Unknown mstep_compute_dtype: {self.mstep_compute_dtype!r}")
        if self.nr_classes < 1:
            raise ValueError("nr_classes must be >= 1")
        if self.nr_iter < 1:
            raise ValueError("nr_iter must be >= 1")
        if self.optimizer not in {"vdam", "momentum_sgd"}:
            raise ValueError(f"unknown InitialModel optimizer: {self.optimizer!r}")
        if self.optimizer == "momentum_sgd" and self.oversampling != 0:
            raise ValueError("momentum_sgd requires --oversampling 0 (one coarse pose grid)")
        if self.optimizer == "momentum_sgd" and self.mstep_compute_dtype != "float32":
            raise ValueError("momentum_sgd is a production float32 optimizer")
        if self.optimizer == "momentum_sgd" and self.fourier_radius_schedule is None:
            raise ValueError("momentum_sgd requires an explicit Fourier radius schedule")
        if self.optimizer == "momentum_sgd" and self.fixed_healpix_order is None:
            raise ValueError("momentum_sgd requires a fixed coarse HEALPix grid")
        if self.optimizer == "momentum_sgd" and self.diagnostic_continue_optimiser is not None:
            raise ValueError("momentum_sgd cannot continue a VDAM optimizer checkpoint")
        if not (0.0 < self.sgd_learning_rate < float("inf")):
            raise ValueError("sgd_learning_rate must be positive and finite")
        if self.fourier_radius_schedule is not None:
            if len(self.fourier_radius_schedule) != self.nr_iter or any(r < 1 for r in self.fourier_radius_schedule):
                raise ValueError("fourier_radius_schedule needs one positive radius per iteration")
        if self.fixed_healpix_order is not None and self.fixed_healpix_order != self.healpix_order:
            raise ValueError("fixed_healpix_order must equal the initial healpix_order")
        if self.stochastic_all_iterations and (
            self.grad_em_iters != 0 or self.pilot_controls is None or self.pilot_controls.stochastic_batch_size is None
        ):
            raise ValueError("stochastic_all_iterations requires grad_em_iters=0 and a fixed stochastic batch")
        if self.grad_write_iter < 1:
            raise ValueError("grad_write_iter must be >= 1")
        if self.pilot_controls is not None:
            self.pilot_controls.validate(initial_healpix_order=self.healpix_order)
        if self.diagnostic_stop_after_iteration is not None and not (
            1 <= int(self.diagnostic_stop_after_iteration) <= int(self.nr_iter)
        ):
            raise ValueError("diagnostic_stop_after_iteration must be between 1 and nr_iter")
        if (
            self.diagnostic_continue_optimiser is not None
            and self.diagnostic_stop_after_iteration is None
        ):
            raise ValueError(
                "diagnostic_continue_optimiser requires diagnostic_stop_after_iteration; "
                "unbounded continuation is intentionally unsupported"
            )
        if self.optimizer == "vdam" and self.diagnostic_continue_optimiser is None:
            # RELION's tau2-fudge sigmoid has length grad_inbetween_iter // 4; when that is 0 (4 or 5 iterations at
            # the default fractions) the fudge at iteration grad_ini_iter is 0/0 = NaN (relax.vdam.schedules keeps
            # it for RELION parity). The VDAM M-step's SSNR and its FSC estimate read it, so the resolution update
            # of that iteration reads NaN. A diagnostic continuation replays RELION's own schedule and is exempt.
            phases = compute_phase_lengths(self.nr_iter, self.grad_ini_frac, self.grad_fin_frac)
            unusable = [
                it for it in range(1, self.nr_iter + 1)
                if not math.isfinite(compute_tau2_fudge(it, phases, True, 3, tau2_fudge_arg=self.tau2_fudge))
            ]
            if unusable:
                raise ValueError(
                    f"VDAM's tau2-fudge schedule for {self.nr_iter} iterations is not finite at iteration(s) "
                    f"{unusable} (RELION's sigmoid length grad_inbetween_iter // 4 is 0); choose another --nr-iter "
                    "or --optimizer momentum_sgd"
                )
        if self.fn_tomograms is not None and (self.optimizer != "vdam" or not self.do_run_C1):
            raise NotImplementedError("subtomogram InitialModel runs RELION's VDAM in C1")
        if self.padding_factor not in (1, 2):
            raise NotImplementedError("native InitialModel currently supports RELION GUI --pad 1 or 2 only")
        if not self.do_run_C1 and self.sym_name.lower() != "c1":
            raise NotImplementedError(
                "native InitialModel direct refinement currently supports C1 only; "
                "use the GUI-default do_run_C1 mode until symmetry-restricted sampling is implemented"
            )

    def symmetry_mode_warning(self) -> str | None:
        """The notice for a non-C1 ``--sym`` refined in C1, or None.

        RELION's InitialModel GUI job refines in C1 and then runs ``relion_align_symmetry --apply_sym
        --select_largest_class`` (pipeline_jobs.cpp:3572-3587); relax does the same
        (:mod:`relax.vdam.align_symmetry`), so the requested group reaches only ``initial_model.mrc``.
        """
        if not self.do_run_C1 or is_identity_symmetry(self.sym_name):
            return None
        return (
            f"WARNING: --sym {self.sym_name} with --run-in-c1: C1 refinement, then RELION-style symmetry alignment and "
            f"symmetrization of the final map (initial_model.mrc, as relion_align_symmetry --sym {self.sym_name} "
            "--apply_sym --select_largest_class); the class maps stay C1, and direct symmetric VDAM is not implemented"
        )
