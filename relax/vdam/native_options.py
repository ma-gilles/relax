"""Shared run defaults and native InitialModel options.

Sampling and continuation read these records without importing the driver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from relax.helpers.particle_io import DEFAULT_KEEP_FREE_SCRATCH_GB
from relax.vdam.schedules import (
    DEFAULT_GRAD_EM_ITERS,
    DEFAULT_GRAD_FIN_FRAC,
    DEFAULT_GRAD_INI_FRAC,
    DEFAULT_GRAD_MU,
    DEFAULT_STEPSIZE_3D_INITIAL_MODEL,
    GUI_DEFAULT_NR_CLASSES,
    GUI_DEFAULT_NR_ITER,
    GUI_DEFAULT_TAU2_FUDGE,
)


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
    pilot_controls: object | None = None  # relax.ppca_initial_model.vdam_controls.VdamPilotControls
    pass2_engine: str = "auto"
    bootstrap_min_particles: int = 1000
    sigma2_min_particles: int = 1000
    padding_factor: int = 1
    # RELION GUI defaults: no pre-read, no scratch copy (relax.helpers.particle_io).
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
    optimizer: Literal["vdam", "cryosparc_sgd"] = "vdam"
    sgd_learning_rate: float = 1.0
    fourier_radius_schedule: tuple[int, ...] | None = None
    fixed_healpix_order: int | None = None
    stochastic_all_iterations: bool = False


@dataclass(frozen=True, kw_only=True)
class NativeInitialModelOptions(InitialModelDefaults):
    """Options for a native InitialModel run; defaults mirror the GUI command."""

    fn_img: str
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

    def validate_run(self) -> None:
        """Check supported settings before loading particles or creating run state."""
        if self.nr_classes < 1:
            raise ValueError("nr_classes must be >= 1")
        if self.nr_iter < 1:
            raise ValueError("nr_iter must be >= 1")
        if self.optimizer not in {"vdam", "cryosparc_sgd"}:
            raise ValueError(f"unknown InitialModel optimizer: {self.optimizer!r}")
        if self.optimizer == "cryosparc_sgd" and self.oversampling != 0:
            raise ValueError("cryosparc_sgd requires --oversampling 0 (one coarse pose grid)")
        if self.optimizer == "cryosparc_sgd" and self.mstep_compute_dtype != "float32":
            raise ValueError("cryosparc_sgd is a production float32 optimizer")
        if self.optimizer == "cryosparc_sgd" and self.fourier_radius_schedule is None:
            raise ValueError("cryosparc_sgd requires an explicit Fourier radius schedule")
        if self.optimizer == "cryosparc_sgd" and self.fixed_healpix_order is None:
            raise ValueError("cryosparc_sgd requires a fixed coarse HEALPix grid")
        if self.optimizer == "cryosparc_sgd" and self.diagnostic_continue_optimiser is not None:
            raise ValueError("cryosparc_sgd cannot continue a VDAM optimizer checkpoint")
        if not (0.0 < self.sgd_learning_rate < float("inf")):
            raise ValueError("sgd_learning_rate must be positive and finite")
        if self.fourier_radius_schedule is not None:
            if len(self.fourier_radius_schedule) != self.nr_iter or any(
                r < 1 for r in self.fourier_radius_schedule
            ):
                raise ValueError("fourier_radius_schedule needs one positive radius per iteration")
        if self.fixed_healpix_order is not None and self.fixed_healpix_order != self.healpix_order:
            raise ValueError("fixed_healpix_order must equal the initial healpix_order")
        if self.stochastic_all_iterations and (
            self.grad_em_iters != 0 or self.pilot_controls is None
            or self.pilot_controls.stochastic_batch_size is None
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
        if self.padding_factor not in (1, 2):
            raise NotImplementedError("native InitialModel currently supports RELION GUI --pad 1 or 2 only")
        if not self.do_run_C1 and self.sym_name.lower() != "c1":
            raise NotImplementedError(
                "native InitialModel direct refinement currently supports C1 only; "
                "use the GUI-default do_run_C1 mode until symmetry-restricted sampling is implemented"
            )
