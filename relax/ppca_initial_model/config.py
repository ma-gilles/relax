"""Explicit provisional pilot policy (implementation plan sections 3 and 5)."""

from dataclasses import dataclass

import numpy as np

from relax.steps.sgd_schedule import (
    DEFAULT_GRAD_EM_ITERS,
    compute_phase_lengths,
    compute_stepsize,
    compute_subset_size,
    compute_tau2_fudge,
    default_subset_sizes_for_3d_initial_model,
)


@dataclass(frozen=True)
class Config:
    q: int = 2
    iterations: int = 200
    seed: int = 11
    stages: tuple = ((1, 4, 1), (61, 8, 2), (111, 16, 3), (161, 32, 3))
    # The PCs the model learns, by iteration: ((first iteration, PCs), ...), the first entry at iteration 1, the counts
    # not decreasing and the last equal to q. A PC joins at the start of its stage as a small random loading
    # (initialization.birth_loadings) of size pc_birth_scale times the mean's standard deviation; until then its
    # column is zero and the E-step and the update see only the active PCs. None learns all q PCs from the first
    # update, seeded by the bootstrap.
    pc_stages: tuple | None = None
    pc_birth_scale: float = 0.1
    # Diagnostic: a map in RELION's convention (a relax or RELION map file) that the mean starts from instead of the
    # seed's random one, e.g. the ground truth of a simulated fixture; the loadings keep the seed's values
    # (initialization.with_mean_volume). None starts from the seed.
    init_mean_map: str | None = None
    # RELION's --tau2_fudge and --tau2_fudge_scheme (ml_optimiser.cpp updateTau2Fudge) for the VDAM gate; None takes
    # the InitialModel defaults, a fudge of 4 on the scheme 4-step (1 at the start, 4 from the middle of the run).
    # A fudge of 1 with the scheme plain keeps the gate at its strongest shrinkage throughout.
    tau2_fudge: float | None = None
    tau2_fudge_scheme: str | None = None
    # The Fourier radius of each iteration. "stages": the stages table's. "data_vs_prior": RELION's InitialModel rule
    # (relax/ppca_initial_model/radius_rule.py): the table's first radius at iteration 1, then the last shell where the
    # mean's data_vs_prior stays at or above 1 plus incr_size shells (RELION's --incr_size), never decreasing. The
    # table's later radii do not apply; the HEALPix orders follow order_rule. "data_vs_prior_pcs": the same, where a
    # shell counts only if the loadings' total variance also has data_vs_prior at or above 1
    # (radius_rule.loading_data_vs_prior); without active loadings it is the mean's rule.
    radius_rule: str = "stages"
    incr_size: int = 10
    # The prior over the HEALPix directions of the global search. "uniform": every direction equally likely throughout.
    # "learned": RELION's pdf_direction, blended from each iteration's posterior mass (it can collapse onto a few
    # directions when the poses are wrong).
    direction_prior: str = "uniform"
    # The HEALPix order of each iteration. "stages": the stages table's. "accuracy": RELION's updateAngularSampling
    # rule with the expected rotation accuracy of the PPCA model (relax/ppca_initial_model/order_rule.py): the table's
    # first order at iteration 1, then, every accuracy_interval iterations, one order finer while the angular step is
    # at least 75% of the accuracy, up to order_max. The final embedding pass takes the first order from there that is
    # fine enough, up to final_order_max (None: order_max). The table's later orders do not apply.
    order_rule: str = "stages"
    order_max: int = 4
    final_order_max: int | None = None
    accuracy_interval: int = 10
    # The accuracy the order rule reads: "divergence" (the PPCA image laws of the two poses, doc section 4.1) or
    # "mean_only" (the mean image alone, RELION's quantity under the engine's noise and window).
    accuracy_signal: str = "divergence"
    # The divergence at which two poses count as told apart, which sets the expected accuracy: a higher value gives a
    # larger accuracy angle and so fewer order increases. None takes RELION's PVALUE, ln 100.
    accuracy_kl_threshold: float | None = None
    # Adaptive oversampling: 0 scores every stage's full pose grid; 1 runs the last stage with RELION's two passes
    # (each image's significant coarse samples, then their children; relax/ppca_refinement/oversampled_stream.py)
    # and the earlier stages dense (iteration_loop.oversampled_update).
    oversampling: int = 0
    # The adaptive fraction of the posterior mass the significant samples hold (RELION's --adaptive_fraction).
    target_mass: float = 0.999
    # Significant coarse samples kept per image at most (RELION's --maxsig: 100 per class for gradient runs).
    max_significant: int = 100
    shift_range: float = 6
    shift_step: float = 2
    image_batch_size: int = 150
    rotation_block_size: int = 512
    stream_full_fine_rows: bool = False
    # The default engine: the GPU stream over the full pose grid of each stage. False selects the
    # host-mask dense engine (q <= 2 only), the float32 reference of the stream's tests.
    stream_coarse_recompute: bool = True
    fine_devices: int = 1
    stochastic_batch_size: int | None = None
    stochastic_all_iterations: bool = False
    balanced_stochastic_halves: bool = False
    checkpoint_interval: int = 1
    skip_final_embeddings: bool = False
    optimizer: str = "vdam"
    sgd_learning_rate: float = 0.4
    # Momentum SGD only: the loading columns' learning rate relative to the mean's (1: the same). 1/16 keeps the mean and
    # the pose estimate sound at q = 10 (docs/math/ppca_momentum_sgd.md).
    sgd_loading_scale: float = 1 / 16
    # Precision of the streamed engine's four GEMMs: "auto" (the default: tf32 on GPUs of compute
    # capability 8.0+, fp32 elsewhere), "tf32" (TF32 tensor-core products, float32 accumulation) or
    # "fp32" (exact float32 products). Section 14 of docs/math/vdam_ppca_algorithm.md has the evidence.
    gemm_precision: str = "auto"
    # Particle images in host memory: "auto" (the default) reads the whole stack once at start-up
    # when it takes at most PREREAD_MEMORY_FRACTION of the job's memory, otherwise reads each tile
    # from disk; "on" and "off" force either. A runtime setting, like gemm_precision.
    preread_images: str = "auto"
    # Pass 2 skips the pose rows on which every image of a tile has less posterior mass than this
    # (0: visit every row), when that leaves at most half the rows. A runtime setting;
    # full_row_kernels._pass2_rows has the bound; section 14 of docs/math/vdam_ppca_algorithm.md the evidence.
    pass2_mass_floor: float = 1e-10

    def __post_init__(self):
        if isinstance(self.q, bool) or not isinstance(self.q, (int, np.integer)) or self.q <= 0 or self.seed <= 0 or self.iterations <= 0:
            raise ValueError("PPCA requires positive integer q, seed and iteration count")
        if not self.stages or self.stages[0][0] != 1:
            raise ValueError("Stages must begin at iteration 1")
        if any(isinstance(hp, bool) or not isinstance(hp, (int, np.integer)) for _, _, hp in self.stages):
            # A fractional order built the truncated order's rotation grid but sized pass 1 from the fraction.
            raise ValueError(f"Stage HEALPix orders must be integers, got {[hp for _, _, hp in self.stages]}")
        if list(self.stages) != sorted(self.stages) or any(r <= 0 or hp < 0 for _, r, hp in self.stages):
            raise ValueError("Invalid radius/HEALPix schedule")
        if not 0 < self.target_mass <= 1 or self.oversampling < 0:
            raise ValueError("Invalid pose support")
        if self.oversampling > 1 or self.stream_full_fine_rows or self.fine_devices > 1:
            raise ValueError(
                "PPCA supports oversampling 0 and 1; the --stream-full-fine-rows / --fine-devices fine pass is "
                "retired (section 14 of docs/math/vdam_ppca_algorithm.md)"
            )
        if self.max_significant <= 0:
            raise ValueError("max_significant must be positive")
        if min(self.image_batch_size, self.rotation_block_size, self.checkpoint_interval) <= 0:
            raise ValueError("Batch/checkpoint sizes must be positive")
        if self.fine_devices < 1:
            raise ValueError("fine_devices must be positive")
        if self.oversampling and not self.stream_coarse_recompute:
            raise ValueError("Adaptive oversampling runs on the streamed engine (stream_coarse_recompute)")
        if self.q > 2 and not self.stream_coarse_recompute:
            raise ValueError("Higher-rank InitialModel requires the streamed coarse recompute engine")
        if self.pc_stages is not None:
            self._validate_pc_stages()
        if not np.isfinite(self.pc_birth_scale) or self.pc_birth_scale <= 0:
            raise ValueError("pc_birth_scale must be finite and positive")
        if self.radius_rule not in ("stages", "data_vs_prior", "data_vs_prior_pcs"):
            raise ValueError("The radius rule must be stages, data_vs_prior or data_vs_prior_pcs")
        if self.direction_prior not in ("learned", "uniform"):
            raise ValueError("The direction prior must be learned or uniform")
        if isinstance(self.incr_size, bool) or not isinstance(self.incr_size, (int, np.integer)) or self.incr_size <= 0:
            raise ValueError("incr_size must be a positive integer")
        self._validate_order_rule()
        if self.stochastic_batch_size is not None and self.stochastic_batch_size <= 0:
            raise ValueError("Stochastic batch size must be positive")
        if self.stochastic_all_iterations and self.stochastic_batch_size is None:
            raise ValueError("Stochastic all iterations requires a fixed stochastic batch size")
        if self.balanced_stochastic_halves and (
            not self.stochastic_all_iterations
            or self.stochastic_batch_size is None
            or self.stochastic_batch_size < 2
            or self.stochastic_batch_size % 2
        ):
            raise ValueError("Balanced stochastic halves require an even fixed batch on all iterations")
        if self.shift_range < 0 or self.shift_step <= 0:
            raise ValueError("Shift range must be nonnegative and step positive")
        if self.optimizer not in ("vdam", "momentum_sgd"):
            raise ValueError("Optimizer must be vdam or momentum_sgd")
        if self.optimizer == "vdam":
            # RELION's tau2-fudge sigmoid has length grad_inbetween_iter // 4; for 4 and 5 iterations that is 0 and
            # the fudge at iteration grad_ini_iter is 0/0 = NaN (kept for RELION parity in relax.vdam.schedules).
            # The VDAM update multiplies the fudge into every gate, so such a run cannot produce a finite model.
            phases = compute_phase_lengths(self.iterations)
            unusable = [
                i for i in range(1, self.iterations + 1) if not np.isfinite(self.tau2_fudge_at(i, phases))
            ]
            if unusable:
                raise ValueError(
                    f"VDAM's tau2-fudge schedule for {self.iterations} iterations is not finite at iteration(s) "
                    f"{unusable} (RELION's sigmoid length grad_inbetween_iter // 4 is 0); choose another iteration "
                    "count or optimizer momentum_sgd"
                )
        if not np.isfinite(self.sgd_learning_rate) or self.sgd_learning_rate <= 0:
            raise ValueError("Momentum SGD learning rate must be finite and positive")
        if not np.isfinite(self.sgd_loading_scale) or self.sgd_loading_scale <= 0:
            raise ValueError("Momentum SGD loading scale must be finite and positive")
        if self.gemm_precision not in ("auto", "fp32", "tf32"):
            raise ValueError("GEMM precision must be auto, fp32 or tf32")
        if self.gemm_precision == "tf32" and not self.stream_coarse_recompute:
            raise ValueError("TF32 GEMMs apply to the streamed engines only")
        if self.preread_images not in ("auto", "on", "off"):
            raise ValueError("Image preread must be auto, on or off")
        if not 0 <= self.pass2_mass_floor < 1:
            raise ValueError("The pass-2 mass floor must be in [0, 1)")

    def _validate_order_rule(self) -> None:
        """Refuse an order rule other than the table or the accuracy rule, or an accuracy rule it cannot run."""
        if self.order_rule not in ("stages", "accuracy"):
            raise ValueError("The order rule must be stages or accuracy")
        if self.order_rule == "stages":
            return
        if self.accuracy_signal not in ("divergence", "mean_only"):
            raise ValueError("The accuracy signal must be divergence or mean_only")
        if self.accuracy_kl_threshold is not None and not (
            np.isfinite(self.accuracy_kl_threshold) and self.accuracy_kl_threshold > 0
        ):
            raise ValueError("accuracy_kl_threshold must be finite and positive")
        for name in ("order_max", "accuracy_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.order_max < self.stages[0][2]:
            raise ValueError(f"order_max {self.order_max} is below the first stage's order {self.stages[0][2]}")
        if self.final_order_max is not None and (
            isinstance(self.final_order_max, bool)
            or not isinstance(self.final_order_max, (int, np.integer))
            or self.final_order_max < self.order_max
        ):
            raise ValueError("final_order_max must be an integer of at least order_max")
        if self.oversampling or not self.stream_coarse_recompute:
            raise ValueError("The accuracy order rule needs the streamed engine without oversampling")

    def _validate_pc_stages(self) -> None:
        """Refuse a PC schedule that does not start at iteration 1, shrinks, does not end at q or is out of range."""
        stages = self.pc_stages
        if not stages or any(len(stage) != 2 for stage in stages):
            raise ValueError("PC stages are (first iteration, PCs) pairs")
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for stage in stages for v in stage):
            raise ValueError(f"PC stages must hold integers, got {stages}")
        starts, counts = [s for s, _ in stages], [n for _, n in stages]
        if starts[0] != 1 or any(a >= b for a, b in zip(starts, starts[1:])):
            raise ValueError("PC stages must begin at iteration 1 and have increasing first iterations")
        if starts[-1] > self.iterations:
            raise ValueError(
                f"The last PC stage starts at iteration {starts[-1]}, after the {self.iterations} iterations"
            )
        if counts[0] < 0 or any(a > b for a, b in zip(counts, counts[1:])) or counts[-1] != self.q:
            raise ValueError(f"PC counts must not decrease, start at 0 or more and end at q = {self.q}, got {counts}")
        if not self.stream_coarse_recompute:
            raise ValueError("PC stages run on the streamed engine (stream_coarse_recompute)")

    def tau2_fudge_at(self, iteration: int, phases) -> float:
        """RELION's tau2 fudge at ``iteration`` of the VDAM schedule with ``phases`` under the run's options."""
        return compute_tau2_fudge(
            iteration,
            phases,
            True,
            3,
            tau2_fudge_arg=self.tau2_fudge,
            tau2_fudge_scheme=self.tau2_fudge_scheme,
        )

    def active_pcs(self, iteration: int) -> int:
        """The number of PCs the model learns at ``iteration``: all q without ``pc_stages``."""
        if self.pc_stages is None:
            return self.q
        return next(count for start, count in reversed(self.pc_stages) if start <= iteration)

    def step_factor(self, iteration, n_images):
        """``min(1, count / scheduled count)``: the factor on VDAM's step for a batch below its own subset.

        VDAM's step and subset schedules together set how many particles its moving average spans
        (about subset / step). A fixed ``stochastic_batch_size`` below VDAM's subset size shortens that
        window; scaling the step by the batch's share of the scheduled subset restores it
        (docs/math/vdam_ppca_algorithm.md section 17). It is one for VDAM's own schedule.
        """
        phases = compute_phase_lengths(self.iterations)
        first, last = default_subset_sizes_for_3d_initial_model(n_images)
        scheduled = compute_subset_size(
            iteration, phases, first, last, n_images, self.iterations, grad_em_iters=DEFAULT_GRAD_EM_ITERS
        )
        scheduled = n_images if scheduled < 0 or iteration == self.iterations else min(scheduled, n_images)
        return min(1.0, self.schedule(iteration, n_images)[0] / scheduled)

    def stage(self, iteration):
        return next((r, hp) for start, r, hp in reversed(self.stages) if start <= iteration)

    def schedule(self, iteration, n_images):
        phases = compute_phase_lengths(self.iterations)
        first, last = default_subset_sizes_for_3d_initial_model(n_images)
        count = compute_subset_size(
            iteration, phases, first, last, n_images, self.iterations, grad_em_iters=DEFAULT_GRAD_EM_ITERS
        )
        if self.stochastic_batch_size is not None and (iteration != self.iterations or self.stochastic_all_iterations):
            count = self.stochastic_batch_size
        if (iteration == self.iterations and not self.stochastic_all_iterations) or count < 0:
            count = n_images
        return (
            min(count, n_images),
            compute_stepsize(iteration, phases, True, 3),
            self.tau2_fudge_at(iteration, phases),
        )
