"""Explicit provisional pilot policy (implementation plan sections 3 and 5)."""

from dataclasses import dataclass

import numpy as np

from relax.vdam.schedules import (
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
    # Oversampling > 0 (coarse significance, then a finer pass over the significant poses) is
    # refused: measured slower than the dense grid (section 14 of docs/math/vdam_ppca_algorithm.md).
    oversampling: int = 0
    target_mass: float = 0.999
    shift_range: float = 6
    shift_step: float = 2
    image_batch_size: int = 150
    rotation_block_size: int = 512
    fine_image_tile_size: int = 1
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
    # full_row_stream._pass2_rows has the bound; section 14 of docs/math/vdam_ppca_algorithm.md the evidence.
    pass2_mass_floor: float = 1e-10

    def __post_init__(self):
        if isinstance(self.q, bool) or not isinstance(self.q, (int, np.integer)) or self.q <= 0 or self.seed <= 0 or self.iterations <= 0:
            raise ValueError("PPCA requires positive integer q, seed and iteration count")
        if not self.stages or self.stages[0][0] != 1:
            raise ValueError("Stages must begin at iteration 1")
        if list(self.stages) != sorted(self.stages) or any(r <= 0 or hp < 0 for _, r, hp in self.stages):
            raise ValueError("Invalid radius/HEALPix schedule")
        if not 0 < self.target_mass <= 1 or self.oversampling < 0:
            raise ValueError("Invalid pose support")
        if self.oversampling > 0 or self.stream_full_fine_rows or self.fine_devices > 1:
            raise ValueError(
                "PPCA oversampling > 0 (and its --stream-full-fine-rows / --fine-devices pass) is not supported: "
                "it measured slower than the dense pose grid; see section 14 of docs/math/vdam_ppca_algorithm.md"
            )
        if min(self.image_batch_size, self.rotation_block_size, self.fine_image_tile_size, self.checkpoint_interval) <= 0:
            raise ValueError("Batch/checkpoint sizes must be positive")
        if self.fine_devices < 1 or (self.fine_devices > 1 and not self.stream_full_fine_rows):
            raise ValueError("Multiple fine devices require the streamed fine engine")
        if self.stream_coarse_recompute and (
            self.oversampling != 0 or self.fine_devices != 1 or self.stream_full_fine_rows
        ):
            raise ValueError("Coarse recomputation requires oversampling=0 and one device")
        if self.q > 2 and not self.stream_coarse_recompute:
            raise ValueError("Higher-rank InitialModel requires the streamed coarse recompute engine")
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
        if not np.isfinite(self.sgd_learning_rate) or self.sgd_learning_rate <= 0:
            raise ValueError("Momentum SGD learning rate must be finite and positive")
        if self.gemm_precision not in ("auto", "fp32", "tf32"):
            raise ValueError("GEMM precision must be auto, fp32 or tf32")
        if self.gemm_precision == "tf32" and not (self.stream_coarse_recompute or self.stream_full_fine_rows):
            raise ValueError("TF32 GEMMs apply to the streamed engines only")
        if self.preread_images not in ("auto", "on", "off"):
            raise ValueError("Image preread must be auto, on or off")
        if not 0 <= self.pass2_mass_floor < 1:
            raise ValueError("The pass-2 mass floor must be in [0, 1)")

    def step_factor(self, iteration, n_images):
        """``min(1, count / scheduled count)``: the factor on VDAM's step for a batch below its own subset.

        VDAM's step and subset schedules together set how many particles its moving average spans
        (about subset / step). A fixed ``stochastic_batch_size`` below VDAM's subset size shortens that
        window; scaling the step by the batch's share of the scheduled subset restores it
        (docs/math/vdam_ppca_algorithm.md section 17). It is one for VDAM's own schedule.
        """
        phases = compute_phase_lengths(self.iterations)
        first, last = default_subset_sizes_for_3d_initial_model(n_images)
        scheduled = compute_subset_size(iteration, phases, first, last, n_images, self.iterations)
        scheduled = n_images if scheduled < 0 or iteration == self.iterations else min(scheduled, n_images)
        return min(1.0, self.schedule(iteration, n_images)[0] / scheduled)

    def stage(self, iteration):
        return next((r, hp) for start, r, hp in reversed(self.stages) if start <= iteration)

    def schedule(self, iteration, n_images):
        phases = compute_phase_lengths(self.iterations)
        first, last = default_subset_sizes_for_3d_initial_model(n_images)
        count = compute_subset_size(iteration, phases, first, last, n_images, self.iterations)
        if self.stochastic_batch_size is not None and (iteration != self.iterations or self.stochastic_all_iterations):
            count = self.stochastic_batch_size
        if (iteration == self.iterations and not self.stochastic_all_iterations) or count < 0:
            count = n_images
        return (
            min(count, n_images),
            compute_stepsize(iteration, phases, True, 3),
            compute_tau2_fudge(iteration, phases, True, 3),
        )
