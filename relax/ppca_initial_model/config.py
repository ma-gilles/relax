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
    oversampling: int = 1
    target_mass: float = 0.999
    shift_range: float = 6
    shift_step: float = 2
    image_batch_size: int = 16
    rotation_block_size: int = 128
    fine_image_tile_size: int = 1
    stream_full_fine_rows: bool = False
    stream_coarse_recompute: bool = False
    fine_devices: int = 1
    stochastic_batch_size: int | None = None
    stochastic_all_iterations: bool = False
    balanced_stochastic_halves: bool = False
    checkpoint_interval: int = 1
    skip_final_embeddings: bool = False
    optimizer: str = "vdam"
    sgd_learning_rate: float = 0.4
    # Precision of the streamed engine's four GEMMs: "fp32" (exact float32 products, the default)
    # or "tf32" (TF32 tensor-core products with float32 accumulation; GPUs of compute capability 8.0+).
    gemm_precision: str = "fp32"

    def __post_init__(self):
        if isinstance(self.q, bool) or not isinstance(self.q, (int, np.integer)) or self.q <= 0 or self.seed <= 0 or self.iterations <= 0:
            raise ValueError("PPCA requires positive integer q, seed and iteration count")
        if not self.stages or self.stages[0][0] != 1:
            raise ValueError("Stages must begin at iteration 1")
        if list(self.stages) != sorted(self.stages) or any(r <= 0 or hp < 0 for _, r, hp in self.stages):
            raise ValueError("Invalid radius/HEALPix schedule")
        if not 0 < self.target_mass <= 1 or self.oversampling < 0:
            raise ValueError("Invalid pose support")
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
        if self.gemm_precision not in ("fp32", "tf32"):
            raise ValueError("GEMM precision must be fp32 or tf32")
        if self.gemm_precision != "fp32" and not (self.stream_coarse_recompute or self.stream_full_fine_rows):
            raise ValueError("TF32 GEMMs apply to the streamed engines only")

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
