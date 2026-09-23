"""Explicit provisional pilot policy (implementation plan sections 3 and 5)."""

from dataclasses import dataclass

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
    stochastic_batch_size: int | None = None
    checkpoint_interval: int = 1

    def __post_init__(self):
        if self.q != 2 or self.seed <= 0 or self.iterations <= 0:
            raise ValueError("First version requires q=2, positive seed and iteration count")
        if not self.stages or self.stages[0][0] != 1:
            raise ValueError("Stages must begin at iteration 1")
        if list(self.stages) != sorted(self.stages) or any(r <= 0 or hp < 0 for _, r, hp in self.stages):
            raise ValueError("Invalid radius/HEALPix schedule")
        if not 0 < self.target_mass <= 1 or self.oversampling < 0:
            raise ValueError("Invalid pose support")
        if min(self.image_batch_size, self.rotation_block_size, self.checkpoint_interval) <= 0:
            raise ValueError("Batch/checkpoint sizes must be positive")
        if self.stochastic_batch_size is not None and self.stochastic_batch_size <= 0:
            raise ValueError("Stochastic batch size must be positive")
        if self.shift_range < 0 or self.shift_step <= 0:
            raise ValueError("Shift range must be nonnegative and step positive")

    def stage(self, iteration):
        return next((r, hp) for start, r, hp in reversed(self.stages) if start <= iteration)

    def schedule(self, iteration, n_images):
        phases = compute_phase_lengths(self.iterations)
        first, last = default_subset_sizes_for_3d_initial_model(n_images)
        count = compute_subset_size(iteration, phases, first, last, n_images, self.iterations)
        if self.stochastic_batch_size is not None and iteration != self.iterations:
            count = self.stochastic_batch_size
        if iteration == self.iterations or count < 0:
            count = n_images
        return (
            min(count, n_images),
            compute_stepsize(iteration, phases, True, 3),
            compute_tau2_fudge(iteration, phases, True, 3),
        )
