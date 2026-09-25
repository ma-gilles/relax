"""Opt-in caps for the native VDAM comparison arm of the PPCA pilots.

The matched VDAM runs of ``docs/math/vdam_ppca_algorithm.md`` (small 2,000-particle
pilot) fix the non-final subset size and cap the Fourier radius and HEALPix order.
Native frequency and angular adaptation stays active within the caps. The VDAM
controller holds one of these as ``NativeInitialModelOptions.pilot_controls``;
``None`` there is native VDAM, unchanged.
"""

from dataclasses import dataclass
from pathlib import Path

from relax.vdam.schedules import default_subset_sizes_for_3d_initial_model


@dataclass(frozen=True)
class VdamPilotControls:
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
