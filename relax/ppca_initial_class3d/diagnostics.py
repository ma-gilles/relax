"""Observations of mixture updates; these values never steer optimization."""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np


class MetricFailure(ValueError):
    """Retain the exact offending class/half operands for controller-owned output."""

    def __init__(self, class_index, half_index, component, message):
        super().__init__(f"Class {class_index}, half {half_index}: {message}")
        self.class_index = class_index
        self.half_index = half_index
        self.component = component


@dataclass
class PosteriorDiagnostics:
    """Tile sums of joint class/pose diagnostics, normalized by particle count."""

    count: int = 0
    pmax_sum: float = 0.0
    entropy_sum: float = 0.0
    covariance_trace_sum: float = 0.0
    scored_rows: int = 0
    pass2_rows: int = 0
    tile_sizes: set = field(default_factory=set)
    precisions: set = field(default_factory=set)

    def add(self, statistics):
        info, n = statistics.diagnostics, statistics.n_images
        self.count += n
        self.pmax_sum += info["pmax_mean"] * n
        self.entropy_sum += info["pose_entropy_mean"] * n
        self.covariance_trace_sum += info["latent_covariance_trace_mean"] * n
        self.scored_rows += info["scored_rows"]
        self.pass2_rows += info["pass2_rows"]
        self.tile_sizes.add(info["tile_size"])
        self.precisions.add(info["gemm_precision"])

    def record(self):
        return {
            "weighting": "joint class/pose contribution per physical particle, not conditional on class",
            "particle_count": self.count,
            "pmax_mean": self.pmax_sum / self.count,
            "pose_entropy_mean": self.entropy_sum / self.count,
            "latent_covariance_trace_mean": self.covariance_trace_sum / self.count,
            "tile_sizes": sorted(self.tile_sizes),
            "gemm_precision": sorted(self.precisions),
            "pass2_row_fraction": self.pass2_rows / self.scored_rows,
        }


def model_summary(model):
    """Reduce Fourier-half model powers and a small loading Gram matrix on device."""
    loading = model[:, 1:]
    gram = jnp.matmul(loading.conj().T, loading, precision=jax.lax.Precision.HIGHEST)
    singular_values = jnp.sqrt(jnp.maximum(jnp.linalg.eigvalsh(gram), 0))[::-1]
    mean_power, loading_power, singular_values = jax.device_get((
        jnp.sum(jnp.abs(model[:, 0]) ** 2), jnp.sum(jnp.abs(loading) ** 2), singular_values,
    ))
    return {"mean_power": float(mean_power), "loading_power": float(loading_power),
            "loading_singular_values": np.asarray(singular_values)}
