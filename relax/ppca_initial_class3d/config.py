"""Opt-in mixture policy; the single-model PPCA defaults remain unchanged."""

from dataclasses import dataclass

import numpy as np

from relax.ppca_initial_model.config import Config as InitialModelConfig

DEFAULT_RADII = (4, 8, 16, 32)
DEFAULT_ORDERS = (1, 2, 2, 2)  # coarse HEALPix orders; stages 2-4 add order-1 children (3.75 degrees) under oversampling
DEFAULT_STAGE_FRACTIONS = (0.0, 0.30, 0.55, 0.80)  # of the iteration count: 1, 61, 111, 161 for 200 updates


def default_stages(iterations: int) -> tuple:
    """The four-stage schedule scaled to ``iterations``: the single model's absolute starts (1, 61, 111, 161 of
    200) as fractions, so a short run still marches through every radius; stages that would start after the
    run ends or on the same update as the previous one are dropped."""
    stages = []
    for fraction, radius, order in zip(DEFAULT_STAGE_FRACTIONS, DEFAULT_RADII, DEFAULT_ORDERS):
        start = int(round(fraction * iterations)) + 1
        if start > iterations or (stages and start <= stages[-1][0]):
            break
        stages.append((start, radius, order))
    return tuple(stages)


@dataclass(frozen=True)
class Config(InitialModelConfig):
    n_classes: int = 2
    # None: default_stages(iterations). The single model's fixed 200-update starts do not fit shorter runs.
    stages: tuple | None = None
    # RELION's two adaptive passes per class from the second stage on (oversampling_start None): pass 1 on the
    # stage's coarse order, pass 2 on the order-1 children of each particle's significant samples.
    oversampling: int = 1
    # Minibatches drawn as whole tomograms (tilt particles only; see tomogram_batches below).
    tomogram_batches: bool = True
    # Dirichlet(alpha=1+pseudocount) MAP update, per physical particle.
    # Positive mass prevents an underflowed class becoming irreversibly absent.
    class_pseudocount: float = 1.0
    # With oversampling 1, the first update that runs RELION's two passes (pass 1 on the stage's coarse grid, pass
    # 2 on the children of each particle's significant samples, per class); None: the last stage's first update,
    # as the single-model controller. Earlier updates score the stage's full grid.
    oversampling_start: int | None = None
    # RELION's --zero_mask: the tilt images are zeroed outside the particle diameter under a 5-pixel raised-cosine
    # edge before any Fourier operand, start-up noise included (images.py). Tilt particles only; the command
    # turns it on for --ios inputs unless --no-zero-mask (RELION's default), off for single-particle manifests.
    zero_mask: bool = False
    # RELION's local angular searches (PRIOR_ROTTILT_PSI) from this update on: each particle's candidate coarse
    # poses lie within 3 sigma of its stored pose (local_search.py, membership pose memory); None: only when
    # auto-sampling switches them on. local_search_sigma_deg None: RELION's twice the children's angular step.
    local_search_start: int | None = None
    local_search_sigma_deg: float | None = None
    # RELION's gradient auto-sampling (--auto_sampling, auto_sampling.py): every accuracy_interval updates, the
    # expected angular and shift accuracy of accuracy_particles particles (calculateExpectedAngularErrors) sets
    # the coarse HEALPix order (at most max_order; local searches from local_order on, RELION's
    # --auto_local_healpix_order 4 counted on its base grid: coarse 3 with children is the same 3.75 degrees) and
    # the offset step and range. The stages then set the resolution support only. The CLI resolves this to
    # True for --ios unless --no-auto-sampling; the input-independent Config remains opt-in for direct callers.
    auto_sampling: bool = False
    accuracy_interval: int = 10
    accuracy_particles: int = 100
    local_order: int = 3
    max_order: int = 3
    # tomogram_batches: each stochastic minibatch is drawn as whole tilt groups (tomograms) in a fresh random
    # order, so a tile holds a tomogram's particles instead of about one: the streamed engine projects every
    # pose once per tile. The pseudo-halves still split by particle parity. False: random particles.

    def __post_init__(self):
        if self.stages is None:
            object.__setattr__(self, "stages", default_stages(self.iterations))
        super().__post_init__()
        if isinstance(self.n_classes, bool) or not isinstance(self.n_classes, (int, np.integer)) or self.n_classes < 1:
            raise ValueError("K must be a positive integer")
        if self.q > 16:
            raise ValueError("The native PPCA kernels support at most 16 PCs per class")
        if not np.isfinite(self.class_pseudocount) or self.class_pseudocount <= 0:
            raise ValueError("class_pseudocount must be finite and positive")
        if not self.stream_coarse_recompute:
            raise ValueError("Mixture PPCA runs on the streamed engine (stream_coarse_recompute)")
        if self.oversampling_start is not None and (
            isinstance(self.oversampling_start, bool) or not isinstance(self.oversampling_start, (int, np.integer))
            or self.oversampling_start < 1
        ):
            raise ValueError("oversampling_start must be a positive iteration")
        if self.oversampling_start is not None and not self.oversampling:
            raise ValueError("oversampling_start needs oversampling 1")
        if self.tomogram_batches and self.balanced_stochastic_halves:
            raise ValueError("Tomogram batches and balanced stochastic halves are different draws; choose one")
        if self.optimizer != "vdam":
            raise ValueError("Mixture InitialModel currently uses the VDAM optimizer")
        for name in ("local_search_start", "accuracy_interval", "accuracy_particles", "local_order", "max_order"):
            value = getattr(self, name)
            if value is None and name == "local_search_start":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.local_search_sigma_deg is not None and not (
            np.isfinite(self.local_search_sigma_deg) and self.local_search_sigma_deg > 0
        ):
            raise ValueError("local_search_sigma_deg must be finite and positive")
        if self.local_order > self.max_order:
            raise ValueError("local_order cannot exceed max_order")
        if self.auto_sampling and not self.oversampling:
            raise ValueError("Auto-sampling needs the two adaptive passes (oversampling 1)")
        if self.auto_sampling and self.local_search_start is not None:
            raise ValueError("Auto-sampling decides when local searches start; drop local_search_start")
