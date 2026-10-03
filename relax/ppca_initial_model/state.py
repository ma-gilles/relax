"""One shared model, two gradient histories; these are not gold-standard maps."""

from dataclasses import dataclass

from relax.ppca_initial_model.update import Moments


@dataclass
class State:
    theta: object
    moments: Moments | None
    noise: object
    iteration: int
    order: object
    rng_state: dict
    offset_variance: float
    radius: int
    initialization: dict

    direction_prior: object = None
    direction_order: int = -1
    sgd_momentum: object = None
    # Per-particle contrast point estimates (Config.contrast_estimate, algorithm section 16.10); None without.
    contrast: object = None
