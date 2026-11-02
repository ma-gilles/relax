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
    # Under the data_vs_prior radius rule, the Fourier radius of the next iteration; None under the stages table.
    next_radius: int | None = None
    # Under the accuracy order rule, the HEALPix order of the next E-step (the final embeddings after the last
    # iteration); None under the stages table.
    next_order: int | None = None
