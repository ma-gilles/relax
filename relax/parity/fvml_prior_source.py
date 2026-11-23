"""The FvML-mixture direction prior as the input source of a K=1 refinement (code rule 15).

``relax refine --fvml-direction-prior N`` replaces RELION's learned direction prior with a uniform component plus
``N`` FvML lobes fitted, after every numbered M-step, to each half's posterior mass over the directions (the learned
prior the run computed, ``relax.sampling.fvml_direction_prior``). Each half fits its own mixture from its own
particles, warm-started from its previous fit. The fitted prior then scores every expectation of the half, as a
fixed prior does (``relax.parity.fixed_direction_prior``): each global search, each local search (which keeps its
selection of orientations within three widths of each particle's pose and weights them by this prior alone, without
the Gaussian log priors) and the final pass. A mixture is a density on the sphere, so it is evaluated on whatever
HEALPix order a search samples, with no reset at an order change. The first iteration, before any fit, scores as
RELION's does (no direction prior).

The model STARs hold the mixture's cell probabilities as ``rlnOrientationDistribution`` (the prior the next
expectation scores with when its order is unchanged) and the fit in two extra blocks
(``relax.refinement.run_files``), which a continued run warm-starts from.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from relax.refinement.ports import InputSource
from relax.sampling import fvml_direction_prior
from relax.sampling.orientation_priors import DirectionPrior, HalfDirectionLogPriors

logger = logging.getLogger(__name__)

PROVENANCE_FILE = "fvml_direction_prior.json"


class FvmlDirectionPriorSource(InputSource):
    """The native source, except that each half's learned prior is the FvML mixture with ``n_lobes`` lobes fitted to
    its posterior masses, and every expectation after the first scores with it."""

    def __init__(self, n_lobes: int, *, dtype: np.dtype):
        if n_lobes < 1:
            raise ValueError(f"an FvML direction prior needs at least one lobe, got {n_lobes}")
        self.n_lobes = int(n_lobes)
        self.dtype = np.dtype(dtype)

    @classmethod
    def for_run(
        cls, n_lobes: int, replacing: InputSource, options, *, tomographic: bool, multi_gpu: bool
    ) -> FvmlDirectionPriorSource:
        """The source of a run with ``options`` whose prior is the ``n_lobes``-lobe mixture.

        ``replacing`` is the source the command chose: only the native one can be replaced. K=1, C1, single-particle
        runs on one GPU are admitted.
        """
        if type(replacing) is not InputSource:
            raise ValueError(
                "--fvml-direction-prior cannot be combined with a RELION replay, a frozen boundary or a fixed prior"
            )
        if options.k_class.n_classes != 1:
            raise ValueError("--fvml-direction-prior is for K=1 refinement")
        if options.symmetry.point_group != "C1":
            raise ValueError("--fvml-direction-prior is for C1 symmetry")
        if tomographic:
            raise ValueError("--fvml-direction-prior is for single-particle data")
        if multi_gpu:
            raise ValueError("--fvml-direction-prior runs on one GPU")
        return cls(n_lobes, dtype=options.precision.rotation_real_dtype)

    def learned_direction_priors(self, learned: tuple, *, previous: list) -> tuple:
        """Each half's mixture fitted to its ``learned`` masses, from its ``previous`` fit when it has one; a half
        without learned masses keeps its prior (None)."""
        priors = []
        for half_index, (masses, scored) in enumerate(zip(learned, previous, strict=True)):
            if masses is None:
                priors.append(None)
                continue
            order = int(masses.healpix_order)
            fit = fvml_direction_prior.fit_direction_prior(
                masses.values, order, n_lobes=self.n_lobes, previous=scored.fit, seed=half_index
            )
            fvml_direction_prior.log_fit(half_index, order, fit)
            values = np.exp(fvml_direction_prior.direction_log_probabilities(fit, order))
            priors.append(DirectionPrior(values.astype(np.asarray(masses.values).dtype), order, fit))
        return tuple(priors)

    def direction_log_prior(
        self, half_index: int, log_prior: HalfDirectionLogPriors, *, prior: DirectionPrior, healpix_order: int,
        use_local: bool,
    ) -> HalfDirectionLogPriors:
        """The half's mixture on the rotation grid of ``healpix_order``; the run's own ``log_prior`` before the half
        has a fit."""
        if prior.fit is None:
            return log_prior
        logger.info(
            "FvML direction prior half-%d: %s search at healpix_order=%d",
            half_index + 1, "local" if use_local else "global", healpix_order,
        )
        rotation_log_prior = fvml_direction_prior.rotation_log_prior(prior.fit, healpix_order, dtype=self.dtype)
        rotation_log_prior.setflags(write=False)
        return HalfDirectionLogPriors(rotation_log_prior=rotation_log_prior, class_rotation_log_prior=None)

    def write_provenance(self, directory: str | Path) -> Path:
        """Record the prior's settings in ``directory`` for whoever reads the run."""
        record = {
            "lobes": self.n_lobes,
            "fit_max_order": fvml_direction_prior.FIT_MAX_ORDER,
            "cell_levels": fvml_direction_prior.CELL_LEVELS,
            "cap_spacings": fvml_direction_prior.CAP_SPACINGS,
            "cold_starts": fvml_direction_prior.N_COLD_STARTS,
            "max_em_iterations": fvml_direction_prior.MAX_EM_ITERATIONS,
            "tol": fvml_direction_prior.TOL,
        }
        path = Path(directory) / PROVENANCE_FILE
        path.write_text(json.dumps(record, indent=2) + "\n")
        return path
