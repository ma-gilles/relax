"""The FvML-mixture direction prior as the input source of a refinement (code rule 15).

``relax refine --fvml-direction-prior N`` replaces RELION's learned direction prior with a uniform component plus
``N`` FvML lobes fitted, after every numbered M-step, to the posterior mass over the directions (the learned prior
the run computed, ``relax.sampling.fvml_direction_prior``), warm-started from the previous fit:

* K=1 (Refine3D): each half fits its own mixture from its own particles. The fitted prior scores every expectation
  of the half, as a fixed prior does (``relax.parity.fixed_direction_prior``): each global search, each local search
  (which keeps its selection of orientations within three widths of each particle's pose and weights them by this
  prior alone, without the Gaussian log priors) and the final pass.
* K classes (Class3D): RELION keeps one ``pdf_direction`` per class, the direction's distribution given the class,
  from the posteriors of all particles (the halves only split the work, and both hold the same learned prior).
  Each class's mixture is fitted to that conditional mass once and shared by the halves, and scores every search
  of the class: a global search's rotations, and a local search's (``--sigma_ang``) rows of that class, which
  RELION scores against every class at each particle's local orientations
  (``relax.local_search.layout.with_class_rotation_log_priors``).

Under a point group the directions are those of one asymmetric unit, the mixture is invariant under the group, and
``N`` counts the lobes of one unit. A mixture is a density on the sphere, so it is evaluated on whatever HEALPix
order a search samples, with no reset at an order change. The first iteration, before any fit, scores as RELION's
does.

The model STARs hold the mixtures' cell probabilities as ``rlnOrientationDistribution`` (the prior the next
expectation scores with when its order is unchanged) and each fit in two extra blocks
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
    """The native source, except that each learned prior (a K=1 half's, or each class's) is the FvML mixture with
    ``n_lobes`` lobes fitted to its posterior masses under point group ``symmetry``, and every expectation after
    the first scores with it."""

    def __init__(self, n_lobes: int, *, n_classes: int, symmetry: str, dtype: np.dtype):
        if n_lobes < 1:
            raise ValueError(f"an FvML direction prior needs at least one lobe, got {n_lobes}")
        if n_classes < 1:
            raise ValueError(f"an FvML direction prior needs at least one class, got {n_classes}")
        self.n_lobes = int(n_lobes)
        self.n_classes = int(n_classes)
        self.symmetry = symmetry
        self.dtype = np.dtype(dtype)

    @classmethod
    def for_run(
        cls, n_lobes: int, replacing: InputSource, options, *, tomographic: bool, multi_gpu: bool
    ) -> FvmlDirectionPriorSource:
        """The source of a run with ``options`` whose prior is the ``n_lobes``-lobe mixture.

        ``replacing`` is the source the command chose: only the native one can be replaced. Single-particle runs on
        one GPU are admitted, with any point group and class count; the asymmetric unit at the first sampling order
        (or the fit's coarsest, ``FIT_MAX_ORDER``) must hold at least ``n_lobes`` directions.
        """
        if type(replacing) is not InputSource:
            raise ValueError(
                "--fvml-direction-prior cannot be combined with a RELION replay, a frozen boundary or a fixed prior"
            )
        if tomographic:
            raise ValueError("--fvml-direction-prior is for single-particle data")
        if multi_gpu:
            raise ValueError("--fvml-direction-prior runs on one GPU")
        n_classes = int(options.k_class.n_classes)
        symmetry = options.symmetry.point_group
        first_order = min(int(options.schedule.init_healpix_order), fvml_direction_prior.FIT_MAX_ORDER)
        n_directions = fvml_direction_prior.direction_pixels(first_order, symmetry).size
        if n_directions < n_lobes:
            raise ValueError(
                f"--fvml-direction-prior {n_lobes}: the {symmetry} asymmetric unit has only {n_directions} directions "
                f"at HEALPix order {first_order}; start at a finer order (--healpix_order) or fit fewer lobes"
            )
        return cls(n_lobes, n_classes=n_classes, symmetry=symmetry, dtype=options.precision.rotation_real_dtype)

    def learned_direction_priors(self, learned: tuple, *, previous: list) -> tuple:
        """The next direction prior of each half from the run's ``learned`` ones (None: the half keeps its prior),
        each warm-started from the fits of the half's ``previous`` prior when it has them."""
        if self.n_classes == 1:
            return self._learned_k1(learned, previous)
        return self._learned_classes(learned, previous)

    def _learned_k1(self, learned: tuple, previous: list) -> tuple:
        """Each half's mixture fitted to its own masses (cold starts seeded by the half)."""
        priors = []
        for half_index, (masses, scored) in enumerate(zip(learned, previous, strict=True)):
            if masses is None:
                priors.append(None)
                continue
            order = int(masses.healpix_order)
            fit = fvml_direction_prior.fit_direction_prior(
                masses.values, order, symmetry=self.symmetry, n_lobes=self.n_lobes,
                previous=None if scored.fits is None else scored.fits[0], seed=half_index,
            )
            fvml_direction_prior.log_fit(f"half-{half_index + 1}", order, fit)
            values = np.exp(fvml_direction_prior.direction_log_probabilities(fit, order))
            priors.append(DirectionPrior(values.astype(np.asarray(masses.values).dtype), order, (fit,)))
        return tuple(priors)

    def _learned_classes(self, learned: tuple, previous: list) -> tuple:
        """Each class's mixture fitted once to its conditional masses (cold starts seeded by the class), the same
        prior for both halves; the halves must hold the same learned prior, as Class3D's one model does."""
        if all(masses is None for masses in learned):
            return tuple(learned)
        if any(masses is None for masses in learned) or not all(
            masses.healpix_order == learned[0].healpix_order and np.array_equal(masses.values, learned[0].values)
            for masses in learned
        ):
            raise ValueError("a Class3D run's halves must hold the same learned direction prior")
        order = int(learned[0].healpix_order)
        values = np.asarray(learned[0].values)
        if values.shape[0] != self.n_classes:
            raise ValueError(f"a learned prior of {values.shape[0]} classes, the run has {self.n_classes}")
        warm = previous[0].fits
        fits = []
        for class_index in range(self.n_classes):
            fit = fvml_direction_prior.fit_direction_prior(
                values[class_index], order, symmetry=self.symmetry, n_lobes=self.n_lobes,
                previous=None if warm is None else warm[class_index], seed=class_index,
            )
            fvml_direction_prior.log_fit(f"class {class_index + 1}", order, fit)
            fits.append(fit)
        fitted = np.stack([np.exp(fvml_direction_prior.direction_log_probabilities(fit, order)) for fit in fits])
        prior = DirectionPrior(fitted.astype(values.dtype), order, tuple(fits))
        return tuple(prior for _ in learned)

    def direction_log_prior(
        self, half_index: int, log_prior: HalfDirectionLogPriors, *, prior: DirectionPrior, healpix_order: int,
        use_local: bool,
    ) -> HalfDirectionLogPriors:
        """The half's mixtures on the rotation grid of ``healpix_order`` (one per class in Class3D), for a global
        or a local search alike; the run's own ``log_prior`` before the half has fits."""
        if prior.fits is None:
            return log_prior
        logger.info(
            "FvML direction prior half-%d: %s search at healpix_order=%d",
            half_index + 1, "local" if use_local else "global", healpix_order,
        )
        tables = [
            fvml_direction_prior.rotation_log_prior(fit, healpix_order, dtype=self.dtype) for fit in prior.fits
        ]
        if self.n_classes == 1:
            rotation_log_prior = tables[0]
            rotation_log_prior.setflags(write=False)
            return HalfDirectionLogPriors(rotation_log_prior=rotation_log_prior, class_rotation_log_prior=None)
        class_rotation_log_prior = np.stack(tables)
        class_rotation_log_prior.setflags(write=False)
        return HalfDirectionLogPriors(rotation_log_prior=None, class_rotation_log_prior=class_rotation_log_prior)

    def write_provenance(self, directory: str | Path) -> Path:
        """Record the prior's settings in ``directory`` for whoever reads the run."""
        record = {
            "lobes": self.n_lobes,
            "classes": self.n_classes,
            "symmetry": self.symmetry,
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
