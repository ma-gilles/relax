"""A fixed direction prior as the input source of a K=1 refinement (code rule 15).

``relax refine --fixed-direction-prior PRIOR.npy`` scores every expectation of the run (each global search, each
local search and the final pass) with one given probability per HEALPix direction. RELION's own prior is learned per
half set, reset to uniform whenever the sampling order changes, and not applied in a local search. The learned prior
is still estimated and written to the run files, but it is not fed back, so a run shows how well the prior is
estimated and how much a given prior is worth to the refinement.

The file is a one-dimensional ``.npy`` of positive floats that sum to one, indexed like a model.star's
``rlnOrientationDistribution`` (HEALPix NEST order). Its length gives its HEALPix order; a coarser order is the
exact sum of each direction's children. ``max_healpix_order`` of the run must be given and at most the file's order.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from recovar.utils.file_hash import sha256_file

from relax.refinement.ports import InputSource
from relax.sampling.orientation_priors import (
    HalfDirectionLogPriors,
    infer_direction_prior_healpix_order,
    make_relion_direction_log_prior,
)

logger = logging.getLogger(__name__)

PROVENANCE_FILE = "fixed_direction_prior.json"
# The file's probabilities may sum to one only up to what a float32 file holds.
SUM_REL_TOL = 1e-6


@dataclass(frozen=True)
class FixedDirectionPrior:
    """A probability per HEALPix direction at ``order``, as read from ``path`` (whose digest is ``sha256``)."""

    values: np.ndarray
    order: int
    path: str
    sha256: str

    @classmethod
    def from_file(cls, path: str | Path) -> FixedDirectionPrior:
        """Read and validate a prior file: 1-D, a HEALPix direction count, finite, positive and summing to one."""
        values = np.load(path, allow_pickle=False)
        if not isinstance(values, np.ndarray) or values.ndim != 1 or not np.issubdtype(values.dtype, np.floating):
            raise ValueError(f"{path}: a fixed direction prior is the .npy of a 1-D float array")
        order = infer_direction_prior_healpix_order(values)
        values = values.astype(np.float64)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(
                f"{path}: every direction needs a finite positive prior "
                f"({int(np.sum(~(values > 0.0)))} of {values.size} are not); a zero prior removes a direction from "
                "every search, which can leave a local search with no hypothesis"
            )
        if not math.isclose(float(values.sum()), 1.0, rel_tol=SUM_REL_TOL):
            raise ValueError(f"{path}: the prior sums to {values.sum():.9g}, not 1")
        return cls(values=values, order=order, path=str(Path(path).resolve()), sha256=sha256_file(path))

    def at_order(self, order: int) -> np.ndarray:
        """The prior on the directions of HEALPix ``order`` <= the file's: each direction's children summed."""
        if order > self.order:
            raise ValueError(
                f"the run needs a direction prior at HEALPix order {order}, finer than {self.path} "
                f"(order {self.order})"
            )
        return self.values.reshape(-1, 4 ** (self.order - order)).sum(axis=1)


class FixedDirectionPriorSource(InputSource):
    """The native source, except that every expectation scores with ``prior``, on both halves."""

    def __init__(self, prior: FixedDirectionPrior, *, dtype: np.dtype):
        self.prior = prior
        self.dtype = np.dtype(dtype)
        self._rotation_log_priors: dict[int, np.ndarray] = {}

    @classmethod
    def for_run(
        cls, path: str | Path, replacing: InputSource, options, *, tomographic: bool, multi_gpu: bool
    ) -> FixedDirectionPriorSource:
        """The source of a run with ``options`` that scores with the prior in ``path``.

        ``replacing`` is the source the command chose: only the native one can be replaced. K=1, C1, single-particle
        runs on one GPU with a capped HEALPix order are admitted.
        """
        if type(replacing) is not InputSource:
            raise ValueError("--fixed-direction-prior cannot be combined with a RELION replay or a frozen boundary")
        if options.k_class.n_classes != 1:
            raise ValueError("--fixed-direction-prior is for K=1 refinement")
        if options.symmetry.point_group != "C1":
            raise ValueError("--fixed-direction-prior is for C1 symmetry")
        if tomographic:
            raise ValueError("--fixed-direction-prior is for single-particle data")
        if multi_gpu:
            raise ValueError("--fixed-direction-prior runs on one GPU")
        prior = FixedDirectionPrior.from_file(path)
        cap = options.schedule.max_healpix_order
        if cap is None or cap > prior.order:
            raise ValueError(
                f"--fixed-direction-prior needs --max_healpix_order of at most {prior.order}, the order of {path} "
                f"(got {'none' if cap is None else cap})"
            )
        return cls(prior, dtype=options.precision.rotation_real_dtype)

    def direction_log_prior(
        self, half_index: int, log_prior: HalfDirectionLogPriors, *, healpix_order: int, use_local: bool
    ) -> HalfDirectionLogPriors:
        """The fixed prior on the rotation grid of ``healpix_order``, whatever the half, the search or the run's own
        ``log_prior``."""
        logger.info(
            "Fixed direction prior half-%d: %s search at healpix_order=%d",
            half_index + 1, "local" if use_local else "global", healpix_order,
        )
        if healpix_order not in self._rotation_log_priors:
            expanded = make_relion_direction_log_prior(
                self.prior.at_order(healpix_order), healpix_order, symmetry="C1", dtype=self.dtype
            )
            expanded.setflags(write=False)
            self._rotation_log_priors[healpix_order] = expanded
        return HalfDirectionLogPriors(
            rotation_log_prior=self._rotation_log_priors[healpix_order], class_rotation_log_prior=None
        )

    def write_provenance(self, directory: str | Path) -> Path:
        """Record the prior file (path, digest, order) in ``directory`` for whoever reads the run."""
        record = {"path": self.prior.path, "sha256": self.prior.sha256, "healpix_order": self.prior.order}
        path = Path(directory) / PROVENANCE_FILE
        path.write_text(json.dumps(record, indent=2) + "\n")
        return path
