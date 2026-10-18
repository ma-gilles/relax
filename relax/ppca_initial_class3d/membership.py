"""Latest observed class posteriors, not an extra all-particle inference pass.

An update's E-step uses the model from the preceding update and the current
iteration's pose grid. Unvisited particles have no inferred class. These
diagnostics never enter model, noise, prior or optimizer updates.
"""

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FIELDS = ("particle_ids", "class_probabilities", "last_evaluated_iteration", "model_iteration")
# Pose memory (additive to v1): the marginal-MAP class's conditional MAP pose of the last evaluation, RELION
# Euler angles in degrees and the 3D (ET) or 2D-padded shift in pixels; NaN rows are unknown. A later update's
# local searches (``local_search``) start from them, as relion_refine starts from the data STAR's angles and
# offsets. Checkpoints without them load as unknown.
POSE_FIELDS = ("euler_deg", "translation_px")
SCHEMA = "relax-ppca-membership-v1"


def _poses(values, n):
    if values is None:
        return np.full((n, 3), np.nan, np.float32)
    values = np.asarray(values, np.float32)
    if values.shape != (n, 3):
        raise ValueError("Membership poses must be (n, 3) Euler-degree and pixel-shift rows")
    return values


def _ids(values):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iu" or np.any(values < 0):
        raise ValueError("Membership particle IDs must be nonnegative integers")
    if len(np.unique(values)) != len(values):
        raise ValueError("Membership particle IDs must be unique")
    return values


def _probabilities(values):
    if (not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1)
            or not np.allclose(values.sum(axis=1), 1, rtol=1e-6, atol=1e-6)):
        raise ValueError("Membership probabilities must be finite, nonnegative and normalized")


@dataclass
class Membership:
    particle_ids: np.ndarray
    class_probabilities: np.ndarray
    last_evaluated_iteration: np.ndarray
    model_iteration: np.ndarray
    euler_deg: np.ndarray | None = None
    translation_px: np.ndarray | None = None

    def __post_init__(self):
        self.euler_deg = _poses(self.euler_deg, len(self.particle_ids))
        self.translation_px = _poses(self.translation_px, len(self.particle_ids))

    @classmethod
    def empty(cls, particle_ids, n_classes):
        ids = np.sort(_ids(particle_ids)).copy()
        return cls(ids, np.full((len(ids), n_classes), np.nan, np.float32),
                   np.full(len(ids), -1, np.int64), np.full(len(ids), -1, np.int64))

    def update(self, particle_ids, probabilities, *, iteration, model_iteration, eulers_deg=None,
               translations_px=None):
        """Scatter one existing E-step tile by original physical-particle ID.

        ``eulers_deg`` and ``translations_px`` are the tile's MAP poses (unknown when absent). Returns the summed
        squared shift change and the count of particles that had a known shift before, RELION's
        ``sum_changes_optimal_offsets`` terms (ml_optimiser.cpp ``calculateRunningAveragesOfChanges``).
        """
        ids = _ids(particle_ids)
        probabilities = np.asarray(probabilities)
        if probabilities.shape != (len(ids), self.class_probabilities.shape[1]):
            raise ValueError("Membership probability shape differs from particle IDs/classes")
        _probabilities(probabilities)
        if iteration < 1 or model_iteration != iteration - 1:
            raise ValueError("Training membership must describe the pre-update model")
        rows = np.searchsorted(self.particle_ids, ids)
        if np.any(rows >= len(self.particle_ids)) or not np.array_equal(self.particle_ids[rows], ids):
            raise ValueError("Membership tile contains an unknown particle ID")
        if np.any(self.last_evaluated_iteration[rows] >= iteration):
            raise ValueError("Duplicate or out-of-order membership update")
        eulers_deg = _poses(eulers_deg, len(ids))
        translations_px = _poses(translations_px, len(ids))
        if (np.isfinite(eulers_deg).all(axis=1) != np.isfinite(translations_px).all(axis=1)).any():
            raise ValueError("Membership poses must give both angles and shifts of a particle or neither")
        previous = self.translation_px[rows]
        known = np.isfinite(previous).all(axis=1) & np.isfinite(translations_px).all(axis=1)
        change = float(np.sum((translations_px[known] - previous[known]) ** 2, dtype=np.float64))
        self.class_probabilities[rows] = probabilities
        self.last_evaluated_iteration[rows] = iteration
        self.model_iteration[rows] = model_iteration
        self.euler_deg[rows] = eulers_deg
        self.translation_px[rows] = translations_px
        return change, int(known.sum())

    def known_poses(self, particle_ids):
        """Rows of ``particle_ids`` and whether each has a stored pose."""
        ids = np.asarray(particle_ids, dtype=np.int64)
        rows = np.searchsorted(self.particle_ids, ids)
        if np.any(rows >= len(self.particle_ids)) or not np.array_equal(self.particle_ids[rows], ids):
            raise ValueError("Unknown membership particle ID")
        return rows, np.isfinite(self.euler_deg[rows]).all(axis=1)

    def validate(self, *, n_particles, n_classes, iteration):
        ids = _ids(self.particle_ids)
        if len(ids) != n_particles or not np.array_equal(ids, np.sort(ids)):
            raise ValueError("Membership IDs must cover the sorted particle set")
        if self.class_probabilities.shape != (n_particles, n_classes):
            raise ValueError("Membership probability shape differs from particle count/classes")
        if self.class_probabilities.dtype != np.float32:
            raise TypeError("Membership probabilities must be float32")
        for values in (self.last_evaluated_iteration, self.model_iteration):
            if values.shape != (n_particles,) or values.dtype.kind not in "iu":
                raise ValueError("Membership iteration metadata must be per-particle integers")
        seen = self.last_evaluated_iteration >= 1
        if (np.any(self.last_evaluated_iteration > iteration)
                or np.any(self.last_evaluated_iteration[~seen] != -1)
                or np.any(self.model_iteration[~seen] != -1)
                or np.any(self.model_iteration[seen] != self.last_evaluated_iteration[seen] - 1)):
            raise ValueError("Invalid membership evaluation/model iteration")
        if not np.all(np.isnan(self.class_probabilities[~seen])):
            raise ValueError("Unvisited membership probabilities must be NaN")
        _probabilities(self.class_probabilities[seen])
        for name in POSE_FIELDS:
            values = getattr(self, name)
            if values.shape != (n_particles, 3) or values.dtype != np.float32:
                raise ValueError("Membership poses must be float32 (n_particles, 3) rows")
            finite = np.isfinite(values)
            if np.any(finite[~seen]) or np.any(finite.any(axis=1) & ~finite.all(axis=1)):
                raise ValueError("Membership poses must be fully known or unknown, and unknown when unvisited")
        if not np.array_equal(np.isfinite(self.euler_deg).all(axis=1), np.isfinite(self.translation_px).all(axis=1)):
            raise ValueError("Membership angles and shifts must be known together")

    def arrays(self, checkpoint_iteration):
        seen = self.last_evaluated_iteration >= 1
        labels = np.full(len(seen), -1, np.int32)
        labels[seen] = self.class_probabilities[seen].argmax(axis=1)
        return {
            **{name: getattr(self, name) for name in FIELDS},
            **{name: getattr(self, name) for name in POSE_FIELDS},
            "class_labels": labels,
            "evaluated": seen,
            "evaluated_in_checkpoint_iteration": seen & (self.last_evaluated_iteration == checkpoint_iteration),
        }

    def export(self, checkpoint_path, checkpoint_iteration):
        """Small, atomic NPZ/TSV views alongside the authoritative checkpoint."""
        path = Path(checkpoint_path)
        arrays = self.arrays(checkpoint_iteration)
        metadata = {
            "schema": SCHEMA, "checkpoint": path.name, "checkpoint_iteration": checkpoint_iteration,
            "class_index_base": 0, "unassigned_label": -1,
            "particle_ids": "original dataset indices; zero-based native particle STAR rows for ET, image indices for SPA",
            "scope": "latest observed training E-step membership; NOT all-particle inference at the saved model",
            "posterior": "class posterior marginalized over poses and latent coordinates",
            "timing": "model_iteration is before the M-step; pose grid belongs to last_evaluated_iteration",
            "unvisited": "label -1, probabilities NaN, iteration fields -1",
            "poses": "euler_deg (rot, tilt, psi; RELION convention) and translation_px of the marginal-MAP class's "
                     "conditional MAP pose on the grid of last_evaluated_iteration; NaN rows unknown",
        }
        # Keep checkpoint_*.npz discovery in the run root unchanged.
        directory = path.parent / "memberships"
        directory.mkdir(exist_ok=True)
        target = directory / (path.stem + ".npz")
        temporary = target.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays, metadata=json.dumps(metadata))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        target = directory / (path.stem + ".tsv")
        temporary = target.with_suffix(".tmp")
        columns = ("particle_ids", "class_labels", "last_evaluated_iteration", "model_iteration",
                   "evaluated_in_checkpoint_iteration")
        with temporary.open("w", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow([*columns, *[f"class_probability_{k}" for k in range(self.class_probabilities.shape[1])]])
            for i in range(len(self.particle_ids)):
                writer.writerow([*[int(arrays[name][i]) for name in columns],
                                 *[format(float(p), ".9g") for p in self.class_probabilities[i]]])
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
