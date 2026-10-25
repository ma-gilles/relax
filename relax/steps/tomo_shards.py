"""A subtomogram half scored in shards: disjoint subsets of the half's particles, each scored by
:func:`relax.steps.tomo_scoring.score_tomo_half_in_loop` against the half's model operands, and joined
by :func:`relax.steps.score_outputs.merge_half_shards`.

A shard changes only which particles one call scores. Every particle keeps the half's reference, noise,
projector, scale groups and coarse orientations: in a local search the coarse grid is the half's union of
the particles' orientations (:class:`TomoLocalRotations`), built once per half by the caller, not the
shard's own union. Only the K=1 subtomogram route is sharded.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from relax.steps.half_inputs import HalfScoringData, HalfSet, TiltPrior
from relax.steps.tomo_half import TomoLocalRotations

# The per-particle fields of a half (HalfSet) and of its scoring data (HalfScoringData): one row per particle.
HALF_SET_UNIT_FIELDS = (
    "optics_group_ids",
    "rotation_eulers",
    "translations",
    "image_corrections",
    "scale_corrections",
    "group_ids",
)
SCORING_DATA_UNIT_FIELDS = ("scale_group_ids", "image_seed_classes")


def _unit_rows(value, units: np.ndarray, n_units: int, name: str):
    if value is None:
        return None
    array = np.asarray(value)
    if array.shape[:1] != (n_units,):
        raise ValueError(f"{name} has {array.shape[:1]} rows for a half of {n_units} particles")
    return array[units]


def half_set_shard(half: HalfSet, units: np.ndarray) -> HalfSet:
    """The half's particles ``units`` (half-local indices, in this order), with their poses and corrections."""

    return dataclasses.replace(
        half,
        dataset=half.dataset.subset(units),
        **{name: _unit_rows(getattr(half, name), units, half.dataset.n_units, f"HalfSet.{name}") for name in HALF_SET_UNIT_FIELDS},
    )


def scoring_data_shard(data: HalfScoringData, units: np.ndarray) -> HalfScoringData:
    """The scoring data of the half's particles ``units``: their particles and per-particle groups, with the
    half's model operands."""

    n_units = data.particles.dataset.n_units
    return dataclasses.replace(
        data,
        particles=half_set_shard(data.particles, units),
        tilt_prior=(
            None
            if data.tilt_prior is None
            else TiltPrior(
                sigma_deg=data.tilt_prior.sigma_deg,
                centre_deg=_unit_rows(data.tilt_prior.centre_deg, units, n_units, "TiltPrior.centre_deg"),
            )
        ),
        **{
            name: _unit_rows(getattr(data, name), units, n_units, f"HalfScoringData.{name}")
            for name in SCORING_DATA_UNIT_FIELDS
        },
    )


def local_rotations_shard(rotations: TomoLocalRotations, units: np.ndarray) -> TomoLocalRotations:
    """The particles ``units``' own orientations, still indexing the half's union (the coarse grid every
    shard of the half scores on)."""

    return TomoLocalRotations(
        coarse_rotation_ids=rotations.coarse_rotation_ids,
        unit_rotation_ids=tuple(rotations.unit_rotation_ids[u] for u in units),
        unit_rotation_log_priors=tuple(rotations.unit_rotation_log_priors[u] for u in units),
    )


@dataclasses.dataclass(frozen=True)
class TomoHalfJob:
    """One subtomogram half's scoring call, ``score_tomo_half_in_loop(data, local_rotations=...,
    unit_seed_classes=..., **arguments)``, kept apart so that its particles can be scored in shards.

    ``data``, ``local_rotations`` (None on a global search) and ``unit_seed_classes`` (None, or one class per
    particle) are the whole half's; ``arguments`` are the call's other keyword arguments, which every shard
    shares.
    """

    data: HalfScoringData
    local_rotations: TomoLocalRotations | None
    unit_seed_classes: np.ndarray | None
    arguments: dict

    @property
    def n_units(self) -> int:
        return int(self.data.particles.dataset.n_units)

    def score(self):
        """The whole half in one call (the one-GPU run's call)."""

        from relax.steps.tomo_scoring import score_tomo_half_in_loop

        return score_tomo_half_in_loop(
            self.data,
            local_rotations=self.local_rotations,
            unit_seed_classes=self.unit_seed_classes,
            **self.arguments,
        )

    def coarse_plan(self):
        """The coarse pass plan of the whole half's call (``tomo_coarse.CoarsePassPlan``, with its frame order),
        sized on this device now; nothing is scored."""

        from relax.scoring.tomo_coarse import CoarsePlanUse
        from relax.steps.tomo_scoring import score_tomo_half_in_loop

        return score_tomo_half_in_loop(
            self.data,
            local_rotations=self.local_rotations,
            unit_seed_classes=self.unit_seed_classes,
            **self.arguments,
            plan_use=CoarsePlanUse(plan_only=True),
        )

    def with_coarse_plan(self, plan) -> "TomoHalfJob":
        """The call scoring with the coarse pass plan ``plan`` (the whole half's, for its shards)."""

        from relax.scoring.tomo_coarse import CoarsePlanUse

        return dataclasses.replace(
            self, arguments={**self.arguments, "plan_use": CoarsePlanUse(pinned=plan._replace(frame_order=None))}
        )

    def cut(self, costs, ranks: int) -> tuple["TomoHalfJob", list]:
        """The call its shards are taken from and the shards for ``ranks`` ranks, from the particles' predicted
        ``costs``: the call scores with the whole half's coarse plan, and a frame-shared global pass's shards are
        whole groups of its frame order (``relax.parallel.shards``).

        The ``--firstiter_cc`` iteration has no coarse pass (each particle's best CC is its support): its half is one
        shard, the whole call, so the pool scores it on one rank per half."""

        from relax.parallel.shards import cut_group_shards, cut_shards

        if self.arguments.get("normalized_cc"):
            return self, [np.arange(self.n_units, dtype=np.int64)]
        plan = self.coarse_plan()
        if plan.frame_order is None:
            return self.with_coarse_plan(plan), cut_shards(costs, ranks)
        size = max(plan.flush_size, 1)
        groups = [plan.frame_order[start : start + size] for start in range(0, self.n_units, size)]
        return self.with_coarse_plan(plan), cut_group_shards(costs, groups, ranks)

    def shard(self, units) -> "TomoHalfJob":
        """The call for the half's particles ``units`` (half-local indices, in this order). A shard of a call with
        the half's coarse plan (:meth:`with_coarse_plan`) and made of whole groups of the plan's frame order scores
        each particle as the half's call does."""

        units = np.asarray(units, dtype=np.int64).reshape(-1)
        if np.array_equal(units, np.arange(self.n_units)):
            return self  # the whole half: the call itself
        if units.size == 0 or units.min() < 0 or units.max() >= self.n_units:
            raise ValueError(f"a shard takes 1 to {self.n_units} of the half's particles, got {units.size}")
        if self.arguments.get("class_log_priors") is not None:
            raise NotImplementedError("only the K=1 subtomogram route is scored in shards")
        return TomoHalfJob(
            data=scoring_data_shard(self.data, units),
            local_rotations=None
            if self.local_rotations is None
            else local_rotations_shard(self.local_rotations, units),
            unit_seed_classes=_unit_rows(self.unit_seed_classes, units, self.n_units, "unit_seed_classes"),
            arguments=self.arguments,
        )

    def score_in_shards(self, shards):
        """The half scored shard by shard in this process and joined (``merge_half_shards``): ``shards`` are
        lists of the half's particles that together cover each particle once."""

        from relax.steps.score_outputs import merge_half_shards

        shards = [np.asarray(units, dtype=np.int64).reshape(-1) for units in shards]
        return merge_half_shards([self.shard(units).score() for units in shards], shards, self.n_units)


@dataclasses.dataclass(frozen=True)
class DeferredTomoHalf:
    """A half's scorer stopped at its subtomogram call: ``job`` is the call, and ``complete(result)`` makes the
    scorer's return value from the call's result (what the scorer does after the call). A multi-GPU run scores
    the job's shards on its ranks and completes the half on rank 0."""

    job: TomoHalfJob
    complete: object
