"""How a half's particles are cut into shards for several GPUs, and the order the shards are handed out.

A shard is a set of one half's particles scored in one call (:mod:`relax.steps.tomo_shards`). A global coarse pass
whose tilt frames are shared cuts its particles in groups (``tomo_coarse.CoarsePassPlan.frame_order``, ``flush_size``
at a time) whose composition decides which near-cut rotations it scores again, so its shards are made of whole
groups (:func:`cut_group_shards`); other passes cut any particle set alike (:func:`cut_shards`). The cut follows
each particle's predicted cost (its significant samples in the previous iteration times its tilt images):
a particle predicted to cost at least ``1 / (2 * ranks)`` of its half is a shard of its own, since one GPU
must run it alone; the others are cut, in particle order, into ``2 * ranks`` blocks of about equal predicted
cost, so the particles of one tomogram mostly stay together. The shards of both halves are then handed out
heaviest first, each to the first rank that is free (longest-processing-time first).

Neither the cut nor the order changes a result: each particle is scored as in one call over the half, and
the shards are joined in shard order (:func:`relax.steps.score_outputs.merge_half_shards`), whichever rank
ran them and whenever.
"""

from __future__ import annotations

import numpy as np


def particle_costs(significant_counts, images_per_particle) -> np.ndarray:
    """Each particle's predicted cost: its significant samples (previous iteration; 1 when unknown) times its
    tilt images."""

    images = np.asarray(images_per_particle, dtype=np.float64).reshape(-1)
    if significant_counts is None:
        return images
    counts = np.asarray(significant_counts, dtype=np.float64).reshape(-1)
    if counts.shape != images.shape:
        raise ValueError(f"{counts.size} significant counts for {images.size} particles")
    return np.maximum(counts, 1.0) * images


def cut_shards(costs, ranks: int) -> list[np.ndarray]:
    """One half's shards: the particles predicted at or above ``1 / (2 * ranks)`` of the half's cost alone, the
    others in up to ``2 * ranks`` contiguous blocks of about equal cost. Shards are in particle order (by first
    particle); every particle is in exactly one."""

    costs = np.asarray(costs, dtype=np.float64).reshape(-1)
    if ranks < 1:
        raise ValueError(f"shards need at least one rank, got {ranks}")
    if costs.size == 0:
        return []
    if np.any(~np.isfinite(costs)) or np.any(costs < 0):
        raise ValueError("particle costs must be finite and non-negative")
    total = float(costs.sum())
    lone = costs >= total / (2 * ranks) if total > 0 else np.zeros(costs.size, dtype=bool)
    shards = [np.array([unit], dtype=np.int64) for unit in np.flatnonzero(lone)]
    rest = np.flatnonzero(~lone)
    if rest.size:
        n_blocks = min(2 * ranks, rest.size)
        # Block b ends where the running cost first reaches (b + 1) / n_blocks of the rest's cost.
        running = np.cumsum(costs[rest])
        edges = np.searchsorted(running, running[-1] * np.arange(1, n_blocks) / n_blocks, side="left") + 1
        edges = np.unique(np.clip(edges, 1, rest.size - 1)) if rest.size > 1 else np.zeros(0, dtype=np.int64)
        shards.extend(block for block in np.split(rest, edges) if block.size)
    shards.sort(key=lambda shard: int(shard[0]))
    return shards


def cut_group_shards(costs, groups, ranks: int) -> list[np.ndarray]:
    """One half's shards made of whole ``groups`` (lists of particles that must be scored together, in order;
    together every particle once): a group holding a particle at or above ``1 / (2 * ranks)`` of the half's cost
    is a shard of its own, the others go in up to ``2 * ranks`` runs of consecutive groups of about equal cost.
    Each shard keeps its groups' particles in group order."""

    costs = np.asarray(costs, dtype=np.float64).reshape(-1)
    groups = [np.asarray(group, dtype=np.int64).reshape(-1) for group in groups]
    if ranks < 1:
        raise ValueError(f"shards need at least one rank, got {ranks}")
    if not groups:
        return []
    joined = np.concatenate(groups)
    if joined.size != costs.size or not np.array_equal(np.sort(joined), np.arange(costs.size)):
        raise ValueError("the groups must hold every particle once")
    total = float(costs.sum())
    heavy = [total > 0 and float(costs[group].max()) >= total / (2 * ranks) for group in groups]
    shards = [group for group, alone in zip(groups, heavy) if alone]
    rest = [group for group, alone in zip(groups, heavy) if not alone]
    if rest:
        n_runs = min(2 * ranks, len(rest))
        # Run b ends where the running cost first reaches (b + 1) / n_runs of the rest's cost.
        running = np.cumsum([float(costs[group].sum()) for group in rest])
        edges = np.searchsorted(running, running[-1] * np.arange(1, n_runs) / n_runs, side="left") + 1
        edges = np.unique(np.clip(edges, 1, len(rest) - 1)) if len(rest) > 1 else []
        bounds = [0, *[int(e) for e in edges], len(rest)]
        shards.extend(np.concatenate(rest[a:b]) for a, b in zip(bounds[:-1], bounds[1:]) if b > a)
    return shards


def heaviest_first(shard_costs) -> list:
    """The order shards are handed out: by predicted cost, heaviest first; ties keep their given order.
    ``shard_costs`` is ``[(key, cost), ...]``; returns the keys."""

    order = sorted(range(len(shard_costs)), key=lambda i: (-float(shard_costs[i][1]), i))
    return [shard_costs[i][0] for i in order]
