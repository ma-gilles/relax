"""RELION's expected-accuracy trial order: the particle-row shuffles of ``Experiment::randomiseParticlesOrder``.

The estimator that scores the trials is ``relax.steps.expected_accuracy``.
"""

from __future__ import annotations

import numpy as np

from relax.numerics import relion_random


def relion_auto_refine_half_orders(
    random_subsets,
    random_seed: int,
    first_iteration: int = 1,
    *,
    optics_group_ids=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return RELION's fresh AutoRefine particle-row order for both halves.

    RELION 5.0.1 (f2c1a384) shuffles half 1 and then half 2 with one
    ``mt19937(seed + iter)`` generator (``std::shuffle``) and stable-sorts each
    half by numeric optics group (``Experiment::randomiseParticlesOrder``,
    exp_model.cpp:406-456). Returned values index the supplied full particle
    table, including the first 100 trials consumed by the expected-accuracy
    estimator.
    """
    subsets = np.asarray(random_subsets, dtype=np.int64).reshape(-1)
    if set(subsets.tolist()) != {1, 2}:
        raise ValueError("random_subsets must contain exactly RELION half labels 1 and 2")
    base_orders = (
        np.flatnonzero(subsets == 1).astype(np.int64),
        np.flatnonzero(subsets == 2).astype(np.int64),
    )
    positions = relion_random.shuffled_orders(
        [base_orders[0].size, base_orders[1].size],
        int(random_seed) + int(first_iteration),
    )
    orders = [
        base[np.asarray(permutation, dtype=np.int64)]
        for base, permutation in zip(base_orders, positions, strict=True)
    ]

    if optics_group_ids is not None:
        optics = np.asarray(optics_group_ids, dtype=np.int64).reshape(-1)
        if optics.shape != subsets.shape:
            raise ValueError(
                f"optics_group_ids must have shape {subsets.shape}, got {optics.shape}"
            )
        orders = [
            order[np.argsort(optics[order], kind="stable")]
            for order in orders
        ]
    return orders[0], orders[1]


def relion_half1_trial_order(
    n_particles: int,
    random_seed: int,
    first_iteration: int = 1,
    *,
    base_order_local=None,
    optics_group_ids=None,
) -> np.ndarray:
    """Return RELION's one-time randomized half-1 local particle order.

    ``Experiment::randomiseParticlesOrder`` shuffles half 1 first with a fresh
    ``mt19937(random_seed + iter)``, so its order does not depend on half 2.
    Full-data AutoRefine performs that shuffle only once per process; a fresh
    refinement therefore uses ``first_iteration=1``.
    """
    shuffled_positions = relion_random.shuffled_orders([int(n_particles)], int(random_seed) + int(first_iteration))[0]
    if base_order_local is None:
        base_order = np.arange(int(n_particles), dtype=np.int64)
    else:
        base_order = np.asarray(base_order_local, dtype=np.int64).reshape(-1)
        if base_order.shape != (int(n_particles),):
            raise ValueError(
                f"base_order_local must have shape ({n_particles},), got {base_order.shape}",
            )
        if not np.array_equal(np.sort(base_order), np.arange(int(n_particles), dtype=np.int64)):
            raise ValueError("base_order_local must be a permutation of half-local particle indices")
    order = base_order[shuffled_positions]
    if optics_group_ids is not None:
        optics = np.asarray(optics_group_ids, dtype=np.int64).reshape(-1)
        if optics.shape != (int(n_particles),):
            raise ValueError(f"optics_group_ids must have shape ({n_particles},), got {optics.shape}")
        # Experiment::randomiseParticlesOrder stable-sorts the already
        # shuffled half by numeric optics group.
        order = order[np.argsort(optics[order], kind="stable")]
    return order


def relion_class3d_trial_layout(
    sorted_rows,
    random_seed: int,
    first_iteration: int = 1,
    *,
    optics_group_ids=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Class3D's expected-accuracy trial order and RELION particle ids.

    Without random halves ``Experiment::randomiseParticlesOrder`` shuffles the
    whole micrograph-sorted vector once with ``std::shuffle(mt19937(seed +
    iter))`` and stable-sorts it by optics group (exp_model.cpp:449-456); the
    first 100 entries are the trials (ml_optimiser.cpp:3930-3934).
    ``sorted_rows[j]`` is the input row of RELION ``part_id`` ``j``. Returns
    the trial order over input rows and each input row's ``part_id``, which
    seeds the per-trial random draws.
    """
    rows = np.asarray(sorted_rows, dtype=np.int64).reshape(-1)
    n_particles = int(rows.size)
    if not np.array_equal(np.sort(rows), np.arange(n_particles, dtype=np.int64)):
        raise ValueError("sorted_rows must be a permutation of input rows")
    positions = relion_random.shuffled_orders([n_particles], int(random_seed) + int(first_iteration))[0]
    order = rows[positions]
    if optics_group_ids is not None:
        optics = np.asarray(optics_group_ids, dtype=np.int64).reshape(-1)
        if optics.shape != (n_particles,):
            raise ValueError(f"optics_group_ids must have shape ({n_particles},), got {optics.shape}")
        order = order[np.argsort(optics[order], kind="stable")]
    particle_ids = np.empty(n_particles, dtype=np.int64)
    particle_ids[rows] = np.arange(n_particles, dtype=np.int64)
    return order, particle_ids


def prepare_relion_half1_trial_order(
    *,
    expected_accuracy,
    half1_dataset,
    optimizer_random_seed,
    init_relion_iteration,
    log,
) -> np.ndarray | None:
    """Resolve the controller's half-1 expected-accuracy trial order.

    Explicit metadata must be a full local permutation. Native setup failures
    retain the existing warning/None fallback, so unavailable exact accuracy
    cannot trigger convergence. The caller supplies its logging context.
    """
    expected_accuracy_trial_order = None
    if expected_accuracy.half1_trial_order_local is not None:
        expected_accuracy_trial_order = np.asarray(
            expected_accuracy.half1_trial_order_local,
            dtype=np.int64,
        ).reshape(-1)
        expected_trial_count = int(half1_dataset.n_units)
        if expected_accuracy_trial_order.shape != (expected_trial_count,):
            raise ValueError(
                "expected_accuracy.half1_trial_order_local must have shape "
                f"({expected_trial_count},), got {expected_accuracy_trial_order.shape}"
            )
        if not np.array_equal(
            np.sort(expected_accuracy_trial_order),
            np.arange(expected_trial_count, dtype=np.int64),
        ):
            raise ValueError(
                "expected_accuracy.half1_trial_order_local must be a permutation "
                "of half-1 local particle indices"
            )
        log.info(
            "RELION expected accuracy consumes an explicit physical trial order "
            "(%d particles)",
            expected_trial_count,
        )
    elif optimizer_random_seed is not None and int(half1_dataset.n_units) > 0:
        try:
            expected_accuracy_trial_order = relion_half1_trial_order(
                int(half1_dataset.n_units),
                int(optimizer_random_seed),
                first_iteration=max(1, int(init_relion_iteration) + 1),
                base_order_local=expected_accuracy.half1_base_order_local,
                optics_group_ids=expected_accuracy.half1_optics_group_ids,
            )
        except Exception as exc:
            expected_accuracy_trial_order = None
            log.warning("RELION exact expected-accuracy particle order unavailable: %s", exc)

    return expected_accuracy_trial_order
