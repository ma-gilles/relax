"""Strict RELION MPI-follower group-scale state and its reduction.

How far RELION 5.0.1 reduces the group-scale XA/AA statistics across
followers depends on how the oracle combined its weighted sums. Line numbers
refer to RELION f2c1a38 (read-only tree /scratch/gpfs/GILLES/mg6942/relion;
its ml_model.cpp is byte-identical to the dispatch-capture oracle source):

* ``combine_weights_thru_disc = !--dont_combine_weights_via_disc``
  (ml_optimiser.cpp:708) selects ``combineAllWeightedSumsViaFile`` or
  ``combineAllWeightedSums`` (ml_optimiser_mpi.cpp:3951-3954).
* Without the flag, ``combineAllWeightedSumsViaFile`` (ml_optimiser_mpi.cpp:1924)
  writes one ``MlWsumModel::pack(Mpack)`` per follower (:1941). That message
  loops over the member ``nr_groups`` (ml_model.cpp:1920 pack, :2003 unpack),
  so every physical group is reduced.
* With the flag, ``combineAllWeightedSums`` (ml_optimiser_mpi.cpp:2028) uses a
  single collective pack only under ``USE_MPI_COLLECTIVE`` (:2048), which
  CMakeLists.txt:248,254 defines for SYCL/ALTCPU builds only; CUDA builds use
  the segmented ``pack(Mpack, piece, nr_pieces)`` (:2078). Its local
  ``nr_groups = sigma2_noise.size()`` (ml_model.cpp:2056 pack, :2241 unpack)
  is the optics-group count (``sigma2_noise.resize(nr_optics_groups)``,
  ml_model.cpp:57). In an all-data Class3D run with more scale groups than
  optics groups only the leading optics-group-sized prefix is MPI-reduced; the
  remaining scale statistics and resulting scale vectors stay follower-local
  (per-rank dump evidence in docs/math/em_parity_program.md).

The mode is read from the oracle's recorded command line. These helpers
reproduce both behaviors without leaking them into ordinary RECOVAR
refinement. ``RelionFollowerScaleState`` holds the follower-local scale
vectors; its construction, per-group selection, update and restart checks
moved verbatim from ``relax.refinement.follower_scale``, which prepares the
follower topology and dispatches these updates.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RelionFollowerScaleState:
    """Follower-local physical-group scale vectors and normalization counts."""

    scales: np.ndarray
    group_counts: np.ndarray
    n_optics_groups: int

    @property
    def n_followers(self) -> int:
        return int(self.scales.shape[0])

    @property
    def n_groups(self) -> int:
        return int(self.scales.shape[1])


def validate_relion_follower_scale_start(*, n_followers: int, init_relion_iteration: int) -> None:
    """Reject continuations whose follower-local scale state is unavailable.

    A RELION model STAR serializes only follower 1's scale vector.  Therefore
    it cannot initialize the complete MPI follower state after iteration 0.
    Supporting that case requires an explicit checkpoint containing every
    follower vector; no such input is currently accepted by the refinement
    API.
    """

    if int(n_followers) > 0 and int(init_relion_iteration) > 0:
        raise ValueError(
            "RELION follower-local scale emulation cannot cold-start at "
            f"init_relion_iteration={int(init_relion_iteration)} from a leader-serialized STAR; "
            "start from iteration 0 (full follower-scale checkpoint input is not implemented)"
        )


def make_relion_follower_scale_state(
    *,
    n_followers: int,
    group_counts,
    n_optics_groups: int,
    initial_group_scales=None,
) -> RelionFollowerScaleState:
    """Construct a validated follower state, initially equal on all followers."""

    if n_followers < 1:
        raise ValueError(f"n_followers must be positive, got {n_followers}")
    counts = np.asarray(group_counts, dtype=np.float64).reshape(-1)
    if counts.size < 1 or np.any(~np.isfinite(counts)) or np.any(counts < 0.0):
        raise ValueError("group_counts must be a non-empty finite non-negative vector")
    if n_optics_groups < 0 or n_optics_groups > counts.size:
        raise ValueError(
            f"n_optics_groups must be in [0, {counts.size}], got {n_optics_groups}",
        )
    if initial_group_scales is None:
        initial = np.ones(counts.size, dtype=np.float64)
    else:
        initial = np.asarray(initial_group_scales, dtype=np.float64).reshape(-1)
        if initial.shape != counts.shape:
            raise ValueError(
                f"initial_group_scales has shape {initial.shape}, expected {counts.shape}",
            )
        if np.any(~np.isfinite(initial)) or np.any(initial <= 0.0):
            raise ValueError("initial_group_scales must be finite and positive")
    return RelionFollowerScaleState(
        scales=np.repeat(initial[None, :], n_followers, axis=0),
        group_counts=counts.copy(),
        n_optics_groups=n_optics_groups,
    )


def relion_worker_group_ids(group_ids, follower_owners, *, n_groups: int) -> np.ndarray:
    """Encode ``(follower, physical group)`` as one accumulator group axis."""

    groups = np.asarray(group_ids, dtype=np.int64).reshape(-1)
    owners = np.asarray(follower_owners, dtype=np.int64).reshape(-1)
    if groups.shape != owners.shape:
        raise ValueError("group_ids and follower_owners must have the same shape")
    if n_groups < 1:
        raise ValueError(f"n_groups must be positive, got {n_groups}")
    if groups.size and (int(np.min(groups)) < 0 or int(np.max(groups)) >= n_groups):
        raise ValueError("group_ids are out of bounds")
    if owners.size and int(np.min(owners)) < 0:
        raise ValueError("follower_owners must be non-negative")
    return owners * n_groups + groups


def select_relion_follower_scales(
    state: RelionFollowerScaleState,
    *,
    group_ids,
    follower_owners,
) -> np.ndarray:
    """Select the current runtime scale for each particle."""

    groups = np.asarray(group_ids, dtype=np.int64).reshape(-1)
    owners = np.asarray(follower_owners, dtype=np.int64).reshape(-1)
    if groups.shape != owners.shape:
        raise ValueError("group_ids and follower_owners must have the same shape")
    if groups.size and (int(np.min(groups)) < 0 or int(np.max(groups)) >= state.n_groups):
        raise ValueError("group_ids are out of bounds")
    if owners.size and (int(np.min(owners)) < 0 or int(np.max(owners)) >= state.n_followers):
        raise ValueError("follower_owners are out of bounds")
    return np.asarray(state.scales[owners, groups], dtype=np.float64)


RELION_SCALE_REDUCTION_OPTICS_PREFIX = "optics_prefix"


RELION_SCALE_REDUCTION_ALL_GROUPS = "all_groups"


RELION_SCALE_REDUCTION_MODES = (RELION_SCALE_REDUCTION_OPTICS_PREFIX, RELION_SCALE_REDUCTION_ALL_GROUPS)


def relion_scale_reduction_mode_from_command(command: str) -> str:
    """Return how the RELION run behind ``command`` reduced group-scale XA/AA.

    See the module docstring: the MPI (segmented) combine reduces only the
    optics-group prefix, the file combine reduces every physical group. A CPU
    command with ``--dont_combine_weights_via_disc`` is rejected because an
    ALTCPU/SYCL build would use single collective messages instead.
    """
    tokens = str(command).split()
    if "--dont_combine_weights_via_disc" not in tokens:
        return RELION_SCALE_REDUCTION_ALL_GROUPS
    if "--gpu" not in tokens:
        raise ValueError(
            "cannot infer RELION group-scale reduction for a CPU run with "
            "--dont_combine_weights_via_disc: ALTCPU/SYCL builds use collective single messages"
        )
    return RELION_SCALE_REDUCTION_OPTICS_PREFIX


def relion_scale_reduction_mode_from_optimiser_star(path: str | Path) -> tuple[str, str]:
    """Read the command RELION recorded in an optimiser STAR header and its reduction mode."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("# --"):
                command = line[2:].strip()
                return relion_scale_reduction_mode_from_command(command), command
            if line.startswith("data_"):
                break
    raise ValueError(f"{path} does not record the RELION command line")


def update_relion_follower_scales(
    state: RelionFollowerScaleState,
    *,
    wsum_signal_product,
    wsum_reference_power,
    reduction_mode: str,
    relion_firstiter_cc_this_iter: bool = False,
    scale_relaxation_mu: float = 0.0,
) -> RelionFollowerScaleState:
    """Apply RELION's follower-local scale update after its MPI reduction.

    ``reduction_mode`` selects which physical groups RELION reduced across
    followers (see the module docstring and
    :func:`relion_scale_reduction_mode_from_command`).
    """

    if reduction_mode not in RELION_SCALE_REDUCTION_MODES:
        raise ValueError(
            f"reduction_mode must be one of {RELION_SCALE_REDUCTION_MODES}, got {reduction_mode!r}"
        )
    if relion_firstiter_cc_this_iter:
        return state
    mu = float(scale_relaxation_mu)
    if not 0.0 <= mu <= 1.0:
        raise ValueError(f"scale_relaxation_mu must be in [0, 1], got {mu}")
    expected_shape = (state.n_followers, state.n_groups)
    xa = np.asarray(wsum_signal_product, dtype=np.float64).reshape(expected_shape).copy()
    aa = np.asarray(wsum_reference_power, dtype=np.float64).reshape(expected_shape).copy()
    if np.any(~np.isfinite(xa)) or np.any(~np.isfinite(aa)) or np.any(aa < 0.0):
        raise ValueError("scale XA/AA must be finite and AA must be non-negative")

    if reduction_mode == RELION_SCALE_REDUCTION_ALL_GROUPS:
        combined_count = state.n_groups
    else:
        combined_count = min(int(state.n_optics_groups), state.n_groups)
    if combined_count:
        xa_combined = np.sum(xa[:, :combined_count], axis=0)
        aa_combined = np.sum(aa[:, :combined_count], axis=0)
        xa[:, :combined_count] = xa_combined[None, :]
        aa[:, :combined_count] = aa_combined[None, :]

    target = np.ones_like(xa)
    np.divide(xa, aa, out=target, where=aa > 0.0)
    updated = mu * np.asarray(state.scales, dtype=np.float64) + (1.0 - mu) * target
    for follower in range(state.n_followers):
        median = float(np.sort(updated[follower])[state.n_groups // 2])
        if np.isfinite(median) and median > 0.0:
            updated[follower] = np.clip(updated[follower], median / 5.0, 5.0 * median)
        count_sum = float(np.sum(state.group_counts))
        if count_sum > 0.0:
            avg_scale = float(np.sum(state.group_counts * updated[follower]) / count_sum)
            if avg_scale > 0.0 and np.isfinite(avg_scale):
                updated[follower] /= avg_scale

    return RelionFollowerScaleState(
        scales=updated,
        group_counts=np.asarray(state.group_counts, dtype=np.float64).copy(),
        n_optics_groups=int(state.n_optics_groups),
    )


def relion_rank1_serialized_scales(state: RelionFollowerScaleState) -> np.ndarray:
    """Return a copy of the scale vector written by RELION MPI follower 1."""

    return np.asarray(state.scales[0], dtype=np.float64).copy()


def validate_coupled_relion_restart_state(
    perturb_restart_state_iterations,
    follower_replay_by_iteration,
    follower_replay_source_artifacts,
) -> None:
    """Fail closed when a strict Class3D continuation restores partial state."""
    restart_numbered_iterations = {
        int(saved_state_iteration) + 1
        for saved_state_iteration in perturb_restart_state_iterations
    }
    follower_replay_numbered_iterations = {
        int(relion_iteration) for relion_iteration in follower_replay_by_iteration
    }
    missing_follower_restarts = sorted(
        restart_numbered_iterations - follower_replay_numbered_iterations
    )
    if missing_follower_restarts:
        raise ValueError(
            "RELION perturbation restart boundaries require matching follower-scale "
            "replay at numbered_pre_score; missing numbered iterations "
            f"{missing_follower_restarts}"
        )

    model_source_restart_iterations = set()
    for source_path in follower_replay_source_artifacts:
        match = re.fullmatch(r"run_it(\d+)_model\.star", os.path.basename(str(source_path)))
        if match is not None:
            model_source_restart_iterations.add(int(match.group(1)) + 1)
    unmatched_model_restarts = sorted(
        model_source_restart_iterations - restart_numbered_iterations
    )
    if unmatched_model_restarts:
        raise ValueError(
            "RELION follower-scale replay loaded from continuation model state requires "
            "a matching perturbation restart boundary; unmatched numbered iterations "
            f"{unmatched_model_restarts}"
        )
