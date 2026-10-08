"""The RELION state a diagnostic replay hands the refinement: STAR replays, the final-only substitution and
RELION's iteration-0 particle state (``--perturb_replay_relion_dir``, ``--final-replay-relion-dir``,
``--relion_init_dir``).

The command decides which replays a run takes and passes the resolved fields; these operations read RELION's run
files (``relion_replay._build_replay_iteration_overrides``) and return the overrides. No CPU run of the command
reaches them with real RELION files: the GPU tiers' replay cases do.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from relax.diagnostics import relion_replay
from relax.relion import input_poses


@dataclass(frozen=True)
class ReplayTarget:
    """The live run a RELION replay is mapped onto: each half's input rows, the particles' image names (RELION's
    data STAR may order rows differently), the image geometry and the noise precision."""

    half1_rows: np.ndarray
    half2_rows: np.ndarray
    particle_names: np.ndarray
    voxel_size: float
    box_size: int
    volume_shape: tuple
    noise_dtype: type


@dataclass(frozen=True)
class FinalReplay:
    """A diagnostic final-only substitution: the override of the selected field groups, RELION's references when
    requested, the source iteration, and RELION's directory when its sampling is requested."""

    override: dict | None = None
    reference_maps: object | None = None
    source_iteration: int | None = None
    sampling_dir: str | None = None


_FINAL_REPLAY_GROUP_KEYS = {
    "poses": {
        "previous_best_translations",
        "previous_best_rotations",
        "previous_best_rotation_eulers",
    },
    "sampling": {
        "translation_sigma_angstrom",
        "translation_sigma_angstrom_per_half",
    },
    "corrections": {
        "noise_variance",
        "direction_prior",
        "image_corrections",
        "serialized_scale_corrections",
    },
    "noise": {"noise_variance"},
    "direction_prior": {"direction_prior"},
    # The serialized image correction is norm-factor * source-scale.  Pair it
    # with that source scale so the replay layer preserves only the RELION
    # norm factor on the resident scoring scale.
    "norm_factor": {"image_corrections", "serialized_scale_corrections"},
    # An explicit live scoring-scale oracle.  This is intentionally excluded
    # from ``all`` because a generic leader model STAR need not represent the
    # scale resident on every follower rank.
    "scoring_scale": {"scoring_scale_corrections"},
    "references": set(),
}
_FINAL_REPLAY_ALL_GROUPS = {"poses", "sampling", "corrections", "references"}


def select_final_replay_fields(source_override, requested_fields):
    """Select diagnostic final-boundary groups without touching numbered state."""
    requested_groups = {
        token.strip().lower()
        for token in str(requested_fields).split(",")
        if token.strip()
    }
    valid_groups = set(_FINAL_REPLAY_GROUP_KEYS) | {"all"}
    unknown_groups = sorted(requested_groups - valid_groups)
    if not requested_groups or unknown_groups:
        raise ValueError(
            "--final-replay-fields requires one or more of "
            "poses,sampling,corrections,noise,direction_prior,norm_factor,"
            "scoring_scale,references,all; "
            f"unknown={unknown_groups}"
        )
    if "all" in requested_groups:
        requested_groups = (requested_groups - {"all"}) | set(_FINAL_REPLAY_ALL_GROUPS)
    selected_keys = set().union(*(_FINAL_REPLAY_GROUP_KEYS[group] for group in requested_groups))
    selected_override = {
        key: value for key, value in source_override.items() if key in selected_keys
    }
    return requested_groups, selected_override


def numbered_star_replay(
    replay_dir,
    target: ReplayTarget,
    *,
    max_iter: int,
    init_relion_iteration: int,
    include_normcorr: bool,
    include_k1_state_swap: bool,
    process_start_noise_broadcast,
) -> list:
    """RELION's per-iteration replay overrides from its run files: slot k is the state numbered iteration k+1
    consumes."""
    return relion_replay._build_replay_iteration_overrides(
        replay_dir,
        target.half1_rows,
        target.half2_rows,
        # Numbered expectation k consumes the state written before it, so
        # iterations 1..N use run_it000..run_it{N-1}.  After convergence,
        # RELION's unnumbered all-data expectation consumes the state just
        # written by iteration N and therefore needs run_it{N} as the
        # extra final-only override.
        int(max_iter),
        ds_voxel=target.voxel_size,
        ds_grid=target.box_size,
        include_normcorr=include_normcorr,
        init_relion_iteration=init_relion_iteration,
        particle_names=target.particle_names,
        include_k1_mean_variance=include_k1_state_swap,
        include_k1_scoring_scale=include_k1_state_swap,
        strict=True,
        process_start_noise_broadcast=process_start_noise_broadcast,
        noise_dtype=target.noise_dtype,
    )


def final_only_replay(
    final_replay_dir,
    target: ReplayTarget,
    *,
    max_iter: int,
    explicit_source_iteration,
    fields,
    init_relion_iteration: int,
    n_classes: int,
    log: logging.Logger,
) -> FinalReplay:
    """The final-only substitution from a converged RELION run's last numbered state."""
    final_replay_dir = Path(final_replay_dir).resolve()
    complete_iterations = relion_replay._complete_relion_numbered_state_iterations(final_replay_dir)
    source_iteration = relion_replay._resolve_final_replay_source_iteration(
        configured_max_iter=max_iter,
        explicit_source_iteration=explicit_source_iteration,
        complete_iterations=complete_iterations,
    )
    final_optimiser_path = final_replay_dir / "run_optimiser.star"
    final_sampling_path = final_replay_dir / "run_sampling.star"
    if not final_optimiser_path.is_file() or not final_sampling_path.is_file():
        raise ValueError(
            "diagnostic final-only substitution requires unnumbered run_optimiser.star "
            f"and run_sampling.star in {final_replay_dir}"
        )
    from relax.relion.relion_metadata import read_relion_optimiser_metadata

    final_optimiser_metadata = read_relion_optimiser_metadata(final_optimiser_path)
    if not bool(final_optimiser_metadata.get("has_converged", False)):
        raise ValueError(
            f"diagnostic final-only oracle does not report convergence: {final_optimiser_path}"
        )
    final_overrides = relion_replay._build_replay_iteration_overrides(
        final_replay_dir,
        target.half1_rows,
        target.half2_rows,
        source_iteration,
        ds_voxel=target.voxel_size,
        ds_grid=target.box_size,
        include_normcorr=True,
        init_relion_iteration=init_relion_iteration,
        particle_names=target.particle_names,
        strict=True,
        noise_dtype=target.noise_dtype,
    )
    source_override = final_overrides[-1]
    if source_override is None:
        raise ValueError("diagnostic final-only substitution did not load a last-numbered override")
    # Expose the model-STAR scale as a scorer oracle only at this explicit
    # final diagnostic boundary.  Numbered replay continues to treat it as
    # serialization provenance because general MPI leader/follower layouts
    # do not guarantee that it is every scorer's resident scale.
    source_override = dict(source_override)
    serialized_scale = source_override.get("serialized_scale_corrections")
    if serialized_scale is not None:
        source_override["scoring_scale_corrections"] = serialized_scale
    requested_groups, final_replay_override = select_final_replay_fields(source_override, fields)
    final_replay_reference_maps = None
    if "references" in requested_groups:
        if n_classes != 1:
            raise ValueError(
                "diagnostic final-only reference substitution currently requires --n-classes=1"
            )
        final_replay_reference_maps = relion_replay._load_final_replay_reference_maps(
            final_replay_dir,
            source_iteration,
            target.volume_shape,
        )
    log.info(
        "Diagnostic final-only substitution: source_iteration=%d groups=%s fields=%s source=%s",
        source_iteration,
        sorted(requested_groups),
        sorted(final_replay_override),
        final_replay_dir,
    )
    return FinalReplay(
        override=final_replay_override,
        reference_maps=final_replay_reference_maps,
        source_iteration=source_iteration,
        sampling_dir=str(final_replay_dir) if "sampling" in requested_groups else None,
    )


def k1_initial_state(init_dir, target: ReplayTarget, *, explicit_noise, live_noise, log: logging.Logger):
    """RELION's complete run_it000 cold-start state for the first K=1 expectation, or None when the directory
    lacks it. Explicit noise (``--init_noise_from_npz``) or the live binary64 start-up noise replaces the model
    STAR's rounded noise."""
    initial_overrides = relion_replay._build_replay_iteration_overrides(
        init_dir,
        target.half1_rows,
        target.half2_rows,
        0,
        ds_voxel=target.voxel_size,
        ds_grid=target.box_size,
        include_normcorr=True,
        init_relion_iteration=0,
        particle_names=target.particle_names,
        include_initial_state=True,
        strict=True,
        noise_dtype=target.noise_dtype,
    )
    first = initial_overrides[0]
    if first is None:
        return None
    if explicit_noise is not None:
        first = dict(first)
        noise = list(explicit_noise) if isinstance(explicit_noise, (list, tuple)) else [explicit_noise, explicit_noise]
        first["noise_variance"] = [np.asarray(value, dtype=np.float64).copy() for value in noise]
        log.info(
            "STRICT-PARITY: first expectation preserves explicit "
            "--init-noise-from-npz instead of rounded model-STAR noise",
        )
    elif live_noise is not None:
        first = dict(first)
        first["noise_variance"] = [
            np.asarray(live_noise, dtype=np.float64).copy(),
            np.asarray(live_noise, dtype=np.float64).copy(),
        ]
        log.info(
            "STRICT-PARITY: first expectation consumes computed live "
            "binary64 K=1 startup noise instead of rounded model-STAR noise",
        )
    return first


def class3d_initial_translations(init_dir, target: ReplayTarget, *, n_classes: int, init_relion_iteration: int):
    """The run_it000 input origins a fresh Class3D start keeps for its FFT pre-shifts, and their data STAR."""
    initial_overrides = relion_replay._build_replay_iteration_overrides(
        init_dir,
        target.half1_rows,
        target.half2_rows,
        0,
        ds_voxel=target.voxel_size,
        ds_grid=target.box_size,
        include_normcorr=False,
        init_relion_iteration=0,
        particle_names=target.particle_names,
        include_initial_state=True,
        strict=True,
    )
    translations = input_poses._kclass_firstiter_translation_seed(
        initial_overrides[0],
        n_classes=n_classes,
        init_relion_iteration=init_relion_iteration,
    )
    return translations, Path(init_dir).expanduser().resolve() / "run_it000_data.star"
