"""The RELION state a diagnostic replay hands the refinement: STAR replays, the final-only substitution and
RELION's iteration-0 particle state (``--perturb_replay_relion_dir``, ``--final-replay-relion-dir``,
``--relion_init_dir``).

The command decides which replays a run takes and passes the resolved fields; these operations read RELION's run
files (``_build_replay_iteration_overrides``) and return the overrides. No CPU run of the command
reaches them with real RELION files: the GPU tiers' replay cases do.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from recovar import utils

from relax.relion.initial_noise import read_relion_sigma2_noise_by_group, relion_mpi_process_start_scoring_noise_pair
from relax.star import input_poses
from relax.star import metadata as relion_metadata

# The replay readers moved here from relion_replay log under its name, so their log rows keep it.
logger = logging.getLogger("relax.diagnostics.relion_replay")


def _resolve_final_replay_source_iteration(
    *, configured_max_iter, explicit_source_iteration, complete_iterations
):
    """Bind a finite RELION oracle to one exact last-numbered boundary."""
    complete = sorted({int(value) for value in complete_iterations})
    if not complete:
        raise ValueError("final replay oracle has no complete numbered RELION states")
    source_iteration = (
        max(complete)
        if explicit_source_iteration is None
        else int(explicit_source_iteration)
    )
    if source_iteration not in complete:
        raise ValueError(
            f"requested final replay source iteration {source_iteration} is not a complete oracle state; "
            f"available={complete}"
        )
    if source_iteration > int(configured_max_iter):
        raise ValueError(
            f"final replay source iteration {source_iteration} exceeds configured max_iter={configured_max_iter}"
        )
    missing_prefix = sorted(set(range(0, source_iteration + 1)) - set(complete))
    if missing_prefix:
        raise ValueError(
            f"final replay oracle is not contiguous through iteration {source_iteration}; missing={missing_prefix}"
        )
    return source_iteration


def _load_final_replay_reference_maps(relion_dir, source_iteration, volume_shape):
    """Load the exact K=1 half references consumed by RELION finalization."""
    from recovar.core import fourier_transform_utils
    from recovar.utils.helpers import load_relion_volume

    relion_dir = Path(relion_dir)
    references = []
    for half_number in (1, 2):
        map_path = relion_dir / (
            f"run_it{int(source_iteration):03d}_half{half_number}_class001.mrc"
        )
        if not map_path.is_file():
            raise ValueError(
                "diagnostic final-only reference substitution is missing "
                f"{map_path}"
            )
        reference_real = np.asarray(load_relion_volume(str(map_path)), dtype=np.float32)
        if tuple(reference_real.shape) != tuple(volume_shape):
            raise ValueError(
                f"diagnostic final-only reference {map_path} has shape "
                f"{reference_real.shape}, expected {tuple(volume_shape)}"
            )
        references.append(
            jnp.asarray(fourier_transform_utils.get_dft3(reference_real).reshape(-1))
        )
        logger.info(
            "Diagnostic final-only reference half %d <- %s",
            half_number,
            map_path,
        )
    return references


def _complete_relion_numbered_state_iterations(relion_dir):
    """Return iterations with data, sampling, and half/shared model state."""
    relion_dir = Path(relion_dir).resolve()
    complete = []
    for data_path in relion_dir.glob("run_it[0-9][0-9][0-9]_data.star"):
        match = re.fullmatch(r"run_it([0-9]{3})_data\.star", data_path.name)
        if match is None:
            continue
        iteration = int(match.group(1))
        sampling = relion_dir / f"run_it{iteration:03d}_sampling.star"
        half_models = (
            relion_dir / f"run_it{iteration:03d}_half1_model.star",
            relion_dir / f"run_it{iteration:03d}_half2_model.star",
        )
        shared_model = relion_dir / f"run_it{iteration:03d}_model.star"
        if sampling.is_file() and (all(path.is_file() for path in half_models) or shared_model.is_file()):
            complete.append(iteration)
    return sorted(complete)


def _build_replay_iteration_overrides(
    relion_dir,
    half1_idx,
    half2_idx,
    max_iter,
    ds_voxel,
    ds_grid,
    *,
    include_normcorr,
    init_relion_iteration=0,
    particle_names=None,
    include_initial_state=False,
    include_k1_mean_variance=False,
    include_k1_scoring_scale=False,
    strict=False,
    noise_dtype=np.float32,
):
    """Build per-iter replay overrides keyed on recovar iteration index.

    For each recovar iteration k >= 1 (i.e. iter 2 onwards in RELION terms),
    reads RELION's run_it{k:03d}_data.star + half1/half2 model.star
    (or the shared Class3D run_it{k:03d}_model.star) and builds an
    override dict containing:
      * image_corrections: per-image (avg_norm/normcorr) * group_scale
      * serialized_scale_corrections: per-image model-STAR group scale,
        retained as provenance rather than forced onto the live scorer
      * scoring_scale_corrections: optional exact K=1 split-half scorer scale
        for state-swap diagnostics whose half-specific model STARs are owned
        by the two scoring ranks
      * previous_best_translations / previous_best_rotation_eulers: RELION's
        previous hard assignments for local-search centering

    This matches scripts/run_multi_iter_parity.py::_load_relion_iteration_override
    (the proven replay logic). The recovar iter-k override is read from
    RELION iter-k's model+data (since recovar iter-k corresponds to RELION
    iter-(k+1), and the per-image scalings used at the start of RELION
    iter-(k+1) are the ones written by RELION iter-k's M-step).

    ``init_relion_iteration`` is normally zero. Diagnostic profile runs can
    set it to a later RELION iteration to jump directly into local search; in
    that case override slot 0 is sourced from the upstream RELION iteration
    instead of being left empty.

    When ``include_initial_state`` is true, slot 0 is loaded from RELION
    iteration 0 as well. This is required for a strict cold-start replay:
    run_it000 carries the particle pre-centering offsets, initial orientations,
    image/scale corrections, and direction prior that RELION uses in its first
    expectation step.

    ``noise_dtype`` controls only the expanded pixel representation of the
    RFLOAT model-STAR spectrum. Double-scoring replays must retain float64 so
    the reciprocal is not narrowed before construction of ``Minvsigma2``.
    """
    import re as _re
    from pathlib import Path as _Path

    import starfile as _sf

    relion_dir = _Path(relion_dir).resolve()

    def _model_has_class_direction_priors(model):
        return any(str(key).startswith("model_pdf_orient_class_") for key in model)

    def _read_model_direction_prior(model_path, model):
        if not _model_has_class_direction_priors(model):
            return None
        from relax.star.metadata import (
            read_relion_direction_prior,
            read_relion_direction_priors,
        )

        has_multiple_classes = any(
            str(key).startswith("model_pdf_orient_class_") and not str(key).endswith("_1")
            for key in model
        )
        if has_multiple_classes:
            return read_relion_direction_priors(model_path)
        return read_relion_direction_prior(model_path)

    noise_dtype = np.dtype(noise_dtype)
    if noise_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(f"noise_dtype must be float32 or float64, got {noise_dtype}")

    def _read_model_noise_variance(model, *, image_shape):
        """One half's pixel noise: flat for one optics group, ``[G, P]`` rows for G > 1.

        RELION keeps ``sigma2_noise`` per optics group (subtomograms: one group
        per tilt); ``[G, P]`` is the noise model's per-group layout (relion/optics_groups.py).
        """
        radial_by_group = read_relion_sigma2_noise_by_group(
            model,
            context="replay model",
        )
        if radial_by_group is None:
            return None
        rows = [
            np.asarray(
                utils.make_radial_image(
                    jnp.asarray(radial * float(ds_grid) ** 4), image_shape, extend_last_frequency=True,
                ),
                dtype=noise_dtype,
            ).reshape(-1)
            for radial in radial_by_group
        ]
        return rows[0] if len(rows) == 1 else np.stack(rows)

    def _read_model_class_tau2(model):
        if not isinstance(model, dict):
            return None
        class_tau2 = []
        for key, table in model.items():
            match = _re.fullmatch(r"model_class_(\d+)", str(key))
            if match is None:
                continue
            col = "rlnReferenceTau2" if "rlnReferenceTau2" in table.columns else None
            if col is None and "rlnReferenceSigma2" in table.columns:
                col = "rlnReferenceSigma2"
            if col is None:
                continue
            class_tau2.append(
                (
                    int(match.group(1)),
                    np.asarray(table[col], dtype=np.float64) * float(ds_grid) ** 4,
                )
            )
        if not class_tau2:
            return None
        class_tau2.sort(key=lambda item: item[0])
        return np.stack([tau2 for _, tau2 in class_tau2], axis=0)

    # Index i is consumed by iteration_loop for recovar iter i+1 during the
    # numbered refinement, and by the final all-data pass as len(current_sizes).
    # Allocate one extra slot so convergence on the last configured numbered
    # iteration can replay RELION run_it{max_iter:03d}_data.star.
    overrides = [None] * (max_iter + 1)
    for recovar_iter in range(0, max_iter + 1):
        # recovar iter k uses corrections computed by RELION iter k (which were
        # written into run_it{k}_data.star). Fresh non-replay runs retain the
        # historical empty slot 0; strict cold-start replay explicitly loads
        # run_it000 because it contains nonzero particle pre-centering offsets
        # and the other state consumed by RELION's first expectation step.
        relion_iter = init_relion_iteration + recovar_iter
        if relion_iter < 0 or (relion_iter == 0 and not include_initial_state):
            continue
        data_star = relion_dir / f"run_it{relion_iter:03d}_data.star"
        model_h1 = relion_dir / f"run_it{relion_iter:03d}_half1_model.star"
        model_h2 = relion_dir / f"run_it{relion_iter:03d}_half2_model.star"
        model_shared = relion_dir / f"run_it{relion_iter:03d}_model.star"
        if model_h1.exists() and model_h2.exists():
            model_paths = (model_h1, model_h2)
        elif model_shared.exists():
            model_paths = (model_shared, model_shared)
        else:
            model_paths = None
        if not data_star.exists() or model_paths is None:
            missing = []
            if not data_star.exists():
                missing.append(str(data_star))
            if model_paths is None:
                missing.append(f"{model_h1} + {model_h2} or {model_shared}")
            message = (
                f"Replay override for recovar iter {recovar_iter + 1} "
                f"(RELION iter {relion_iter:03d}) is missing {'; '.join(missing)}"
            )
            if strict:
                raise ValueError(message)
            logger.warning("%s — leaving unset", message)
            continue

        data = _sf.read(str(data_star))
        parts = data["particles"] if isinstance(data, dict) else data
        m1 = _sf.read(str(model_paths[0]))
        m2 = _sf.read(str(model_paths[1]))

        replay_identity_rows = relion_metadata.particle_identity_rows(
            parts,
            label=f"RELION replay STAR {data_star}",
        )

        nc = np.asarray(parts["rlnNormCorrection"], dtype=np.float64)

        def _scalar(table, key):
            v = table[key]
            return float(v if isinstance(v, (int, float)) else v.iloc[0] if hasattr(v, "iloc") else v[0])

        avg_norm_h1 = _scalar(m1["model_general"], "rlnNormCorrectionAverage")
        avg_norm_h2 = _scalar(m2["model_general"], "rlnNormCorrectionAverage")

        # rlnSigmaOffsetsAngst is RELION's per-iter translation sigma. RELION
        # iter (k+1) loads it from the iter-k model.star and uses it to build
        # pdf_offset (acc_ml_optimiser_impl.h::pdf_offset). recovar's iter-1
        # does not accumulate sigma2_offset moments (no per-image prior centers
        # exist yet), so without an explicit override the iter-2 E-step uses
        # the default init sigma (10 Å) instead of the data-driven RELION
        # value, which is ~6× too wide and depresses iter-2 Pmax by ~22%.
        sigma_offset_h1 = _scalar(m1["model_general"], "rlnSigmaOffsetsAngst")
        sigma_offset_h2 = _scalar(m2["model_general"], "rlnSigmaOffsetsAngst")
        sigma_offset_per_half = [float(sigma_offset_h1), float(sigma_offset_h2)]
        sigma_offset_avg = 0.5 * (sigma_offset_per_half[0] + sigma_offset_per_half[1])
        noise_h1 = _read_model_noise_variance(m1, image_shape=(int(ds_grid), int(ds_grid)))
        noise_h2 = _read_model_noise_variance(m2, image_shape=(int(ds_grid), int(ds_grid)))
        direction_prior_h1 = _read_model_direction_prior(model_paths[0], m1)
        direction_prior_h2 = _read_model_direction_prior(model_paths[1], m2)
        class_tau2 = _read_model_class_tau2(m1)

        groups_h1 = m1.get("model_groups")
        groups_h2 = m2.get("model_groups")
        scale_h1 = (
            np.asarray(groups_h1["rlnGroupScaleCorrection"], dtype=np.float64)
            if groups_h1 is not None and "rlnGroupScaleCorrection" in groups_h1.columns
            else np.array([1.0])
        )
        scale_h2 = (
            np.asarray(groups_h2["rlnGroupScaleCorrection"], dtype=np.float64)
            if groups_h2 is not None and "rlnGroupScaleCorrection" in groups_h2.columns
            else np.array([1.0])
        )
        group_no = (
            np.asarray(parts["rlnGroupNumber"], dtype=int)
            if "rlnGroupNumber" in parts.columns
            else np.ones(len(parts), dtype=int)
        )
        pp_scale_h1 = scale_h1[np.clip(group_no - 1, 0, len(scale_h1) - 1)]
        pp_scale_h2 = scale_h2[np.clip(group_no - 1, 0, len(scale_h2) - 1)]
        combined_h1 = (avg_norm_h1 / nc) * pp_scale_h1
        combined_h2 = (avg_norm_h2 / nc) * pp_scale_h2

        # Map RELION particle order to recovar's half1/half2 ordering.
        # half1_idx / half2_idx are row positions in RECOVAR's input STAR,
        # Match the complete ``(index, stack path)`` identity. Numeric stack
        # indices can repeat across multi-stack real-data STAR files.
        if particle_names is None:
            particle_identities = None
        else:
            particle_identities = [
                relion_metadata.relion_image_identity(name, label="RECOVAR input STAR")
                for name in particle_names
            ]
            if len(set(particle_identities)) != len(particle_identities):
                raise ValueError("RECOVAR input contains duplicate rlnImageName/stack identities")

        def _to_half(values, half_idx):
            rows = np.asarray(half_idx, dtype=np.int64)
            if particle_identities is None:
                return np.asarray(values, dtype=np.float32)[rows]
            identities = [particle_identities[int(row)] for row in rows]
            missing = sorted({identity for identity in identities if identity not in replay_identity_rows})
            if missing:
                preview = ", ".join(f"{index}@{stack}" for index, stack in missing[:8])
                raise ValueError(
                    f"RELION replay STAR is missing {len(missing)} RECOVAR particle identities "
                    f"(preview: {preview})"
                )
            return np.asarray(
                [values[replay_identity_rows[identity]] for identity in identities],
                dtype=np.float32,
            )

        corr_h1 = _to_half(combined_h1, half1_idx)
        corr_h2 = _to_half(combined_h2, half2_idx)
        scale_corr_h1 = _to_half(pp_scale_h1, half1_idx)
        scale_corr_h2 = _to_half(pp_scale_h2, half2_idx)

        trans_h1 = None
        trans_h2 = None
        if "rlnOriginXAngst" in parts.columns and "rlnOriginYAngst" in parts.columns:
            offsets = np.stack(
                [
                    np.asarray(parts["rlnOriginXAngst"], dtype=np.float64) / float(ds_voxel),
                    np.asarray(parts["rlnOriginYAngst"], dtype=np.float64) / float(ds_voxel),
                ],
                axis=1,
            )
            trans_h1 = _to_half(offsets, half1_idx)
            trans_h2 = _to_half(offsets, half2_idx)
        elif "rlnOriginX" in parts.columns and "rlnOriginY" in parts.columns:
            offsets = np.stack(
                [
                    np.asarray(parts["rlnOriginX"], dtype=np.float64),
                    np.asarray(parts["rlnOriginY"], dtype=np.float64),
                ],
                axis=1,
            )
            trans_h1 = _to_half(offsets, half1_idx)
            trans_h2 = _to_half(offsets, half2_idx)

        angle_cols = ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")
        rot_h1 = None
        rot_h2 = None
        euler_h1 = None
        euler_h2 = None
        if all(col in parts.columns for col in angle_cols):
            eulers = np.stack([np.asarray(parts[col], dtype=np.float64) for col in angle_cols], axis=1)
            rotations = utils.R_from_relion(eulers, degrees=True).astype(np.float32)
            rot_h1 = _to_half(rotations, half1_idx)
            rot_h2 = _to_half(rotations, half2_idx)
            euler_h1 = _to_half(eulers, half1_idx)
            euler_h2 = _to_half(eulers, half2_idx)

        override_k = {
            "translation_sigma_angstrom": sigma_offset_avg,
            "translation_sigma_angstrom_per_half": sigma_offset_per_half,
            "previous_best_translations": [trans_h1, trans_h2],
            "previous_best_rotations": [rot_h1, rot_h2],
            "previous_best_rotation_eulers": [euler_h1, euler_h2],
        }
        if noise_h1 is not None and noise_h2 is not None:
            override_k["noise_variance"] = relion_mpi_process_start_scoring_noise_pair(
                noise_h1,
                noise_h2,
                # RELION performs this broadcast once in MPI initialise().
                # Later uninterrupted iterations update each follower's noise
                # independently, so only replay slot 0 is process-start state.
                split_random_halves=(recovar_iter == 0 and model_paths[0] != model_paths[1]),
            )
        if direction_prior_h1 is not None and direction_prior_h2 is not None:
            override_k["direction_prior"] = [direction_prior_h1, direction_prior_h2]
        if class_tau2 is not None and class_tau2.shape[0] > 1:
            override_k["class_tau2"] = class_tau2
        elif class_tau2 is not None and include_k1_mean_variance:
            override_k["mean_variance"] = np.asarray(
                utils.make_radial_image(
                    class_tau2[0],
                    (int(ds_grid), int(ds_grid), int(ds_grid)),
                    extend_last_frequency=True,
                ),
                dtype=np.float64,
            ).reshape(-1)
        if include_normcorr:
            override_k["image_corrections"] = [corr_h1, corr_h2]
            override_k["serialized_scale_corrections"] = [scale_corr_h1, scale_corr_h2]
            if include_k1_scoring_scale:
                if model_paths[0] == model_paths[1]:
                    raise ValueError(
                        "exact K=1 scoring-scale replay requires distinct half-specific "
                        f"model STARs; got shared source {model_paths[0]}"
                    )
                override_k["scoring_scale_corrections"] = [scale_corr_h1, scale_corr_h2]
        overrides[recovar_iter] = override_k
        if include_normcorr:
            logger.info(
                "Replay override recovar iter %d: image_corr means=(%s, %s), serialized_scale_corr means=(%s, %s), "
                "sigma_offset=(half1 %.4f Å, half2 %.4f Å, mean %.4f Å)",
                recovar_iter + 1,
                _format_replay_mean_for_log(corr_h1),
                _format_replay_mean_for_log(corr_h2),
                _format_replay_mean_for_log(scale_corr_h1),
                _format_replay_mean_for_log(scale_corr_h2),
                sigma_offset_per_half[0],
                sigma_offset_per_half[1],
                sigma_offset_avg,
            )
        else:
            logger.info(
                "Replay override recovar iter %d: sigma_offset=(half1 %.4f Å, half2 %.4f Å, mean %.4f Å) "
                "(normcorr replay disabled)",
                recovar_iter + 1,
                sigma_offset_per_half[0],
                sigma_offset_per_half[1],
                sigma_offset_avg,
            )

    return overrides


def _format_replay_mean_for_log(values) -> str:
    arr = np.asarray(values)
    if arr.size == 0:
        return "empty"
    return f"{float(arr.mean()):.4f}"


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
) -> list:
    """RELION's per-iteration replay overrides from its run files: slot k is the state numbered iteration k+1
    consumes."""
    return _build_replay_iteration_overrides(
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
    complete_iterations = _complete_relion_numbered_state_iterations(final_replay_dir)
    source_iteration = _resolve_final_replay_source_iteration(
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
    from relax.star.metadata import read_relion_optimiser_metadata

    final_optimiser_metadata = read_relion_optimiser_metadata(final_optimiser_path)
    if not bool(final_optimiser_metadata.get("has_converged", False)):
        raise ValueError(
            f"diagnostic final-only oracle does not report convergence: {final_optimiser_path}"
        )
    final_overrides = _build_replay_iteration_overrides(
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
        final_replay_reference_maps = _load_final_replay_reference_maps(
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
    initial_overrides = _build_replay_iteration_overrides(
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
    initial_overrides = _build_replay_iteration_overrides(
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
    translations = input_poses.kclass_firstiter_translation_seed(
        initial_overrides[0],
        n_classes=n_classes,
        init_relion_iteration=init_relion_iteration,
    )
    return translations, Path(init_dir).expanduser().resolve() / "run_it000_data.star"
