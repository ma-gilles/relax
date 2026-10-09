"""RELION InitialModel optics, particle-state conversion and STAR serialization."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from recovar.data_io.starfile import star_column, write_star
from recovar.utils.helpers import R_from_relion, R_to_relion

from relax import sampling
from relax.vdam.state import InitialModelState, NativeOpticsState, NativeParticleState


def _optics_group_indices(main_star) -> np.ndarray:
    if "_rlnOpticsGroup" not in main_star.columns:
        return np.zeros(len(main_star), dtype=np.int64)
    raw = main_star["_rlnOpticsGroup"].to_numpy()
    try:
        labels = np.asarray(raw, dtype=np.int64)
    except (TypeError, ValueError):
        labels = np.asarray(raw, dtype=str)
    _, indices = np.unique(labels, return_inverse=True)
    return indices.astype(np.int64, copy=False)


def _particle_optics(main_star, optics_star, ds) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Each particle's voltage, Cs and amplitude contrast (its optics group's), and the model pixel size.

    Optics groups may differ in their CTF constants; groups on other pixel sizes or boxes need
    a dataset with one class per image shape (``MultiShapeDataset``, which has ``datasets``).
    """

    pixel_size = float(ds.voxel_size)
    names = ("_rlnVoltage", "_rlnSphericalAberration", "_rlnAmplitudeContrast")
    if optics_star is None:
        missing = [name for name in names if name not in main_star.columns]
        if missing:
            raise ValueError(
                "native InitialModel needs voltage/Cs/amplitude contrast in the STAR file; "
                f"missing {', '.join(missing)}"
            )
        values = [np.asarray(main_star[name].astype(float).to_numpy(), dtype=np.float64) for name in names]
        return values[0], values[1], values[2], pixel_size

    for name in ("_rlnImagePixelSize", "_rlnImageSize"):
        several = name in optics_star.columns and np.unique(optics_star[name].astype(float).to_numpy()).size != 1
        if several and getattr(ds, "datasets", None) is None:
            raise ValueError(f"optics groups with several {name[4:]} values need one dataset per image shape")
    labels = optics_star["_rlnOpticsGroup"].to_numpy() if "_rlnOpticsGroup" in optics_star.columns else [1]
    if "_rlnOpticsGroup" in main_star.columns:
        particle_labels = main_star["_rlnOpticsGroup"].to_numpy()
    else:
        if len(optics_star) != 1:
            raise ValueError("particles without _rlnOpticsGroup need a one-row optics table")
        particle_labels = np.full(len(main_star), labels[0])
    row_of = {str(label): row for row, label in enumerate(labels)}
    missing = sorted({str(label) for label in particle_labels} - set(row_of))
    if missing:
        raise ValueError(f"particles name optics groups missing from the optics table: {', '.join(missing)}")
    rows = np.asarray([row_of[str(label)] for label in particle_labels], dtype=np.int64)
    values = [np.asarray(optics_star[name].astype(float).to_numpy(), dtype=np.float64)[rows] for name in names]
    return values[0], values[1], values[2], pixel_size


def _phase_shift(main_star) -> np.ndarray:
    if "_rlnPhaseShift" not in main_star.columns:
        return np.zeros(len(main_star), dtype=np.float64)
    return np.asarray(main_star["_rlnPhaseShift"].astype(float).to_numpy(), dtype=np.float64)


def _native_optics_state(main_star, optics_star, dataset) -> NativeOpticsState:
    voltage, Cs, Q0, pixel_size = _particle_optics(main_star, optics_star, dataset)
    required = ("_rlnDefocusU", "_rlnDefocusV", "_rlnDefocusAngle")
    missing = [name for name in required if name not in main_star.columns]
    if missing:
        raise ValueError(f"native InitialModel needs per-particle CTF columns: {', '.join(missing)}")
    grids = {}
    if getattr(dataset, "datasets", None) is not None:
        # Each particle's own grid, for the expected accuracy of groups on other grids.
        grids = dict(image_pixel_size=np.empty(len(main_star)), image_box=np.empty(len(main_star), dtype=np.int64))
        for class_dataset, rows in zip(dataset.datasets, dataset.rows):
            grids["image_pixel_size"][rows] = float(class_dataset.voxel_size)
            grids["image_box"][rows] = int(class_dataset.image_shape[0])
    return NativeOpticsState(
        voltage=voltage,
        Cs=Cs,
        Q0=Q0,
        pixel_size=float(pixel_size),
        **grids,
        defU=np.asarray(main_star["_rlnDefocusU"].astype(float).to_numpy(), dtype=np.float64),
        defV=np.asarray(main_star["_rlnDefocusV"].astype(float).to_numpy(), dtype=np.float64),
        defAngle=np.asarray(main_star["_rlnDefocusAngle"].astype(float).to_numpy(), dtype=np.float64),
        phase_shift=_phase_shift(main_star),
    )


def _experiment_read_order(main_star) -> np.ndarray:
    """RELION Experiment::read order for bootstrap, noise and subset scheduling."""

    mic_col = star_column(main_star, "_rlnMicrographName")
    if mic_col is None:
        return np.arange(len(main_star), dtype=np.int64)
    mic_names = mic_col.astype(str).to_numpy()
    return np.asarray(sorted(range(len(mic_names)), key=lambda i: mic_names[i]), dtype=np.int64)


def _stack_star_pair(main_star, x_name: str, y_name: str) -> np.ndarray | None:
    x = star_column(main_star, x_name)
    y = star_column(main_star, y_name)
    if (x is None) != (y is None):
        raise ValueError(f"STAR file must provide both {x_name} and {y_name}")
    if x is None:
        return None
    return np.stack(
        [
            np.asarray(x.astype(float).to_numpy(), dtype=np.float64),
            np.asarray(y.astype(float).to_numpy(), dtype=np.float64),
        ],
        axis=1,
    )


def _image_origin_offsets_pixels_from_star(main_star, dataset) -> np.ndarray:
    n_images = int(len(main_star))
    angst = _stack_star_pair(main_star, "_rlnOriginXAngst", "_rlnOriginYAngst")
    if angst is not None:
        pixel_size = float(dataset.voxel_size)
        if pixel_size <= 0.0:
            raise ValueError("dataset voxel_size must be positive to convert STAR origins from Angstroms")
        shifts = angst / pixel_size
    else:
        pixels = _stack_star_pair(main_star, "_rlnOriginX", "_rlnOriginY")
        if pixels is None:
            return np.zeros((n_images, 2), dtype=np.float32)
        shifts = pixels
    if not np.all(np.isfinite(shifts)):
        raise ValueError("STAR origin shifts must be finite")
    return shifts.astype(np.float64, copy=False)


def _star_source_poses(main_star, n: int) -> dict:
    """The STAR's input orientations as particle-state fields: RELION's expected accuracy and change monitor use
    them until a particle's first visit (ml_optimiser.cpp:9505-9507). Each missing ``rlnAngleRot/Tilt/Psi`` column
    gives every particle the angle 0 for that label, as RELION's ``Experiment::read`` sets it
    (exp_model.cpp:1104-1106, 1131-1136)."""
    columns = [star_column(main_star, f"_rlnAngle{name}") for name in ("Rot", "Tilt", "Psi")]
    eulers = np.stack(
        [
            np.zeros(n, dtype=np.float64)
            if column is None
            else np.asarray(column.astype(float).to_numpy(), dtype=np.float64)
            for column in columns
        ],
        axis=1,
    )
    if not np.all(np.isfinite(eulers)):
        raise ValueError("STAR Euler angles must be finite")
    return dict(
        best_pose_rotations=np.asarray(R_from_relion(eulers, degrees=True), dtype=np.float32),
        best_pose_eulers_deg=eulers,
        best_pose_eulers_valid=np.ones(n, dtype=bool),
    )


def _tomo_particle_state_from_star(main_star, *, pixel_size: float) -> NativeParticleState:
    """Fresh state of subtomogram particles: their 3D offsets (pixels) from ``rlnOrigin{X,Y,Z}Angst``, class 0,
    and the input orientations (:func:`_star_source_poses`)."""

    n = int(len(main_star))
    offsets = np.zeros((n, 3), dtype=np.float64)
    for axis, label in enumerate(("_rlnOriginXAngst", "_rlnOriginYAngst", "_rlnOriginZAngst")):
        column = star_column(main_star, label)
        if column is not None:
            offsets[:, axis] = np.asarray(column.astype(float).to_numpy(), dtype=np.float64) / float(pixel_size)
    return NativeParticleState(
        translation_offsets=offsets,
        class_assignments=np.zeros(n, dtype=np.int32),
        max_posterior=np.zeros(n, dtype=np.float32),
        **_star_source_poses(main_star, n),
    )


def tomo_checkpoint_particle_state(main_star, *, pixel_size: float) -> NativeParticleState:
    """Subtomogram particles of a RELION VDAM checkpoint's data STAR (``--diagnostic-continue-optimiser``).

    As :func:`_tomo_particle_state_from_star`, plus each particle's class (0: not yet visited) and Pmax, which the
    change monitor and the subset bookkeeping read; a fresh run starts every particle in class 0 with Pmax 0.
    """
    state = _tomo_particle_state_from_star(main_star, pixel_size=pixel_size)
    numbers = np.asarray(star_column(main_star, "_rlnClassNumber", required=True).astype(int), dtype=np.int32)
    pmax = star_column(main_star, "_rlnMaxValueProbDistribution", required=True).astype(float)
    return replace(
        state,
        class_assignments=np.maximum(numbers - 1, 0).astype(np.int32),
        max_posterior=np.asarray(pmax, dtype=np.float32),
        visited=numbers > 0,
    )


def _particle_state_from_star(
    main_star,
    dataset,
    *,
    allow_unvisited_class_zero: bool = False,
    nr_classes: int | None = None,
) -> NativeParticleState:
    """Load particle state, optionally accepting RELION's unvisited-class sentinel.

    Fresh production inputs retain the ordinary one-indexed positive-class
    contract.  Native InitialModel continuation STARs use class zero only for
    particles that the stochastic gradient schedule has not visited yet; those
    rows start in class index 0, as in a fresh run.
    """

    if allow_unvisited_class_zero and (nr_classes is None or int(nr_classes) < 1):
        raise ValueError("diagnostic continuation requires the checkpoint class count")
    n_images = int(getattr(dataset, "n_images", len(main_star)))
    if len(main_star) != n_images:
        raise ValueError(f"STAR table has {len(main_star)} particles but dataset has {n_images} images")
    class_col = star_column(main_star, "_rlnClassNumber")
    if class_col is None:
        if allow_unvisited_class_zero:
            raise ValueError(
                "diagnostic continuation requires _rlnClassNumber",
            )
        class_numbers = None
        class_assignments = np.zeros(n_images, dtype=np.int32)
    else:
        class_numbers = np.asarray(class_col.astype(int).to_numpy(), dtype=np.int32)
        if allow_unvisited_class_zero:
            if np.any((class_numbers < 0) | (class_numbers > int(nr_classes))):
                raise ValueError(
                    f"diagnostic continuation _rlnClassNumber values must be in 0..{int(nr_classes)}",
                )
            class_assignments = np.maximum(class_numbers - 1, 0).astype(np.int32)
        else:
            class_assignments = class_numbers - 1
        if not allow_unvisited_class_zero and np.any(class_assignments < 0):
            raise ValueError("_rlnClassNumber values must be one-indexed positive class ids")

    pmax_col = star_column(main_star, "_rlnMaxValueProbDistribution")
    if pmax_col is None:
        max_posterior = np.zeros(n_images, dtype=np.float32)
        max_posterior_values = None
    else:
        max_posterior_values = np.asarray(
            pmax_col.astype(float).to_numpy(),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(max_posterior_values)):
            raise ValueError("_rlnMaxValueProbDistribution values must be finite")
        max_posterior = max_posterior_values.astype(np.float32)

    if allow_unvisited_class_zero:
        assert class_numbers is not None
        zero_state_evidence: list[np.ndarray] = []
        if max_posterior_values is not None:
            if np.any(max_posterior_values < 0.0):
                raise ValueError(
                    "diagnostic continuation probability values must be non-negative",
                )
            zero_state_evidence.append(max_posterior_values == 0.0)
        significant_col = star_column(main_star, "_rlnNrOfSignificantSamples")
        if significant_col is not None:
            significant_samples = np.asarray(
                significant_col.astype(float).to_numpy(),
                dtype=np.float64,
            )
            if (
                not np.all(np.isfinite(significant_samples))
                or np.any(significant_samples < 0.0)
                or np.any(significant_samples != np.floor(significant_samples))
            ):
                raise ValueError(
                    "diagnostic continuation significant-sample counts must be "
                    "finite non-negative integers",
                )
            zero_state_evidence.append(significant_samples == 0.0)
        if not zero_state_evidence:
            raise ValueError(
                "diagnostic continuation cannot validate unvisited class-zero rows "
                "without posterior or significant-sample state",
            )
        state_is_unvisited = np.logical_and.reduce(zero_state_evidence)
        class_is_unvisited = class_numbers == 0
        if not np.array_equal(class_is_unvisited, state_is_unvisited):
            mismatched_rows = np.flatnonzero(class_is_unvisited != state_is_unvisited)
            raise ValueError(
                "diagnostic continuation class-zero sentinels disagree with "
                f"unvisited particle state at rows {mismatched_rows[:8].tolist()}",
            )
        visited = ~class_is_unvisited
    else:
        visited = max_posterior > 0.0

    return NativeParticleState(
        translation_offsets=_image_origin_offsets_pixels_from_star(main_star, dataset),
        class_assignments=class_assignments,
        max_posterior=max_posterior,
        pose_assignments=np.full(n_images, -1, dtype=np.int32),
        visited=visited,
        **_star_source_poses(main_star, n_images),
    )


def _write_model_star(path: str, state: InitialModelState, class_mrcs: tuple[str, ...]) -> None:
    current_resolution_angstrom = 1.0 / float(state.current_resolution) if state.current_resolution > 0 else np.inf
    pixel_x_ori = float(state.pixel_size) * float(state.box_size)
    n_shells = int(state.box_size) // 2 + 1
    pdf_direction = np.asarray(state.pdf_direction, dtype=np.float64)

    lines: list[str] = [
        "# Created by recovar native InitialModel\n",
        "\ndata_model_general\n\n",
        f"_rlnCurrentResolution {current_resolution_angstrom:.12g}\n",
        f"_rlnCurrentImageSize {int(state.current_size)}\n",
        f"_rlnCurrentIteration {int(state.iter)}\n",
        f"_rlnNrClasses {int(state.K)}\n",
        f"_rlnTau2FudgeFactor {float(state.tau2_fudge_factor):.12g}\n",
        f"_rlnAveragePmax {float(state.ave_Pmax):.12g}\n",
        f"_rlnSigmaOffsetsAngst {float(np.sqrt(max(float(state.sigma2_offset), 0.0))):.12g}\n\n",
        "data_model_classes\n\nloop_\n_rlnReferenceImage #1\n_rlnClassDistribution #2\n_rlnEstimatedResolution #3\n",
    ]
    for k, (class_mrc, probability) in enumerate(zip(class_mrcs, np.asarray(state.pdf_class))):
        # RELION's per-class resolution: last shell before the SSNR first drops below one (ml_model.cpp:1644-1657).
        maxres = int(np.argmin(np.append(state.data_vs_prior_class[k], 0.0) >= 1.0)) - 1
        lines.append(f"{class_mrc} {float(probability):.12g} {pixel_x_ori / maxres if maxres > 0 else np.inf:.12g}\n")

    for k in range(int(state.K)):
        lines.append(
            f"\n\ndata_model_class_{k + 1}\n\nloop_\n"
            "_rlnSpectralIndex #1\n_rlnResolution #2\n_rlnAngstromResolution #3\n"
            "_rlnSsnrMap #4\n_rlnGoldStandardFsc #5\n_rlnFourierCompleteness #6\n"
            "_rlnReferenceSigma2 #7\n_rlnReferenceTau2 #8\n"
        )
        tau2 = np.asarray(state.tau2_class[k], dtype=np.float64)
        dvp = np.asarray(state.data_vs_prior_class[k], dtype=np.float64)
        fsc = np.asarray(state.fsc_halves_class[k], dtype=np.float64)
        sigma2_class = np.asarray(state.sigma2_class[k], dtype=np.float64)
        fourier_coverage = np.asarray(state.fourier_coverage_class[k], dtype=np.float64)
        for shell in range(n_shells):
            resolution = float(shell) / pixel_x_ori
            resolution_angstrom = pixel_x_ori / float(shell) if shell > 0 else 999.0
            lines.append(
                f"{int(shell)} {resolution:.12g} {resolution_angstrom:.12g} "
                f"{float(dvp[shell]):.12g} {float(fsc[shell]):.12g} "
                f"{float(fourier_coverage[shell]):.12g} "
                f"{float(sigma2_class[shell]):.12g} {float(tau2[shell]):.12g}\n"
            )
        if pdf_direction.ndim == 2 and k < pdf_direction.shape[0]:
            lines.append(f"\n\ndata_model_pdf_orient_class_{k + 1}\n\nloop_\n_rlnOrientationDistribution #1\n")
            lines.extend(f"{float(p):.12g}\n" for p in pdf_direction[k])

    for g, spectrum in enumerate(np.asarray(state.sigma2_noise)):
        lines.append(
            f"\n\ndata_model_optics_group_{g + 1}\n\nloop_\n_rlnSpectralIndex #1\n_rlnResolution #2\n_rlnSigma2Noise #3\n"
        )
        lines.extend(f"{int(shell)} 0 {float(sigma2):.12g}\n" for shell, sigma2 in enumerate(spectrum))

    with open(path, "w") as f:
        f.writelines(lines)


def _set_star_column(table, column: str, values) -> None:
    target = column
    no_prefix = column[1:] if column.startswith("_") else column
    if target not in table.columns and no_prefix in table.columns:
        target = no_prefix
    table[target] = values


def _format_float_column(values: np.ndarray, precision: int = 6) -> list[str]:
    return [f"{float(value):.{precision}f}" for value in np.asarray(values).reshape(-1)]


def _initial_model_random_subsets(main_star) -> np.ndarray:
    """Return RELION's one-based pseudo-halfset for every input-table row.

    InitialModel routes ``Experiment`` part ids by ``part_id % 2`` even when
    ordinary split-half refinement is disabled.  ``_experiment_read_order``
    maps those internal part ids to RECOVAR's input-table rows; invert that
    map here so the written data STAR records the same persistent identity.
    """

    order = np.asarray(_experiment_read_order(main_star), dtype=np.int64)
    n_images = len(main_star)
    if (
        order.shape != (n_images,)
        or np.unique(order).size != n_images
        or np.any(order < 0)
        or np.any(order >= n_images)
    ):
        raise ValueError("RELION experiment read order must be a particle-row permutation")
    part_ids = np.empty(n_images, dtype=np.int64)
    part_ids[order] = np.arange(n_images, dtype=np.int64)
    return (part_ids % 2 + 1).astype(np.int32, copy=False)


def _write_data_star(path: str, main_star, optics_star, dataset, particle_state: NativeParticleState) -> None:
    n_images = int(len(particle_state.translation_offsets))
    if len(main_star) != n_images:
        raise ValueError(f"STAR table has {len(main_star)} particles but the particle state has {n_images}")

    output_order = _experiment_read_order(main_star)
    table = main_star.copy()
    visited = particle_state.visited
    if visited is None:
        visited = (
            np.asarray(particle_state.pose_assignments, dtype=np.int32) >= 0
            if particle_state.pose_assignments is not None
            else np.ones(n_images, dtype=bool)
        )
    visited = np.asarray(visited, dtype=bool).reshape(-1)

    offsets_angstrom = np.asarray(particle_state.translation_offsets, dtype=np.float64) * float(dataset.voxel_size)
    _set_star_column(table, "_rlnOriginXAngst", _format_float_column(offsets_angstrom[:, 0]))
    _set_star_column(table, "_rlnOriginYAngst", _format_float_column(offsets_angstrom[:, 1]))
    if offsets_angstrom.shape[1] == 3:
        # Subtomogram particles carry 3D offsets.
        _set_star_column(table, "_rlnOriginZAngst", _format_float_column(offsets_angstrom[:, 2]))
    if star_column(table, "_rlnOriginX") is not None or star_column(table, "_rlnOriginY") is not None:
        offsets_pixels = np.asarray(particle_state.translation_offsets, dtype=np.float64)
        _set_star_column(table, "_rlnOriginX", _format_float_column(offsets_pixels[:, 0]))
        _set_star_column(table, "_rlnOriginY", _format_float_column(offsets_pixels[:, 1]))
    class_numbers = np.zeros(n_images, dtype=np.int32)
    class_numbers[visited] = np.asarray(particle_state.class_assignments, dtype=np.int32)[visited] + 1
    _set_star_column(table, "_rlnClassNumber", class_numbers)
    _set_star_column(table, "_rlnRandomSubset", _initial_model_random_subsets(main_star))
    _set_star_column(table, "_rlnMaxValueProbDistribution", _format_float_column(particle_state.max_posterior))
    if particle_state.significant_counts is not None:
        _set_star_column(table, "_rlnNrOfSignificantSamples", np.asarray(particle_state.significant_counts, np.int64))
    if particle_state.log_likelihood_contribution is not None:
        _set_star_column(table, "_rlnLogLikeliContribution", _format_float_column(particle_state.log_likelihood_contribution))

    has_rotations = (
        particle_state.best_pose_rotation_ids is not None
        or particle_state.best_pose_rotations is not None
        or particle_state.best_pose_eulers_deg is not None
    )
    if has_rotations:

        def _angle(col):
            return table[col].astype(float).to_numpy(copy=True) if col in table else np.zeros(n_images)

        angle_rot, angle_tilt, angle_psi = (_angle(c) for c in ("_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"))
        remaining_rot = visited.copy()
        if particle_state.best_pose_eulers_deg is not None and particle_state.best_pose_eulers_valid is not None:
            source_valid = remaining_rot & np.asarray(particle_state.best_pose_eulers_valid, dtype=bool)
            eulers = np.asarray(particle_state.best_pose_eulers_deg, dtype=np.float64)[source_valid]
            angle_rot[source_valid], angle_tilt[source_valid], angle_psi[source_valid] = eulers.T
            remaining_rot[source_valid] = False
        if particle_state.best_pose_rotations is not None:
            rotations = np.asarray(particle_state.best_pose_rotations, dtype=np.float64)
            valid_matrix = remaining_rot & np.any(np.abs(rotations.reshape(n_images, -1)) > 0.0, axis=1)
            if np.any(valid_matrix):
                eulers = np.asarray(R_to_relion(rotations[valid_matrix], degrees=True), dtype=np.float64)
                angle_rot[valid_matrix] = eulers[:, 0]
                angle_tilt[valid_matrix] = eulers[:, 1]
                angle_psi[valid_matrix] = eulers[:, 2]
                remaining_rot[valid_matrix] = False

        if particle_state.best_pose_rotation_ids is not None:
            rotation_ids = np.asarray(particle_state.best_pose_rotation_ids, dtype=np.int64).reshape(-1)
            valid_rot = remaining_rot & (rotation_ids >= 0)
            if np.any(valid_rot):
                rotation_orders = (
                    np.asarray(particle_state.best_pose_rotation_orders, dtype=np.int32).reshape(-1)
                    if particle_state.best_pose_rotation_orders is not None
                    else None
                )
                if rotation_orders is None:
                    max_rotations = int(np.max(rotation_ids[valid_rot])) + 1
                    inferred_order = next(
                        (o for o in range(16) if sampling.rotation_grid_size(o) >= max_rotations), None
                    )
                    if inferred_order is None:
                        raise ValueError(
                            f"cannot infer HEALPix order for max rotation id {int(np.max(rotation_ids[valid_rot]))}"
                        )
                    rotation_orders = np.full(n_images, inferred_order, dtype=np.int32)
                for order in np.unique(rotation_orders[valid_rot]).tolist():
                    if order < 0:
                        continue
                    order_mask = valid_rot & (rotation_orders == order)
                    eulers = sampling.get_relion_rotation_grid_eulers(order, rotation_index_order="relion")
                    angle_rot[order_mask] = eulers[rotation_ids[order_mask], 0]
                    angle_tilt[order_mask] = eulers[rotation_ids[order_mask], 1]
                    angle_psi[order_mask] = eulers[rotation_ids[order_mask], 2]
        _set_star_column(table, "_rlnAngleRot", _format_float_column(angle_rot))
        _set_star_column(table, "_rlnAngleTilt", _format_float_column(angle_tilt))
        _set_star_column(table, "_rlnAnglePsi", _format_float_column(angle_psi))

    table = table.iloc[output_order].reset_index(drop=True)
    out_path = Path(path)
    if str(out_path.parent) not in ("", "."):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    # Rows written from the table's array: the same bytes as the per-row Series writer
    # (tests/unit/test_star_array_rows.py), 0.06 s instead of 0.5 s for 10k particles.
    write_star(str(out_path), table, optics_star.copy() if optics_star is not None else None, array_rows=True)
