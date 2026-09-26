"""RELION's per-iteration run files for auto-refine and Class3D, and ``--continue``.

After every numbered iteration RELION writes (``MlOptimiser::write``,
ml_optimiser.cpp:1359-1557, RELION 5.0.1 f2c1a38):

* ``run_itNNN_optimiser.star``: ``optimiser_general`` (ml_optimiser.cpp:1376-1520);
* ``run_itNNN_half{1,2}_model.star`` for split-half auto-refine, or
  ``run_itNNN_model.star`` for Class3D (``MlModel::write``, ml_model.cpp:469-823),
  with the reference maps ``..._class001.mrc`` it names;
* ``run_itNNN_data.star`` (``Experiment::write``);
* ``run_itNNN_sampling.star`` (``HealpixSampling::write``, healpix_sampling.cpp:232-289);
* for auto-refine, the unregularized half maps ``run_itNNN_half{1,2}_class001_unfil.mrc``
  (ml_optimiser_mpi.cpp:3237-3272).

:class:`RunFileWriter` writes the same files, with RELION's block and label names, from an
:class:`~relax.refinement.iteration_snapshot.IterationSnapshot`.
:func:`read_run_files` reads them back into a snapshot for ``--continue``.

Two deliberate differences from RELION's files:

* Floats are written with full precision (Python ``repr``). RELION's ``%12.6f`` keeps four
  significant digits of a 0.00x noise power (metadata_table.cpp:257-278), which a restart
  would inherit.
* ``run_itNNN_optimiser.star`` carries two extra blocks, ``data_relax_state`` (every
  ``RefinementState`` scalar, the sampling perturbation and the particle-order provenance)
  and ``data_relax_shells`` (the FSC that drives current-size growth). RELION's files do not
  hold the latched fine-enough flag or the growth FSC, which the next iteration reads.
  ``MlOptimiser::read`` parses only ``data_optimiser_general`` (ml_optimiser.cpp:1107), so
  RELION ignores the extra blocks.

``--continue`` keeps each half's own noise spectrum. RELION's MPI restart broadcasts
rank 1's ``sigma2_noise`` to every rank (ml_optimiser_mpi.cpp:750-758), so its half 2
restarts on half 1's noise (relax issue #7); an uninterrupted run keeps both.

Maps use RELION's sign and axis convention (``relax.helpers.map_io``). Relax's Fourier means
round-trip through the float32 real-space MRC that RELION's ``--continue`` also reads.
"""

from __future__ import annotations

import logging
import math
import os
import re
import shlex
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from relax.refinement.iteration_snapshot import (
    REFINEMENT_STATE_SCALAR_FIELDS,
    IterationSnapshot,
)

logger = logging.getLogger(__name__)

RUN_FILES_FORMAT = "relax.run_files.v1"


# ---------------------------------------------------------------------------
# STAR text
# ---------------------------------------------------------------------------


def _format_value(value) -> str:
    """One STAR cell. Floats keep every bit (``repr`` is the shortest round-trip form)."""

    if isinstance(value, (bool, np.bool_)):
        return "1" if value else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    text = str(value)
    if text == "" or any(ch.isspace() for ch in text):
        return '"' + text.replace('"', "'") + '"'
    return text


def _format_column(values) -> list[str]:
    if isinstance(values, list) and (not values or isinstance(values[0], str)):
        return [_format_value(v) for v in values]  # input text, kept as written
    array = np.asarray(values)
    if array.dtype.kind == "f":
        return [_format_value(v) for v in array.astype(np.float64).tolist()]
    if array.dtype.kind in "iub":
        return [_format_value(v) for v in array.tolist()]
    return [_format_value(v) for v in array.tolist()]


def _list_block(name: str, items) -> str:
    lines = [f"\n# version 50001\n\ndata_{name}\n\n"]
    width = max(len(label) for label, _ in items) + 1
    for label, value in items:
        lines.append(f"_{label:<{width}} {_format_value(value)}\n")
    return "".join(lines) + "\n"


def _loop_block(name: str, columns) -> str:
    labels = list(columns)
    cells = [_format_column(columns[label]) for label in labels]
    n_rows = len(cells[0]) if cells else 0
    if any(len(c) != n_rows for c in cells):
        raise ValueError(f"STAR loop {name}: columns have different lengths")
    header = [f"\n# version 50001\n\ndata_{name}\n\nloop_\n"]
    header.extend(f"_{label} #{i + 1}\n" for i, label in enumerate(labels))
    body = "\n".join(" ".join(row) for row in zip(*cells))
    return "".join(header) + body + ("\n" if body else "") + "\n"


def read_star_blocks(path) -> dict:
    """Parse a STAR file into ``{block: {label: str}}`` (list) or ``{block: {label: [str]}}`` (loop).

    Values stay strings so numbers convert without the last-bit error of pandas' default
    float parser; :func:`_floats` converts with NumPy's correctly rounded ``strtod``.
    """

    blocks: dict = {}
    name = None
    in_loop = False
    labels: list[str] = []
    rows: list[list[str]] = []

    def _close_loop():
        nonlocal in_loop, labels, rows
        if in_loop and name is not None:
            columns = list(zip(*rows)) if rows else [()] * len(labels)
            blocks[name] = {label: list(col) for label, col in zip(labels, columns)}
        in_loop = False
        labels, rows = [], []

    with open(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("data_"):
                _close_loop()
                name = line[5:]
                blocks[name] = {}
                continue
            if line == "loop_":
                in_loop = True
                labels, rows = [], []
                continue
            if line.startswith("_"):
                tokens = shlex.split(line, comments=False)
                label = tokens[0][1:]
                if in_loop and not rows:
                    labels.append(label)
                else:
                    _close_loop()
                    blocks[name][label] = tokens[1] if len(tokens) > 1 else ""
                continue
            if in_loop:
                tokens = shlex.split(line) if '"' in line or "'" in line else line.split()
                if len(tokens) != len(labels):
                    raise ValueError(f"{path}: block {name} row has {len(tokens)} fields, expected {len(labels)}")
                rows.append(tokens)
    _close_loop()
    return blocks


def _floats(values) -> np.ndarray:
    return np.asarray(list(values), dtype=np.float64)


def _ints(values) -> np.ndarray:
    return np.asarray([int(v) for v in values], dtype=np.int64)


def _bool(text) -> bool:
    return bool(int(float(text)))


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSettings:
    """Run-constant values RELION records in ``optimiser_general`` and the model STAR."""

    output_root: str
    random_seed: int
    nr_iter: int
    particle_diameter: float
    width_mask_edge: int = 5
    adaptive_oversampling: int = 1
    adaptive_fraction: float = 0.999
    low_resol_join_halves: float = 40.0
    auto_local_healpix_order: int = 4
    do_solvent_fsc: bool = False
    max_significants: int = -1
    symmetry: str = "C1"
    healpix_order_original: int = 2
    offset_range_original_angstrom: float = 0.0
    offset_step_original_angstrom: float = 0.0
    perturbation_factor: float = 0.5
    padding_factor: float = 2.0
    command_line: str = ""


class RunFileWriter:
    """Writes RELION's ``run_itNNN_*`` files from snapshots of a refinement.

    ``input_star`` is the run's input particle STAR; ``half_rows`` holds, per half, the input
    rows of that half's images in the loop's local order. ``data.star`` keeps the input row
    order and every input column as written in the input (text unchanged), with the refined
    columns replaced, as ``Experiment::write`` does.

    With ``background`` (the default) a call returns once the snapshot is handed over and a
    thread writes the files while the next iteration runs; at most one write is in flight,
    and :meth:`wait` joins it and re-raises its error. The snapshot is a host copy, so the
    loop may overwrite its arrays. ``run_itNNN_optimiser.star`` is written last and renamed
    into place, so an iteration can be continued only once all its files exist.
    """

    def __init__(
        self,
        output_dir,
        *,
        settings: RunSettings,
        input_star,
        half_rows,
        group_names=None,
        prefix: str = "run",
        write_every: int = 1,
        write_unfiltered_maps: bool = True,
        background: bool = True,
        keep_iterations: int = 0,
    ):
        if int(write_every) < 1:
            raise ValueError(f"write_every must be positive, got {write_every}")
        self.output_dir = Path(output_dir)
        self.settings = settings
        input_blocks = read_star_blocks(input_star)
        self.particles = input_blocks.get("particles") or input_blocks.get("")
        self.optics = input_blocks.get("optics")
        if not self.particles or "rlnImageName" not in self.particles:
            raise ValueError(f"{input_star} has no particle table with rlnImageName")
        self.half_rows = [np.asarray(rows, dtype=np.int64) for rows in half_rows]
        self.group_names = group_names
        self.prefix = str(prefix)
        self.write_every = int(write_every)
        if int(keep_iterations) < 0:
            raise ValueError(f"keep_iterations must be >= 0, got {keep_iterations}")
        self.keep_iterations = int(keep_iterations)
        self.write_unfiltered_maps = bool(write_unfiltered_maps)
        self.background = bool(background)
        self.seconds: dict[int, float] = {}
        self._thread = None
        self._error = None
        n_rows = len(self.particles["rlnImageName"])
        covered = np.concatenate(self.half_rows) if self.half_rows else np.empty(0, np.int64)
        if covered.size != n_rows or np.unique(covered).size != n_rows:
            raise ValueError("half_rows must partition the particle table rows")

    # The loop asks before it builds a snapshot or reconstructs unfiltered maps.
    def due(self, relion_iteration: int) -> bool:
        return int(relion_iteration) % self.write_every == 0

    def wants_unfiltered_maps(self, relion_iteration: int, *, n_classes: int) -> bool:
        return self.write_unfiltered_maps and int(n_classes) == 1 and self.due(relion_iteration)

    def root(self, relion_iteration: int) -> Path:
        return self.output_dir / f"{self.prefix}_it{int(relion_iteration):03d}"

    def __call__(self, snapshot: IterationSnapshot) -> Path:
        """Write ``snapshot``'s files; returns the optimiser STAR path (complete after :meth:`wait`)."""

        self.wait()
        path = Path(f"{self.root(snapshot.relion_iteration)}_optimiser.star")
        if not self.background:
            self._write(snapshot)
            return path
        self._thread = threading.Thread(
            target=self._write_in_thread,
            args=(snapshot,),
            name=f"relax-run-files-it{int(snapshot.relion_iteration):03d}",
        )
        self._thread.start()
        return path

    def wait(self) -> None:
        """Join the write in flight and re-raise its error."""

        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            error, self._error = self._error, None
            raise RuntimeError("writing the RELION run files failed") from error

    def _write_in_thread(self, snapshot):
        try:
            self._write(snapshot)
        except BaseException as exc:  # re-raised by wait() on the refinement's thread
            self._error = exc

    def _write(self, snapshot: IterationSnapshot) -> Path:
        t0 = time.time()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        root = self.root(snapshot.relion_iteration)
        _write_maps(root, snapshot)
        model_paths = _write_model_stars(root, snapshot, self.settings, self._group_rows(snapshot))
        _write_sampling_star(root, snapshot, self.settings)
        _write_data_star(root, snapshot, self.particles, self.optics, self.half_rows)
        optimiser = _write_optimiser_star(root, snapshot, self.settings, model_paths)
        self.seconds[int(snapshot.relion_iteration)] = time.time() - t0
        logger.info(
            "Wrote RELION run files for iteration %d to %s* in %.2f s",
            int(snapshot.relion_iteration),
            root,
            self.seconds[int(snapshot.relion_iteration)],
        )
        if self.keep_iterations:
            self._remove_old_iterations(int(snapshot.relion_iteration))
        return optimiser

    def _remove_old_iterations(self, written: int) -> None:
        """Keep the ``keep_iterations`` newest complete iterations up to ``written``.

        Older run files in the output directory go, including a crashed run's incomplete
        ones; later iterations (left by an earlier run this one continues) stay until they are
        overwritten. Only the file names this writer produces are removed, the optimiser STAR
        first so a partly removed iteration can no longer be continued. The run's final
        outputs are not run_itNNN files and are never touched.
        """

        pattern = re.compile(
            rf"{re.escape(self.prefix)}_it(\d{{3}})_(optimiser\.star|data\.star|sampling\.star|model\.star"
            rf"|half[12]_model\.star|class\d{{3}}\.mrc|half[12]_class\d{{3}}\.mrc|half[12]_class001_unfil\.mrc)"
        )
        by_iteration: dict[int, list[Path]] = {}
        for path in self.output_dir.iterdir():
            match = pattern.fullmatch(path.name)
            if match:
                by_iteration.setdefault(int(match.group(1)), []).append(path)
        complete = sorted(
            it
            for it, paths in by_iteration.items()
            if it <= written and any(p.name.endswith("_optimiser.star") for p in paths)
        )
        keep = set(complete[-self.keep_iterations :])
        for it in sorted(by_iteration):
            if it in keep or it >= written:
                continue
            paths = sorted(by_iteration[it], key=lambda p: not p.name.endswith("_optimiser.star"))
            for path in paths:
                path.unlink(missing_ok=True)
            logger.info("Removed the run files of iteration %d (--keep-iterations %d)", it, self.keep_iterations)

    def _group_rows(self, snapshot):
        """``(group number, name, particle count)`` per scale group, 1-based as RELION."""

        ids = np.concatenate([np.asarray(g, dtype=np.int64) for g in snapshot.group_ids if g is not None])
        n_groups = int(ids.max()) + 1 if ids.size else 1
        counts = np.bincount(ids, minlength=n_groups)
        names = (
            list(self.group_names)
            if self.group_names is not None and len(self.group_names) == n_groups
            else [f"group_{g + 1}" for g in range(n_groups)]
        )
        return [(g + 1, names[g], int(counts[g])) for g in range(n_groups)]


def _class_map_paths(root: Path, snapshot: IterationSnapshot, half: int | None):
    stem = f"{root}_half{half}" if half is not None else str(root)
    return [Path(f"{stem}_class{k + 1:03d}.mrc") for k in range(int(snapshot.n_classes))]


def _volume_shape(snapshot):
    n = int(snapshot.ori_size)
    return (n, n, n)


def _write_maps(root: Path, snapshot: IterationSnapshot) -> None:
    from relax.helpers.map_io import write_map

    shape = _volume_shape(snapshot)

    def _write(path, volume_ft):
        write_map(path, _real_from_fourier(volume_ft, shape), voxel_size=snapshot.pixel_size)

    if snapshot.k_class:
        stack = np.asarray(snapshot.means[0]).reshape(int(snapshot.n_classes), -1)
        for k, path in enumerate(_class_map_paths(root, snapshot, None)):
            _write(path, stack[k])
        return
    for h in range(2):
        (path,) = _class_map_paths(root, snapshot, h + 1)
        _write(path, snapshot.means[h])
        if snapshot.unfiltered_means is not None and snapshot.unfiltered_means[h] is not None:
            _write(Path(f"{root}_half{h + 1}_class001_unfil.mrc"), snapshot.unfiltered_means[h])


def _resolution_columns(snapshot, n_shells):
    shells = np.arange(n_shells)
    box = float(snapshot.pixel_size) * float(snapshot.ori_size)
    resolution = shells / box
    with np.errstate(divide="ignore"):
        angstrom = np.where(shells > 0, box / np.maximum(shells, 1), 999.0)
    return shells, resolution, angstrom


def _write_model_stars(root: Path, snapshot: IterationSnapshot, settings: RunSettings, group_rows):
    frame = float(snapshot.ori_size) ** 4
    state = snapshot.state_fields
    current_resolution = float(state["current_resolution"])
    inverse_resolution = (
        0.0 if not np.isfinite(current_resolution) or current_resolution <= 0 else 1.0 / current_resolution
    )
    halves = (None,) if snapshot.k_class else (1, 2)
    paths = []
    for half in halves:
        h = 0 if half is None else half - 1
        stem = str(root) if half is None else f"{root}_half{half}"
        path = Path(f"{stem}_model.star")
        noise = np.asarray(snapshot.noise_shells[h], dtype=np.float64)
        noise = noise.reshape(1, -1) if noise.ndim == 1 else noise
        general = [
            ("rlnReferenceDimensionality", 3),
            ("rlnDataDimensionality", 2),
            ("rlnOriginalImageSize", int(snapshot.ori_size)),
            ("rlnCurrentResolution", inverse_resolution),
            ("rlnCurrentImageSize", int(snapshot.current_size)),
            ("rlnPaddingFactor", float(settings.padding_factor)),
            ("rlnIsHelix", False),
            ("rlnFourierSpaceInterpolator", 1),
            ("rlnMinRadiusNnInterpolation", 10),
            ("rlnPixelSize", float(snapshot.pixel_size)),
            ("rlnNrClasses", int(snapshot.n_classes)),
            ("rlnNrBodies", 1),
            ("rlnNrGroups", len(group_rows)),
            ("rlnNrOpticsGroups", int(noise.shape[0])),
            ("rlnTau2FudgeFactor", float(snapshot.tau2_fudge)),
            ("rlnNormCorrectionAverage", float(snapshot.avg_norm_correction[h])),
            ("rlnSigmaOffsetsAngst", float(snapshot.sigma_offset_angstrom[h])),
            ("rlnOrientationalPriorMode", 1 if state["do_local_search"] else 0),
            ("rlnSigmaPriorRotAngle", float(np.degrees(state["sigma_rot"]))),
            ("rlnSigmaPriorTiltAngle", float(np.degrees(state["sigma_rot"]))),
            ("rlnSigmaPriorPsiAngle", float(np.degrees(state["sigma_psi"]))),
            ("rlnLogLikelihood", 0.0),
            ("rlnAveragePmax", float(state["ave_Pmax"])),
        ]
        text = [f"# Created by relax ({RUN_FILES_FORMAT})\n", _list_block("model_general", general)]
        class_paths = _class_map_paths(root, snapshot, half)
        class_weights = (
            np.asarray(snapshot.class_weights, dtype=np.float64)
            if snapshot.class_weights is not None
            else np.ones(int(snapshot.n_classes)) / int(snapshot.n_classes)
        )
        n_classes = int(snapshot.n_classes)
        text.append(
            _loop_block(
                "model_classes",
                {
                    "rlnReferenceImage": [p.name for p in class_paths],
                    "rlnClassDistribution": class_weights,
                    "rlnAccuracyRotations": np.full(n_classes, float(state["acc_rot"])),
                    "rlnAccuracyTranslationsAngst": np.full(n_classes, float(state["acc_trans"])),
                },
            )
        )
        tau2 = np.asarray(snapshot.tau2_shells, dtype=np.float64)
        dvp = np.asarray(snapshot.data_vs_prior, dtype=np.float64)
        for k in range(n_classes):
            tau2_k = tau2[k] if snapshot.k_class else tau2[h]
            dvp_k = dvp[k] if snapshot.k_class else dvp
            n_shells = tau2_k.shape[-1]
            shells, resolution, angstrom = _resolution_columns(snapshot, n_shells)
            fsc = (
                np.asarray(snapshot.fsc, dtype=np.float64)[:n_shells]
                if snapshot.fsc is not None and not snapshot.k_class
                else np.zeros(n_shells)
            )
            dvp_k = np.pad(dvp_k[:n_shells], (0, max(0, n_shells - dvp_k.shape[-1])))
            fsc = np.pad(fsc, (0, max(0, n_shells - fsc.shape[-1])))
            text.append(
                _loop_block(
                    f"model_class_{k + 1}",
                    {
                        "rlnSpectralIndex": shells,
                        "rlnResolution": resolution,
                        "rlnAngstromResolution": angstrom,
                        "rlnSsnrMap": dvp_k,
                        "rlnGoldStandardFsc": fsc,
                        "rlnReferenceTau2": tau2_k / frame,
                    },
                )
            )
        scale = _group_scales(snapshot, h, len(group_rows))
        text.append(
            _loop_block(
                "model_groups",
                {
                    "rlnGroupNumber": [g for g, _, _ in group_rows],
                    "rlnGroupName": [name for _, name, _ in group_rows],
                    "rlnGroupNrParticles": [count for _, _, count in group_rows],
                    "rlnGroupScaleCorrection": scale,
                },
            )
        )
        for g in range(noise.shape[0]):
            positive = np.flatnonzero(noise[g] > 0.0)  # ml_model.cpp:792-801 skips unused shells
            shells, resolution, _ = _resolution_columns(snapshot, noise.shape[1])
            text.append(
                _loop_block(
                    f"model_optics_group_{g + 1}",
                    {
                        "rlnSpectralIndex": shells[positive],
                        "rlnResolution": resolution[positive],
                        "rlnSigma2Noise": noise[g][positive] / frame,
                    },
                )
            )
        if snapshot.direction_prior is not None and snapshot.direction_prior[h] is not None:
            prior = np.asarray(snapshot.direction_prior[h], dtype=np.float64)
            prior = prior.reshape(n_classes, -1) if snapshot.k_class else prior.reshape(1, -1)
            for k in range(prior.shape[0]):
                text.append(_loop_block(f"model_pdf_orient_class_{k + 1}", {"rlnOrientationDistribution": prior[k]}))
        path.write_text("".join(text))
        paths.append(path)
    return paths


def _group_scales(snapshot, h, n_groups):
    """Per-group scale of half ``h``: every image of a group carries its group's scale."""

    scale = np.ones(n_groups, dtype=np.float64)
    ids = snapshot.group_ids[h]
    per_image = snapshot.scale_corrections[h]
    if ids is None or per_image is None or len(ids) == 0:
        return scale
    ids = np.asarray(ids, dtype=np.int64)
    per_image = np.asarray(per_image, dtype=np.float64)
    scale[ids] = per_image
    if not np.array_equal(scale[ids], per_image):
        raise ValueError("per-image scale corrections differ within a scale group")
    return scale


def _write_sampling_star(root: Path, snapshot: IterationSnapshot, settings: RunSettings) -> None:
    state = snapshot.state_fields
    pixel = float(snapshot.pixel_size)
    items = [
        ("rlnIs3DSampling", True),
        ("rlnIs3DTranslationalSampling", False),
        ("rlnHealpixOrder", int(state["healpix_order"])),
        ("rlnSymmetryGroup", settings.symmetry),
        ("rlnTiltAngleLimit", -91.0),
        ("rlnPsiStep", float(state["angular_step"])),
        ("rlnOffsetRange", float(state["translation_range"]) * pixel),
        ("rlnOffsetStep", float(state["translation_step"]) * pixel),
        ("rlnHelicalOffsetStep", -1.0),
        ("rlnSamplingPerturbInstance", float(snapshot.random_perturbation)),
        ("rlnSamplingPerturbFactor", float(settings.perturbation_factor)),
        ("rlnHealpixOrderOriginal", int(settings.healpix_order_original)),
        ("rlnPsiStepOriginal", -1.0),
        ("rlnOffsetRangeOriginal", float(settings.offset_range_original_angstrom)),
        ("rlnOffsetStepOriginal", float(settings.offset_step_original_angstrom)),
    ]
    Path(f"{root}_sampling.star").write_text(
        f"# Created by relax ({RUN_FILES_FORMAT})\n" + _list_block("sampling_general", items)
    )


def _per_row(values_per_half, half_rows, n_rows, fill, dtype):
    out = np.full(n_rows, fill, dtype=dtype)
    for rows, values in zip(half_rows, values_per_half):
        if values is not None and rows.size:
            out[rows] = np.asarray(values, dtype=dtype).reshape(rows.size, *out.shape[1:])
    return out


def _write_data_star(root: Path, snapshot: IterationSnapshot, particles, optics, half_rows) -> None:
    """``particles``/``optics`` are the input tables as text columns (``read_star_blocks``)."""

    n_rows = len(particles["rlnImageName"])
    eulers = np.zeros((n_rows, 3), dtype=np.float64)
    offsets = np.zeros((n_rows, 2), dtype=np.float64)
    for rows, e, t in zip(half_rows, snapshot.rotation_eulers, snapshot.translations):
        if rows.size:
            if e is not None:
                eulers[rows] = np.asarray(e, dtype=np.float64)
            if t is not None:
                offsets[rows] = np.asarray(t, dtype=np.float64)
    offsets *= float(snapshot.pixel_size)
    # rlnNormCorrection = avg_norm * scale / image_correction: the loop's image correction
    # is (avg_norm / normcorr) * scale (relion_normalization.update_relion_norm_scale_corrections).
    norm = np.ones(n_rows, dtype=np.float64)
    for h, rows in enumerate(half_rows):
        if rows.size and snapshot.image_corrections[h] is not None:
            image = np.asarray(snapshot.image_corrections[h], dtype=np.float64)
            scale = np.asarray(snapshot.scale_corrections[h], dtype=np.float64)
            norm[rows] = float(snapshot.avg_norm_correction[h]) * scale / image
    table = dict(particles)
    columns = {
        "rlnAngleRot": eulers[:, 0],
        "rlnAngleTilt": eulers[:, 1],
        "rlnAnglePsi": eulers[:, 2],
        "rlnOriginXAngst": offsets[:, 0],
        "rlnOriginYAngst": offsets[:, 1],
        "rlnNormCorrection": norm,
        "rlnGroupNumber": _per_row(
            [None if g is None else np.asarray(g) + 1 for g in snapshot.group_ids], half_rows, n_rows, 1, np.int64
        ),
    }
    if snapshot.class_assignments is not None:
        columns["rlnClassNumber"] = _per_row(
            [None if c is None else np.asarray(c) + 1 for c in snapshot.class_assignments],
            half_rows,
            n_rows,
            1,
            np.int64,
        )
    else:
        columns["rlnClassNumber"] = np.ones(n_rows, dtype=np.int64)
    if not snapshot.k_class:
        columns["rlnRandomSubset"] = _per_row(
            [np.full(rows.size, h + 1) for h, rows in enumerate(half_rows)], half_rows, n_rows, 0, np.int64
        )
    if snapshot.max_posterior is not None:
        columns["rlnMaxValueProbDistribution"] = _per_row(snapshot.max_posterior, half_rows, n_rows, 0.0, np.float64)
    if snapshot.significant_counts is not None:
        columns["rlnNrOfSignificantSamples"] = _per_row(snapshot.significant_counts, half_rows, n_rows, 0, np.int64)
    for label, values in columns.items():
        table[label] = values
    blocks = []
    if optics is not None:
        blocks.append(_loop_block("optics", optics))
    blocks.append(_loop_block("particles", table))
    Path(f"{root}_data.star").write_text(f"# Created by relax ({RUN_FILES_FORMAT})\n" + "".join(blocks))


def _write_optimiser_star(root: Path, snapshot: IterationSnapshot, settings: RunSettings, model_paths):
    state = snapshot.state_fields
    k_class = snapshot.k_class
    pixel = float(snapshot.pixel_size)
    items = [
        ("rlnOutputRootName", settings.output_root),
        ("rlnModelStarFile", model_paths[0].name),
    ]
    if not k_class:
        items.append(("rlnModelStarFile2", model_paths[1].name))
    best_resolution = 0.0 if not np.isfinite(state["current_resolution"]) else 1.0 / float(state["current_resolution"])
    items += [
        ("rlnExperimentalDataStarFile", f"{root.name}_data.star"),
        ("rlnOrientSamplingStarFile", f"{root.name}_sampling.star"),
        ("rlnCurrentIteration", int(snapshot.relion_iteration)),
        ("rlnNumberOfIterations", int(settings.nr_iter)),
        ("rlnDoSplitRandomHalves", not k_class),
        ("rlnJoinHalvesUntilThisResolution", float(settings.low_resol_join_halves)),
        ("rlnAdaptiveOversampleOrder", int(settings.adaptive_oversampling)),
        ("rlnAdaptiveOversampleFraction", float(settings.adaptive_fraction)),
        ("rlnRandomSeed", int(settings.random_seed)),
        ("rlnParticleDiameter", float(settings.particle_diameter)),
        ("rlnWidthMaskEdge", int(settings.width_mask_edge)),
        ("rlnDoZeroMask", True),
        ("rlnDoSolventFlattening", False),
        ("rlnDoSolventFscCorrection", bool(settings.do_solvent_fsc)),
        ("rlnSolventMaskName", "None"),
        ("rlnSolventMask2Name", "None"),
        ("rlnBodyStarFile", "None"),
        ("rlnTauSpectrumName", "None"),
        ("rlnMaximumCoarseImageSize", -1),
        ("rlnHighresLimitExpectation", -1.0),
        ("rlnLowresLimitExpectation", -1.0),
        ("rlnIncrementImageSize", int(snapshot.incr_size)),
        ("rlnDoMapEstimation", True),
        ("rlnDoGradientRefine", False),
        ("rlnDoStochasticGradientDescent", False),
        ("rlnTau2FudgeArg", float(snapshot.tau2_fudge)),
        ("rlnMaximumSignificantPoses", int(settings.max_significants)),
        ("rlnDoAutoRefine", not k_class),
        ("rlnDoAutoSampling", not k_class),
        ("rlnAutoLocalSearchesHealpixOrder", int(settings.auto_local_healpix_order)),
        ("rlnNumberOfIterWithoutResolutionGain", int(state["nr_iter_wo_resol_gain"])),
        ("rlnBestResolutionThusFar", best_resolution),
        ("rlnNumberOfIterWithoutChangingAssignments", int(state["nr_iter_wo_large_hidden_variable_changes"])),
        ("rlnDoSkipAlign", False),
        ("rlnDoSkipRotate", False),
        ("rlnOverallAccuracyRotations", float(state["acc_rot"])),
        ("rlnOverallAccuracyTranslationsAngst", float(state["acc_trans"])),
        ("rlnChangesOptimalOrientations", float(state["current_changes_optimal_orientations"])),
        ("rlnChangesOptimalOffsets", float(state["current_changes_optimal_offsets_angstrom"])),
        ("rlnChangesOptimalClasses", float(state["current_changes_optimal_classes"])),
        ("rlnSmallestChangesOrientations", float(state["smallest_changes_optimal_orientations"])),
        ("rlnSmallestChangesOffsets", float(state["smallest_changes_optimal_offsets_angstrom"])),
        ("rlnSmallestChangesClasses", int(round(float(state["smallest_changes_optimal_classes"])))),
        ("rlnLocalSymmetryFile", "None"),
        ("rlnDoHelicalRefine", False),
        ("rlnIgnoreHelicalSymmetry", False),
        ("rlnFourierMask", "None"),
        ("rlnHasConverged", bool(state["has_converged"])),
        ("rlnHasHighFscAtResolLimit", bool(snapshot.has_high_fsc_at_limit)),
        ("rlnHasLargeSizeIncreaseIterationsAgo", 0),
        ("rlnDoCorrectNorm", True),
        ("rlnDoCorrectScale", True),
        ("rlnDoCorrectCtf", True),
        ("rlnDoCenterClasses", False),
        ("rlnDoIgnoreCtfUntilFirstPeak", False),
        ("rlnCtfDataArePhaseFlipped", False),
        ("rlnDoOnlyFlipCtfPhases", False),
        ("rlnRefsAreCtfCorrected", True),
        ("rlnFixSigmaNoiseEstimates", False),
        ("rlnFixSigmaOffsetEstimates", False),
        ("rlnMaxNumberOfPooledParticles", 1),
        ("rlnOffsetRangeX", -1.0),
        ("rlnOffsetRangeY", -1.0),
        ("rlnOffsetRangeZ", -1.0),
    ]
    relax_items = [
        ("relax_format", RUN_FILES_FORMAT),
        ("relax_relion_iteration", int(snapshot.relion_iteration)),
        ("relax_n_classes", int(snapshot.n_classes)),
        ("relax_ori_size", int(snapshot.ori_size)),
        ("relax_pixel_size", pixel),
        ("relax_current_size", int(snapshot.current_size)),
        ("relax_random_perturbation", float(snapshot.random_perturbation)),
        ("relax_sigma_offset_half1", float(snapshot.sigma_offset_angstrom[0])),
        ("relax_sigma_offset_half2", float(snapshot.sigma_offset_angstrom[1])),
        *((f"relax_{key}", value) for key, value in sorted(snapshot.extra.items())),
        *((f"relax_state_{name}", state[name]) for name in REFINEMENT_STATE_SCALAR_FIELDS),
    ]
    text = [
        f"# RELION optimiser; written by relax ({RUN_FILES_FORMAT})\n",
        f"# {settings.command_line}\n" if settings.command_line else "",
        _list_block("optimiser_general", items),
        _list_block("relax_state", relax_items),
    ]
    if snapshot.fsc_for_growth is not None:
        text.append(_loop_block("relax_shells", {"relax_fsc_for_growth": np.asarray(snapshot.fsc_for_growth)}))
    path = Path(f"{root}_optimiser.star")
    staging = path.with_name(path.name + ".partial")
    staging.write_text("".join(text))
    os.replace(staging, path)
    return path


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def _state_value(name, text):
    from relax.helpers.convergence import RefinementState

    default = RefinementState.__dataclass_fields__[name].default
    if isinstance(default, bool):
        return _bool(text)
    if isinstance(default, int):
        return int(text)
    return float(text)


def read_run_files(optimiser_star, *, image_names, half_rows) -> IterationSnapshot:
    """Read the ``run_itNNN_*`` files named by ``optimiser_star`` into a snapshot.

    ``image_names`` are the input particle STAR's ``rlnImageName`` values in input order and
    ``half_rows`` its half layout, as for :class:`RunFileWriter`; ``data.star`` must list the
    same images in the same order. Per-image arrays come back in the dtypes the writing
    loop held them in (``relax_*_dtype`` in ``data_relax_state``).
    """

    from relax.helpers.map_io import load_relax_map

    optimiser_star = Path(optimiser_star).resolve()
    directory = optimiser_star.parent
    blocks = read_star_blocks(optimiser_star)
    general = blocks.get("optimiser_general")
    relax_state = blocks.get("relax_state")
    if general is None:
        raise ValueError(f"{optimiser_star} has no data_optimiser_general block")
    if relax_state is None or relax_state.get("relax_format") != RUN_FILES_FORMAT:
        raise ValueError(
            f"{optimiser_star} was not written by relax ({RUN_FILES_FORMAT}); --continue needs "
            "relax's data_relax_state block, which RELION's own files do not carry"
        )
    relion_iteration = int(general["rlnCurrentIteration"])
    n_classes = int(relax_state["relax_n_classes"])
    k_class = n_classes > 1
    ori_size = int(relax_state["relax_ori_size"])
    pixel_size = float(relax_state["relax_pixel_size"])
    frame = float(ori_size) ** 4
    state_fields = {
        name: _state_value(name, relax_state[f"relax_state_{name}"]) for name in REFINEMENT_STATE_SCALAR_FIELDS
    }
    extra = {
        key[len("relax_") :]: value
        for key, value in relax_state.items()
        if key.startswith("relax_")
        and not key.startswith("relax_state_")
        and key
        not in {
            "relax_format",
            "relax_relion_iteration",
            "relax_n_classes",
            "relax_ori_size",
            "relax_pixel_size",
            "relax_current_size",
            "relax_random_perturbation",
            "relax_sigma_offset_half1",
            "relax_sigma_offset_half2",
        }
    }
    if int(relax_state["relax_relion_iteration"]) != relion_iteration:
        raise ValueError(f"{optimiser_star}: relax and RELION iteration numbers disagree")

    model_names = [general["rlnModelStarFile"]] + ([] if k_class else [general["rlnModelStarFile2"]])
    models = [read_star_blocks(directory / name) for name in model_names]
    halves = [models[0], models[0]] if k_class else models

    def _general(model, label):
        return model["model_general"][label]

    volume_shape = (ori_size, ori_size, ori_size)
    means = []
    for model in models:
        paths = [directory / name for name in model["model_classes"]["rlnReferenceImage"]]
        volumes = [_fourier_from_map(load_relax_map(p), volume_shape) for p in paths]
        means.append(volumes[0] if not k_class else np.stack(volumes, axis=0))
    if k_class:
        means = [means[0], means[0]]
    unfiltered = None
    if not k_class:
        stem = optimiser_star.name[: -len("_optimiser.star")]
        unfil_paths = [directory / f"{stem}_half{h + 1}_class001_unfil.mrc" for h in range(2)]
        if all(p.exists() for p in unfil_paths):
            unfiltered = [_fourier_from_map(load_relax_map(p), volume_shape) for p in unfil_paths]

    def _class_table(model, k):
        return model[f"model_class_{k + 1}"]

    if k_class:
        tau2 = np.stack([_floats(_class_table(models[0], k)["rlnReferenceTau2"]) for k in range(n_classes)]) * frame
        data_vs_prior = np.stack([_floats(_class_table(models[0], k)["rlnSsnrMap"]) for k in range(n_classes)])
        fsc = None
    else:
        tau2 = np.stack([_floats(_class_table(m, 0)["rlnReferenceTau2"]) for m in models]) * frame
        data_vs_prior = _floats(_class_table(models[0], 0)["rlnSsnrMap"])
        fsc = _floats(_class_table(models[0], 0)["rlnGoldStandardFsc"])

    noise_shells = []
    for model in halves:
        n_optics = int(_general(model, "rlnNrOpticsGroups"))
        n_shells = ori_size // 2 + 1
        rows = np.zeros((n_optics, n_shells), dtype=np.float64)
        for g in range(n_optics):
            table = model[f"model_optics_group_{g + 1}"]
            index = _ints(table.get("rlnSpectralIndex", []))
            if index.size and int(index.max()) >= n_shells:
                n_shells = int(index.max()) + 1
                rows = np.pad(rows, ((0, 0), (0, n_shells - rows.shape[1])))
            rows[g, index] = _floats(table.get("rlnSigma2Noise", [])) * frame
        noise_shells.append(rows[0] if n_optics == 1 else rows)

    direction_prior = []
    for model in halves:
        tables = [model.get(f"model_pdf_orient_class_{k + 1}") for k in range(n_classes)]
        if any(t is None or not t.get("rlnOrientationDistribution") for t in tables):
            direction_prior.append(None)
            continue
        prior = np.stack([_floats(t["rlnOrientationDistribution"]) for t in tables])
        direction_prior.append(prior if k_class else prior[0])
    if all(p is None for p in direction_prior):
        direction_prior = None

    class_weights = _floats(models[0]["model_classes"]["rlnClassDistribution"]) if k_class else None
    avg_norm = tuple(float(_general(m, "rlnNormCorrectionAverage")) for m in halves)
    sigma_offset = (float(relax_state["relax_sigma_offset_half1"]), float(relax_state["relax_sigma_offset_half2"]))

    data = read_star_blocks(directory / general["rlnExperimentalDataStarFile"])["particles"]
    names = data["rlnImageName"]
    expected = [str(name) for name in image_names]
    if len(names) != len(expected) or any(a != b for a, b in zip(names, expected)):
        raise ValueError(
            f"{general['rlnExperimentalDataStarFile']} does not list the input particles in the input order"
        )
    eulers = np.stack([_floats(data[c]) for c in ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")], axis=1)
    offsets = np.stack([_floats(data[c]) for c in ("rlnOriginXAngst", "rlnOriginYAngst")], axis=1) / pixel_size
    norm = _floats(data["rlnNormCorrection"])
    group_number = _ints(data["rlnGroupNumber"])
    class_number = _ints(data["rlnClassNumber"]) if k_class else None
    pmax = _floats(data["rlnMaxValueProbDistribution"]) if "rlnMaxValueProbDistribution" in data else None
    nsig = _ints(data["rlnNrOfSignificantSamples"]) if "rlnNrOfSignificantSamples" in data else None

    group_scales = [_floats(m["model_groups"]["rlnGroupScaleCorrection"]) for m in halves]
    euler_dtype = np.dtype(extra.pop("euler_dtype", "float64"))
    translation_dtype = np.dtype(extra.pop("translation_dtype", "float32"))
    correction_dtype = np.dtype(extra.pop("correction_dtype", "float32"))
    rotation_eulers, translations, image_corr, scale_corr, group_ids, classes = [], [], [], [], [], []
    for h, rows in enumerate(half_rows):
        rows = np.asarray(rows, dtype=np.int64)
        ids = group_number[rows] - 1
        scale = group_scales[h][ids]
        rotation_eulers.append(eulers[rows].astype(euler_dtype))
        translations.append(offsets[rows].astype(translation_dtype))
        # The loop's image correction is (avg_norm / normcorr) * group scale.
        image_corr.append(((avg_norm[h] / norm[rows]) * scale).astype(correction_dtype))
        scale_corr.append(scale.astype(correction_dtype))
        group_ids.append(ids)
        classes.append(None if class_number is None else class_number[rows] - 1)

    shells = blocks.get("relax_shells")
    fsc_for_growth = _floats(shells["relax_fsc_for_growth"]) if shells and "relax_fsc_for_growth" in shells else None

    return IterationSnapshot(
        relion_iteration=relion_iteration,
        n_classes=n_classes,
        ori_size=ori_size,
        pixel_size=pixel_size,
        tau2_fudge=float(general["rlnTau2FudgeArg"]),
        means=means,
        tau2_shells=tau2,
        data_vs_prior=data_vs_prior,
        noise_shells=noise_shells,
        sigma_offset_angstrom=sigma_offset,
        current_size=int(relax_state["relax_current_size"]),
        incr_size=int(general["rlnIncrementImageSize"]),
        has_high_fsc_at_limit=_bool(general["rlnHasHighFscAtResolLimit"]),
        random_perturbation=float(relax_state["relax_random_perturbation"]),
        state_fields=state_fields,
        rotation_eulers=rotation_eulers,
        translations=translations,
        image_corrections=image_corr,
        scale_corrections=scale_corr,
        group_ids=group_ids,
        fsc=fsc,
        fsc_for_growth=fsc_for_growth,
        class_weights=class_weights,
        direction_prior=direction_prior,
        class_assignments=classes if k_class else None,
        max_posterior=None if pmax is None else [pmax[np.asarray(r)] for r in half_rows],
        significant_counts=None if nsig is None else [nsig[np.asarray(r)] for r in half_rows],
        avg_norm_correction=avg_norm,
        unfiltered_means=unfiltered,
        extra=extra,
    )


# Host FFTs with recovar's centered convention (fourier_transform_utils.get_dft3/get_idft3,
# norm "backward"), so writing and reading a map needs no device memory: at box 800 one
# complex volume is 4 GB.
def _fft_workers() -> int:
    """Half the CPUs this process may use: a background write shares them with the refinement."""

    return max(1, len(os.sched_getaffinity(0)) // 2)


def _real_from_fourier(volume_ft, volume_shape) -> np.ndarray:
    import scipy.fft

    volume = np.asarray(volume_ft).reshape(volume_shape)
    volume = scipy.fft.ifftn(scipy.fft.ifftshift(volume), workers=_fft_workers())
    return np.ascontiguousarray(scipy.fft.ifftshift(volume).real, dtype=np.float32)


def _fourier_from_map(volume_real, volume_shape) -> np.ndarray:
    """The loop's centered complex64 Fourier volume of an internal-frame real map."""

    import scipy.fft

    real = np.asarray(volume_real, dtype=np.float32).reshape(volume_shape)
    volume = scipy.fft.fftn(scipy.fft.fftshift(real), workers=_fft_workers())
    return np.asarray(scipy.fft.fftshift(volume), dtype=np.complex64).reshape(-1)


def output_root_for(output_dir, prefix: str = "run") -> str:
    return os.path.join(str(output_dir), prefix)
