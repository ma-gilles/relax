"""RELION 5 subtomogram (2D-stack) input.

relion_refine reads a tomo particle as one ``ExpImage`` per visible tilt, each with
its own projection ``Aproj A``, depth-corrected CTF and cumulative dose
(``Experiment::read``, RELION 5.0.1 ``src/exp_model.cpp:988-1026``). relax reads the
same data through RECOVAR's RELION 5 reader, ``parse_relion5_tomo.convert``, which
writes one STAR row per particle-tilt with per-tilt Euler angles, projected origin,
depth-corrected defocus, ``rlnCtfScalefactor``, ``rlnMicrographPreExposure`` and the
particle's optics group and half set. :class:`TomoParticleIndex` records which rows
belong to which particle.

The per-tilt CTF damping follows RELION exactly (:func:`relax.relion.ctf.relion_tomo_damping`); the
exact RELION CTF operand applies it to rows that carry ``rlnMicrographPreExposure``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
from recovar.data_io.starfile import read_star, star_column


def read_optimisation_set(path) -> tuple[Path, Path]:
    """Particles and tomograms STAR paths of a RELION optimisation set.

    ``relion_refine --ios`` fills ``--i`` and ``--tomograms`` from
    ``rlnTomoParticlesFile`` and ``rlnTomoTomogramsFile``; the paths are relative to
    the project directory, which is where relion_refine runs.
    """

    path = Path(path)
    table, _ = read_star(str(path))
    root = path.parent
    particles = star_column(table, "rlnTomoParticlesFile", required=True)
    tomograms = star_column(table, "rlnTomoTomogramsFile", required=True)
    return root / str(np.asarray(particles)[0]), root / str(np.asarray(tomograms)[0])


def _star_loop_bounds(lines: list[str], block: str) -> tuple[int, int, int]:
    """``(first label line, first data row, end)`` of the ``loop_`` of STAR block ``data_<block>``."""

    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == f"data_{block}")
        loop = next(i for i in range(start + 1, len(lines)) if lines[i].strip() == "loop_")
    except StopIteration:
        raise ValueError(f"STAR text has no data_{block} loop") from None
    first = loop + 1
    while first < len(lines) and lines[first].strip().startswith("_rln"):
        first += 1
    end = first
    while end < len(lines) and lines[end].strip() and not lines[end].strip().startswith("data_"):
        end += 1
    return loop + 1, first, end


def _with_star_columns(path, block: str, defaults: dict[str, str]) -> str | None:
    """The text of STAR ``path`` with each absent label of ``defaults`` added to ``data_<block>``'s loop.

    Every row gets the default's token appended; every other token of the file is unchanged. ``None``
    when the loop already has all the labels.
    """

    lines = Path(path).read_text().splitlines()
    labels_at, first, end = _star_loop_bounds(lines, block)
    present = {lines[i].split()[0] for i in range(labels_at, first)}
    missing = [(label, value) for label, value in defaults.items() if label not in present]
    if not missing:
        return None
    n_labels = first - labels_at
    head = [f"{label} #{n_labels + 1 + k}" for k, (label, _) in enumerate(missing)]
    tail = "".join(f" {value}" for _, value in missing)
    return "\n".join(lines[:first] + head + [line + tail for line in lines[first:end]] + lines[end:]) + "\n"


def _reader_inputs(particles_star, tomograms_star, output_star) -> tuple[Path, Path]:
    """The particle and tomogram STARs RECOVAR's reader is given: RELION's files, or copies beside
    ``output_star`` in which the values RELION takes for an absent label are written out (relax#79).

    RECOVAR's reader needs these labels on every row; RELION's own tools leave them out:

    - ``rlnRandomSubset`` (particles): ``relion_tomo_subtomo`` writes none. relion_refine draws the halves
      itself (``Experiment::divideParticlesInRandomHalves``, where 0 means not assigned), and so does
      relax, from the particle STAR (:func:`relax.star.input_particle_table.relion_random_subsets`).
      The copy carries 0.
    - ``rlnCtfScalefactor`` and ``rlnTomoXTilt`` (each tilt-series STAR): 1 and 0 when absent
      (``TomogramSet::loadTomogram``, jaz/tomography/tomogram_set.cpp:397-429). The copied tilt-series
      STARs are named by absolute path in a copy of the tomogram STAR.

    Every other token of the files is unchanged, and nothing is written into the project.
    """

    particles_star, tomograms_star, output_star = Path(particles_star), Path(tomograms_star), Path(output_star)
    stem = output_star.stem
    text = _with_star_columns(particles_star, "particles", {"_rlnRandomSubset": "0"})
    if text is not None:
        particles_star = output_star.with_name(f"{stem}_input_particles.star")
        particles_star.parent.mkdir(parents=True, exist_ok=True)
        particles_star.write_text(text)

    tomograms, _ = read_star(str(tomograms_star))
    copies, originals = {}, {}
    for name, series in zip(
        np.asarray(star_column(tomograms, "rlnTomoName", required=True)).astype(str),
        np.asarray(star_column(tomograms, "rlnTomoTiltSeriesStarFile", required=True)).astype(str),
    ):
        text = _with_star_columns(
            tomograms_star.parent / series, name, {"_rlnCtfScalefactor": "1.000000", "_rlnTomoXTilt": "0.000000"}
        )
        if text is not None:
            copy = output_star.with_name(f"{stem}_input_tilt_series") / f"{name}.star"
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_text(text)
            copies[series] = str(copy.resolve())
        else:
            originals[series] = str((tomograms_star.parent / series).resolve())
    if copies:
        # The copy lives outside the project, so it names every tilt-series STAR by absolute path.
        copies.update(originals)
        lines = tomograms_star.read_text().splitlines()
        _, first, end = _star_loop_bounds(lines, "global")  # TomogramSet's table name
        for i in range(first, end):
            lines[i] = "\t".join(copies.get(token, token) for token in lines[i].split())
        tomograms_star = output_star.with_name(f"{stem}_input_tomograms.star")
        tomograms_star.write_text("\n".join(lines) + "\n")
    return particles_star, tomograms_star


def flatten_relion5_tomo(particles_star, tomograms_star, output_star) -> Path:
    """Write the per-tilt STAR relax reads (RECOVAR's RELION 5 reader) and return it.

    Tomograms with ``rlnCtfBfactorPerElectronDose`` carry it on every row as the
    column ``_rlnCtfBfactorPerElectronDose``. The per-tilt optics table keeps the particle
    STAR's optics features (:data:`relax.star.metadata.OPTICS_FEATURE_LABELS`:
    ``rlnCtfDataAreCtfPremultiplied``, beam tilt, odd and even Zernike terms, ``rlnMagMat``):
    relion_refine reads them from the particle STAR's optics groups for every tilt image
    (``Experiment::read``, exp_model.cpp:863), and relax's exact CTF and aberration operands
    read them from the per-tilt STAR.
    """

    from recovar.commands.parse_relion5_tomo import convert
    from recovar.data_io import starfile

    output_star = Path(output_star)
    output_star.parent.mkdir(parents=True, exist_ok=True)
    particles, particle_optics = read_star(str(particles_star))
    reader_particles, reader_tomograms = _reader_inputs(particles_star, tomograms_star, output_star)
    convert(str(reader_tomograms), str(reader_particles), str(output_star))
    rows, optics = read_star(str(output_star))
    changed = False
    tomograms, _ = read_star(str(tomograms_star))
    bfactor = star_column(tomograms, "rlnCtfBfactorPerElectronDose")
    if bfactor is not None and np.any(np.asarray(bfactor, dtype=np.float64) > 0.0):
        tomo_of_particle = dict(
            zip(
                np.asarray(star_column(particles, "rlnTomoParticleName", required=True)),
                np.asarray(star_column(particles, "rlnTomoName", required=True)),
            )
        )
        per_tomo = dict(
            zip(np.asarray(star_column(tomograms, "rlnTomoName", required=True)), np.asarray(bfactor, dtype=np.float64))
        )
        names = np.asarray(star_column(rows, "rlnGroupName", required=True))
        rows["_rlnCtfBfactorPerElectronDose"] = [per_tomo[tomo_of_particle[name]] for name in names]
        changed = True
    from relax.star.metadata import OPTICS_FEATURE_LABELS

    features = {label for labels in OPTICS_FEATURE_LABELS.values() for label in labels}
    written = {str(name).lstrip("_") for name in optics.columns}
    missing = [
        name
        for name in particle_optics.columns
        if str(name).lstrip("_") in features and str(name).lstrip("_") not in written
    ]
    if missing:
        source_row = {
            int(group): row
            for row, group in enumerate(np.asarray(star_column(particle_optics, "rlnOpticsGroup", required=True)))
        }
        rows_of_flat = [
            source_row[int(group)] for group in np.asarray(star_column(optics, "rlnOpticsGroup", required=True))
        ]
        for name in missing:
            label = "_" + str(name).lstrip("_")
            optics[label] = np.asarray(particle_optics[name])[rows_of_flat]
        changed = True
    if changed:
        starfile.write_star(str(output_star), rows, data_optics=optics)
    return output_star


@dataclasses.dataclass(frozen=True)
class TomoParticleIndex:
    """Rows of the per-tilt STAR that make up each particle.

    Rows ``image_offsets[p]:image_offsets[p + 1]`` are the visible tilts of particle
    ``p``, in increasing cumulative dose. One particle has one half set and one optics
    group, as in RELION (the half set and optics group are per particle there).
    """

    particle_names: np.ndarray
    image_offsets: np.ndarray
    half_set: np.ndarray
    optics_group: np.ndarray

    @property
    def n_particles(self) -> int:
        return int(self.particle_names.size)

    @property
    def n_images(self) -> int:
        return int(self.image_offsets[-1])

    def image_particle(self) -> np.ndarray:
        """Particle index of every row."""

        return np.repeat(np.arange(self.n_particles), np.diff(self.image_offsets))


def tomo_particle_index(rows: pd.DataFrame) -> TomoParticleIndex:
    """Group per-tilt rows by ``rlnGroupName``; rows of one particle must be contiguous."""

    names = np.asarray(star_column(rows, "rlnGroupName", required=True))
    starts = np.flatnonzero(np.r_[True, names[1:] != names[:-1]])
    offsets = np.r_[starts, names.size].astype(np.int64)
    particle_names = names[starts]
    if np.unique(particle_names).size != particle_names.size:
        raise ValueError("tilt rows of one particle are not contiguous in the per-tilt STAR")

    def per_particle(label):
        values = np.asarray(star_column(rows, label, required=True)).astype(np.int64)
        first = values[starts]
        if np.any(values != np.repeat(first, np.diff(offsets))):
            raise ValueError(f"tilts of one particle disagree on {label}")
        return first

    return TomoParticleIndex(
        particle_names=particle_names,
        image_offsets=offsets,
        half_set=per_particle("rlnRandomSubset"),
        optics_group=per_particle("rlnOpticsGroup"),
    )


def relion_tilt_rotations(x_tilt_deg, y_tilt_deg, z_rot_deg) -> np.ndarray:
    """``[F, 3, 3]`` rotation part ``r2 r1 r0`` of RELION's tilt projection matrices.

    ``Tomogram::setProjectionMatrix`` (jaz/tomography/tomogram.cpp:57-62) builds
    ``Rz(rlnTomoZRot) Ry(rlnTomoYTilt) Rx(rlnTomoXTilt)`` with gravis
    ``t3Matrix::rotation``, the GL axis-angle formula in degrees (t3Matrix.h:478-496).
    The shifts ``s0``, ``s1``, ``s2`` only move positions, not directions.
    """

    def axis_rotation(angle_deg, i, j):
        angle = np.deg2rad(np.asarray(angle_deg, dtype=np.float64).reshape(-1))
        m = np.tile(np.eye(3), (angle.size, 1, 1))
        m[:, i, i] = m[:, j, j] = np.cos(angle)
        m[:, i, j] = -np.sin(angle)
        m[:, j, i] = np.sin(angle)
        return m

    return axis_rotation(z_rot_deg, 0, 1) @ axis_rotation(y_tilt_deg, 2, 0) @ axis_rotation(x_tilt_deg, 1, 2)


@dataclasses.dataclass(frozen=True)
class TomoImageGeometry:
    """Each image's place in RELION's particle, in RELION's ``img_id`` order.

    relion_refine adds a particle's visible tilts in tilt-series frame order
    (exp_model.cpp:1012-1025), not in dose order. Image ``i`` is per-tilt STAR row
    ``rows[i]``; particle ``p`` owns images ``index.image_offsets[p]:[p + 1]``.
    ``projections[i]`` is RELION's ``Aproj``: the frame's tilt rotation times the
    particle's subtomogram matrix (``P = projectionMatrices[f] * A``, exp_model.cpp:1007,
    1020; its 3x3 part is copied in addImageToParticle, exp_model.cpp:204-208).
    """

    rows: np.ndarray
    frames: np.ndarray
    projections: np.ndarray


def relion_image_geometry(
    rows: pd.DataFrame, index: TomoParticleIndex, particles_star, tomograms_star
) -> TomoImageGeometry:
    """Put ``rows`` (the per-tilt STAR, ``index`` its particles) in RELION's image order.

    The subtomogram matrix is ``Euler::anglesToMatrix3`` of ``rlnTomoSubtomogramRot/Tilt/Psi``
    (particle_set.cpp:381-398, Euler_angles_relion.h:38-47, the same matrix as RELION's
    ``Euler_angles2matrix``) and the identity without those columns.
    """

    from ast import literal_eval

    from relax.sampling.healpix import euler_angles_to_matrix

    particles_star, tomograms_star = Path(particles_star), Path(tomograms_star)
    particles, _ = read_star(str(particles_star))
    tomograms, _ = read_star(str(tomograms_star))
    project_root = tomograms_star.parent
    names = np.asarray(star_column(particles, "rlnTomoParticleName", required=True))
    source = {name: p for p, name in enumerate(names)}
    tomo_names = np.asarray(star_column(particles, "rlnTomoName", required=True))
    visible = np.asarray(star_column(particles, "rlnTomoVisibleFrames", required=True))
    if star_column(particles, "rlnTomoSubtomogramRot") is not None:
        subtomogram = euler_angles_to_matrix(
            np.stack(
                [
                    np.asarray(star_column(particles, label, required=True), dtype=np.float64)
                    for label in ("rlnTomoSubtomogramRot", "rlnTomoSubtomogramTilt", "rlnTomoSubtomogramPsi")
                ],
                axis=1,
            )
        )
    else:
        subtomogram = np.tile(np.eye(3), (names.size, 1, 1))

    series = {}
    series_file = dict(
        zip(
            np.asarray(star_column(tomograms, "rlnTomoName", required=True)),
            np.asarray(star_column(tomograms, "rlnTomoTiltSeriesStarFile", required=True)),
        )
    )

    def tilt_series(tomo):
        if tomo not in series:
            table, _ = read_star(str(project_root / str(series_file[tomo])))
            x_tilt = star_column(table, "rlnTomoXTilt")
            rotations = relion_tilt_rotations(
                np.zeros(len(table)) if x_tilt is None else x_tilt,
                star_column(table, "rlnTomoYTilt", required=True),
                star_column(table, "rlnTomoZRot", required=True),
            )
            frame_of = {str(m): f for f, m in enumerate(star_column(table, "rlnMicrographName", required=True))}
            series[tomo] = (rotations, frame_of)
        return series[tomo]

    micrographs = np.asarray(star_column(rows, "rlnMicrographName", required=True)).astype(str)
    out_rows = np.empty(index.n_images, dtype=np.int64)
    frames = np.empty(index.n_images, dtype=np.int64)
    projections = np.empty((index.n_images, 3, 3), dtype=np.float64)
    for p, name in enumerate(index.particle_names):
        s = source[name]
        rotations, frame_of = tilt_series(tomo_names[s])
        start, stop = int(index.image_offsets[p]), int(index.image_offsets[p + 1])
        particle_rows = np.arange(start, stop)
        particle_frames = np.array([frame_of[m] for m in micrographs[particle_rows]], dtype=np.int64)
        order = np.argsort(particle_frames, kind="stable")
        expected = [f for f, v in enumerate(literal_eval(str(visible[s]))) if int(v) == 1]
        if particle_frames[order].tolist() != expected:
            raise ValueError(f"per-tilt rows of {name} are not its visible frames {expected}")
        out_rows[start:stop] = particle_rows[order]
        frames[start:stop] = particle_frames[order]
        projections[start:stop] = rotations[particle_frames[order]] @ subtomogram[s]
    return TomoImageGeometry(rows=out_rows, frames=frames, projections=projections)
