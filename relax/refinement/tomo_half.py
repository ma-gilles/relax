"""Subtomogram particles (RELION 5 2D stacks) as the units of a refinement half.

RELION refines a subtomogram particle as one unit over its tilt images: every image is scored with
its own projection matrix, CTF and noise, the images' diff2 is summed per particle, and the particle
has one pose (rotation in its subtomogram frame, 3D offset) and one posterior (ml_optimiser.cpp:7650;
acc_ml_optimiser_impl.h:1190-1402). A :class:`TomoHalf` presents a half as those particles, in
particle-STAR row order, over a flat dataset of their tilt images in RELION's ``img_id`` order
(:func:`relax.relion.tomo_input.relion_image_geometry`). The half's scoring is
:func:`relax.refinement.tomo_scoring.score_tomo_half`.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from relax.refinement.optics_shapes import RowLayout


def is_relion5_2d_stack_star(particles_star) -> bool:
    """Whether a particle STAR is RELION 5 subtomogram 2D stacks (``data_general`` flag, ParticleSet::read)."""

    import starfile

    star = starfile.read(particles_star)
    general = star.get("general") if isinstance(star, dict) else None
    if general is None or "rlnTomoSubTomosAre2DStacks" not in getattr(general, "columns", ()):
        return False
    return bool(int(np.asarray(general["rlnTomoSubTomosAre2DStacks"]).ravel()[0]))


class TomoDataset:
    """All particles of a 2D-stack project: unit ``u`` is particle-STAR row ``u``.

    ``images`` is the loaded flat per-tilt dataset (:func:`relax.relion.tomo_input.flatten_relion5_tomo`).
    Particle ``u`` owns images ``image_rows[unit_image_offsets[u]:unit_image_offsets[u + 1]]`` of it, in
    ``img_id`` order, with RELION's ``Aproj`` in ``image_projections`` (and the projection's left matrix,
    :func:`tilt_left_matrices`, in ``image_left``), its tilt-series frame in
    ``image_frames`` and the particle's tomogram in ``unit_tomogram``.
    """

    def __init__(self, images, flat_rows, particles_star, tomograms_star):
        from recovar.data_io.starfile import read_star, star_column

        from relax.relion import tomo_input

        index = tomo_input.tomo_particle_index(flat_rows)
        geometry = tomo_input.relion_image_geometry(flat_rows, index, particles_star, tomograms_star)
        particles, _ = read_star(str(particles_star))
        names = np.asarray(star_column(particles, "rlnTomoParticleName", required=True))
        position = {name: p for p, name in enumerate(index.particle_names)}
        missing = [name for name in names if name not in position]
        if missing or names.size != index.n_particles:
            raise ValueError(
                f"the per-tilt STAR and the particle STAR disagree on the particles ({len(missing)} missing, "
                f"{names.size} vs {index.n_particles})"
            )
        order = np.asarray([position[name] for name in names], dtype=np.int64)
        counts = np.diff(index.image_offsets)[order]
        self.unit_image_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        spans = [np.arange(index.image_offsets[p], index.image_offsets[p + 1]) for p in order]
        self.image_rows = np.concatenate([geometry.rows[s] for s in spans]).astype(np.int64)
        self.image_projections = np.concatenate([geometry.projections[s] for s in spans])
        self.image_left = tilt_left_matrices(images, self.image_projections, self.image_rows)
        # Frame of each image in its tilt series and tomogram of each particle (PPCA tilt groups).
        self.image_frames = np.concatenate([geometry.frames[s] for s in spans]).astype(np.int64)
        self.unit_tomogram = np.asarray(star_column(particles, "rlnTomoName", required=True)).astype(str)
        optics = np.asarray(star_column(particles, "rlnOpticsGroup", required=True), dtype=np.int64)
        if not np.array_equal(optics, index.optics_group[order]):
            raise ValueError("the per-tilt STAR's optics groups disagree with the particle STAR's")
        self.unit_optics_group = optics
        self.particle_names = names
        self.particles_star = str(particles_star)  # its data_general block goes into every output data STAR
        self.images = images
        self.n_units = int(names.size)
        self.image_shape = tuple(int(size) for size in images.image_shape)
        self.volume_shape = tuple(int(size) for size in images.volume_shape)
        self.voxel_size = float(images.voxel_size)
        self.grid_size = self.image_shape[0]

    def subset(self, units):
        """The :class:`TomoHalf` of these particle rows, in this order."""

        units = np.asarray(units, dtype=np.int64).reshape(-1)
        counts = np.diff(self.unit_image_offsets)[units]
        images = (
            np.concatenate([np.arange(self.unit_image_offsets[u], self.unit_image_offsets[u + 1]) for u in units])
            if units.size
            else np.zeros(0, dtype=np.int64)
        )
        return TomoHalf(
            self.images.subset(self.image_rows[images]),
            unit_image_offsets=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
            image_projections=self.image_projections[images],
            image_left=self.image_left[images],
            unit_optics_group=self.unit_optics_group[units],
            rows=units,
            image_frames=self.image_frames[images],
            unit_tomogram=self.unit_tomogram[units],
            image_shape=self.image_shape,
            volume_shape=self.volume_shape,
            voxel_size=self.voxel_size,
        )

    def startup_noise_images(self, units, *, unit_groups, minimum_nr_particles: int = 10):
        """``(group, real-space image)`` of the tilt images RELION's start-up noise estimate reads.

        calculateSumOfPowerSpectraAndAverageImage (ml_optimiser.cpp:2806-3080) visits particles in
        ``units`` order and skips a particle whose optics group already counts
        ``minimum_nr_particles_sigma2_noise`` (10 for subtomograms, :2574). The count goes up once per
        *image* (:3058-3059), so a group's first particle contributes all its tilt images and fills it;
        the loop stops after the particle that fills the last group. Each image is counted once in the
        group's ``sumw`` (the per-image average of setSigmaNoiseEstimatesAndSetAverageImage).
        """

        units = np.asarray(units, dtype=np.int64).reshape(-1)
        unit_groups = np.asarray(unit_groups, dtype=np.int64).reshape(-1)
        group_of = dict(zip(units.tolist(), unit_groups.tolist()))
        for unit in self.startup_units(units, unit_groups=unit_groups, minimum_nr_particles=minimum_nr_particles):
            for tilt, image in enumerate(self.unit_images(unit)):
                if not np.all(np.isfinite(image)):
                    raise RuntimeError(
                        f"start-up noise estimate: particle {self.particle_names[int(unit)]} (row {int(unit)}) has a "
                        f"non-finite pixel in its tilt image {tilt}"
                    )
                yield group_of[int(unit)], image

    def startup_units(self, units, *, unit_groups, minimum_nr_particles: int = 10) -> np.ndarray:
        """The particles (in ``units`` order) whose tilt images RELION's start-up loop reads (:meth:`startup_noise_images`)."""

        from relax.relion.initial_noise import relion_startup_positions

        units = np.asarray(units, dtype=np.int64).reshape(-1)
        sizes = self.unit_image_offsets[units + 1] - self.unit_image_offsets[units]
        return units[relion_startup_positions(unit_groups, sizes, minimum_nr_particles)]

    def unit_images(self, unit) -> np.ndarray:
        """Particle ``unit``'s real-space tilt images in ``img_id`` order, ``[S, N, N]``."""

        rows = self.image_rows[self.unit_image_offsets[unit] : self.unit_image_offsets[unit + 1]]
        batches = [
            np.asarray(batch_images)
            for batch_images, _particles, _local in self.images.image_source.iter_batches(
                batch_size=int(rows.size), batch_mode="images", subset_indices=rows
            )
        ]
        return np.concatenate(batches, axis=0)


def tilt_left_matrices(images, image_projections, image_rows) -> np.ndarray:
    """Each tilt image's projection left matrix ``L``: RELION's ``applyAnisoMag(Aproj)``.

    relion_refine projects and backprojects a tilt image at ``L R`` with
    ``L = inv(M3) Aproj`` (acc_ml_optimiser_impl.h:1088-1104, 1711-1727, 3212-3227;
    ``ObservationModel::applyAnisoMag``), ``M3`` the image's optics-group ``rlnMagMat`` in a 3x3
    identity; without magnification ``L = Aproj``. The 3D shift still projects through ``Aproj``
    (``Experiment::getTranslationInTiltSeries``). ``images`` is the flat per-tilt dataset and
    ``image_rows`` the dataset rows of ``image_projections``. The scale difference
    (``applyScaleDifference``) is 1: tilt images are on the reference grid.
    """

    from recovar.data_io.starfile import star_column

    from relax.relion import optics_aberrations
    from relax.relion.ctf import exact_ctf_source_cache

    image_projections = np.asarray(image_projections, dtype=np.float64)
    _, cache = exact_ctf_source_cache(images, tuple(int(n) for n in images.image_shape))
    labels = {str(label).lstrip("_") for row in cache["optics"].values() for label in row.keys()}
    if not any(label.startswith("rlnMagMat") for label in labels):
        return image_projections
    inverse = {}
    for group, row in cache["optics"].items():
        mag3 = np.eye(3)
        mag3[:2, :2] = optics_aberrations.optics_group_mag_matrix(row)
        inverse[group] = np.linalg.inv(mag3)
    groups = np.asarray(star_column(cache["particles"], "rlnOpticsGroup", required=True), dtype=np.int64)
    if image_projections.shape[0] == 0:
        return image_projections
    left = np.stack([inverse[int(group)] for group in groups[np.asarray(image_rows, dtype=np.int64)]])
    return np.einsum("nij,njk->nik", left.reshape(-1, 3, 3), image_projections)


def load_tomo_dataset(
    particles_star, tomograms_star, flat_star, *, datadir, lazy: bool, read_policy=None
) -> TomoDataset:
    """A RELION 5 2D-stack project as a :class:`TomoDataset`; the per-tilt STAR is written to ``flat_star``.

    ``read_policy`` (:class:`relax.io.particle_io.ParticleReadPolicy`) applies ``--scratch_dir`` to
    the stacks the per-tilt STAR names: the particle STAR has no ``_rlnImageName`` rows to stage from.
    """

    from recovar.data_io.cryoem_dataset import load_dataset
    from recovar.data_io.starfile import read_star

    from relax.io.particle_io import assert_reads_from_scratch, prepare_particle_reads
    from relax.relion.tomo_input import flatten_relion5_tomo

    flat_star = flatten_relion5_tomo(particles_star, tomograms_star, flat_star)
    scratch = None if read_policy is None else prepare_particle_reads(
        str(flat_star), read_policy, datadir=datadir, compact=False
    )
    images = load_dataset(str(flat_star), datadir=datadir, lazy=lazy, dtype=np.complex64, absent_angles_zero=True)
    assert_reads_from_scratch(images, scratch)
    return TomoDataset(images, read_star(str(flat_star))[0], particles_star, tomograms_star)


class TomoHalf:
    """One half as subtomogram particles over their tilt images.

    ``n_units`` counts particles; ``images`` is the flat dataset of their tilt images, particle ``u``'s
    at ``unit_image_offsets[u]:[u + 1]`` in ``img_id`` order, with ``image_projections`` (RELION's
    ``Aproj``, which places the particle's 3D shifts) and ``image_left`` (the projection's left matrix,
    :func:`tilt_left_matrices`). ``image_shape``, ``volume_shape`` and ``voxel_size`` are the reference model's.
    """

    def __init__(
        self,
        images,
        *,
        unit_image_offsets,
        image_projections,
        image_left,
        unit_optics_group,
        rows,
        image_shape,
        volume_shape,
        voxel_size,
        image_frames,
        unit_tomogram,
    ):
        self.images = images
        self.unit_image_offsets = np.asarray(unit_image_offsets, dtype=np.int64)
        self.image_projections = np.asarray(image_projections, dtype=np.float64)
        self.image_left = np.asarray(image_left, dtype=np.float64)
        self.unit_optics_group = np.asarray(unit_optics_group, dtype=np.int64)
        # Each image's tilt-series frame and each unit's tomogram, as in TomoDataset.
        self.image_frames = np.asarray(image_frames, dtype=np.int64)
        self.unit_tomogram = np.asarray(unit_tomogram)
        self.image_shape = tuple(int(size) for size in image_shape)
        self.volume_shape = tuple(int(size) for size in volume_shape)
        self.voxel_size = float(voxel_size)
        self.grid_size = self.image_shape[0]
        self.n_units = int(self.unit_image_offsets.size - 1)
        self.n_images = int(self.unit_image_offsets[-1])
        if int(images.n_units) != self.n_images or not (
            self.image_projections.shape == self.image_left.shape == (self.n_images, 3, 3)
        ):
            raise ValueError("a tomo half needs one dataset image and one Aproj per tilt image")
        # Particle-STAR row of each unit, as a loaded dataset's index layout reports it.
        self._index_layout = RowLayout(np.asarray(rows, dtype=np.int64))

    def subset(self, units):
        """The :class:`TomoHalf` of these units (particles), in this order, with their tilt images."""

        units = np.asarray(units, dtype=np.int64).reshape(-1)
        images = (
            np.concatenate([np.arange(self.unit_image_offsets[u], self.unit_image_offsets[u + 1]) for u in units])
            if units.size
            else np.zeros(0, dtype=np.int64)
        )
        return TomoHalf(
            self.images.subset(images),
            unit_image_offsets=np.concatenate([[0], np.cumsum(np.diff(self.unit_image_offsets)[units])]).astype(np.int64),
            image_projections=self.image_projections[images],
            image_left=self.image_left[images],
            unit_optics_group=self.unit_optics_group[units],
            rows=self._index_layout.rows[units],
            image_shape=self.image_shape,
            volume_shape=self.volume_shape,
            voxel_size=self.voxel_size,
            image_frames=self.image_frames[images],
            unit_tomogram=self.unit_tomogram[units],
        )

    def image_particle(self) -> np.ndarray:
        """Unit of every tilt image."""

        return np.repeat(np.arange(self.n_units), np.diff(self.unit_image_offsets))

    def __getattr__(self, name):
        raise AttributeError(f"a subtomogram half has no {name!r}; its images are in `images`")


@dataclasses.dataclass(frozen=True)
class TomoSampling:
    """One iteration's sampling of a subtomogram half (RELION HealpixSampling with 3D translations)."""

    healpix_order: int
    oversampling_order: int
    offset_range_angst: float
    offset_step_angst: float
    random_perturbation: float
    coarse_size: int
    fine_size: int


def numbered_iteration_tomo_sampling(
    state,
    image_geometry,
    *,
    local_sampling,
    grid_healpix_order: int,
    random_perturbation: float,
    coarse_size: int | None,
    fine_size: int | None,
) -> TomoSampling:
    """A numbered iteration's subtomogram sampling from the refinement state.

    ``local_sampling`` is the iteration's local-search sampling, or None for a global search on the
    exhaustive grid of ``grid_healpix_order``. ``coarse_size`` and ``fine_size`` are the pass-1 and
    pass-2 Fourier windows; None is the full image.
    """
    use_local = local_sampling is not None
    tomo_oversampling = int(state.adaptive_oversampling)
    box_size = image_geometry.image_shape[0]
    return local_tomo_sampling(
        # A global search's grid order is its pass-1 order.
        fine_order=int(local_sampling.search.healpix_order) if use_local else int(grid_healpix_order) + tomo_oversampling,
        oversampling_order=tomo_oversampling,
        translation_range_px=state.translation_range,
        translation_step_px=state.translation_step,
        voxel_size=image_geometry.pixel_size_angstrom,
        random_perturbation=local_sampling.perturbation if use_local else random_perturbation,
        pass1_size=box_size if coarse_size is None else coarse_size,
        current_size=box_size if fine_size is None else fine_size,
    )


def local_tomo_sampling(
    *,
    fine_order: int,
    oversampling_order: int,
    translation_range_px: float,
    translation_step_px: float,
    voxel_size: float,
    random_perturbation: float,
    pass1_size: int | None,
    current_size: int,
) -> TomoSampling:
    """A :class:`TomoSampling` with pass 1 one oversampling order below the fine order (a local search, the
    final pass, and through :func:`numbered_iteration_tomo_sampling` a numbered iteration).

    ``pass1_size`` is RELION's parent-pass image size; without one (no adaptive parent pass, e.g. the
    final pass at a high order) the coarse size is the current size, as RELION's coarse_size is.
    The refinement state keeps the offset range and step in pixels of the model grid.
    """

    return TomoSampling(
        healpix_order=int(fine_order) - int(oversampling_order),
        oversampling_order=int(oversampling_order),
        offset_range_angst=float(translation_range_px) * float(voxel_size),
        offset_step_angst=float(translation_step_px) * float(voxel_size),
        random_perturbation=float(random_perturbation),
        coarse_size=int(current_size if pass1_size is None else pass1_size),
        fine_size=int(current_size),
    )


def tomo_translation_grids(sampling: TomoSampling, pixel_size: float):
    """RELION's coarse 3D grid in Angstrom and pixels and its oversampled children (healpix_sampling.cpp)."""

    from relax import sampling as relax_sampling

    coarse_angst = relax_sampling.get_relion_translation_grid_3d(
        sampling.offset_range_angst, sampling.offset_step_angst
    )
    coarse_px, _ = relax_sampling.relion_translations_in_pixel_3d(
        coarse_angst,
        sampling.offset_step_angst,
        oversampling_order=0,
        pixel_size=pixel_size,
        random_perturbation=sampling.random_perturbation,
    )
    fine_px, parent = relax_sampling.relion_translations_in_pixel_3d(
        coarse_angst,
        sampling.offset_step_angst,
        oversampling_order=sampling.oversampling_order,
        pixel_size=pixel_size,
        random_perturbation=sampling.random_perturbation,
    )
    return coarse_angst, coarse_px, fine_px, parent


@dataclasses.dataclass(frozen=True, kw_only=True)
class TiltImageAccuracyInputs:
    """Per-tilt-image inputs of RELION's expected-accuracy estimate for a tomo half, by tilt image."""

    # (n_particles + 1,): particle p's tilt images are image_offsets[p]:image_offsets[p + 1].
    image_offsets: np.ndarray
    # (n_images, 3, 3): each tilt image's projection (Aproj).
    image_projections: np.ndarray
    # (n_images, 7): defocus U, V (A), angle (deg), B-factor, scale, phase shift, dose (-999: the B-factor
    # carries the dose).
    image_ctf: np.ndarray
    # (n_images,) each: the image's optics group's voltage (kV), Cs (mm) and amplitude contrast.
    voltage: np.ndarray
    spherical_aberration: np.ndarray
    amplitude_contrast: np.ndarray


def tilt_image_accuracy_inputs(half: TomoHalf) -> TiltImageAccuracyInputs:
    """Per-tilt-image inputs of RELION's expected-accuracy estimate for a tomo half.

    Each image's ``Aproj`` and CTF as Experiment::addImageToParticle stores them (exp_model.cpp:196-243):
    defocus, astigmatism angle, scale and phase shift from the tilt's CTF; the cumulative dose as the
    CTF dose (Grant-Grigorieff damping, ctf.h:219-231), or, with a B-factor per electron dose, dose
    ``-999`` and B-factor ``B_dose * dose``. Also each image's optics constants (voltage, Cs, Q0).
    """

    from recovar.data_io.starfile import read_star, star_column

    rows = np.asarray(
        half.images._index_layout.original_image_indices_for_local(np.arange(half.n_images)), dtype=np.int64
    )
    table, optics = read_star(str(half.images.particles_file))

    def column(name, default=None):
        values = star_column(table, name)
        if values is None:
            if default is None:
                raise ValueError(f"the per-tilt STAR has no {name}")
            return np.full(rows.size, float(default))
        return np.asarray(values, dtype=np.float64)[rows]

    dose = column("rlnMicrographPreExposure")
    b_per_dose = column("rlnCtfBfactorPerElectronDose", 0.0)
    per_dose = b_per_dose > 0.0
    image_ctf = np.stack(
        [
            column("rlnDefocusU"),
            column("rlnDefocusV"),
            column("rlnDefocusAngle"),
            np.where(per_dose, b_per_dose * dose, 0.0),
            column("rlnCtfScalefactor", 1.0),
            column("rlnPhaseShift", 0.0),
            np.where(per_dose, -999.0, dose),
        ],
        axis=1,
    )
    group_of = np.asarray(star_column(table, "rlnOpticsGroup", required=True), dtype=np.int64)[rows]
    labels = np.asarray(star_column(optics, "rlnOpticsGroup", required=True), dtype=np.int64)
    row_of_label = {int(label): i for i, label in enumerate(labels)}
    optics_rows = np.asarray([row_of_label[int(g)] for g in group_of], dtype=np.int64)

    def optics_column(name):
        return np.asarray(star_column(optics, name, required=True), dtype=np.float64)[optics_rows]

    return TiltImageAccuracyInputs(
        image_offsets=half.unit_image_offsets,
        image_projections=half.image_projections,
        image_ctf=image_ctf,
        voltage=optics_column("rlnVoltage"),
        spherical_aberration=optics_column("rlnSphericalAberration"),
        amplitude_contrast=optics_column("rlnAmplitudeContrast"),
    )


@dataclasses.dataclass(frozen=True)
class TomoLocalRotations:
    """The particles' local orientations at one HEALPix order (RELION's nonzero-prior orientations).

    ``coarse_rotation_ids`` are the union's grid rotations (ascending, the loop's index order);
    ``unit_rotation_ids[u]`` index that union (ascending) with ``unit_rotation_log_priors[u]``.
    """

    coarse_rotation_ids: np.ndarray
    unit_rotation_ids: tuple
    unit_rotation_log_priors: tuple

    def support_priors(self, supports, n_coarse_trans: int):
        """Each unit's prior over its support's coarse rotations, ascending (the pass-2 tables' contract)."""

        out = []
        for unit, cells in enumerate(supports):
            rotations = np.unique(np.asarray(cells, dtype=np.int64) // int(n_coarse_trans))
            position = np.searchsorted(self.unit_rotation_ids[unit], rotations)
            if np.any(position >= self.unit_rotation_ids[unit].size) or np.any(
                self.unit_rotation_ids[unit][np.minimum(position, self.unit_rotation_ids[unit].size - 1)] != rotations
            ):
                raise ValueError(f"particle {unit}'s support leaves its local rotations")
            out.append(np.asarray(self.unit_rotation_log_priors[unit], dtype=np.float32)[position])
        return out


def tomo_local_rotations(
    previous_eulers_deg,
    *,
    sigma_rot,
    sigma_psi,
    healpix_order: int,
    random_perturbation: float,
    voxel_size: float,
    symmetry: str = "C1",
) -> TomoLocalRotations:
    """Every particle's local orientations and log priors around its previous pose (subtomogram local search).

    The single-particle local search's neighbourhoods (relax.local_search.layout.build_local_hypothesis_layout,
    RELION's selectOrientationsWithNonZeroPriorProbability) with the particle's subtomogram-frame angles;
    only the rotation part is used (the 3D offset prior is the particle's own, score_tomo_half).
    """

    from relax.local_search.layout import build_local_hypothesis_layout
    from relax.sampling import build_local_search_grid_metadata, relion_angular_sampling_deg

    eulers = np.asarray(previous_eulers_deg, dtype=np.float64)
    n_units = int(eulers.shape[0])
    layout = build_local_hypothesis_layout(
        eulers,
        None,
        sigma_rot,
        sigma_psi,
        int(healpix_order),
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((n_units, 2), dtype=np.float32),
        1.0,
        None,
        float(voxel_size),
        grid_metadata=build_local_search_grid_metadata(int(healpix_order), symmetry=symmetry),
        rotation_log_prior=None,
        rotation_grid_random_perturbation=float(random_perturbation),
        rotation_grid_angular_sampling_deg=relion_angular_sampling_deg(int(healpix_order), adaptive_oversampling=0),
        dtype=np.float32,
    )
    offsets = np.asarray(layout.rotation_offsets, dtype=np.int64)
    ids = np.asarray(layout.rotation_ids_flat, dtype=np.int64)
    priors = np.asarray(layout.rotation_log_priors_flat, dtype=np.float32)
    union = np.unique(ids)
    unit_ids, unit_priors = [], []
    for unit in range(n_units):
        rows = slice(int(offsets[unit]), int(offsets[unit + 1]))
        order = np.argsort(ids[rows], kind="stable")
        unit_ids.append(np.searchsorted(union, ids[rows][order]).astype(np.int64))
        unit_priors.append(priors[rows][order])
    return TomoLocalRotations(
        coarse_rotation_ids=union, unit_rotation_ids=tuple(unit_ids), unit_rotation_log_priors=tuple(unit_priors)
    )
