"""Subtomogram particles (RELION 5 2D stacks) as the units of a refinement half (S4.2).

RELION refines a subtomogram particle as one unit over its tilt images: every image is scored with
its own projection matrix, CTF and noise, the images' diff2 is summed per particle, and the particle
has one pose (rotation in its subtomogram frame, 3D offset) and one posterior (ml_optimiser.cpp:7650;
acc_ml_optimiser_impl.h:1737-2194). A :class:`TomoHalf` presents a half as those particles, in
particle-STAR row order, over a flat dataset of their tilt images in RELION's ``img_id`` order
(:func:`relax.relion.tomo_input.relion_image_geometry`). The half's scoring goes through
:func:`score_tomo_half`: the coarse pass per particle (:mod:`relax.scoring.tomo_coarse`) and the
resident fine pass and M-step over tilt units (``resident_pass2._resident_pass2(tilt=...)``).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from relax.refinement import tomo_particles
from relax.refinement.optics_shapes import _RowLayout


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
    ``img_id`` order, with RELION's ``Aproj`` in ``image_projections``.
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
        optics = np.asarray(star_column(particles, "rlnOpticsGroup", required=True), dtype=np.int64)
        if not np.array_equal(optics, index.optics_group[order]):
            raise ValueError("the per-tilt STAR's optics groups disagree with the particle STAR's")
        self.unit_optics_group = optics
        self.particle_names = names
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
            unit_optics_group=self.unit_optics_group[units],
            rows=units,
            image_shape=self.image_shape,
            volume_shape=self.volume_shape,
            voxel_size=self.voxel_size,
        )

    def startup_noise_images(self, units, *, unit_groups, minimum_nr_particles: int = 10):
        """``(group, real-space image)`` of the tilt images RELION's start-up noise estimate reads.

        calculateSumOfPowerSpectraAndAverageImage (ml_optimiser.cpp:3063-3372) visits particles in
        ``units`` order and skips a particle whose optics group already counts
        ``minimum_nr_particles_sigma2_noise`` (10 for subtomograms, :2813). The count goes up once per
        *image* (:3350-3351), so a group's first particle contributes all its tilt images and fills it;
        the loop stops after the particle that fills the last group. Each image is counted once in the
        group's ``sumw`` (the per-image average of setSigmaNoiseEstimatesAndSetAverageImage).
        """

        units = np.asarray(units, dtype=np.int64).reshape(-1)
        unit_groups = np.asarray(unit_groups, dtype=np.int64).reshape(-1)
        all_groups = set(unit_groups.tolist())
        done = {group: 0 for group in all_groups}
        for unit, group in zip(units.tolist(), unit_groups.tolist()):
            if done[group] >= int(minimum_nr_particles):
                continue
            rows = self.image_rows[self.unit_image_offsets[unit] : self.unit_image_offsets[unit + 1]]
            for batch_images, _particles, _local in self.images.image_source.iter_batches(
                batch_size=int(rows.size), batch_mode="images", subset_indices=rows
            ):
                for image in np.asarray(batch_images):
                    yield group, image
            done[group] += int(rows.size)
            if all(count >= int(minimum_nr_particles) for count in done.values()):
                return


class TomoHalf:
    """One half as subtomogram particles over their tilt images.

    ``n_units`` counts particles; ``images`` is the flat dataset of their tilt images, particle ``u``'s
    at ``unit_image_offsets[u]:[u + 1]`` in ``img_id`` order, with ``image_projections`` (RELION's
    ``Aproj``). ``image_shape``, ``volume_shape`` and ``voxel_size`` are the reference model's.
    """

    def __init__(
        self,
        images,
        *,
        unit_image_offsets,
        image_projections,
        unit_optics_group,
        rows,
        image_shape,
        volume_shape,
        voxel_size,
    ):
        self.images = images
        self.unit_image_offsets = np.asarray(unit_image_offsets, dtype=np.int64)
        self.image_projections = np.asarray(image_projections, dtype=np.float64)
        self.unit_optics_group = np.asarray(unit_optics_group, dtype=np.int64)
        self.image_shape = tuple(int(size) for size in image_shape)
        self.volume_shape = tuple(int(size) for size in volume_shape)
        self.voxel_size = float(voxel_size)
        self.grid_size = self.image_shape[0]
        self.n_units = int(self.unit_image_offsets.size - 1)
        self.n_images = int(self.unit_image_offsets[-1])
        if int(images.n_units) != self.n_images or self.image_projections.shape != (self.n_images, 3, 3):
            raise ValueError("a tomo half needs one dataset image and one Aproj per tilt image")
        # Particle-STAR row of each unit, as a loaded dataset's index layout reports it.
        self._index_layout = _RowLayout(np.asarray(rows, dtype=np.int64))

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


def tilt_pass_inputs(
    half: TomoHalf,
    *,
    fine_px,
    fine_parent,
    unit_coarse_prior,
    old_offsets_px,
    pixel_size: float,
    fine_source_eulers,
    fine_rotations,
    unit_groups,
):
    """The resident pass's :class:`TiltPassInputs` for one half and one sampling."""

    from relax.sparse_pass2.resident_tilts import TiltPassInputs

    image_particle = half.image_particle()
    rounded_old = tomo_particles.relion_gpu_old_offsets(old_offsets_px)
    # sigma2_offset sums: pixel_size^2 |rounded old + trial shift - prior|^2, prior zero in auto-refine
    # (acc_ml_optimiser_impl.h:3999-4035, :4144).
    shifted = rounded_old[:, None, :] + np.asarray(fine_px, dtype=np.float64)[None, :, :]
    return TiltPassInputs(
        unit_image_offsets=half.unit_image_offsets,
        image_left=half.image_projections,
        image_angles=tomo_particles.tilt_translation_angles(
            fine_px, rounded_old, half.image_projections, image_particle, half.grid_size
        ),
        image_noise_scale=tomo_particles.image_noise_scale(image_particle, half.n_units).astype(np.float32),
        unit_translation_prior=np.asarray(np.asarray(unit_coarse_prior)[:, fine_parent], dtype=np.float32),
        unit_translation_sqdist_ang=float(pixel_size) ** 2 * np.sum(shifted * shifted, axis=-1),
        unit_optics_groups=np.asarray(unit_groups, dtype=np.int32),
        fine_source_eulers=fine_source_eulers,
        fine_rotations=fine_rotations,
        slot_capacity=int(np.max(np.diff(half.unit_image_offsets))) if half.n_units else 1,
    )


@dataclasses.dataclass(frozen=True)
class TomoScoreResult:
    """One tomo half's E- and M-step, per particle (the units of the half)."""

    pass2: object  # SparsePass2Output of compute_tilt_pass2_stats_resident: Ft, per-particle poses and stats
    coarse_hard_assignment: np.ndarray  # int32 [P], coarse rotation * T_coarse + coarse translation
    best_translations_px: np.ndarray  # [P, 3] rounded old offset + the winning trial shift (RELION's new offset)
    significant_counts: np.ndarray  # int32 [P], coarse significant samples
    coarse_max_posterior: np.ndarray  # [P]


def score_tomo_half(
    half: TomoHalf,
    *,
    volume,
    noise_variance,
    relion_projector_half,
    relion_projector_r_max: int,
    sampling: TomoSampling,
    rotation_log_prior,
    old_offsets_px,
    sigma_offset_angst: float,
    adaptive_fraction: float,
    max_significants,
    unit_groups,
    padding_factor: int,
    scale_corrections=None,
    group_ids=None,
    scale_correction_group_count=None,
    scale_correction_data_vs_prior=None,
    reconstruction_current_size=None,
    rotation_index_order: str = "recovar",
    local_rotations=None,
) -> TomoScoreResult:
    """RELION's adaptive two-pass E-step and M-step of subtomogram particles (global search).

    Pass 1 scores every particle's tilt images on the coarse grid and cuts the particle's summed
    posterior (:func:`relax.scoring.tomo_coarse.particle_coarse_supports`); pass 2 scores the
    significant samples' oversampled children and backprojects every tilt image
    (``resident_pass2.compute_tilt_pass2_stats_resident``). ``old_offsets_px`` are the particles'
    previous 3D offsets (unrounded, pixels); ``noise_variance`` is one spectrum or ``[G, N^2]`` rows
    with ``unit_groups`` the dense optics group of each particle. The flags are the production K=1
    ones (the one-iteration RELION-pinned replay, em_work/cryoet_s42_20260925/tomo_replay_it1.py).
    The coarse grid is RELION's HEALPix grid at ``sampling.healpix_order`` in ``rotation_index_order``
    (the loop's order, so ``rotation_log_prior`` and the returned rotation sums share it).

    A local search passes ``local_rotations`` (:func:`tomo_local_rotations`): the coarse grid is then
    the union of the particles' local rotations at ``sampling.healpix_order`` and each particle is
    scored over its own with its own prior (``rotation_log_prior`` must be None); the returned rotation
    sums are over that union.
    """

    import jax.numpy as jnp
    from recovar.reconstruction import noise as recon_noise

    from relax import sampling as relax_sampling
    from relax.helpers.projection import relion_projector_half_to_texture_full
    from relax.scoring import tomo_coarse
    from relax.sparse_pass2.resident_pass2 import compute_tilt_pass2_stats_resident

    pixel = float(half.voxel_size)
    size = int(half.grid_size)
    unit_groups = np.asarray(unit_groups, dtype=np.int32)
    image_groups = np.repeat(unit_groups, np.diff(half.unit_image_offsets))
    old_offsets_px = np.asarray(old_offsets_px, dtype=np.float64).reshape(half.n_units, 3)
    coarse_angst, coarse_px, fine_px, fine_parent = tomo_translation_grids(sampling, pixel)
    if local_rotations is not None and rotation_log_prior is not None:
        raise ValueError("a local search's priors are the particles' own")
    coarse_ids = (
        np.arange(int(relax_sampling.rotation_grid_size(sampling.healpix_order)))
        if local_rotations is None
        else local_rotations.coarse_rotation_ids
    )
    n_rot = int(coarse_ids.size)
    coarse_eulers_deg = relax_sampling.rotation_indices_to_relion_eulers(
        coarse_ids, sampling.healpix_order, rotation_index_order=rotation_index_order
    )
    fine_rot, rot_parent, fine_mstep, fine_eulers = relax_sampling.get_oversampled_rotation_grid_from_samples(
        coarse_ids,
        sampling.healpix_order,
        oversampling_order=sampling.oversampling_order,
        random_perturbation=sampling.random_perturbation,
        return_mstep_rotations=True,
        return_source_eulers=True,
        rotation_index_order=rotation_index_order,
        dtype=np.float32,
    )
    unit_coarse_prior = tomo_particles.relion_offset_log_prior_3d(
        coarse_angst, old_offsets_px, pixel_size=pixel, sigma_offset_angst=sigma_offset_angst
    )
    noise = jnp.asarray(noise_variance)
    noise_half = recon_noise.to_batched_half_pixel_noise(noise, (size, size))
    if noise_half.ndim == 2 and noise_half.shape[0] == 1:
        noise_half = noise_half[0]
    layout = tomo_coarse.coarse_score_layout(
        (size, size), sampling.coarse_size, half_spectrum_scoring=True, square_window=False
    )
    image_scale = None if scale_corrections is None else np.repeat(
        np.asarray(scale_corrections, dtype=np.float32), np.diff(half.unit_image_offsets)
    )
    supports, coarse_pmax = tomo_coarse.particle_coarse_supports(
        half.images,
        unit_image_offsets=half.unit_image_offsets,
        image_projections=half.image_projections,
        unit_old_offsets_px=old_offsets_px,
        coarse_eulers_deg=coarse_eulers_deg,
        random_perturbation=sampling.random_perturbation,
        angular_sampling_deg=relax_sampling.relion_angular_sampling_deg(sampling.healpix_order),
        coarse_translations_px=coarse_px,
        projector_full=relion_projector_half_to_texture_full(jnp.asarray(relion_projector_half)).astype(jnp.complex64),
        layout=layout,
        noise_variance_half=noise_half,
        rotation_log_prior=rotation_log_prior,
        unit_translation_log_prior=unit_coarse_prior,
        adaptive_fraction=adaptive_fraction,
        max_significants=max_significants,
        model_max_r=int(relion_projector_r_max),
        padding_factor=int(padding_factor),
        image_size=size,
        optics_group_ids=image_groups,
        scale_corrections=image_scale,
        **(
            {}
            if local_rotations is None
            else {
                "unit_rotation_ids": local_rotations.unit_rotation_ids,
                "unit_rotation_log_priors": local_rotations.unit_rotation_log_priors,
            }
        ),
    )
    tilt = tilt_pass_inputs(
        half,
        fine_px=fine_px,
        fine_parent=fine_parent,
        unit_coarse_prior=unit_coarse_prior,
        old_offsets_px=old_offsets_px,
        pixel_size=pixel,
        fine_source_eulers=fine_eulers,
        fine_rotations=fine_rot,
        unit_groups=unit_groups,
    )
    image_group_ids = None if group_ids is None else np.repeat(
        np.asarray(group_ids, dtype=np.int32), np.diff(half.unit_image_offsets)
    )
    pass2 = compute_tilt_pass2_stats_resident(
        experiment_dataset=half.images,
        volume=volume,
        noise_variance=noise,
        translations=coarse_px,
        significant_sample_indices=supports,
        nside_level=sampling.healpix_order,
        disc_type="linear_interp",
        oversampling_order=sampling.oversampling_order,
        current_size=sampling.fine_size,
        reconstruction_current_size=(
            sampling.fine_size if reconstruction_current_size is None else int(reconstruction_current_size)
        ),
        translation_step=None,
        rotation_log_prior=rotation_log_prior,
        score_with_masked_images=True,
        return_stats=True,
        translation_log_prior=None,
        accumulate_noise=True,
        half_spectrum_scoring=True,
        projection_padding_factor=int(padding_factor),
        reconstruction_padding_factor=int(padding_factor),
        image_corrections=None,  # RELION neither normalises nor pre-shifts a tilt image (acc :872-919)
        scale_corrections=image_scale,
        image_pre_shifts=None,
        use_float64_scoring=False,
        do_gridding_correction=True,
        random_perturbation=sampling.random_perturbation,
        group_ids=image_group_ids,
        scale_correction_group_count=scale_correction_group_count,
        scale_correction_data_vs_prior=scale_correction_data_vs_prior,
        normalization_score_mode="gaussian",
        return_score_log_z=True,
        return_source_eulers=True,
        fine_source_eulers_override=fine_eulers,
        fine_rotations_override=fine_rot,
        fine_mstep_rotations_override=fine_mstep,
        fine_rotation_parent_override=rot_parent,
        fine_translations_override=fine_px,
        fine_translation_parent_override=fine_parent,
        relion_x_half_mstep=True,
        relion_fine_mstep_prune=True,
        relion_exact_fine_gaussian=True,
        relion_fine_diff2_fused_ffi=True,
        relion_f32_fine_posterior=True,
        relion_exact_fine_normalized_cc=True,
        relion_projector_half=np.asarray(relion_projector_half),
        relion_projector_r_max=int(relion_projector_r_max),
        adaptive_fraction=adaptive_fraction,
        include_unweighted_norm_high_shell=True,
        preserve_bpref_particle_order=True,
        source_faithful_spectrum_norm=True,
        optics_group_ids=image_groups if np.asarray(noise_variance).ndim == 2 else None,
        tilt=tilt,
        **(
            {}
            if local_rotations is None
            else {
                "coarse_rotation_ids": coarse_ids,
                "unit_rotation_log_prior": local_rotations.support_priors(supports, coarse_px.shape[0]),
            }
        ),
    )
    hard = np.asarray(pass2.hard_assignment, dtype=np.int64)
    n_fine_trans = int(fine_px.shape[0])
    coarse_hard = rot_parent[hard // n_fine_trans] * int(coarse_px.shape[0]) + fine_parent[hard % n_fine_trans]
    return TomoScoreResult(
        pass2=pass2,
        coarse_hard_assignment=coarse_hard.astype(np.int32),
        best_translations_px=np.asarray(pass2.best_translations, dtype=np.float64)
        + tomo_particles.relion_gpu_old_offsets(old_offsets_px),
        significant_counts=np.asarray([s.size for s in supports], dtype=np.int32),
        coarse_max_posterior=np.asarray(coarse_pmax, dtype=np.float64),
    )


def score_tomo_half_in_loop(
    half: TomoHalf,
    *,
    use_local: bool,
    use_adaptive: bool,
    volume,
    noise_variance,
    relion_projector_half,
    relion_projector_r_max,
    sampling: TomoSampling,
    rotation_log_prior,
    previous_translations,
    sigma_offset_angst: float,
    max_significants,
    unit_groups,
    scale_corrections,
    group_ids,
    scale_correction_group_count,
    scale_correction_data_vs_prior,
    reconstruction_current_size,
    outputs,
    k: int,
    local_search=None,
):
    """The refinement loop's E+M step for a tomo half: :func:`score_tomo_half` as a ``HalfScoreResult``.

    Per-unit fields are the particles'. The best translations are the winning trial shifts in pixels
    (3D); the loop adds the rounded previous offset, as RELION writes ``old + shift``. The explicit
    best poses also go to ``outputs`` (the loop's pose update reads them there).

    A local-search iteration passes ``local_search``: a dict with the particles' previous angles
    (``previous_eulers_deg``), ``sigma_rot`` and ``sigma_psi`` and the pass-1 HEALPix order
    (``parent_order``); ``sampling`` then carries that order and the local grid's perturbation.
    """

    from relax.dense.score_outputs import HalfScoreResult
    from relax.dense.scoring_policy import PADDING_FACTOR, PROJECTION_PADDING_FACTOR
    from relax.helpers.half_volume_mstep import relion_backprojector_volume_shape
    from relax.sampling import rotation_grid_size

    if bool(use_local) != (local_search is not None):
        raise ValueError("a local-search iteration needs its local-search inputs, and only it")
    if not (use_adaptive or use_local) or int(sampling.oversampling_order) < 1:
        raise NotImplementedError("subtomogram particles run RELION's adaptive two-pass E-step only")
    if PADDING_FACTOR != PROJECTION_PADDING_FACTOR:
        raise ValueError("the tomo half pass projects and backprojects with one padding factor")
    if relion_projector_half is None or relion_projector_r_max is None:
        raise ValueError("the tomo half pass needs RELION's Projector::data half map")
    relion_projector_half = np.asarray(relion_projector_half)
    if relion_projector_half.ndim == 4:
        # The loop keeps a class axis; K = 1 here.
        if relion_projector_half.shape[0] != 1:
            raise ValueError("the tomo half pass refines one class")
        relion_projector_half = relion_projector_half[0]
    relion_projector_r_max = np.asarray(relion_projector_r_max).reshape(-1)[0]
    if np.ndim(volume) == 2:
        if np.shape(volume)[0] != 1:
            raise ValueError("the tomo half pass refines one class")
        volume = volume[0]
    local_rotations = None
    if local_search is not None:
        local_rotations = tomo_local_rotations(
            local_search["previous_eulers_deg"],
            sigma_rot=local_search["sigma_rot"],
            sigma_psi=local_search["sigma_psi"],
            healpix_order=int(sampling.healpix_order),
            random_perturbation=float(sampling.random_perturbation),
            voxel_size=half.voxel_size,
        )
        prior = None
    else:
        n_rot = int(rotation_grid_size(sampling.healpix_order))
        prior = (
            np.zeros(n_rot, dtype=np.float32)
            if rotation_log_prior is None
            else np.asarray(rotation_log_prior, dtype=np.float32).reshape(-1)
        )
        if prior.shape != (n_rot,):
            raise ValueError(f"a global tomo pass needs one rotation log prior per coarse rotation, got {prior.shape}")
    old = (
        np.zeros((half.n_units, 3), dtype=np.float64)
        if previous_translations is None
        else np.asarray(previous_translations, dtype=np.float64)
    )
    if old.shape != (half.n_units, 3):
        raise ValueError(f"tomo particles carry 3D offsets, got previous translations of shape {old.shape}")
    groups = np.zeros(half.n_units, dtype=np.int32) if unit_groups is None else np.asarray(unit_groups, dtype=np.int32)
    result = score_tomo_half(
        half,
        volume=volume,
        noise_variance=noise_variance,
        relion_projector_half=relion_projector_half,
        relion_projector_r_max=int(relion_projector_r_max),
        sampling=sampling,
        rotation_log_prior=prior,
        old_offsets_px=old,
        sigma_offset_angst=sigma_offset_angst,
        adaptive_fraction=0.999,
        max_significants=max_significants,
        unit_groups=groups,
        padding_factor=PROJECTION_PADDING_FACTOR,
        scale_corrections=scale_corrections,
        group_ids=group_ids,
        scale_correction_group_count=scale_correction_group_count,
        scale_correction_data_vs_prior=scale_correction_data_vs_prior,
        reconstruction_current_size=reconstruction_current_size,
        local_rotations=local_rotations,
    )
    pass2 = result.pass2
    outputs.best_pose_rotations[k] = np.asarray(pass2.best_rotations, dtype=np.float32)
    outputs.best_pose_rotation_eulers[k] = np.asarray(pass2.source_eulers, dtype=np.float64)
    outputs.best_pose_translations[k] = np.asarray(pass2.best_translations, dtype=np.float32)
    mstep_size = sampling.fine_size if reconstruction_current_size is None else int(reconstruction_current_size)
    return HalfScoreResult(
        ha=np.asarray(pass2.hard_assignment, dtype=np.int32),
        Ft_y=pass2.Ft_y,
        Ft_ctf=pass2.Ft_ctf,
        em_stats=pass2.relion_stats,
        noise_stats=pass2.noise_stats,
        best_pose_rotations=outputs.best_pose_rotations[k],
        best_pose_rotation_eulers=outputs.best_pose_rotation_eulers[k],
        best_pose_translations=outputs.best_pose_translations[k],
        coarse_ha=result.coarse_hard_assignment,
        significant_counts=result.significant_counts,
        mstep_full_half_axis=0,
        mstep_accumulator_shape=relion_backprojector_volume_shape(
            half.volume_shape, PADDING_FACTOR, current_size=mstep_size
        ),
    )


def tilt_image_accuracy_inputs(half: TomoHalf) -> dict:
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

    return {
        "image_offsets": half.unit_image_offsets,
        "image_projections": half.image_projections,
        "image_ctf": image_ctf,
        "voltage": optics_column("rlnVoltage"),
        "spherical_aberration": optics_column("rlnSphericalAberration"),
        "amplitude_contrast": optics_column("rlnAmplitudeContrast"),
    }



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
    previous_eulers_deg, *, sigma_rot, sigma_psi, healpix_order: int, random_perturbation: float, voxel_size: float
) -> TomoLocalRotations:
    """Every particle's local orientations and log priors around its previous pose (subtomogram local search).

    The single-particle local search's neighbourhoods (relax.local.local_layout.build_local_hypothesis_layout,
    RELION's selectOrientationsWithNonZeroPriorProbability) with the particle's subtomogram-frame angles;
    only the rotation part is used (the 3D offset prior is the particle's own, score_tomo_half).
    """

    from relax.local.local_layout import build_local_hypothesis_layout
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
        grid_metadata=build_local_search_grid_metadata(int(healpix_order)),
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
