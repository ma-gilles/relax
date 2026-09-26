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

    def startup_noise_images(self, units, *, unit_groups, particles_per_group: int = 10):
        """``(group, real-space image)`` of the tilt images RELION's start-up noise estimate reads.

        calculateSumOfPowerSpectraAndAverageImage (ml_optimiser.cpp:3100-3300) visits particles in
        ``units`` order, skips a particle whose optics group already has ``minimum_nr_particles_sigma2_noise``
        particles (10 for subtomograms, :2813) and adds every tilt image of the others, each counted
        once in the group's ``sumw`` (the per-image average of setSigmaNoiseEstimatesAndSetAverageImage).
        """

        units = np.asarray(units, dtype=np.int64).reshape(-1)
        unit_groups = np.asarray(unit_groups, dtype=np.int64).reshape(-1)
        taken = {}
        chosen = []
        for unit, group in zip(units.tolist(), unit_groups.tolist()):
            if taken.get(group, 0) >= int(particles_per_group):
                continue
            taken[group] = taken.get(group, 0) + 1
            chosen.append((unit, group))
        for unit, group in chosen:
            rows = self.image_rows[self.unit_image_offsets[unit] : self.unit_image_offsets[unit + 1]]
            for batch_images, _particles, _local in self.images.image_source.iter_batches(
                batch_size=int(rows.size), batch_mode="images", subset_indices=rows
            ):
                for image in np.asarray(batch_images):
                    yield group, image


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

    pass2: object  # SparsePass2Output of compute_pass2_stats_resident: Ft, per-particle poses and stats
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
    coarse_eulers_deg,
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
) -> TomoScoreResult:
    """RELION's adaptive two-pass E-step and M-step of subtomogram particles (global search).

    Pass 1 scores every particle's tilt images on the coarse grid and cuts the particle's summed
    posterior (:func:`relax.scoring.tomo_coarse.particle_coarse_supports`); pass 2 scores the
    significant samples' oversampled children and backprojects every tilt image
    (``resident_pass2.compute_pass2_stats_resident(tilt=...)``). ``old_offsets_px`` are the particles'
    previous 3D offsets (unrounded, pixels); ``noise_variance`` is one spectrum or ``[G, N^2]`` rows
    with ``unit_groups`` the dense optics group of each particle. The flags are the production K=1
    ones (the one-iteration RELION-pinned replay, em_work/cryoet_s42_20260925/tomo_replay_it1.py).
    """

    import jax.numpy as jnp
    from recovar.reconstruction import noise as recon_noise

    from relax import sampling as relax_sampling
    from relax.helpers.projection import relion_projector_half_to_texture_full
    from relax.scoring import tomo_coarse
    from relax.sparse_pass2.resident_pass2 import compute_pass2_stats_resident

    pixel = float(half.voxel_size)
    size = int(half.grid_size)
    unit_groups = np.asarray(unit_groups, dtype=np.int32)
    image_groups = np.repeat(unit_groups, np.diff(half.unit_image_offsets))
    old_offsets_px = np.asarray(old_offsets_px, dtype=np.float64).reshape(half.n_units, 3)
    coarse_angst, coarse_px, fine_px, fine_parent = tomo_translation_grids(sampling, pixel)
    n_rot = int(np.asarray(coarse_eulers_deg).shape[0])
    fine_rot, rot_parent, fine_mstep, fine_eulers = relax_sampling.get_oversampled_rotation_grid_from_samples(
        np.arange(n_rot),
        sampling.healpix_order,
        oversampling_order=sampling.oversampling_order,
        random_perturbation=sampling.random_perturbation,
        return_mstep_rotations=True,
        return_source_eulers=True,
        rotation_index_order="relion",
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
    pass2 = compute_pass2_stats_resident(
        half.images,
        volume,
        noise,
        coarse_px,
        supports,
        sampling.healpix_order,
        "linear_interp",
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
