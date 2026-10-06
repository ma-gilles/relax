"""Denovo Iref seeding (RELION ``--pad 1`` parity).

``compute_bootstrap_iref`` and ``postprocess_bootstrap_iref`` run relax's port of
``calculateSumOfPowerSpectraAndAverageImage`` (ml_optimiser.cpp:3127-3205), the
reconstruct and ``initialLowPassFilterReferences`` (:mod:`relax.vdam.bootstrap_reconstruction`).
Parity target: ``run_it000_class001.mrc`` (|CC|>0.998).
"""

from __future__ import annotations

import os

import numpy as np

from relax.relion import initial_model_io
from relax.relion.initial_model_io import _experiment_read_order
from relax.relion.initial_noise import _image_sigma2_iter, compute_avg_unaligned_and_sigma2, relion_startup_positions
from relax.vdam import bootstrap_reconstruction, output
from relax.vdam.init import initialise_data_vs_prior_from_references, initialise_denovo_state, seed_noise_from_mavg
from relax.vdam.native_options import NativeInitialModelOptions
from relax.vdam.native_sampling import _n_directions_for_healpix_order
from relax.vdam.state import InitialModelState


def compute_bootstrap_iref(
    *,
    images: np.ndarray,
    defU: np.ndarray,
    defV: np.ndarray,
    defAngle: np.ndarray,
    phase_shift: np.ndarray,
    voltage,
    Cs,
    Q0,
    pixel_size: float,
    ori_size: int,
    nr_classes: int,
    particle_diameter_ang: float,
    width_mask_edge_px: float,
    do_zero_mask: bool,
    do_ctf_correction: bool,
    random_seed: int,
    padding_factor: int = 1,
    current_size: int = -1,
    minimum_nr_particles: int = 1000,
    particle_seed_ids: np.ndarray | None = None,
    particle_positions: np.ndarray | None = None,
    image_gamma_offsets=None,
):
    """RELION's random-orientation bootstrap reference per class, in RECOVAR's frame.

    ``voltage``, ``Cs`` and ``Q0`` are scalars or one value per image (its optics group's).
    ``particle_positions`` are the images' particles' positions in RELION's order (default
    ``0..n-1``): each orientation's seed and its class, position % K
    (:func:`relax.vdam.bootstrap_reconstruction.bootstrap_references`).
    ``image_gamma_offsets`` is ``(image_group [n], {group: even Zernike gamma offset or None})``
    for optics groups with even aberrations (:func:`_bootstrap_gamma_offsets`); CTF-premultiplied
    images are multiplied by the CTF once more, as RELION's start-up does (ml_optimiser.cpp:3337-3341).
    Returns ``(Iref, rand_state)``: the C ``rand()`` stream where the particle loop
    leaves it, which :func:`postprocess_bootstrap_iref` continues for the blobs.
    """
    from recovar.utils.helpers import relion_volume_to_recovar

    from relax.relion.relion_ctf import relion_ctf_fftw_half

    if current_size <= 0:
        # RELION wsum_model.current_size = ROUND(0.07 * ori_size) (shell count, not A).
        current_size = int(np.floor(0.07 * ori_size + 0.5))
    todo = min(max(int(minimum_nr_particles), int(nr_classes) * 5), int(images.shape[0]))
    ctf_images = None
    if do_ctf_correction:
        ones = np.ones(todo)
        params = np.column_stack(
            [
                np.asarray(defU, dtype=np.float64)[:todo],
                np.asarray(defV, dtype=np.float64)[:todo],
                np.asarray(defAngle, dtype=np.float64)[:todo],
                np.broadcast_to(np.asarray(voltage, dtype=np.float64), (images.shape[0],))[:todo],
                np.broadcast_to(np.asarray(Cs, dtype=np.float64), (images.shape[0],))[:todo],
                np.broadcast_to(np.asarray(Q0, dtype=np.float64), (images.shape[0],))[:todo],
                0.0 * ones,
                ones,
                np.asarray(phase_shift, dtype=np.float64)[:todo],
            ]
        )
        if image_gamma_offsets is None:
            ctf_images = relion_ctf_fftw_half(params, int(ori_size), float(pixel_size))
        else:
            # Each optics group's even Zernike phase (ObservationModel::getGammaOffset), as
            # RELION's start-up CTF::getFftwImage applies it (ml_optimiser.cpp:3306-3307).
            image_group, gamma_by_group = image_gamma_offsets
            image_group = np.asarray(image_group)[:todo]
            ctf_images = np.empty((todo, int(ori_size), int(ori_size) // 2 + 1), dtype=np.float64)
            for group, gamma in gamma_by_group.items():
                rows = image_group == group
                ctf_images[rows] = relion_ctf_fftw_half(
                    params[rows], int(ori_size), float(pixel_size), gamma_offset=gamma
                )
    iref_relion, rand_state = bootstrap_reconstruction.bootstrap_references(
        images=np.asarray(images[:todo], dtype=np.float64),
        ctf_images=ctf_images,
        ori_size=int(ori_size),
        pixel_size=float(pixel_size),
        nr_classes=int(nr_classes),
        particle_diameter_ang=float(particle_diameter_ang),
        width_mask_edge_px=float(width_mask_edge_px),
        do_zero_mask=bool(do_zero_mask),
        random_seed=int(random_seed),
        padding_factor=int(padding_factor),
        minimum_nr_particles=int(minimum_nr_particles),
        particle_seed_ids=particle_seed_ids,
        particle_positions=particle_positions,
        current_size=int(current_size),
    )
    return np.asarray([relion_volume_to_recovar(vol) for vol in iref_relion], dtype=np.float64), rand_state


def _bootstrap_gamma_offsets(dataset, image_indices, ori_size: int):
    """``(image_group, {group: gamma})`` for the bootstrap CTF, or None without even Zernike terms.

    The groups' even Zernike gamma offsets on the model grid, as relax's exact CTF rows
    use them (:func:`relax.relion.relion_ctf._optics_group_ctf_geometry`). Magnification is
    refused upstream for InitialModel.
    """

    from recovar.data_io.starfile import star_column

    from relax.relion import relion_ctf
    from relax.relion.optics_aberrations import dataset_needs_exact_ctf

    if not dataset_needs_exact_ctf(dataset):
        return None
    _, cache = relion_ctf._exact_ctf_source_cache(dataset, (int(ori_size), int(ori_size)))
    original = np.asarray(relion_ctf.original_image_indices(dataset, np.asarray(image_indices, dtype=np.int64)))
    groups = np.asarray(star_column(cache["particles"], "rlnOpticsGroup", required=True), dtype=np.int64)[original]
    gamma_by_group = {}
    for group in np.unique(groups):
        gamma, mag = relion_ctf._optics_group_ctf_geometry(cache, int(group), int(ori_size))
        if mag is not None:
            raise NotImplementedError("InitialModel does not implement anisotropic magnification")
        gamma_by_group[int(group)] = gamma
    if all(gamma is None for gamma in gamma_by_group.values()):
        return None
    return groups, gamma_by_group


def postprocess_bootstrap_iref(
    Iref: np.ndarray,
    *,
    rand_state,
    pixel_size: float,
    ini_high_ang: float,
    particle_diameter_ang: float,
    width_mask_edge_px: float,
    do_init_blobs: bool = True,
    is_helical_segment: bool = False,
) -> np.ndarray:
    """RELION's post-bootstrap low-pass, blobs and soft mask (ml_optimiser.cpp:2940-2980).

    ``rand_state`` is the second value :func:`compute_bootstrap_iref` returns.
    """
    from recovar.utils.helpers import recovar_volume_to_relion, relion_volume_to_recovar

    if is_helical_segment:
        raise NotImplementedError("helical InitialModel blobs are not ported")
    arr = np.asarray(Iref, dtype=np.float64)
    if arr.ndim != 4 or arr.shape[1] != arr.shape[2] or arr.shape[2] != arr.shape[3]:
        raise ValueError(f"Iref must have shape (K, N, N, N), got {arr.shape}")
    post_relion = bootstrap_reconstruction.postprocess_references(
        np.asarray([recovar_volume_to_relion(vol) for vol in arr], dtype=np.float64),
        generator=rand_state,
        pixel_size=float(pixel_size),
        ini_high_ang=float(ini_high_ang),
        particle_diameter_ang=float(particle_diameter_ang),
        width_mask_edge_px=float(width_mask_edge_px),
        do_init_blobs=bool(do_init_blobs),
    )
    return np.asarray([relion_volume_to_recovar(vol) for vol in post_relion], dtype=np.float64)


def _group_pixel_sizes(dataset, optics_group_by_particle) -> np.ndarray | None:
    """Each optics group's image pixel size for a dataset on several image shapes, else None."""

    if getattr(dataset, "datasets", None) is None:
        return None
    sizes = np.zeros(int(np.max(optics_group_by_particle)) + 1)
    for class_dataset, rows in zip(dataset.datasets, dataset.rows):
        sizes[np.asarray(optics_group_by_particle)[rows]] = float(class_dataset.voxel_size)
    return sizes


def _model_grid_startup_images(dataset, rows, pixel_sizes, opts: NativeInitialModelOptions, model_pixel_size: float):
    """The start-up images of a dataset on several shapes, as RELION's start-up loop leaves them.

    Each image is soft-masked with its own pixel size, resized to the model pixel size and
    windowed to the model box (ml_optimiser.cpp:2905-2955); the bootstrap transforms these.
    """

    from relax.relion.initial_noise import _rescale_to_model_grid

    ori_size = int(dataset.grid_size)
    out = np.empty((len(rows), ori_size, ori_size), dtype=np.float64)
    for i, (_row, image) in enumerate(dataset.iter_images(rows, batch_size=max(1, int(opts.image_batch_size)))):
        image = np.asarray(image, dtype=np.float64)
        if bool(opts.do_zero_mask):
            radius = float(opts.particle_diameter) / (2.0 * float(pixel_sizes[i]))
            image = bootstrap_reconstruction.soft_mask_outside_map(image, radius, float(opts.width_mask_edge_px))
        out[i] = _rescale_to_model_grid(image, float(pixel_sizes[i]), float(model_pixel_size), ori_size)
    return out


def _load_raw_images(dataset, image_indices: np.ndarray, *, batch_size: int) -> np.ndarray:
    """Load raw real-space particle images through ``CryoEMDataset`` I/O."""

    images: list[np.ndarray] = []
    for batch_images, _particle_indices, _local_indices in dataset.image_source.iter_batches(
        batch_size=batch_size,
        batch_mode="images",
        subset_indices=np.asarray(image_indices, dtype=np.int64),
    ):
        images.append(np.asarray(batch_images))
    if not images:
        return np.empty((0, dataset.grid_size, dataset.grid_size), dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(images, axis=0))


def _initial_state_from_particles(
    dataset,
    main_star,
    optics_star,
    opts: NativeInitialModelOptions,
) -> tuple[InitialModelState, np.ndarray]:
    profile = output._StageProfile(opts.environment.profile)

    ori_size = int(dataset.grid_size)
    pixel_size = float(dataset.voxel_size)
    order = _experiment_read_order(main_star)
    optics_group_by_particle = initial_model_io._optics_group_indices(main_star)
    nr_optics_groups = int(np.unique(optics_group_by_particle).size)
    ordered_groups = optics_group_by_particle[order]
    ones = np.ones(order.size, dtype=np.int64)
    profile.record("setup")

    # The start-up loop's particles (relion_startup_positions): each optics group's first
    # quota of particles in RELION's order, for the noise spectra and for the bootstrap.
    noise_order = order[relion_startup_positions(ordered_groups, ones, int(opts.sigma2_min_particles))]
    group_pixel_sizes = _group_pixel_sizes(dataset, optics_group_by_particle)
    batch_size = max(1, int(opts.image_batch_size))
    Mavg, sigma2_per_group = compute_avg_unaligned_and_sigma2(
        (
            _image_sigma2_iter(dataset, noise_order, optics_group_by_particle, batch_size=batch_size)
            if group_pixel_sizes is None
            else (
                (int(optics_group_by_particle[row]), image)
                for row, image in dataset.iter_images(noise_order, batch_size=batch_size)
            )
        ),
        ori_size=ori_size,
        pixel_size=pixel_size,
        particle_diameter_ang=float(opts.particle_diameter),
        width_mask_edge_px=int(opts.width_mask_edge_px),
        do_zero_mask=bool(opts.do_zero_mask),
        nr_optics_groups=nr_optics_groups,
        minimum_nr_particles=int(opts.sigma2_min_particles),
        **({} if group_pixel_sizes is None else {"group_pixel_sizes": group_pixel_sizes, "model_pixel_size": pixel_size}),
    )
    profile.record("average_unaligned")

    # A particle's position in RELION's order is its part_id: its random orientation's seed and
    # its class (part_id % K) in the bootstrap (ml_optimiser.cpp:3254-3270).
    bootstrap_positions = relion_startup_positions(ordered_groups, ones, int(opts.bootstrap_min_particles))
    bootstrap_order = order[bootstrap_positions]
    if group_pixel_sizes is None:
        images = _load_raw_images(dataset, bootstrap_order, batch_size=batch_size)
    else:
        images = _model_grid_startup_images(
            dataset, bootstrap_order, group_pixel_sizes[optics_group_by_particle[bootstrap_order]], opts, pixel_size
        )
    profile.record("raw_images")
    sorted_star = main_star.iloc[bootstrap_order]
    voltage, Cs, Q0, pixel_size = initial_model_io._particle_optics(sorted_star, optics_star, dataset)
    profile.record("optics_metadata")

    # RELAX_INITIAL_IREF_OVERRIDE (parity hook): RELION's iter000 ref replaces the bootstrap below.
    override_path = os.environ.get("RELAX_INITIAL_IREF_OVERRIDE")
    bootstrap_kwargs = dict(
        images=images,
        defU=np.asarray(sorted_star["_rlnDefocusU"].astype(float).to_numpy(), dtype=np.float64),
        defV=np.asarray(sorted_star["_rlnDefocusV"].astype(float).to_numpy(), dtype=np.float64),
        defAngle=np.asarray(sorted_star["_rlnDefocusAngle"].astype(float).to_numpy(), dtype=np.float64),
        phase_shift=initial_model_io._phase_shift(sorted_star),
        voltage=voltage,
        Cs=Cs,
        Q0=Q0,
        pixel_size=pixel_size,
        ori_size=ori_size,
        nr_classes=int(opts.nr_classes),
        particle_diameter_ang=float(opts.particle_diameter),
        width_mask_edge_px=float(opts.width_mask_edge_px),
        # Images on several shapes arrive masked and on the model grid.
        do_zero_mask=bool(opts.do_zero_mask) and group_pixel_sizes is None,
        do_ctf_correction=bool(opts.do_ctf_correction),
        random_seed=int(opts.random_seed),
        padding_factor=int(opts.padding_factor),
        current_size=-1,
        minimum_nr_particles=int(bootstrap_positions.size),
        particle_positions=bootstrap_positions,
        image_gamma_offsets=None if group_pixel_sizes is not None else _bootstrap_gamma_offsets(
            dataset, bootstrap_order, ori_size
        ),
    )
    iref, rand_state = (None, None) if override_path else compute_bootstrap_iref(**bootstrap_kwargs)
    profile.record("bootstrap")

    state = initialise_denovo_state(
        ori_size=ori_size,
        pixel_size=pixel_size,
        K=int(opts.nr_classes),
        nr_iter=int(opts.nr_iter),
        n_directions=_n_directions_for_healpix_order(int(opts.healpix_order)),
        nr_optics_groups=nr_optics_groups,
        pseudo_halfsets=True,
        padding_factor=int(opts.padding_factor),
    )
    state = seed_noise_from_mavg(state, sigma2_per_group)
    init_sigma_offset_angstrom = (
        opts.translation_sigma_angstrom if opts.translation_sigma_angstrom is not None else 10.0
    )
    state.sigma2_offset = float(init_sigma_offset_angstrom) ** 2
    state.Mavg = Mavg
    profile.record("state_init")
    if override_path:
        # Parity hook: load Iref directly. Comma-separated paths for K-class,
        # single path broadcast across K, or a "{k}" template expanded k=1..K.
        from recovar.utils.helpers import load_relion_volume

        K = int(opts.nr_classes)
        paths = [p.strip() for p in override_path.split(",") if p.strip()]
        if len(paths) == 1 and "{k" in paths[0]:
            paths = [paths[0].format(k=k + 1) for k in range(K)]
        if len(paths) not in (1, K):
            raise ValueError(f"RELAX_INITIAL_IREF_OVERRIDE expects 1 or K={K} paths, got {len(paths)}")
        vols = np.stack(
            [np.asarray(load_relion_volume(p), dtype=np.float64) for p in paths],
            axis=0,
        )
        if vols.shape[1:] != (ori_size, ori_size, ori_size):
            raise ValueError(f"RELAX_INITIAL_IREF_OVERRIDE volume shape {vols.shape[1:]} != {(ori_size,) * 3}")
        state.Iref = np.broadcast_to(vols, (K, ori_size, ori_size, ori_size)).copy() if len(paths) == 1 else vols
    else:
        state.Iref = postprocess_bootstrap_iref(
            iref,
            rand_state=rand_state,
            pixel_size=pixel_size,
            ini_high_ang=float(state.ini_high),
            particle_diameter_ang=float(opts.particle_diameter),
            width_mask_edge_px=float(opts.width_mask_edge_px),
            do_init_blobs=True,
            is_helical_segment=False,
        )
    profile.record("initial_reference")
    state = initialise_data_vs_prior_from_references(
        state,
        nr_particles=len(main_star),
        fix_tau=False,
    )
    profile.record("data_vs_prior")
    profile.report("initial state")
    return state, optics_group_by_particle


def _initial_state_from_tomo_particles(dataset, particles_table, opts: NativeInitialModelOptions):
    """The de novo start of subtomogram particles (RELION 5 2D stacks): noise, bootstrap and priors.

    ``dataset`` is a :class:`relax.refinement.tomo_half.TomoDataset`; its units are the particles.
    RELION's start-up loop (calculateSumOfPowerSpectraAndAverageImage, ml_optimiser.cpp:2806-3080)
    counts tilt images against ``minimum_nr_particles_sigma2_noise`` (10 for subtomograms, :2574), so
    the first particle of each optics group fills it: its images give the noise spectrum and the
    bootstrap, each image backprojected at ``Aproj R`` (no magnification, :3010-3015) with its own
    dose-damped CTF, which carries its optics group's even Zernike terms and magnified frequencies
    (setValuesByGroup, :3036-3047). The reference is low-passed but has no blobs or soft mask (:2707). The initial
    ``data_vs_prior`` counts particles (ml_model.cpp:1609-1617). Returns ``(state, optics_group_by_particle)``.
    """

    from relax.helpers.expected_accuracy import _trial_ctf_images
    from relax.refinement.tomo_half import tilt_image_accuracy_inputs
    from relax.relion.optics_aberrations import dataset_needs_exact_ctf
    from relax.relion.relion_ctf import relion_fftw_ctf_rows

    ori_size = int(dataset.grid_size)
    pixel_size = float(dataset.voxel_size)
    order = _experiment_read_order(particles_table)
    optics_group_by_particle = initial_model_io._optics_group_indices(particles_table)
    nr_optics_groups = int(np.unique(optics_group_by_particle).size)
    units = dataset.startup_units(order, unit_groups=optics_group_by_particle[order], minimum_nr_particles=10)
    half = dataset.subset(units)
    unit_images = [dataset.unit_images(unit) for unit in units]
    images = np.concatenate(unit_images, axis=0)
    image_groups = np.repeat(optics_group_by_particle[units], [block.shape[0] for block in unit_images])
    Mavg, sigma2_per_group = compute_avg_unaligned_and_sigma2(
        zip(image_groups.tolist(), images),
        ori_size=ori_size,
        pixel_size=pixel_size,
        particle_diameter_ang=float(opts.particle_diameter),
        width_mask_edge_px=int(opts.width_mask_edge_px),
        do_zero_mask=bool(opts.do_zero_mask),
        nr_optics_groups=nr_optics_groups,
        minimum_nr_particles=int(images.shape[0]),
    )

    tilt = tilt_image_accuracy_inputs(half)
    ctf_images = None
    if bool(opts.do_ctf_correction) and dataset_needs_exact_ctf(half.images):
        # Optics-table terms: each tilt's CTF from setValuesByGroup, i.e. with its group's even Zernike
        # phase and magnified frequencies (ml_optimiser.cpp:3036-3047); a premultiplied image keeps the
        # plain CTF here and is multiplied by it once more, as RELION's start-up does.
        ctf_images = relion_fftw_ctf_rows(
            half.images, np.arange(half.n_images), (ori_size, ori_size), square_premultiplied=False
        )
    elif bool(opts.do_ctf_correction):
        ctf_images = _trial_ctf_images(
            np.arange(half.n_units),
            defocus=None,
            optics=None,
            pixel_size=pixel_size,
            image_full_size=ori_size,
            current_image_size=ori_size,
            tilt_images=tilt,
        )
    # Each image's particle as its position in RELION's order (the seed and the class, part_id_sorted % K).
    position = np.empty(order.size, dtype=np.int64)
    position[order] = np.arange(order.size)
    image_particle = np.repeat(position[units], np.diff(half.unit_image_offsets))
    iref_relion, _ = bootstrap_reconstruction.bootstrap_references(
        images=images,
        ctf_images=ctf_images,
        ori_size=ori_size,
        pixel_size=pixel_size,
        nr_classes=int(opts.nr_classes),
        particle_diameter_ang=float(opts.particle_diameter),
        width_mask_edge_px=float(opts.width_mask_edge_px),
        do_zero_mask=bool(opts.do_zero_mask),
        random_seed=int(opts.random_seed),
        padding_factor=int(opts.padding_factor),
        minimum_nr_particles=int(units.size),
        current_size=int(np.floor(0.07 * ori_size + 0.5)),
        image_particle=image_particle,
        image_projections=half.image_projections,
    )
    from recovar.utils.helpers import relion_volume_to_recovar

    iref = np.asarray([relion_volume_to_recovar(vol) for vol in iref_relion], dtype=np.float64)

    state = initialise_denovo_state(
        ori_size=ori_size,
        pixel_size=pixel_size,
        K=int(opts.nr_classes),
        nr_iter=int(opts.nr_iter),
        n_directions=_n_directions_for_healpix_order(int(opts.healpix_order)),
        nr_optics_groups=nr_optics_groups,
        pseudo_halfsets=True,
        padding_factor=int(opts.padding_factor),
    )
    state = seed_noise_from_mavg(state, sigma2_per_group)
    init_sigma_offset_angstrom = opts.translation_sigma_angstrom if opts.translation_sigma_angstrom is not None else 10.0
    state.sigma2_offset = float(init_sigma_offset_angstrom) ** 2
    state.Mavg = Mavg
    state.Iref = postprocess_bootstrap_iref(
        iref,
        rand_state=None,
        pixel_size=pixel_size,
        ini_high_ang=float(state.ini_high),
        particle_diameter_ang=float(opts.particle_diameter),
        width_mask_edge_px=float(opts.width_mask_edge_px),
        do_init_blobs=False,
        is_helical_segment=False,
    )
    state = initialise_data_vs_prior_from_references(state, nr_particles=int(dataset.n_units), fix_tau=False)
    return state, optics_group_by_particle
