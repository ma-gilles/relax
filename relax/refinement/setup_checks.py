"""Set-up checks of a refinement run: what refine_single_volume refuses or resolves from its inputs before the
first iteration (code rule 12: check at the edge, then trust), and the run context they resolve.
"""

import logging
from dataclasses import dataclass

import numpy as np

from relax.helpers import optics_scale
from relax.helpers.expected_accuracy import (
    RELION_DEFAULT_SIGMA2_FUDGE,
    Half1AccuracyInputs,
    prepare_relion_half1_trial_order,
)
from relax.helpers.relion_random import GlibcRand, init_random_generator
from relax.helpers.resolution import ImageGeometry
from relax.reconstruction.regularization_relion import RELION_MINRES_MAP
from relax.refinement.expectation_batches import BatchPlanner
from relax.refinement.half_inputs import configure_half_image_preprocessing
from relax.refinement.iteration_planning import RunOptics
from relax.refinement.iteration_snapshot import SnapshotCapture
from relax.refinement.mean_helpers import ReconstructionSettings
from relax.refinement.optics_shapes import MultiShapeHalf
from relax.refinement.ports import (
    ExpectationProbe,
    MaximizationProbe,
    ObserverExpectationProbe,
    ObserverMaximizationProbe,
    RunObserver,
)
from relax.refinement.refinement_options import (
    OpticsGeometry,
    RefinementOptions,
    RelionConsistencyOptions,
    require_consistency_route,
)
from relax.refinement.tomo_half import TomoHalf
from relax.relion.geometry import (
    PROJECTION_PADDING_FACTOR,
    RECONSTRUCTION_PADDING_FACTOR,
    REFERENCE_FILTER_EDGE_SHELLS,
)

# The set-up logs as part of the controller's run.
logger = logging.getLogger("relax.refinement.iteration_loop")


def _relion_translation_angle_scale(*, model_pixel_size: float, optics_pixel_sizes) -> float:
    """Convert model-pixel trial translations to the shared optics pixel size.

    RELION builds its trial translations in Angstrom with the model pixel (ml_optimiser.cpp:590, 597,
    2592-2593) and converts them with the particle's optics pixel size
    (HealpixSampling::getTranslationsInPixel), for every class count, while RECOVAR's translation grid
    is in model pixels. The scale model_pixel_size / optics_pixel_size is exactly 1.0 when the
    serialized sizes agree; it multiplies only the RELION translation-phase operand.
    """

    if optics_pixel_sizes is None:
        return 1.0
    optics = np.asarray(optics_pixel_sizes, dtype=np.float64).reshape(-1)
    if optics.size == 0 or not np.all(np.isfinite(optics)) or np.any(optics <= 0.0):
        raise ValueError("RELION optics pixel sizes must be non-empty, positive, and finite")
    unique_optics = np.unique(optics)
    if unique_optics.size != 1:
        raise NotImplementedError(
            "exact RELION translation phases on one image shape require one shared optics pixel size; "
            "per-particle optics scaling is not yet implemented"
        )
    # A Python float: the model pixel size may arrive as the input's float32, and the scale is float64 arithmetic.
    return float(model_pixel_size) / float(unique_optics[0])


def checked_optics_group_ids(optics_group_ids_per_half, noise_variance_per_half, experiment_datasets):
    """Each half's per-image optics-group rows, or ``[None, None]`` for one group.

    A half's noise is a flat vector (one optics group) or ``[G, P]`` rows
    (:mod:`relax.helpers.optics_noise`); with rows every image needs its group.
    """

    n_groups = 1 if noise_variance_per_half[0].ndim == 1 else int(noise_variance_per_half[0].shape[0])
    if n_groups == 1:
        return [None, None]
    if optics_group_ids_per_half is None or len(optics_group_ids_per_half) != 2:
        raise ValueError(
            f"a {n_groups}-optics-group noise table needs optics_geometry.optics_group_ids_per_half for both halves"
        )
    ids = []
    for half, (values, dataset) in enumerate(zip(optics_group_ids_per_half, experiment_datasets)):
        values = np.asarray(values, dtype=np.int32).reshape(-1)
        if values.shape != (int(dataset.n_units),) or np.any(values < 0) or np.any(values >= n_groups):
            raise ValueError(
                f"half {half + 1} optics-group ids must give each of {int(dataset.n_units)} images "
                f"a row 0..{n_groups - 1}"
            )
        ids.append(values)
    return ids


def _internal_solvent_mask(path, box_size, pixel_size_angstrom):
    """RELION --solvent_mask on the model grid in relax's internal (z, y, x) frame, or None.

    Map files hold RELION's axis order, the transpose of the internal frame (relax.helpers.map_io);
    a mask has no sign to undo.
    """
    if path is None:
        return None
    from relax.reconstruction.solvent_mask import read_solvent_mask

    mask = read_solvent_mask(path, box_size=box_size, pixel_size=float(pixel_size_angstrom))
    return np.ascontiguousarray(np.transpose(mask, (2, 1, 0)))


def checked_run_optics(geometry: OpticsGeometry, image_geometry: ImageGeometry, *, multi_shape_halves: bool) -> RunOptics:
    """The run's optics geometry, from the options' RELION optics arrays and model pixel size.

    Refuses optics image and pixel sizes given one without the other, empty or of different lengths, and a model
    pixel size that is not positive and finite; without ``relion_model_pixel_size`` the model pixel size
    is the image pixel size.
    """
    if (geometry.relion_optics_image_sizes is None) != (geometry.relion_optics_pixel_sizes is None):
        raise ValueError(
            "relion_optics_image_sizes and relion_optics_pixel_sizes must be supplied together",
        )
    optics_image_sizes = None
    optics_pixel_sizes = None
    if geometry.relion_optics_image_sizes is not None:
        optics_image_sizes = np.asarray(geometry.relion_optics_image_sizes, dtype=np.int64).reshape(-1)
        optics_pixel_sizes = np.asarray(geometry.relion_optics_pixel_sizes, dtype=np.float64).reshape(-1)
        if optics_image_sizes.shape != optics_pixel_sizes.shape or optics_image_sizes.size == 0:
            raise ValueError("RELION optics image geometry arrays must be non-empty and aligned")
    model_pixel_size = (
        image_geometry.pixel_size_angstrom
        if geometry.relion_model_pixel_size is None
        else float(geometry.relion_model_pixel_size)
    )
    if not np.isfinite(model_pixel_size) or model_pixel_size <= 0.0:
        raise ValueError(f"RELION model pixel size must be positive, got {model_pixel_size}")
    return RunOptics(
        image_geometry=image_geometry,
        model_pixel_size=model_pixel_size,
        optics_image_sizes=optics_image_sizes,
        optics_pixel_sizes=optics_pixel_sizes,
        multi_shape_halves=multi_shape_halves,
    )


def translation_angle_scale_for_run(optics: RunOptics, *, subtomograms: bool) -> float:
    """The scale of RELION's translation phases for the run: 1.0 for several image shapes (their classes scale
    their own grids, ``ShapeClass.trial_grid_factor``) and for subtomograms, else the model-to-optics pixel-size
    ratio (``_relion_translation_angle_scale``), logged when not 1.0."""
    scale = (
        # Shape classes carry their translations in class pixels already; tilt images have their own phases.
        1.0
        if optics.multi_shape_halves or subtomograms
        else _relion_translation_angle_scale(
            model_pixel_size=optics.model_pixel_size,
            optics_pixel_sizes=optics.optics_pixel_sizes,
        )
    )
    if scale != 1.0:
        logger.info(
            "RELION translation phases: model_pixel_size=%.12g "
            "optics_pixel_size=%.12g angle_scale=%.17g",
            optics.model_pixel_size,
            float(optics.optics_pixel_sizes[0]),
            scale,
        )
    return scale


def projection_scale_for_run(optics: RunOptics, *, subtomograms: bool) -> float:
    """The scale difference ``s`` of a single-shape run's images against RELION's model grid, or 1.0.

    RELION scales every optics group's matrices by ``s = (box_g pix_g) / (ori model_pix)``
    (``applyScaleDifference``, obs_model.cpp:1332-1340) with the model pixel the reference header's,
    so a header that is not the STAR pixel gives single-shape data an ``s != 1`` too; whether it
    projects with it is RELION's identity test (:func:`relax.relion.optics_aberrations.relion_projection_optics`).
    Several image shapes carry their own class scales; tilt images their own left matrices.
    """

    if optics.multi_shape_halves or subtomograms or optics.optics_pixel_sizes is None:
        return 1.0
    pixel, box = optics.first_optics_group_geometry()
    scale = optics_scale.scale_difference(box, pixel, optics.image_geometry.box_size, optics.model_pixel_size)
    if scale != 1.0:
        logger.info("RELION single-shape scale difference: s=%.17g (model pixel %.12g)", scale, optics.model_pixel_size)
    return scale


def reconstruction_settings_for_run(
    options: RefinementOptions,
    image_geometry: ImageGeometry,
    volume_shape,
    consistency: RelionConsistencyOptions,
) -> ReconstructionSettings:
    """The run's one ``ReconstructionSettings``, every M-step's: RELION's padding and filter constants, the
    options' mask, fudge and programs, ``consistency``'s gridding and shell counting, and the ``--solvent_mask``
    read on the model grid (None without one)."""
    return ReconstructionSettings(
        box_size=image_geometry.box_size,
        voxel_size=image_geometry.pixel_size_angstrom,
        volume_shape=volume_shape,
        padding_factor=RECONSTRUCTION_PADDING_FACTOR,
        projection_padding_factor=PROJECTION_PADDING_FACTOR,
        minres_map=RELION_MINRES_MAP,
        width_mask_edge=options.schedule.width_mask_edge_px,
        fmask_edge=REFERENCE_FILTER_EDGE_SHELLS,
        tau2_fudge=options.parity.tau2_fudge,
        particle_diameter_angstrom=options.schedule.particle_diameter_ang,
        first_iteration_lowpass_angstrom=options.parity.relion_firstiter_ini_high_angstrom,
        gridding_kernel=consistency.gridding_kernel,
        shell_pair_counting=consistency.shell_pair_counting,
        solvent_mask=_internal_solvent_mask(
            options.solvent.mask_path, image_geometry.box_size, image_geometry.pixel_size_angstrom
        ),
        solvent_correct_fsc=options.solvent.correct_fsc,
        programs=options.variants.reconstruction,
        solvent_phase_stream=_solvent_phase_stream(options) if options.solvent.correct_fsc else None,
    )


def _solvent_phase_stream(options: RefinementOptions) -> GlibcRand:
    """RELION's MPI leader ``rand()`` stream for the corrected FSC's phases, at the run's first draw.

    The auto-refine reference is always MPI relion_refine, whose leader seeds once with
    ``init_random_generator(random_seed)`` (ml_optimiser_mpi.cpp:827) and never reseeds; the split's
    draws (``options.solvent.split_draws``) come first. relax.reconstruction.solvent_mask.
    """
    seed = (
        options.parity.perturb_seed
        if options.parity.optimizer_random_seed is None
        else options.parity.optimizer_random_seed
    )
    stream = init_random_generator(int(seed or 0))
    stream.rand_array(int(options.solvent.split_draws))
    return stream


def expected_accuracy_inputs_for_run(
    options: RefinementOptions, half1_dataset, volume_shape, *, optics_group_ids, gridding_kernel: str
) -> Half1AccuracyInputs:
    """The run-constant inputs of RELION's expected-accuracy estimate on half 1, with its trial order.

    ``optics_group_ids`` are half 1's per-image optics-group rows (None for one group); ``gridding_kernel`` is
    the projector's gridding-correction window.
    """
    # RELION randomises each half once at the first iteration and then uses
    # the first 100 half-1 particles for calculateExpectedAngularErrors.
    # Build that immutable local order once.  A missing/rebuilt-without-this-
    # helper binding is handled fail-closed below: acc_rot stays infinite and
    # cannot trigger convergence.
    effective_optimizer_random_seed = (
        options.parity.perturb_seed
        if options.parity.optimizer_random_seed is None
        else options.parity.optimizer_random_seed
    )
    expected_accuracy_trial_order = prepare_relion_half1_trial_order(
        expected_accuracy=options.expected_accuracy,
        half1_dataset=half1_dataset,
        optimizer_random_seed=effective_optimizer_random_seed,
        init_relion_iteration=options.schedule.init_relion_iteration,
        log=logger,
    )
    return Half1AccuracyInputs(
        trial_order_local=expected_accuracy_trial_order,
        dataset=half1_dataset,
        volume_shape=volume_shape,
        padding_factor=PROJECTION_PADDING_FACTOR,
        sigma2_fudge=RELION_DEFAULT_SIGMA2_FUDGE,
        optimizer_random_seed=effective_optimizer_random_seed,
        expected_accuracy=options.expected_accuracy,
        optics_group_ids=optics_group_ids,
        gridding_kernel=gridding_kernel,
    )


@dataclass(frozen=True, kw_only=True)
class RunContext:
    """What a refinement run resolves once from its datasets and options before the first iteration, and only
    reads afterwards (the run's options stay the controller's own record).

    ``source_pixel_size_angstrom`` is the datasets' own voxel-size scalar, kept with its input type for host
    arithmetic (``image_geometry.pixel_size_angstrom`` is its validated Python float). Frozen fields, not
    deeply immutable: ``perturb_rng`` (the native perturbation's RNG, None with a perturbation seed) advances
    each time the run draws a perturbation; the other members are not modified.
    """

    # The dtype of the controller's float64-sensitive host operands (rotation grids, priors), from the precision.
    scoring_dtype: object
    volume_shape: tuple
    # The volume on the reconstruction's padded grid.
    padded_volume_shape: tuple
    source_pixel_size_angstrom: object
    image_geometry: ImageGeometry
    k_class_enabled: bool
    # Subtomogram particles (units are particles over their tilt images, offsets are 3D); halves of several
    # image shapes.
    tomo_halves: bool
    multi_shape_halves: bool
    optics: RunOptics
    # Opt-in corrections of RELION's inconsistencies, refused on the routes that keep RELION's rules.
    consistency: RelionConsistencyOptions
    relion_translation_angle_scale: float
    # A single-shape run's scale difference against the model grid (``projection_scale_for_run``).
    relion_projection_scale: float
    reconstruction_settings: ReconstructionSettings
    snapshot_capture: SnapshotCapture
    batch_planner: BatchPlanner
    collect_local_search_profile: bool
    perturb_rng: object
    # The run observer's mid-step hooks, the only part of it a step sees (relax.refinement.ports).
    expectation_probe: ExpectationProbe
    maximization_probe: MaximizationProbe

    @property
    def offset_dims(self) -> int:
        """The number of translation components: 3 for subtomograms, 2 for single particles."""
        return 3 if self.tomo_halves else 2


def build_run_context(
    experiment_datasets,
    options: RefinementOptions,
    *,
    replays_relion_state: bool,
    observer: RunObserver,
) -> RunContext:
    """Resolve the run's context from its two half datasets and validated ``options``, refusing unsupported
    routes, and configure the datasets' image preprocessing (Fourier backend and masks) in place: the datasets'
    images are prepared before anything reads them.

    ``replays_relion_state`` whether the input source replays a RELION run's state (the consistency options are
    refused then); ``observer`` is the run's observer: the context keeps its mid-step hooks as probes and
    whether it asks for the local searches' profiles.
    """
    volume_shape = experiment_datasets[0].volume_shape
    # Keep the input scalar type for host arithmetic; geometry validates its value.
    source_pixel_size_angstrom = experiment_datasets[0].voxel_size
    image_geometry = ImageGeometry(
        image_shape=experiment_datasets[0].image_shape, pixel_size_angstrom=source_pixel_size_angstrom,
    )
    multi_shape_halves = isinstance(experiment_datasets[0], MultiShapeHalf)
    optics = checked_run_optics(options.optics_geometry, image_geometry, multi_shape_halves=multi_shape_halves)
    tomo_halves = isinstance(experiment_datasets[0], TomoHalf)
    consistency = require_consistency_route(
        options, subtomograms=tomo_halves, several_image_shapes=multi_shape_halves,
        replays_relion_state=replays_relion_state,
    )
    relion_translation_angle_scale = translation_angle_scale_for_run(optics, subtomograms=tomo_halves)
    relion_projection_scale = projection_scale_for_run(optics, subtomograms=tomo_halves)
    reconstruction_settings = reconstruction_settings_for_run(options, image_geometry, volume_shape, consistency)
    snapshot_capture = SnapshotCapture(
        n_classes=options.k_class.n_classes, box_size=image_geometry.box_size,
        voxel_size=image_geometry.pixel_size_angstrom, tau2_fudge=options.parity.tau2_fudge,
        consistency=consistency.non_default(),
    )
    configure_half_image_preprocessing(
        experiment_datasets,
        pixel_size_angstrom=source_pixel_size_angstrom,
        particle_diameter_angstrom=options.schedule.particle_diameter_ang,
        width_mask_edge_px=options.schedule.width_mask_edge_px,
        fourier_backend=options.parity.image_fourier_backend,
        # RELION's source-faithful powerClass normalisation applies wherever its particle order is preserved.
        source_faithful_spectrum_norm=options.parity.preserve_bpref_particle_order,
        log=logger,
    )
    return RunContext(
        scoring_dtype=options.precision.rotation_real_dtype,
        volume_shape=volume_shape,
        padded_volume_shape=tuple(d * RECONSTRUCTION_PADDING_FACTOR for d in volume_shape),
        source_pixel_size_angstrom=source_pixel_size_angstrom,
        image_geometry=image_geometry,
        k_class_enabled=options.k_class.n_classes > 1,
        tomo_halves=tomo_halves,
        multi_shape_halves=multi_shape_halves,
        optics=optics,
        consistency=consistency,
        relion_translation_angle_scale=relion_translation_angle_scale,
        relion_projection_scale=relion_projection_scale,
        reconstruction_settings=reconstruction_settings,
        snapshot_capture=snapshot_capture,
        batch_planner=BatchPlanner(
            requested=options.execution, image_shape=image_geometry.image_shape, volume_shape=volume_shape,
            n_classes=options.k_class.n_classes, precision=options.precision, log=logger,
        ),
        collect_local_search_profile=options.local_search.collects_profile(observer.collects_local_search_profiles),
        perturb_rng=None if options.parity.perturb_seed is not None else np.random.default_rng(),
        expectation_probe=ObserverExpectationProbe(observer),
        maximization_probe=ObserverMaximizationProbe(observer),
    )
