"""Set-up checks of a refinement run: what refine_single_volume refuses or resolves from its inputs before the
first iteration (code rule 12: check at the edge, then trust).
"""

import logging

import numpy as np

from relax.helpers.expected_accuracy import (
    RELION_DEFAULT_SIGMA2_FUDGE,
    Half1AccuracyInputs,
    prepare_relion_half1_trial_order,
)
from relax.helpers.resolution import ImageGeometry
from relax.reconstruction.regularization_relion import RELION_MINRES_MAP
from relax.refinement.iteration_planning import RunOptics
from relax.refinement.mean_helpers import ReconstructionSettings
from relax.refinement.refinement_options import RefinementOptions, RelionConsistencyOptions, RelionParityOptions
from relax.relion.geometry import (
    PROJECTION_PADDING_FACTOR,
    RECONSTRUCTION_PADDING_FACTOR,
    REFERENCE_FILTER_EDGE_SHELLS,
)

# The set-up logs as part of the controller's run.
logger = logging.getLogger("relax.refinement.iteration_loop")


def _relion_k1_translation_angle_scale(
    *,
    n_classes: int,
    model_pixel_size: float,
    optics_pixel_sizes,
) -> float:
    """Convert K=1 model-pixel translations to the shared optics pixel size.

    RELION keeps sampling translations in Angstrom and converts them with the
    particle's optics pixel size (HealpixSampling::getTranslationsInPixel),
    while RECOVAR's translation grid is in model pixels. The scale
    model_pixel_size / optics_pixel_size is exactly 1.0 when the serialized
    sizes agree; it multiplies only the RELION translation-phase operand.
    """

    if int(n_classes) != 1 or optics_pixel_sizes is None:
        return 1.0
    optics = np.asarray(optics_pixel_sizes, dtype=np.float64).reshape(-1)
    if optics.size == 0 or not np.all(np.isfinite(optics)) or np.any(optics <= 0.0):
        raise ValueError("RELION optics pixel sizes must be non-empty, positive, and finite")
    unique_optics = np.unique(optics)
    if unique_optics.size != 1:
        raise NotImplementedError(
            "K=1 exact RELION translation phases currently require one shared optics pixel size; "
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
            f"a {n_groups}-optics-group noise table needs parity.optics_group_ids_per_half for both halves"
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


def checked_run_optics(parity: RelionParityOptions, image_geometry: ImageGeometry, *, multi_shape_halves: bool) -> RunOptics:
    """The run's optics geometry, from ``parity``'s RELION optics arrays and model pixel size.

    Refuses optics image and pixel sizes given one without the other, empty or of different lengths, and a model
    pixel size that is not positive and finite; without ``parity.relion_model_pixel_size`` the model pixel size
    is the image pixel size.
    """
    if (parity.relion_optics_image_sizes is None) != (parity.relion_optics_pixel_sizes is None):
        raise ValueError(
            "relion_optics_image_sizes and relion_optics_pixel_sizes must be supplied together",
        )
    optics_image_sizes = None
    optics_pixel_sizes = None
    if parity.relion_optics_image_sizes is not None:
        optics_image_sizes = np.asarray(parity.relion_optics_image_sizes, dtype=np.int64).reshape(-1)
        optics_pixel_sizes = np.asarray(parity.relion_optics_pixel_sizes, dtype=np.float64).reshape(-1)
        if optics_image_sizes.shape != optics_pixel_sizes.shape or optics_image_sizes.size == 0:
            raise ValueError("RELION optics image geometry arrays must be non-empty and aligned")
    model_pixel_size = (
        image_geometry.pixel_size_angstrom
        if parity.relion_model_pixel_size is None
        else float(parity.relion_model_pixel_size)
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


def translation_angle_scale_for_run(optics: RunOptics, *, n_classes: int, subtomograms: bool) -> float:
    """The scale of RELION's translation phases for the run: 1.0 for several image shapes and for subtomograms,
    else the K=1 model-to-optics pixel-size ratio (``_relion_k1_translation_angle_scale``), logged when not 1.0."""
    scale = (
        # Shape classes carry their translations in class pixels already; tilt images have their own phases.
        1.0
        if optics.multi_shape_halves or subtomograms
        else _relion_k1_translation_angle_scale(
            n_classes=n_classes,
            model_pixel_size=optics.model_pixel_size,
            optics_pixel_sizes=optics.optics_pixel_sizes,
        )
    )
    if scale != 1.0:
        logger.info(
            "RELION K=1 translation phases: model_pixel_size=%.12g "
            "optics_pixel_size=%.12g angle_scale=%.17g",
            optics.model_pixel_size,
            float(optics.optics_pixel_sizes[0]),
            scale,
        )
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
        solvent_fsc_seed=int(
            (
                options.parity.perturb_seed
                if options.parity.optimizer_random_seed is None
                else options.parity.optimizer_random_seed
            )
            or 0
        ),
    )


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
