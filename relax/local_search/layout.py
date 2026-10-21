"""Per-image local hypothesis layout: the flat rotation, prior and translation-mask rows of each image."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from relax.fine_pass.significant_samples import significant_sample_ids
from relax.local_search.direction_supports import (
    build_factorized_local_entries,
    build_parent_expanded_local_entries,
    flat_parts,
)
from relax.sampling.device_rotations import DevicePass1Source, project_rows, relion_adaptive_pass1_rotations
from relax.sampling.grids import (
    build_local_search_grid_metadata,
    relion_mstep_rotations_from_eulers,
    rotation_grid_n_in_planes,
    rotation_grid_size,
)
from relax.sampling.orientation_priors import make_relion_translation_log_prior
from relax.sampling.oversampled_grids import (
    compute_oversampled_rotation_grid,
    get_local_rotation_grid_fast,
    get_oversampled_rotation_grid_from_samples,
    get_oversampled_translation_grid,
    infer_translation_step,
    unique_nonnegative_ids,
)
from relax.sampling.perturbation import apply_relion_rotation_perturbation_to_eulers
from relax.sampling.symmetry import canonicalize_rotational_symmetry


@dataclass(frozen=True)
class LocalHypothesisLayout:
    """Flat per-image local hypothesis storage."""

    n_global_rotations: int
    n_pixels: int
    n_psi: int
    rotation_offsets: np.ndarray
    rotation_ids_flat: np.ndarray
    rotations_flat: np.ndarray
    rotation_log_priors_flat: np.ndarray
    rotation_counts: np.ndarray
    translation_grid: np.ndarray
    translation_log_priors: np.ndarray
    rotation_posterior_ids_flat: np.ndarray | None = None
    sample_mask_bits: np.ndarray | None = None  # uint8, translation bits packed little-endian
    mstep_rotations_flat: np.ndarray | None = None
    source_eulers_flat: np.ndarray | None = None
    # The rows are oversampled children of parents (RELION's oversampled orientations): rows and M-step rows
    # are the host inverse matrices of ``source_eulers_flat``, and ``rotation_ids_flat`` index the fine grid.
    oversampled_rows: bool = False
    # Other rows were built from ``rotation_ids_flat`` on the local grid of this (healpix_order,
    # random_perturbation, angular_sampling_deg): the builder's arguments, which a caller's grid need not repeat.
    id_rows_source: tuple[int, float, float | None] | None = None
    symmetry: str = "C1"
    # Class3D (K>1) local searches: every image's rows repeated once per class, class-major within the
    # image (:func:`expand_local_layout_classes`); ``row_class_flat`` is each row's class. None: K=1.
    row_class_flat: np.ndarray | None = None
    n_classes: int = 1

    def sample_mask_rows(self, start=0, stop=None) -> np.ndarray | None:
        """Expand only the requested rotation rows to the kernel's boolean mask."""
        if self.sample_mask_bits is None:
            return None
        return np.unpackbits(
            self.sample_mask_bits[start:stop], axis=1,
            count=int(self.translation_grid.shape[0]), bitorder="little",
        ).view(np.bool_)

    @property
    def n_images(self) -> int:
        return int(self.rotation_counts.shape[0])


def _rotation_eulers_from_grid_metadata(
    rotation_ids: np.ndarray,
    grid_metadata,
    *,
    dtype=np.float32,
) -> np.ndarray:
    """Return canonical RELION Euler angles for selected rotation ids only."""

    rotation_ids = np.asarray(rotation_ids, dtype=np.int64).reshape(-1)
    if rotation_ids.size == 0:
        return np.zeros((0, 3), dtype=dtype)
    if "eulers_full" in grid_metadata:
        return np.asarray(grid_metadata["eulers_full"], dtype=dtype)[rotation_ids]
    if str(grid_metadata["mode"]) != "factorized":
        raise ValueError("Selected rotation eulers require factorized metadata or eulers_full")
    n_pixels = int(grid_metadata["n_pixels"])
    pixel_idx = rotation_ids % n_pixels
    psi_idx = rotation_ids // n_pixels
    return np.stack(
        [
            np.asarray(grid_metadata["rot_deg"], dtype=dtype)[pixel_idx],
            np.asarray(grid_metadata["tilt_deg"], dtype=dtype)[pixel_idx],
            np.asarray(grid_metadata["psi_deg"], dtype=dtype)[psi_idx],
        ],
        axis=1,
    ).astype(dtype, copy=False)


def selected_rotation_matrices(
    rotation_ids: np.ndarray,
    rotation_grid_rotations: np.ndarray | None,
    grid_metadata,
    *,
    random_perturbation: float = 0.0,
    angular_sampling_deg: float | None = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Build matrices for selected local ids without materializing the full grid."""

    rotation_ids = np.asarray(rotation_ids, dtype=np.int64).reshape(-1)
    if rotation_ids.size == 0:
        return np.zeros((0, 3, 3), dtype=dtype)
    if rotation_grid_rotations is not None:
        return np.asarray(rotation_grid_rotations, dtype=dtype).reshape(-1, 3, 3)[rotation_ids]
    unique_ids, inverse = unique_nonnegative_ids(rotation_ids)
    # Eulers are staged at the requested dtype; apply_relion_rotation_perturbation_to_eulers
    # works in float64 internally, so a dtype=float64 caller keeps full precision.
    # The mstep sibling below always stages them in float64.
    selected_eulers = _rotation_eulers_from_grid_metadata(unique_ids, grid_metadata, dtype=dtype)
    if abs(float(random_perturbation)) > 1e-12:
        if angular_sampling_deg is None:
            raise ValueError("angular_sampling_deg is required when random_perturbation is nonzero")
        rotations, _ = apply_relion_rotation_perturbation_to_eulers(
            selected_eulers,
            float(random_perturbation),
            float(angular_sampling_deg),
            dtype=dtype,
        )
    else:
        # Preserve RELION's accelerated-path handoff: host RFLOAT inverse
        # matrices are cast to XFLOAT before scoring on the device (a no-op
        # cast under ACC_DOUBLE_PRECISION, when dtype=float64).
        rotations, _ = apply_relion_rotation_perturbation_to_eulers(
            selected_eulers,
            0.0,
            0.0,
            dtype=dtype,
        )
    return rotations.astype(dtype, copy=False)[inverse]


def _selected_mstep_rotation_matrices(
    rotation_ids: np.ndarray,
    rotation_grid_mstep_rotations: np.ndarray | None,
    grid_metadata,
    *,
    random_perturbation: float = 0.0,
    angular_sampling_deg: float | None = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Build RELION host-path adjoint matrices for selected local ids."""

    rotation_ids = np.asarray(rotation_ids, dtype=np.int64).reshape(-1)
    if rotation_ids.size == 0:
        return np.zeros((0, 3, 3), dtype=dtype)
    if rotation_grid_mstep_rotations is not None:
        return np.asarray(rotation_grid_mstep_rotations, dtype=dtype).reshape(-1, 3, 3)[rotation_ids]
    unique_ids, inverse = unique_nonnegative_ids(rotation_ids)
    selected_eulers = _rotation_eulers_from_grid_metadata(unique_ids, grid_metadata, dtype=np.float64)
    if angular_sampling_deg is None:
        if abs(float(random_perturbation)) > 1e-12:
            raise ValueError("angular_sampling_deg is required when random_perturbation is nonzero")
        angular_sampling_deg = 0.0
    mstep_rotations, _ = apply_relion_rotation_perturbation_to_eulers(
        selected_eulers,
        float(random_perturbation),
        float(angular_sampling_deg),
        dtype=dtype,
    )
    return np.asarray(mstep_rotations, dtype=dtype)[inverse]


def local_layout_device_rotations(layout: LocalHypothesisLayout, scale: float, magnification, *, use_float64: bool):
    """A parent layout's scoring rows as RELION's pass-1 projector plan builds them, or None on the CPU.

    RELION's coarse pass scores a local search's parent orientations through an ``AccProjectorPlan`` set up
    with ``coarse = true`` (``getAllSquaredDifferencesCoarse``, acc_ml_optimiser_impl.h:1063-1145): the plan
    forms ``L (A R)`` on the device in XFLOAT from the Euler angles, the perturbation matrix ``R`` and the
    optics' left matrix ``L`` (``acc_make_eulers_3D``, acc_projector_plan_impl.h:158-392), as it does for a
    global search. These are not the host inverse matrices of the fine pass and the weighted sums
    (:func:`local_layout_host_rotations`): the two differ by a few float32 ulps in every row, which moves the
    score of a few percent of the orientations by about 1e-4 and reorders near-tied parents.

    The rows are those of the global pass 1 (:func:`relax.sampling.device_rotations.relion_adaptive_pass1_rotations` and, for
    images on another grid or magnified, :func:`relax.sampling.device_rotations.project_rows` with the device source), built
    from the unperturbed Euler rows of ``rotation_ids_flat`` in the layout's dtype, with the perturbation and
    angular step the layout was built with. Only id rows have a plan: oversampled children are fine-pass rows.
    """

    if layout.oversampled_rows or layout.id_rows_source is None:
        raise ValueError("only a parent layout's id rows are built by RELION's pass-1 projector plan")
    healpix_order, random_perturbation, angular_sampling_deg = layout.id_rows_source
    dtype = np.asarray(layout.rotations_flat).dtype
    unique_ids, inverse = unique_nonnegative_ids(np.asarray(layout.rotation_ids_flat, dtype=np.int64).reshape(-1))
    source_eulers = _rotation_eulers_from_grid_metadata(
        unique_ids, build_local_search_grid_metadata(healpix_order, symmetry=layout.symmetry), dtype=dtype
    )
    source = DevicePass1Source(
        source_eulers_deg=np.asarray(source_eulers, dtype=np.float64),
        random_perturbation=float(random_perturbation),
        angular_sampling_deg=0.0 if angular_sampling_deg is None else float(angular_sampling_deg),
        use_float64=bool(use_float64),
    )
    rows = relion_adaptive_pass1_rotations(
        source.source_eulers_deg,
        source.random_perturbation,
        source.angular_sampling_deg,
        use_float64=source.use_float64,
    )
    if rows is None:
        return None
    rows = project_rows(np.asarray(rows), scale, magnification, device_source=source, what="local parent rows")
    return np.asarray(rows, dtype=dtype)[inverse]


def local_layout_host_rotations(layout: LocalHypothesisLayout, *, mstep: bool) -> np.ndarray:
    """A layout's scoring (or, with ``mstep``, M-step) rows as float64 matrices, by the rule that built them.

    The rows of images on another grid or magnified are composed from these and cast once
    (:func:`relax.sampling.device_rotations.project_rows`). Oversampled children are the host inverse matrices of their
    source Euler rows (:func:`relax.sampling.oversampled_grids.get_oversampled_rotation_grid_from_samples`); other rows are
    rebuilt from ``rotation_ids_flat`` with the grid and perturbation the layout was built with
    (``id_rows_source``), not the caller's.
    """

    if layout.oversampled_rows:
        if layout.source_eulers_flat is None:
            raise ValueError("oversampled local rows carry no source Euler rows to rebuild them from")
        return relion_mstep_rotations_from_eulers(np.asarray(layout.source_eulers_flat), dtype=np.float64)
    if layout.id_rows_source is None:
        raise ValueError("local rows record no builder arguments to rebuild them from")
    healpix_order, random_perturbation, angular_sampling_deg = layout.id_rows_source
    builder = _selected_mstep_rotation_matrices if mstep else selected_rotation_matrices
    return builder(
        layout.rotation_ids_flat,
        None,
        build_local_search_grid_metadata(healpix_order, symmetry=layout.symmetry),
        random_perturbation=random_perturbation,
        angular_sampling_deg=angular_sampling_deg,
        dtype=np.float64,
    )


def build_local_hypothesis_layout(
    prior_rotations: np.ndarray,
    rotation_grid_rotations: np.ndarray | None,
    sigma_rot: float,
    sigma_psi: float,
    healpix_order: int,
    translations: np.ndarray,
    prior_translations: np.ndarray,
    sigma_offset_angstrom: float,
    offset_range_pixels: float | None,
    voxel_size: float,
    *,
    grid_metadata,
    translation_prior_reference_translations: np.ndarray | None = None,
    rotation_log_prior: np.ndarray | None = None,
    rotation_grid_random_perturbation: float = 0.0,
    rotation_grid_angular_sampling_deg: float | None = None,
    local_parent_oversampling_order: int = 0,
    rotation_grid_mstep_rotations: np.ndarray | None = None,
    generate_relion_mstep_rotations: bool = False,
    dtype: np.dtype = np.float32,
    sigma_tilt_deg: float,
    prior_tilt_deg: np.ndarray | None = None,
    engine_tuning,
) -> LocalHypothesisLayout:
    """Build exact per-image local neighborhoods and translation priors.

    ``engine_tuning`` (the command's ``EngineTuning``) sizes the orientation selector's image chunks.

    ``sigma_tilt_deg`` with ``prior_tilt_deg`` (and ``sigma_rot == sigma_psi == 0``) is RELION's prior on
    the tilt angle alone (``--sigma_tilt``, in degrees as given): every in-plane angle and the directions
    within three widths of the image's prior tilt (``direction_supports._tilt_prior_direction_supports``).

    ``dtype`` controls the precision of every rotation/translation/prior
    array built here (default float32, matching RELION's accelerated-GPU
    single-precision path). Pass ``np.float64`` to keep this local-search
    hypothesis grid genuinely double precision end to end; the caller is
    responsible for deriving this from ``use_float64_scoring`` /
    ``use_float64_projections`` so the default stays unchanged.
    """

    # The priors keep their own precision: local supports are selected in double (direction_supports._resolve_prior_rotations).
    prior_rotations = np.asarray(prior_rotations)
    symmetry = canonicalize_rotational_symmetry(str(grid_metadata.get("symmetry", "C1")))
    if rotation_grid_rotations is not None:
        rotation_grid_rotations = np.asarray(rotation_grid_rotations, dtype=dtype).reshape(-1, 3, 3)
    if rotation_grid_mstep_rotations is not None:
        rotation_grid_mstep_rotations = np.asarray(rotation_grid_mstep_rotations, dtype=dtype).reshape(-1, 3, 3)
        expected_rotation_count = (
            int(rotation_grid_rotations.shape[0])
            if rotation_grid_rotations is not None
            else int(grid_metadata["n_pixels"]) * int(grid_metadata["n_psi"])
        )
        if int(rotation_grid_mstep_rotations.shape[0]) != expected_rotation_count:
            raise ValueError(
                "rotation_grid_mstep_rotations must match the scoring grid size: "
                f"{rotation_grid_mstep_rotations.shape[0]} vs {expected_rotation_count}",
            )
    generate_relion_mstep_rotations = bool(
        generate_relion_mstep_rotations or rotation_grid_mstep_rotations is not None
    )
    translations = np.asarray(translations, dtype=dtype)
    prior_translations = np.asarray(prior_translations, dtype=dtype).reshape(-1, translations.shape[1])
    rotation_log_prior_np = None if rotation_log_prior is None else np.asarray(rotation_log_prior, dtype=dtype)

    source_eulers_flat = None
    rotations_flat_override = None
    mstep_rotations_flat_override = None
    rotation_posterior_ids_flat_override = None
    if int(local_parent_oversampling_order) > 0:
        (
            offsets,
            counts,
            rotation_ids_flat,
            rotation_log_priors_flat,
            rotations_flat_override,
            mstep_rotations_flat_override,
            source_eulers_flat,
            rotation_posterior_ids_flat_override,
        ) = build_parent_expanded_local_entries(
            prior_rotations,
            healpix_order,
            sigma_rot,
            sigma_psi,
            oversampling_order=int(local_parent_oversampling_order),
            rotation_log_prior=rotation_log_prior_np,
            random_perturbation=float(rotation_grid_random_perturbation),
            generate_relion_mstep_rotations=generate_relion_mstep_rotations,
            dtype=dtype,
            symmetry=symmetry,
            sigma_tilt_deg=sigma_tilt_deg,
            prior_tilt_deg=prior_tilt_deg,
            engine_tuning=engine_tuning,
        )
    elif str(grid_metadata["mode"]) == "factorized":
        offsets, counts, rotation_ids_flat, rotation_log_priors_flat = build_factorized_local_entries(
            prior_rotations,
            healpix_order,
            sigma_rot,
            sigma_psi,
            grid_metadata,
            dtype=dtype,
            sigma_tilt_deg=sigma_tilt_deg,
            prior_tilt_deg=prior_tilt_deg,
            engine_tuning=engine_tuning,
        )
    else:
        if sigma_tilt_deg > 0.0:
            raise NotImplementedError("a tilt prior needs the factorized HEALPix x psi grid")
        n_images = int(prior_rotations.shape[0])
        offsets = np.zeros(n_images + 1, dtype=np.int64)
        counts = np.zeros(n_images, dtype=np.int32)
        rotation_ids_parts: list[np.ndarray] = []
        log_prior_parts: list[np.ndarray] = []

        for image_idx in range(n_images):
            local_ids, local_log_prior = get_local_rotation_grid_fast(
                prior_rotations[image_idx : image_idx + 1],
                sigma_rot,
                sigma_psi,
                healpix_order,
                sigma_cutoff=3.0,
                per_image=True,
                grid_metadata=grid_metadata,
            )
            local_ids = np.asarray(local_ids, dtype=np.int64).reshape(-1)
            local_log_prior = np.asarray(local_log_prior[0], dtype=dtype).reshape(-1)
            counts[image_idx] = int(local_ids.shape[0])
            offsets[image_idx + 1] = offsets[image_idx] + local_ids.shape[0]
            rotation_ids_parts.append(local_ids)
            log_prior_parts.append(local_log_prior)

        rotation_ids_flat = flat_parts(rotation_ids_parts, empty_shape=0, dtype=np.int64, cast=np.int64)
        rotation_log_priors_flat = flat_parts(log_prior_parts, empty_shape=0, dtype=dtype)
    if rotation_log_prior_np is not None and int(local_parent_oversampling_order) <= 0:
        if rotation_log_prior_np.shape[0] != int(grid_metadata["n_pixels"]) * int(grid_metadata["n_psi"]):
            raise ValueError(
                "rotation_log_prior must have one value per local-grid rotation "
                f"({int(grid_metadata['n_pixels']) * int(grid_metadata['n_psi'])}); "
                f"got {rotation_log_prior_np.shape}"
            )
        rotation_log_priors_flat = (
            rotation_log_priors_flat + rotation_log_prior_np[np.asarray(rotation_ids_flat, dtype=np.int64)]
        )
    rotations_flat = (
        rotations_flat_override
        if rotations_flat_override is not None
        else selected_rotation_matrices(
            rotation_ids_flat,
            rotation_grid_rotations,
            grid_metadata,
            random_perturbation=rotation_grid_random_perturbation,
            angular_sampling_deg=rotation_grid_angular_sampling_deg,
            dtype=dtype,
        )
    )
    mstep_rotations_flat = None
    if generate_relion_mstep_rotations:
        mstep_rotations_flat = (
            mstep_rotations_flat_override
            if mstep_rotations_flat_override is not None
            else _selected_mstep_rotation_matrices(
                rotation_ids_flat,
                rotation_grid_mstep_rotations,
                grid_metadata,
                random_perturbation=rotation_grid_random_perturbation,
                angular_sampling_deg=rotation_grid_angular_sampling_deg,
                dtype=dtype,
            )
        )
    translation_grid = translations
    translation_parent = None
    if int(local_parent_oversampling_order) > 0:
        translation_grid, translation_parent = get_oversampled_translation_grid(
            translations,
            infer_translation_step(translations),
            oversampling_order=int(local_parent_oversampling_order),
        )
        translation_grid = np.asarray(translation_grid, dtype=dtype)
        translation_parent = np.asarray(translation_parent, dtype=np.int32)

    reference_translations = (
        np.asarray(translation_prior_reference_translations, dtype=dtype)
        if translation_prior_reference_translations is not None
        else translations
    )

    coarse_translation_log_priors = make_relion_translation_log_prior(
        reference_translations,
        voxel_size,
        sigma_offset_angstrom,
        prior_translations,
        offset_range_pixels=offset_range_pixels,
        dtype=dtype,
    ).astype(dtype, copy=False)
    if translation_parent is None:
        translation_log_priors = coarse_translation_log_priors
    else:
        translation_log_priors = _fine_translation_log_prior(
            coarse_translation_log_priors,
            translation_parent,
            int(prior_translations.shape[0]),
            int(translation_grid.shape[0]),
            dtype=dtype,
        )

    if rotation_posterior_ids_flat_override is not None:
        parent_order = int(healpix_order) - int(local_parent_oversampling_order)
        n_global_rotations = rotation_grid_size(parent_order, symmetry)
    elif rotation_grid_rotations is not None:
        n_global_rotations = int(rotation_grid_rotations.shape[0])
    else:
        n_global_rotations = int(grid_metadata["n_pixels"]) * int(grid_metadata["n_psi"])

    if (
        source_eulers_flat is None
        and int(local_parent_oversampling_order) == 0
        and rotation_grid_rotations is None
        and str(grid_metadata["mode"]) == "factorized"
    ):
        source_eulers_flat = get_oversampled_rotation_grid_from_samples(
            rotation_ids_flat,
            healpix_order,
            oversampling_order=0,
            random_perturbation=float(rotation_grid_random_perturbation),
            return_source_eulers=True,
            dtype=dtype,
            symmetry=symmetry,
        )[-1]
    if rotation_posterior_ids_flat_override is not None:
        posterior_ids = np.asarray(rotation_posterior_ids_flat_override, dtype=np.int64)
        if posterior_ids.shape != np.asarray(rotation_ids_flat).shape:
            raise RuntimeError(
                "parent-expanded local posterior ids must match the scored rotation rows"
            )
        if np.any(posterior_ids < 0) or int(posterior_ids.max(initial=-1)) >= n_global_rotations:
            raise RuntimeError(
                f"parent-expanded {symmetry} posterior ids must index the reduced "
                f"parent grid of size {n_global_rotations}"
            )

    return LocalHypothesisLayout(
        n_global_rotations=n_global_rotations,
        n_pixels=int(grid_metadata["n_pixels"]),
        n_psi=int(grid_metadata["n_psi"]),
        rotation_offsets=offsets,
        rotation_ids_flat=rotation_ids_flat,
        rotations_flat=rotations_flat,
        rotation_log_priors_flat=rotation_log_priors_flat,
        rotation_counts=counts,
        translation_grid=translation_grid,
        translation_log_priors=np.asarray(translation_log_priors, dtype=dtype),
        mstep_rotations_flat=mstep_rotations_flat,
        source_eulers_flat=source_eulers_flat,
        oversampled_rows=int(local_parent_oversampling_order) > 0,
        id_rows_source=(
            None
            if int(local_parent_oversampling_order) > 0
            else (
                int(healpix_order),
                float(rotation_grid_random_perturbation),
                None if rotation_grid_angular_sampling_deg is None else float(rotation_grid_angular_sampling_deg),
            )
        ),
        rotation_posterior_ids_flat=rotation_posterior_ids_flat_override,
        symmetry=symmetry,
    )


def expand_local_layout_classes(layout: LocalHypothesisLayout, n_classes: int) -> LocalHypothesisLayout:
    """The layout with every image's rows repeated for each of ``n_classes`` classes, class-major per image.

    RELION's Class3D local search centres one search per particle on its previous best angles and scores
    every class at the same orientations (ml_optimiser.cpp, getFourierTransformsAndCtfs and
    getAllSquaredDifferences loop over classes inside one particle). A row's prior, rotation, mask and ids
    are its K=1 row's; ``row_class_flat`` says which class's reference it is scored against.
    """

    if n_classes < 2:
        raise ValueError(f"a class expansion needs at least two classes, got {n_classes}")
    if layout.n_classes != 1:
        raise ValueError("the layout is already expanded over classes")
    offsets = np.asarray(layout.rotation_offsets, dtype=np.int64)
    counts = np.diff(offsets)
    n_images = int(counts.shape[0])
    # Source row of every expanded row: image i's rows once per class, in class order.
    image_of = np.repeat(np.arange(n_images, dtype=np.int64), counts * n_classes)
    within = np.arange(int(counts.sum()) * n_classes, dtype=np.int64) - np.repeat(
        offsets[:-1] * n_classes, counts * n_classes
    )
    image_counts = counts[image_of]
    source = offsets[:-1][image_of] + within % np.maximum(image_counts, 1)
    row_class = (within // np.maximum(image_counts, 1)).astype(np.int32)

    def take(values):
        return None if values is None else np.asarray(values)[source]

    new_counts = (counts * n_classes).astype(np.asarray(layout.rotation_counts).dtype)
    new_offsets = np.zeros(n_images + 1, dtype=np.int64)
    new_offsets[1:] = np.cumsum(new_counts, dtype=np.int64)
    return replace(
        layout,
        rotation_offsets=new_offsets,
        rotation_ids_flat=take(layout.rotation_ids_flat),
        rotations_flat=take(layout.rotations_flat),
        rotation_log_priors_flat=take(layout.rotation_log_priors_flat),
        rotation_counts=new_counts,
        rotation_posterior_ids_flat=take(layout.rotation_posterior_ids_flat),
        sample_mask_bits=take(layout.sample_mask_bits),
        mstep_rotations_flat=take(layout.mstep_rotations_flat),
        source_eulers_flat=take(layout.source_eulers_flat),
        row_class_flat=row_class,
        n_classes=n_classes,
    )


def restrict_local_layout_classes(layout: LocalHypothesisLayout, image_classes) -> LocalHypothesisLayout:
    """A class-expanded layout keeping, for each image, only the rows of its given class.

    RELION's first Class3D iteration from one reference scores each particle against one random class
    only (do_generate_seeds: exp_iclass_min = exp_iclass_max), with or without local searches.
    """

    if layout.n_classes == 1:
        raise ValueError("only a class-expanded layout has classes to restrict")
    image_classes = np.asarray(image_classes, dtype=np.int64).reshape(-1)
    offsets = np.asarray(layout.rotation_offsets, dtype=np.int64)
    counts = np.diff(offsets)
    if image_classes.shape != counts.shape:
        raise ValueError(f"one class per image is needed, got {image_classes.shape} for {counts.shape[0]} images")
    row_image = np.repeat(np.arange(counts.shape[0], dtype=np.int64), counts)
    keep = np.asarray(layout.row_class_flat, dtype=np.int64) == image_classes[row_image]
    new_counts = np.bincount(row_image[keep], minlength=counts.shape[0])
    if np.any(new_counts == 0):
        raise ValueError(f"image {int(np.flatnonzero(new_counts == 0)[0])} has no row of its seed class")

    def take(values):
        return None if values is None else np.asarray(values)[keep]

    new_offsets = np.zeros(counts.shape[0] + 1, dtype=np.int64)
    new_offsets[1:] = np.cumsum(new_counts, dtype=np.int64)
    return replace(
        layout,
        rotation_offsets=new_offsets,
        rotation_ids_flat=take(layout.rotation_ids_flat),
        rotations_flat=take(layout.rotations_flat),
        rotation_log_priors_flat=take(layout.rotation_log_priors_flat),
        rotation_counts=new_counts.astype(np.asarray(layout.rotation_counts).dtype),
        rotation_posterior_ids_flat=take(layout.rotation_posterior_ids_flat),
        sample_mask_bits=take(layout.sample_mask_bits),
        mstep_rotations_flat=take(layout.mstep_rotations_flat),
        source_eulers_flat=take(layout.source_eulers_flat),
        row_class_flat=take(layout.row_class_flat).astype(np.int32),
    )


def drop_local_layout_classes(layout: LocalHypothesisLayout, empty_classes) -> LocalHypothesisLayout:
    """A class-expanded layout without the rows of ``empty_classes``.

    RELION leaves a class with ``pdf_class == 0`` out of the expectation (acc_ml_optimiser_impl.h:1069,
    :1617, :2378), also under local searches, whose weights carry no class prior
    (``pdf_orientation = directions_prior * psi_prior``): the class is not scored, so it cannot win an
    image, enter a per-particle minimum or return. Every image keeps its rows of the other classes.
    """

    empty = np.unique(np.asarray(list(empty_classes), dtype=np.int64))
    if empty.size == 0:
        return layout
    if layout.n_classes == 1:
        raise ValueError("only a class-expanded layout has classes to drop")
    if empty.size >= layout.n_classes or empty.min() < 0 or empty.max() >= layout.n_classes:
        raise ValueError(f"cannot drop classes {empty.tolist()} of {layout.n_classes}")
    offsets = np.asarray(layout.rotation_offsets, dtype=np.int64)
    counts = np.diff(offsets)
    row_image = np.repeat(np.arange(counts.shape[0], dtype=np.int64), counts)
    keep = ~np.isin(np.asarray(layout.row_class_flat, dtype=np.int64), empty)
    new_counts = np.bincount(row_image[keep], minlength=counts.shape[0])

    def take(values):
        return None if values is None else np.asarray(values)[keep]

    new_offsets = np.zeros(counts.shape[0] + 1, dtype=np.int64)
    new_offsets[1:] = np.cumsum(new_counts, dtype=np.int64)
    return replace(
        layout,
        rotation_offsets=new_offsets,
        rotation_ids_flat=take(layout.rotation_ids_flat),
        rotations_flat=take(layout.rotations_flat),
        rotation_log_priors_flat=take(layout.rotation_log_priors_flat),
        rotation_counts=new_counts.astype(np.asarray(layout.rotation_counts).dtype),
        rotation_posterior_ids_flat=take(layout.rotation_posterior_ids_flat),
        sample_mask_bits=take(layout.sample_mask_bits),
        mstep_rotations_flat=take(layout.mstep_rotations_flat),
        source_eulers_flat=take(layout.source_eulers_flat),
        row_class_flat=take(layout.row_class_flat).astype(np.int32),
    )


def build_local_adaptive_pass2_hypothesis_layout(
    parent_layout: LocalHypothesisLayout,
    significant_sample_indices,
    parent_healpix_order: int,
    *,
    oversampling_order: int,
    random_perturbation: float = 0.0,
    translation_step: float | None = None,
    dtype: np.dtype = np.float32,
    symmetry: str | None = None,
) -> LocalHypothesisLayout:
    """Expand local adaptive parent support while preserving significant pairs.

    RELION's adaptive local pass 2 first expands significant coarse orientation
    parents, then only scores fine translation children for coarse
    ``(orientation, translation)`` pairs that survived pass 1. ``parent_layout``
    carries the image-specific local Gaussian priors from that coarse pass.

    A class-expanded parent layout (``n_classes > 1``) has significant samples
    ``(class * n_global_rotations + rotation) * n_trans + t``, so each class keeps
    its own surviving parents; the children carry their parent's class.
    """

    if oversampling_order <= 0:
        raise ValueError("oversampling_order must be positive for adaptive local pass 2")
    fine_healpix_order = parent_healpix_order + oversampling_order
    symmetry = canonicalize_rotational_symmetry(
        parent_layout.symmetry if symmetry is None else symmetry
    )
    if len(significant_sample_indices) != parent_layout.n_images:
        raise ValueError(
            "significant_sample_indices must have one entry per image; "
            f"got {len(significant_sample_indices)} for {parent_layout.n_images} images",
        )

    coarse_translations = np.asarray(parent_layout.translation_grid, dtype=dtype)
    n_coarse_trans = int(coarse_translations.shape[0])
    if translation_step is None:
        translation_step = infer_translation_step(coarse_translations)
    fine_translations, fine_translation_parent = get_oversampled_translation_grid(
        coarse_translations,
        float(translation_step),
        oversampling_order=oversampling_order,
    )
    fine_translations = np.asarray(fine_translations, dtype=dtype)
    fine_translation_parent = np.asarray(fine_translation_parent, dtype=np.int32)
    n_fine_trans = int(fine_translations.shape[0])

    n_real_global = int(parent_layout.n_global_rotations)
    # Class-expanded layouts number a (class, rotation) parent class * n_global + rotation.
    n_parent_global = parent_layout.n_classes * n_real_global
    parent_offsets = np.asarray(parent_layout.rotation_offsets, dtype=np.int64)
    parent_ids_flat = np.asarray(parent_layout.rotation_ids_flat, dtype=np.int64)
    if parent_layout.n_classes > 1:
        parent_ids_flat = np.asarray(parent_layout.row_class_flat, dtype=np.int64) * n_real_global + parent_ids_flat
    parent_log_prior_flat = np.asarray(parent_layout.rotation_log_priors_flat, dtype=dtype)
    parent_counts = np.diff(parent_offsets)
    if parent_layout.n_images and np.any(parent_counts == 0):
        image_idx = int(np.flatnonzero(parent_counts == 0)[0])
        raise ValueError(f"Image {image_idx} has no local parent rotations for adaptive pass 2")

    # Each image's parent rotations, flat and image-major: the sorted unique
    # rotations of its significant samples, or every local parent (in the
    # parent layout's order) when it has none. One pass over all images; the
    # per-image statements are unchanged.
    sig_arrays = [
        np.zeros(0, dtype=np.int64) if samples is None else np.asarray(samples, dtype=np.int64).reshape(-1)
        for samples in significant_sample_indices
    ]
    sig_counts = np.fromiter((a.size for a in sig_arrays), dtype=np.int64, count=parent_layout.n_images)
    full_images = sig_counts == 0
    sig_flat = np.concatenate(sig_arrays) if sig_arrays else np.zeros(0, dtype=np.int64)
    sig_image = np.repeat(np.arange(parent_layout.n_images, dtype=np.int64), sig_counts)
    sig_rot = sig_flat // n_coarse_trans
    sig_trans = sig_flat % n_coarse_trans
    key_scale = np.int64(max(n_parent_global, 1)) + np.int64(1)
    sparse_keys = np.unique(sig_image * key_scale + sig_rot)
    sparse_image = sparse_keys // key_scale
    sparse_rot = sparse_keys % key_scale
    full_parent_rows = np.flatnonzero(np.repeat(full_images, parent_counts))
    full_image = np.repeat(np.arange(parent_layout.n_images, dtype=np.int64), parent_counts)[full_parent_rows]
    unit_image = np.concatenate([sparse_image, full_image])
    unit_rot = np.concatenate([sparse_rot, parent_ids_flat[full_parent_rows]])
    order = np.argsort(unit_image, kind="stable")
    unit_image, unit_rot = unit_image[order], unit_rot[order]
    bad_rot = (unit_rot < 0) | (unit_rot >= n_parent_global)
    if np.any(bad_rot):
        image_idx = int(unit_image[np.flatnonzero(bad_rot)[0]])
        raise ValueError(f"Image {image_idx} has significant rotation ids outside the parent grid")

    # Each unit's log prior from its image's local parents (a later duplicate
    # id wins, as in the id lookup of tests/unit/test_refine_relion_mode.py).
    parent_image = np.repeat(np.arange(parent_layout.n_images, dtype=np.int64), parent_counts)
    parent_keys = parent_image * key_scale + parent_ids_flat
    parent_order = np.argsort(parent_keys, kind="stable")
    sorted_parent_keys = parent_keys[parent_order]
    unit_keys = unit_image * key_scale + unit_rot
    position = np.searchsorted(sorted_parent_keys, unit_keys, side="right") - 1
    matched = position >= 0
    matched[matched] = sorted_parent_keys[position[matched]] == unit_keys[matched]
    if not np.all(matched):
        image_idx = int(unit_image[np.flatnonzero(~matched)[0]])
        missing = unit_rot[(unit_image == image_idx) & ~matched]
        raise ValueError(
            f"Image {image_idx} has significant rotations outside its local parent support: {missing[:8].tolist()}"
        )
    unit_log_prior = parent_log_prior_flat[parent_order[position]]
    unit_class = (unit_rot // n_real_global).astype(np.int32)
    unit_rot = unit_rot % n_real_global

    # Every image's oversampled children in one call: each parent's children
    # depend on that parent only, so the rows are the per-image calls' rows.
    oversampled_rots, parent_map, oversampled_rot_indices, oversampled_mstep_rots, source_eulers = (
        compute_oversampled_rotation_grid(
            unit_rot,
            parent_healpix_order,
            oversampling_order=oversampling_order,
            random_perturbation=float(random_perturbation),
            return_rotation_indices=True,
            return_mstep_rotations=True,
            return_source_eulers=True,
            rotation_index_order="recovar",
            dtype=dtype,
            symmetry=symmetry,
        )
    )
    rotations_flat = np.asarray(oversampled_rots, dtype=dtype).reshape(-1, 3, 3)
    mstep_rotations_flat = np.asarray(oversampled_mstep_rots, dtype=dtype).reshape(-1, 3, 3)
    parent_map = np.asarray(parent_map, dtype=np.int64)
    rotation_ids_flat = np.asarray(oversampled_rot_indices, dtype=np.int32).astype(np.int64)
    posterior_ids_flat = unit_rot[parent_map].astype(np.int32)
    rotation_log_priors_flat = unit_log_prior[parent_map].astype(dtype, copy=False)
    child_image = unit_image[parent_map]
    counts = np.bincount(child_image, minlength=parent_layout.n_images).astype(np.int32)
    offsets = np.zeros(parent_layout.n_images + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts, dtype=np.int64)

    if np.all(full_images):
        # ``None`` is the exact-local engine's compact representation of full
        # per-rotation/per-translation support. Avoid materializing massive
        # all-ones masks for RELION full-parent local pass 2.
        sample_mask_bits = None if parent_layout.n_images else np.zeros((0, (n_fine_trans + 7) // 8), dtype=np.uint8)
    else:
        coarse_mask = np.repeat(full_images[unit_image][:, None], n_coarse_trans, axis=1)
        sig_unit = np.searchsorted(unit_keys, sig_image * key_scale + sig_rot)
        coarse_mask[sig_unit, sig_trans] = True
        sample_mask_bits = np.packbits(
            coarse_mask[parent_map][:, fine_translation_parent], axis=1, bitorder="little"
        )

    fine_metadata = build_local_search_grid_metadata(fine_healpix_order, symmetry=symmetry)
    return LocalHypothesisLayout(
        oversampled_rows=True,
        n_global_rotations=rotation_grid_size(parent_healpix_order, symmetry),
        n_pixels=int(fine_metadata["n_pixels"]),
        n_psi=int(fine_metadata["n_psi"]),
        rotation_offsets=offsets,
        rotation_ids_flat=rotation_ids_flat,
        rotations_flat=rotations_flat,
        source_eulers_flat=(
            np.empty((0, 3), dtype=np.float64)
            if not parent_layout.n_images
            else (None if source_eulers is None else np.asarray(source_eulers))
        ),
        rotation_log_priors_flat=rotation_log_priors_flat,
        rotation_counts=counts,
        translation_grid=fine_translations,
        translation_log_priors=np.asarray(parent_layout.translation_log_priors, dtype=dtype)[
            :, fine_translation_parent
        ],
        rotation_posterior_ids_flat=posterior_ids_flat,
        sample_mask_bits=sample_mask_bits,
        mstep_rotations_flat=mstep_rotations_flat,
        symmetry=symmetry,
        row_class_flat=None if parent_layout.n_classes == 1 else unit_class[parent_map],
        n_classes=parent_layout.n_classes,
    )


def _positions_in_sorted_unique_ids(sorted_ids: np.ndarray, query_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map query ids to row positions in a sorted unique id array."""

    sorted_np = np.asarray(sorted_ids, dtype=np.int64).reshape(-1)
    query_np = np.asarray(query_ids, dtype=np.int64).reshape(-1)
    if query_np.size == 0:
        return np.zeros(0, dtype=np.int64), np.ones(0, dtype=bool)
    pos = np.searchsorted(sorted_np, query_np)
    valid = pos < sorted_np.size
    matched = np.zeros(query_np.shape, dtype=bool)
    if np.any(valid):
        matched[valid] = sorted_np[pos[valid]] == query_np[valid]
    return pos.astype(np.int64, copy=False), matched


def _fine_translation_log_prior(
    translation_log_prior: np.ndarray | None,
    fine_translation_parent: np.ndarray,
    n_images: int,
    n_fine_translations: int,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    if translation_log_prior is None:
        return np.zeros((n_images, n_fine_translations), dtype=dtype)
    translation_log_prior_np = np.asarray(translation_log_prior, dtype=dtype)
    if translation_log_prior_np.ndim == 1:
        fine = translation_log_prior_np[fine_translation_parent]
        return np.broadcast_to(fine[None, :], (n_images, n_fine_translations)).astype(dtype, copy=False)
    if translation_log_prior_np.ndim == 2:
        if translation_log_prior_np.shape[0] != n_images:
            raise ValueError(
                "translation_log_prior must have one row per image when 2D; "
                f"got {translation_log_prior_np.shape[0]} rows for {n_images} images",
            )
        return translation_log_prior_np[:, fine_translation_parent].astype(dtype, copy=False)
    raise ValueError(f"translation_log_prior must be 1D or 2D, got {translation_log_prior_np.ndim} dimensions")


def _pass2_translation_log_prior(
    translation_log_prior: np.ndarray | None,
    fine_translation_log_prior: np.ndarray | None,
    fine_translation_parent: np.ndarray,
    n_images: int,
    n_fine_translations: int,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    if fine_translation_log_prior is None:
        return _fine_translation_log_prior(
            translation_log_prior,
            fine_translation_parent,
            n_images,
            n_fine_translations,
            dtype=dtype,
        )
    if translation_log_prior is not None:
        raise ValueError("translation_log_prior and fine_translation_log_prior are mutually exclusive")

    prior_np = np.asarray(fine_translation_log_prior, dtype=dtype)
    if prior_np.ndim == 1:
        if prior_np.shape[0] != n_fine_translations:
            raise ValueError(
                "fine_translation_log_prior must have one value per fine translation; "
                f"got {prior_np.shape[0]} values for {n_fine_translations} translations",
            )
        return np.broadcast_to(prior_np[None, :], (n_images, n_fine_translations)).astype(dtype, copy=False)
    if prior_np.ndim == 2:
        if prior_np.shape != (n_images, n_fine_translations):
            raise ValueError(
                f"fine_translation_log_prior must have shape ({n_images}, {n_fine_translations}); got {prior_np.shape}",
            )
        return prior_np.astype(dtype, copy=False)
    raise ValueError(f"fine_translation_log_prior must be 1D or 2D, got {prior_np.ndim} dimensions")


def build_pass2_hypothesis_layout(
    significant_sample_indices,
    n_coarse_rotations: int,
    n_coarse_translations: int,
    nside_level: int,
    translations: np.ndarray,
    *,
    oversampling_order: int,
    translation_step: float | None = None,
    rotation_log_prior: np.ndarray | None = None,
    translation_log_prior: np.ndarray | None = None,
    fine_translation_log_prior: np.ndarray | None = None,
    random_perturbation: float = 0.0,
    rotation_index_order: str = "recovar",
    allow_empty: bool = False,
    dtype: np.dtype = np.float32,
    symmetry: str = "C1",
) -> LocalHypothesisLayout:
    """Build exact-local layout for RELION adaptive pass-2 hypotheses.

    Pass 2 is not a Gaussian local search around one previous best pose. RELION
    oversamples the coarse ``(rotation, translation)`` samples that survived
    pass 1. The exact local engine can score the same structure if each image
    carries its own oversampled rotations plus a sparse ``(R, T)`` mask.
    """

    symmetry = canonicalize_rotational_symmetry(symmetry)
    dtype = np.dtype(dtype)
    translations_np = np.asarray(translations, dtype=dtype)
    if translation_step is None:
        translation_step = infer_translation_step(translations_np)
    fine_translations, fine_translation_parent = get_oversampled_translation_grid(
        translations_np,
        float(translation_step),
        oversampling_order=oversampling_order,
    )
    fine_translations = np.asarray(fine_translations, dtype=dtype)
    fine_translation_parent = np.asarray(fine_translation_parent, dtype=np.int32)
    n_fine_translations = int(fine_translations.shape[0])
    n_images = len(significant_sample_indices)
    rotation_log_prior_np = None if rotation_log_prior is None else np.asarray(rotation_log_prior, dtype=dtype)

    coarse_rows = []
    for image_idx, sig_samples in enumerate(significant_sample_indices):
        if sig_samples is None:
            unique_rot = np.arange(n_coarse_rotations, dtype=np.int32)
            coarse_rot = unique_rot
            coarse_trans = None
            use_full_candidate_mask = True
        else:
            sig_samples = significant_sample_ids(
                sig_samples, int(n_coarse_rotations) * int(n_coarse_translations),
            )
            if sig_samples.size == 0:
                if not allow_empty:
                    raise ValueError(f"Image {image_idx} has no significant coarse samples for sparse pass 2")
                unique_rot = np.zeros(1, dtype=np.int64)
                coarse_rot = unique_rot
                coarse_trans = np.zeros(0, dtype=np.int64)
            else:
                coarse_rot = sig_samples // int(n_coarse_translations)
                coarse_trans = sig_samples % int(n_coarse_translations)
                unique_rot = np.unique(coarse_rot).astype(np.int64, copy=False)
            use_full_candidate_mask = False

        if np.any(unique_rot < 0) or np.any(unique_rot >= int(n_coarse_rotations)):
            raise ValueError(f"Image {image_idx} has significant rotation ids outside the coarse grid")

        coarse_rows.append((unique_rot, coarse_rot, coarse_trans, use_full_candidate_mask))

    # Child orientations depend on the coarse sample and iteration perturbation,
    # not on the image. Generate only the requested union once, then gather each
    # image's rows in the same parent/child order as independent generation.
    shared_parent_ids = (
        np.unique(np.concatenate([row[0] for row in coarse_rows]))
        if coarse_rows else np.zeros(0, dtype=np.int64)
    )
    shared_rotations, shared_parent_map, shared_rotation_ids, shared_eulers = (
        get_oversampled_rotation_grid_from_samples(
            shared_parent_ids,
            int(nside_level),
            oversampling_order=oversampling_order,
            random_perturbation=random_perturbation,
            return_rotation_indices=True,
            return_source_eulers=True,
            rotation_index_order=rotation_index_order,
            dtype=dtype,
            symmetry=symmetry,
        )
    )
    children_per_parent = 8 ** int(oversampling_order)
    if not np.array_equal(
        shared_parent_map,
        np.repeat(np.arange(shared_parent_ids.size), children_per_parent),
    ):
        raise RuntimeError("Pass-2 oversampling must retain contiguous children per parent")
    child_offsets = np.arange(children_per_parent, dtype=np.int64)

    # Allocate the final layout once; retaining every image's arrays until
    # concatenation otherwise doubles the largest host allocations.
    counts = np.asarray([row[0].size * children_per_parent for row in coarse_rows], dtype=np.int32)
    offsets = np.zeros(n_images + 1, dtype=np.int64)
    np.cumsum(counts, dtype=np.int64, out=offsets[1:])
    n_rows = int(offsets[-1])
    rotations_flat = np.empty_like(shared_rotations, shape=(n_rows, 3, 3), dtype=dtype)
    rotation_ids_flat = np.empty(n_rows, dtype=np.int64)
    posterior_ids_flat = np.empty(n_rows, dtype=np.int32)
    rotation_log_priors_flat = np.empty(n_rows, dtype=dtype)
    sample_mask_bits = np.empty((n_rows, (n_fine_translations + 7) // 8), dtype=np.uint8)
    source_eulers_flat = np.empty_like(shared_eulers, shape=(n_rows, 3)) if shared_eulers is not None else None
    if not n_images:
        source_eulers_flat = np.empty((0, 3), dtype=np.float64)

    for image_idx, (unique_rot, coarse_rot, coarse_trans, use_full_candidate_mask) in enumerate(coarse_rows):
        shared_positions = np.searchsorted(shared_parent_ids, unique_rot)
        rows = (shared_positions[:, None] * children_per_parent + child_offsets).reshape(-1)
        oversampled_rots = np.asarray(shared_rotations[rows], dtype=dtype)
        oversampled_rot_indices = np.asarray(shared_rotation_ids[rows], dtype=np.int64)
        parent_map = np.repeat(np.arange(unique_rot.size, dtype=np.int32), children_per_parent)
        coarse_parent_ids = unique_rot[parent_map].astype(np.int32, copy=False)

        if rotation_log_prior_np is None:
            local_rotation_log_prior = np.zeros(oversampled_rots.shape[0], dtype=dtype)
        else:
            local_rotation_log_prior = rotation_log_prior_np[unique_rot][parent_map].astype(dtype, copy=False)

        if use_full_candidate_mask:
            sample_mask = np.ones((oversampled_rots.shape[0], n_fine_translations), dtype=bool)
        else:
            sample_mask = np.zeros((oversampled_rots.shape[0], n_fine_translations), dtype=bool)
            if coarse_trans.size:
                # Vectorized replacement of the inner-image Python loop over
                # unique_rot. Build a (n_unique_rot, n_coarse_trans) mask of
                # significant (rot, trans) pairs, then expand by parent_map
                # and fine_translation_parent. At 50k/256 K=1 this cuts pass2
                # layout-build time from ~10 s/iter to <1 s.
                significance_mask_coarse = np.zeros(
                    (unique_rot.shape[0], int(n_coarse_translations)),
                    dtype=bool,
                )
                local_idx_per_sample, matched_coarse_rot = _positions_in_sorted_unique_ids(unique_rot, coarse_rot)
                if not np.all(matched_coarse_rot):
                    raise ValueError(f"Image {image_idx} has significant samples outside unique coarse rotations")
                significance_mask_coarse[local_idx_per_sample, coarse_trans] = True
                sample_mask = significance_mask_coarse[parent_map][:, fine_translation_parent]

        if not np.any(sample_mask) and not allow_empty:
            raise ValueError(f"Image {image_idx} has no valid sparse pass-2 candidates after oversampling")

        target = slice(offsets[image_idx], offsets[image_idx + 1])
        rotations_flat[target] = oversampled_rots
        rotation_ids_flat[target] = oversampled_rot_indices
        posterior_ids_flat[target] = coarse_parent_ids
        rotation_log_priors_flat[target] = local_rotation_log_prior
        sample_mask_bits[target] = np.packbits(sample_mask, axis=1, bitorder="little")
        if source_eulers_flat is not None:
            source_eulers_flat[target] = shared_eulers[rows]

    n_pixels = (12 * (2 ** int(nside_level)) ** 2 if symmetry == "C1" else int(
        build_local_search_grid_metadata(int(nside_level), symmetry=symmetry)["n_pixels"]
    ))

    return LocalHypothesisLayout(
        oversampled_rows=True,
        n_global_rotations=int(n_coarse_rotations),
        n_pixels=int(n_pixels),
        n_psi=int(rotation_grid_n_in_planes(int(nside_level))),
        rotation_offsets=offsets,
        rotation_ids_flat=rotation_ids_flat,
        rotations_flat=rotations_flat,
        source_eulers_flat=source_eulers_flat,
        rotation_log_priors_flat=rotation_log_priors_flat,
        rotation_counts=counts,
        translation_grid=fine_translations,
        translation_log_priors=_pass2_translation_log_prior(
            translation_log_prior,
            fine_translation_log_prior,
            fine_translation_parent,
            n_images,
            n_fine_translations,
            dtype=dtype,
        ),
        rotation_posterior_ids_flat=posterior_ids_flat,
        sample_mask_bits=sample_mask_bits,
        symmetry=symmetry,
    )
