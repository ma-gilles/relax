"""Halves whose optics groups differ in pixel size or box (RELION S3b).

Images of different shapes cannot share one dataset or one compiled program, so a
half is split into *shape classes* (optics groups with the same box and pixel size).
Half scoring runs the unchanged single-shape route once per class on that class's
dataset, with RELION's rules for a group on another grid (:mod:`relax.helpers.optics_scale`):

- projection/backprojection matrices divided by the class scale ``s_g``
  (``applyScaleDifference``); poses stay unscaled everywhere else;
- translations converted from reference pixels to class pixels;
- image sizes remapped (``updateImageSizeAndResolutionPointers``), while the
  backprojector keeps the reference model size;
- noise read from the reference-shell spectrum (E-step remap) and the class's noise
  sums added back onto the reference shells (M-step remap).

Per-image outputs return to the half's image order by index, and additive sums add.
A half with one shape class is a plain dataset and never reaches this module.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from relax.helpers import optics_scale
from relax.helpers.orientation_priors import (
    make_relion_translation_log_prior,
    relion_half_translation_prior_inputs,
    relion_translation_search_base,
)
from relax.helpers.resolution import relion_coarse_image_size


@dataclasses.dataclass(frozen=True)
class ShapeClass:
    """The images of one half that share a box and a pixel size."""

    dataset: object
    image_indices: np.ndarray  # positions in the half's image order
    box_size: int
    pixel_size: float
    scale: float  # s_g = box_g angpix_g / (ori angpix_ref)
    translation_factor: float  # reference pixels -> class pixels: angpix_ref / angpix_g


@dataclasses.dataclass(frozen=True, kw_only=True)
class ShapeTranslations:
    """Pre-shifts and priors computed in one shape class's own pixels."""

    search_base: object | None
    local_prior_center: object | None
    engine_prior_center: object
    log_prior: object | None


@dataclasses.dataclass(frozen=True, kw_only=True)
class OpticsSpec:
    """Optics operands for a half or its selected shape class.

    ``class_translations`` follows ``MultiShapeHalf.classes`` in order. After
    selection, only projection scale and reference support remain class-specific.
    """

    noise_radial_k: object | None = None
    coarse_sizing: tuple[float, float | None] | None = None
    class_translations: tuple[ShapeTranslations, ...] | None = None
    projection_scale: float = 1.0
    reference_current_size: int | None = None


@dataclasses.dataclass(frozen=True)
class _RowLayout:
    rows: np.ndarray

    def original_image_indices_for_local(self, local):
        return self.rows[np.asarray(local)]


class MultiShapeHalf:
    """One half as several shape classes, presenting the reference geometry.

    ``image_shape``, ``volume_shape`` and ``voxel_size`` are the reference model's
    (optics group 0's grid); ``n_units`` counts every image of the half. Anything
    that needs the images themselves must go through ``classes``.
    """

    def __init__(self, classes, *, image_shape, volume_shape, voxel_size, rows=None):
        self.classes = tuple(classes)
        self.image_shape = tuple(int(size) for size in image_shape)
        self.volume_shape = tuple(int(size) for size in volume_shape)
        self.voxel_size = float(voxel_size)
        self.grid_size = self.image_shape[0]
        self.n_units = self.n_images = int(sum(c.image_indices.size for c in self.classes))
        # Class3D's second accumulator is an empty half: no shape classes.
        order = np.concatenate([c.image_indices for c in self.classes] + [np.zeros(0, dtype=np.int64)])
        if not np.array_equal(np.sort(order), np.arange(self.n_units)):
            raise ValueError("shape classes must partition the half's images")
        # Particle-STAR row of each image, as a loaded dataset's index layout reports it.
        self._index_layout = None if rows is None else _RowLayout(np.asarray(rows, dtype=np.int64))

    def __getattr__(self, name):
        raise AttributeError(f"a half with several image shapes has no single {name!r}; use its shape classes")


def image_translation_factors(dataset):
    """Each image's reference-pixel -> own-pixel factor (``ShapeClass.translation_factor``), or None for
    a dataset on one grid, whose images are all on the reference pixel."""

    if not isinstance(dataset, MultiShapeHalf):
        return None
    factors = np.ones(dataset.n_units, dtype=np.float64)
    for shape_class in dataset.classes:
        factors[shape_class.image_indices] = shape_class.translation_factor
    return factors


def prepare_optics(
    dataset,
    *,
    noise_radial,
    previous_translations,
    sigma_offset_angstrom,
    base_translations,
    current_translations,
    with_log_prior: bool,
    zero_cold_center: bool,
    coarse_step_deg,
    particle_diameter_ang,
    dtype,
) -> OpticsSpec:
    """Prepare half scoring operands for its image shape classes.

    Stored offsets are converted before rounding in each class's pixels
    (RELION ``my_old_offset.selfROUND()``, ml_optimiser.cpp:6085). Scaling an
    already-rounded reference offset gives different pre-shifts and priors.
    A single-shape half needs no additional arrays or translation computation.
    """

    if not isinstance(dataset, MultiShapeHalf):
        return OpticsSpec()

    noise_radial = np.asarray(noise_radial, dtype=np.float64)
    coarse_sizing = None if coarse_step_deg is None else (float(coarse_step_deg), particle_diameter_ang)
    translations = []
    for shape_class in dataset.classes:
        factor = shape_class.translation_factor
        previous = (
            None
            if previous_translations is None
            else np.asarray(previous_translations, dtype=np.float64)[shape_class.image_indices] * factor
        )
        inputs = relion_half_translation_prior_inputs(
            previous,
            voxel_size=shape_class.pixel_size,
            base_translations=None if base_translations is None else np.asarray(base_translations) * factor,
            current_translations=np.asarray(current_translations) * factor,
            dtype=dtype,
        )
        search_base = relion_translation_search_base(previous, dtype=dtype)
        log_prior = None
        if with_log_prior:
            center = inputs.prior_center
            if center is None and zero_cold_center:
                center = np.zeros(2, dtype=dtype)
            log_prior = make_relion_translation_log_prior(
                inputs.prior_translations,
                shape_class.pixel_size,
                sigma_offset_angstrom,
                center,
                offset_range_pixels=None,
                dtype=dtype,
            )
        translations.append(
            ShapeTranslations(
                search_base=search_base,
                local_prior_center=inputs.local_prior_center,
                engine_prior_center=inputs.engine_prior_center,
                log_prior=log_prior,
            )
        )
    return OpticsSpec(
        noise_radial_k=noise_radial,
        coarse_sizing=coarse_sizing,
        class_translations=tuple(translations),
    )


def make_shape_classes(datasets_and_indices, *, ref_box, ref_pixel):
    """``ShapeClass`` records from ``(dataset, half positions)`` pairs.

    A class at scale ``s >= sqrt(2)`` is refused. There, RELION's fine kernels project
    a moved pixel inside the model sphere for the image rows beyond it
    (:func:`relax.helpers.projection.relion_kernel_zero_rows`), which relax does not reproduce.
    Listing the optics group with the largest box x pixel size first avoids it.
    """

    classes = []
    for dataset, indices in datasets_and_indices:
        box = int(dataset.image_shape[0])
        pixel = float(dataset.voxel_size)
        scale = optics_scale.scale_difference(box, pixel, ref_box, ref_pixel)
        if scale >= math.sqrt(2.0):
            raise NotImplementedError(
                f"an optics group of {box} px at {pixel} A spans {scale:.3f} times the reference field of view "
                f"({ref_box} px at {ref_pixel} A). From sqrt(2) on, RELION's fine and weighted-sum kernels project "
                "the pixel (maxR, i) for every image row i beyond the model radius, and shift the image at that "
                "pixel too (acc/cuda/cuda_kernels/diff2.cuh:494-530, wavg.cuh:81-86), a RELION defect relax will "
                "reproduce with the S4.2 scorer changes. Until then, list the optics group with the largest "
                "box x pixel size first."
            )
        classes.append(
            ShapeClass(
                dataset=dataset,
                image_indices=np.asarray(indices, dtype=np.int64),
                box_size=box,
                pixel_size=pixel,
                scale=scale,
                translation_factor=float(ref_pixel) / pixel,
            )
        )
    return classes


def optics_shape_class_rows(particles_star):
    """Particle rows per image shape when optics groups differ in box or pixel size.

    ``None`` when every optics group has one box and pixel size (the single-dataset
    path). Otherwise one row array per (box, pixel size), in optics-table order, so
    the first class holds optics group 1, whose grid is RELION's model grid.
    """
    import starfile

    star = starfile.read(particles_star)
    optics = star.get("optics") if isinstance(star, dict) else None
    if optics is None or not {"rlnImageSize", "rlnImagePixelSize"}.issubset(optics.columns):
        return None
    shapes = list(zip(np.asarray(optics["rlnImageSize"], dtype=np.int64).tolist(),
                      np.asarray(optics["rlnImagePixelSize"], dtype=np.float64).tolist()))
    unique_shapes = list(dict.fromkeys(shapes))
    if len(unique_shapes) == 1:
        return None
    labels = np.asarray(optics["rlnOpticsGroup"], dtype=np.int64)
    if not np.array_equal(labels, np.arange(1, labels.size + 1)):
        raise ValueError(f"optics groups must be numbered 1..{labels.size} in table order, got {labels.tolist()}")
    shape_of_label = np.asarray([unique_shapes.index(shape) for shape in shapes], dtype=np.int64)
    particle_shapes = shape_of_label[np.asarray(star["particles"]["rlnOpticsGroup"], dtype=np.int64) - 1]
    return [np.flatnonzero(particle_shapes == c) for c in range(len(unique_shapes))]


class MultiShapeDataset:
    """All particles as one loaded dataset per shape class.

    ``rows[c]`` are the particle-STAR rows held, in order, by ``datasets[c]``; together
    they cover every row once. The reference geometry (``image_shape``,
    ``volume_shape``, ``voxel_size``) is that of ``datasets[0]``, the class of the first
    optics group (RELION's model ``ori_size`` and pixel size, ml_model.cpp:1090-1091).
    """

    def __init__(self, datasets, rows):
        self.datasets = tuple(datasets)
        self.rows = tuple(np.asarray(r, dtype=np.int64) for r in rows)
        if len(self.datasets) != len(self.rows) or len(self.datasets) < 2:
            raise ValueError("a multi-shape dataset needs one row list per class and at least two classes")
        self.n_units = self.n_images = int(sum(r.size for r in self.rows))
        self._class_of_row = np.full(self.n_units, -1, dtype=np.int64)
        self._local_of_row = np.full(self.n_units, -1, dtype=np.int64)
        for c, (dataset, rows_c) in enumerate(zip(self.datasets, self.rows)):
            if int(dataset.n_units) != rows_c.size:
                raise ValueError(f"class {c} holds {dataset.n_units} images for {rows_c.size} rows")
            self._class_of_row[rows_c] = c
            self._local_of_row[rows_c] = np.arange(rows_c.size)
        if np.any(self._class_of_row < 0):
            raise ValueError("shape classes must cover every particle row once")
        ref = self.datasets[0]
        self.image_shape = tuple(int(size) for size in ref.image_shape)
        self.volume_shape = tuple(int(size) for size in ref.volume_shape)
        self.voxel_size = float(ref.voxel_size)
        self.grid_size = self.image_shape[0]

    def __getattr__(self, name):
        raise AttributeError(f"a dataset with several image shapes has no single {name!r}; use its classes")

    def subset(self, rows):
        """The ``MultiShapeHalf`` of these particle rows, in this order."""

        rows = np.asarray(rows, dtype=np.int64)
        pairs = []
        for c, dataset in enumerate(self.datasets):
            positions = np.flatnonzero(self._class_of_row[rows] == c)
            if positions.size:
                pairs.append((dataset.subset(self._local_of_row[rows[positions]]), positions))
        classes = make_shape_classes(pairs, ref_box=self.grid_size, ref_pixel=self.voxel_size)
        return MultiShapeHalf(
            classes,
            image_shape=self.image_shape,
            volume_shape=self.volume_shape,
            voxel_size=self.voxel_size,
            rows=rows,
        )

    def iter_images(self, rows, *, batch_size):
        """``(row, real-space image)`` in the given row order, each on its own class's grid."""

        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        for start in range(0, rows.size, batch_size):
            chunk = rows[start : start + batch_size]
            images = {}
            for c in np.unique(self._class_of_row[chunk]):
                local = self._local_of_row[chunk[self._class_of_row[chunk] == c]]
                for batch_images, _particles, local_indices in self.datasets[c].image_source.iter_batches(
                    batch_size=local.size, batch_mode="images", subset_indices=local
                ):
                    for image, local_row in zip(np.asarray(batch_images), np.asarray(local_indices).reshape(-1)):
                        images[int(self.rows[c][int(local_row)])] = image
            for row in chunk:
                yield int(row), images.pop(int(row))


# Keywords of the half scoring functions whose leading axis is the half's images.
PER_IMAGE_KWARGS = (
    "previous_best_rotation_eulers_k",
    "trans_prior_center",
    "trans_prior_center_for_engine",
    "image_corrections_k",
    "scale_corrections_k",
    "translation_search_base",
    "group_ids_k",
    "optics_group_ids_k",
    "image_seed_classes",
)
# Keywords that may be per image when two-dimensional (image x hypothesis).
PER_IMAGE_IF_2D_KWARGS = ("translation_log_prior", "rotation_log_prior_k")
# Per-class keywords ([K, hypothesis]) that are per image when three-dimensional
# ([K, image, hypothesis]).
PER_CLASS_IMAGE_IF_3D_KWARGS = ("class_rotation_log_prior_k",)
# Keywords in reference pixels.
TRANSLATION_KWARGS = (
    "current_translations",
    "base_translations",
    "trans_prior_center",
    "trans_prior_center_for_engine",
    "translation_search_base",
    "replay_prior_translations",
)
# Image-side Fourier sizes in reference pixels (None means the full box).
IMAGE_SIZE_KWARGS = (
    "cs_for_engine",
    "local_pass1_current_size",
    "firstiter_coarse_current_size",
    "firstiter_fine_current_size",
)
# Pass-1 sizes; with ``coarse_sizing`` they come from RELION's adaptive formula.
COARSE_SIZE_KWARGS = ("local_pass1_current_size", "firstiter_coarse_current_size")


def _class_images(value, shape_class: ShapeClass, n_half: int, name: str):
    """The class's rows of a per-image array (the half's images on its leading axis)."""

    array = np.asarray(value)
    if array.ndim == 0 or array.shape[0] != n_half:
        raise ValueError(f"{name} must have the half's {n_half} images on its leading axis")
    return array[shape_class.image_indices]


def class_kwargs(kwargs, shape_class: ShapeClass, n_half: int) -> dict:
    """One shape class's keywords: its images, class pixels and class Fourier sizes."""

    out = dict(kwargs)
    coarse_sizing = out.pop("coarse_sizing", None)
    index = shape_class.image_indices
    for name in PER_IMAGE_KWARGS + PER_IMAGE_IF_2D_KWARGS:
        value = out.get(name)
        if value is None:
            continue
        if name in PER_IMAGE_IF_2D_KWARGS and np.ndim(value) != 2:
            continue
        out[name] = _class_images(value, shape_class, n_half, name)
    for name in PER_CLASS_IMAGE_IF_3D_KWARGS:
        value = out.get(name)
        if value is not None and np.ndim(value) == 3:
            out[name] = np.moveaxis(_class_images(np.moveaxis(np.asarray(value), 1, 0), shape_class, n_half, name), 0, 1)
    for name in TRANSLATION_KWARGS:
        if out.get(name) is not None:
            out[name] = np.asarray(out[name]) * shape_class.translation_factor
    translation_step = out.get("translation_step")
    if translation_step is not None and shape_class.translation_factor != 1.0:
        # The oversampled translation grid is built from the step (RELION samples offsets in
        # Angstrom and converts them with the image's pixel size, getTranslationsInPixel).
        out["translation_step"] = float(translation_step) * shape_class.translation_factor
    for name in IMAGE_SIZE_KWARGS:
        if out.get(name) is not None:
            out[name] = optics_scale.group_current_size(out[name], shape_class.box_size, shape_class.scale)
    if coarse_sizing is not None:
        for name in COARSE_SIZE_KWARGS:
            if kwargs.get(name) is not None:
                out[name] = _class_coarse_size(coarse_sizing, shape_class, out.get("cs_for_engine"))
    # The backprojector stays on the reference model grid; the class's image window
    # for the M-step is the remapped size.
    reference_size = out.get("model_current_size_for_engine")
    if reference_size is None:
        reference_size = kwargs.get("cs_for_engine")
    if reference_size is not None:
        out["model_current_size_for_engine"] = optics_scale.group_current_size(
            reference_size, shape_class.box_size, shape_class.scale
        )
    half = kwargs.get("experiment_dataset")
    if reference_size is None and half is not None and (
        shape_class.box_size != int(half.image_shape[0]) or shape_class.scale != 1.0
    ):
        # The full reference box, stated explicitly: the class's own "full box" is another size.
        reference_size = int(half.image_shape[0])
    out["reference_current_size"] = reference_size
    out["projection_scale"] = shape_class.scale
    out["experiment_dataset"] = (
        shape_class.dataset if half is None else _engine_dataset(shape_class, half.volume_shape)
    )
    return out


def reconstruction_image_radius(reference_current_size, scale: float):
    """The M-step's image-space radius for images on another grid.

    RELION's backprojector keeps rotated samples inside the reference model's
    ``r_max = current_size / 2`` (BackProjector::backproject, ``max_r2``); an image pixel
    at radius ``|k|`` lands at reference radius ``|k| / s``, so the image-side bound is
    ``r_max * s``. None keeps the engines' own bound (one grid).
    """
    if reference_current_size is None:
        return None
    return float(int(reference_current_size) // 2) * float(scale)


def reference_grid_kwargs(reference_current_size, scale: float) -> dict:
    """Engine kwargs of images on another grid: the reference-model M-step size and image radius."""

    if reference_current_size is None:
        return {}
    return {
        "reconstruction_volume_current_size": int(reference_current_size),
        "reconstruction_image_radius": reconstruction_image_radius(reference_current_size, scale),
    }


def engine_projection_inputs(dataset, *, scale, reference_current_size, rotations):
    """``(projection matrices, engine kwargs)`` for images on ``dataset``'s grid and optics.

    ``rotations`` maps names to pose matrices (None entries stay None); each becomes its
    projection matrix (``applyScaleDifference`` for ``scale``, ``applyAnisoMag`` for the
    dataset's magnification). The kwargs are :func:`reference_grid_kwargs`. Every
    adaptive engine call of a shape class (half scoring and VDAM) takes its geometry here.
    """

    from relax.relion.optics_aberrations import dataset_projection_magnification, projection_rotations

    magnification = dataset_projection_magnification(dataset)
    projected = {name: projection_rotations(value, scale, magnification) for name, value in rotations.items()}
    return projected, reference_grid_kwargs(reference_current_size, scale)


# Engine keywords (run_dense_k_class_em_adaptive) whose leading axis is the images.
ENGINE_PER_IMAGE_KWARGS = (
    "image_corrections",
    "scale_corrections",
    "group_ids",
    "optics_group_ids",
    "reconstruction_group_ids",
    "image_seed_classes",
)
# Engine keywords that are per image when two-dimensional (image x hypothesis).
ENGINE_PER_IMAGE_IF_2D_KWARGS = ("translation_log_prior", "coarse_translation_log_prior", "rotation_log_prior")
# Engine keywords in reference pixels whose RELION value is rounded in the image's own
# pixels (ml_optimiser.cpp:6085); a caller rebuilds them per class (see
# iteration_loop._class_translation_kwargs) instead of scaling the reference values.
ENGINE_ROUNDED_TRANSLATION_KWARGS = ("image_pre_shifts", "translation_prior_centers")


@dataclasses.dataclass(frozen=True)
class ShapeClassEngineInputs:
    """One shape class's arguments for ``run_dense_k_class_em_adaptive``."""

    dataset: object
    noise_variance: np.ndarray
    coarse_rotations: np.ndarray
    fine_rotations: np.ndarray
    fine_mstep_rotations: np.ndarray | None
    coarse_translations: np.ndarray
    fine_translations: np.ndarray
    coarse_current_size: int | None
    fine_current_size: int | None
    engine_kwargs: dict


def shape_class_engine_inputs(
    shape_class: ShapeClass,
    half: MultiShapeHalf,
    *,
    noise_radial,
    coarse_rotations,
    fine_rotations,
    fine_mstep_rotations,
    coarse_translations,
    fine_translations,
    coarse_current_size,
    fine_current_size,
    reference_current_size,
    engine_kwargs,
    coarse_sizing=None,
) -> ShapeClassEngineInputs:
    """A shape class's adaptive-engine call from the half's reference-grid arguments.

    Rotations become the class's projection matrices, translations go to class pixels,
    the pass-1/pass-2 sizes are remapped (``coarse_sizing``, as in :func:`class_kwargs`,
    gives the class its own adaptive pass-1 size), the noise rows are read from the
    reference shells, per-image keywords keep the class's rows and the backprojector stays
    on the reference model grid. Parent maps and Euler overrides index the shared grids
    and pass through. Results merge with :func:`merge_k_class_engine_results`.
    """

    n_half = half.n_units
    kwargs = dict(engine_kwargs)
    for name in ENGINE_ROUNDED_TRANSLATION_KWARGS:
        if kwargs.get(name) is not None and np.any(np.asarray(kwargs[name]) != 0):
            raise ValueError(f"{name} must be rebuilt in each shape class's pixels, not scaled")
    for name in ENGINE_PER_IMAGE_KWARGS + ENGINE_PER_IMAGE_IF_2D_KWARGS:
        value = kwargs.get(name)
        if value is None or (name in ENGINE_PER_IMAGE_IF_2D_KWARGS and np.ndim(value) != 2):
            continue
        kwargs[name] = _class_images(value, shape_class, n_half, name)
    for name in ENGINE_ROUNDED_TRANSLATION_KWARGS:
        if kwargs.get(name) is not None:
            kwargs[name] = _class_images(kwargs[name], shape_class, n_half, name)
    sizes = class_kwargs(
        {
            "cs_for_engine": fine_current_size,
            "firstiter_coarse_current_size": coarse_current_size,
            "coarse_sizing": coarse_sizing,
        },
        shape_class,
        0,
    )
    if reference_current_size is None:
        # The backprojector's model size: the reference pass-2 size, else the full reference box.
        reference_current_size = int(half.image_shape[0]) if fine_current_size is None else int(fine_current_size)
    for name in ("current_size", "reconstruction_current_size"):
        if kwargs.get(name) is not None:
            kwargs[name] = optics_scale.group_current_size(kwargs[name], shape_class.box_size, shape_class.scale)
    dataset = _engine_dataset(shape_class, half.volume_shape)
    projected, grid_kwargs = engine_projection_inputs(
        dataset,
        scale=shape_class.scale,
        reference_current_size=reference_current_size,
        rotations={"coarse": coarse_rotations, "fine": fine_rotations, "mstep": fine_mstep_rotations},
    )
    kwargs.update(grid_kwargs)
    factor = shape_class.translation_factor
    return ShapeClassEngineInputs(
        dataset=dataset,
        noise_variance=class_noise_table(noise_radial, shape_class, int(half.image_shape[0])),
        coarse_rotations=projected["coarse"],
        fine_rotations=projected["fine"],
        fine_mstep_rotations=projected["mstep"],
        coarse_translations=np.asarray(coarse_translations) * factor,
        fine_translations=np.asarray(fine_translations) * factor,
        coarse_current_size=sizes["firstiter_coarse_current_size"],
        fine_current_size=sizes["cs_for_engine"],
        engine_kwargs=kwargs,
    )


class _ReferenceGridView:
    """A shape class's dataset as the engines see it: its own images on the reference volume grid.

    The engines read ``volume_shape`` as the grid of the reference they project and of the
    backprojector they fill (both on the model's ``ori_size``); every other attribute is the
    class dataset's.
    """

    def __init__(self, dataset, volume_shape):
        self._dataset = dataset
        self.volume_shape = tuple(int(size) for size in volume_shape)

    def __getattr__(self, name):
        return getattr(self._dataset, name)


def _engine_dataset(shape_class: ShapeClass, reference_volume_shape):
    dataset = shape_class.dataset
    if tuple(dataset.volume_shape) == tuple(int(size) for size in reference_volume_shape):
        return dataset
    return _ReferenceGridView(dataset, reference_volume_shape)


def class_adaptive_sizes(shape_class: ShapeClass, cs_for_engine, coarse_cs, coarse_sizing):
    """A class's (pass-2, pass-1) image sizes, exactly as ``class_kwargs`` gives them."""

    out = class_kwargs(
        {"cs_for_engine": cs_for_engine, "firstiter_coarse_current_size": coarse_cs, "coarse_sizing": coarse_sizing},
        shape_class,
        0,
    )
    return out["cs_for_engine"], out["firstiter_coarse_current_size"]


def _class_coarse_size(coarse_sizing, shape_class: ShapeClass, class_current_size):
    """A class's adaptive pass-1 size, None for its full box (ml_optimiser.cpp:5761-5777).

    ``2 CEIL(remap * pixel_ref * ori / coarse_resolution)`` is the reference formula at
    the class's own pixel size and box, clamped to the class's current size.
    """
    angular_step_deg, particle_diameter_ang = coarse_sizing
    size = relion_coarse_image_size(
        angular_step_deg,
        shape_class.pixel_size,
        shape_class.box_size,
        particle_diameter=particle_diameter_ang,
        current_size=class_current_size,
    )
    return size if size < shape_class.box_size else None


def class_noise_table(noise_radial_ref, shape_class: ShapeClass, ref_box: int):
    """The class's per-pixel noise rows, read from the reference-shell spectra.

    ``noise_radial_ref`` is ``[G, n_ref]`` in RECOVAR's native units on the reference
    grid (RELION sigma2 times ``ref_box**4``); the class grid's native unit is RELION
    sigma2 times ``box_g**4``.
    """

    from recovar.reconstruction import noise

    radial = np.atleast_2d(np.asarray(noise_radial_ref, dtype=np.float64))
    to_class = (float(shape_class.box_size) / float(ref_box)) ** 4
    n_shells = shape_class.box_size // 2 + 1
    shape = (shape_class.box_size, shape_class.box_size)
    beyond = optics_scale.reference_shell_of_group_shell(n_shells, shape_class.scale) >= radial.shape[-1]
    rows = []
    for row in radial:
        group_row = optics_scale.group_noise_from_reference(row, n_shells, shape_class.scale)
        # RELION gives these shells (beyond the reference Nyquist) zero weight; they lie
        # outside every scored window, so hold the last reference shell there instead of
        # a zero variance that would turn the unused pixels into inf/nan.
        group_row[beyond] = row[-1]
        rows.append(np.asarray(noise.make_radial_noise(group_row * to_class, shape)).reshape(-1))
    return np.stack(rows)


def noise_sums_to_reference(values, shape_class: ShapeClass, ref_box: int):
    """Class per-shell sums ``[G, n_g]`` in class native units onto ``[G, n_ref]`` reference shells."""

    values = np.atleast_2d(np.asarray(values, dtype=np.float64))
    to_reference = (float(ref_box) / float(shape_class.box_size)) ** 4
    n_ref = ref_box // 2 + 1
    return np.stack(
        [
            optics_scale.add_group_shells_to_reference(np.zeros(n_ref), row * to_reference, shape_class.scale)
            for row in values
        ]
    )


def place_by_index(parts, classes, n_half):
    """Per-image arrays of each class back into the half's image order."""

    parts = [None if part is None else np.asarray(part) for part in parts]
    present = [part for part in parts if part is not None]
    if not present:
        return None
    if len(present) != len(parts):
        raise ValueError("a per-image output is missing for some shape classes")
    out = np.empty((n_half,) + present[0].shape[1:], dtype=np.result_type(*present))
    for part, shape_class in zip(parts, classes):
        if part.shape[0] != shape_class.image_indices.size:
            raise ValueError("a shape class returned the wrong number of images")
        out[shape_class.image_indices] = part
    return out


def _to_reference_units(value, shape_class: ShapeClass, ref_box: int, power: int):
    """A class's backprojected sum in the reference class's native units.

    A class's native image Fourier values carry ``box_g**2`` and its noise ``box_g**4``
    (RELION's normalised values times those), so its data sum (image / noise) carries
    ``box_g**-2`` and its weight sum (1 / noise) ``box_g**-4``; RELION adds both in its own
    normalisation. Rescaled by ``(box_g / ref_box) ** power`` they add to the reference
    class's sums, as the noise sums do (``noise_sums_to_reference``).
    """
    if value is None or shape_class.box_size == ref_box:
        return value
    return value * ((float(shape_class.box_size) / float(ref_box)) ** power)


def _common_centered_cubes(values):
    """The classes' flattened odd BPref cubes cut to the smallest one, around their common centre.

    A pass on the model grid may keep its BPref at a stable-window physical cube while a class
    at its own box returns the logical cube (VDAM's ``keep_physical_bpref``). Both are centred on
    the same Fourier origin, so the smaller cube's voxels are the centre of the larger one: the
    cut selects the elements ``crop_public_full_volume`` would, with no arithmetic.
    """
    sizes = {int(np.shape(value)[-1]) for value in values if value is not None and np.ndim(value)}
    if len(sizes) < 2:
        return values
    edges = {size: round(size ** (1.0 / 3.0)) for size in sizes}
    if any(edge**3 != size or edge % 2 == 0 for size, edge in edges.items()):
        raise ValueError(f"shape classes returned BPref accumulators that are not odd cubes: sizes {sorted(sizes)}")
    edge = min(edges.values())
    out = []
    for value in values:
        if value is None or not np.ndim(value) or edges[int(np.shape(value)[-1])] == edge:
            out.append(value)
            continue
        big = edges[int(np.shape(value)[-1])]
        start = (big - edge) // 2
        cube = np.reshape(value, np.shape(value)[:-1] + (big, big, big))
        cut = cube[..., start : start + edge, start : start + edge, start : start + edge]
        out.append(np.reshape(cut, np.shape(value)[:-1] + (edge**3,)))
    return out


def _sum(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    total = values[0]
    for value in values[1:]:
        total = total + value
    return total


def _merge_noise_stats(stats, classes, n_half, ref_box):
    from relax.helpers.types import make_noise_stats

    if all(stat is None for stat in stats):
        return None
    if any(stat is None for stat in stats):
        raise ValueError("noise statistics are missing for some shape classes")
    if any(stat.wsum_noise_a2 is not None for stat in stats):
        raise NotImplementedError("split noise diagnostics are not merged across shape classes")
    return make_noise_stats(
        wsum_sigma2_noise=_sum(
            [noise_sums_to_reference(s.wsum_sigma2_noise, c, ref_box) for s, c in zip(stats, classes)]
        ),
        wsum_img_power=_sum(
            [noise_sums_to_reference(s.wsum_img_power, c, ref_box) for s, c in zip(stats, classes)]
        ),
        wsum_sigma2_offset=float(sum(float(s.wsum_sigma2_offset) for s in stats)),
        sumw=_sum([np.asarray(s.sumw, dtype=np.float64) for s in stats]),
        # Each image's norm residual is a power sum (box_g**4 in native units); the norm
        # corrections and their average are compared across all particles, so every class's
        # residuals go to the reference box's units.
        wsum_norm_correction=place_by_index(
            [_to_reference_units(s.wsum_norm_correction, c, ref_box, -4) for s, c in zip(stats, classes)],
            classes,
            n_half,
        ),
        wsum_scale_correction_xa=_sum([s.wsum_scale_correction_xa for s in stats]),
        wsum_scale_correction_aa=_sum([s.wsum_scale_correction_aa for s in stats]),
    )


def _require_one_backprojector_layout(results) -> None:
    """Refuse shape-class results whose M-step accumulators differ in half axis or shape: they cannot be summed."""

    first = results[0]
    for result in results[1:]:
        if (
            result.mstep_full_half_axis != first.mstep_full_half_axis
            or result.mstep_accumulator_shape != first.mstep_accumulator_shape
        ):
            raise ValueError("shape classes returned different backprojector layouts")


def merge_class_results(results, classes, n_half, ref_box):
    """One ``HalfScoreResult`` for the half from its shape classes' results."""

    from relax.dense.score_outputs import HalfScoreResult

    first = results[0]
    _require_one_backprojector_layout(results)

    def per_image(name):
        return place_by_index([getattr(result, name) for result in results], classes, n_half)

    em_stats = _merge_relion_stats([result.em_stats for result in results], classes, n_half)
    translations = [
        None if result.best_pose_translations is None
        else np.asarray(result.best_pose_translations) / shape_class.translation_factor
        for result, shape_class in zip(results, classes)
    ]
    return HalfScoreResult(
        ha=per_image("ha"),
        Ft_y=_sum([_to_reference_units(r.Ft_y, c, ref_box, 2) for r, c in zip(results, classes)]),
        Ft_ctf=_sum([_to_reference_units(r.Ft_ctf, c, ref_box, 4) for r, c in zip(results, classes)]),
        em_stats=em_stats,
        noise_stats=_merge_noise_stats([result.noise_stats for result in results], classes, n_half, ref_box),
        best_pose_rotations=per_image("best_pose_rotations"),
        best_pose_rotation_eulers=per_image("best_pose_rotation_eulers"),
        best_pose_translations=(
            None if translations[0] is None else place_by_index(translations, classes, n_half).astype(
                translations[0].dtype
            )
        ),
        coarse_ha=per_image("coarse_ha"),
        pose_rotations=first.pose_rotations,
        pose_rotation_eulers=first.pose_rotation_eulers,
        significant_counts=per_image("significant_counts"),
        profile_summary=first.profile_summary,
        mstep_full_half_axis=first.mstep_full_half_axis,
        mstep_accumulator_shape=first.mstep_accumulator_shape,
        classes=_merge_class_scores([result.classes for result in results], classes, n_half, ref_box),
    )


def _merge_class_scores(summaries, classes, n_half, ref_box):
    """Merge class assignments, posterior sums and noise into one half's result."""

    from relax.dense.score_outputs import ClassScoreSummary

    if all(summary is None for summary in summaries):
        return None
    if any(summary is None for summary in summaries):
        raise ValueError("class statistics are missing for some shape classes")
    per_class = [summary.noise_stats for summary in summaries]
    if all(stats is None for stats in per_class):
        noise_stats = None
    else:
        if any(stats is None for stats in per_class) or len({len(stats) for stats in per_class}) != 1:
            raise ValueError("per-class noise statistics are missing for some shape classes")
        noise_stats = [
            _merge_noise_stats([stats[c] for stats in per_class], classes, n_half, ref_box)
            for c in range(len(per_class[0]))
        ]
    return ClassScoreSummary(
        assignments=place_by_index([summary.assignments for summary in summaries], classes, n_half),
        mstep_mass=_sum([np.asarray(summary.mstep_mass, dtype=np.float64) for summary in summaries]),
        evidence_mass=_sum([np.asarray(summary.evidence_mass, dtype=np.float64) for summary in summaries]),
        rotation_mass=_sum([np.asarray(summary.rotation_mass, dtype=np.float64) for summary in summaries]),
        noise_stats=noise_stats,
    )


def _per_image_axis(values, classes, n_half, axis):
    """Per-image arrays with the images on ``axis``, placed back in the half's image order."""

    if all(value is None for value in values):
        return None
    moved = [None if value is None else np.moveaxis(np.asarray(value), axis, 0) for value in values]
    return np.moveaxis(place_by_index(moved, classes, n_half), 0, axis)


def _merge_relion_stats(stats, classes, n_half):
    from relax.helpers.types import RelionStats

    return RelionStats(
        log_evidence_per_image=place_by_index([s.log_evidence_per_image for s in stats], classes, n_half),
        best_log_score_per_image=place_by_index([s.best_log_score_per_image for s in stats], classes, n_half),
        max_posterior_per_image=place_by_index([s.max_posterior_per_image for s in stats], classes, n_half),
        rotation_posterior_sums=_sum([np.asarray(s.rotation_posterior_sums, dtype=np.float64) for s in stats]),
    )


def merge_k_class_engine_results(results, classes, n_half, ref_box):
    """One ``KClassEMResult`` for the half from its shape classes' engine results.

    The shape classes share the pose grids, so pose and rotation indices carry over;
    per-image fields return to the half's image order (``[K, N]`` fields on their image
    axis), backprojected sums go to the reference class's units, posterior sums add,
    noise sums land on the reference shells and best translations return to reference
    pixels.
    """

    first = results[0]
    _require_one_backprojector_layout(results)
    if any(result.new_means is not None for result in results):
        raise NotImplementedError("closed-form class means are not merged across shape classes")

    def per_image(name, axis=0):
        return _per_image_axis([getattr(r, name) for r in results], classes, n_half, axis)

    def per_class_tuple(name, transform=lambda value, shape_class: value):
        values = [getattr(r, name) for r in results]
        if all(value is None for value in values):
            return None
        n_classes = len(values[0])
        return tuple(
            place_by_index(
                [np.asarray(transform(np.asarray(value[c]), shape_class)) for value, shape_class in zip(values, classes)],
                classes,
                n_half,
            )
            for c in range(n_classes)
        )

    def to_reference_pixels(value, shape_class):
        return value / shape_class.translation_factor

    def accumulators(name, power):
        return _sum(
            _common_centered_cubes(
                [_to_reference_units(np.asarray(getattr(r, name)), c, ref_box, power) for r, c in zip(results, classes)]
            )
        )

    noise = [r.noise_stats for r in results]
    if all(value is None for value in noise):
        noise_stats = None
    else:
        if any(value is None for value in noise):
            raise ValueError("per-class noise statistics are missing for some shape classes")
        noise_stats = tuple(
            _merge_noise_stats([value[c] for value in noise], classes, n_half, ref_box) for c in range(len(noise[0]))
        )
    best_translations = per_class_tuple("per_class_best_pose_translations", to_reference_pixels)
    joint_translations = [
        None if r.best_pose_translations is None else to_reference_pixels(np.asarray(r.best_pose_translations), c)
        for r, c in zip(results, classes)
    ]
    return first._replace(
        Ft_y=accumulators("Ft_y", 2),
        Ft_ctf=accumulators("Ft_ctf", 4),
        per_class_hard_assignments=per_image("per_class_hard_assignments", axis=1),
        class_assignments=per_image("class_assignments"),
        pose_assignments=per_image("pose_assignments"),
        class_responsibilities=per_image("class_responsibilities", axis=1),
        class_posterior_sums=_sum([np.asarray(r.class_posterior_sums, dtype=np.float64) for r in results]),
        class_mstep_posterior_sums=_sum(
            [None if r.class_mstep_posterior_sums is None else np.asarray(r.class_mstep_posterior_sums, np.float64)
             for r in results]
        ),
        stats=_merge_relion_stats([r.stats for r in results], classes, n_half),
        per_class_stats=tuple(
            _merge_relion_stats([r.per_class_stats[c] for r in results], classes, n_half)
            for c in range(len(first.per_class_stats))
        ),
        noise_stats=noise_stats,
        aggregate_noise_stats=_merge_noise_stats([r.aggregate_noise_stats for r in results], classes, n_half, ref_box),
        per_class_best_pose_rotations=per_class_tuple("per_class_best_pose_rotations"),
        per_class_best_pose_translations=best_translations,
        per_class_best_pose_rotation_ids=per_class_tuple("per_class_best_pose_rotation_ids"),
        per_class_best_pose_eulers_deg=per_class_tuple("per_class_best_pose_eulers_deg"),
        best_pose_rotations=per_image("best_pose_rotations"),
        best_pose_translations=_per_image_axis(joint_translations, classes, n_half, 0),
        best_pose_rotation_ids=per_image("best_pose_rotation_ids"),
        best_pose_eulers_deg=per_image("best_pose_eulers_deg"),
        significant_counts=per_image("significant_counts"),
        profile_summary=first.profile_summary,
        uncast_log_evidence_per_image=per_image("uncast_log_evidence_per_image"),
    )


def require_exact_local_parent_windows(kwargs) -> None:
    """Refuse a local search whose parent pass would need RELION's wrapped coarse rows.

    The local parent pass (RELION's pass 1) projects in Python and zeroes the rows beyond
    ``maxR``. That is RELION's coarse kernel unless the class's pass-1 window lies strictly
    between ``2 r_max`` and about ``2 s r_max`` (:func:`relax.helpers.optics_scale.coarse_rows_wrap_inside`).
    """

    half = kwargs["experiment_dataset"]
    ref_box = int(half.image_shape[0])
    reference_size = kwargs.get("cs_for_engine") or ref_box
    for shape_class in half.classes:
        window = class_kwargs(kwargs, shape_class, half.n_units).get("local_pass1_current_size") or shape_class.box_size
        if optics_scale.coarse_rows_wrap_inside(window, int(reference_size) // 2, shape_class.scale):
            raise NotImplementedError(
                f"optics group class of {shape_class.box_size} px at scale {shape_class.scale:.3f}: its local pass-1 "
                f"window {window} lies between 2 r_max and 2 s r_max (r_max {int(reference_size) // 2}), where "
                "RELION's coarse kernel projects its wrapped outer rows (diff2.cuh:86-90). Only the fused coarse "
                "scorer reproduces that so far."
            )
