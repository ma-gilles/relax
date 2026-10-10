"""Halves whose optics groups differ in pixel size or box (RELION S3b).

Images of different shapes cannot share one dataset or one compiled program, so a
half is split into *shape classes* (optics groups with the same box and pixel size).
Half scoring runs the unchanged single-shape route once per class on that class's
dataset, with RELION's rules for a group on another grid (:mod:`relax.relion.optics_scale`):

- projection/backprojection matrices divided by the class scale ``s_g``
  (``applyScaleDifference``); poses stay unscaled everywhere else;
- translations converted from reference pixels to class pixels;
- image sizes remapped (``updateImageSizeAndResolutionPointers``), while the
  backprojector keeps the reference model size;
- noise read from the reference-shell spectrum (E-step remap) and the class's noise
  sums added back onto the reference shells (M-step remap).

Per-image outputs return to the half's image order by index, and additive sums add.
A half with one shape class is a plain dataset and never reaches this module.

This module holds the shape classes and the datasets built from them; the per-class operand mapping and the
merging of the classes' results are :mod:`relax.refinement.shape_class_scoring`.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from relax.relion import optics_scale


@dataclasses.dataclass(frozen=True)
class ShapeClass:
    """The images of one half that share a box and a pixel size."""

    dataset: object
    image_indices: np.ndarray  # positions in the half's image order
    box_size: int
    pixel_size: float
    scale: float  # s_g = box_g angpix_g / (ori angpix_ref)
    translation_factor: float  # reference pixels -> class pixels: angpix_ref / angpix_g
    # The trial translation grid's reference pixels -> class pixels: RELION builds the grid in Angstrom
    # with the model pixel (ml_optimiser.cpp:590, 597) and converts it with the class's pixel
    # (getTranslationsInPixel), model_pix / angpix_g. None: translation_factor (the model pixel is the STAR's).
    grid_factor: float | None = None

    def trial_grid_factor(self) -> float:
        """Reference pixels -> class pixels for the trial grid (``grid_factor``, else ``translation_factor``)."""
        return self.translation_factor if self.grid_factor is None else self.grid_factor


@dataclasses.dataclass(frozen=True)
class RowLayout:
    """The original dataset index of each of a half's local image rows."""

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
        self._index_layout = None if rows is None else RowLayout(np.asarray(rows, dtype=np.int64))

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


def shape_datasets(dataset) -> tuple:
    """The loaded datasets behind a half or a particle set: one per image shape.

    A ``MultiShapeHalf`` or ``MultiShapeDataset`` has no single source STAR, so questions
    about its optics table (CTF-premultiplied groups) go to these.
    """

    if isinstance(dataset, MultiShapeHalf):
        return tuple(shape_class.dataset for shape_class in dataset.classes)
    if isinstance(dataset, MultiShapeDataset):
        return dataset.datasets
    return (dataset,)


def average_ctf2_parts(half, scales, *, current_size, image_current_size) -> list:
    """One half's ``relion_ctf.premultiplied_average_ctf2`` parts.

    A half on one grid is one part at ``image_current_size``. A ``MultiShapeHalf`` gives
    one part per shape class, with the class's scale corrections, its own image current
    size (``updateImageSizeAndResolutionPointers``) and its scale ``s_g`` for the shell remap.
    """

    if not isinstance(half, MultiShapeHalf):
        return [(half, scales, int(image_current_size), 1.0)]
    return [
        (
            shape_class.dataset,
            None if scales is None else np.asarray(scales).reshape(-1)[shape_class.image_indices],
            optics_scale.group_current_size(current_size, shape_class.box_size, shape_class.scale),
            shape_class.scale,
        )
        for shape_class in half.classes
    ]


def make_shape_classes(datasets_and_indices, *, model_box_size, ref_pixel, model_pixel=None):
    """``ShapeClass`` records from ``(dataset, half positions)`` pairs.

    ``model_pixel`` is RELION's model pixel size (the reference map's header value, ml_model.cpp:944-962;
    None: ``ref_pixel``): each group's scale and current-size remap read it
    (``remap_sizes``, ml_optimiser.cpp:6917-6923). The translations keep ``ref_pixel``, the first optics
    group's pixel, as the stored offsets do.

    A class at scale ``s >= sqrt(2)`` is refused. There, RELION's fine kernels project
    a moved pixel inside the model sphere for the image rows beyond it
    (:func:`relax.projection.projection.relion_kernel_zero_rows`), which relax does not reproduce.
    Listing the optics group with the largest box x pixel size first avoids it.
    """

    classes = []
    for dataset, indices in datasets_and_indices:
        box_size = int(dataset.image_shape[0])
        pixel = float(dataset.voxel_size)
        scale = optics_scale.scale_difference(
            box_size, pixel, model_box_size, ref_pixel if model_pixel is None else model_pixel
        )
        if scale >= math.sqrt(2.0):
            raise NotImplementedError(
                f"an optics group of {box_size} px at {pixel} A spans {scale:.3f} times the reference field of view "
                f"({model_box_size} px at {ref_pixel} A). From sqrt(2) on, RELION's fine and weighted-sum kernels "
                "project the pixel (maxR, i) for every image row i beyond the model radius, and shift the image at that "
                "pixel too (acc/cuda/cuda_kernels/diff2.cuh:494-530, wavg.cuh:81-86), a RELION defect relax does not "
                "reproduce. List the optics group with the largest box x pixel size first."
            )
        classes.append(
            ShapeClass(
                dataset=dataset,
                image_indices=np.asarray(indices, dtype=np.int64),
                box_size=box_size,
                pixel_size=pixel,
                scale=scale,
                translation_factor=float(ref_pixel) / pixel,
                grid_factor=None if model_pixel is None else float(model_pixel) / pixel,
            )
        )
    return classes


_MAG_MATRIX_COLUMNS = ("rlnMagMat00", "rlnMagMat01", "rlnMagMat10", "rlnMagMat11")


def optics_shape_class_rows(particles_star):
    """Particle rows per image shape when optics groups differ in box, pixel size or magnification.

    ``None`` when every optics group has one box, pixel size and ``rlnMagMat`` (the
    single-dataset path). Otherwise one row array per (box, pixel size, magnification
    matrix), in optics-table order, so the first class holds optics group 1, whose grid
    is RELION's model grid. RELION applies each group's ``rlnMagMat`` to its images'
    projection and backprojection matrices (``applyAnisoMag``); a class holds one
    matrix, which its scoring calls apply (relax#48).
    """
    import starfile

    star = starfile.read(particles_star)
    optics = star.get("optics") if isinstance(star, dict) else None
    if optics is None or not {"rlnImageSize", "rlnImagePixelSize"}.issubset(optics.columns):
        return None
    mags = (
        [tuple(row) for row in np.asarray(optics[list(_MAG_MATRIX_COLUMNS)], dtype=np.float64).tolist()]
        if set(_MAG_MATRIX_COLUMNS).issubset(optics.columns)
        else [None] * len(optics)
    )
    shapes = list(zip(np.asarray(optics["rlnImageSize"], dtype=np.int64).tolist(),
                      np.asarray(optics["rlnImagePixelSize"], dtype=np.float64).tolist(),
                      mags))
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
    ``model_pixel_size`` is RELION's model pixel size when a reference map sets it
    (:meth:`with_model_pixel_size`); the halves' shape classes scale against it.
    """

    def __init__(self, datasets, rows, model_pixel_size=None):
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
        self.model_pixel_size = self.voxel_size if model_pixel_size is None else float(model_pixel_size)

    def __getattr__(self, name):
        raise AttributeError(f"a dataset with several image shapes has no single {name!r}; use its classes")

    def class_rows(self, rows):
        """``(dataset, local rows)``: the shape class holding all these particle rows, and their rows in it."""

        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        classes = np.unique(self._class_of_row[rows])
        if classes.size != 1:
            raise ValueError(f"these particle rows span shape classes {classes.tolist()}")
        return self.datasets[int(classes[0])], self._local_of_row[rows]

    def with_model_pixel_size(self, model_pixel_size):
        """The same particles scaled against RELION's model pixel size (the reference map header's)."""

        return MultiShapeDataset(self.datasets, self.rows, model_pixel_size=model_pixel_size)

    def subset(self, rows):
        """The ``MultiShapeHalf`` of these particle rows, in this order."""

        rows = np.asarray(rows, dtype=np.int64)
        pairs = []
        for c, dataset in enumerate(self.datasets):
            positions = np.flatnonzero(self._class_of_row[rows] == c)
            if positions.size:
                pairs.append((dataset.subset(self._local_of_row[rows[positions]]), positions))
        classes = make_shape_classes(
            pairs, model_box_size=self.grid_size, ref_pixel=self.voxel_size, model_pixel=self.model_pixel_size
        )
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
