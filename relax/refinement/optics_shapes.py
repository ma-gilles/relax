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

import numpy as np

from relax.helpers import optics_scale
from relax.helpers.resolution import clamp_relion_coarse_image_size, compute_coarse_image_size


@dataclasses.dataclass(frozen=True)
class ShapeClass:
    """The images of one half that share a box and a pixel size."""

    dataset: object
    image_indices: np.ndarray  # positions in the half's image order
    box_size: int
    pixel_size: float
    scale: float  # s_g = box_g angpix_g / (ori angpix_ref)
    translation_factor: float  # reference pixels -> class pixels: angpix_ref / angpix_g


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
        order = np.concatenate([c.image_indices for c in self.classes])
        if not np.array_equal(np.sort(order), np.arange(self.n_units)):
            raise ValueError("shape classes must partition the half's images")
        # Particle-STAR row of each image, as a loaded dataset's index layout reports it.
        self._index_layout = None if rows is None else _RowLayout(np.asarray(rows, dtype=np.int64))

    def __getattr__(self, name):
        raise AttributeError(f"a half with several image shapes has no single {name!r}; use its shape classes")


def make_shape_classes(datasets_and_indices, *, ref_box, ref_pixel):
    """``ShapeClass`` records from ``(dataset, half positions)`` pairs."""

    classes = []
    for dataset, indices in datasets_and_indices:
        box = int(dataset.image_shape[0])
        pixel = float(dataset.voxel_size)
        classes.append(
            ShapeClass(
                dataset=dataset,
                image_indices=np.asarray(indices, dtype=np.int64),
                box_size=box,
                pixel_size=pixel,
                scale=optics_scale.scale_difference(box, pixel, ref_box, ref_pixel),
                translation_factor=float(ref_pixel) / pixel,
            )
        )
    return classes


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
    "replay_prior_translations",
)
# Keywords that may be per image when two-dimensional (image x hypothesis).
PER_IMAGE_IF_2D_KWARGS = ("translation_log_prior", "rotation_log_prior_k", "class_rotation_log_prior_k")
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


def class_kwargs(kwargs, shape_class: ShapeClass, n_half: int) -> dict:
    """One shape class's keywords: its images, class pixels and class Fourier sizes."""

    out = dict(kwargs)
    coarse_sizing = out.pop("coarse_sizing", None)
    index = shape_class.image_indices
    for name in PER_IMAGE_KWARGS + PER_IMAGE_IF_2D_KWARGS:
        value = out.get(name)
        if value is None:
            continue
        array = np.asarray(value)
        if name in PER_IMAGE_IF_2D_KWARGS and array.ndim != 2:
            continue
        if array.ndim == 0 or array.shape[0] != n_half:
            raise ValueError(f"{name} must have the half's {n_half} images on its leading axis")
        out[name] = array[index]
    for name in TRANSLATION_KWARGS:
        if out.get(name) is not None:
            out[name] = np.asarray(out[name]) * shape_class.translation_factor
    state = out.get("state")
    if state is not None and getattr(state, "translation_step", None) is not None and shape_class.translation_factor != 1.0:
        # The oversampled translation grid is built from the step (RELION samples offsets in
        # Angstrom and converts them with the image's pixel size, getTranslationsInPixel).
        out["state"] = dataclasses.replace(
            state, translation_step=float(state.translation_step) * shape_class.translation_factor
        )
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
    size = clamp_relion_coarse_image_size(
        compute_coarse_image_size(
            angular_step_deg, shape_class.pixel_size, shape_class.box_size, particle_diameter=particle_diameter_ang
        ),
        class_current_size,
        shape_class.box_size,
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
        wsum_norm_correction=place_by_index([s.wsum_norm_correction for s in stats], classes, n_half),
        wsum_scale_correction_xa=_sum([s.wsum_scale_correction_xa for s in stats]),
        wsum_scale_correction_aa=_sum([s.wsum_scale_correction_aa for s in stats]),
    )


def merge_class_results(results, classes, n_half, ref_box):
    """One ``HalfScoreResult`` for the half from its shape classes' results."""

    from relax.dense.score_outputs import HalfScoreResult
    from relax.helpers.types import RelionStats

    first = results[0]
    for result in results[1:]:
        if (
            result.mstep_full_half_axis != first.mstep_full_half_axis
            or result.mstep_accumulator_shape != first.mstep_accumulator_shape
        ):
            raise ValueError("shape classes returned different backprojector layouts")

    def per_image(name):
        return place_by_index([getattr(result, name) for result in results], classes, n_half)

    stats = [result.em_stats for result in results]
    em_stats = RelionStats(
        log_evidence_per_image=place_by_index([s.log_evidence_per_image for s in stats], classes, n_half),
        best_log_score_per_image=place_by_index([s.best_log_score_per_image for s in stats], classes, n_half),
        max_posterior_per_image=place_by_index([s.max_posterior_per_image for s in stats], classes, n_half),
        rotation_posterior_sums=_sum([np.asarray(s.rotation_posterior_sums, dtype=np.float64) for s in stats]),
    )
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
    )


def score_half_by_shape(score_fn, kwargs):
    """Run ``score_fn`` (a half scoring function) once per shape class and merge.

    ``kwargs`` are ``score_fn``'s keywords for the whole half plus ``noise_radial_k``,
    the half's ``[G, n_ref]`` reference-shell noise spectra, and optionally
    ``class_batch_overrides``, one dict of batch-size keywords per class planned for
    that class's own image box and sizes, and ``class_translation_overrides``, each
    class's translation operands rebuilt in its own pixels.
    """

    from relax.dense.score_outputs import PerHalfOutputs

    kwargs = dict(kwargs)
    half = kwargs["experiment_dataset"]
    noise_radial = kwargs.pop("noise_radial_k")
    batch_overrides = kwargs.pop("class_batch_overrides", None)
    translation_overrides = kwargs.pop("class_translation_overrides", None)
    if batch_overrides is not None and len(batch_overrides) != len(half.classes):
        raise ValueError("class_batch_overrides needs one entry per shape class")
    if kwargs.get("optics_group_ids_k") is None:
        raise ValueError("a half with several image shapes needs each image's optics group")
    outputs, k = kwargs["outputs"], kwargs["k"]
    ref_box = int(half.image_shape[0])
    results = []
    for index, shape_class in enumerate(half.classes):
        class_kw = class_kwargs(kwargs, shape_class, half.n_units)
        if batch_overrides is not None:
            class_kw.update(batch_overrides[index])
        if translation_overrides is not None:
            # The class's own rounded pre-shifts, prior centers and pdf_offset replace the
            # scaled reference values, for the operands this call takes.
            class_kw.update({key: value for key, value in translation_overrides[index].items() if key in class_kw})
        class_kw["noise_variance_k"] = class_noise_table(noise_radial, shape_class, ref_box)
        class_kw["outputs"] = PerHalfOutputs()
        results.append(score_fn(**class_kw))
    merged = merge_class_results(results, half.classes, half.n_units, ref_box)
    outputs.best_pose_rotations[k] = merged.best_pose_rotations
    outputs.best_pose_rotation_eulers[k] = merged.best_pose_rotation_eulers
    outputs.best_pose_translations[k] = merged.best_pose_translations
    return merged
