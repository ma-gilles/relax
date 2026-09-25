"""Host data model for the device-resident K=1 sparse pass-2 prototype.

This module builds a flat, row-major (CSR-by-image) table of pass-2 candidate
rows from :func:`recovar.em.scoring.sparse_bucket_arrays._prepare_per_image_pass2_inputs`
output, and chunks those rows into fixed-capacity "programs" so a future
device-resident kernel can be traced once per (row capacity, image capacity,
pixel count) triple instead of once per bucket shape.

Everything here is host planning: plain numpy (int32/float32/uint32/int8) plus
one jax.numpy reference twin (:func:`expand_mask_jnp`) used only to validate
that the same bit-unpacking arithmetic works on device arrays. No scoring,
projection or M-step arithmetic lives in this module; see
``../em_device_resident_pass2_design_20260918.md`` for the stages this table
feeds.

Index conventions
------------------
* "image" always means the *local* position (0..n_images-1) of an image
  within the ``per_image_inputs`` list passed to
  :func:`build_resident_candidate_tables`, i.e. exactly the index used to
  index every ``per_image_inputs[...][i]`` list. It is not a dataset/original
  particle id.
* "row" means one (image, fine rotation) candidate: one entry of
  ``per_image_inputs["oversampled_rot_indices"][i]``. Rows are laid out
  image-major (CSR), and never reordered relative to
  ``per_image_inputs`` — the row order for image ``i`` is exactly
  ``per_image_inputs["oversampled_rot_indices"][i]``'s order.
* "fine rotation id" (``row_fine_rot``) indexes the shared, parent-major fine
  rotation grid the compact engine already uses (see the design doc); this
  module never looks at rotation matrices, only integer ids.
* "parent" means an image-local coarse rotation: the position of a coarse
  rotation within that image's own ``unique_rot`` array, i.e. exactly
  ``per_image_inputs["parent_map"][i]``'s values. Two different images' parent
  index ``0`` generally refer to different coarse rotations.
* Coarse translation bitsets pack bit ``k`` = "coarse translation ``k`` is a
  valid candidate for this (image, parent)". Bit order is the raw coarse
  translation id (0..n_coarse_trans-1), matching
  ``SparseCandidateMask.coarse_valid``'s column order exactly. A bitset is
  ``n_mask_words(n_coarse_trans)`` little-endian uint32 words: translation
  ``k`` is bit ``k % 32`` of word ``k // 32``, so any translation count fits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from relax.scoring.compact_candidates import SparseCandidateMask

__all__ = [
    "CapacityChunk",
    "merge_class_tables",
    "coarse_winner_cells",
    "n_mask_words",
    "ResidentCandidateTables",
    "build_resident_candidate_tables",
    "expand_chunk_mask_jnp",
    "expand_mask_jnp",
    "expand_mask_rows",
    "materialize_chunk",
    "plan_capacity_chunks",
]

# Per-image mask modes (int8). Kept as module-level constants so both the
# numpy and jax.numpy expanders and the tests share one vocabulary.
_MASK_MODE_FULL = np.int8(0)
_MASK_MODE_BITSET = np.int8(1)
_MASK_MODE_EMPTY = np.int8(2)

_ROW_FINE_ROT_PAD = np.int32(0)
_ROW_PARENT_LOCAL_PAD = np.int32(0)
_ROW_LOG_PRIOR_PAD = np.float32(-1e30)
_ROW_MASK_BITS_PAD = np.uint32(0)
_MASK_WORD_BITS = 32


def n_mask_words(n_coarse_trans: int) -> int:
    """uint32 words per coarse-translation bitset (at least one)."""

    n_coarse_trans = int(n_coarse_trans)
    if n_coarse_trans <= 0:
        raise ValueError(f"n_coarse_trans must be positive, got {n_coarse_trans}")
    return -(-n_coarse_trans // _MASK_WORD_BITS)


def all_translations_words(n_coarse_trans: int) -> np.ndarray:
    """The bitset with every coarse translation set, as uint32 words."""

    words = np.zeros(n_mask_words(n_coarse_trans), dtype=np.uint32)
    for word in range(words.size):
        n_bits = min(_MASK_WORD_BITS, int(n_coarse_trans) - word * _MASK_WORD_BITS)
        words[word] = np.uint32((1 << n_bits) - 1)
    return words


def translation_word_and_bit(trans) -> tuple[np.ndarray, np.ndarray]:
    """(word index, single-bit uint32 mask) of each coarse translation id."""

    trans = np.asarray(trans, dtype=np.int64)
    return trans // _MASK_WORD_BITS, np.uint32(1) << (trans % _MASK_WORD_BITS).astype(np.uint32)


@dataclass(frozen=True)
class ResidentCandidateTables:
    """Flat, image-CSR table of pass-2 candidate rows plus their masks.

    All arrays are plain numpy; nothing here is device-resident yet (that is
    the job of the chunk programs this table feeds). Units: ``row_log_prior``
    is a natural-log prior density in the same units as
    ``per_image_inputs["log_prior"]`` (nats); every other field is an integer
    id or bitset.
    """

    n_images: int
    n_rows: int
    n_fine_trans: int
    n_coarse_trans: int
    # CSR row ranges: image i owns rows [row_offsets[i], row_offsets[i + 1]).
    row_offsets: np.ndarray  # int32 [n_images + 1]
    row_image: np.ndarray  # int32 [n_rows], non-decreasing, values in [0, n_images)
    row_fine_rot: np.ndarray  # int32 [n_rows]
    row_parent_local: np.ndarray  # int32 [n_rows]
    row_log_prior: np.ndarray  # float32 [n_rows]
    # Per-image mask mode and, for bitset-mode images only, a CSR table of
    # per-parent coarse-translation bitsets.
    mask_mode: np.ndarray  # int8 [n_images]
    parent_offsets: np.ndarray  # int32 [n_images + 1]
    parent_trans_bits: np.ndarray  # uint32 [parent_offsets[-1], n_mask_words(n_coarse_trans)]
    # Class of each row (K>1 Class3D); None means every row is class 0. Rows are
    # image-major, then class-major (RELION's iorientclass = iclass * nr_dir *
    # nr_psi + iorient, ml_optimiser.cpp:8450), so one image's posterior
    # segment spans all its classes.
    row_class: np.ndarray | None = None  # int32 [n_rows]
    n_classes: int = 1

    def __post_init__(self):
        if self.row_offsets.shape != (self.n_images + 1,):
            raise ValueError("row_offsets must have shape (n_images + 1,)")
        if self.parent_offsets.shape != (self.n_images + 1,):
            raise ValueError("parent_offsets must have shape (n_images + 1,)")
        if int(self.row_offsets[-1]) != int(self.n_rows):
            raise ValueError("row_offsets[-1] must equal n_rows")
        for name in ("row_image", "row_fine_rot", "row_parent_local", "row_log_prior"):
            arr = getattr(self, name)
            if arr.shape != (self.n_rows,):
                raise ValueError(f"{name} must have shape (n_rows,), got {arr.shape}")
        if self.mask_mode.shape != (self.n_images,):
            raise ValueError("mask_mode must have shape (n_images,)")
        expected_bits = (int(self.parent_offsets[-1]), n_mask_words(self.n_coarse_trans))
        if self.parent_trans_bits.shape != expected_bits:
            raise ValueError(
                f"parent_trans_bits must have shape {expected_bits}, got {self.parent_trans_bits.shape}"
            )
        if self.row_class is not None:
            if self.row_class.shape != (self.n_rows,):
                raise ValueError(f"row_class must have shape (n_rows,), got {self.row_class.shape}")
            if self.row_class.size and (
                int(self.row_class.min()) < 0 or int(self.row_class.max()) >= int(self.n_classes)
            ):
                raise ValueError(f"row_class values must lie in [0, {self.n_classes})")
        elif int(self.n_classes) != 1:
            raise ValueError("a K>1 table needs row_class")


@dataclass(frozen=True)
class CapacityChunk:
    """One contiguous, capacity-padded image range.

    ``[image_start, image_stop)`` and the matching ``[row_start, row_stop)``
    (exactly ``row_offsets[image_start]`` and ``row_offsets[image_stop]``) are
    the *valid* extents; ``row_capacity``/``image_capacity`` are the padded
    program shape this chunk will run at (from the capacity ladders, or, for
    a single image whose own rows exceed the largest row class, the smallest
    multiple of that largest class that covers it).
    """

    image_start: int
    image_stop: int
    row_start: int
    row_stop: int
    row_capacity: int
    image_capacity: int

    @property
    def n_valid_images(self) -> int:
        return self.image_stop - self.image_start

    @property
    def n_valid_rows(self) -> int:
        return self.row_stop - self.row_start


def _pack_bits_rows(bool_rows: np.ndarray) -> np.ndarray:
    """Pack each row of a boolean matrix into uint32 words, column k = bit k % 32 of word k // 32."""

    n_rows, n_bits = bool_rows.shape
    n_words = n_mask_words(n_bits)
    padded = np.zeros((n_rows, n_words * _MASK_WORD_BITS), dtype=np.uint32)
    padded[:, :n_bits] = bool_rows
    weights = np.uint32(1) << np.arange(_MASK_WORD_BITS, dtype=np.uint32)
    return (padded.reshape(n_rows, n_words, _MASK_WORD_BITS) * weights).sum(axis=2, dtype=np.uint32)


def _bits_for_mask(mask: SparseCandidateMask, n_coarse_trans: int) -> np.ndarray:
    """Return one bitset (uint32 words) per image-local parent for a bitset-mode mask.

    Only valid for ``mask.mode in {"coarse", "coarse_exclude"}``. The number
    of parents returned is the number of distinct parent indices actually
    referenced by ``mask.parent_map`` (``max(parent_map) + 1``), which is
    exactly the range ``row_parent_local`` can take for this image -- there is
    no need to size the table to the full coarse rotation grid.
    """

    if mask.mode == "coarse":
        if mask.coarse_valid is None or mask.fine_translation_parent is None:
            raise ValueError("coarse candidate mask spec is missing coarse_valid")
        if mask.coarse_valid.shape[1] != n_coarse_trans:
            raise ValueError(
                "coarse candidate mask coarse_valid width does not match n_coarse_trans: "
                f"{mask.coarse_valid.shape[1]} vs {n_coarse_trans}",
            )
        return _pack_bits_rows(mask.coarse_valid)
    if mask.mode == "coarse_exclude":
        if mask.coarse_excluded is None or mask.parent_map is None:
            raise ValueError("coarse_exclude candidate mask spec is missing excluded/parent arrays")
        n_parents = int(mask.parent_map.max(initial=-1)) + 1
        bits = np.tile(all_translations_words(n_coarse_trans), (n_parents, 1))
        excluded = np.unique(np.asarray(mask.coarse_excluded, dtype=np.int64).reshape(-1))
        if excluded.size:
            excluded_rot = excluded // int(n_coarse_trans)
            excluded_trans = excluded % int(n_coarse_trans)
            if int(excluded_rot.max(initial=-1)) >= n_parents:
                raise ValueError("coarse_exclude excluded rotation is outside this image's referenced parents")
            words, clear_bits = translation_word_and_bit(excluded_trans)
            # A parent can appear more than once in excluded_rot (several
            # excluded translations for the same rotation); fold with a
            # scatter-AND so every exclusion is applied.
            np.bitwise_and.at(bits, (excluded_rot, words), ~clear_bits)
        return bits
    raise ValueError(f"_bits_for_mask does not support mode {mask.mode!r}")


def build_resident_candidate_tables(
    per_image_inputs: dict,
    *,
    n_coarse_trans: int,
    n_fine_trans: int,
    fine_translation_parent,
) -> ResidentCandidateTables:
    """Flatten ``_prepare_per_image_pass2_inputs`` output into one CSR table.

    ``per_image_inputs`` is exactly the dict returned by
    :func:`recovar.em.scoring.sparse_bucket_arrays._prepare_per_image_pass2_inputs`;
    only the ``oversampled_rot_indices``, ``parent_map``, ``log_prior`` and
    ``candidate_mask`` entries are read (rotations/mstep rotations/source
    Eulers stay on the shared fine grid and are looked up later by
    ``row_fine_rot``, not copied here).

    ``fine_translation_parent`` is accepted (rather than read off any single
    image) because it is one shared, iteration-global table; every image's
    mask expands against the same array.
    """

    n_coarse_trans = int(n_coarse_trans)
    n_words = n_mask_words(n_coarse_trans)
    n_fine_trans = int(n_fine_trans)
    fine_translation_parent = np.asarray(fine_translation_parent)
    if fine_translation_parent.shape != (n_fine_trans,):
        raise ValueError(
            f"fine_translation_parent must have shape (n_fine_trans,)={(n_fine_trans,)}, "
            f"got {fine_translation_parent.shape}",
        )

    oversampled_rot_indices = per_image_inputs["oversampled_rot_indices"]
    parent_map_list = per_image_inputs["parent_map"]
    log_prior_list = per_image_inputs["log_prior"]
    candidate_mask_list = per_image_inputs["candidate_mask"]
    n_images = len(oversampled_rot_indices)
    if not (len(parent_map_list) == len(log_prior_list) == len(candidate_mask_list) == n_images):
        raise ValueError("per_image_inputs lists disagree on image count")

    row_offsets = np.zeros(n_images + 1, dtype=np.int32)
    parent_offsets = np.zeros(n_images + 1, dtype=np.int32)
    mask_mode = np.empty(n_images, dtype=np.int8)

    row_image_parts: list[np.ndarray] = []
    row_fine_rot_parts: list[np.ndarray] = []
    row_parent_local_parts: list[np.ndarray] = []
    row_log_prior_parts: list[np.ndarray] = []
    parent_trans_bits_parts: list[np.ndarray] = []

    int32_max = np.iinfo(np.int32).max
    for i in range(n_images):
        rot_ids = np.asarray(oversampled_rot_indices[i])
        parent_map = np.asarray(parent_map_list[i], dtype=np.int32)
        log_prior = np.asarray(log_prior_list[i], dtype=np.float32)
        n_rows_i = int(rot_ids.shape[0])
        if parent_map.shape != (n_rows_i,) or log_prior.shape != (n_rows_i,):
            raise ValueError(f"image {i}: oversampled_rot_indices/parent_map/log_prior disagree on row count")
        if rot_ids.size and int(rot_ids.max()) > int32_max:
            raise ValueError(f"image {i}: a fine rotation id overflows int32")

        row_offsets[i + 1] = row_offsets[i] + n_rows_i
        row_image_parts.append(np.full(n_rows_i, i, dtype=np.int32))
        row_fine_rot_parts.append(rot_ids.astype(np.int32, copy=False))
        row_parent_local_parts.append(parent_map)
        row_log_prior_parts.append(log_prior)

        mask = candidate_mask_list[i]
        if not isinstance(mask, SparseCandidateMask):
            raise TypeError(f"image {i}: candidate_mask must be a SparseCandidateMask, got {type(mask)!r}")
        if mask.n_fine_trans != n_fine_trans:
            raise ValueError(f"image {i}: candidate mask n_fine_trans={mask.n_fine_trans} != {n_fine_trans}")
        if mask.mode == "full":
            mask_mode[i] = _MASK_MODE_FULL
            parent_offsets[i + 1] = parent_offsets[i]
        elif mask.mode == "empty":
            mask_mode[i] = _MASK_MODE_EMPTY
            parent_offsets[i + 1] = parent_offsets[i]
        elif mask.mode in ("coarse", "coarse_exclude"):
            mask_mode[i] = _MASK_MODE_BITSET
            bits_i = _bits_for_mask(mask, n_coarse_trans)
            if n_rows_i and int(parent_map.max(initial=-1)) >= bits_i.shape[0]:
                raise ValueError(f"image {i}: row_parent_local references a parent outside its bitset table")
            parent_offsets[i + 1] = parent_offsets[i] + bits_i.shape[0]
            parent_trans_bits_parts.append(bits_i)
        else:
            raise ValueError(f"image {i}: unknown candidate mask mode {mask.mode!r}")

    n_rows = int(row_offsets[-1])
    row_image = np.concatenate(row_image_parts) if row_image_parts else np.zeros(0, dtype=np.int32)
    row_fine_rot = np.concatenate(row_fine_rot_parts) if row_fine_rot_parts else np.zeros(0, dtype=np.int32)
    row_parent_local = (
        np.concatenate(row_parent_local_parts) if row_parent_local_parts else np.zeros(0, dtype=np.int32)
    )
    row_log_prior = np.concatenate(row_log_prior_parts) if row_log_prior_parts else np.zeros(0, dtype=np.float32)
    parent_trans_bits = (
        np.concatenate(parent_trans_bits_parts)
        if parent_trans_bits_parts
        else np.zeros((0, n_words), dtype=np.uint32)
    )

    return ResidentCandidateTables(
        n_images=n_images,
        n_rows=n_rows,
        n_fine_trans=n_fine_trans,
        n_coarse_trans=n_coarse_trans,
        row_offsets=row_offsets,
        row_image=row_image,
        row_fine_rot=row_fine_rot,
        row_parent_local=row_parent_local,
        row_log_prior=row_log_prior,
        mask_mode=mask_mode,
        parent_offsets=parent_offsets,
        parent_trans_bits=parent_trans_bits,
    )


def merge_class_tables(tables_by_class) -> ResidentCandidateTables:
    """One K-class candidate table from K single-class tables of the same images.

    Each class's table holds that class's rows (its own significant coarse
    support, rotation prior and bitsets, built exactly as for K=1). The merged
    table orders rows image-major, then class-major, then in the class table's
    own row order, so an image's posterior segment is RELION's class-major
    hidden space (ml_optimiser.cpp:8406, :8450) and the segmented posterior
    normalizes jointly over classes (:9225, :9602-9660).

    Every image of the merged table is in bitset mode: a class that was full
    for an image contributes all-set bitsets for its parents, an empty class
    all-clear ones, so a row's mask never depends on another class's mode.
    """

    tables_by_class = list(tables_by_class)
    n_classes = len(tables_by_class)
    if n_classes == 0:
        raise ValueError("merge_class_tables needs at least one class table")
    first = tables_by_class[0]
    for tables in tables_by_class[1:]:
        if (tables.n_images, tables.n_fine_trans, tables.n_coarse_trans) != (
            first.n_images,
            first.n_fine_trans,
            first.n_coarse_trans,
        ):
            raise ValueError("class tables must cover the same images and translation grids")
    n_images = int(first.n_images)
    n_words = n_mask_words(first.n_coarse_trans)
    image_ids = np.arange(n_images, dtype=np.int64)

    row_parts, parent_parts = [], []
    for class_index, tables in enumerate(tables_by_class):
        row_image = np.asarray(tables.row_image, dtype=np.int64)
        parent_local = np.asarray(tables.row_parent_local, dtype=np.int64)
        # Parents an image's rows reference in this class (bitset images store
        # exactly these; full and empty images store none).
        n_parents = np.zeros(n_images, dtype=np.int64)
        np.maximum.at(n_parents, row_image, parent_local + 1)
        mode = np.asarray(tables.mask_mode)
        stored = np.diff(tables.parent_offsets.astype(np.int64))
        bitset = mode == _MASK_MODE_BITSET
        if np.any(bitset & (stored < n_parents)):
            raise ValueError("a class table's rows reference parents outside its bitset table")
        n_parents = np.where(bitset, stored, n_parents)
        parent_image = np.repeat(image_ids, n_parents)
        bits = np.zeros((int(n_parents.sum()), n_words), dtype=np.uint32)
        first_parent = np.concatenate([[0], np.cumsum(n_parents)])[:-1]
        within = np.arange(bits.shape[0], dtype=np.int64) - first_parent[parent_image]
        parent_mode = mode[parent_image]
        from_table = parent_mode == _MASK_MODE_BITSET
        bits[from_table] = tables.parent_trans_bits[
            tables.parent_offsets[parent_image[from_table]].astype(np.int64) + within[from_table]
        ]
        bits[parent_mode == _MASK_MODE_FULL] = all_translations_words(first.n_coarse_trans)
        row_parts.append(
            dict(
                image=row_image,
                klass=np.full(row_image.size, class_index, dtype=np.int64),
                order=np.arange(row_image.size, dtype=np.int64),
                fine_rot=np.asarray(tables.row_fine_rot, dtype=np.int32),
                parent_local=parent_local,
                log_prior=np.asarray(tables.row_log_prior, dtype=np.float32),
            )
        )
        parent_parts.append(
            dict(image=parent_image, klass=np.full(parent_image.size, class_index), bits=bits, n_parents=n_parents)
        )

    # Parents of an image's earlier classes shift this class's local parent ids.
    class_parent_counts = np.stack([part["n_parents"] for part in parent_parts])  # [K, n_images]
    parent_shift = np.cumsum(class_parent_counts, axis=0) - class_parent_counts
    rows = {key: np.concatenate([part[key] for part in row_parts]) for key in row_parts[0]}
    rows["parent_local"] = rows["parent_local"] + parent_shift[rows["klass"], rows["image"]]
    row_order = np.lexsort((rows["order"], rows["klass"], rows["image"]))
    parents_image = np.concatenate([part["image"] for part in parent_parts])
    parents_class = np.concatenate([part["klass"] for part in parent_parts])
    parents_bits = np.concatenate([part["bits"] for part in parent_parts])
    parent_order = np.lexsort((np.arange(parents_image.size), parents_class, parents_image))

    row_offsets = np.zeros(n_images + 1, dtype=np.int32)
    row_offsets[1:] = np.cumsum(np.bincount(rows["image"], minlength=n_images))
    parent_offsets = np.zeros(n_images + 1, dtype=np.int32)
    parent_offsets[1:] = np.cumsum(class_parent_counts.sum(axis=0))
    return ResidentCandidateTables(
        n_images=n_images,
        n_rows=int(row_offsets[-1]),
        n_fine_trans=int(first.n_fine_trans),
        n_coarse_trans=int(first.n_coarse_trans),
        row_offsets=row_offsets,
        row_image=rows["image"][row_order].astype(np.int32),
        row_fine_rot=rows["fine_rot"][row_order],
        row_parent_local=rows["parent_local"][row_order].astype(np.int32),
        row_log_prior=rows["log_prior"][row_order],
        mask_mode=np.full(n_images, _MASK_MODE_BITSET, dtype=np.int8),
        parent_offsets=parent_offsets,
        parent_trans_bits=parents_bits[parent_order],
        row_class=rows["klass"][row_order].astype(np.int32),
        n_classes=n_classes,
    )


def coarse_winner_cells(
    tables: ResidentCandidateTables,
    coarse_pose_ids,
    *,
    fine_rotation_parent,
    fine_translation_parent,
) -> np.ndarray:
    """Segment-relative cell ``r_local * T + t`` of each image's retained coarse winner.

    Zero oversampling (``--adaptive_oversampling 0``) gives every coarse
    (rotation, translation) exactly one fine child, and RELION's fine pass keeps
    the coarse pass's winner (acc_ml_optimiser_impl.h:3268-3269). This locates
    that child among an image's candidate rows, the resident counterpart of
    :func:`relax.scoring.sparse_bucket_arrays.coarse_winner_local_pose_ids`, and
    refuses a winner that is not a unique, selected candidate.
    """

    n_coarse_trans = int(tables.n_coarse_trans)
    poses = np.asarray(coarse_pose_ids)
    if poses.shape != (tables.n_images,) or not np.all(np.isfinite(poses)):
        raise ValueError("coarse winners must be one finite pose ID per image")
    if np.any(poses < 0) or not np.array_equal(poses, poses.astype(np.int64)):
        raise ValueError("coarse winners must be nonnegative integer pose IDs")
    winner_rot, winner_trans = np.divmod(poses.astype(np.int64), n_coarse_trans)

    fine_translation_parent = np.asarray(fine_translation_parent, dtype=np.int64)
    trans_children = np.bincount(fine_translation_parent, minlength=n_coarse_trans)
    if trans_children.size != n_coarse_trans or np.any(trans_children != 1):
        raise ValueError("zero-oversampling coarse winners need exactly one fine child per coarse translation")
    fine_of_coarse_trans = np.empty(n_coarse_trans, dtype=np.int64)
    fine_of_coarse_trans[fine_translation_parent] = np.arange(fine_translation_parent.size, dtype=np.int64)

    row_image = np.asarray(tables.row_image, dtype=np.int64)
    row_parent_rot = np.asarray(fine_rotation_parent, dtype=np.int64)[np.asarray(tables.row_fine_rot, dtype=np.int64)]
    winner_rows = np.flatnonzero(row_parent_rot == winner_rot[row_image])
    counts = np.bincount(row_image[winner_rows], minlength=tables.n_images)
    if np.any(counts != 1):
        raise ValueError("zero-oversampling coarse winner must have exactly one selected fine child")
    winner_row = np.empty(tables.n_images, dtype=np.int64)
    winner_row[row_image[winner_rows]] = winner_rows

    mode = np.asarray(tables.mask_mode)
    selected = mode == _MASK_MODE_FULL
    bitset = mode == _MASK_MODE_BITSET
    if np.any(bitset):
        images = np.flatnonzero(bitset)
        parent = tables.parent_offsets[images].astype(np.int64) + tables.row_parent_local[winner_row[images]]
        word, bit = translation_word_and_bit(winner_trans[images])
        selected[images] = (tables.parent_trans_bits[parent, word] & bit) != 0
    if not np.all(selected):
        raise ValueError("coarse winner is missing from selected fine support")

    row_local = winner_row - tables.row_offsets[:-1].astype(np.int64)
    return row_local * int(fine_translation_parent.size) + fine_of_coarse_trans[winner_trans]


def expand_mask_rows(tables: ResidentCandidateTables, image: int, fine_translation_parent) -> np.ndarray:
    """Reference (numpy) dense mask for one image; matches ``_candidate_mask_to_dense`` exactly.

    Returns a ``bool[n_rows_i, len(fine_translation_parent)]`` array. This is
    the ground truth :func:`expand_mask_jnp` and the eventual device gather
    kernel must reproduce.
    """

    image = int(image)
    start, stop = int(tables.row_offsets[image]), int(tables.row_offsets[image + 1])
    n_rows_i = stop - start
    fine_translation_parent = np.asarray(fine_translation_parent)
    n_fine_trans = int(fine_translation_parent.shape[0])
    mode = int(tables.mask_mode[image])

    if mode == _MASK_MODE_FULL:
        return np.ones((n_rows_i, n_fine_trans), dtype=bool)
    if mode == _MASK_MODE_EMPTY:
        return np.zeros((n_rows_i, n_fine_trans), dtype=bool)
    if mode != _MASK_MODE_BITSET:
        raise ValueError(f"image {image}: unknown mask_mode {mode}")

    p0, p1 = int(tables.parent_offsets[image]), int(tables.parent_offsets[image + 1])
    bits = tables.parent_trans_bits[p0:p1]
    row_parent = tables.row_parent_local[start:stop]
    row_bits = bits[row_parent].astype(np.uint32)
    words, shifts = np.divmod(fine_translation_parent.astype(np.int64), _MASK_WORD_BITS)
    return (((row_bits[:, words] >> shifts.astype(np.uint32)[None, :]) & np.uint32(1)) != 0)


def expand_mask_jnp(tables: ResidentCandidateTables, image: int, fine_translation_parent):
    """``jax.numpy`` twin of :func:`expand_mask_rows` for the same one image.

    Same contract and same result as :func:`expand_mask_rows` (validated by
    the unit tests bitwise), but written with ``jax.numpy`` primitives so the
    bit-unpacking arithmetic can be copied verbatim into a jitted device
    gather kernel in a later ticket. Runs on whatever platform JAX is
    configured for (CPU in this ticket); nothing here is CUDA-specific.
    """

    import jax.numpy as jnp

    image = int(image)
    start, stop = int(tables.row_offsets[image]), int(tables.row_offsets[image + 1])
    n_rows_i = stop - start
    fine_translation_parent = jnp.asarray(fine_translation_parent, dtype=jnp.uint32)
    n_fine_trans = int(fine_translation_parent.shape[0])
    mode = int(tables.mask_mode[image])

    if mode == _MASK_MODE_FULL:
        return jnp.ones((n_rows_i, n_fine_trans), dtype=bool)
    if mode == _MASK_MODE_EMPTY:
        return jnp.zeros((n_rows_i, n_fine_trans), dtype=bool)
    if mode != _MASK_MODE_BITSET:
        raise ValueError(f"image {image}: unknown mask_mode {mode}")

    p0, p1 = int(tables.parent_offsets[image]), int(tables.parent_offsets[image + 1])
    bits = jnp.asarray(tables.parent_trans_bits[p0:p1], dtype=jnp.uint32)
    row_parent = jnp.asarray(tables.row_parent_local[start:stop], dtype=jnp.int32)
    return _test_translation_bits(bits[row_parent], fine_translation_parent)


def _test_translation_bits(row_bits, fine_translation_parent):
    """bool[rows, T]: bit ``fine_translation_parent[t]`` of each row's word bitset."""

    import jax.numpy as jnp

    words = (fine_translation_parent // jnp.uint32(_MASK_WORD_BITS)).astype(jnp.int32)
    shifts = fine_translation_parent % jnp.uint32(_MASK_WORD_BITS)
    return ((row_bits[:, words] >> shifts[None, :]) & jnp.uint32(1)) != 0


def expand_chunk_mask_jnp(row_mask_bits, row_mask_mode, fine_translation_parent):
    """Dense candidate mask of one padded chunk, from its per-row mask fields.

    ``jax.numpy`` twin of :func:`expand_mask_rows` evaluated for a whole
    :func:`materialize_chunk` output at once, so a jitted device program can
    rebuild the ``bool[row_capacity, n_fine_trans]`` mask without any host
    array. Inputs are the chunk fields ``row_mask_bits`` (uint32
    ``[row_capacity, n_words]``) and ``row_mask_mode`` (int8 ``[row_capacity]``), plus
    the iteration-global ``fine_translation_parent`` (int32
    ``[n_fine_trans]``).

    Mask modes follow the module vocabulary: ``0`` accepts every translation,
    ``1`` tests bit ``fine_translation_parent[t]`` of that row's bitset and
    ``2`` rejects every translation. Padded rows carry mode ``2``, so they are
    all-false whatever their other fields hold.
    """

    import jax.numpy as jnp

    row_mask_bits = jnp.asarray(row_mask_bits, dtype=jnp.uint32)
    row_mask_mode = jnp.asarray(row_mask_mode, dtype=jnp.int8)
    fine_translation_parent = jnp.asarray(fine_translation_parent, dtype=jnp.uint32)
    if row_mask_bits.ndim != 2 or row_mask_mode.shape != row_mask_bits.shape[:1]:
        raise ValueError(
            "row_mask_bits must be [rows, words] and row_mask_mode [rows], got "
            f"{row_mask_bits.shape} and {row_mask_mode.shape}",
        )
    if fine_translation_parent.ndim != 1:
        raise ValueError(
            f"fine_translation_parent must be 1-D, got {fine_translation_parent.shape}",
        )

    bitset = _test_translation_bits(row_mask_bits, fine_translation_parent)
    is_full = (row_mask_mode == _MASK_MODE_FULL)[:, None]
    is_empty = (row_mask_mode == _MASK_MODE_EMPTY)[:, None]
    return jnp.where(is_full, True, jnp.where(is_empty, False, bitset))


def _smallest_fit(ladder: tuple[int, ...], value: int) -> int | None:
    for capacity in ladder:
        if value <= capacity:
            return capacity
    return None


def plan_capacity_chunks(
    tables: ResidentCandidateTables,
    *,
    row_capacity_ladder=(8192, 32768, 131072, 524288),
    image_capacity_ladder=(32, 128, 512),
) -> list[CapacityChunk]:
    """Greedily group images (in image order) into fixed-capacity chunks.

    Images are never reordered (RELION particle order is preserved for the
    x-half BPref). The chunker grows a chunk one image at a time while both
    its row count and its image count still fit some (row_capacity,
    image_capacity) pair from the ladders, then closes it and starts the next
    chunk at the first image that no longer fits. A single image whose own
    row count exceeds the largest row class becomes a one-image chunk whose
    row capacity is rounded up to the smallest multiple of the largest row
    class that covers it (its image capacity is the smallest image class,
    i.e. the ladder's first entry, since one image always fits there).
    """

    row_capacity_ladder = tuple(int(v) for v in row_capacity_ladder)
    image_capacity_ladder = tuple(int(v) for v in image_capacity_ladder)
    if not row_capacity_ladder or not image_capacity_ladder:
        raise ValueError("capacity ladders must be non-empty")
    if list(row_capacity_ladder) != sorted(row_capacity_ladder):
        raise ValueError("row_capacity_ladder must be increasing")
    if list(image_capacity_ladder) != sorted(image_capacity_ladder):
        raise ValueError("image_capacity_ladder must be increasing")

    row_offsets = tables.row_offsets
    n_images = tables.n_images
    chunks: list[CapacityChunk] = []
    i = 0
    while i < n_images:
        row_start = int(row_offsets[i])
        best_row_cap = None
        best_img_cap = None
        best_stop = None
        j = i
        while j < n_images:
            candidate_images = j - i + 1
            candidate_rows = int(row_offsets[j + 1]) - row_start
            row_cap = _smallest_fit(row_capacity_ladder, candidate_rows)
            img_cap = _smallest_fit(image_capacity_ladder, candidate_images)
            if row_cap is None or img_cap is None:
                break
            best_row_cap, best_img_cap, best_stop = row_cap, img_cap, j + 1
            j += 1

        if best_stop is None:
            # Even the lone first image overflows the largest row class.
            row_stop = int(row_offsets[i + 1])
            candidate_rows = row_stop - row_start
            largest_row = row_capacity_ladder[-1]
            row_cap = -(-candidate_rows // largest_row) * largest_row  # ceil to a multiple
            chunks.append(
                CapacityChunk(
                    image_start=i,
                    image_stop=i + 1,
                    row_start=row_start,
                    row_stop=row_stop,
                    row_capacity=row_cap,
                    image_capacity=image_capacity_ladder[0],
                ),
            )
            i += 1
        else:
            chunks.append(
                CapacityChunk(
                    image_start=i,
                    image_stop=best_stop,
                    row_start=row_start,
                    row_stop=int(row_offsets[best_stop]),
                    row_capacity=best_row_cap,
                    image_capacity=best_img_cap,
                ),
            )
            i = best_stop

    return chunks


def materialize_chunk(tables: ResidentCandidateTables, chunk: CapacityChunk) -> dict:
    """Gather one chunk's rows into padded, capacity-shaped numpy arrays.

    Padding contract (padded rows must never validate any (row, t) cell):
    ``row_image_local`` pads to ``image_capacity - 1`` (a padded image slot,
    see ``image_ids`` below), ``row_fine_rot``/``row_parent_local`` pad to 0,
    ``row_log_prior`` pads to -1e30, ``row_mask_bits`` pads to 0 and
    ``row_mask_mode`` pads to 2 (empty) -- the mode alone is sufficient to
    invalidate every padded cell regardless of bits, so padding the other
    fields to 0 is a safety margin, not a correctness requirement.
    """

    row_capacity = int(chunk.row_capacity)
    image_capacity = int(chunk.image_capacity)
    n_valid_rows = chunk.n_valid_rows
    n_valid_images = chunk.n_valid_images
    if n_valid_rows > row_capacity:
        raise ValueError(f"chunk has {n_valid_rows} valid rows but capacity {row_capacity}")
    if n_valid_images > image_capacity:
        raise ValueError(f"chunk has {n_valid_images} valid images but capacity {image_capacity}")

    rs, re = chunk.row_start, chunk.row_stop

    row_image_local = np.full(row_capacity, image_capacity - 1, dtype=np.int32)
    row_fine_rot = np.full(row_capacity, _ROW_FINE_ROT_PAD, dtype=np.int32)
    row_parent_local = np.full(row_capacity, _ROW_PARENT_LOCAL_PAD, dtype=np.int32)
    row_log_prior = np.full(row_capacity, _ROW_LOG_PRIOR_PAD, dtype=np.float32)
    n_words = n_mask_words(tables.n_coarse_trans)
    row_mask_bits = np.full((row_capacity, n_words), _ROW_MASK_BITS_PAD, dtype=np.uint32)
    row_mask_mode = np.full(row_capacity, _MASK_MODE_EMPTY, dtype=np.int8)
    row_class = np.zeros(row_capacity, dtype=np.int32)
    image_ids = np.full(image_capacity, -1, dtype=np.int32)

    if n_valid_rows:
        row_image_global = tables.row_image[rs:re]
        row_image_local[:n_valid_rows] = row_image_global - chunk.image_start
        row_fine_rot[:n_valid_rows] = tables.row_fine_rot[rs:re]
        row_parent_local_valid = tables.row_parent_local[rs:re]
        row_parent_local[:n_valid_rows] = row_parent_local_valid
        row_log_prior[:n_valid_rows] = tables.row_log_prior[rs:re]
        if tables.row_class is not None:
            row_class[:n_valid_rows] = tables.row_class[rs:re]

        row_mode_valid = tables.mask_mode[row_image_global]
        row_mask_mode[:n_valid_rows] = row_mode_valid

        bits_out = np.zeros((n_valid_rows, n_words), dtype=np.uint32)
        bitset_rows = row_mode_valid == _MASK_MODE_BITSET
        if np.any(bitset_rows):
            flat_idx = tables.parent_offsets[row_image_global[bitset_rows]] + row_parent_local_valid[bitset_rows]
            bits_out[bitset_rows] = tables.parent_trans_bits[flat_idx]
        row_mask_bits[:n_valid_rows] = bits_out

    if n_valid_images:
        image_ids[:n_valid_images] = np.arange(chunk.image_start, chunk.image_stop, dtype=np.int32)

    return {
        "row_image_local": row_image_local,
        "row_fine_rot": row_fine_rot,
        "row_parent_local": row_parent_local,
        "row_log_prior": row_log_prior,
        "row_mask_bits": row_mask_bits,
        "row_mask_mode": row_mask_mode,
        "row_class": row_class,
        "n_valid_rows": np.int32(n_valid_rows),
        "n_valid_images": np.int32(n_valid_images),
        "image_ids": image_ids,
    }
