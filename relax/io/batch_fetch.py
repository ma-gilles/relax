"""Dataset batch fetch helpers shared by dense/local EM paths."""

from __future__ import annotations

import threading
from contextlib import contextmanager

import numpy as np

# A half's raw images held on the host for one exact-local call (its pass-1 probe, then the fine pass over the
# same images: relax#96), up to this many bytes; a larger half rereads them (100k images at 256 px are 13 GB).
HELD_HALF_IMAGES_MAX_BYTES = 8 << 30

# The held images by image source. A registry, not a context variable: the fine pass reads in a prefetch thread.
_HELD_HALF_IMAGES: dict[int, "_HeldHalfImages"] = {}
_HELD_HALF_IMAGES_LOCK = threading.Lock()


class _HeldHalfImages:
    """The raw host images of one half read so far, by local image index; disabled past the byte cap."""

    def __init__(self, source, n_images: int, max_bytes: int):
        self.source = source
        self.n_images = n_images
        self.max_bytes = max_bytes
        self.images = None
        self.held = None
        self.disabled = False
        self.lock = threading.Lock()

    def _holds_range(self, image_indices) -> bool:
        return image_indices.size > 0 and int(image_indices.min()) >= 0 and int(image_indices.max()) < self.n_images

    def take(self, image_indices):
        """Copies of the held images at ``image_indices``, or None when any of them was not read yet."""

        with self.lock:
            if self.images is None or self.disabled or not self._holds_range(image_indices):
                return None
            if not np.all(self.held[image_indices]):
                return None
            return self.images[image_indices]

    def put(self, image_indices, images) -> None:
        with self.lock:
            if self.disabled or not self._holds_range(image_indices) or images.shape[0] != image_indices.size:
                return
            if self.images is None:
                if self.n_images * int(images[0].nbytes) > self.max_bytes:
                    self.disabled = True
                    return
                self.images = np.empty((self.n_images,) + tuple(images.shape[1:]), dtype=images.dtype)
                self.held = np.zeros(self.n_images, dtype=bool)
            self.images[image_indices] = images
            self.held[image_indices] = True


@contextmanager
def held_half_images(experiment_dataset, *, max_bytes: int = HELD_HALF_IMAGES_MAX_BYTES):
    """Serve repeated reads of this half's raw images from host memory inside the scope (relax#96).

    The exact-local route reads a half's images for the pass-1 probe and again for the fine pass. Inside
    this scope :func:`fetch_indexed_batch` keeps every image it reads from this dataset's image source
    and serves a later read of the same images from that copy: the bytes the source returned, so the
    passes compute exactly what they computed before. A half above ``max_bytes`` reads as before. Only the
    vectorized host path is held (single-particle sources with ``host_images``).
    """

    source = getattr(experiment_dataset, "image_source", None)
    if source is None or getattr(source, "host_images", None) is None or getattr(source, "tilt_series", False):
        yield
        return
    key = id(source)
    with _HELD_HALF_IMAGES_LOCK:
        if key in _HELD_HALF_IMAGES:
            raise RuntimeError("this image source already has a held-images scope")
        _HELD_HALF_IMAGES[key] = _HeldHalfImages(source, int(experiment_dataset.n_units), max_bytes)
    try:
        yield
    finally:
        with _HELD_HALF_IMAGES_LOCK:
            _HELD_HALF_IMAGES.pop(key, None)


def _held_half_images_of(source):
    with _HELD_HALF_IMAGES_LOCK:
        held = _HELD_HALF_IMAGES.get(id(source))
    return held if held is not None and held.source is source else None


def fetch_indexed_batch(experiment_dataset, image_indices):
    """Fetch one explicitly indexed image batch and return dataset indices.

    Single-particle sources serve the batch's host images in one vectorized
    read (``host_images``) and the CTF rows from the metadata. The dataset's
    batch iterator reads and collates every image as its own item through a
    thread pool: 340 us per image against 40 us for the vectorized read (5k
    128-px stack), about 0.9 s per half-set on each of the resident operand
    preparations. Sources without a host path, and tilt series, keep the
    iterator. Inside :func:`held_half_images` the host images of a repeated
    read come from the held copy.
    """

    image_indices = np.asarray(image_indices)
    source = getattr(experiment_dataset, "image_source", None)
    host_images = getattr(source, "host_images", None)
    if host_images is not None and not getattr(source, "tilt_series", False):
        held = _held_half_images_of(source)
        images = None if held is None else held.take(image_indices)
        if images is None:
            try:
                images = host_images(image_indices)
            except NotImplementedError:
                images = None
            if images is not None and held is not None:
                held.put(image_indices, np.asarray(images))
        if images is not None:
            _, _, ctf_params = experiment_dataset.metadata.get_batch(image_indices)
            return np.asarray(images), ctf_params, image_indices.astype(np.int32, copy=True)
    return fetch_indexed_batch_via_iterator(experiment_dataset, image_indices)


def iter_indexed_batches(experiment_dataset, image_indices, batch_size):
    """Yield the dataset iterator's batch tuples for ``image_indices``, fetched per batch.

    Each item is ``(images, None, None, ctf_params, None, None, indices)``, the
    fields of ``CryoEMDataset.iter_batches`` that indexed consumers read, with
    the batch served by :func:`fetch_indexed_batch` (one vectorized host read
    for single-particle sources).
    """

    image_indices = np.asarray(image_indices)
    for start in range(0, int(image_indices.shape[0]), int(batch_size)):
        images, ctf_params, indices = fetch_indexed_batch(
            experiment_dataset, image_indices[start : start + int(batch_size)]
        )
        yield images, None, None, ctf_params, None, None, indices


def fetch_indexed_batch_via_iterator(experiment_dataset, image_indices):
    """The dataset batch iterator's route for one explicitly indexed batch."""

    batch_iter = experiment_dataset.iter_batches(
        len(image_indices),
        indices=np.asarray(image_indices),
        by_image=False,
    )
    batch_data, _, _, ctf_params, _, _, indices = next(batch_iter)
    return batch_data, ctf_params, np.asarray(indices)


def original_image_indices(experiment_dataset, local_indices) -> np.ndarray:
    """Map local batch image indices to original image ids for debug dumps."""
    local_indices = np.asarray(local_indices, dtype=np.int64)
    mapper = getattr(experiment_dataset, "original_image_indices_from_local", None)
    if mapper is not None:
        return np.asarray(mapper(local_indices), dtype=np.int64)
    original_indices_all = getattr(experiment_dataset, "dataset_indices", None)
    if original_indices_all is None:
        return local_indices
    return np.asarray(original_indices_all, dtype=np.int64)[local_indices]


@contextmanager
def prefetched_batches(iterable, *, depth: int):
    """Own an ordered, bounded batch iterator and close it on every exit.

    Reuse the dataset loader's cancellable queue: closing a consumer must not
    leave a producer blocked on a full queue with device batches retained.
    Depth zero preserves synchronous iteration. Source values and exceptions
    retain their order; the consumer owns the lifetime explicitly.
    """
    from recovar.data_io.image_backends import PrefetchIterator as _PrefetchIterator

    if depth < 0:
        raise ValueError("prefetch depth must be non-negative")
    iterator = iter(_PrefetchIterator(iterable, buffer_size=depth)) if depth else iter(iterable)
    try:
        yield iterator
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
