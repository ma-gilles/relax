"""Dataset batch fetch helpers shared by dense/local EM paths."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np


def fetch_indexed_batch(experiment_dataset, image_indices):
    """Fetch one explicitly indexed image batch and return dataset indices.

    Single-particle sources serve the batch's host images in one vectorized
    read (``host_images``) and the CTF rows from the metadata. The dataset's
    batch iterator reads and collates every image as its own item through a
    thread pool: 340 us per image against 40 us for the vectorized read (5k
    128-px stack), about 0.9 s per half-set on each of the resident operand
    preparations. Sources without a host path, and tilt series, keep the
    iterator.
    """

    image_indices = np.asarray(image_indices)
    source = getattr(experiment_dataset, "image_source", None)
    host_images = getattr(source, "host_images", None)
    if host_images is not None and not getattr(source, "tilt_series", False):
        try:
            images = host_images(image_indices)
        except NotImplementedError:
            images = None
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
