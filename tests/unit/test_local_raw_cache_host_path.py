"""The local raw-image cache must not change when it stops going through JAX.

`_fetch_local_raw_rows_once` used to route every image through
`CryoEMDataset.iter_batches`, which collates each item into a JAX array on the
device; the caller then pulled it straight back to host NumPy. Under
``RECOVAR_PREREAD_IMAGES`` the loader already holds those images in host memory,
so the round trip bought nothing and cost about 270 s of a 6673 s K=1 100k/256
run at 200 mini-batches -- 4.1%.

The fast path asks the image source for host images directly. These tests pin
the property that makes it safe to prefer: identical images, identical CTF rows,
identical index bookkeeping, for arbitrary index order and repeats -- checked
against the original route rather than against my reading of it. They also pin
that a source without a host path still works.
"""

import numpy as np
import pytest

pytest.importorskip("jax")

import recovar.data_io.cryoem_dataset as dataset  # noqa: E402

from relax.io.batch_fetch import fetch_indexed_batch  # noqa: E402

pytestmark = pytest.mark.unit

N_IMAGES = 6
D = 4


class _HostImageStack:
    """Minimal non-tilt backend serving distinguishable host images."""

    def __init__(self, n_images=N_IMAGES, d=D):
        self.n_images = n_images
        self.D = d
        self.unpadded_D = d
        self.padding = 0
        self.image_shape = (d, d)
        self.Np = n_images
        self.mask = np.ones((d, d), dtype=np.float32)
        self.particles = [np.asarray([i], dtype=np.int32) for i in range(n_images)]
        self.dataset_tilt_indices = np.arange(n_images, dtype=np.int32)
        # Every image is a distinct constant, so a mis-ordered gather cannot pass.
        self._images = np.arange(n_images, dtype=np.float32)[:, None, None] * np.ones(
            (1, d, d), dtype=np.float32
        )

    def __getitem__(self, index):
        idx = np.atleast_1d(np.asarray(index, dtype=np.int32))
        return self._images[idx], idx, idx

    def get_image_generator(self, batch_size, num_workers=0):
        _ = (batch_size, num_workers)
        idx = np.arange(self.n_images, dtype=np.int32)
        yield self._images, idx, idx

    def get_image_subset_generator(self, batch_size, subset_indices, num_workers=0):
        _ = (batch_size, num_workers)
        idx = np.asarray(subset_indices, dtype=np.int32)
        yield self._images[idx], idx, idx

    def get_dataset_generator(self, batch_size, num_workers=0, **kwargs):
        yield from self.get_image_generator(batch_size, num_workers)

    def get_dataset_subset_generator(self, batch_size, subset_indices, num_workers=0, **kwargs):
        yield from self.get_image_subset_generator(batch_size, subset_indices, num_workers)

    def process_images(self, images, apply_image_mask=False):
        return images


def _dataset():
    source = dataset.BackendImageSource(
        _HostImageStack(), info=dataset.ImageSourceInfo(tilt_series=False),
    )
    rots = np.tile(np.eye(3, dtype=np.float32), (N_IMAGES, 1, 1))
    trans = np.zeros((N_IMAGES, 2), dtype=np.float32)
    ctf = np.arange(N_IMAGES * 9, dtype=np.float32).reshape(N_IMAGES, 9)
    return dataset.CryoEMDataset(
        image_source=source,
        voxel_size=1.0,
        metadata=dataset.ImageMetadata(rots, trans, ctf),
        dataset_indices=np.arange(N_IMAGES, dtype=np.int32),
        tilt_series_flag=False,
    )


@pytest.mark.parametrize(
    "indices",
    [
        np.arange(N_IMAGES, dtype=np.int32),
        np.asarray([3, 0, 5, 1], dtype=np.int32),
        np.asarray([2, 2, 4, 2], dtype=np.int32),
        np.asarray([4], dtype=np.int32),
    ],
)
def test_fetch_indexed_batch_host_path_matches_the_iterator(indices):
    """fetch_indexed_batch's vectorized host read equals the batch iterator's route."""

    from relax.io.batch_fetch import fetch_indexed_batch_via_iterator

    cryo = _dataset()
    images, ctf, fetched = fetch_indexed_batch(cryo, indices)
    assert isinstance(images, np.ndarray)
    reference_images, reference_ctf, reference_indices = fetch_indexed_batch_via_iterator(cryo, indices)
    lookup = {int(i): pos for pos, i in enumerate(np.asarray(reference_indices))}
    np.testing.assert_array_equal(np.asarray(fetched), indices)
    assert np.asarray(fetched).dtype == np.int32
    for row, index in enumerate(np.asarray(fetched)):
        np.testing.assert_array_equal(images[row], np.asarray(reference_images)[lookup[int(index)]])
        np.testing.assert_array_equal(np.asarray(ctf)[row], np.asarray(reference_ctf)[lookup[int(index)]])


def test_fetch_indexed_batch_without_a_host_path_uses_the_iterator(monkeypatch):
    cryo = _dataset()
    monkeypatch.delattr(type(cryo.image_source), "host_images", raising=True)
    indices = np.asarray([1, 3], dtype=np.int32)
    images, ctf, fetched = fetch_indexed_batch(cryo, indices)
    assert np.asarray(images).shape[0] == indices.size
    np.testing.assert_array_equal(np.sort(np.asarray(fetched)), np.sort(indices))


def _counted_host_reads(cryo):
    """Count the image source's host reads (the held images must replace the repeated ones)."""

    reads = []
    original = cryo.image_source.host_images

    def host_images(image_indices):
        reads.append(np.asarray(image_indices).copy())
        return original(image_indices)

    cryo.image_source.host_images = host_images
    return reads


def test_held_half_images_serve_a_repeated_read_with_the_same_bytes():
    from relax.io.batch_fetch import held_half_images

    cryo = _dataset()
    reads = _counted_host_reads(cryo)
    first_indices = np.asarray([3, 0, 5], dtype=np.int32)
    with held_half_images(cryo):
        first = fetch_indexed_batch(cryo, first_indices)
        again = fetch_indexed_batch(cryo, np.asarray([5, 3], dtype=np.int32))
        # One image not read yet: the whole batch comes from the source.
        mixed = fetch_indexed_batch(cryo, np.asarray([0, 1], dtype=np.int32))
    after = fetch_indexed_batch(cryo, np.asarray([5, 3], dtype=np.int32))
    assert [r.tolist() for r in reads] == [[3, 0, 5], [0, 1], [5, 3]]
    np.testing.assert_array_equal(again[0], first[0][[2, 0]])
    assert again[0].dtype == first[0].dtype
    np.testing.assert_array_equal(again[0], after[0])
    np.testing.assert_array_equal(np.asarray(again[1]), np.asarray(after[1]))
    np.testing.assert_array_equal(again[2], np.asarray([5, 3], dtype=np.int32))
    np.testing.assert_array_equal(mixed[0], fetch_indexed_batch(cryo, np.asarray([0, 1], dtype=np.int32))[0])


def test_held_half_images_above_the_byte_cap_reread():
    from relax.io.batch_fetch import held_half_images

    cryo = _dataset()
    reads = _counted_host_reads(cryo)
    indices = np.asarray([1, 2], dtype=np.int32)
    # N_IMAGES float32 D x D images do not fit under one image's bytes.
    with held_half_images(cryo, max_bytes=D * D * 4):
        first = fetch_indexed_batch(cryo, indices)
        second = fetch_indexed_batch(cryo, indices)
    assert len(reads) == 2
    np.testing.assert_array_equal(first[0], second[0])


def test_held_half_images_serve_a_read_from_another_thread():
    """The fine pass reads its operands in a prefetch thread; the hold is not thread-local."""

    import threading

    from relax.io.batch_fetch import held_half_images

    cryo = _dataset()
    reads = _counted_host_reads(cryo)
    indices = np.asarray([4, 2], dtype=np.int32)
    out = {}
    with held_half_images(cryo):
        first = fetch_indexed_batch(cryo, indices)
        worker = threading.Thread(target=lambda: out.setdefault("images", fetch_indexed_batch(cryo, indices)[0]))
        worker.start()
        worker.join()
    assert len(reads) == 1
    np.testing.assert_array_equal(out["images"], first[0])
