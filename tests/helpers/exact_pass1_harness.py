"""A CPU harness for pass 1 on RELION's exact coarse operands.

Production pass 1 runs RELION's CUDA preprocessing, translation and projector; these
helpers stand them in on CPU (a tiny strict-preprocess dataset, unit CTFs, no
high-resolution image power, translation as repetition and a coded projector) so the
pass-1 program, its posterior and its outputs can be tested without a GPU.
"""

import jax.numpy as jnp
import numpy as np

from relax.relion import relion_ctf


class ExactPass1Dataset:
    """Tiny strict-preprocess dataset for the live shared significance path.

    Image ``i`` (original index ``i``) is the constant ``i + 1``; its half-spectrum
    preprocessing repeats that value over the ``N (N // 2 + 1)`` half-spectrum pixels of
    an ``N x N`` image (``N = box``, 4 by default).
    """

    padding = 0
    voxel_size = 1.0
    dtype = jnp.complex64
    premultiplied_ctf = False

    def __init__(self, original_indices=None, box=4):
        self.box = int(box)
        self.image_shape = (self.box, self.box)
        self.image_size = self.box * self.box
        self.grid_size = self.box
        self.volume_shape = (self.box,) * 3
        self.volume_size = self.box**3
        self._original_indices = np.asarray(
            [0, 1, 2] if original_indices is None else original_indices,
            dtype=np.int64,
        )
        self.n_images = int(self._original_indices.size)
        self.n_units = self.n_images
        self._images = np.stack(
            [
                np.full(self.image_shape, int(original_index) + 1, dtype=np.float32)
                for original_index in self._original_indices
            ]
        )
        self.CTF_params = np.zeros((self.n_units, 9), dtype=np.float32)
        self.rotation_matrices = np.tile(np.eye(3, dtype=np.float32), (self.n_units, 1, 1))
        self.translations = np.zeros((self.n_units, 2), dtype=np.float32)

        class _Backend:
            image_mask = np.ones(self.image_shape, dtype=np.float32)
            image_mask_mode = "relion_background_fill"
            relion_fourier_backend = "relion_cuda"

        class _ImageSource:
            backend = _Backend()

        self.image_source = _ImageSource()

    @staticmethod
    def ctf_evaluator(params, image_shape=None, voxel_size=None, *, half_image=False):
        del voxel_size
        if half_image:
            pixel_count = int(image_shape[0]) * (int(image_shape[1]) // 2 + 1)
        else:
            pixel_count = int(image_shape[0]) * int(image_shape[1])
        return jnp.ones((params.shape[0], pixel_count), dtype=jnp.float32)

    def process_images(self, batch, apply_image_mask=False, **kwargs):
        del apply_image_mask, kwargs
        batch = jnp.asarray(batch)
        return jnp.repeat(
            batch[:, :1, :1].reshape(batch.shape[0], 1).astype(jnp.complex64),
            self.image_size,
            axis=1,
        )

    def process_images_half(self, batch, apply_image_mask=False, **kwargs):
        del apply_image_mask, kwargs
        batch = jnp.asarray(batch)
        return jnp.repeat(
            batch[:, :1, :1].reshape(batch.shape[0], 1).astype(jnp.complex64),
            self.box * (self.box // 2 + 1),
            axis=1,
        )

    @property
    def image_mask(self):
        return np.ones(self.image_shape, dtype=np.float32)

    def iter_batches(self, batch_size, *, indices=None, by_image=False, **kwargs):
        del by_image, kwargs
        if indices is None:
            indices = np.arange(self.n_units)
        indices = np.asarray(indices, dtype=np.int64)
        for start in range(0, indices.size, int(batch_size)):
            selected = indices[start : start + int(batch_size)]
            yield (
                jnp.asarray(self._images[selected]),
                self.rotation_matrices[selected],
                self.translations[selected],
                jnp.asarray(self.CTF_params[selected]),
                None,
                selected,
                selected,
            )

    def original_image_indices_from_local(self, indices):
        return self._original_indices[np.asarray(indices, dtype=np.int64)]

    def subset(self, indices):
        return type(self)(
            self._original_indices[np.asarray(indices, dtype=np.int64)],
            box=self.box,
        )


def mock_unit_ctf_and_zero_highres_power(monkeypatch):
    """Use unit CTFs and no high-resolution image power in live-path CPU tests."""
    from relax.sparse_pass2 import sparse_pass2_scoring

    monkeypatch.setattr(
        relion_ctf,
        "_relion_exact_ctf_half_from_source_star",
        lambda _dataset, indices, image_shape, *, pixel_indices=None: jnp.ones(
            (
                len(indices),
                (int(image_shape[0]) * (int(image_shape[1]) // 2 + 1) if pixel_indices is None else len(pixel_indices)),
            ),
            dtype=jnp.float64,
        ),
    )
    monkeypatch.setattr(
        relion_ctf,
        "_relion_exact_ctf_half_from_source_star_host",
        lambda _dataset, indices, image_shape, *, pixel_indices=None: np.ones(
            (
                len(indices),
                (int(image_shape[0]) * (int(image_shape[1]) // 2 + 1) if pixel_indices is None else len(pixel_indices)),
            ),
            dtype=np.float64,
        ),
    )
    monkeypatch.setattr(
        sparse_pass2_scoring,
        "_relion_cuda_powerclass_highres_xi2_half",
        lambda processed, **_kwargs: jnp.zeros(
            processed.shape[0],
            dtype=jnp.float32,
        ),
    )


def coded_class_projectors(n_classes):
    """``[K, 3, 3, 2]`` RELION projector halves (r_max 1) whose entries are the class index."""

    return jnp.stack([jnp.full((3, 3, 2), class_index, dtype=jnp.complex64) for class_index in range(n_classes)])


def install_exact_pass1_mocks(monkeypatch):
    """Run pass 1 on CPU: its CUDA check passes, CTFs are one, translation repeats the image and the
    projector returns, for class ``k`` and a rotation whose ``[0, 1]`` entry is ``c``, the row
    ``(k + 1 + c) * exp(0.3 i p)`` over the requested pixels ``p``."""

    from relax.cuda import kernels as em_cuda_kernels
    from relax.helpers import projection as projection_helpers
    from relax.scoring import gaussian_plan

    monkeypatch.setattr(gaussian_plan, "_custom_cuda_ready", lambda: True)
    mock_unit_ctf_and_zero_highres_power(monkeypatch)
    monkeypatch.setattr(
        em_cuda_kernels,
        "relion_translate_score_f32",
        lambda images, translation_angles, pixel_indices, image_shape: jnp.repeat(
            images[:, None, :],
            int(translation_angles.shape[0]),
            axis=1,
        ).reshape(images.shape[0] * int(translation_angles.shape[0]), -1),
    )

    def coded_projection(projector_half, rotations_block, image_shape, **kwargs):
        class_value = float(np.asarray(projector_half)[0, 0, 0].real)
        codes = class_value + 1.0 + np.asarray(rotations_block)[:, 0, 1].astype(np.float64)
        pixel_indices = kwargs.get("pixel_indices")
        n_pixels = (
            int(image_shape[0]) * (int(image_shape[1]) // 2 + 1) if pixel_indices is None else len(pixel_indices)
        )
        pixels = np.arange(n_pixels)
        projected = jnp.asarray(codes[:, None] * np.exp(0.3j * pixels)[None, :], dtype=jnp.complex64)
        return projected, (jnp.abs(projected) ** 2 if kwargs.get("return_abs2", True) else None)

    monkeypatch.setattr(projection_helpers, "compute_relion_projector_projections_block", coded_projection)
