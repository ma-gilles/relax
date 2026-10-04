"""A tiny K=1 / K-class refinement on the CPU stand-in engine, for tests of what the controller hands its sites.

``run_tiny_refinement`` runs ``refine_single_volume`` on 8-pixel mock half sets with
``helpers.fake_adaptive_engine`` in place of the global E-step (its pass 2 needs a GPU), through the
numbered iterations and, for K=1, the final all-data pass. ``record_calls`` wraps a callee so a test
can read the arguments each call received.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from helpers.em_arrays import _hermitian_volume
from helpers.fake_adaptive_engine import install_fake_adaptive_engine

IMAGE_SHAPE = (8, 8)
IMAGE_SIZE = 64
VOLUME_SHAPE = (8, 8, 8)
VOLUME_SIZE = 512
N_IMAGES = 10
N_ROTATIONS = 5


def _identity_ctf(params, image_shape=None, voxel_size=None, *, half_image=False):
    h, w = IMAGE_SHAPE if image_shape is None else image_shape
    return jnp.ones((params.shape[0], h * (w // 2 + 1) if half_image else h * w), dtype=jnp.float32)


def _identity_process(batch, apply_image_mask=False):
    return batch


def _identity_process_half(batch, apply_image_mask=False):
    from recovar.core import fourier_transform_utils as ftu

    return ftu.full_image_to_half_image(batch, IMAGE_SHAPE)


class MockHalfSet:
    """The subset of the dataset API the refinement loop reads, over random Hermitian images."""

    def __init__(self, n_images, rng):
        self.image_shape = IMAGE_SHAPE
        self.image_size = IMAGE_SIZE
        self.grid_size = IMAGE_SHAPE[0]
        self.volume_shape = VOLUME_SHAPE
        self.volume_size = VOLUME_SIZE
        self.n_images = self.n_units = n_images
        self.voxel_size = 1.0
        self.dtype = jnp.complex64
        self.CTF_params = np.zeros((n_images, 9), dtype=np.float32)
        self.ctf_evaluator = staticmethod(_identity_ctf)
        self.process_images = staticmethod(_identity_process)
        self.process_images_half = staticmethod(_identity_process_half)
        self.premultiplied_ctf = False
        real = rng.standard_normal((n_images, *IMAGE_SHAPE)).astype(np.float32)
        self._images = np.fft.fftshift(np.fft.fft2(real), axes=(-2, -1)).reshape(n_images, -1).astype(np.complex64)
        self.rotation_matrices = np.tile(np.eye(3, dtype=np.float32), (n_images, 1, 1))
        self.translations = np.zeros((n_images, 2), dtype=np.float32)

        class _ImageSource:
            process_images = staticmethod(_identity_process)

        self.image_source = _ImageSource()

    def iter_batches(self, batch_size, *, indices=None, by_image=False, **kwargs):
        indices = np.arange(self.n_images) if indices is None else np.asarray(indices)
        for start in range(0, len(indices), max(1, batch_size)):
            idx = indices[start : start + max(1, batch_size)]
            yield (
                jnp.asarray(self._images[idx]),
                self.rotation_matrices[idx],
                self.translations[idx],
                jnp.asarray(self.CTF_params[idx]),
                None,
                idx,
                idx,
            )

    def update_poses(self, rots, trans):
        self.rotation_matrices = np.asarray(rots)
        self.translations = np.asarray(trans)

    def get_valid_frequency_indices(self, pixel_res):
        return np.ones(self.volume_size, dtype=bool)

    def original_image_indices_from_local(self, indices=None):
        return np.arange(self.n_images, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)


def record_calls(monkeypatch, owner, name):
    """Replace ``owner.name`` by a recorder that forwards to it; returns the ``(args, kwargs)`` list."""

    calls = []
    original = getattr(owner, name)

    def recorder(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, recorder)
    return calls


def run_tiny_refinement(monkeypatch, *, parity=None, n_classes=1, max_iter=2, engine_calls=None, **option_groups):
    """Run the controller for ``max_iter`` numbered iterations and (K=1) the final all-data pass.

    ``parity`` is a mapping of ``RelionParityOptions`` fields; ``option_groups`` are further
    ``RefinementOptions`` groups. ``engine_calls`` receives the fake engine's call records.
    """

    import relax.sampling as sampling
    from relax.refinement import iteration_loop
    from relax.refinement.refinement_options import (
        AdaptiveOptions,
        KClassOptions,
        RefinementBatching,
        RefinementOptions,
        RefinementSchedule,
        RelionParityOptions,
    )

    calls = [] if engine_calls is None else engine_calls

    def random_half_map(k, size):
        side = round(size ** (1.0 / 3.0))
        return _hermitian_volume((side, side, side), seed=1000 + len(calls) * 7 + k)

    install_fake_adaptive_engine(monkeypatch, calls, Ft_y=random_half_map)

    def identity_rotation_grid(order, dtype=None, *, symmetry="C1"):
        n_rotations = iteration_loop.rotation_grid_size(order, symmetry=symmetry)
        return sampling.RotationGrid(
            rotations=np.repeat(np.eye(3, dtype=np.float32)[None], n_rotations, axis=0),
            rotation_eulers=np.zeros((n_rotations, 3), dtype=np.float32),
            healpix_order=order,
            symmetry=symmetry,
        )

    monkeypatch.setattr(sampling, "relion_scoring_rotation_grid", identity_rotation_grid)
    for name in ("RELAX_PARITY_DUMP_DIR", "RELAX_PARITY_TIMING_DIR"):
        monkeypatch.delenv(name, raising=False)
    # K=1: run the final all-data pass after the last numbered iteration without waiting for convergence.
    monkeypatch.setenv("RELAX_FINAL_ALL_DATA_AFTER_MAX_ITER", "1")
    rng = np.random.default_rng(42)
    halves = [MockHalfSet(N_IMAGES // 2, rng), MockHalfSet(N_IMAGES // 2, rng)]
    if n_classes > 1:
        option_groups.setdefault(
            "k_class",
            KClassOptions(n_classes=n_classes, init_class_log_priors=np.log(np.full(n_classes, 1.0 / n_classes))),
        )
    return iteration_loop.refine_single_volume(
        halves,
        _hermitian_volume(VOLUME_SHAPE, seed=42),
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 100.0,
        jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32),
        options=RefinementOptions(
            disc_type="linear_interp",
            schedule=RefinementSchedule(max_iter=max_iter, init_current_size=4, init_healpix_order=2, max_healpix_order=2),
            batching=RefinementBatching(image_batch_size=N_IMAGES, rotation_block_size=N_ROTATIONS),
            adaptive=AdaptiveOptions(adaptive_oversampling=1),
            parity=RelionParityOptions(**(parity or {})),
            **option_groups,
        ),
    )


class _Reached(Exception):
    """Raised by a recorder standing in for a GPU-only engine stage."""


def engine_stage_kwargs(monkeypatch, **engine_kwargs):
    """What ``run_dense_k_class_em_adaptive`` hands its GPU-only stages, without running them.

    The coarse significance pass (Gaussian and first-iteration CC routes) and the resident pass 2
    are replaced by recorders; returns ``{"pass1": kwargs, "pass1_cc": kwargs, "pass2": kwargs}``
    for a K=1 half of three mock images with ``engine_kwargs`` added to the engine's keywords.
    """

    from relax.classification import k_class
    from relax.scoring import significance
    from relax.sparse_pass2 import resident_pass2

    recorded = []

    def recorder(*_args, **kwargs):
        recorded.append(kwargs)
        raise _Reached

    monkeypatch.setattr(significance, "_compute_k_class_significance_batched", recorder)
    monkeypatch.setattr(resident_pass2, "compute_pass2_stats_resident", recorder)
    dataset = MockHalfSet(3, np.random.default_rng(0))
    coarse = np.repeat(np.eye(3, dtype=np.float32)[None], 4, axis=0)
    fine = np.repeat(np.eye(3, dtype=np.float32)[None], 8, axis=0)
    translations, fine_translations = np.zeros((2, 2), np.float32), np.zeros((4, 2), np.float32)
    rotation_parents, translation_parents = np.repeat(np.arange(4), 2), np.repeat(np.arange(2), 2)
    means = jnp.zeros((1, VOLUME_SIZE), jnp.complex64)
    keywords = dict(half_spectrum_scoring=True, mstep_relion_x_half=True, **engine_kwargs)
    stages = {}
    for name, cc in (("pass1", False), ("pass1_cc", True)):
        try:
            k_class.run_dense_k_class_em_adaptive(
                dataset, means, None, jnp.ones(IMAGE_SIZE), coarse, translations, fine, fine_translations,
                rotation_parents, translation_parents, "linear_interp",
                oversampling_order=1, coarse_healpix_order=0, firstiter_cc_pass2_only_best_coarse=cc, **keywords,
            )
        except _Reached:
            stages[name] = recorded[-1]
    try:
        k_class._run_sparse_k_class_adaptive_pass2(
            dataset, means, jnp.ones((1, VOLUME_SIZE)), jnp.ones(IMAGE_SIZE), coarse, translations, fine, None,
            rotation_parents, fine_translations, translation_parents, [[np.array([0], np.int32)] * 3], "linear_interp",
            class_log_priors=np.zeros(1), accumulate_noise=True, return_best_pose_details=False,
            coarse_healpix_order=0, oversampling_order=1, random_perturbation=0.0,
            engine_kwargs=dict(keywords, current_size=None),
        )
    except _Reached:
        stages["pass2"] = recorded[-1]
    return stages
