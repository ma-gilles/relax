"""A CPU stand-in for ``run_dense_k_class_em_adaptive`` in refinement-loop tests.

The global E-step of every refinement (K=1 and K-class, any adaptive
oversampling) runs ``run_dense_k_class_em_adaptive``, whose pass 2 runs only on
the device-resident GPU engine. Tests of the iteration loop (scheduling, routing,
state updates) replace it with :func:`fake_adaptive_engine`, which returns a
``KClassEMResult`` in the engine's layout: full-volume accumulators at the
reconstruction padding, fine-grid pose ids and per-fine-rotation posterior sums.
The engine's numerics are checked against the RELION E-step reference in
``test_resident_relion_reference``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from relax.classification.k_class_results import KClassEMResult
from relax.helpers.types import NoiseStats, RelionStats


def accumulator_size(experiment_dataset, kwargs) -> int:
    """Voxels of one class's full-volume BPref accumulator at the reconstruction padding."""

    padding_factor = int(kwargs.get("reconstruction_padding_factor", 1))
    return int(np.prod([int(size) * padding_factor for size in experiment_dataset.volume_shape]))


def adaptive_result(
    experiment_dataset,
    means,
    fine_rotations,
    kwargs,
    *,
    max_posterior=None,
    Ft_y=None,
    Ft_ctf=None,
    sigma2_offset=0.0,
    significant_counts=None,
    pose_assignments=None,
    best_pose_translations=None,
):
    """A ``KClassEMResult`` for ``means`` (``[K, V]``) over the dataset's images.

    Defaults: zero ``Ft_y``, unit ``Ft_ctf``, every image in class 0 at fine pose 0 with
    posterior 1, identity best rotation with zero best translation, unit noise sums and one
    significant sample per image. Arguments override one field; ``Ft_y``/``Ft_ctf`` are ``[K, V]`` arrays or callables of ``(k, size)``.
    """

    n_classes = int(np.asarray(means).shape[0]) if np.ndim(means) >= 2 else 1
    n_images = int(experiment_dataset.n_units)
    n_shells = int(experiment_dataset.image_shape[0]) // 2 + 1
    n_fine_rot = int(np.asarray(fine_rotations).shape[0])
    size = accumulator_size(experiment_dataset, kwargs)

    def accumulators(value, default):
        if value is None:
            return default
        if callable(value):
            return jnp.stack([jnp.asarray(value(k, size)) for k in range(n_classes)])
        return jnp.asarray(value)

    pmax = (
        jnp.ones(n_images, dtype=jnp.float32)
        if max_posterior is None
        else jnp.asarray(max_posterior, dtype=jnp.float32)
    )
    stats = tuple(
        RelionStats(
            log_evidence_per_image=jnp.zeros(n_images, dtype=jnp.float32),
            best_log_score_per_image=jnp.zeros(n_images, dtype=jnp.float32),
            max_posterior_per_image=pmax,
            rotation_posterior_sums=jnp.ones(n_fine_rot, dtype=jnp.float32),
        )
        for _ in range(n_classes)
    )

    def noise(sumw):
        return NoiseStats(
            wsum_sigma2_noise=jnp.ones(n_shells, dtype=jnp.float32),
            wsum_img_power=jnp.ones(n_shells, dtype=jnp.float32),
            wsum_sigma2_offset=sigma2_offset,
            sumw=float(sumw),
        )

    poses = (
        jnp.zeros(n_images, dtype=jnp.int32)
        if pose_assignments is None
        else jnp.asarray(pose_assignments, dtype=jnp.int32)
    )
    return KClassEMResult(
        new_means=None,
        Ft_y=accumulators(Ft_y, jnp.zeros((n_classes, size), dtype=jnp.complex64)),
        Ft_ctf=accumulators(Ft_ctf, jnp.ones((n_classes, size), dtype=jnp.complex64)),
        per_class_hard_assignments=jnp.broadcast_to(poses, (n_classes, n_images)),
        class_assignments=jnp.zeros(n_images, dtype=jnp.int32),
        pose_assignments=poses,
        class_responsibilities=jnp.full((n_classes, n_images), 1.0 / n_classes, dtype=jnp.float32),
        class_posterior_sums=jnp.full(n_classes, n_images / n_classes, dtype=jnp.float32),
        stats=stats[0],
        per_class_stats=stats,
        noise_stats=tuple(noise(n_images / n_classes) for _ in range(n_classes)),
        aggregate_noise_stats=noise(n_images),
        best_pose_rotations=jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (n_images, 3, 3)),
        best_pose_translations=(
            jnp.zeros((n_images, 2), dtype=jnp.float32)
            if best_pose_translations is None
            else jnp.asarray(best_pose_translations, dtype=jnp.float32)
        ),
        best_pose_rotation_ids=jnp.zeros(n_images, dtype=jnp.int32),
        significant_counts=(
            jnp.ones(n_images, dtype=jnp.int32)
            if significant_counts is None
            else jnp.asarray(significant_counts, dtype=jnp.int32)
        ),
    )


def fake_adaptive_engine(calls=None, **result_kwargs):
    """A ``run_dense_k_class_em_adaptive`` replacement returning :func:`adaptive_result`.

    Each call appends a record to ``calls`` (when given): the dataset, the means, the
    coarse and fine grids, and the keyword arguments.
    """

    def run(
        experiment_dataset,
        means,
        mean_variance,
        noise_variance,
        coarse_rotations,
        coarse_translations,
        fine_rotations,
        fine_translations,
        rot_parent_map,
        trans_parent_map,
        disc_type,
        **kwargs,
    ):
        del mean_variance, disc_type
        if calls is not None:
            calls.append(
                dict(
                    dataset=experiment_dataset,
                    means=means,
                    noise_variance=noise_variance,
                    coarse_rotations=coarse_rotations,
                    coarse_translations=coarse_translations,
                    fine_rotations=fine_rotations,
                    fine_translations=fine_translations,
                    rot_parent_map=rot_parent_map,
                    trans_parent_map=trans_parent_map,
                    kwargs=kwargs,
                )
            )
        return adaptive_result(experiment_dataset, means, fine_rotations, kwargs, **result_kwargs)

    return run


def install_fake_adaptive_engine(monkeypatch, calls=None, **result_kwargs):
    """Route every global E-step (the adaptive and the --firstiter_cc dispatch) to the fake."""

    from relax.refinement import firstiter_cc, half_scoring

    fake = fake_adaptive_engine(calls, **result_kwargs)
    monkeypatch.setattr(half_scoring, "run_dense_k_class_em_adaptive", fake)
    monkeypatch.setattr(firstiter_cc, "run_dense_k_class_em_adaptive", fake)
    return fake
