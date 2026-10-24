"""The first-iteration (``--firstiter_cc``) routes of the dense K-class engine.

``run_dense_k_class_joint_firstiter_score_probe`` scores every class jointly to find each image's global
winner; ``run_sparse_firstiter_global_winner_subset_pass2`` runs pass 2 per class on the images that class
won and assembles the per-class subset results (``_PerClassSubsetResults``). The engine entry
(``k_class.run_dense_k_class_em_adaptive``) selects them. Imports nothing from ``k_class``.
"""

from __future__ import annotations

import logging
import time
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from relax.classification.k_class_inputs import (
    as_class_means,
    dataset_image_count,
    projection_float64_from_kwargs,
    select_projector_half_for_class,
    translation_angle_scale_kwargs,
)
from relax.classification.k_class_results import (
    KClassEMResult,
    assemble_result,
    expand_subset_noise_stats,
    expand_subset_pose_details,
    full_stats_from_subset,
    zero_subset_noise_stats,
)
from relax.classification.k_class_sparse_pass2 import (
    as_host_accumulator,
    resident_production_arithmetic,
    sparse_pose_ids_to_fine_grid,
)
from relax.classification.probes import KClassProbe
from relax.reconstruction.half_volume_mstep import relion_backprojector_volume_shape
from relax.relion.scale_groups import prepare_scale_correction_groups
from relax.types import RelionStats, make_relion_stats

logger = logging.getLogger("relax.classification.k_class")  # the engine's logger before the split: log rows keep it


class _DenseKClassScoreProbeResult(NamedTuple):
    """Joint ``--firstiter_cc`` coarse probe output: per-class evidence, best poses and global class winners."""

    class_log_evidence: np.ndarray
    per_class_hard_assignments: np.ndarray
    per_class_stats: tuple[RelionStats, ...]
    class_assignments: np.ndarray
    coarse_score_backend: str | None = None


def run_dense_k_class_joint_firstiter_score_probe(
    experiment_dataset,
    means_array,
    noise_variance,
    rotations,
    translations,
    *,
    engine_kwargs: dict,
    probe: KClassProbe,
    scoring_overrides,
) -> _DenseKClassScoreProbeResult:
    """Score RELION firstiter-CC K-class coarse poses in one shared pass; ``probe`` observes it
    (``scoring_overrides``: the run's ScoringOverrides)."""

    from relax.projection.projection import compact_relion_projector_half_for_centered_indices
    from relax.scoring.significance import compute_k_class_significance_batched

    means_array = as_class_means(means_array)
    n_classes = int(means_array.shape[0])
    n_rot = int(np.asarray(rotations).shape[0])
    n_images = dataset_image_count(experiment_dataset)

    # Keep the full Projector::data host slab for fine scoring, but transfer
    # only the centered support consumed by this coarse normalized-CC probe.
    # At box 800, staging the full PPref plus its CUDA texture can otherwise
    # exhaust an 80-GB device before the first particle is scored.
    score_projector_half = engine_kwargs.get("relion_projector_half")
    score_projector_r_max = engine_kwargs.get("relion_projector_r_max")
    score_texture_interp = engine_kwargs.get(
        "coarse_relion_projector_texture_interp",
        False,
    )
    if score_texture_interp is None:
        score_texture_interp = True  # RELION's texture interpolation, pass 1's only coarse projector
    if score_projector_half is not None and n_classes == 1 and score_texture_interp:
        from relax.fourier.fourier_window import make_fourier_window_spec

        image_shape = tuple(int(value) for value in experiment_dataset.image_shape)
        n_half = image_shape[0] * (image_shape[1] // 2 + 1)
        score_window = make_fourier_window_spec(
            image_shape,
            engine_kwargs.get("current_size"),
            n_half,
            square=bool(engine_kwargs.get("square_window", False)),
            include_recon_window=False,
            score_square=True,
            score_include_dc=True,
        )
        if score_window.score_indices_np is not None:
            score_projector_half = select_projector_half_for_class(
                score_projector_half,
                0,
                1,
            )
            original_projector_shape = tuple(int(value) for value in score_projector_half.shape)
            original_projector_r_max = int(score_projector_r_max)
            score_projector_half, score_projector_r_max = (
                compact_relion_projector_half_for_centered_indices(
                    score_projector_half,
                    score_window.score_indices_np,
                    image_shape,
                    r_max=int(score_projector_r_max),
                    padding_factor=int(engine_kwargs.get("projection_padding_factor", 1)),
                )
            )
            logger.info(
                "RELION firstiter-CC coarse PPref host compaction: "
                "r_max=%d->%d shape=%s->%s allocated=%.4f GiB",
                original_projector_r_max,
                int(score_projector_r_max),
                original_projector_shape,
                tuple(int(value) for value in score_projector_half.shape),
                float(score_projector_half.nbytes / 2**30),
            )

    # RELION's iter-1 firstiter_cc path performs WTA on raw normalized-CC
    # scores before the non-firstiter prior-weighting branch is reached.
    firstiter_class_log_priors = np.zeros(n_classes, dtype=np.float64)
    pass1_result = compute_k_class_significance_batched(
        experiment_dataset,
        noise_variance,
        rotations,
        translations,
        class_log_priors=firstiter_class_log_priors,
        engine_tuning=engine_kwargs["engine_tuning"],
        scoring_overrides=scoring_overrides,
        adaptive_fraction=1.0,
        max_significants=1,
        image_batch_size=int(engine_kwargs.get("image_batch_size", 500)),
        rotation_block_size=int(engine_kwargs.get("rotation_block_size", 5000)),
        current_size=engine_kwargs.get("current_size"),
        score_with_masked_images=bool(engine_kwargs.get("score_with_masked_images", False)),
        rotation_log_prior=None,
        translation_log_prior=None,
        image_corrections=engine_kwargs.get("image_corrections"),
        scale_corrections=engine_kwargs.get("scale_corrections"),
        image_pre_shifts=engine_kwargs.get("image_pre_shifts"),
        # The normalized CC does not weight by the noise, but the scorer reads each image's row.
        optics_group_ids=engine_kwargs.get("optics_group_ids"),
        half_spectrum_scoring=bool(engine_kwargs.get("half_spectrum_scoring", False)),
        projection_padding_factor=int(engine_kwargs.get("projection_padding_factor", 1)),
        square_window=bool(engine_kwargs.get("square_window", False)),
        use_float64_scoring=bool(engine_kwargs.get("use_float64_scoring", False)),
        use_float64_projections=projection_float64_from_kwargs(engine_kwargs),
        relion_projector_half=score_projector_half,
        relion_projector_r_max=score_projector_r_max,
        relion_projector_texture_interp=engine_kwargs.get(
            "coarse_relion_projector_texture_interp",
            False,
        ),
        score_mode="normalized_cc",
        firstiter_cc_support=engine_kwargs.get("firstiter_cc_support", "relion"),
        nyquist_column_counting=engine_kwargs.get("nyquist_column_counting", "relion"),
        tree_rescore_max_margin=engine_kwargs.get("firstiter_cc_tree_rescore_max_margin"),
        collect_significance=probe.score_dump_requested(
            current_size=engine_kwargs.get("current_size"),
            debug_iteration=engine_kwargs.get("debug_iteration"),
        ),
        return_class_best=True,
        return_class_second=probe.wants_class_runner_up(),
        debug_iteration=engine_kwargs.get("debug_iteration"),
        probe=probe,
        coarse_healpix_order=engine_kwargs.get("coarse_healpix_order"),
        coarse_rotation_ids=engine_kwargs.get("coarse_rotation_ids"),
        translation_phase_source=engine_kwargs.get("translation_phase_source"),
        **({"symmetry_label": engine_kwargs["symmetry_label"]} if engine_kwargs.get("symmetry_label", "C1") != "C1" else {}),
        **translation_angle_scale_kwargs(engine_kwargs),
    )
    probe.firstiter_winners_scored(
        experiment_dataset=experiment_dataset,
        stats=pass1_result.stats,
        class_assignments=pass1_result.class_assignment,
        n_classes=n_classes,
        n_rotations=n_rot,
        n_translations=int(np.asarray(translations).shape[0]),
        iteration=engine_kwargs.get("debug_iteration"),
    )
    class_log_evidence = np.asarray(pass1_result.stats.class_log_evidence_per_image, dtype=np.float64)
    per_class_hard = np.asarray(pass1_result.stats.class_hard_assignments, dtype=np.int32)
    score_dtype = _score_dtype_from_kwargs(engine_kwargs)
    class_best_log_score = np.asarray(pass1_result.stats.class_best_log_score_per_image, dtype=score_dtype)
    class_assignments = np.asarray(pass1_result.class_assignment, dtype=np.int32)
    per_class_stats = tuple(
        make_relion_stats(
            log_evidence_per_image=np.asarray(class_log_evidence[class_index], dtype=score_dtype),
            best_log_score_per_image=np.asarray(class_best_log_score[class_index], dtype=score_dtype),
            max_posterior_per_image=np.ones(n_images, dtype=score_dtype),
            rotation_posterior_sums=np.zeros(n_rot, dtype=score_dtype),
        )
        for class_index in range(n_classes)
    )

    return _DenseKClassScoreProbeResult(
        class_log_evidence=class_log_evidence,
        per_class_hard_assignments=per_class_hard,
        per_class_stats=per_class_stats,
        class_assignments=class_assignments,
        coarse_score_backend=pass1_result.stats.executed_coarse_backend,
    )


# The image-axis inputs a firstiter-CC subset pass forwards for each winner class.
_SUBSET_PASS_CLASS_KWARGS = (
    "image_corrections", "scale_corrections", "image_pre_shifts", "translation_prior_centers", "image_translations",
)


_IMAGE_AXIS_ENGINE_KWARGS = (
    "image_corrections",
    "scale_corrections",
    "group_ids",
    "optics_group_ids",
    "image_pre_shifts",
    "image_translations",
    "translation_prior_centers",
    "translation_log_prior",
    "rotation_log_prior",
    "normalization_log_evidence",
    "relion_f32_normalization_sum_weight",
)


def _subset_image_axis_engine_kwargs(kwargs: dict, image_indices: np.ndarray, n_images: int) -> dict:
    """Slice image-axis kwargs when running a class-specific dataset subset."""

    out = dict(kwargs)
    image_indices = np.asarray(image_indices, dtype=np.int64)
    for name in _IMAGE_AXIS_ENGINE_KWARGS:
        value = out.get(name)
        if value is None:
            continue
        array = np.asarray(value)
        if array.ndim > 0 and int(array.shape[0]) == int(n_images):
            out[name] = array[image_indices]
    return out


def _pose_dtype_from_kwargs(kwargs: dict):
    """Host pose dtype: float64 when scoring or projections run in double, else float32."""

    return np.float64 if kwargs.get("use_float64_scoring", False) or projection_float64_from_kwargs(kwargs) else np.float32


def _score_dtype_from_kwargs(kwargs: dict):
    """Host score dtype: float64 when scoring runs in double, else float32."""

    return np.float64 if kwargs.get("use_float64_scoring", False) else np.float32


def _full_group_count_from_kwargs(kwargs: dict) -> int | None:
    _, group_count = prepare_scale_correction_groups(
        kwargs.get("group_ids"), kwargs.get("scale_correction_group_count"),
    )
    return group_count or None


class _PerClassSubsetResults:
    """Per-class outputs of a firstiter-CC global-winner subset pass, in class order.

    Every image belongs to exactly one winning class, so each class contributes
    accumulators, assignments, statistics, noise and best poses over its own
    image subset, expanded back to the full image axis. A class without images
    contributes zero accumulators, ``-inf`` best scores and zero posteriors.
    ``host_accumulators`` says whether the appended M-step accumulators are moved
    to the host; the sparse route hosts engine outputs and keeps its empty-class
    zeros on the device.
    """

    def __init__(
        self, *, n_images: int, accumulate_noise: bool, return_best_pose_details: bool, full_group_count, pose_dtype, score_dtype
    ):
        self.Ft_y = []
        self.Ft_ctf = []
        self.hard_assignments = []
        self.per_class_stats = []
        self.per_class_noise = [] if accumulate_noise else None
        self.best_pose_rotations = [] if return_best_pose_details else None
        self.best_pose_translations = [] if return_best_pose_details else None
        self.best_pose_rotation_ids = [] if return_best_pose_details else None
        self.subset_counts = []
        self.n_images = n_images
        self.full_group_count = full_group_count
        self.pose_dtype = pose_dtype
        self.score_dtype = score_dtype
        self.return_best_pose_details = return_best_pose_details

    def append_empty_class(self, *, mean, class_log_evidence, noise_variance, n_rot, host_accumulators):
        zero = jnp.zeros_like(mean)
        zero_ctf = jnp.zeros_like(jnp.real(mean))
        self.Ft_y.append(as_host_accumulator(zero) if host_accumulators else zero)
        self.Ft_ctf.append(as_host_accumulator(zero_ctf) if host_accumulators else zero_ctf)
        self.hard_assignments.append(np.zeros(self.n_images, dtype=np.int32))
        self.per_class_stats.append(
            make_relion_stats(
                log_evidence_per_image=np.asarray(class_log_evidence, dtype=self.score_dtype),
                best_log_score_per_image=np.full(self.n_images, -np.inf, dtype=self.score_dtype),
                max_posterior_per_image=np.zeros(self.n_images, dtype=self.score_dtype),
                rotation_posterior_sums=np.zeros(n_rot, dtype=self.score_dtype),
            ),
        )
        if self.per_class_noise is not None:
            self.per_class_noise.append(
                zero_subset_noise_stats(
                    noise_variance,
                    n_images=self.n_images,
                    full_group_count=self.full_group_count,
                ),
            )
        if self.return_best_pose_details:
            self.best_pose_rotations.append(np.zeros((self.n_images, 3, 3), dtype=self.pose_dtype))
            self.best_pose_translations.append(np.zeros((self.n_images, 2), dtype=self.pose_dtype))
            self.best_pose_rotation_ids.append(np.zeros(self.n_images, dtype=np.int32))

    def append_class(self, *, image_indices, Ft_y, Ft_ctf, hard_full, stats_subset, class_log_evidence, noise, best_pose, host_accumulators):
        self.Ft_y.append(as_host_accumulator(Ft_y) if host_accumulators else Ft_y)
        self.Ft_ctf.append(as_host_accumulator(Ft_ctf) if host_accumulators else Ft_ctf)
        self.hard_assignments.append(hard_full)
        self.per_class_stats.append(
            full_stats_from_subset(
                stats_subset,
                image_indices,
                self.n_images,
                class_log_evidence=class_log_evidence,
            ),
        )
        if self.per_class_noise is not None:
            self.per_class_noise.append(
                expand_subset_noise_stats(
                    noise,
                    image_indices,
                    self.n_images,
                    full_group_count=self.full_group_count,
                ),
            )
        if self.return_best_pose_details:
            best_rots, best_trans, best_rot_ids = best_pose
            best_rots_full, best_trans_full, best_rot_ids_full = expand_subset_pose_details(
                best_rots, best_trans, best_rot_ids, image_indices, self.n_images
            )
            self.best_pose_rotations.append(best_rots_full)
            self.best_pose_translations.append(best_trans_full)
            self.best_pose_rotation_ids.append(best_rot_ids_full)

    def assemble(self, class_log_evidence, *, profile_summary, **assemble_kwargs):
        """The winner-take-all K-class result of a subset pass; the subset counts are its class posterior sums."""

        if any(self.subset_counts):
            # A class without images adds zeros in the scored classes' layout (x-half BPref accumulators, noise per
            # shell), not the full-volume, per-pixel placeholders: a seed iteration's CC pass puts every image in
            # class 1.
            scored = next(k for k, n in enumerate(self.subset_counts) if n)
            for k, n in enumerate(self.subset_counts):
                if n:
                    continue
                self.Ft_y[k] = np.zeros_like(np.asarray(self.Ft_y[scored]))
                self.Ft_ctf[k] = np.zeros_like(np.asarray(self.Ft_ctf[scored]))
                if self.per_class_noise is not None:
                    self.per_class_noise[k] = _zero_noise_stats_like(self.per_class_noise[scored])
        return assemble_result(
            class_log_evidence=class_log_evidence,
            Ft_y=self.Ft_y,
            Ft_ctf=self.Ft_ctf,
            per_class_hard_assignments=np.stack(self.hard_assignments, axis=0),
            per_class_stats=tuple(self.per_class_stats),
            noise_stats=None if self.per_class_noise is None else tuple(self.per_class_noise),
            per_class_best_pose_rotations=self.best_pose_rotations,
            per_class_best_pose_translations=self.best_pose_translations,
            per_class_best_pose_rotation_ids=self.best_pose_rotation_ids,
            class_posterior_sums_override=np.asarray(self.subset_counts, dtype=np.float64),
            firstiter_winner_take_all=True,
            profile_summary=profile_summary,
            **assemble_kwargs,
        )


def _zero_noise_stats_like(stats):
    """``stats`` with every sum zero, in its layout."""

    return stats._replace(
        **{
            name: (None if value is None else 0.0 if np.ndim(value) == 0 else np.zeros_like(np.asarray(value)))
            for name, value in stats._asdict().items()
        }
    )


def run_sparse_firstiter_global_winner_subset_pass2(
    experiment_dataset,
    means_array,
    mean_variance,
    noise_variance,
    coarse_translations_np,
    fine_rotations_np,
    fine_mstep_rotations_np,
    fine_translations_np,
    rot_parent_map_np: np.ndarray,
    trans_parent_map_np: np.ndarray,
    sig_sample_indices_by_class,
    disc_type: str,
    *,
    coarse_result: _DenseKClassScoreProbeResult | None,
    coarse_class_assignments: np.ndarray,
    n_rot_coarse: int,
    n_fine_trans: int,
    healpix_order: int,
    oversampling_order: int,
    accumulate_noise: bool,
    return_best_pose_details: bool,
    pass2_kwargs: dict,
    given_pose_evidence: np.ndarray | None = None,
) -> KClassEMResult:
    """Sparse RELION firstiter_cc fine pass over global-winner image subsets.

    ``coarse_result`` None is the pass at given poses (``--skip_align``): there is no coarse probe, each
    image's one sample is its only candidate, and a class's evidence for its images is the fine pass's
    own (the coarse and the fine sample are the same). ``given_pose_evidence`` (float64 ``[K, N]``) then
    receives that evidence as the pass reduced it, before the result's cast to the score dtype.
    """

    from relax.fine_pass.dispatch import compute_pass2_stats_sparse

    n_classes = int(means_array.shape[0])
    n_images = int(coarse_class_assignments.shape[0])
    relion_projector_half_by_class = pass2_kwargs.get("relion_projector_half")
    relion_projector_r_max = pass2_kwargs.get("relion_projector_r_max")
    source_faithful_spectrum_norm = bool(
        pass2_kwargs.get("source_faithful_spectrum_norm", False)
    )
    if source_faithful_spectrum_norm and n_classes != 1:
        raise ValueError("source-faithful powerClass normalization is K=1-only")
    score_dtype = _score_dtype_from_kwargs(pass2_kwargs)
    pose_dtype = _pose_dtype_from_kwargs(pass2_kwargs)

    def _class_rotation_prior(class_index: int):
        del class_index
        return np.zeros(int(n_rot_coarse), dtype=score_dtype)

    common = dict(
        nside_level=int(healpix_order),
        disc_type=disc_type,
        oversampling_order=int(oversampling_order),
        current_size=pass2_kwargs.get("current_size"),
        reconstruction_current_size=pass2_kwargs.get("reconstruction_current_size"),
        translation_step=None,
        score_with_masked_images=bool(pass2_kwargs.get("score_with_masked_images", False)),
        return_stats=True,
        half_spectrum_scoring=bool(pass2_kwargs.get("half_spectrum_scoring", False)),
        projection_padding_factor=int(pass2_kwargs.get("projection_padding_factor", 1)),
        reconstruction_padding_factor=int(pass2_kwargs.get("reconstruction_padding_factor", 1)),
        use_float64_scoring=bool(pass2_kwargs.get("use_float64_scoring", False)),
        do_gridding_correction=bool(pass2_kwargs.get("do_gridding_correction", False)),
        square_window=bool(pass2_kwargs.get("square_window", False)),
        random_perturbation=0.0,
        fine_rotations_override=fine_rotations_np,
        fine_mstep_rotations_override=fine_mstep_rotations_np,
        fine_rotation_parent_override=rot_parent_map_np,
        fine_translations_override=fine_translations_np,
        fine_translation_parent_override=trans_parent_map_np,
        relion_x_half_mstep=bool(pass2_kwargs.get("mstep_relion_x_half", False)),
        relion_firstiter_score_mode="normalized_cc",
        relion_firstiter_winner_take_all=True,
        # K=1 production firstiter-CC must use the literal RELION fine
        # numerator/denominator reduction.  Previously only the coarse probe
        # enabled this path, leaving the actual pass-2 M-step on the folded
        # algebraic shortcut.  Both are algebraically equivalent, but the
        # literal route follows RELION's reduction contract directly.
        relion_exact_fine_normalized_cc=n_classes == 1,
        source_faithful_spectrum_norm=source_faithful_spectrum_norm,
        nyquist_column_counting=pass2_kwargs.get("nyquist_column_counting", "relion"),
        firstiter_cc_support=pass2_kwargs.get("firstiter_cc_support", "relion"),
        # Images on another grid (K=1 shape classes): the reference-model M-step size and
        # image radius, as the Gaussian route passes them.
        reconstruction_volume_current_size=pass2_kwargs.get("reconstruction_volume_current_size"),
        reconstruction_image_radius=pass2_kwargs.get("reconstruction_image_radius"),
        **({"symmetry_label": pass2_kwargs["symmetry_label"]}
           if pass2_kwargs.get("symmetry_label", "C1") != "C1" else {}),
        **translation_angle_scale_kwargs(pass2_kwargs),
    )
    if pass2_kwargs.get("wsum_current_size") is not None:
        # --strict_highres_exp: the weighted sums' image size, above the E-step's (the resident engine).
        common["wsum_current_size"] = int(pass2_kwargs["wsum_current_size"])
    resident_arithmetic = n_classes > 1
    if resident_arithmetic:
        # Each image's fine pass is a K=1 pass inside its coarse winner class,
        # so on the resident engine it is the K=1 production pass.
        common = resident_production_arithmetic(common)
    # Images on another grid fill the backprojector at the reference model size.
    mstep_current_size = (
        common["reconstruction_volume_current_size"]
        if common.get("reconstruction_volume_current_size") is not None
        else common["current_size"]
        if common["reconstruction_current_size"] is None
        else common["reconstruction_current_size"]
    )
    mstep_accumulator_shape = (
        relion_backprojector_volume_shape(
            experiment_dataset.volume_shape,
            common["reconstruction_padding_factor"],
            current_size=mstep_current_size,
        )
        if common["relion_x_half_mstep"]
        else None
    )

    class_log_evidence = (
        np.full((n_classes, n_images), -np.inf, dtype=score_dtype)
        if coarse_result is None
        else np.asarray(coarse_result.class_log_evidence)
    )
    results = _PerClassSubsetResults(
        n_images=n_images,
        accumulate_noise=accumulate_noise,
        return_best_pose_details=return_best_pose_details,
        full_group_count=_full_group_count_from_kwargs(pass2_kwargs),
        pose_dtype=pose_dtype,
        score_dtype=score_dtype,
    )
    t0 = time.time()
    for class_index in range(n_classes):
        image_indices = np.nonzero(coarse_class_assignments == class_index)[0].astype(np.int64, copy=False)
        results.subset_counts.append(int(image_indices.size))
        if image_indices.size == 0:
            results.append_empty_class(
                mean=means_array[class_index],
                class_log_evidence=class_log_evidence[class_index],
                noise_variance=noise_variance,
                n_rot=n_rot_coarse,
                host_accumulators=False,
            )
            continue

        subset_dataset = experiment_dataset.subset(image_indices)
        subset_sig = [sig_sample_indices_by_class[class_index][int(i)] for i in image_indices]
        class_kwargs = _subset_image_axis_engine_kwargs(
            {name: pass2_kwargs.get(name) for name in _SUBSET_PASS_CLASS_KWARGS}, image_indices, n_images
        )
        scale_groups = {"group_ids": None, "scale_correction_group_count": None}
        if resident_arithmetic:
            # The K=1 production pass of each winner class carries the scale groups as the
            # K=1 route does. RELION accumulates the scale sums in the firstiter_cc iteration
            # and does not apply them (ml_optimiser.cpp:6190).
            scale_groups = _subset_image_axis_engine_kwargs(
                {"group_ids": pass2_kwargs.get("group_ids")}, image_indices, n_images
            ) | {"scale_correction_group_count": pass2_kwargs.get("scale_correction_group_count")}

        result = compute_pass2_stats_sparse(
            subset_dataset,
            means_array[class_index],
            noise_variance,
            coarse_translations_np,
            subset_sig,
            engine_tuning=pass2_kwargs["engine_tuning"],
            scoring_overrides=pass2_kwargs["scoring_overrides"],
            rotation_log_prior=_class_rotation_prior(class_index),
            translation_log_prior=None,
            accumulate_noise=accumulate_noise,
            image_corrections=class_kwargs.get("image_corrections"),
            scale_corrections=class_kwargs.get("scale_corrections"),
            **scale_groups,
            image_pre_shifts=class_kwargs.get("image_pre_shifts"),
            image_translations=class_kwargs.get("image_translations"),
            translation_prior_centers=class_kwargs.get("translation_prior_centers"),
            # Each image's noise row.
            optics_group_ids=_subset_image_axis_engine_kwargs(
                {"optics_group_ids": pass2_kwargs.get("optics_group_ids")}, image_indices, n_images
            )["optics_group_ids"],
            relion_projector_half=select_projector_half_for_class(
                relion_projector_half_by_class,
                class_index,
                n_classes,
            ),
            relion_projector_r_max=relion_projector_r_max,
            **common,
        )
        if coarse_result is None:
            class_log_evidence[class_index, image_indices] = np.asarray(
                result.relion_stats.log_evidence_per_image, dtype=class_log_evidence.dtype
            )
            if given_pose_evidence is not None:
                given_pose_evidence[class_index, image_indices] = np.asarray(
                    result.relion_stats.log_evidence_per_image, dtype=np.float64
                )
        hard_full = np.zeros(n_images, dtype=np.int32)
        hard_full[image_indices] = sparse_pose_ids_to_fine_grid(
            result.hard_assignment, result.best_rotation_indices, n_fine_trans
        )
        results.append_class(
            image_indices=image_indices,
            Ft_y=result.Ft_y,
            Ft_ctf=result.Ft_ctf,
            hard_full=hard_full,
            stats_subset=result.relion_stats,
            class_log_evidence=class_log_evidence[class_index],
            noise=result.noise_stats,
            best_pose=(
                (result.best_rotations, result.best_translations, result.best_rotation_indices)
                if return_best_pose_details
                else None
            ),
            host_accumulators=True,
        )

    logger.info(
        "Sparse firstiter-CC global-winner subset pass2: classes=%d images=%d subset_counts=%s total=%.1fs",
        n_classes,
        n_images,
        results.subset_counts,
        time.time() - t0,
    )
    return results.assemble(
        class_log_evidence,
        profile_summary={"sparse_firstiter_subset_pass2_s": np.float64(time.time() - t0)},
        host_accumulators=True,
        mstep_full_half_axis=0 if common["relion_x_half_mstep"] else None,
        mstep_accumulator_shape=mstep_accumulator_shape,
    )
