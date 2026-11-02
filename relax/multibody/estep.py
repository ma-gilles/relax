"""One body's expectation on one half: its subtracted images scored by the K=1 local route.

RELION processes the bodies of each particle in turn and writes a body's new pose before it prepares the next body
(``accDoExpectationOneParticle``, acc_ml_optimiser_impl.h:3725-3760 at f2c1a38), with the maps fixed until the
maximisation. Nothing couples two particles inside an expectation, so scoring every particle of a half for body 1,
then every particle for body 2 with body 1's new poses, is the same iteration. :func:`score_body_half` is one such
step: the body's images (:mod:`relax.multibody.subtraction`), its search (:mod:`relax.multibody.layout`) and the
existing exact-local engine (``relax.refinement.local_half``) with the body's translation prior: centred on the
body's refined offset, which the images carry as a Fourier pre-shift, with the body's own width.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from relax.multibody import geometry
from relax.multibody.bodies import Body
from relax.multibody.layout import BodySearch
from relax.multibody.subtraction import (
    BodyImagePrediction,
    BodyPoses,
    ParticleFrames,
    dataset_with_images,
    other_body_spectrum,
    translated_body_images,
)
from relax.refinement.local_half import (
    LocalBatchPolicy,
    LocalDiagnosticPolicy,
    LocalExecutionPolicy,
    LocalPriorSpec,
    score_half_local_in_bpref_scope,
)
from relax.refinement.ports import NoProbe
from relax.sampling.orientation_priors import relion_half_translation_prior_inputs, relion_translation_search_base
from relax.steps.half_inputs import HalfScoringData, HalfSet
from relax.steps.projector_preparation import PreparedProjector
from relax.steps.score_outputs import HalfScoreResult
from relax.steps.scoring_policy import local_precision
from relax.steps.shape_class_scoring import OpticsSpec

# Images subtracted at once: the prediction's phase table and projections of one batch stay well inside a GPU.
SUBTRACTION_BATCH_IMAGES = 2048


@dataclass(frozen=True)
class HalfParticles:
    """One half's particles as the bodies read them, in the half's local order."""

    index: int
    rows: np.ndarray  # rows of the whole particle table
    frames: ParticleFrames
    image_corrections: np.ndarray  # (avg_norm / norm) * scale
    scale_corrections: np.ndarray  # scale of the particle's group
    optics_group_ids: np.ndarray | None
    group_ids: np.ndarray  # 0-based RELION group of each particle
    group_count: int


@dataclass(frozen=True)
class BodyScore:
    """One body's expectation on one half: the engine's result and the body's new residual poses."""

    result: HalfScoreResult
    poses: BodyPoses


def score_body_half(
    target: int,
    bodies: list[Body],
    *,
    dataset,
    particles: HalfParticles,
    poses: list[BodyPoses],
    projectors: list[PreparedProjector],
    reference,
    noise_variance,
    noise_radial,
    data_vs_prior,
    sampling,
    offset_range_angstrom: float,
    current_size: int,
    padding_factor: int,
    options,
    iteration: int,
) -> BodyScore:
    """Score body ``target`` on one half and return its result and new poses.

    ``dataset`` is the whole particle dataset (its images are replaced by the body's); ``poses`` are every body's
    current poses of this half's particles (already updated for the bodies scored before ``target`` in this
    iteration); ``projectors`` are this half's projectors of every body's masked, centred map at the iteration's
    model size and ``reference`` the target body's map in the same form (recovar Fourier); ``data_vs_prior`` the body's
    curve the engine's scale statistics read (gated shells only); ``sampling`` is the
    iteration's ``LocalSampling``; ``options`` the run's ``RefinementOptions``.
    """

    body = bodies[target]
    n = int(particles.rows.size)
    images = np.zeros((dataset.n_images, *dataset.image_shape), dtype=np.float32)  # only this half's rows get pages
    half_dataset, frames = dataset.subset(particles.rows), particles.frames
    for start in range(0, n, SUBTRACTION_BATCH_IMAGES):
        batch = np.arange(start, min(start + SUBTRACTION_BATCH_IMAGES, n))
        images[particles.rows[batch]] = translated_body_images(
            half_dataset, batch, target, bodies,
            ParticleFrames(frames.consensus[batch], frames.offsets_px[batch], frames.norm_factors[batch]),
        )
    radius_px = geometry.body_mask_radius_px(body, offset_range_angstrom, dataset.voxel_size)
    body_dataset = dataset_with_images(
        dataset, images, mask_diameter_angstrom=2.0 * radius_px * float(dataset.voxel_size)
    ).subset(particles.rows)

    def prediction(image_indices):
        indices = np.asarray(image_indices).reshape(-1)
        return other_body_spectrum(
            half_dataset, indices, target, bodies, projectors,
            [BodyPoses(pose.eulers_deg[indices], pose.offsets_px[indices]) for pose in poses],
            ParticleFrames(frames.consensus[indices], frames.offsets_px[indices], frames.norm_factors[indices]),
            current_size=current_size, padding_factor=padding_factor,
        )

    pixel_size, _, width_mask_edge_px = dataset.image_source.backend._relion_image_mask_params
    image_operations = BodyImagePrediction(
        np.asarray(poses[target].offsets_px, dtype=np.float64), prediction,
        n_images=n, image_shape=dataset.image_shape, mask_radius_px=radius_px, mask_width_px=float(width_mask_edge_px),
        current_size=current_size, batch_images=SUBTRACTION_BATCH_IMAGES,
    )
    half = HalfScoringData(
        particles=HalfSet(
            index=particles.index,
            dataset=body_dataset,
            optics_group_ids=particles.optics_group_ids,
            rotation_eulers=poses[target].eulers_deg,
            translations=np.zeros((n, 2)),
            image_corrections=particles.image_corrections,
            scale_corrections=particles.scale_corrections,
        ),
        reference=reference,
        noise_variance=noise_variance,
        noise_radial=noise_radial,
        projector=projectors[target],
        # The engine's group-scale sums (RELION accumulates them; a multi-body run never applies them).
        scale_group_ids=particles.group_ids,
        scale_group_count=particles.group_count,
        scale_correction_data_vs_prior=data_vs_prior,
    )
    # RELION's body search: old offset and prior both zero (acc_ml_optimiser_impl.h:97-98, 2000-2009, 2097-2099),
    # with the refined offset applied to the image instead.
    zero = np.zeros((n, 2), dtype=np.float64)
    prior_inputs = relion_half_translation_prior_inputs(
        zero, voxel_size=dataset.voxel_size, base_translations=sampling.base_translations,
        current_translations=np.asarray(sampling.translations), dtype=options.precision.rotation_real_dtype,
    )
    left, right = geometry.body_left_right(particles.frames.consensus, body)
    result = score_half_local_in_bpref_scope(
        half=half,
        sampling=sampling,
        priors=LocalPriorSpec(
            trans_prior_center=prior_inputs.local_prior_center,
            trans_prior_center_for_engine=prior_inputs.engine_prior_center,
            current_sigma_offset_angstrom=body.sigma_offset_angstrom,
            translation_search_base=relion_translation_search_base(zero, dtype=options.precision.rotation_real_dtype),
            image_translations=image_operations,
            body_search=BodySearch(body=body, left=left, right=right),
        ),
        batching=LocalBatchPolicy(max_significants=options.adaptive.max_significants),
        execution=LocalExecutionPolicy(
            disc_type="linear_interp",
            source_faithful_spectrum_norm=options.parity.preserve_bpref_particle_order,
            relion_translation_angle_scale=1.0,
            nyquist_column_counting=options.consistency.nyquist_column_counting,
            relion_x_half_mstep=options.variants.k1_relion_x_half_mstep,
            adaptive_pass2=options.variants.local_adaptive_pass2.at(sampling.search.oversampling_order),
            precision=options.precision,
            fine_precision=local_precision(
                options.precision, iteration + 1, pass_index=2,
                float64_pass2_iterations=options.environment.diagnostic_float64_pass2_iterations,
            ),
            score_only=False,
            firstiter_cc=False,
            engine_tuning=options.environment.engine,
            scoring_overrides=options.environment.scoring,
            padding_factor=padding_factor,
        ),
        diagnostics=LocalDiagnosticPolicy(
            iteration=iteration,
            collect_local_search_profile=False,
            local_profile_history=[],
            debug_iteration=iteration + 1,
            bpref_device_signature_active=False,
            probe=NoProbe(),
        ),
        optics=OpticsSpec.single_shape(1.0),
    )
    # The new offset is the refined offset plus the best translation (acc_ml_optimiser_impl.h:2905-2912).
    new_offsets = np.asarray(poses[target].offsets_px, dtype=np.float64) + np.asarray(
        result.best_pose_translations, dtype=np.float64
    )
    return BodyScore(
        result=result,
        poses=BodyPoses(eulers_deg=np.asarray(result.best_pose_rotation_eulers, dtype=np.float64), offsets_px=new_offsets),
    )
