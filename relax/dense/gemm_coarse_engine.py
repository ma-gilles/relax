"""Exact full-grid GEMM global pass using the resident pass's numerical owners.

The caller supplies the canonical expanded grid and the preparation state just
before resident candidate tables would be built.  This route streams fixed
rotation and translation tiles, without making a full image-by-rotation table.
Only the E-step and collapsed BPref are new; windowing, particle operands,
Wavg/noise statistics, and result finalization use their existing owners.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from relax.dense.gemm_coarse_statistics import (
    DenseGemmStatisticsOperands,
    DenseGemmStatisticsPlan,
    accumulate_batch_statistics,
    initial_statistics_carry,
    make_statistics_callbacks,
)
from relax.dense.gemm_experiment import (
    DenseGemmTileConfig,
    make_joint_k_batch_program,
    native_phase_table,
    native_relion_callbacks,
    pad_batch,
    pad_grid,
)


@dataclass(frozen=True)
class DenseGemmPreparedState:
    """State already resolved by the resident driver before sparse table build."""

    dataset: object
    class_volumes: tuple
    class_projector_halves: tuple
    class_rotation_priors: tuple
    fine_rotations: object
    fine_mstep_rotations: object
    fine_rotation_parent: object
    fine_translations_source: object
    fine_translations: object
    fine_translation_parent: object
    fine_translation_prior: object
    translation_prior_centers: object
    noise_variance_half: object
    n_coarse_rot: int
    nside_level: int
    oversampling_order: int
    random_perturbation: float
    relion_parent_execution_order: bool
    current_size: int
    mstep_current_size: int
    n_half: int
    window_spec_kwargs: dict
    disc_type: str
    projector_r_max: int | None
    projection_padding_factor: int
    reconstruction_padding_factor: int
    reconstruction_volume_current_size: int | None
    reconstruction_image_radius: float | None
    image_corrections: object
    scale_corrections: object
    image_pre_shifts: object
    group_ids: object
    scale_correction_group_count: int | None
    scale_correction_data_vs_prior: object
    optics_group_ids: object
    reconstruction_group_ids: object
    reconstruction_group_count: int | None
    score_with_masked_images: bool
    half_spectrum_scoring: bool
    square_window: bool
    accumulate_noise: bool
    use_exact_relion_gaussian: bool
    source_faithful_spectrum_norm: bool
    relion_wavg_atomic_direct_noise: bool
    relion_wavg_atomic_scale_aa: bool
    accumulate_scale: bool
    relion_exact_bpref_operands: bool
    relion_native_fine_units: bool
    relion_firstiter_score_mode: str
    mstep_subtract_ctf_projection: bool
    include_unweighted_norm_high_shell: bool
    relion_translation_angle_scale: float
    precision_policy: object
    symmetry_label: str


def _memory_tiles(
    state: DenseGemmPreparedState, *, n_score: int, n_recon: int,
    n_rect: int, n_groups: int, bp_size: int,
):
    """Conservative bounded workspace policy, including statistics and priors."""

    from relax.sparse_pass2.resident_operands import resident_half_operand_bytes
    from relax.sparse_pass2.resident_statistics import posterior_translation_bucket_scratch_bytes
    from relax.sparse_pass2.sparse_pass2_budget import (
        _device_free_memory_bytes,
        _jax_allocator_free_memory_bytes,
        _jax_allocator_pool_free_bytes,
        device_available_bytes,
    )

    available = device_available_bytes(
        _device_free_memory_bytes(), _jax_allocator_free_memory_bytes(),
        _jax_allocator_pool_free_bytes(),
    )
    budget = int(available * 0.60) if available is not None else 4 << 30
    k = len(state.class_volumes)
    r = int(np.asarray(state.fine_rotations).shape[0])
    t = int(np.asarray(state.fine_translations).shape[0])
    # Account for both padded x-half BPref arrays, the already-live projector
    # slabs, the phase grids, and global image/rotation statistics. Free memory
    # is measured before the new accumulators are allocated.
    ref_bytes = sum(int(ref.size) * int(ref.dtype.itemsize) for ref in state.class_projector_halves)
    shared_priors = all(source is None or np.asarray(source).ndim == 1
                        for source in state.class_rotation_priors)
    fixed = (k * n_groups * bp_size * 12 + ref_bytes
             + t * (n_score + n_recon) * 8 + r * 9 * 4 * 2
             + state.dataset.n_units * (64 + 16 * k)
             + (k * r * 4 if shared_priors else 0))
    # The collapsed BPref pays one projection and adjoint per Q tile, so large
    # Q is valuable even when B is about 100. Expand all translations when
    # the geometry fits; the budget loop below reduces U/B/Q as needed.
    n_rotation_tiles = (r + 3071) // 3072
    q = (r + n_rotation_tiles - 1) // n_rotation_tiles
    u = t
    b = min(int(state.dataset.n_units), 128)
    while True:
        # Keep statistics rows bounded even when the score/adjoint Q is large.
        # Choose a divisor so every Q tile uses the same native row shape.
        qs = min(q, 128)
        while q % qs:
            qs -= 1
        # The canonical half-preparation has a minimum 256-row capacity even
        # when the GEMM batch is smaller; include it and its short-lived raw
        # image staging before admitting a tile.
        prepared = resident_half_operand_bytes(
            n_images=b, n_score_pixels=n_score, n_recon_pixels=n_recon,
            n_rect_pixels=n_rect, n_noise_shells=state.dataset.image_shape[0] // 2 + 1,
            n_fine_trans=t, norm_high_shell_bytes=8,
        ) + max(256, b) * int(np.prod(state.dataset.image_shape)) * 4
        # The statistics rows use a bounded Q_stat. The 40-byte term covers the translated/projected row arrays
        # and Wavg triplet; both are doubled below for XLA/native temporaries.
        # The 2x margin covers XLA fused temporaries and native workspaces.
        tile = (prepared + b * q * u * 16 + b * u * n_rect * 16
                + b * qs * n_recon * 40 + b * n_rect * 16
                + q * (n_score + n_recon) * 24
                + (k * b * r * 4 if not shared_priors else 0) + k * b * t * 4
                + posterior_translation_bucket_scratch_bytes(b * k, b, t)
                + 2 * posterior_translation_bucket_scratch_bytes(b * qs, b, 1))
        if fixed + 2 * tile <= budget:
            break
        if u > 1 and b * u * n_rect >= b * qs * n_recon:
            u = max(1, u // 2)
        elif b > 1:
            b = max(1, b // 2)
        elif q > 1:
            q = max(1, q // 2)
        elif u > 1:
            u = max(1, u // 2)
        else:
            raise MemoryError(
                "dense GEMM minimum B=Q=U=1 tile and resident BPref/statistics "
                f"need {fixed + 2 * tile} bytes, above the {budget}-byte safe budget"
            )
    logging.getLogger(__name__).info(
        "Dense GEMM memory plan: B=%d Q=%d U=%d Q_stat=%d fixed=%.2f GiB "
        "tile=%.2f GiB safe_budget=%.2f GiB",
        b, q, u, qs, fixed / 1024**3, tile / 1024**3, budget / 1024**3,
    )
    return b, q, u, qs


def _expanded_rotation_priors(priors, parent, image_indices, n_images: int, n_classes: int, rpad: int):
    """Broadcast RELION's parent log PDF onto every configured child."""

    parent = np.asarray(parent, np.int32)
    shared = all(source is None or np.asarray(source).ndim == 1 for source in priors)
    rows = []
    for class_id in range(n_classes):
        source = priors[class_id]
        if source is None:
            fine = np.zeros((1 if shared else len(image_indices), len(parent)), np.float32)
        else:
            source = np.asarray(source, np.float32)
            if source.ndim == 1:
                fine = np.broadcast_to(source[parent], (1 if shared else len(image_indices), len(parent)))
            elif source.ndim == 2 and source.shape[0] == n_images:
                fine = source[np.asarray(image_indices), :][:, parent]
            else:
                raise ValueError("class rotation prior must be [coarse R] or [images, coarse R]")
        rows.append(np.pad(fine, ((0, 0), (0, rpad - len(parent))), constant_values=-np.inf))
    return jnp.asarray(np.stack(rows), jnp.float32)


def _full_grid_local_rotation_slots(state: DenseGemmPreparedState) -> np.ndarray:
    """Map a global fine ID to its slot in one complete resident class table.

    Every image has the same full grid here.  Reuse the resident table's
    parent execution key and child order without allocating image-by-rotation
    candidate rows.  K-class tables concatenate these slots class-major.
    """

    from relax.scoring.sparse_bucket_arrays import relion_parent_execution_key
    from relax.sparse_pass2.resident_significance import _ragged_gather, fine_rotation_children

    n_fine = len(state.fine_rotations)
    if state.relion_parent_execution_order:
        child_offsets, child_ids = fine_rotation_children(
            n_coarse_rot=state.n_coarse_rot, nside_level=state.nside_level,
            oversampling_order=state.oversampling_order,
            random_perturbation=state.random_perturbation,
            fine_rotation_parent_override=state.fine_rotation_parent,
            symmetry_label=state.symmetry_label,
        )
        parents = np.arange(state.n_coarse_rot, dtype=np.int64)
        parents = parents[np.argsort(relion_parent_execution_key(
            parents, n_coarse_rot=state.n_coarse_rot, nside_level=state.nside_level,
        ), kind="stable")]
        child_counts = child_offsets[parents + 1] - child_offsets[parents]
        ordered_fine = _ragged_gather(child_offsets, child_ids, parents, child_counts)
    else:
        # Full-support host preparation uses np.flatnonzero over the global
        # fine grid when RELION parent execution order is disabled.
        ordered_fine = np.arange(n_fine, dtype=np.int64)
    if (ordered_fine.shape != (n_fine,) or np.any(ordered_fine < 0)
            or np.any(ordered_fine >= n_fine)
            or np.unique(ordered_fine).size != n_fine):
        raise ValueError("the complete fine grid must give each global rotation one resident slot")
    inverse = np.empty(n_fine, np.int32)
    inverse[ordered_fine] = np.arange(n_fine, dtype=np.int32)
    return inverse


def _dense_reconstruction_operands(resident, *, exact_bpref: bool, n_images: int):
    """Select the resident owner's two BPref operand representations.

    With exact BPref the image is raw and weighted CTF is applied after
    translation. Otherwise ``recon_image`` is already weighted. Both use the
    same CTF-squared/noise denominator supplied by the preparation owner.
    """

    has_weight = resident.recon_weight is not None
    has_direct_ctf = resident.direct_ctf_rfloat_recon is not None
    if has_weight != bool(exact_bpref) or has_direct_ctf != bool(exact_bpref):
        raise ValueError("resident BPref operand presence disagrees with the resolved arithmetic mode")
    image = resident.recon_image[:n_images]
    numerator_image = (image * resident.recon_weight[:n_images]
                       if exact_bpref else image)
    return numerator_image, resident.ctf2_over_nv_recon[:n_images]


def run_dense_gemm_full_grid(state: DenseGemmPreparedState):
    """Return the resident driver's private result after exact full-grid E/M."""

    from recovar.reconstruction import noise as noise_utils

    from relax.helpers.adjoint import mstep_adjoint_max_r
    from relax.helpers.half_spectrum import (
        make_relion_noise_shell_indices_half,
        mask_relion_noise_shell_indices_to_current_window,
    )
    from relax.helpers.half_volume_mstep import half_volume_accumulator_shape, relion_backprojector_volume_shape
    from relax.helpers.scale_groups import prepare_scale_correction_groups
    from relax.sparse_pass2.resident_operands import prepare_resident_half_operands
    from relax.sparse_pass2.resident_pass2 import (
        _make_chunk_translation_sqdist,
        _relion_native_fine_units_in_place,
        _relion_scale_correction_pixel_mask,
    )
    from relax.sparse_pass2.resident_statistics import (
        make_resident_statistics,
        resolve_statistics_config,
    )
    from relax.sparse_pass2.sparse_pass2_bucket_io import _relion_cuda_score_translation_angles_if_available
    from relax.sparse_pass2.sparse_pass2_scoring import _relion_cuda_fine_pixel_weights
    from relax.sparse_pass2.sparse_pass2_wavg import _make_relion_wavg_rectangle
    from relax.sparse_pass2.sparse_pass2_window import _pass2_half_weights, _sparse_pass2_window_setup

    dataset = state.dataset
    image_shape = tuple(dataset.image_shape)
    n_images = int(dataset.n_units)
    k = len(state.class_volumes)
    if n_images <= 0 or k <= 0:
        raise ValueError("dense GEMM needs images and classes")
    if state.precision_policy.score_real_dtype != jnp.float32:
        raise ValueError("dense GEMM production score precision is float32")
    if not state.score_with_masked_images or not state.half_spectrum_scoring:
        raise NotImplementedError("dense GEMM requires masked half-spectrum scoring")
    if state.projector_r_max is None or any(ref is None for ref in state.class_projector_halves):
        raise NotImplementedError("dense GEMM requires supplied RELION projector slabs")
    if state.fine_rotations is None or state.fine_rotation_parent is None or state.fine_mstep_rotations is None:
        raise ValueError("dense GEMM requires the complete configured expanded rotation grid")
    if state.reconstruction_group_ids is None:
        if state.reconstruction_group_count is not None:
            raise ValueError("reconstruction group count needs IDs")
        reconstruction_groups = np.zeros(n_images, np.int32)
        n_groups = 1
    else:
        reconstruction_groups = np.asarray(state.reconstruction_group_ids, np.int32)
        n_groups = int(state.reconstruction_group_count or 0)
        if (reconstruction_groups.shape != (n_images,) or n_groups <= 0
                or np.any(reconstruction_groups < 0) or np.any(reconstruction_groups >= n_groups)):
            raise ValueError("reconstruction group IDs must be one valid 0-based ID per image")

    window = _sparse_pass2_window_setup(
        dataset, disc_type=state.disc_type, image_shape=image_shape,
        current_size=state.current_size, n_half=state.n_half,
        mstep_current_size=state.mstep_current_size,
        square_window=state.square_window,
        window_spec_kwargs=state.window_spec_kwargs,
        use_relion_x_half_mstep=True, log_label="Dense GEMM global pass",
    )
    if window.window_indices is None or window.recon_window_indices is None:
        raise NotImplementedError("dense GEMM requires bounded score and reconstruction windows")
    score_indices = window.window_indices
    recon_indices = window.recon_window_indices
    n_score, n_recon = int(window.n_windowed), int(window.n_recon_windowed)
    _, half_weights_score = _pass2_half_weights(
        image_shape, window.window_spec, half_spectrum_scoring=True,
        relion_firstiter_score_mode=state.relion_firstiter_score_mode, use_float64_scoring=False,
    )
    noise_half = noise_utils.to_batched_half_pixel_noise(state.noise_variance_half, image_shape).squeeze()
    n_optics = 1 if noise_half.ndim == 1 else int(noise_half.shape[0])
    optics = None if state.optics_group_ids is None else np.asarray(state.optics_group_ids, np.int32)
    if n_optics > 1 and (optics is None or optics.shape != (n_images,) or np.any(optics < 0) or np.any(optics >= n_optics)):
        raise ValueError("optics-group IDs must match the per-group noise table")
    angles = _relion_cuda_score_translation_angles_if_available(
        state.fine_translations_source, image_shape, enabled=True, dtype=np.float32,
        angle_scale=state.relion_translation_angle_scale,
    )
    if angles is None:
        raise NotImplementedError("dense GEMM needs canonical RELION translation angles")
    rect = _make_relion_wavg_rectangle(
        image_shape, state.current_size, recon_indices,
        reconstruction_current_size=state.mstep_current_size,
    )
    n_rect = int(rect.centered_indices.size)
    shell_half = mask_relion_noise_shell_indices_to_current_window(
        make_relion_noise_shell_indices_half(image_shape), image_shape,
        state.current_size, score_indices,
    )
    shell_noise = window.window_spec.recon_values(shell_half)
    noise_recon = window.window_spec.recon_values(noise_half)
    n_shells = image_shape[0] // 2 + 1
    scale_group_ids, n_scale_groups = prepare_scale_correction_groups(
        state.group_ids, state.scale_correction_group_count, n_images=n_images,
    )
    class_masks = []
    dvp = state.scale_correction_data_vs_prior
    for class_id in range(k):
        class_dvp = None if dvp is None else (np.asarray(dvp)[class_id] if np.asarray(dvp).ndim == 2 else dvp)
        mask = np.zeros(n_rect, bool)
        mask[rect.exact_positions] = np.asarray(
            _relion_scale_correction_pixel_mask(class_dvp, shell_noise, n_shells=n_shells), bool
        )
        class_masks.append(mask)
    masks = jnp.asarray(class_masks[0] if k == 1 else np.stack(class_masks))
    stats_config = resolve_statistics_config(
        n_shells=n_shells, n_fine_trans=len(state.fine_translations), n_images=n_images,
        n_coarse_rot=k * state.n_coarse_rot, n_scale_groups=n_scale_groups,
        current_size=state.current_size,
        include_unweighted_high_shell=state.include_unweighted_norm_high_shell,
        use_exact_relion_gaussian=state.use_exact_relion_gaussian,
        relion_wavg_atomic_direct_noise=state.relion_wavg_atomic_direct_noise,
        relion_wavg_atomic_scale_aa=state.relion_wavg_atomic_scale_aa,
        accumulate_scale=state.accumulate_scale,
        source_faithful_spectrum_norm=state.source_faithful_spectrum_norm,
        n_optics_groups=n_optics, n_classes=k,
    )
    stats = make_resident_statistics(stats_config, max_posterior_dtype=jnp.float32)

    recon_current = (state.mstep_current_size if state.reconstruction_volume_current_size is None
                     else int(state.reconstruction_volume_current_size))
    bp_shape = relion_backprojector_volume_shape(
        dataset.volume_shape, state.reconstruction_padding_factor, current_size=recon_current,
    )
    bp_size = int(np.prod(half_volume_accumulator_shape(bp_shape)))
    bp_radius = mstep_adjoint_max_r(recon_current, state.reconstruction_image_radius, state.reconstruction_padding_factor)
    b, q, u, qs = _memory_tiles(
        state, n_score=n_score, n_recon=n_recon, n_rect=n_rect,
        n_groups=n_groups, bp_size=bp_size,
    )
    score_phase = native_phase_table(angles, score_indices, image_shape)
    rec_phase = native_phase_table(angles, recon_indices, image_shape)
    grid = pad_grid(
        state.fine_rotations, state.fine_mstep_rotations, score_phase, rec_phase,
        rotation_tile=q, translation_tile=u,
    )
    rpad = int(grid.score_rotations.shape[0])
    tpad = int(grid.score_phase.shape[0])
    fine_local_slots = np.pad(
        _full_grid_local_rotation_slots(state), (0, rpad - len(state.fine_rotations)),
    )
    shared_class_prior = None
    if all(source is None or np.asarray(source).ndim == 1 for source in state.class_rotation_priors):
        shared_class_prior = _expanded_rotation_priors(
            state.class_rotation_priors, state.fine_rotation_parent, (), n_images, k, rpad,
        )
    source_refs = jnp.stack([jnp.asarray(v, jnp.complex64) for v in state.class_projector_halves])
    native_score_project, backproject = native_relion_callbacks(
        image_shape=image_shape, score_indices=score_indices,
        rec_indices=window.relion_x_half_recon_indices,
        r_max=state.projector_r_max, projector_output_size=state.current_size,
        volume_shape=bp_shape, padding_factor=state.projection_padding_factor,
        backprojection_r_max=bp_radius,
    )
    project_rec, _ = native_relion_callbacks(
        image_shape=image_shape, score_indices=recon_indices,
        rec_indices=window.relion_x_half_recon_indices,
        r_max=state.projector_r_max,
        projector_output_size=max(state.current_size, state.mstep_current_size),
        volume_shape=bp_shape, padding_factor=state.projection_padding_factor,
        backprojection_r_max=bp_radius,
    )
    def project(reference, rotations):
        score = native_score_project(reference, rotations)
        return (_relion_native_fine_units_in_place(score, int(np.prod(image_shape)))
                if state.relion_native_fine_units else score)
    stat_plan = DenseGemmStatisticsPlan(
        image_shape=image_shape, image_capacity=b, rotation_tile=q, translation_tile=u,
        rows_per_statistics_block=qs, n_recon_pixels=n_recon, n_rect_pixels=n_rect,
        n_shells=n_shells, n_classes=k, n_coarse_rot=state.n_coarse_rot,
        n_fine_rotations=len(state.fine_rotations),
        n_optics_groups=n_optics, use_rfloat_ctf_wavg=state.relion_exact_bpref_operands,
        score_mode=state.relion_firstiter_score_mode, stats_config=stats_config,
    )
    step, finish_class = make_statistics_callbacks(stat_plan)
    program = make_joint_k_batch_program(
        DenseGemmTileConfig(b, q, u, "image", "exact"), project, backproject,
        score_mode=state.relion_firstiter_score_mode,
        project_reconstruction=project_rec,
        mstep_subtract_ctf_projection=state.mstep_subtract_ctf_projection,
        statistics_step=step, statistics_finish_class=finish_class,
    )
    numerator = jnp.zeros((k, n_groups, bp_size), jnp.complex64)
    denominator = jnp.zeros((k, n_groups, bp_size), jnp.float32)
    invalid_pair = jnp.zeros((n_images,), bool)
    invalid_mass = jnp.zeros((n_images,), bool)
    bucket_kwargs = dict(
        noise_variance_half=noise_half, fine_translations=state.fine_translations,
        config=window.config, n_trans=len(state.fine_translations),
        score_with_masked_images=True, half_spectrum_scoring=True,
        image_corrections=state.image_corrections, scale_corrections=state.scale_corrections,
        image_pre_shifts=state.image_pre_shifts, use_float64_scoring=False,
        score_only=False, score_mode=state.relion_firstiter_score_mode,
        window_indices=score_indices, recon_window_indices=recon_indices,
        translation_phases_half=None,
        relion_score_translation_angles=angles, return_windowed_shifted=window.windowed_prepare,
        relion_exact_normalized_cc_operands=state.relion_firstiter_score_mode == "normalized_cc",
        relion_exact_bpref_operands=state.relion_exact_bpref_operands, noise_optics_groups=optics,
    )
    for start in range(0, n_images, b):
        count = min(b, n_images - start)
        image_ids = np.arange(start, start + count, dtype=np.int32)
        resident = prepare_resident_half_operands(
            dataset, image_ids, bucket_io_kwargs=bucket_kwargs,
            window_indices=score_indices, recon_window_indices=recon_indices,
            wavg_rect_indices=rect.centered_indices,
            noise_shell_indices_half=shell_half, n_noise_shells=n_shells,
            image_shape=image_shape, current_size=state.current_size,
            n_fine_trans=len(state.fine_translations),
            use_exact_relion_gaussian=state.use_exact_relion_gaussian,
            accumulate_noise=state.accumulate_noise,
            source_faithful_spectrum_norm=state.source_faithful_spectrum_norm,
            fine_translation_prior_2d=state.fine_translation_prior,
            scale_corrections_np=state.scale_corrections,
            group_ids_np=scale_group_ids, precision_policy=state.precision_policy,
            image_batch_size=b, optics_groups_np=optics,
            relion_native_fine_units=state.relion_native_fine_units,
            log_summary=False, allow_normalized_cc=True,
        )
        score_weight = _relion_cuda_fine_pixel_weights(resident.corr_img_score[:count], half_weights_score)
        initial_diff2 = (jnp.zeros((count,), jnp.float32) if resident.highres_xi2_half is None
                         else resident.highres_xi2_half[:count])
        rec_image, rec_denominator = _dense_reconstruction_operands(
            resident, exact_bpref=state.relion_exact_bpref_operands, n_images=count,
        )
        batch = pad_batch(
            resident.score_input[:count], score_weight, initial_diff2,
            rec_image, rec_denominator,
            None,
            resident.translation_prior[:count], image_ids,
            image_capacity=b, grid=grid, sentinel_id=n_images,
            **(
                dict(rec_raw_image=resident.recon_image[:count],
                     rec_weighted_ctf=resident.recon_weight[:count])
                if state.relion_exact_bpref_operands else {}
            ),
        )
        class_prior = shared_class_prior
        if class_prior is None:
            class_prior = _expanded_rotation_priors(
                state.class_rotation_priors, state.fine_rotation_parent,
                image_ids, n_images, k, rpad,
            )
            class_prior = jnp.pad(class_prior, ((0, 0), (0, b - count), (0, 0)), constant_values=-jnp.inf)
        recon_groups = jnp.pad(jnp.asarray(reconstruction_groups[start:start + count]), (0, b - count), constant_values=-1)
        def take_and_pad(array, *, fill=0):
            return jnp.pad(jnp.asarray(array[:count]), ((0, b - count),) + ((0, 0),) * (array.ndim - 1), constant_values=fill)
        fine_parent = np.pad(np.asarray(state.fine_rotation_parent, np.int32), (0, rpad - len(state.fine_rotations)), constant_values=state.n_coarse_rot)
        fine_ids = np.pad(np.arange(len(state.fine_rotations), dtype=np.int32), (0, rpad - len(state.fine_rotations)))
        sqdist = _make_chunk_translation_sqdist(
            None, translation_prior_centers_np=state.translation_prior_centers,
            image_indices=image_ids, image_capacity=b, n_valid_images=count,
            fine_translations=state.fine_translations, voxel_size=dataset.voxel_size,
        )
        ops = DenseGemmStatisticsOperands(
            recon_image=take_and_pad(resident.recon_image),
            recon_weight=(None if resident.recon_weight is None
                          else take_and_pad(resident.recon_weight)),
            noise_image=take_and_pad(resident.noise_image),
            ctf2_over_nv_recon=take_and_pad(resident.ctf2_over_nv_recon),
            direct_ctf_rfloat_recon=(None if resident.direct_ctf_rfloat_recon is None
                                    else take_and_pad(resident.direct_ctf_rfloat_recon)),
            wavg_image_rect=take_and_pad(resident.wavg_image_rect),
            scale=take_and_pad(resident.scale),
            optics_groups=None if resident.optics_groups is None else take_and_pad(resident.optics_groups),
            image_noise_scale=None,
            noise_variance=noise_recon, shell_indices_noise=shell_noise,
            translation_angles=jnp.pad(jnp.asarray(angles), ((0, tpad - len(angles)), (0, 0))),
            recon_pixel_indices=jnp.asarray(recon_indices, jnp.int32),
            rect_indices=jnp.asarray(rect.centered_indices, jnp.int32),
            exact_positions=jnp.asarray(rect.exact_positions, jnp.int32),
            logical_recon_pixels=jnp.asarray(n_recon, jnp.int32),
            logical_rect_pixels=jnp.asarray(n_rect, jnp.int32),
            wavg_scale_pixel_mask=masks,
            image_power_shells=take_and_pad(resident.image_power_shells),
            relion_norm_high_shell=(
                jnp.zeros((b,), jnp.float32) if resident.relion_norm_high_shell is None
                else take_and_pad(resident.relion_norm_high_shell)
            ),
            image_ids=jnp.pad(jnp.asarray(image_ids), (0, b - count), constant_values=-1),
            group_ids=jnp.pad(
                jnp.asarray(np.full(count, -1, np.int32) if scale_group_ids is None
                            else scale_group_ids[start:start + count]),
                (0, b - count), constant_values=-1,
            ),
            translation_sqdist_ang=sqdist,
            fine_to_coarse_parent=jnp.asarray(fine_parent),
            fine_rotation_ids=jnp.asarray(fine_ids),
            fine_local_slots=jnp.asarray(fine_local_slots),
            cc_half_batch_norm=(
                jnp.zeros((b,), jnp.float32) if resident.cc_half_batch_norm is None
                else take_and_pad(resident.cc_half_batch_norm)
            ),
            wavg_shell_indices=jnp.asarray(rect.shell_indices, jnp.int32),
            norm_shell_cutoff=None,
        )
        result = program(
            source_refs, numerator, denominator, batch, grid, class_prior,
            # A2/XA are statistics, accumulated across every rotation tile.
            # Keep their F32 row arithmetic, then retain the small per-image
            # block totals in the same F64 metadata stream as the final norm.
            recon_groups, initial_statistics_carry(stat_plan, noise_real_dtype=jnp.float64), ops,
        )
        numerator, denominator = result.numerator, result.denominator
        stats = accumulate_batch_statistics(stats, result, result.statistics_state, ops, batch, plan=stat_plan)
        invalid_pair = invalid_pair.at[start:start + count].set(result.invalid_normalizer[:count])
        invalid_mass = invalid_mass.at[start:start + count].set(result.invalid_weight[:count])
    if bool(jnp.any(invalid_pair | invalid_mass)):
        raise FloatingPointError("dense GEMM produced a nonfinite posterior or empty reconstruction weight")
    return _finalize_dense_result(
        state, numerator, denominator, stats, stats_config, bp_shape,
        n_groups=n_groups,
    )


def _finalize_dense_result(state, numerator, denominator, stats, stats_config, bp_shape, *, n_groups):
    """Use the resident BPref/statistics finalizers, with one terminal host pull."""

    from relax.helpers.half_volume_mstep import (
        finalize_half_volume_bpref,
        relion_x_half_accumulators_to_public_layout,
    )
    from relax.helpers.types import make_noise_stats
    from relax.sparse_pass2.resident_pass2 import _ResidentPass2Result
    from relax.sparse_pass2.resident_statistics import finalize_statistics

    k = int(numerator.shape[0])
    final_y, final_w = [], []
    for group in range(n_groups):
        for class_id in range(k):
            y, w = finalize_half_volume_bpref(
                numerator[class_id, group], denominator[class_id, group], bp_shape,
                logger=logging.getLogger(__name__), label="Dense GEMM global pass",
                symmetry_label=state.symmetry_label, relion_x_half=True,
            )
            y, w = relion_x_half_accumulators_to_public_layout(y, w, bp_shape)
            final_y.append(y)
            final_w.append(w)
    finalized = finalize_statistics(stats, config=stats_config, n_images=state.dataset.n_units)
    noise = make_noise_stats(
        wsum_sigma2_noise=finalized.wsum_sigma2_noise,
        wsum_img_power=finalized.wsum_img_power,
        wsum_sigma2_offset=finalized.wsum_sigma2_offset,
        sumw=finalized.sumw,
        wsum_norm_correction=finalized.wsum_norm_correction,
        wsum_scale_correction_xa=finalized.wsum_scale_correction_xa,
        wsum_scale_correction_aa=finalized.wsum_scale_correction_aa,
    )
    return _ResidentPass2Result(
        Ft_y=tuple(final_y), Ft_ctf=tuple(final_w), n_classes=k,
        n_slot_groups=n_groups, finalized=finalized, noise_stats=noise,
        fine_translations=np.asarray(state.fine_translations, np.float32),
        score_real_dtype=jnp.float32,
    )
