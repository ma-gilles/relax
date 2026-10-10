"""Matched native fixed-state checks of the public resident and dense GEMM pass."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from helpers.float_compare import assert_matches


def _install_native_preprocessing(dataset):
    """Use RECOVAR's registered RELION CUDA preprocessor and FFT owner."""
    from recovar.data_io import image_backends

    backend = SimpleNamespace(
        image_mask=np.ones(dataset.image_shape, np.float32),
        image_mask_mode="relion_background_fill",
        relion_fourier_backend="relion_cuda",
    )
    dataset.image_source.backend = backend
    dataset.image_source.image_mask = backend.image_mask

    def process_half(
        batch, apply_image_mask=False, *, relion_normalization_factors,
        relion_integer_shifts, relion_fft_per_image=False,
    ):
        _, processed = image_backends.relion_cuda_preprocessor()(
            jnp.asarray(batch, jnp.float32),
            jnp.asarray(relion_normalization_factors, jnp.float32),
            jnp.asarray(relion_integer_shifts, jnp.int32),
            3.5, 1.0, bool(apply_image_mask),
            native_lane_reduction=False, native_atomic_reduction=False,
        )
        fft = (image_backends._centered_rfft2_jax_per_image if relion_fft_per_image
               else image_backends._centered_rfft2_jax)
        transformed = fft(processed)
        return transformed.reshape((transformed.shape[0], -1)).astype(jnp.complex64)

    dataset.process_images_half = process_half


def _variable_ctf_rows(_dataset, image_indices, image_shape, *, pixel_indices=None):
    all_pixels = int(image_shape[0]) * (int(image_shape[1]) // 2 + 1)
    pixels = np.arange(all_pixels if pixel_indices is None else len(pixel_indices), dtype=np.float64)
    if pixel_indices is not None:
        pixels = np.asarray(pixel_indices, np.float64)
    images = np.asarray(image_indices, np.float64)
    return jnp.asarray(0.7 + 0.007 * images[:, None] + 0.002 * pixels[None, :], jnp.float64)


def _native_k1_args(monkeypatch, *, seed=20260918):
    """A full-support fixture with canonical CUDA preprocessing and exact BPref."""
    from relax.relion import ctf as relion_ctf

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "unit"))
    from unit.test_resident_pass2_driver import _driver_fixture_args
    from unit.test_resident_relion_reference import _ppref, _tie_free_noise

    monkeypatch.setenv("RELAX_EM_PROTOTYPE_SOFT_POSTERIOR_BLOCK_BPREF", "1")
    monkeypatch.setenv("RELAX_RELION_WAVG_ATOMIC_SCALE_AA", "1")
    monkeypatch.setenv("RELAX_RELION_WAVG_ATOMIC_DIRECT_NOISE_ONLY", "1")
    args = _driver_fixture_args(seed=seed)
    dataset = args["experiment_dataset"]
    _install_native_preprocessing(dataset)
    monkeypatch.setattr(relion_ctf, "relion_exact_ctf_half_from_source_star", _variable_ctf_rows)
    half, r_max = _ppref(args["volume"])
    noise = _tie_free_noise(6, 200.0).reshape(-1)
    noise = np.where(noise > 1e20, noise, noise * (1 + 0.001 * np.arange(noise.size)))
    args.update(
        significant_sample_indices=[None] * dataset.n_units,
        score_with_masked_images=True,
        source_faithful_spectrum_norm=True,
        relion_projector_half=jnp.asarray(half, jnp.complex64),
        relion_projector_r_max=r_max,
        noise_variance=jnp.asarray(noise, jnp.float32),
        normalization_other_score_log_z=None,
        adaptive_fraction=1.0,
        relion_fine_mstep_prune=False,
    )
    return args


def _k1_output_arrays(control, candidate):
    arrays = {}
    for arm_name, arm in (("control", control), ("candidate", candidate)):
        for name in ("hard_assignment", "best_rotation_indices", "best_translations", "score_log_z", "Ft_y", "Ft_ctf"):
            value = getattr(arm, name, None)
            if value is not None:
                arrays[f"{arm_name}_{name}"] = np.asarray(value)
        for record_name in ("relion_stats", "noise_stats"):
            record = getattr(arm, record_name, None)
            if record is None:
                continue
            for field in record._fields:
                value = getattr(record, field)
                if value is not None:
                    arrays[f"{arm_name}_{record_name}_{field}"] = np.asarray(value)
    return arrays


def _assert_paired_native_noise(candidate, control):
    """Apply the existing native-reference noise and scale gates to a pair.

    See tests/unit/test_resident_relion_reference.py::_assert_k1_pass.
    These are the approved float32-derived comparison bands for this test.
    """
    from unit.test_resident_relion_reference import F32, _rel_l2

    got, want = candidate.noise_stats, control.noise_stats
    got_total = np.asarray(got.wsum_sigma2_noise) + np.asarray(got.wsum_img_power)
    want_total = np.asarray(want.wsum_sigma2_noise) + np.asarray(want.wsum_img_power)
    assert _rel_l2(got_total, want_total) < 1e-6
    assert_matches(got.wsum_img_power, want.wsum_img_power, rtol=F32)
    assert_matches(got.wsum_sigma2_offset, want.wsum_sigma2_offset, rtol=F32)
    assert _rel_l2(got.wsum_norm_correction, want.wsum_norm_correction) < 1e-6
    for name in ("wsum_scale_correction_xa", "wsum_scale_correction_aa"):
        assert _rel_l2(getattr(got, name), getattr(want, name)) < 1e-6, name
    assert float(got.sumw) == pytest.approx(float(want.sumw), rel=1e-6)


def _capture_chunk_norm(collector, original, stats, operands, tables, *, config):
    """Opt-in per-image norm boundary, using the same public statistics owners."""
    from relax.fine_pass.resident_statistics import (
        segment_sum_by_image, weighted_image_power_from_shells,
    )

    image_capacity = int(operands.image_ids.shape[0])
    posterior = operands.row_posterior
    marginal = segment_sum_by_image(
        posterior, operands.row_image_local, image_capacity,
        float32_posterior_bucket_size=128 if posterior.dtype == jnp.float32 else None,
    )
    support_mass = jnp.where(operands.image_ids >= 0, jnp.sum(marginal, axis=1), 0.0)
    cutoff = (tables.norm_shell_cutoff if tables.norm_shell_cutoff is not None
              else config.norm_unweighted_shell_cutoff)
    _, weighted_per_image = weighted_image_power_from_shells(
        operands.image_power_shells, support_mass, operands.relion_norm_high_shell,
        operands.image_ids >= 0, norm_unweighted_shell_cutoff=cutoff,
        include_unweighted_high_shell=config.include_unweighted_high_shell,
        deterministic_norm_reduction=config.deterministic_norm_reduction,
    )
    result = original(stats, operands, tables, config=config)
    names = ("image_ids", "support_mass", "weighted_image_power", "high_shell",
             "a2", "xa", "wavg_triplet", "norm_before", "norm_after")
    high_shell = (jnp.zeros_like(support_mass) if operands.relion_norm_high_shell is None
                  else operands.relion_norm_high_shell)
    jax.debug.callback(
        lambda *values: collector.append(dict(zip(names, map(np.asarray, values)))),
        operands.image_ids, support_mass, weighted_per_image, high_shell,
        operands.a2_per_image, operands.xa_per_image, operands.wavg_triplet_pixels,
        stats.norm_correction, result.norm_correction,
    )
    return result


def _capture_pre_scatter_norm(
    collector, original, proj, proj_abs2, summed_masked, ctf_probs,
    noise_variance, shell_indices, row_image_local, row_optics_groups=None,
    *, n_shells, image_capacity,
):
    """Save only scalar A2/XA rows and their existing F32 block scatters."""
    from relax.fine_pass.resident_statistics import _flat_row_norm_and_scale_terms

    result = original(
        proj, proj_abs2, summed_masked, ctf_probs, noise_variance,
        shell_indices, row_image_local, row_optics_groups,
        n_shells=n_shells, image_capacity=image_capacity,
    )
    row_noise = (noise_variance if row_optics_groups is None
                 else jnp.asarray(noise_variance)[row_optics_groups])
    row_a2, row_xa = _flat_row_norm_and_scale_terms(
        proj, proj_abs2, summed_masked, ctf_probs, row_noise,
    )
    names = ("row_image", "row_a2", "row_xa", "block_a2", "block_xa")
    jax.debug.callback(
        lambda *values: collector.append(dict(zip(names, map(np.asarray, values)))),
        row_image_local, row_a2, row_xa, result[1], result[2], ordered=True,
    )
    return result


def _save_gradient_row_diagnostic(path, control_rows, candidate_rows,
                                  control_norm_parts, candidate_norm_parts):
    """Compare F64 diagnostic row sums with the two F32 aggregation stages."""
    arrays = {}
    for label, rows, norm_parts in (
        ("control", control_rows, control_norm_parts),
        ("candidate", candidate_rows, candidate_norm_parts),
    ):
        if not rows or len(norm_parts) != 1:
            raise AssertionError(f"{label}: expected captured A2/XA rows and one image chunk")
        image_ids = np.asarray(norm_parts[0]["image_ids"], np.int32)
        n_images = int(np.count_nonzero(image_ids >= 0))
        arrays[f"{label}_image_ids"] = image_ids
        row_offsets = np.concatenate((
            np.zeros(1, np.int64),
            np.cumsum([len(block["row_image"]) for block in rows], dtype=np.int64),
        ))
        arrays[f"{label}_row_offsets"] = row_offsets
        arrays[f"{label}_block_image_capacity"] = np.asarray(
            [len(block["block_a2"]) for block in rows], np.int32,
        )
        for name in ("row_image", "row_a2", "row_xa"):
            arrays[f"{label}_{name}"] = np.concatenate([np.asarray(block[name]) for block in rows])
        for block_id, block in enumerate(rows):
            for name in ("block_a2", "block_xa"):
                arrays[f"{label}_block_{block_id:04d}_{name}"] = np.asarray(block[name])
        slots = arrays[f"{label}_row_image"]
        live = (slots >= 0) & (slots < image_ids.size)
        global_ids = np.full(slots.shape, -1, np.int32)
        global_ids[live] = image_ids[slots[live]]
        valid = (global_ids >= 0) & (global_ids < n_images)
        arrays[f"{label}_row_global_image"] = global_ids
        for term in ("a2", "xa"):
            row_values = np.asarray(arrays[f"{label}_row_{term}"], np.float64)
            arrays[f"{label}_unmapped_row_{term}"] = np.sum(row_values[~valid], dtype=np.float64)
            f64_rows = np.zeros(n_images, np.float64)
            np.add.at(f64_rows, global_ids[valid], row_values[valid])
            arrays[f"{label}_f64_row_{term}_by_image"] = f64_rows
            f64_blocks = np.zeros(n_images, np.float64)
            unmapped_blocks = np.float64(0.0)
            for block in rows:
                partial = np.asarray(block[f"block_{term}"], np.float64)
                for slot in range(min(len(partial), len(image_ids))):
                    global_image = int(image_ids[slot])
                    if 0 <= global_image < n_images:
                        f64_blocks[global_image] += partial[slot]
                    else:
                        unmapped_blocks += partial[slot]
                unmapped_blocks += np.sum(partial[len(image_ids):], dtype=np.float64)
            arrays[f"{label}_f64_block_{term}_by_image"] = f64_blocks
            arrays[f"{label}_unmapped_block_{term}"] = unmapped_blocks
            arrays[f"{label}_actual_{term}_by_image"] = np.asarray(
                norm_parts[0][term], np.float64,
            )[:n_images]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"Dense GEMM pre-scatter gradient norm diagnostic saved: {path}")


@pytest.mark.unit
def test_gradient_row_diagnostic_captures_jitted_float32_rows(tmp_path):
    """The opt-in callback retains row dtype and maps image-local rows offline."""
    captured = []

    def original(_proj, _abs2, _summed, _ctf, _noise, _shells, _ids, _optics,
                 *, n_shells, image_capacity):
        return (jnp.zeros((n_shells,), jnp.float64),
                jnp.zeros((image_capacity,), jnp.float32),
                jnp.zeros((image_capacity,), jnp.float32))

    @jax.jit
    def run():
        proj = jnp.asarray([[1.0 + 0j, 2.0 + 0j], [3.0 + 0j, 4.0 + 0j]], jnp.complex64)
        return _capture_pre_scatter_norm(
            captured, original, proj, jnp.abs(proj) ** 2,
            jnp.ones_like(proj), jnp.ones((2, 2), jnp.float32),
            jnp.ones((2,), jnp.float32), jnp.asarray([0, 1], jnp.int32),
            jnp.asarray([0, 1], jnp.int32),
            n_shells=2, image_capacity=2,
        )

    jax.block_until_ready(run())
    assert len(captured) == 1
    assert captured[0]["row_a2"].dtype == np.float32
    assert captured[0]["row_xa"].dtype == np.float32
    norm_parts = [{"image_ids": np.asarray([0, 1], np.int32),
                   "a2": np.zeros(2, np.float32), "xa": np.zeros(2, np.float32)}]
    path = tmp_path / "rows.npz"
    _save_gradient_row_diagnostic(path, captured, captured, norm_parts, norm_parts)
    saved = np.load(path)
    np.testing.assert_allclose(saved["control_f64_row_a2_by_image"], [5.0, 25.0])
    assert saved["control_row_a2"].dtype == np.float32


@pytest.mark.unit
def test_gradient_row_diagnostic_serializes_ragged_blocks_and_image_ids(tmp_path):
    """Different row lengths and image capacities retain their own slot maps."""

    def block(ids, a2, xa, block_a2, block_xa):
        return dict(
            row_image=np.asarray(ids, np.int32), row_a2=np.asarray(a2, np.float32),
            row_xa=np.asarray(xa, np.float32),
            block_a2=np.asarray(block_a2, np.float32),
            block_xa=np.asarray(block_xa, np.float32),
        )

    control = [
        block([0, 1], [1, 2], [.1, .2], [1, 2, 0, 0], [.1, .2, 0, 0]),
        block([2, 0, -1], [3, 4, 0], [.3, .4, 0], [4, 0, 3], [.4, 0, .3]),
    ]
    candidate = [
        block([1], [2], [.2], [0, 2], [0, .2]),
        block([0, 2, 0, -1], [1, 3, 4, 0], [.1, .3, .4, 0],
              [5, 0, 3, 0], [.5, 0, .3, 0]),
    ]
    norm = [{"image_ids": np.asarray([2, 0, 1, -1], np.int32),
             "a2": np.asarray([5, 2, 3, 0], np.float32),
             "xa": np.asarray([.5, .2, .3, 0], np.float32)}]
    path = tmp_path / "ragged.npz"
    _save_gradient_row_diagnostic(path, control, candidate, norm, norm)
    data = np.load(path)
    np.testing.assert_array_equal(data["control_row_offsets"], [0, 2, 5])
    np.testing.assert_array_equal(data["candidate_row_offsets"], [0, 1, 5])
    np.testing.assert_array_equal(data["control_block_image_capacity"], [4, 3])
    np.testing.assert_array_equal(data["candidate_block_image_capacity"], [2, 4])
    np.testing.assert_array_equal(data["control_row_global_image"], [2, 0, 1, 2, -1])
    np.testing.assert_array_equal(data["candidate_row_global_image"], [0, 2, 1, 2, -1])
    for label in ("control", "candidate"):
        np.testing.assert_allclose(data[f"{label}_f64_row_a2_by_image"], [2, 3, 5])
        np.testing.assert_allclose(data[f"{label}_f64_block_a2_by_image"], [2, 3, 5])
        np.testing.assert_allclose(data[f"{label}_f64_row_xa_by_image"], [.2, .3, .5], rtol=2e-7)
        assert data[f"{label}_block_0000_block_a2"].dtype == np.float32


def _save_k4_output_diagnostic(path, control, candidate, *, resident_mass_parts=(),
                               dense_batches=(), control_norm_parts=(), candidate_norm_parts=()):
    """Retain every returned K-class state field before comparison assertions."""
    arrays = {}
    for arm_name, arm in (("control", control), ("candidate", candidate)):
        for name in (
            "class_log_evidence_per_image", "class_best_log_score_per_image",
            "class_rotation_posterior_sums", "class_reconstruction_posterior_sums",
            "per_class_hard_assignments",
        ):
            arrays[f"{arm_name}_{name}"] = np.asarray(getattr(arm, name))
        for name in ("Ft_y", "Ft_ctf"):
            for class_id, value in enumerate(getattr(arm, name)):
                arrays[f"{arm_name}_{name}_{class_id}"] = np.asarray(value)
        for record_name in ("stats", "noise_stats"):
            record = getattr(arm, record_name)
            for field in record._fields:
                value = getattr(record, field)
                if value is not None:
                    arrays[f"{arm_name}_{record_name}_{field}"] = np.asarray(value)
    for batch_id, (image_ids, mass, posterior, row_image) in enumerate(resident_mass_parts):
        arrays[f"control_batch_{batch_id}_image_ids"] = np.asarray(image_ids)
        arrays[f"control_batch_{batch_id}_support_mass"] = np.asarray(mass)
        arrays[f"control_batch_{batch_id}_row_posterior"] = np.asarray(posterior)
        arrays[f"control_batch_{batch_id}_row_image_local"] = np.asarray(row_image)
    for batch_id, result in enumerate(dense_batches):
        for field in (
            "joint_normalizer", "class_normalizers", "image_posterior_mass",
            "class_posterior_sums", "class_translation_marginals",
        ):
            arrays[f"candidate_batch_{batch_id}_{field}"] = np.asarray(getattr(result, field))
    for label, parts in (("control", control_norm_parts), ("candidate", candidate_norm_parts)):
        for batch_id, part in enumerate(parts):
            for field, value in part.items():
                arrays[f"{label}_norm_batch_{batch_id}_{field}"] = np.asarray(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"Dense GEMM K4 output diagnostic saved: {path}")


def _save_k1_score_diagnostic(path, state, batch, control, candidate):
    """Persist outputs first, then matched native/GEMM scores at both winners."""
    from relax.cuda import kernels as em_cuda_kernels
    from relax.dense.gemm_experiment import native_phase_table, native_relion_callbacks
    from relax.dense.gemm_experiment_kernels import score_tile
    from relax.dense.gemm_coarse_engine import _expanded_rotation_priors
    from relax.fine_pass.resident_pass2 import _relion_native_fine_units_in_place
    from relax.fine_pass.bucket_io import _relion_cuda_score_translation_angles_if_available
    from relax.fine_pass.projection_blocks import _compute_sparse_pass2_windowed_projections_block
    from relax.fine_pass.scoring import _relion_cuda_fine_full_to_compact_lookup
    from relax.fine_pass.window import _sparse_pass2_window_setup

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _k1_output_arrays(control, candidate)
    np.savez_compressed(path, **arrays)

    b = int(state.dataset.n_units)
    t = int(np.asarray(state.fine_translations).shape[0])
    winner_pose = np.stack([np.asarray(control.hard_assignment), np.asarray(candidate.hard_assignment)], axis=1)
    winner_rot = winner_pose // t
    winner_trans = winner_pose % t
    unique_rot, inverse = np.unique(winner_rot, return_inverse=True)
    inverse = inverse.reshape(b, 2)
    window = _sparse_pass2_window_setup(
        state.dataset, disc_type=state.disc_type, image_shape=tuple(state.dataset.image_shape),
        current_size=state.current_size, n_half=state.n_half,
        mstep_current_size=state.mstep_current_size, square_window=state.square_window,
        window_spec_kwargs=state.window_spec_kwargs, use_relion_x_half_mstep=True,
        log_label="Dense GEMM score diagnostic",
    )
    score_indices = window.window_indices
    score_project, _ = native_relion_callbacks(
        image_shape=tuple(state.dataset.image_shape), score_indices=score_indices,
        rec_indices=window.relion_x_half_recon_indices,
        r_max=state.projector_r_max, projector_output_size=state.current_size,
        volume_shape=tuple(state.dataset.volume_shape), padding_factor=state.projection_padding_factor,
    )
    rotations = jnp.asarray(state.fine_rotations)[unique_rot]
    slab = jnp.asarray(state.class_projector_halves[0], jnp.complex64)
    projected_dense = score_project(slab, rotations)
    projected_resident, _, _ = _compute_sparse_pass2_windowed_projections_block(
        state.class_volumes[0], rotations, tuple(state.dataset.image_shape),
        tuple(state.dataset.volume_shape), state.disc_type,
        score_indices=score_indices, recon_indices=None,
        relion_projector_half=slab, relion_projector_r_max=state.projector_r_max,
        projection_padding_factor=state.projection_padding_factor,
        projector_output_size=state.current_size, mask_current_image_disk=False,
        **window.window_spec.projection_kwargs(return_abs2=False),
    )
    if state.relion_native_fine_units:
        projected_dense = _relion_native_fine_units_in_place(projected_dense, int(np.prod(state.dataset.image_shape)))
        projected_resident = _relion_native_fine_units_in_place(projected_resident, int(np.prod(state.dataset.image_shape)))
    angles = _relion_cuda_score_translation_angles_if_available(
        state.fine_translations_source, tuple(state.dataset.image_shape), enabled=True,
        dtype=np.float32, angle_scale=state.relion_translation_angle_scale,
    )
    phase = native_phase_table(angles, score_indices, tuple(state.dataset.image_shape))
    full_to_compact = _relion_cuda_fine_full_to_compact_lookup(
        tuple(state.dataset.image_shape), state.current_size, window.window_indices_np,
    )
    priors = _expanded_rotation_priors(
        state.class_rotation_priors, state.fine_rotation_parent, (), b, 1, len(state.fine_rotations),
    )[0]
    selected_prior = jnp.broadcast_to(priors[:, unique_rot], (b, len(unique_rot)))
    native_row_image = np.repeat(np.arange(b, dtype=np.int32), 2)
    native_project_index = inverse.reshape(-1)

    def native_scores(projected):
        raw = em_cuda_kernels.relion_fine_diff2_fused_translate_runtime_flat_rows_f32(
            projected[native_project_index], jnp.asarray(native_row_image), batch.score_image,
            jnp.asarray(angles, jnp.float32), batch.score_weight,
            jnp.asarray(full_to_compact, jnp.int32), jnp.asarray(state.current_size, jnp.int32),
            batch.initial_diff2,
        ).reshape(b, 2, t)
        prior = selected_prior[jnp.arange(b)[:, None], jnp.asarray(inverse)]
        return -raw + prior[:, :, None] + batch.translation_prior[:b, None, :]

    def gemm_scores(projected):
        all_scores = score_tile(
            projected, batch.score_image[:b], batch.score_weight[:b], batch.initial_diff2[:b],
            phase, selected_prior, batch.translation_prior[:b, :t],
            jnp.ones((b,), bool), jnp.ones((len(unique_rot),), bool), jnp.ones((t,), bool),
            translation_side="image",
        )
        return all_scores[jnp.arange(b)[:, None], jnp.asarray(inverse), :]

    arrays.update({
        "winner_pose": winner_pose, "winner_rot": winner_rot, "winner_trans": winner_trans,
        "unique_rot": unique_rot, "projection_dense": np.asarray(projected_dense),
        "projection_resident": np.asarray(projected_resident),
        "native_scores_dense_projection": np.asarray(native_scores(projected_dense)),
        "native_scores_resident_projection": np.asarray(native_scores(projected_resident)),
        "gemm_scores_dense_projection": np.asarray(gemm_scores(projected_dense)),
        "gemm_scores_resident_projection": np.asarray(gemm_scores(projected_resident)),
        "score_image": np.asarray(batch.score_image[:b]),
        "score_weight": np.asarray(batch.score_weight[:b]),
        "initial_diff2": np.asarray(batch.initial_diff2[:b]),
        "translation_prior": np.asarray(batch.translation_prior[:b, :t]),
        "rotation_prior": np.asarray(selected_prior),
        "phase": np.asarray(phase), "angles": np.asarray(angles),
        "score_indices": np.asarray(score_indices),
    })
    np.savez_compressed(path, **arrays)
    print(f"Dense GEMM matched-score diagnostic saved: {path}")
    print("max projection delta:", float(np.max(np.abs(arrays["projection_dense"] - arrays["projection_resident"]))))
    for name in ("native_scores_dense_projection", "native_scores_resident_projection", "gemm_scores_dense_projection"):
        print(name, "image 0 winning cells:", arrays[name][0, np.arange(2), winner_trans[0]])


def _save_k1_cc_score_diagnostic(path, state, batch, grid, resident, control_chunks, control, candidate):
    """Keep actual fine-CC inputs and compare fine and GEMM scores at both winners."""
    from relax.dense.gemm_experiment import native_relion_callbacks
    from relax.dense.gemm_experiment_kernels import cc_score_tile
    from relax.scoring.coarse_kernels import relion_coarse_gemm_terms
    from relax.fine_pass.resident_pass2 import _relion_native_fine_units_in_place
    from relax.fine_pass.scoring import (
        _relion_cuda_fine_normalized_cc_score,
        _relion_cuda_fine_full_to_compact_lookup,
    )
    from relax.fine_pass.window import _pass2_half_weights, _sparse_pass2_window_setup

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _k1_output_arrays(control, candidate)
    n_images = int(state.dataset.n_units)
    n_trans = len(state.fine_translations)
    winner_rot = np.stack((np.asarray(control.best_rotation_indices),
                           np.asarray(candidate.best_rotation_indices)), axis=1)
    unique_rot, inverse = np.unique(winner_rot, return_inverse=True)
    inverse = inverse.reshape(n_images, 2)
    window = _sparse_pass2_window_setup(
        state.dataset, disc_type=state.disc_type, image_shape=tuple(state.dataset.image_shape),
        current_size=state.current_size, n_half=state.n_half,
        mstep_current_size=state.mstep_current_size, square_window=state.square_window,
        window_spec_kwargs=state.window_spec_kwargs, use_relion_x_half_mstep=True,
        log_label="Dense GEMM CC score diagnostic",
    )
    project, _ = native_relion_callbacks(
        image_shape=tuple(state.dataset.image_shape), score_indices=window.window_indices,
        rec_indices=window.relion_x_half_recon_indices,
        r_max=state.projector_r_max, projector_output_size=state.current_size,
        volume_shape=tuple(state.dataset.volume_shape),
        padding_factor=state.projection_padding_factor,
    )
    projection = project(
        jnp.asarray(state.class_projector_halves[0], jnp.complex64),
        jnp.asarray(state.fine_rotations)[unique_rot],
    )
    if state.relion_native_fine_units:
        projection = _relion_native_fine_units_in_place(
            projection, int(np.prod(state.dataset.image_shape)),
        )
    _, half_weights = _pass2_half_weights(
        tuple(state.dataset.image_shape), window.window_spec,
        half_spectrum_scoring=True, relion_firstiter_score_mode="normalized_cc",
        use_float64_scoring=False,
    )
    phase = grid.score_phase[:n_trans]
    shifted = batch.score_image[:n_images, None, :] * phase[None, :, :]
    lookup = _relion_cuda_fine_full_to_compact_lookup(
        tuple(state.dataset.image_shape), state.current_size, window.window_indices_np,
    )
    fine_scores = _relion_cuda_fine_normalized_cc_score(
        projection[None, :, None, :], shifted[:, None, :, :],
        resident.corr_img_score[:n_images, None, None, :], half_weights, lookup,
    )
    gemm_scores = cc_score_tile(
        projection, batch.score_image[:n_images], batch.score_weight[:n_images], phase,
        jnp.ones((n_images,), bool), jnp.ones((len(unique_rot),), bool),
        jnp.ones((n_trans,), bool), translation_side="image",
    )
    gemm_cross, gemm_power, _, _ = relion_coarse_gemm_terms(
        projection, shifted, batch.score_weight[:n_images], n_images,
        n_images=n_images, n_trans=n_trans, wide=jnp.float32,
    )
    direct_scores = gemm_cross.swapaxes(1, 2) / jnp.sqrt(
        jnp.maximum(gemm_power[:, :, None], jnp.float32(1e-30))
    )
    arrays.update({
        "winner_rotation_ids": winner_rot,
        "unique_rotation_ids": unique_rot,
        "winner_inverse": inverse,
        "fine_scores_same_dense_operands": np.asarray(fine_scores),
        "gemm_scores_same_dense_operands": np.asarray(gemm_scores),
        "gemm_direct_division_same_dense_operands": np.asarray(direct_scores),
        "dense_projection": np.asarray(projection),
        "dense_score_image": np.asarray(batch.score_image[:n_images]),
        "dense_shifted_score": np.asarray(shifted),
        "dense_corr_img": np.asarray(resident.corr_img_score[:n_images]),
        "dense_combined_pixel_weight": np.asarray(batch.score_weight[:n_images]),
        "dense_half_weights": np.asarray(half_weights),
        "dense_phase": np.asarray(phase),
        "dense_full_to_compact": np.asarray(lookup),
    })
    for chunk_id, chunk in enumerate(control_chunks):
        for name, value in chunk.items():
            arrays[f"control_chunk_{chunk_id}_{name}"] = value
    np.savez_compressed(path, **arrays)
    print(f"Dense GEMM matched CC score diagnostic saved: {path}")


@pytest.mark.unit
def test_native_fixture_forwards_preprocessing_operands(monkeypatch):
    from recovar.data_io import image_backends

    seen = []

    def fake_native(images, factors, shifts, radius, edge, masked, **kwargs):
        seen.append((np.asarray(factors), np.asarray(shifts), radius, edge, masked, kwargs))
        return images, images * factors[:, None, None]

    monkeypatch.setattr(image_backends, "relion_cuda_preprocessor", lambda: fake_native)
    dataset = SimpleNamespace(image_shape=(8, 8), image_source=SimpleNamespace())
    _install_native_preprocessing(dataset)
    transformed = dataset.process_images_half(
        np.ones((2, 8, 8), np.float32), apply_image_mask=True,
        relion_normalization_factors=np.asarray([0.5, 1.25], np.float32),
        relion_integer_shifts=np.asarray([[1, -1], [0, 2]], np.int32),
        relion_fft_per_image=True,
    )
    assert transformed.shape == (2, 40)
    assert len(seen) == 1
    assert_matches(seen[0][0], np.asarray([0.5, 1.25], np.float32))
    np.testing.assert_array_equal(seen[0][1], [[1, -1], [0, 2]])
    assert seen[0][2:5] == (3.5, 1.0, True)


@pytest.mark.gpu
def test_k1_full_grid_matches_resident_exact_operands(monkeypatch):
    """Both arms see the same unpruned child grid and RELION CUDA operands."""
    assert jax.default_backend() == "gpu"
    from relax.dense import gemm_coarse_engine
    from relax.fine_pass import resident_pass2

    args = _native_k1_args(monkeypatch)
    from unit.test_resident_relion_reference import F32, _rel_l2
    dataset = args["experiment_dataset"]
    called = []
    diagnostic = {}
    original = gemm_coarse_engine.run_dense_gemm_full_grid
    original_pad_batch = gemm_coarse_engine.pad_batch

    def capture_batch(*args, **kwargs):
        batch = original_pad_batch(*args, **kwargs)
        diagnostic["batch"] = batch
        return batch

    def checked(state):
        assert state.relion_native_fine_units
        assert state.relion_exact_bpref_operands
        assert state.precision_policy.score_real_dtype == jnp.float32
        called.append(True)
        diagnostic["state"] = state
        return original(state)

    monkeypatch.setattr(gemm_coarse_engine, "run_dense_gemm_full_grid", checked)
    monkeypatch.setattr(gemm_coarse_engine, "pad_batch", capture_batch)
    control = resident_pass2.compute_pass2_stats_resident(**args)
    candidate = resident_pass2.compute_pass2_stats_resident(**dict(args, dense_gemm_full_grid=True))
    assert called == [True]
    if path := os.environ.get("RELAX_DENSE_GEMM_DIAGNOSTIC_NPZ"):
        _save_k1_score_diagnostic(path, diagnostic["state"], diagnostic["batch"], control, candidate)
    np.testing.assert_array_equal(candidate.hard_assignment, control.hard_assignment)
    np.testing.assert_array_equal(candidate.best_rotation_indices, control.best_rotation_indices)
    assert_matches(candidate.best_translations, control.best_translations, rtol=F32)
    # Float32 scores accumulate into float64 metadata; use the established
    # native-reference bands, not the float64 storage dtype's default band.
    # Source: tests/unit/test_resident_relion_reference.py::_assert_k1_pass.
    assert_matches(candidate.score_log_z, control.score_log_z, rtol=F32)
    assert_matches(candidate.relion_stats.log_evidence_per_image,
                   control.relion_stats.log_evidence_per_image, rtol=F32)
    assert_matches(candidate.relion_stats.best_log_score_per_image,
                   control.relion_stats.best_log_score_per_image, rtol=F32)
    posterior_atol = (4.0 * float(np.finfo(np.float32).eps)
                      * float(np.max(np.abs(control.score_log_z))))
    np.testing.assert_allclose(
        candidate.relion_stats.max_posterior_per_image,
        control.relion_stats.max_posterior_per_image, rtol=0.0, atol=posterior_atol,
    )
    np.testing.assert_allclose(
        candidate.relion_stats.rotation_posterior_sums,
        control.relion_stats.rotation_posterior_sums,
        rtol=0.0, atol=posterior_atol * dataset.n_units,
    )
    assert _rel_l2(candidate.Ft_y, control.Ft_y) < 1e-5
    assert _rel_l2(candidate.Ft_ctf, control.Ft_ctf) < 1e-5
    _assert_paired_native_noise(candidate, control)
    assert np.isfinite(np.asarray(candidate.score_log_z)).all()
    assert np.linalg.norm(np.asarray(candidate.Ft_y)) > 0
    assert np.linalg.norm(np.asarray(candidate.Ft_ctf)) > 0


@pytest.mark.gpu
def test_k2_square_cc_full_grid_maps_wavg_and_keeps_one_winner(monkeypatch):
    """A square CC score crop retains its complete reconstruction window."""
    assert jax.default_backend() == "gpu"
    from helpers.em_arrays import _hermitian_volume
    from helpers.sparse_pass2_mock import VOLUME_SHAPE
    from relax.relion import ctf as relion_ctf
    from relax.fine_pass import resident_pass2
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "unit"))
    from unit.test_resident_pass2_driver import _driver_fixture_args
    from unit.test_resident_relion_reference import _ppref, _tie_free_noise

    monkeypatch.setenv("RELAX_EM_PROTOTYPE_SOFT_POSTERIOR_BLOCK_BPREF", "1")
    args = _driver_fixture_args(seed=20260929)
    dataset = args.pop("experiment_dataset")
    _install_native_preprocessing(dataset)
    monkeypatch.setattr(relion_ctf, "relion_exact_ctf_half_from_source_star", _variable_ctf_rows)
    args.pop("volume")
    args.pop("significant_sample_indices")
    prior = args.pop("rotation_log_prior")
    args.pop("normalization_other_score_log_z")
    args.pop("normalization_score_mode")
    volumes = jnp.stack([_hermitian_volume(VOLUME_SHAPE, seed=17 + i) for i in range(2)])
    slabs = [_ppref(volumes[i]) for i in range(2)]
    args.update(
        score_with_masked_images=True,
        relion_projector_half=np.stack([slab for slab, _ in slabs]),
        relion_projector_r_max=slabs[0][1],
        noise_variance=jnp.asarray(_tie_free_noise(6, 200.0).reshape(-1), jnp.float32),
        relion_firstiter_score_mode="normalized_cc",
        relion_firstiter_winner_take_all=True,
        relion_exact_fine_normalized_cc=True,
        square_window=True,
        relion_fine_mstep_prune=False,
        dense_gemm_full_grid=True,
    )
    noise = args.pop("noise_variance")
    translations = args.pop("translations")
    output = resident_pass2.compute_k_class_pass2_stats_resident(
        dataset, volumes, noise, translations, [[None] * dataset.n_units for _ in range(2)],
        args.pop("nside_level"), args.pop("disc_type"),
        rotation_log_priors_by_class=[prior + np.float32(np.log(0.5))] * 2,
        **args,
    )
    assert_matches(output.stats.max_posterior_per_image, np.ones(dataset.n_units), rtol=1e-6)
    assert_matches(np.sum(output.class_reconstruction_posterior_sums), dataset.n_units, rtol=1e-6)
    assert all(np.linalg.norm(np.asarray(value)) > 0 for value in output.Ft_ctf)


@pytest.mark.gpu
def test_public_k2_dense_engine_runs_joint_full_grid(monkeypatch):
    """The selected EM entry point reaches the native joint K-class engine."""
    assert jax.default_backend() == "gpu"
    from helpers.em_arrays import _hermitian_volume
    from helpers.sparse_pass2_mock import VOLUME_SHAPE
    from relax.classification.k_class import run_dense_k_class_em_adaptive
    from relax.relion import ctf as relion_ctf
    from relax.refinement.engine_record import take_coarse_engine_calls
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "unit"))
    from unit.test_resident_pass2_driver import _driver_fixture_args
    from unit.test_resident_relion_reference import _ppref, _tie_free_noise

    monkeypatch.setenv("RELAX_EM_PROTOTYPE_SOFT_POSTERIOR_BLOCK_BPREF", "1")
    args = _driver_fixture_args(seed=20260929)
    dataset = args.pop("experiment_dataset")
    _install_native_preprocessing(dataset)
    monkeypatch.setattr(relion_ctf, "relion_exact_ctf_half_from_source_star", _variable_ctf_rows)
    args.pop("volume")
    args.pop("noise_variance")
    args.pop("significant_sample_indices")
    args.pop("normalization_other_score_log_z")
    args.pop("normalization_score_mode")
    args.pop("relion_x_half_mstep")
    args.pop("return_score_log_z")
    args.pop("preserve_bpref_particle_order")
    fine_rotations = args.pop("fine_rotations_override")
    rot_parent = args.pop("fine_rotation_parent_override")
    fine_translations = args.pop("fine_translations_override")
    trans_parent = args.pop("fine_translation_parent_override")
    translations = args.pop("translations")
    noise = jnp.asarray(_tie_free_noise(6, 200.0).reshape(-1), jnp.float32)
    volumes = jnp.stack([_hermitian_volume(VOLUME_SHAPE, seed=17 + i) for i in range(2)])
    slabs = [_ppref(volumes[i]) for i in range(2)]
    args.update(
        score_with_masked_images=True,
        relion_projector_half=np.stack([slab for slab, _ in slabs]),
        relion_projector_r_max=slabs[0][1],
        mstep_relion_x_half=True,
    )
    take_coarse_engine_calls()
    result = run_dense_k_class_em_adaptive(
        dataset, volumes, None, noise,
        fine_rotations[::2], translations, fine_rotations, fine_translations,
        rot_parent, trans_parent, args.pop("disc_type"),
        class_log_priors=np.log(np.array([0.5, 0.5])),
        accumulate_noise=args.pop("accumulate_noise"),
        coarse_healpix_order=args.pop("nside_level"),
        oversampling_order=args.pop("oversampling_order"),
        fine_current_size=args["current_size"],
        coarse_engine="gemm_dense", **args,
    )
    assert np.all(np.isfinite(np.asarray(result.stats.log_evidence_per_image)))
    assert np.sum(np.asarray(result.class_posterior_sums)) == pytest.approx(
        dataset.n_units, rel=1e-6
    )
    assert all(np.linalg.norm(np.asarray(value)) > 0 for value in result.Ft_ctf)
    calls = take_coarse_engine_calls()
    assert len(calls) == 1
    assert calls[0]["resolved"] == "gemm_dense"
    assert calls[0]["evaluated_fine_candidates_total"] == (
        dataset.n_units * 2 * len(fine_rotations) * len(fine_translations)
    )


@pytest.mark.gpu
@pytest.mark.parametrize("class_fixture", ["skew", "balanced"])
@pytest.mark.parametrize("reconstruction_groups", [1, 2])
def test_k4_full_grid_has_one_joint_posterior_and_all_statistics(
    monkeypatch, class_fixture, reconstruction_groups,
):
    assert jax.default_backend() == "gpu"
    from helpers.em_arrays import _hermitian_volume
    from helpers.sparse_pass2_mock import VOLUME_SHAPE
    from relax.relion import ctf as relion_ctf
    from relax.dense import gemm_coarse_engine
    from relax.fine_pass import resident_pass2
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "unit"))
    from unit.test_resident_pass2_driver import _driver_fixture_args
    from unit.test_resident_relion_reference import F32, _ppref, _rel_l2, _tie_free_noise

    monkeypatch.setenv("RELAX_EM_PROTOTYPE_SOFT_POSTERIOR_BLOCK_BPREF", "1")
    monkeypatch.setenv("RELAX_RELION_WAVG_ATOMIC_SCALE_AA", "1")
    monkeypatch.setenv("RELAX_RELION_WAVG_ATOMIC_DIRECT_NOISE_ONLY", "1")
    args = _driver_fixture_args(seed=20260929)
    dataset = args.pop("experiment_dataset")
    _install_native_preprocessing(dataset)
    monkeypatch.setattr(relion_ctf, "relion_exact_ctf_half_from_source_star", _variable_ctf_rows)
    args.pop("volume")
    args.pop("significant_sample_indices")
    base_rotation_prior = args.pop("rotation_log_prior")
    args.pop("normalization_other_score_log_z")
    args.pop("normalization_score_mode")
    if class_fixture == "skew":
        volumes = jnp.stack([
            _hermitian_volume(VOLUME_SHAPE, seed=17 + 13 * class_id)
            for class_id in range(4)
        ])
        prior_masses = (0.4, 0.3, 0.2, 0.1)
    else:
        base = _hermitian_volume(VOLUME_SHAPE, seed=17)
        volumes = jnp.stack([
            base + jnp.float32(1e-4) * _hermitian_volume(VOLUME_SHAPE, seed=43 + class_id)
            for class_id in range(4)
        ])
        prior_masses = (0.25,) * 4
    slabs = [_ppref(volumes[class_id]) for class_id in range(4)]
    priors = [base_rotation_prior + np.float32(np.log(mass)) for mass in prior_masses]
    dvp = np.stack([np.asarray(args["scale_correction_data_vs_prior"]).copy() for _ in range(4)])
    for class_id in range(4):
        dvp[class_id, dvp.shape[1] - 1 - class_id:] = 1.0
    args.update(
        score_with_masked_images=True,
        source_faithful_spectrum_norm=False,
        relion_projector_half=np.stack([slab for slab, _ in slabs]),
        relion_projector_r_max=slabs[0][1],
        noise_variance=jnp.asarray(_tie_free_noise(6, 200.0).reshape(-1), jnp.float32),
        scale_correction_data_vs_prior=dvp,
        adaptive_fraction=1.0,
        relion_fine_mstep_prune=False,
    )
    if reconstruction_groups == 2:
        args.update(
            reconstruction_group_ids=np.arange(dataset.n_units, dtype=np.int32) % 2,
            reconstruction_group_count=2,
            mstep_subtract_ctf_projection=True,
        )
    supports = [[None] * dataset.n_units for _ in range(4)]

    def call(dense):
        options = dict(args, dense_gemm_full_grid=dense)
        noise = options.pop("noise_variance")
        translations = options.pop("translations")
        return resident_pass2.compute_k_class_pass2_stats_resident(
            dataset, volumes, noise, translations, supports,
            options.pop("nside_level"), options.pop("disc_type"),
            rotation_log_priors_by_class=priors, **options,
        )

    diagnostic_path = os.environ.get("RELAX_DENSE_GEMM_K4_DIAGNOSTIC_NPZ")
    if diagnostic_path and class_fixture == "balanced":
        diagnostic_file = Path(diagnostic_path)
        diagnostic_path = str(diagnostic_file.with_name(diagnostic_file.stem + "_balanced.npz"))
    resident_mass_parts = []
    dense_batches = []
    control_norm_parts = []
    candidate_norm_parts = []
    if diagnostic_path:
        from relax.fine_pass.resident_statistics import segment_sum_by_image

        original_accumulate = resident_pass2._accumulate_chunk_image_terms

        def capture_resident_mass(stats, operands, tables, *, config):
            translation_mass = segment_sum_by_image(
                operands.row_posterior, operands.row_image_local,
                int(operands.image_ids.shape[0]),
                float32_posterior_bucket_size=(
                    128 if operands.row_posterior.dtype == jnp.float32 else None
                ),
            )
            mass = jnp.sum(translation_mass, axis=1)
            jax.debug.callback(
                lambda ids, values, posterior, row_image: resident_mass_parts.append(
                    (np.asarray(ids), np.asarray(values), np.asarray(posterior), np.asarray(row_image))
                ),
                operands.image_ids, mass, operands.row_posterior, operands.row_image_local,
            )
            return _capture_chunk_norm(
                control_norm_parts, original_accumulate, stats, operands, tables, config=config,
            )

        monkeypatch.setattr(resident_pass2, "_accumulate_chunk_image_terms", capture_resident_mass)
    # The dense GEMM engine sums its float32 image rows in buckets; its control is the
    # resident pass with the same bucketed sums (the resident default is the scatter-add).
    monkeypatch.setattr(resident_pass2, "_FLOAT32_BUCKETED_IMAGE_SUMS", True)
    control = call(False)
    monkeypatch.setattr(resident_pass2, "_FLOAT32_BUCKETED_IMAGE_SUMS", False)
    if diagnostic_path:
        monkeypatch.setattr(resident_pass2, "_accumulate_chunk_image_terms", original_accumulate)
        from relax.dense import gemm_coarse_statistics

        original_dense_accumulate = gemm_coarse_statistics._accumulate_chunk_image_terms

        def capture_dense_norm(stats, operands, tables, *, config):
            return _capture_chunk_norm(
                candidate_norm_parts, original_dense_accumulate, stats, operands, tables, config=config,
            )

        monkeypatch.setattr(gemm_coarse_statistics, "_accumulate_chunk_image_terms", capture_dense_norm)
        original_make_program = gemm_coarse_engine.make_joint_k_batch_program

        def capture_dense_program(*args, **kwargs):
            program = original_make_program(*args, **kwargs)

            def run(*program_args, **program_kwargs):
                result = program(*program_args, **program_kwargs)
                dense_batches.append(result)
                return result

            return run

        monkeypatch.setattr(gemm_coarse_engine, "make_joint_k_batch_program", capture_dense_program)
    candidate = call(True)
    if diagnostic_path:
        _save_k4_output_diagnostic(
            diagnostic_path, control, candidate,
            resident_mass_parts=resident_mass_parts, dense_batches=dense_batches,
            control_norm_parts=control_norm_parts, candidate_norm_parts=candidate_norm_parts,
        )
    np.testing.assert_array_equal(candidate.per_class_hard_assignments, control.per_class_hard_assignments)
    # Class and joint posterior categories follow the existing Class3D native
    # reference gate in tests/unit/test_resident_relion_reference.py.
    assert_matches(candidate.class_log_evidence_per_image,
                   control.class_log_evidence_per_image, rtol=F32)
    assert_matches(candidate.class_best_log_score_per_image,
                   control.class_best_log_score_per_image, rtol=F32)
    assert_matches(candidate.stats.log_evidence_per_image,
                   control.stats.log_evidence_per_image, rtol=F32)
    assert_matches(candidate.stats.best_log_score_per_image,
                   control.stats.best_log_score_per_image, rtol=F32)
    posterior_atol = (4.0 * float(np.finfo(np.float32).eps)
                      * float(np.max(np.abs(control.stats.log_evidence_per_image))))
    np.testing.assert_allclose(candidate.stats.max_posterior_per_image,
                               control.stats.max_posterior_per_image, rtol=0.0,
                               atol=posterior_atol)
    np.testing.assert_allclose(candidate.class_rotation_posterior_sums,
                               control.class_rotation_posterior_sums,
                               rtol=0.0, atol=posterior_atol * dataset.n_units)
    np.testing.assert_allclose(candidate.stats.rotation_posterior_sums,
                               control.stats.rotation_posterior_sums,
                               rtol=0.0, atol=posterior_atol * dataset.n_units)
    assert_matches(candidate.class_reconstruction_posterior_sums,
                   control.class_reconstruction_posterior_sums, rtol=F32)
    _assert_paired_native_noise(candidate, control)
    for arm in ("Ft_y", "Ft_ctf"):
        for got, want in zip(getattr(candidate, arm), getattr(control, arm)):
            assert _rel_l2(got, want) < 1e-5
    masses = np.asarray(candidate.class_reconstruction_posterior_sums)
    assert_matches(np.sum(masses), dataset.n_units, rtol=F32)
    if class_fixture == "skew":
        assert masses[2] > 0.9 * dataset.n_units
    else:
        assert np.all(masses > 0.05 * dataset.n_units)


@pytest.mark.gpu
def test_k1_firstiter_cc_keeps_canonical_winner_and_statistics(monkeypatch):
    """The native first-iteration CC scorer selects one full-grid pose."""
    assert jax.default_backend() == "gpu"
    from relax.fine_pass import resident_pass2

    args = _native_k1_args(monkeypatch, seed=20260930)
    from unit.test_resident_relion_reference import F32, _rel_l2

    args.update(
        relion_firstiter_score_mode="normalized_cc",
        relion_firstiter_winner_take_all=True,
        relion_exact_fine_normalized_cc=True,
        normalization_score_mode="normalized_cc",
    )
    diagnostic_path = os.environ.get("RELAX_DENSE_GEMM_CC_DIAGNOSTIC_NPZ")
    control_chunks = []
    dense_capture = {}
    if diagnostic_path:
        from relax.dense import gemm_coarse_engine
        from relax.fine_pass import resident_operands, resident_scoring

        original_cc_scorer = resident_scoring.score_resident_chunk_normalized_cc

        def capture_control_cc(*score_args, **score_kwargs):
            result = original_cc_scorer(*score_args, **score_kwargs)

            def retain(row_image, row_rotation, n_valid, projection, shifted, weight,
                       scores, half_weights, full_to_compact):
                control_chunks.append(dict(
                    row_image_local=np.asarray(row_image), row_fine_rot=np.asarray(row_rotation),
                    n_valid_rows=np.asarray(n_valid), projection_score_cache=np.asarray(projection),
                    score_shifted_cc=np.asarray(shifted), cc_score_weight=np.asarray(weight),
                    scores=np.asarray(scores), half_weights=np.asarray(half_weights),
                    full_to_compact=np.asarray(full_to_compact),
                ))

            jax.debug.callback(
                retain, score_args[0], score_args[1], score_args[4], score_args[5],
                score_args[6], score_args[7], result.scores,
                score_kwargs["half_weights"], score_kwargs["full_to_compact"],
            )
            return result

        monkeypatch.setattr(resident_scoring, "score_resident_chunk_normalized_cc", capture_control_cc)
    control = resident_pass2.compute_pass2_stats_resident(**args)
    if diagnostic_path:
        monkeypatch.setattr(resident_scoring, "score_resident_chunk_normalized_cc", original_cc_scorer)
        original_dense = gemm_coarse_engine.run_dense_gemm_full_grid
        original_pad_batch = gemm_coarse_engine.pad_batch
        original_pad_grid = gemm_coarse_engine.pad_grid
        original_resident_operands = resident_operands.prepare_resident_half_operands

        def capture_dense_state(state):
            dense_capture["state"] = state
            return original_dense(state)

        def capture_batch(*batch_args, **batch_kwargs):
            dense_capture["batch"] = original_pad_batch(*batch_args, **batch_kwargs)
            return dense_capture["batch"]

        def capture_grid(*grid_args, **grid_kwargs):
            dense_capture["grid"] = original_pad_grid(*grid_args, **grid_kwargs)
            return dense_capture["grid"]

        def capture_operands(*operand_args, **operand_kwargs):
            dense_capture["resident"] = original_resident_operands(*operand_args, **operand_kwargs)
            return dense_capture["resident"]

        monkeypatch.setattr(gemm_coarse_engine, "run_dense_gemm_full_grid", capture_dense_state)
        monkeypatch.setattr(gemm_coarse_engine, "pad_batch", capture_batch)
        monkeypatch.setattr(gemm_coarse_engine, "pad_grid", capture_grid)
        monkeypatch.setattr(resident_operands, "prepare_resident_half_operands", capture_operands)
    candidate = resident_pass2.compute_pass2_stats_resident(**dict(args, dense_gemm_full_grid=True))
    if diagnostic_path:
        _save_k1_cc_score_diagnostic(
            diagnostic_path, dense_capture["state"], dense_capture["batch"],
            dense_capture["grid"], dense_capture["resident"], control_chunks,
            control, candidate,
        )
    np.testing.assert_array_equal(candidate.hard_assignment, control.hard_assignment)
    np.testing.assert_array_equal(candidate.best_rotation_indices, control.best_rotation_indices)
    assert_matches(candidate.best_translations, control.best_translations, rtol=F32)
    assert_matches(candidate.relion_stats.max_posterior_per_image,
                   np.ones(args["experiment_dataset"].n_units), rtol=F32)
    assert_matches(candidate.relion_stats.log_evidence_per_image,
                   control.relion_stats.log_evidence_per_image, rtol=F32)
    assert_matches(candidate.relion_stats.best_log_score_per_image,
                   control.relion_stats.best_log_score_per_image, rtol=F32)
    assert _rel_l2(candidate.Ft_y, control.Ft_y) < 1e-5
    assert _rel_l2(candidate.Ft_ctf, control.Ft_ctf) < 1e-5
    _assert_paired_native_noise(candidate, control)


@pytest.mark.gpu
def test_k1_gradient_grouped_bpref_with_distinct_windows_and_translation_tail(monkeypatch):
    """VDAM residuals retain both pseudo-halfsets with a smaller model window."""
    assert jax.default_backend() == "gpu"
    # Import the dense statistics owner before instrumenting the resident
    # function; otherwise its first import can bind the temporary control
    # wrapper and record the dense rows twice.
    from relax.dense import gemm_coarse_statistics
    from relax.fine_pass import resident_pass2

    args = _native_k1_args(monkeypatch, seed=20261001)
    from unit.test_resident_relion_reference import F32, _rel_l2

    dataset = args["experiment_dataset"]
    n_images = dataset.n_units
    translations = np.asarray(args["fine_translations_override"], np.float32)
    parents = np.asarray(args["fine_translation_parent_override"], np.int32)
    args.update(
        mstep_subtract_ctf_projection=True,
        current_size=8,
        reconstruction_current_size=6,
        reconstruction_group_ids=np.arange(n_images, dtype=np.int32) % 2,
        reconstruction_group_count=2,
        # T=9 with U=8 forces a real padded translation tile in the native pass.
        fine_translations_override=np.concatenate((translations, np.asarray([[0.25, -0.25]], np.float32))),
        fine_translation_parent_override=np.concatenate((parents, [0])),
    )
    assert len(args["fine_translations_override"]) == 9
    args.pop("window_at_box", None)
    diagnostic_path = os.environ.get("RELAX_DENSE_GEMM_GRADIENT_DIAGNOSTIC_NPZ")
    row_diagnostic_path = os.environ.get("RELAX_DENSE_GEMM_GRADIENT_ROW_DIAGNOSTIC_NPZ")
    control_norm_parts = []
    candidate_norm_parts = []
    control_rows = []
    candidate_rows = []
    if diagnostic_path or row_diagnostic_path:
        original_control_accumulate = resident_pass2._accumulate_chunk_image_terms

        def capture_control_norm(stats, operands, tables, *, config):
            return _capture_chunk_norm(
                control_norm_parts, original_control_accumulate, stats, operands, tables,
                config=config,
            )

        monkeypatch.setattr(resident_pass2, "_accumulate_chunk_image_terms", capture_control_norm)
    if row_diagnostic_path:
        original_control_block = resident_pass2._resident_block_noise_and_norm

        def capture_control_rows(*block_args, **block_kwargs):
            return _capture_pre_scatter_norm(
                control_rows, original_control_block, *block_args, **block_kwargs,
            )

        monkeypatch.setattr(resident_pass2, "_resident_block_noise_and_norm", capture_control_rows)
        # A preceding fixture may have compiled the same row program without
        # diagnostics. The opt-in probe must be present in the traced call.
        jax.clear_caches()
    control = resident_pass2._resident_pass2(**args)
    if row_diagnostic_path:
        jax.block_until_ready(control.noise_stats.wsum_norm_correction)
        jax.effects_barrier()
    if diagnostic_path or row_diagnostic_path:
        monkeypatch.setattr(resident_pass2, "_accumulate_chunk_image_terms", original_control_accumulate)
        original_dense_accumulate = gemm_coarse_statistics._accumulate_chunk_image_terms

        def capture_dense_norm(stats, operands, tables, *, config):
            return _capture_chunk_norm(
                candidate_norm_parts, original_dense_accumulate, stats, operands, tables,
                config=config,
            )

        monkeypatch.setattr(gemm_coarse_statistics, "_accumulate_chunk_image_terms", capture_dense_norm)
    if row_diagnostic_path:
        monkeypatch.setattr(resident_pass2, "_resident_block_noise_and_norm", original_control_block)
        original_dense_block = gemm_coarse_statistics._resident_block_noise_and_norm

        def capture_dense_rows(*block_args, **block_kwargs):
            return _capture_pre_scatter_norm(
                candidate_rows, original_dense_block, *block_args, **block_kwargs,
            )

        monkeypatch.setattr(gemm_coarse_statistics, "_resident_block_noise_and_norm", capture_dense_rows)
        jax.clear_caches()
    candidate = resident_pass2._resident_pass2(**dict(args, dense_gemm_full_grid=True))
    if path := diagnostic_path or (str(Path(row_diagnostic_path).with_name("gradient.npz"))
                                   if row_diagnostic_path else None):
        arrays = {}
        for label, result in (("control", control), ("candidate", candidate)):
            for group in range(2):
                arrays[f"{label}_Ft_y_group{group}"] = np.asarray(result.Ft_y[group])
                arrays[f"{label}_Ft_ctf_group{group}"] = np.asarray(result.Ft_ctf[group])
            for field in result.finalized._fields:
                value = getattr(result.finalized, field)
                if value is not None:
                    arrays[f"{label}_stats_{field}"] = np.asarray(value)
        for label, parts in (("control", control_norm_parts), ("candidate", candidate_norm_parts)):
            for batch_id, part in enumerate(parts):
                for field, value in part.items():
                    arrays[f"{label}_norm_batch_{batch_id}_{field}"] = np.asarray(value)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
    if row_diagnostic_path:
        jax.block_until_ready(candidate.noise_stats.wsum_norm_correction)
        jax.effects_barrier()
        _save_gradient_row_diagnostic(
            row_diagnostic_path, control_rows, candidate_rows,
            control_norm_parts, candidate_norm_parts,
        )
    assert candidate.n_slot_groups == control.n_slot_groups == 2
    np.testing.assert_array_equal(candidate.finalized.hard_assignment, control.finalized.hard_assignment)
    np.testing.assert_array_equal(candidate.finalized.best_fine_rotation_indices,
                                  control.finalized.best_fine_rotation_indices)
    assert_matches(candidate.finalized.log_evidence_per_image,
                   control.finalized.log_evidence_per_image, rtol=F32)
    for group in range(2):
        assert _rel_l2(candidate.Ft_y[group], control.Ft_y[group]) < 1e-5
        assert _rel_l2(candidate.Ft_ctf[group], control.Ft_ctf[group]) < 1e-5
        assert np.linalg.norm(np.asarray(candidate.Ft_ctf[group])) > 0
    _assert_paired_native_noise(candidate, control)
