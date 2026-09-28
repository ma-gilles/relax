"""Run one bounded, synthetic-operand K1 dense GEMM/CUDA experiment.

This harness exercises the real RELION CUDA projector, translator, direct
coarse scorer and fused x-half backprojector.  Its generated Fourier operands
are diagnostic; it does not claim an end-to-end RELION trajectory or FSC gate.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import re
import statistics
import subprocess
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from relax.cuda.kernels import (
    RelionCapacityHalfTextureF32,
    relion_coarse_diff2_projector_f32,
    relion_coarse_diff2_rectangular_f32,
    relion_fused_x_half_backproject_particle_grid_indexed,
    relion_translate_score_f32,
    relion_translate_sum_flat_rows_f32,
)
from relax.dense.gemm_experiment import (
    DenseGemmBatchResult,
    DenseGemmTileConfig,
    leading_tile_bytes,
    make_batch_program,
    native_phase_table,
    native_relion_callbacks,
    pad_batch,
    pad_grid,
)
from relax.dense.gemm_experiment_kernels import (
    empty_normalizer_table,
    merge_normalizers,
    normalizer_logz,
    score_model_power,
    score_tile,
    tile_normalizer,
    weighted_slices,
)
from relax.helpers.projection import relion_projector_half_to_texture_full
from relax.relion.relion_projector_setup import setup_relion_projector


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--engine", choices=("gemm", "production", "native"), default="gemm")
    parser.add_argument("--mode", choices=("exact", "lagged"), default="exact")
    parser.add_argument("--translation-side", choices=("image", "projection"), required=True)
    parser.add_argument("--images", type=int, default=32)
    parser.add_argument("--rotations", type=int, default=8)
    parser.add_argument("--translations", type=int, default=9)
    parser.add_argument("--rotation-tile", type=int, default=4)
    parser.add_argument("--translation-tile", type=int, default=3)
    parser.add_argument("--box", type=int, default=8)
    parser.add_argument("--padding-factor", type=int, choices=(1, 2), default=2)
    parser.add_argument("--seed", type=int, default=529)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timing-style", choices=("device-reset", "init-inclusive"), default="device-reset")
    parser.add_argument("--profile-warm-repeats", action="store_true", help="CUDA profiler start/stop around warmed operator repeats")
    parser.add_argument("--compare-direct", action="store_true", help="small-case full-grid diagnostics outside timing")
    parser.add_argument("--diagnose-scores", action="store_true", help="matched float32 score and float64 normalization diagnostics")
    parser.add_argument("--save-arrays", action="store_true", help="save final logZ, mass and volumes outside timing")
    parser.add_argument("--tile-budget-gb", type=float, default=1.0)
    parser.add_argument("--no-donation", action="store_true", help="diagnose outer JIT buffer aliasing")
    parser.add_argument("--adjoint-backend", choices=("fused", "separate"), default="fused")
    return parser.parse_args()


def _rotations(count, seed):
    """Representative SO(3) poses with distinct canonical score/M-step matrices."""
    from relax.sampling import _relion_device_scoring_rotations_f32, _relion_mstep_rotations_from_eulers

    rng = np.random.default_rng(seed + 19)
    eulers = np.column_stack(
        (
            rng.uniform(0, 360, count),
            np.rad2deg(np.arccos(rng.uniform(-1, 1, count))),
            rng.uniform(0, 360, count),
        )
    ).astype(np.float32)
    score = _relion_device_scoring_rotations_f32(eulers)
    if score is None:
        raise RuntimeError("SO(3) benchmark requires the native CUDA RELION scoring rotations")
    return score, _relion_mstep_rotations_from_eulers(eulers)


def _source_identity(root):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root).decode().strip()

    paths = (
        "relax/dense/gemm_experiment.py",
        "relax/dense/gemm_experiment_kernels.py",
        "scripts/benchmark_dense_gemm_em.py",
    )
    return {
        "head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "tracked_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=root)).hexdigest(),
        "untracked_source_sha256": {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths},
    }


def _native_reconstruction_indices(image_size, radius):
    """One unique 2*rmax FFTW square; omit the redundant negative-radius row."""
    rows = [*range(radius + 1), *range(image_size - radius + 1, image_size)]
    indices = np.asarray(
        [row * (image_size // 2 + 1) + col for row in rows for col in range(radius + 1)],
        dtype=np.int32,
    )
    native_rows = (indices // (image_size // 2 + 1)) % (2 * radius)
    native_cols = indices % (image_size // 2 + 1)
    destinations = native_rows * (radius + 1) + native_cols
    if len(indices) != 2 * radius * (radius + 1) or len(np.unique(destinations)) != len(indices):
        raise ValueError("reconstruction indices must map one-to-one onto the native FFTW square")
    return jnp.asarray(indices, jnp.int32)


def _centered_reconstruction_indices(native_indices, image_size):
    """Map an FFTW BPref row order to centered packed-half translation rows."""
    half_width = image_size // 2 + 1
    native_indices = jnp.asarray(native_indices, jnp.int32)
    rows = native_indices // half_width
    return ((rows + image_size // 2) % image_size) * half_width + native_indices % half_width


def _native_identity():
    # The paths are selected explicitly by the caller; these reads also make
    # accidental loader auto-builds visible in the report.
    import os

    from recovar import cuda_backproject

    from relax.cuda import kernels

    paths = {"recovar": os.environ.get("RECOVAR_CUDA_LIB"), "relax": os.environ.get("RELAX_CUDA_LIB")}
    if any(path is None for path in paths.values()):
        raise RuntimeError("set both RECOVAR_CUDA_LIB and RELAX_CUDA_LIB to exclusive built libraries")
    if not cuda_backproject.cuda_available():
        raise RuntimeError("pinned RECOVAR CUDA library did not load")
    if not kernels.custom_cuda_requested():
        raise RuntimeError("pinned RELAX CUDA library was disabled")
    return {
        name: {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        for name, path in paths.items()
    }


def _make_case(args):
    rng = np.random.default_rng(args.seed)
    n = args.box
    padding_factor = getattr(args, "padding_factor", 1)
    r_max = n // 4
    if n < 8 or n % 2 or r_max < 2:
        raise ValueError("box must be an even integer of at least 8")
    if args.translations > 128:
        raise ValueError("native direct coarse scorer supports at most 128 translations")
    from relax.helpers.half_volume_mstep import relion_backprojector_volume_shape

    volume_shape = relion_backprojector_volume_shape((n, n, n), padding_factor, current_size=2 * r_max)
    image_shape = (n, n)
    pixels = n * (n // 2 + 1)
    score_indices = jnp.arange(pixels, dtype=jnp.int32)
    # The native 2*r_max FFTW square must receive unique compact pixels.
    # Passing all pixels of the larger image folds multiple rows into the
    # same scatter destination, so its winner depends on XLA buffer layout.
    rec_indices = _native_reconstruction_indices(n, r_max)
    rec_centered_indices = _centered_reconstruction_indices(rec_indices, n)
    real_reference = np.zeros((n, n, n), dtype=np.float32)
    real_reference[n // 2 - 1 : n // 2 + 2, n // 2 - 1 : n // 2 + 2, n // 2 - 1 : n // 2 + 2] = rng.uniform(
        0.1, 0.9, (3, 3, 3)
    )
    projector, _ = setup_relion_projector(
        jnp.asarray(real_reference),
        jnp.int32(r_max),
        ori_size=n,
        padding_factor=padding_factor,
        do_gridding=False,
        compute_dtype=jnp.float32,
    )
    # The CUDA texture path consumes the logical r_max slab including one
    # ghost texel on each side, not setup's full Nyquist-capacity slab.
    logical_size = 2 * (padding_factor * r_max + 1) + 1
    slab_start = (projector.shape[0] - logical_size) // 2
    projector = projector[
        slab_start : slab_start + logical_size,
        slab_start : slab_start + logical_size,
        : logical_size // 2 + 1,
    ]
    score_rotations, bp_rotations = _rotations(args.rotations, args.seed)
    score_image = (rng.normal(0, 0.06, (args.images, pixels)) + 1j * rng.normal(0, 0.06, (args.images, pixels))).astype(
        np.complex64
    )
    # Preserve the pre-fix RNG stream for score operands while dropping the
    # invalid duplicate -r reconstruction row from the historical harness.
    prior_rec_pixels = (2 * r_max + 1) * (r_max + 1)
    keep_rec = np.r_[0 : (r_max + 1) ** 2, (r_max + 2) * (r_max + 1) : prior_rec_pixels]
    prior_rec_image = (
        rng.normal(0, 0.04, (args.images, prior_rec_pixels))
        + 1j * rng.normal(0, 0.04, (args.images, prior_rec_pixels))
    ).astype(np.complex64)
    rec_image = prior_rec_image[:, keep_rec]
    score_weight = rng.uniform(0.4, 2.0, (args.images, pixels)).astype(np.float32)
    score_weight[:, ::11] = 0  # zero CTF positions
    prior_rec_weight = rng.uniform(0.2, 1.5, (args.images, prior_rec_pixels)).astype(np.float32)
    rec_weight = prior_rec_weight[:, keep_rec]
    initial_diff2 = rng.uniform(0.3, 1.1, args.images).astype(np.float32)
    translation_angles = rng.uniform(-0.2, 0.2, (args.translations, 2)).astype(np.float32)
    translation_angles[0] = 0
    score_phase = native_phase_table(jnp.asarray(translation_angles), score_indices, image_shape)
    rec_phase = native_phase_table(jnp.asarray(translation_angles), rec_centered_indices, image_shape)
    grid = pad_grid(
        score_rotations,
        bp_rotations,
        score_phase,
        rec_phase,
        rotation_tile=args.rotation_tile,
        translation_tile=args.translation_tile,
    )
    rotation_prior = rng.uniform(-0.1, 0.1, (args.images, args.rotations)).astype(np.float32)
    translation_prior = rng.uniform(-0.1, 0.1, (args.images, args.translations)).astype(np.float32)
    batch = pad_batch(
        score_image,
        score_weight,
        initial_diff2,
        rec_image,
        rec_weight,
        rotation_prior,
        translation_prior,
        np.arange(args.images, dtype=np.int32),
        image_capacity=args.images,
        grid=grid,
        sentinel_id=args.images,
    )
    project, backproject = native_relion_callbacks(
        image_shape=image_shape,
        score_indices=score_indices,
        rec_indices=rec_indices,
        r_max=r_max,
        projector_output_size=n,
        volume_shape=volume_shape,
        backprojection_backend=getattr(args, "adjoint_backend", "fused"),
        padding_factor=padding_factor,
    )
    volume_size = volume_shape[0] * volume_shape[1] * (volume_shape[2] // 2 + 1)
    return (
        projector,
        batch,
        grid,
        project,
        backproject,
        score_indices,
        rec_indices,
        jnp.asarray(translation_angles),
        volume_size,
    )


def _run_once(program, case, old_table):
    reference, batch, grid, _, _, _, _, _, volume_size = case
    if old_table.ndim == 1:
        old_table = jnp.stack((old_table, jnp.zeros_like(old_table)), axis=1)
    start = time.perf_counter()
    result = program(
        reference,
        jnp.zeros((volume_size,), jnp.complex64),
        jnp.zeros((volume_size,), jnp.float32),
        old_table,
        empty_normalizer_table(batch.particle_ids.shape[0] + 1),
        batch,
        grid,
    )
    jax.block_until_ready(result.numerator)
    jax.block_until_ready(result.denominator)
    elapsed = time.perf_counter() - start
    return result, elapsed


def _reusable_program(program):
    """Reset donated scratch on device before the resident operator executes."""
    @partial(jax.jit, donate_argnums=(1, 2, 4))
    def reset(reference, scratch_y, scratch_w, old_table, scratch_next, batch, grid):
        # A pure zeros_like lets XLA drop the donated arguments, so there is
        # nothing to alias.  Scratch arrays hold the prior iteration's finite
        # outputs; self-subtraction writes exact zero into reusable
        # storage and keeps the donation live in the compiled executable.
        zero_y = scratch_y - scratch_y
        zero_w = scratch_w - scratch_w
        return program(
            reference,
            zero_y,
            zero_w,
            old_table,
            jnp.full_like(scratch_next, -jnp.inf),
            batch,
            grid,
        )

    return reset


def _run_reusable(program, case, old_table, scratch):
    reference, batch, grid, _, _, _, _, _, _ = case
    start = time.perf_counter()
    result = program(reference, *scratch[:2], old_table, scratch[2], batch, grid)
    jax.block_until_ready(result.numerator)
    jax.block_until_ready(result.denominator)
    elapsed = time.perf_counter() - start
    return result, elapsed, (result.numerator, result.denominator, result.next_pair_table)


def _memory_stats(device):
    stats = device.memory_stats()
    return {key: int(value) for key, value in stats.items() if isinstance(value, (int, np.integer))} if stats else None


def _compiled_memory(compiled):
    analysis = compiled.memory_analysis()
    if analysis is None:
        return None
    sizes = {
        key: int(getattr(analysis, key))
        for key in ("temp_size_in_bytes", "argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes")
    }
    sizes["estimated_live_bytes"] = (
        sizes["temp_size_in_bytes"]
        + sizes["argument_size_in_bytes"]
        + sizes["output_size_in_bytes"]
        - sizes["alias_size_in_bytes"]
    )
    return sizes


def _volume_aliases(hlo, compiled, volume_size):
    """Verify that both donated volume outputs alias distinct live inputs."""
    first_line = hlo.splitlines()[0]
    section = re.search(r"input_output_alias=\{(.*?)\}, entry_computation_layout=", first_line)
    entries = {} if section is None else {
        int(output): int(argument)
        for output, argument in re.findall(
            r"\{([01])\}: \((\d+), \{\}, (?:may|must)-alias\)", section.group(1)
        )
    }
    expected_bytes = int(volume_size) * (np.dtype(np.complex64).itemsize + np.dtype(np.float32).itemsize)
    analysis = _compiled_memory(compiled)
    valid = (
        set(entries) == {0, 1}
        and entries[0] != entries[1]
        and analysis is not None
        and analysis["alias_size_in_bytes"] >= expected_bytes
    )
    return {"verified": valid, "output_to_input": entries, "expected_volume_bytes": expected_bytes}


def _profiler_boundary(start):
    cudart = ctypes.CDLL("libcudart.so")
    action = cudart.cudaProfilerStart if start else cudart.cudaProfilerStop
    status = action()
    if status != 0:
        raise RuntimeError(f"CUDA profiler {'start' if start else 'stop'} failed with code {status}")


def _direct_comparison(args, case, result, old_logz_table):
    reference, batch, grid, project, backproject, score_indices, _, angles, volume_size = case
    projected = project(reference, grid.score_rotations[: args.rotations])
    shifted = relion_translate_score_f32(batch.score_image, angles, score_indices, (args.box, args.box)).reshape(
        args.images, args.translations, -1
    )
    direct_diff2 = relion_coarse_diff2_rectangular_f32(
        projected, shifted, batch.score_weight, batch.initial_diff2, score_indices
    )
    direct_scores = (
        -direct_diff2
        + batch.rotation_prior[:, : args.rotations, None]
        + batch.translation_prior[:, None, : args.translations]
    )
    gemm_scores = score_tile(
        projected,
        batch.score_image,
        batch.score_weight,
        batch.initial_diff2,
        grid.score_phase[: args.translations],
        batch.rotation_prior[:, : args.rotations],
        batch.translation_prior[:, : args.translations],
        batch.valid_images,
        grid.valid_rotations[: args.rotations],
        grid.valid_translations[: args.translations],
        translation_side=args.translation_side,
    )
    direct_pair = tile_normalizer(direct_scores)
    direct_logz = normalizer_logz(direct_pair)
    if old_logz_table.ndim == 1:
        old_logz_table = jnp.stack((old_logz_table, jnp.zeros_like(old_logz_table)), axis=1)
    direct_normalizer = direct_pair if args.mode == "exact" else old_logz_table[batch.particle_ids]
    direct_q = jnp.exp((direct_scores - direct_normalizer[:, 0, None, None]) - direct_normalizer[:, 1, None, None])
    direct_y, direct_w = weighted_slices(
        direct_q,
        batch.rec_image,
        batch.rec_weight,
        grid.rec_phase[: args.translations],
        translation_side="image",
    )
    tiled_scores = []
    for rot_start in range(0, args.rotations, args.rotation_tile):
        projected_tile = project(reference, grid.score_rotations[rot_start : rot_start + args.rotation_tile])
        rotation_scores = []
        for trans_start in range(0, args.translations, args.translation_tile):
            scores_tile = score_tile(
                projected_tile,
                batch.score_image,
                batch.score_weight,
                batch.initial_diff2,
                grid.score_phase[trans_start : trans_start + args.translation_tile],
                batch.rotation_prior[:, rot_start : rot_start + args.rotation_tile],
                batch.translation_prior[:, trans_start : trans_start + args.translation_tile],
                batch.valid_images,
                grid.valid_rotations[rot_start : rot_start + args.rotation_tile],
                grid.valid_translations[trans_start : trans_start + args.translation_tile],
                translation_side=args.translation_side,
            )
            rotation_scores.append(scores_tile[:, :, : min(args.translation_tile, args.translations - trans_start)])
        tiled_scores.append(
            jnp.concatenate(rotation_scores, axis=2)[:, : min(args.rotation_tile, args.rotations - rot_start)]
        )
    tiled_scores = jnp.concatenate(tiled_scores, axis=1)
    tiled_normalizer = result.next_pair_table[batch.particle_ids] if args.mode == "exact" else old_logz_table[batch.particle_ids]
    tiled_q = jnp.exp((tiled_scores - tiled_normalizer[:, 0, None, None]) - tiled_normalizer[:, 1, None, None])
    tiled_y, tiled_w = weighted_slices(
        tiled_q,
        batch.rec_image,
        batch.rec_weight,
        grid.rec_phase[: args.translations],
        translation_side="image",
    )

    def sum_slices(y, w, ys, ws, _rotations):
        return y + jnp.sum(ys, axis=0), w + jnp.sum(ws, axis=0)

    slice_probe = make_batch_program(
        DenseGemmTileConfig(args.images, args.rotation_tile, args.translation_tile, args.translation_side, args.mode),
        project,
        sum_slices,
    )
    probed = slice_probe(
        reference,
        jnp.zeros((batch.rec_image.shape[1],), jnp.complex64),
        jnp.zeros((batch.rec_image.shape[1],), jnp.float32),
        old_logz_table,
        empty_normalizer_table(args.images + 1),
        batch,
        grid,
    )

    def capture_slices(y, w, ys, ws, rotations):
        start = rotations[0, 0, 0].astype(jnp.int32)
        return (
            jax.lax.dynamic_update_slice_in_dim(y, ys, start, axis=0),
            jax.lax.dynamic_update_slice_in_dim(w, ws, start, axis=0),
        )

    encoded_rotations = jnp.zeros_like(grid.backprojection_rotations)
    encoded_rotations = encoded_rotations.at[:, 0, 0].set(jnp.arange(grid.backprojection_rotations.shape[0]))
    capture_grid = grid._replace(backprojection_rotations=encoded_rotations)
    capture_probe = make_batch_program(
        DenseGemmTileConfig(args.images, args.rotation_tile, args.translation_tile, args.translation_side, args.mode),
        project,
        capture_slices,
    )
    captured = capture_probe(
        reference,
        jnp.zeros((grid.score_rotations.shape[0], batch.rec_image.shape[1]), jnp.complex64),
        jnp.zeros((grid.score_rotations.shape[0], batch.rec_image.shape[1]), jnp.float32),
        old_logz_table,
        empty_normalizer_table(args.images + 1),
        batch,
        capture_grid,
    )
    direct_y_volume = jnp.zeros((volume_size,), jnp.complex64)
    direct_w_volume = jnp.zeros((volume_size,), jnp.float32)
    for start in range(0, args.rotations, args.rotation_tile):
        stop = min(start + args.rotation_tile, args.rotations)
        padding = args.rotation_tile - (stop - start)
        y_tile = jnp.pad(direct_y[start:stop], ((0, padding), (0, 0)))
        w_tile = jnp.pad(direct_w[start:stop], ((0, padding), (0, 0)))
        direct_y_volume, direct_w_volume = backproject(
            direct_y_volume,
            direct_w_volume,
            y_tile,
            w_tile,
            grid.backprojection_rotations[start : start + args.rotation_tile],
        )

    @jax.jit
    def compiled_direct_adjoint(y_rows, w_rows, rotations):
        y_rows = jnp.pad(y_rows, ((0, grid.score_rotations.shape[0] - args.rotations), (0, 0)))
        w_rows = jnp.pad(w_rows, ((0, grid.score_rotations.shape[0] - args.rotations), (0, 0)))

        def body(tile_index, volumes):
            start = tile_index * args.rotation_tile
            ys = jax.lax.dynamic_slice_in_dim(y_rows, start, args.rotation_tile, axis=0)
            ws = jax.lax.dynamic_slice_in_dim(w_rows, start, args.rotation_tile, axis=0)
            rs = jax.lax.dynamic_slice_in_dim(rotations, start, args.rotation_tile, axis=0)
            return backproject(volumes[0], volumes[1], ys, ws, rs)

        return jax.lax.fori_loop(
            0,
            grid.score_rotations.shape[0] // args.rotation_tile,
            body,
            (jnp.zeros((volume_size,), jnp.complex64), jnp.zeros((volume_size,), jnp.float32)),
        )

    scan_y_volume, scan_w_volume = compiled_direct_adjoint(direct_y, direct_w, grid.backprojection_rotations)

    def maxima(a, b):
        return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))

    return {
        "direct_cuda_score_max_abs": maxima(gemm_scores, direct_scores),
        "tiled_score_vs_full_gemm_max_abs": maxima(tiled_scores, gemm_scores),
        "tiled_score_vs_direct_cuda_max_abs": maxima(tiled_scores, direct_scores),
        "direct_cuda_logz_max_abs": maxima(result.batch_logz[: args.images], direct_logz),
        "tiled_posterior_vs_direct_cuda_max_abs": maxima(tiled_q, direct_q),
        "tiled_numerator_slices_vs_direct_max_abs": maxima(tiled_y, direct_y),
        "tiled_weight_slices_vs_direct_max_abs": maxima(tiled_w, direct_w),
        "compiled_numerator_slice_sum_vs_direct_max_abs": maxima(probed.numerator, jnp.sum(direct_y, axis=0)),
        "compiled_weight_slice_sum_vs_direct_max_abs": maxima(probed.denominator, jnp.sum(direct_w, axis=0)),
        "captured_numerator_slices_vs_direct_max_abs": maxima(captured.numerator[: args.rotations], direct_y),
        "captured_weight_slices_vs_direct_max_abs": maxima(captured.denominator[: args.rotations], direct_w),
        "direct_score_min_winning_margin": float(
            np.min(
                np.sort(np.asarray(direct_scores).reshape(args.images, -1), axis=1)[:, -1]
                - np.sort(np.asarray(direct_scores).reshape(args.images, -1), axis=1)[:, -2]
            )
        ),
        "direct_cuda_mass_range": [
            float(np.min(np.asarray(jnp.sum(direct_q, axis=(1, 2))))),
            float(np.max(np.asarray(jnp.sum(direct_q, axis=(1, 2))))),
        ],
        "gemm_mass_range": [
            float(np.min(np.asarray(result.batch_mass[: args.images]))),
            float(np.max(np.asarray(result.batch_mass[: args.images]))),
        ],
        "direct_slices_norm": float(jnp.linalg.norm(direct_y)),
        "direct_weight_slices_norm": float(jnp.linalg.norm(direct_w)),
        "direct_cuda_numerator_volume_max_abs": maxima(result.numerator, direct_y_volume),
        "direct_cuda_weight_volume_max_abs": maxima(result.denominator, direct_w_volume),
        "gemm_numerator_volume_norm": float(jnp.linalg.norm(result.numerator)),
        "direct_numerator_volume_norm": float(jnp.linalg.norm(direct_y_volume)),
        "gemm_weight_volume_norm": float(jnp.linalg.norm(result.denominator)),
        "direct_weight_volume_norm": float(jnp.linalg.norm(direct_w_volume)),
        "compiled_adjoint_vs_host_numerator_max_abs": maxima(scan_y_volume, direct_y_volume),
        "compiled_adjoint_vs_host_weight_max_abs": maxima(scan_w_volume, direct_w_volume),
        "gemm_vs_compiled_direct_adjoint_numerator_max_abs": maxima(result.numerator, scan_y_volume),
        "gemm_vs_compiled_direct_adjoint_weight_max_abs": maxima(result.denominator, scan_w_volume),
    }


def _production_program(args, backproject, translation_angles):
    """Full-grid fused CUDA scorer plus the same native reconstruction slices.

    This control includes projector sampling inside the fused score launch.
    It retains exact two-sweep/lagged one-sweep semantics and one adjoint per
    rotation tile; it does not emulate RELION significant-support pruning.
    """
    qsize = args.rotation_tile
    exact = args.mode == "exact"
    n = args.box
    radius = n // 4
    native_index = jnp.arange(n * (n // 2 + 1), dtype=jnp.int32)
    half_width = n // 2 + 1
    # The synthetic GEMM operands follow centered-row projection order;
    # the fused coarse ABI enumerates native FFTW rows.
    full_to_compact = ((native_index // half_width + n // 2) % n) * half_width + native_index % half_width
    pf = args.padding_factor
    native_rows = getattr(args, "engine", "production") == "native"
    if native_rows:
        from relax.helpers.half_volume_mstep import relion_backprojector_volume_shape

        rec_native_indices = _native_reconstruction_indices(n, radius)
        rec_centered_indices = _centered_reconstruction_indices(rec_native_indices, n)
        volume_shape = relion_backprojector_volume_shape((n,) * 3, pf, current_size=2 * radius)

    @jax.jit
    def program(reference_full, y_volume, w_volume, old_table, next_table, batch, grid):
        nrot = grid.score_rotations.shape[0] // qsize
        old_pair = old_table[batch.particle_ids]

        def score(rotation_index):
            start = rotation_index * qsize
            rotations = jax.lax.dynamic_slice_in_dim(grid.score_rotations, start, qsize, axis=0)
            diff2 = relion_coarse_diff2_projector_f32(
                reference_full,
                rotations,
                batch.score_image,
                translation_angles,
                batch.score_weight,
                batch.initial_diff2,
                full_to_compact,
                current_size=n,
                physical_image_size=n,
                model_max_r=radius,
                padding_factor=pf,
                canonical_reduction=True,
            )
            rprior = jax.lax.dynamic_slice_in_dim(batch.rotation_prior, start, qsize, axis=1)
            valid_rot = jax.lax.dynamic_slice_in_dim(grid.valid_rotations, start, qsize, axis=0)
            value = -diff2 + rprior[:, :, None] + batch.translation_prior[:, None, : args.translations]
            return jnp.where(batch.valid_images[:, None, None] & valid_rot[None, :, None], value, -jnp.inf)

        if exact:
            def score_rotation(i, pair):
                return merge_normalizers(pair, tile_normalizer(score(i)))

            normalizer = jax.lax.fori_loop(
                0, nrot, score_rotation, empty_normalizer_table(args.images)
            )
        else:
            normalizer = old_pair

        def reconstruct_rotation(i, state):
            y, w, pair, mass = state
            scores = score(i)
            if not exact:
                pair = merge_normalizers(pair, tile_normalizer(scores))
            weights = jnp.where(
                batch.valid_images[:, None, None],
                jnp.exp((scores - normalizer[:, 0, None, None]) - normalizer[:, 1, None, None]),
                0,
            )
            bp_rotations = jax.lax.dynamic_slice_in_dim(grid.backprojection_rotations, i * qsize, qsize, axis=0)
            if native_rows:
                row_ids = jnp.repeat(jnp.arange(args.images, dtype=jnp.int32), qsize)
                data_flat, _, _, weight_flat = relion_translate_sum_flat_rows_f32(
                    batch.rec_raw_image,
                    batch.rec_raw_image,
                    row_ids,
                    weights.reshape(args.images * qsize, args.translations),
                    translation_angles,
                    rec_centered_indices,
                    jnp.asarray(args.images * qsize, dtype=jnp.int32),
                    jnp.asarray(batch.rec_image.shape[1], dtype=jnp.int32),
                    recon_weight=batch.rec_weighted_ctf,
                    ctf2_over_nv=batch.rec_weight,
                    image_shape=(n, n),
                )
                data_rows = data_flat.reshape(args.images, qsize, batch.rec_image.shape[1])
                weight_rows = weight_flat.reshape(args.images, qsize, batch.rec_image.shape[1])
                particle_rotations = jnp.broadcast_to(bp_rotations[None], (args.images, qsize, 3, 3))
                y, w = relion_fused_x_half_backproject_particle_grid_indexed(
                    y, w, data_rows, weight_rows, rec_native_indices, particle_rotations,
                    (n, n), volume_shape, float(radius),
                )
            else:
                ys, ws = weighted_slices(
                    weights, batch.rec_image, batch.rec_weight, grid.rec_phase[: args.translations], translation_side="image"
                )
                y, w = backproject(y, w, ys, ws, bp_rotations)
            return y, w, pair, mass + jnp.sum(weights, axis=(1, 2))

        y_volume, w_volume, calculated_pair, mass = jax.lax.fori_loop(
            0,
            nrot,
            reconstruct_rotation,
            (
                y_volume, w_volume,
                empty_normalizer_table(args.images),
                jnp.zeros((args.images,), jnp.float32),
            ),
        )
        pair = normalizer if exact else calculated_pair
        logz = normalizer_logz(pair)
        next_table = next_table.at[batch.particle_ids].set(pair)
        invalid_normalizer = batch.valid_images & (
            ~jnp.isfinite(pair).all(axis=1) | (~jnp.isfinite(old_pair).all(axis=1) if not exact else False)
        )
        invalid_weight = batch.valid_images & (~jnp.isfinite(mass) | (mass <= 0))
        return DenseGemmBatchResult(
            y_volume, w_volume, normalizer_logz(next_table), next_table,
            logz, mass, invalid_normalizer, invalid_weight
        )

    return program


def _score_diagnostics(args, case, absolute_logz):
    """Compare matched F32 candidate scores, then diagnose their normalization.

    Float64 is used only on materialized F32 scores to identify cancellation;
    it is never used by the EM operator or for a production timing claim.
    """
    reference, batch, grid, project, _, _, _, angles, _ = case
    n, qsize, usize = args.box, args.rotation_tile, args.translation_tile
    radius = n // 4
    half_width = n // 2 + 1
    native_index = jnp.arange(n * half_width, dtype=jnp.int32)
    lookup = ((native_index // half_width + n // 2) % n) * half_width + native_index % half_width
    full_reference = relion_projector_half_to_texture_full(reference)
    gemm_rows, fused_rows = [], []
    fused_repeat_max_abs = 0.0
    for rot_start in range(0, args.rotations, qsize):
        rotations = grid.score_rotations[rot_start : rot_start + qsize]
        projected = project(reference, rotations)
        tiled = []
        for tstart in range(0, args.translations, usize):
            scores = score_tile(
                projected, batch.score_image, batch.score_weight, batch.initial_diff2,
                grid.score_phase[tstart : tstart + usize],
                batch.rotation_prior[:, rot_start : rot_start + qsize],
                batch.translation_prior[:, tstart : tstart + usize],
                batch.valid_images, grid.valid_rotations[rot_start : rot_start + qsize],
                grid.valid_translations[tstart : tstart + usize],
                translation_side=args.translation_side,
            )
            tiled.append(np.asarray(scores)[:, :, : min(usize, args.translations - tstart)])
        gemm_rows.append(np.concatenate(tiled, axis=2)[:, : min(qsize, args.rotations - rot_start)])
        diff2 = relion_coarse_diff2_projector_f32(
            full_reference, rotations, batch.score_image, angles,
            batch.score_weight, batch.initial_diff2, lookup,
            current_size=n, physical_image_size=n, model_max_r=radius,
            padding_factor=args.padding_factor,
            canonical_reduction=True,
        )
        repeated_diff2 = relion_coarse_diff2_projector_f32(
            full_reference, rotations, batch.score_image, angles,
            batch.score_weight, batch.initial_diff2, lookup,
            current_size=n, physical_image_size=n, model_max_r=radius,
            padding_factor=args.padding_factor,
            canonical_reduction=True,
        )
        fused_repeat_max_abs = max(
            fused_repeat_max_abs,
            float(np.max(np.abs(np.asarray(diff2) - np.asarray(repeated_diff2)))),
        )
        fused = (
            -np.asarray(diff2)
            + np.asarray(batch.rotation_prior[:, rot_start : rot_start + qsize, None])
            + np.asarray(batch.translation_prior[:, None, : args.translations])
        )
        fused_rows.append(fused[:, : min(qsize, args.rotations - rot_start)])
    gemm = np.concatenate(gemm_rows, axis=1).reshape(args.images, -1)
    fused = np.concatenate(fused_rows, axis=1).reshape(args.images, -1)
    deltas = gemm - fused

    def normalized(scores):
        wide = scores.astype(np.float64)
        maximum = np.max(wide, axis=1)
        relative = np.exp(wide - maximum[:, None])
        total = np.sum(relative, axis=1)
        q = relative / total[:, None]
        return maximum + np.log(total), q, maximum

    gemm_logz64, gemm_q64, gemm_max = normalized(gemm)
    fused_logz64, fused_q64, fused_max = normalized(fused)
    order = np.argsort(fused, axis=1)
    winning_margin = fused[np.arange(args.images), order[:, -1]] - fused[np.arange(args.images), order[:, -2]]
    naive_gemm_mass = np.sum(np.exp(gemm - np.asarray(absolute_logz)[:, None]), axis=1)
    return {
        "label": "diagnostic float64 normalization of matched float32 candidate scores only",
        "gemm_vs_fused_score_max_abs": float(np.max(np.abs(deltas))),
        "gemm_vs_fused_score_rms": float(np.sqrt(np.mean(deltas.astype(np.float64) ** 2))),
        "fused_repeat_score_max_abs": fused_repeat_max_abs,
        "gemm_vs_fused_winner_flips": int(np.count_nonzero(np.argmax(gemm, axis=1) != np.argmax(fused, axis=1))),
        "fused_winning_margin_min": float(np.min(winning_margin)),
        "fused_winning_margin_median": float(np.median(winning_margin)),
        "gemm_vs_fused_diagnostic_logz64_max_abs": float(np.max(np.abs(gemm_logz64 - fused_logz64))),
        "gemm_vs_fused_diagnostic_q64_max_abs": float(np.max(np.abs(gemm_q64 - fused_q64))),
        "gemm_max_score_range": [float(np.min(gemm_max)), float(np.max(gemm_max))],
        "fused_max_score_range": [float(np.min(fused_max)), float(np.max(fused_max))],
        "gemm_absolute_logz_vs_diagnostic_max_abs": float(np.max(np.abs(np.asarray(absolute_logz) - gemm_logz64))),
        "gemm_naive_mass_error_max_abs": float(np.max(np.abs(naive_gemm_mass - 1))),
    }


def _score_only_program(args, project, translation_angles):
    """One complete fixed-grid score/logZ sweep without reconstruction."""
    qsize, usize = args.rotation_tile, args.translation_tile
    n = args.box
    radius = n // 4
    native_index = jnp.arange(n * (n // 2 + 1), dtype=jnp.int32)
    half_width = n // 2 + 1
    full_to_compact = ((native_index // half_width + n // 2) % n) * half_width + native_index % half_width

    @jax.jit
    def score_all(reference, batch, grid):
        def rotation_loop(i, pair):
            start = i * qsize
            rotations = jax.lax.dynamic_slice_in_dim(grid.score_rotations, start, qsize, axis=0)
            rprior = jax.lax.dynamic_slice_in_dim(batch.rotation_prior, start, qsize, axis=1)
            valid_rot = jax.lax.dynamic_slice_in_dim(grid.valid_rotations, start, qsize, axis=0)
            if args.engine != "gemm":
                diff2 = relion_coarse_diff2_projector_f32(
                    reference, rotations, batch.score_image, translation_angles,
                    batch.score_weight, batch.initial_diff2, full_to_compact,
                    current_size=n, physical_image_size=n, model_max_r=radius,
                    padding_factor=args.padding_factor,
                    canonical_reduction=True,
                )
                scores = -diff2 + rprior[:, :, None] + batch.translation_prior[:, None, : args.translations]
                scores = jnp.where(batch.valid_images[:, None, None] & valid_rot[None, :, None], scores, -jnp.inf)
                return merge_normalizers(pair, tile_normalizer(scores))
            projected = project(reference, rotations)
            model_power = score_model_power(projected, batch.score_weight)

            def translation_loop(j, value):
                tstart = j * usize
                phase = jax.lax.dynamic_slice_in_dim(grid.score_phase, tstart, usize, axis=0)
                tprior = jax.lax.dynamic_slice_in_dim(batch.translation_prior, tstart, usize, axis=1)
                valid_trans = jax.lax.dynamic_slice_in_dim(grid.valid_translations, tstart, usize, axis=0)
                scores = score_tile(
                    projected, batch.score_image, batch.score_weight, batch.initial_diff2,
                    phase, rprior, tprior, batch.valid_images, valid_rot, valid_trans,
                    translation_side=args.translation_side,
                    model_power=model_power,
                )
                return merge_normalizers(value, tile_normalizer(scores))

            return jax.lax.fori_loop(0, grid.score_phase.shape[0] // usize, translation_loop, pair)

        pair = jax.lax.fori_loop(
            0, grid.score_rotations.shape[0] // qsize,
            rotation_loop, empty_normalizer_table(args.images),
        )
        return normalizer_logz(pair)

    return score_all


def main():
    args = arguments()
    if jax.default_backend() != "gpu":
        raise RuntimeError("benchmark requires an explicitly selected CUDA GPU")
    if min(args.images, args.rotations, args.translations, args.rotation_tile, args.translation_tile) <= 0:
        raise ValueError("all sizes and tile capacities must be positive")
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats positive")
    if args.engine != "gemm" and args.compare_direct:
        raise ValueError("--compare-direct currently requires --engine gemm")
    if args.engine != "gemm" and args.diagnose_scores:
        raise ValueError("--diagnose-scores currently requires --engine gemm")
    if args.timing_style == "device-reset" and args.no_donation:
        raise ValueError("--no-donation requires --timing-style init-inclusive")
    native = _native_identity()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    case = _make_case(args)
    reference, batch, grid, project, backproject, _, _, _, _ = case
    texture = None
    if args.engine == "gemm":
        from relax.helpers.half_volume_mstep import relion_backprojector_volume_shape

        texture = RelionCapacityHalfTextureF32(reference, args.box // 4, padding_factor=args.padding_factor)
        project, backproject = native_relion_callbacks(
            image_shape=(args.box, args.box),
            score_indices=case[5],
            rec_indices=case[6],
            r_max=args.box // 4,
            projector_output_size=args.box,
            volume_shape=relion_backprojector_volume_shape(
                (args.box,) * 3, args.padding_factor, current_size=args.box // 2
            ),
            backprojection_backend=args.adjoint_backend,
            padding_factor=args.padding_factor,
            capacity_texture=texture,
        )
        case = (reference, batch, grid, project, backproject, *case[5:])
    else:
        reference = relion_projector_half_to_texture_full(reference)
        case = (reference, batch, grid, project, backproject, *case[5:])
    config = DenseGemmTileConfig(
        args.images, args.rotation_tile, args.translation_tile, args.translation_side, args.mode
    )
    program = (
        make_batch_program(config, project, backproject)
        if args.engine == "gemm"
        else _production_program(args, backproject, case[7])
    )
    if args.no_donation and args.engine == "gemm":
        program = jax.jit(program.__wrapped__)
    tile_bytes = leading_tile_bytes(
        image_capacity=args.images,
        rotation_tile=args.rotation_tile,
        translation_tile=args.translation_tile,
        score_pixels=batch.score_image.shape[1],
        rec_pixels=batch.rec_image.shape[1],
        side=args.translation_side,
    )
    if sum(tile_bytes.values()) > args.tile_budget_gb * 1e9:
        raise MemoryError("leading live tile estimate exceeds explicit tile budget")
    old = empty_normalizer_table(args.images + 1)
    if args.mode == "lagged":
        exact_config = DenseGemmTileConfig(
            args.images, args.rotation_tile, args.translation_tile, args.translation_side, "exact"
        )
        exact_program = (
            make_batch_program(exact_config, project, backproject)
            if args.engine == "gemm"
            else _production_program(argparse.Namespace(**{**vars(args), "mode": "exact"}), backproject, case[7])
        )
        bootstrap, _ = _run_once(exact_program, case, old)
        old = bootstrap.next_pair_table
    device = jax.devices()[0]
    scratch = (
        jnp.zeros((case[-1],), jnp.complex64),
        jnp.zeros((case[-1],), jnp.float32),
        empty_normalizer_table(args.images + 1),
    )
    execution_program = _reusable_program(program) if args.timing_style == "device-reset" else program
    memory_before_compile = _memory_stats(device)
    lower = execution_program.lower(reference, scratch[0], scratch[1], old, scratch[2], batch, grid)
    compile_start = time.perf_counter()
    compiled = lower.compile()
    compile_seconds = time.perf_counter() - compile_start
    hlo = compiled.as_text()
    hlo_path = args.output.with_suffix(".hlo.txt")
    hlo_path.write_text(hlo)
    volume_aliases = _volume_aliases(hlo, compiled, case[-1])
    if args.timing_style == "device-reset" and not volume_aliases["verified"]:
        alias_path = args.output.with_suffix(".alias.json")
        alias_path.write_text(json.dumps({"volume_aliases": volume_aliases, "memory": _compiled_memory(compiled)}, indent=2) + "\n")
        raise RuntimeError(f"device-reset executable did not alias both donated volumes; see {alias_path}")
    memory_after_compile = _memory_stats(device)
    for _ in range(args.warmup):
        if args.timing_style == "device-reset":
            _, _, scratch = _run_reusable(compiled, case, old, scratch)
        else:
            _run_once(compiled, case, old)
    memory_after_warmup = _memory_stats(device)
    timings = []
    result = None
    if args.profile_warm_repeats:
        _profiler_boundary(True)
    try:
        trace_range = jax.profiler.TraceAnnotation("dense_gemm_operator_warmed_repeats") if args.profile_warm_repeats else nullcontext()
        with trace_range:
            for _ in range(args.repeats):
                if args.timing_style == "device-reset":
                    result, elapsed, scratch = _run_reusable(compiled, case, old, scratch)
                else:
                    result, elapsed = _run_once(compiled, case, old)
                timings.append(elapsed)
    finally:
        if args.profile_warm_repeats:
            _profiler_boundary(False)
    memory_after_repeats = _memory_stats(device)
    texture_payload_bytes = 2 * 4 * int(np.prod(texture.shape)) if texture is not None else None
    score_only = _score_only_program(args, project, case[7])
    score_only_logz = score_only(reference, batch, grid)
    jax.block_until_ready(score_only_logz)
    score_timings = []
    last_score_result = score_only_logz
    for _ in range(args.repeats):
        start = time.perf_counter()
        last_score_result = score_only(reference, batch, grid)
        jax.block_until_ready(last_score_result)
        score_timings.append(time.perf_counter() - start)
    score_logz_max_abs = float(np.max(np.abs(np.asarray(score_only_logz[: args.images]) - np.asarray(result.batch_logz[: args.images]))))
    comparison = _direct_comparison(args, case, result, old) if args.compare_direct else None
    score_diagnostics = _score_diagnostics(args, case, result.batch_logz[: args.images]) if args.diagnose_scores else None
    if texture is not None:
        completion = (
            project(reference, grid.score_rotations[: args.rotation_tile])
            if args.compare_direct or args.diagnose_scores else last_score_result
        )
        texture.close_after(completion)
    host_arrays = {
        "logz": np.asarray(result.batch_logz[: args.images]),
        "mass": np.asarray(result.batch_mass[: args.images]),
        "numerator": np.asarray(result.numerator),
        "denominator": np.asarray(result.denominator),
    }
    invalid_arrays = [name for name, value in host_arrays.items() if not np.isfinite(value).all()]
    if np.any(host_arrays["mass"] <= 0):
        invalid_arrays.append("nonpositive_mass")
    arrays_receipt = None
    if args.save_arrays:
        arrays_path = args.output.with_suffix(".arrays.npz")
        np.savez_compressed(arrays_path, **host_arrays)
        arrays_receipt = {"path": str(arrays_path), "sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest()}
    report = {
        "schema": "relax.dense_gemm_em.synthetic_native.v2",
        "source": _source_identity(Path(__file__).resolve().parents[1]),
        "native": native,
        "backend": jax.default_backend(),
        "device": str(device),
        "precision": {
            "score": "float32 F32_F32_F32",
            "weighted_sums": "float32/complex64 HIGHEST",
            "projection": "complex64 CUDA texture",
            "backprojection": "complex64/float32 CUDA fused x-half",
        },
        "workload": {
            "kind": "synthetic_preassembled_operands",
            "engine": args.engine,
            "operator_control": {
                "gemm": "resident GEMM scorer and GEMM 2D reduction with fused adjoint",
                "production": "canonical fused CUDA scorer with shared GEMM 2D reduction and fused adjoint",
                "native": "canonical fused CUDA scorer, native BPref translation and particle-owned fused adjoint",
            }[args.engine],
            "fused_coarse_reduction": "canonical" if args.engine != "gemm" else None,
            "mode": args.mode,
            "side": args.translation_side,
            "B": args.images,
            "R": args.rotations,
            "T": args.translations,
            "Q": args.rotation_tile,
            "U": args.translation_tile,
            "box": args.box,
            "padding_factor": args.padding_factor,
            "seed": args.seed,
            "candidate_count": args.images * args.rotations * args.translations,
            "adjoint_calls_per_batch": int(np.ceil(args.rotations / args.rotation_tile)),
            "native_particle_launches_per_batch": (
                args.images * int(np.ceil(args.rotations / args.rotation_tile)) if args.engine == "native" else None
            ),
            "adjoint_backend": args.adjoint_backend,
            "donated_accumulators": not args.no_donation,
            "timing_style": args.timing_style,
        },
        "tile_bytes_lower_bound": tile_bytes,
        "compile_seconds": compile_seconds,
        "warmed_seconds": timings,
        "median_seconds": statistics.median(timings),
        "score_sweep_seconds": score_timings,
        "score_sweep_median_seconds": statistics.median(score_timings),
        "score_logz_vs_operator_max_abs": score_logz_max_abs,
        "operator_includes_zero_init": True,
        "operator_zero_init_location": "compiled_device_reset" if args.timing_style == "device-reset" else "host_dispatched_allocations",
        "profiler_boundaries": "cudaProfilerStart/Stop plus NVTX dense_gemm_operator_warmed_repeats around warmed operator repeats" if args.profile_warm_repeats else None,
        "reference_conversion_excluded": True,
        "texture_staging_excluded": texture is not None,
        "particles_per_second": args.images / statistics.median(timings),
        "memory": {
            "compiled": _compiled_memory(compiled),
            "compiled_estimate_scope": "XLA arguments + outputs - aliases + temporaries; excludes native CUDA allocations",
            "resident_texture_payload_bytes_lower_bound": texture_payload_bytes,
            "resident_texture_scope": "two float32 CUDA arrays; CUDA allocation overhead excluded" if texture is not None else None,
            "device_stats_scope": "JAX allocator counters, not native CUDA arrays or process RSS",
            "device_before_compile": memory_before_compile,
            "device_after_compile": memory_after_compile,
            "device_after_warmup": memory_after_warmup,
            "device_after_repeats": memory_after_repeats,
        },
        "lowering": {
            "hlo_path": str(hlo_path),
            "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(),
            "volume_aliases": volume_aliases,
            "cublas_gemm_calls_in_hlo": hlo.count('custom_call_target="__cublas$gemm"'),
            "fused_coarse_calls_in_hlo": hlo.count("cuda_relion_coarse_diff2_projector_f32"),
            "contains_cuda_custom_call": "custom-call" in hlo,
            "contains_f32_algorithm": "F32_F32_F32" in hlo,
        },
        "comparison": comparison,
        "score_diagnostics": score_diagnostics,
        "arrays": arrays_receipt,
        "invalid_arrays": invalid_arrays,
        "small_result": (
            {
                "logz": np.asarray(result.batch_logz[: args.images]).tolist(),
                "mass": np.asarray(result.batch_mass[: args.images]).tolist(),
                "numerator_real": np.asarray(result.numerator.real).tolist(),
                "numerator_imag": np.asarray(result.numerator.imag).tolist(),
                "denominator": np.asarray(result.denominator).tolist(),
            }
            if args.images <= 8 and args.box <= 16 else None
        ),
        "invalid_normalizer_ids": np.asarray(batch.particle_ids)[np.asarray(result.invalid_normalizer)].tolist(),
        "invalid_weight_ids": np.asarray(batch.particle_ids)[np.asarray(result.invalid_weight)].tolist(),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if report["invalid_normalizer_ids"] or report["invalid_weight_ids"] or invalid_arrays:
        raise RuntimeError(f"invalid posterior state; see {args.output}")
    print(
        json.dumps(
            {"output": str(args.output), "median_seconds": report["median_seconds"], "comparison": comparison}, indent=2
        )
    )


if __name__ == "__main__":
    main()
