"""Focused tests for RELION's fused fine-Gaussian CUDA FFI."""

from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest
from helpers.cuda_source import read_em_cuda_source
from helpers.float_compare import assert_matches

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from relax.relion import relion_ctf

pytestmark = pytest.mark.unit


def _fma32(left, right, addend):
    return np.asarray(
        np.asarray(left, dtype=np.float64) * np.asarray(right, dtype=np.float64)
        + np.asarray(addend, dtype=np.float64),
        dtype=np.float32,
    )


def _fma64(left, right, addend):
    with localcontext() as context:
        context.prec = 200
        exact = (
            Decimal.from_float(float(left)) * Decimal.from_float(float(right))
            + Decimal.from_float(float(addend))
        )
    return np.float64(float(exact))


def _complex_normal_for_test(rng, shape):
    return (
        rng.normal(0.0, 0.02, shape) + 1j * rng.normal(0.0, 0.02, shape)
    ).astype(np.complex64)


def _fine_diff2_update_reference(
    reference,
    shifted,
    weight,
    lane_sum=np.float32(0),
    *,
    prehalf_weight,
):
    """Binary32 oracle for the two mathematically equivalent source orders."""

    diff_real = np.float32(reference.real - shifted.real)
    diff_imag = np.float32(reference.imag - shifted.imag)
    imag_square = np.float32(diff_imag * diff_imag)
    square_sum = _fma32(diff_real, diff_real, imag_square)
    if prehalf_weight:
        staged_weight = np.float32(weight * np.float32(0.5))
        return _fma32(square_sum, staged_weight, lane_sum)
    half_square_sum = np.float32(square_sum * np.float32(0.5))
    return _fma32(half_square_sum, weight, lane_sum)


def _production_reference(reference, shifted, weight, lookup):
    lanes = np.zeros(256, dtype=np.float32)
    for full_pixel, compact_pixel in enumerate(lookup):
        if compact_pixel < 0:
            continue
        diff_real = np.float32(
            reference[compact_pixel].real - shifted[compact_pixel].real
        )
        diff_imag = np.float32(
            reference[compact_pixel].imag - shifted[compact_pixel].imag
        )
        imag_square = np.float32(diff_imag * diff_imag)
        square_sum = _fma32(diff_real, diff_real, imag_square)
        half_square_sum = np.float32(square_sum * np.float32(0.5))
        lane = full_pixel % 256
        lanes[lane] = _fma32(half_square_sum, weight[compact_pixel], lanes[lane])
    for width in (128, 64, 32, 16, 8, 4, 2, 1):
        lanes[:width] = np.add(lanes[:width], lanes[width : 2 * width], dtype=np.float32)
    return np.float32(lanes[0])


def _production_reference_f64(reference, shifted, weight, lookup):
    lanes = np.zeros(256, dtype=np.float64)
    for full_pixel, compact_pixel in enumerate(lookup):
        if compact_pixel < 0:
            continue
        diff_real = np.float64(reference[compact_pixel].real - shifted[compact_pixel].real)
        diff_imag = np.float64(reference[compact_pixel].imag - shifted[compact_pixel].imag)
        imag_square = np.float64(diff_imag * diff_imag)
        square_sum = _fma64(diff_real, diff_real, imag_square)
        half_square_sum = np.float64(square_sum * np.float64(0.5))
        lane = full_pixel % 256
        lanes[lane] = _fma64(half_square_sum, weight[compact_pixel], lanes[lane])
    for width in (128, 64, 32, 16, 8, 4, 2, 1):
        lanes[:width] = np.add(lanes[:width], lanes[width : 2 * width], dtype=np.float64)
    return np.float64(lanes[0])


def _coarse_production_result(
    reference,
    shifted,
    weight,
    lookup,
    *,
    translation_count,
    initial_diff2=np.float32(0),
):
    active_lanes = 128 // translation_count
    lanes = np.zeros(active_lanes, dtype=np.float32)
    for chunk_start in range(0, lookup.size, 32):
        for lane in range(active_lanes):
            for pixel_in_chunk in range(lane, 32, active_lanes):
                full_pixel = chunk_start + pixel_in_chunk
                if full_pixel >= lookup.size:
                    break
                compact_pixel = lookup[full_pixel]
                if compact_pixel < 0:
                    continue
                diff_real = np.float32(
                    reference[compact_pixel].real - shifted[compact_pixel].real
                )
                diff_imag = np.float32(
                    reference[compact_pixel].imag - shifted[compact_pixel].imag
                )
                imag_square = np.float32(diff_imag * diff_imag)
                square_sum = _fma32(diff_real, diff_real, imag_square)
                half_square_sum = np.float32(square_sum * np.float32(0.5))
                lanes[lane] = _fma32(
                    half_square_sum,
                    weight[compact_pixel],
                    lanes[lane],
                )
    # The kernel adds the lane partials atomically in hardware order; this is
    # one ordering, and the others agree with it in the default float32 band.
    total = np.float32(initial_diff2)
    for lane in range(active_lanes):
        total = np.float32(total + lanes[lane])
    return total


def _operands():
    rng = np.random.default_rng(20)
    pixel_count = 513
    reference = (
        rng.normal(0, 0.02, pixel_count)
        + 1j * rng.normal(0, 0.02, pixel_count)
    ).astype(np.complex64)
    shifted = (
        rng.normal(0, 0.02, pixel_count)
        + 1j * rng.normal(0, 0.02, pixel_count)
    ).astype(np.complex64)
    weight = rng.uniform(0, 150_000, pixel_count).astype(np.float32)
    weight[rng.random(pixel_count) < 0.2] = 0
    lookup = np.arange(pixel_count, dtype=np.int32)
    return reference, shifted, weight, lookup


def test_relion_fine_diff2_cuda_source_pins_production_rounding_order():
    source = read_em_cuda_source()

    start = source.index("relion_fine_diff2_update_f32")
    prehalf_start = source.index("relion_fine_diff2_update_prehalf_f32", start)
    block = source[start:prehalf_start]
    prehalf_block = source[prehalf_start : source.index("__global__", prehalf_start)]
    assert "__fsub_rn(reference.x, shifted_image.x)" in block
    assert "__fmul_rn(diff_imag, diff_imag)" in block
    assert "__fmaf_rn(diff_real, diff_real, imag_square)" in block
    assert "__fmul_rn(square_sum, 0.5f)" in block
    assert "__fmaf_rn(half_square_sum, weight, lane_sum)" in block
    assert "__fmaf_rn(square_sum, prehalved_weight, lane_sum)" in prehalf_block
    assert "0.5f" not in prehalf_block
    kernel_start = source.index("void relion_fine_diff2_rectangular_kernel(")
    kernel = source[kernel_start : source.index("__global__", kernel_start)]
    # PR180 shares the body across precision modes; production must still
    # select the explicit float32 update, reduction, and initial-diff2 add.
    assert "if constexpr (std::is_same_v<T, float>)" in kernel
    assert "lane_sum = relion_fine_diff2_update_f32(" in kernel
    assert "lane_sum = relion_fine_diff2_update_f64(" in kernel
    assert "__shared__ T lane_sums[kRelionFineDiff2BlockSize]" in kernel
    assert "if constexpr (ADD_INITIAL)" in kernel
    assert "__fadd_rn(lane_sums[0], initial_diff2[batch])" in kernel
    assert "__dadd_rn(lane_sums[0], initial_diff2[batch])" in kernel


def test_relion_coarse_prehalf_cpu_oracle_pins_equivalent_source_orders():
    rng = np.random.default_rng(71)
    for _ in range(128):
        reference = np.complex64(np.float32(rng.uniform(-2.0, 2.0)) + 1j * np.float32(rng.uniform(-2.0, 2.0)))
        shifted = np.complex64(np.float32(rng.uniform(-2.0, 2.0)) + 1j * np.float32(rng.uniform(-2.0, 2.0)))
        weight = np.float32(rng.uniform(0.25, 16.0))
        lane_sum = np.float32(rng.uniform(0.25, 16.0))
        production = _fine_diff2_update_reference(
            reference,
            shifted,
            weight,
            lane_sum,
            prehalf_weight=False,
        )
        prehalved = _fine_diff2_update_reference(
            reference,
            shifted,
            weight,
            lane_sum,
            prehalf_weight=True,
        )
        assert_matches(production, prehalved)

    # A normal result built from a subnormal square distinguishes which
    # operand is halved: zero against 2**-23, far outside any rounding band.
    min_subnormal = np.asarray(1, dtype=np.uint32).view(np.float32)[()]
    image = np.complex64(np.sqrt(np.float64(min_subnormal)) + 0j)
    weight = np.float32(2.0**127)
    production = _fine_diff2_update_reference(
        np.complex64(0),
        image,
        weight,
        prehalf_weight=False,
    )
    prehalved = _fine_diff2_update_reference(
        np.complex64(0),
        image,
        weight,
        prehalf_weight=True,
    )
    assert production == np.float32(0)
    assert_matches(prehalved, np.float32(2.0**-23))


def test_relion_fused_translate_cuda_source_pins_native_block_topology():
    source = read_em_cuda_source()

    assert "constexpr int kRelionFineDiff2TranslationCapacity = 7;" in source
    assert "constexpr int kRelionFineDiff2Ref3dJobChunk = 4;" in source
    assert "template <bool FlatRows>" in source
    assert "relion_fine_diff2_fused_translate_rows_f32_kernel" in source
    assert "relion_fine_diff2_fused_translate_rows_f32_kernel<false>" in source
    assert "relion_fine_diff2_fused_translate_rows_f32_kernel<true>" in source
    assert "row_image_ids[row]" in source
    assert "relion_score_translate_f32(" in source
    assert "translation_offset * kRelionFineDiff2BlockSize" in source
    assert "lane_sums[lane_index] = relion_fine_diff2_update_f32(" in source
    assert "initial_diff2[batch]" in source
    assert "runtime_current_size[0]" in source
    assert "logical_full_pixel_count" in source
    assert "RelionFineDiff2FusedTranslateRuntimeFlatRowsF32" in source
    assert "RelionFineDiff2FusedTranslateRuntimeRectangularF32" in source

    pair_start = source.index(
        "relion_fine_diff2_fused_translate_pairs_f32_kernel"
    )
    pair_kernel = source[pair_start : source.index("cudaError_t", pair_start)]
    assert "constexpr int kRelionFineDiff2PairsPerBlock = 16;" in source
    assert "pair_count + kRelionFineDiff2PairsPerBlock - 1" in pair_kernel
    assert "pair_chunk * kRelionFineDiff2PairsPerBlock" in pair_kernel
    assert (
        "kRelionFineDiff2BlockSize * kRelionFineDiff2PairsPerBlock"
        in pair_kernel
    )
    assert "image_value = image[image_index]" in pair_kernel
    assert "pixel_weight = weight[image_index]" in pair_kernel

    launch_start = source.index(
        "cudaError_t launch_relion_fine_diff2_fused_translate_pairs_f32"
    )
    launch = source[launch_start : source.index("__global__", launch_start)]
    assert "const int64_t total_blocks = batch_size * pair_chunks;" in launch
    assert "static_cast<unsigned int>(total_blocks)" in launch

    jobs_start = source.index(
        "relion_fine_diff2_fused_translate_jobs_f32_kernel"
    )
    jobs_kernel = source[jobs_start : source.index("cudaError_t", jobs_start)]
    assert "4 * (job_start + offset)" in jobs_kernel
    assert "job_plan[plan_offset + 1]" in jobs_kernel
    assert "job_plan[plan_offset + 3]" in jobs_kernel
    assert "kRelionFineDiff2BlockSize * kJobsPerBlock" in jobs_kernel
    assert "relion_fine_diff2_update_f32(" in jobs_kernel
    assert "initial_diff2[image_rows[offset]]" in jobs_kernel

    jobs_launch_start = source.index(
        "cudaError_t launch_relion_fine_diff2_fused_translate_jobs_f32"
    )
    jobs_launch = source[
        jobs_launch_start : source.index("__global__", jobs_launch_start)
    ]
    assert "job_count + kRelionFineDiff2Ref3dJobChunk - 1" in jobs_launch
    assert "static_cast<unsigned int>(total_blocks)" in jobs_launch


def test_relion_powerclass_cuda_source_pins_native_atomic_topology():
    source = read_em_cuda_source()

    start = source.index("relion_powerclass_spectrum_highres_f32_kernel")
    block = source[start : source.index("cudaError_t", start)]
    assert "kRelionPowerClassBlockSize = 128" in source
    assert "__float2int_rn(sqrtf(" in block
    assert "atomicAdd(&spectrum[shell], value)" in block
    assert "highres_lanes[tid] += highres_lanes[tid + width]" in block
    assert "atomicAdd(highres_xi2, highres_lanes[0])" in block


def test_relion_coarse_diff2_cuda_source_pins_production_topology():
    source = read_em_cuda_source()

    start = source.index("relion_coarse_diff2_rotation_block_f32")
    block = source[start : source.index("cudaError_t", start)]
    assert "kRelionCoarseDiff2BlockSize = 128" in source
    assert "kRelionCoarseEulersPerBlock = 16" in source
    assert "kRelionCoarsePrefetchFraction = 4" in source
    assert "threadIdx.x % translation_count" in block
    assert "threadIdx.x / translation_count" in block
    assert "pixel_in_chunk += active_lanes" in block
    assert "atomicAdd(" in block
    f64_start = source.index("relion_coarse_diff2_rectangular_f64_kernel")
    f64_block = source[f64_start : source.index("cudaError_t", f64_start)]
    assert "double lane_sums" in f64_block
    assert "relion_fine_diff2_update_f64" in f64_block
    assert "atomicAdd(" in f64_block


    # The rectangular kernel and its runtime-size variant share exactly one
    # production reduction call each; neither duplicates the arithmetic.
    callers = (
        "relion_coarse_diff2_rectangular_f32_kernel",
        "relion_coarse_diff2_rectangular_runtime_f32_kernel",
    )
    for caller in callers:
        caller_start = source.index("void " + caller + "(")
        caller_block = source[caller_start : source.index("__global__", caller_start)]
        assert caller_block.count("relion_coarse_diff2_rotation_block_f32(") == 1
    # One definition plus the explicitly enumerated callers above.
    assert block.count("relion_coarse_diff2_rotation_block_f32(") == 1 + len(callers)


def test_relion_coarse_normalized_cc_source_pins_native_tree_and_atomics():
    source = read_em_cuda_source()

    start = source.index("relion_coarse_normalized_cc_pairs_f32_kernel")
    block = source[start : source.index("cudaError_t", start)]
    assert "packed_pixel += kRelionCoarseDiff2BlockSize" in block
    assert "reference_value.x * image_value.x" in block
    assert "reference_value.y * image_value.y" in block
    assert "reference_value.x * reference_value.x" in block
    assert "reference_value.y * reference_value.y" in block
    assert "for (int width = kRelionCoarseDiff2BlockSize / 2" in block
    assert "sqrtf(" in block
    assert "atomicAdd(&output[candidate], contribution)" in block

    texture_start = source.index(
        "relion_coarse_normalized_cc_native_texture_pairs_f32_kernel"
    )
    texture_block = source[texture_start : source.index("cudaError_t", texture_start)]
    assert "relion_coarse_project_texture_f32(" in texture_block
    assert "relion_coarse_score_translate_f32(" in texture_block
    assert "translation_angles[candidate * 2]" in texture_block
    assert "numerator_weight[operand_index]" in texture_block
    assert "half_weights[compact_pixel]" not in texture_block
    assert "packed_pixel += kRelionCoarseDiff2BlockSize" in texture_block
    assert "for (int width = kRelionCoarseDiff2BlockSize / 2" in texture_block
    assert "sqrtf(" in texture_block
    assert "candidate_output[1] = numerator_lanes[0]" in texture_block
    assert "candidate_output[2] = norm_lanes[0]" in texture_block
    assert "atomicAdd(&candidate_output[0], contribution)" in texture_block
    assert "launch_relion_coarse_normalized_cc_native_texture_pairs_f32" in source
    assert "RelionCoarseNormalizedCcNativeTexturePairsF32" in source


def test_relion_fused_coarse_projector_source_pins_vdam_support_and_segmentation():
    root = Path(__file__).resolve().parents[2]
    em_cuda_dir = root / "relax" / "cuda"
    source = read_em_cuda_source()
    block = (em_cuda_dir / "relion_coarse_diff2_projector_body.inc").read_text()

    launcher_start = source.index("launch_relion_coarse_diff2_projector_f32_impl")
    launcher = source[launcher_start : source.index("__global__ void relion_coarse_diff2_initialize_f32_kernel(", launcher_start)]
    assert "shared_rotations[EULERS_PER_BLOCK * 6]" in block
    assert "shared_references[" in block
    assert "shared_images[kRelionCoarseDiff2BlockSize]" in block
    assert "shared_weights[kRelionCoarseDiff2BlockSize]" in block
    assert "threadIdx.x / kRelionCoarsePrefetchFraction" in block
    assert "threadIdx.x % kRelionCoarsePrefetchFraction" in block
    assert "pixel_in_chunk * EULERS_PER_BLOCK + local_rotation" in block
    assert "relion_score_translate_f32(" in block
    assert "tex3D<float>(" in block
    assert "projector_scale * tex3D<float>(" in block
    assert "RELAX_RELION_COARSE_STAGE_WEIGHT(pixel_weight)" in block
    assert "RELAX_RELION_COARSE_DIFF2_UPDATE(" in block
    assert "relion_fine_diff2_update_f32(" not in block
    assert "__fmul_rn(pixel_weight, 0.5f)" not in block
    start = source.index("relion_coarse_diff2_projector_f32_kernel")
    template_start = source.rfind("template <", 0, start)
    default_kernel = source[
        template_start:
        source.index("relion_coarse_diff2_projector_prehalf_f32_kernel", start)
    ]
    prehalf_kernel = source[
        source.index("relion_coarse_diff2_projector_prehalf_f32_kernel", start) :
        source.index("launch_relion_coarse_diff2_projector_f32_variant", start)
    ]
    assert "bool SINGLE_LANE_CANONICAL = false,\n    bool PER_IMAGE_POSES = false>" in default_kernel
    # Per-image poses (subtomogram tilt images) only offset the rotation and phase reads.
    assert "#define PER_IMAGE_POSES false" in prehalf_kernel
    assert "rotations[rotation_base + rotation * 6 + component]" in block
    assert "image_angles[2 * translation]" in block
    assert "PREHALF_WEIGHT" not in default_kernel
    assert "#define RELAX_RELION_COARSE_STAGE_WEIGHT(pixel_weight)\n" in default_kernel
    assert "relion_fine_diff2_update_f32" in default_kernel
    assert "relion_coarse_diff2_projector_body.inc" in default_kernel
    assert "__fmul_rn(pixel_weight, 0.5f)" in prehalf_kernel
    assert "relion_fine_diff2_update_prehalf_f32" in prehalf_kernel
    assert "relion_coarse_diff2_projector_body.inc" in prehalf_kernel
    assert "prehalved coarse weights require native atomic reduction" in source
    assert "const int score_max_r = min(model_max_r, current_size / 2);" in launcher
    assert "(rotation_count / 128) * 128" in launcher
    assert "SINGLE_LANE_CANONICAL" in launcher
    assert "if constexpr (CAPTURE_LANES)" in block
    assert "if constexpr (CANONICAL_REDUCTION)" in block
    assert "shared_lane_partials[" in block
    assert "CANONICAL_REDUCTION && !SINGLE_LANE_CANONICAL" in block
    assert "if constexpr (SINGLE_LANE_CANONICAL)" in block
    assert "translation = static_cast<int>(threadIdx.x);" in block
    assert "active_thread = threadIdx.x < translation_count;" in block
    assert "SINGLE_LANE_CANONICAL ? 1 : active_lanes" in block
    assert "output[output_index] = __fadd_rn(" in block
    assert "threadIdx.x + lane_index * translation_count" in block
    assert "total = __fadd_rn(" in block
    assert "lane_partials[" in block
    assert "kRelionCoarseDiff2BlockSize +" in block



def test_relion_coarse_prehalf_shared_body_is_built_packaged_and_stale_checked():
    # The EM library (librelax_cuda.so) builds from relax/cuda (relax split S4).
    root = Path(__file__).resolve().parents[2]
    include_name = "relion_coarse_diff2_projector_body.inc"
    makefile = (root / "relax" / "cuda" / "Makefile").read_text()
    manifest = (root / "MANIFEST.in").read_text()

    local_inputs = (
        "relax_kernels.cu",
        include_name,
        "noise_residual.cuh",
        "vdam_trace.cuh",
        "relion_preprocess.cuh",
        "relion_vdam_mstep.cuh",
        "relion_scoring.cuh",
        "relion_posterior.cuh",
        "sparse_pass2_posterior.cuh",
        "relion_translate_sum.cuh",
        "relion_capacity_texture.cuh",
        "ppca_moment_backproject.cuh",
        "ppca_stream.cuh",
    )
    public_headers = ("$(RECOVAR_CUDA_INCLUDE)/device_scratch.cuh", "$(RECOVAR_CUDA_INCLUDE)/recovar_cuda_common.cuh")
    library_rule = next(line for line in makefile.splitlines() if line.startswith("$(LIB):"))
    prerequisites, order_only = library_rule.split(":", 1)[1].split("|", 1)
    assert set(prerequisites.split()) == set(local_inputs) | set(public_headers)
    assert order_only.split() == ["check-nvcc"]
    for source_name in local_inputs:
        assert f"include relax/cuda/{source_name}" in manifest

    from recovar.cuda_build import include_dir

    from relax.cuda import kernels as em_cuda_kernels

    assert em_cuda_kernels._RELAX_CUDA_BUILD_SOURCE_NAMES == (
        *local_inputs[:1],
        *local_inputs[2:],
        include_name,
        str(include_dir() / "recovar_cuda_common.cuh"),
        str(include_dir() / "device_scratch.cuh"),
        "Makefile",
    )
    assert em_cuda_kernels._LIBRARY.source_names == em_cuda_kernels._RELAX_CUDA_BUILD_SOURCE_NAMES














@pytest.mark.parametrize("translation_count", [1, 64, 129])
def test_relion_coarse_single_lane_canonical_rejects_unsupported_counts(
    translation_count,
):
    from relax.cuda import kernels as em_cuda_kernels

    with pytest.raises(ValueError, match="requires 65--128 translations"):
        em_cuda_kernels.relion_coarse_diff2_projector_f32.__wrapped__(
            jnp.zeros((5, 5, 5), dtype=jnp.complex64),
            jnp.eye(3, dtype=jnp.float32)[None, :, :],
            jnp.zeros((1, 1), dtype=jnp.complex64),
            jnp.zeros((translation_count, 2), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
            jnp.asarray([0], dtype=jnp.int32),
            current_size=1,
            physical_image_size=1,
            model_max_r=1,
            canonical_reduction=True,
            single_lane_canonical=True,
        )


def test_relion_coarse_single_lane_canonical_requires_canonical_reduction():
    from relax.cuda import kernels as em_cuda_kernels

    with pytest.raises(ValueError, match="requires canonical_reduction=True"):
        em_cuda_kernels.relion_coarse_diff2_projector_f32.__wrapped__(
            jnp.zeros((5, 5, 5), dtype=jnp.complex64),
            jnp.eye(3, dtype=jnp.float32)[None, :, :],
            jnp.zeros((1, 1), dtype=jnp.complex64),
            jnp.zeros((116, 2), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
            jnp.asarray([0], dtype=jnp.int32),
            current_size=1,
            physical_image_size=1,
            model_max_r=1,
            canonical_reduction=False,
            single_lane_canonical=True,
        )


def test_compact_projection_window_positions_map_full_indices_to_compact_rows():
    from relax.scoring.coarse_layout import compact_projection_window_positions

    compact = np.asarray([20, 21, 25, 26, 10, 11], dtype=np.int32)
    window = np.asarray([10, 20, 26, 11], dtype=np.int32)

    assert_matches(
        compact_projection_window_positions(compact, window),
        [4, 0, 3, 5],
    )
    with pytest.raises(ValueError, match="absent from the compact projection"):
        compact_projection_window_positions(compact, [10, 99])


def test_exact_relion_ctf_source_defaults_to_dataset_star(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from relax.relion.relion_ctf import _relion_exact_ctf_source_star

    dataset_star = tmp_path / "particles.star"
    explicit_star = tmp_path / "override.star"
    monkeypatch.delenv("RELAX_K1_RELION_EXACT_CTF_STAR", raising=False)
    assert _relion_exact_ctf_source_star(
        SimpleNamespace(particles_file=str(dataset_star)),
    ) == dataset_star.resolve()

    monkeypatch.setenv("RELAX_K1_RELION_EXACT_CTF_STAR", str(explicit_star))
    assert _relion_exact_ctf_source_star(
        SimpleNamespace(particles_file=str(dataset_star)),
    ) == explicit_star.resolve()

    monkeypatch.delenv("RELAX_K1_RELION_EXACT_CTF_STAR", raising=False)
    with pytest.raises(ValueError, match="STAR-backed dataset"):
        _relion_exact_ctf_source_star(SimpleNamespace(particles_file="particles.mrcs"))


def test_exact_relion_ctf_source_exposes_host_and_shared_device_boundaries(
    monkeypatch,
    tmp_path,
):
    from types import SimpleNamespace

    import pandas as pd

    monkeypatch.setattr(
        relion_ctf,
        "relion_ctf_fftw_half",
        lambda params, *_a, finish, **_k: finish(
            0, len(params), np.broadcast_to(np.arange(12, dtype=np.float64), (len(params), 12)).copy()
        ),
    )

    source = (tmp_path / "particles.star").resolve()
    cache_key = (str(source), (4, 4))
    particle = {
        "rlnOpticsGroup": 1,
        "rlnDefocusU": 10000.0,
        "rlnDefocusV": 11000.0,
        "rlnDefocusAngle": 12.0,
        "rlnPhaseShift": 3.0,
    }
    optics = {
        "rlnVoltage": 300.0,
        "rlnSphericalAberration": 2.7,
        "rlnAmplitudeContrast": 0.1,
        "rlnImagePixelSize": 1.5,
    }
    monkeypatch.setattr(
        relion_ctf,
        "_relion_exact_ctf_source_star",
        lambda _dataset: source,
    )
    monkeypatch.setitem(
        relion_ctf._RELION_EXACT_CTF_SOURCE_CACHE,
        cache_key,
        {
            "particles": pd.DataFrame([particle]),
            "optics": {1: optics},
            "slots": np.asarray([-1], dtype=np.int64),
            "rows": None,
            "n_cached": 0,
        },
    )
    dataset = SimpleNamespace(
        original_image_indices_from_local=lambda indices: np.asarray(indices),
    )

    pixel_indices = np.asarray([11, 0, 4, 4], dtype=np.int32)
    compact_result = relion_ctf._relion_exact_ctf_half_from_source_star_host(
        dataset,
        np.asarray([0, 0], dtype=np.int32),
        (4, 4),
        pixel_indices=pixel_indices,
    )
    host_result = relion_ctf._relion_exact_ctf_half_from_source_star_host(
        dataset,
        np.asarray([0], dtype=np.int32),
        (4, 4),
    )
    device_result = relion_ctf._relion_exact_ctf_half_from_source_star(
        dataset,
        np.asarray([0], dtype=np.int32),
        (4, 4),
    )

    assert type(host_result) is np.ndarray
    assert host_result.dtype == np.float64
    assert isinstance(device_result, jax.Array)
    assert device_result.dtype == jnp.float64
    assert_matches(
        host_result[0],
        -np.fft.fftshift(np.arange(12, dtype=np.float64).reshape(4, 3), axes=0).reshape(-1),
    )
    assert_matches(np.asarray(device_result), host_result)

    assert compact_result.dtype == np.float64
    assert_matches(compact_result, host_result[[0, 0]][:, pixel_indices])
    # Memoized operands are shared, so callers must not be able to corrupt them.
    with pytest.raises(ValueError, match="read-only"):
        compact_result[:] = 99.0
    assert_matches(
        relion_ctf._relion_exact_ctf_half_from_source_star_host(
            dataset,
            np.asarray([0], dtype=np.int32),
            (4, 4),
        ),
        host_result,
    )


@pytest.mark.gpu
def test_relion_coarse_diff2_rectangular_matches_atomic_envelope(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    from relax.cuda import kernels as em_cuda_kernels

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    rng = np.random.default_rng(29)
    batch_size, rotation_count, translation_count = 2, 17, 29
    compact_pixel_count, full_pixel_count = 421, 513
    reference = (
        rng.normal(0, 0.02, (rotation_count, compact_pixel_count))
        + 1j * rng.normal(0, 0.02, (rotation_count, compact_pixel_count))
    ).astype(np.complex64)
    shifted = (
        rng.normal(
            0,
            0.02,
            (batch_size, translation_count, compact_pixel_count),
        )
        + 1j
        * rng.normal(
            0,
            0.02,
            (batch_size, translation_count, compact_pixel_count),
        )
    ).astype(np.complex64)
    weight = rng.uniform(0, 150_000, (batch_size, compact_pixel_count)).astype(
        np.float32
    )
    initial_diff2 = rng.uniform(10_000, 20_000, batch_size).astype(np.float32)
    retained = np.sort(
        rng.choice(full_pixel_count, compact_pixel_count, replace=False)
    )
    lookup = np.full(full_pixel_count, -1, dtype=np.int32)
    lookup[retained] = np.arange(compact_pixel_count, dtype=np.int32)

    with jax.default_device(gpu_device):
        actual = np.asarray(
            em_cuda_kernels.relion_coarse_diff2_rectangular_f32(
                jnp.asarray(reference),
                jnp.asarray(shifted),
                jnp.asarray(weight),
                jnp.asarray(initial_diff2),
                jnp.asarray(lookup),
            )
        )

    for batch in range(batch_size):
        for rotation in range(rotation_count):
            for translation in range(translation_count):
                expected = _coarse_production_result(
                    reference[rotation],
                    shifted[batch, translation],
                    weight[batch],
                    lookup,
                    translation_count=translation_count,
                    initial_diff2=initial_diff2[batch],
                )
                assert_matches(actual[batch, rotation, translation], expected)












@pytest.mark.gpu
def test_relion_fine_diff2_rectangular_matches_production_tree(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    from relax.cuda import kernels as em_cuda_kernels

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    reference, shifted, weight, lookup = _operands()
    initial_diff2 = np.asarray([0.022644043], dtype=np.float32)
    expected = np.add(
        _production_reference(reference, shifted, weight, lookup),
        initial_diff2[0],
        dtype=np.float32,
    )

    with jax.default_device(gpu_device):
        actual = em_cuda_kernels.relion_fine_diff2_rectangular_f32(
            jnp.asarray(reference[None, None, :]),
            jnp.asarray(shifted[None, None, :]),
            jnp.asarray(weight[None, :]),
            jnp.asarray(lookup),
            initial_diff2=jnp.asarray(initial_diff2),
        )

    assert_matches(
        np.asarray(actual),
        np.asarray([[[expected]]], dtype=np.float32),
    )


@pytest.mark.gpu
def test_relion_fine_diff2_pairs_matches_production_tree(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    from relax.cuda import kernels as em_cuda_kernels

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    reference, shifted, weight, lookup = _operands()
    expected = _production_reference(reference, shifted, weight, lookup)

    with jax.default_device(gpu_device):
        actual = em_cuda_kernels.relion_fine_diff2_pairs_f32(
            jnp.asarray(reference[None, None, :]),
            jnp.asarray(shifted[None, None, :]),
            jnp.asarray(weight[None, :]),
            jnp.asarray(lookup),
        )

    assert_matches(
        np.asarray(actual),
        np.asarray([[expected]], dtype=np.float32),
    )


@pytest.mark.gpu
def test_relion_fine_diff2_rectangular_f64_matches_acc_double_tree(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    from relax.cuda import kernels as em_cuda_kernels

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    reference32, shifted32, weight32, lookup = _operands()
    reference = reference32.astype(np.complex128)
    shifted = shifted32.astype(np.complex128)
    weight = weight32.astype(np.float64)
    expected = _production_reference_f64(reference, shifted, weight, lookup)

    with jax.default_device(gpu_device):
        actual = em_cuda_kernels.relion_fine_diff2_rectangular_f64(
            jnp.asarray(reference[None, None, :]),
            jnp.asarray(shifted[None, None, :]),
            jnp.asarray(weight[None, :]),
            jnp.asarray(lookup),
        )

    assert_matches(
        np.asarray(actual),
        np.asarray([[[expected]]], dtype=np.float64),
    )


@pytest.mark.parametrize(
    "function_name,expected_target,expected_shape",
    [
        (
            "relion_fine_diff2_rectangular_f64",
            "cuda_relion_fine_diff2_rectangular_f64",
            (1, 2, 3),
        ),
        (
            "relion_fine_diff2_pairs_f64",
            "cuda_relion_fine_diff2_pairs_f64",
            (1, 2),
        ),
    ],
)
def test_relion_fine_diff2_f64_uses_double_ffi_target(
    monkeypatch, function_name, expected_target, expected_shape
):
    import recovar.cuda_backproject as cuda_backproject

    from relax.cuda import kernels as em_cuda_kernels

    call = {}

    def fake_ffi_call(target, out_type, **options):
        call.update(target=target, out_type=out_type, options=options)
        return lambda *_args: jnp.zeros(out_type.shape, out_type.dtype)

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(cuda_backproject, "custom_cuda_requested", lambda: True)
    monkeypatch.setattr(em_cuda_kernels, "custom_cuda_requested", lambda: True)
    monkeypatch.setattr(cuda_backproject, "_ensure_ffi", lambda: None)
    monkeypatch.setattr(em_cuda_kernels, "_ensure_ffi", lambda: None)
    monkeypatch.setattr(cuda_backproject.jax.ffi, "ffi_call", fake_ffi_call)
    function = getattr(em_cuda_kernels, function_name).__wrapped__
    reference_shape = (1, 2, 5)
    shifted_shape = (1, 3, 5) if "rectangular" in function_name else reference_shape
    actual = function(
        jnp.zeros(reference_shape, dtype=jnp.complex128),
        jnp.zeros(shifted_shape, dtype=jnp.complex128),
        jnp.ones((1, 5), dtype=jnp.float64),
        jnp.arange(5, dtype=jnp.int32),
    )

    assert actual.shape == expected_shape
    assert actual.dtype == jnp.float64
    assert call["target"] == expected_target


@pytest.mark.parametrize(
    "function_name",
    [
        "relion_fine_diff2_rectangular_f32",
        "relion_fine_diff2_pairs_f32",
        "relion_fine_diff2_rectangular_f64",
        "relion_fine_diff2_pairs_f64",
    ],
)
def test_relion_fine_diff2_fails_closed_without_gpu(monkeypatch, function_name):
    import recovar.cuda_backproject as cuda_backproject

    from relax.cuda import kernels as em_cuda_kernels

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    function = getattr(em_cuda_kernels, function_name).__wrapped__
    is_f64 = function_name.endswith("f64")
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        function(
            jnp.zeros((1, 1, 2), dtype=jnp.complex128 if is_f64 else jnp.complex64),
            jnp.zeros((1, 1, 2), dtype=jnp.complex128 if is_f64 else jnp.complex64),
            jnp.ones((1, 2), dtype=jnp.float64 if is_f64 else jnp.float32),
            jnp.asarray([0, 1], dtype=jnp.int32),
        )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_relion_coarse_diff2_fails_closed_without_gpu(monkeypatch, dtype):
    import recovar.cuda_backproject as cuda_backproject

    from relax.cuda import kernels as em_cuda_kernels

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    is_f64 = dtype == jnp.float64
    function = (
        em_cuda_kernels.relion_coarse_diff2_rectangular_f64
        if is_f64
        else em_cuda_kernels.relion_coarse_diff2_rectangular_f32
    )
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        function.__wrapped__(
            jnp.zeros((1, 2), dtype=jnp.complex128 if is_f64 else jnp.complex64),
            jnp.zeros((1, 29, 2), dtype=jnp.complex128 if is_f64 else jnp.complex64),
            jnp.ones((1, 2), dtype=dtype),
            jnp.zeros((1,), dtype=dtype),
            jnp.asarray([0, 1], dtype=jnp.int32),
        )


def test_relion_coarse_normalized_cc_fails_closed_without_gpu(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject

    from relax.cuda import kernels as em_cuda_kernels

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="require a JAX GPU backend"):
        em_cuda_kernels.relion_coarse_normalized_cc_pairs_f32.__wrapped__(
            jnp.zeros((1, 2, 3), dtype=jnp.complex64),
            jnp.ones((1, 2, 3), dtype=jnp.float32),
            jnp.zeros((1, 2, 3), dtype=jnp.complex64),
            jnp.ones((3,), dtype=jnp.float32),
            jnp.arange(3, dtype=jnp.int32),
        )


def test_relion_coarse_normalized_cc_native_texture_fails_closed_without_gpu(
    monkeypatch,
):
    import recovar.cuda_backproject as cuda_backproject

    from relax.cuda import kernels as em_cuda_kernels

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="require a JAX GPU backend"):
        em_cuda_kernels.relion_coarse_normalized_cc_native_texture_pairs_f32.__wrapped__(
            jnp.zeros((5, 5, 5), dtype=jnp.complex64),
            jnp.eye(3, dtype=jnp.float32)[None, :, :],
            jnp.zeros((1, 1), dtype=jnp.complex64),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
            jnp.asarray([0], dtype=jnp.int32),
            1,
            1,
            1,
        )


def test_relion_projector_half_texture_fails_closed_without_gpu(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject

    from relax.cuda import kernels as em_cuda_kernels

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        em_cuda_kernels.relion_projector_half_texture_f32.__wrapped__(
            jnp.zeros((5, 5, 3), dtype=jnp.complex64),
            jnp.eye(3, dtype=jnp.float32)[None, :, :],
            current_size=2,
            padding_factor=1,
            projector_max_r=1,
        )


def test_relion_coarse_vdam_projector_fails_closed_without_gpu(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject

    from relax.cuda import kernels as em_cuda_kernels

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        em_cuda_kernels.relion_coarse_diff2_projector_f32.__wrapped__(
            jnp.zeros((5, 5, 5), dtype=jnp.complex64),
            jnp.eye(3, dtype=jnp.float32)[None, :, :],
            jnp.zeros((1, 1), dtype=jnp.complex64),
            jnp.zeros((1, 2), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
            jnp.asarray([0], dtype=jnp.int32),
            current_size=1,
            physical_image_size=1,
            model_max_r=1,
        )




@pytest.mark.parametrize(
    "reference_shape,shifted_shape,weight_shape,expected_route,expected_shape",
    [
        ((2, 3, 1, 7), (2, 1, 4, 7), (2, 1, 1, 7), "rectangular", (2, 3, 4)),
        ((3, 1, 7), (1, 4, 7), (1, 1, 7), "rectangular", (3, 4)),
        ((2, 5, 7), (2, 5, 7), (2, 1, 7), "pairs", (2, 5)),
    ],
)
def test_sparse_pass2_fused_flag_routes_supported_operand_layouts(
    monkeypatch,
    reference_shape,
    shifted_shape,
    weight_shape,
    expected_route,
    expected_shape,
):
    from relax.cuda import kernels as em_cuda_kernels
    from relax.sparse_pass2.sparse_pass2_scoring import _relion_cuda_fine_diff2_sum

    routes = []

    def rectangular(reference, shifted_image, weight, full_to_compact):
        routes.append(
            (
                "rectangular",
                reference.shape,
                shifted_image.shape,
                weight.shape,
                full_to_compact.shape,
            )
        )
        return jnp.zeros(
            (reference.shape[0], reference.shape[1], shifted_image.shape[1]),
            dtype=jnp.float32,
        )

    def pairs(reference, shifted_image, weight, full_to_compact):
        routes.append(
            (
                "pairs",
                reference.shape,
                shifted_image.shape,
                weight.shape,
                full_to_compact.shape,
            )
        )
        return jnp.zeros(reference.shape[:2], dtype=jnp.float32)

    monkeypatch.setenv("RELAX_RELION_FINE_DIFF2_FUSED_FFI", "1")
    monkeypatch.setattr(
        em_cuda_kernels,
        "relion_fine_diff2_rectangular_f32",
        rectangular,
    )
    monkeypatch.setattr(em_cuda_kernels, "relion_fine_diff2_pairs_f32", pairs)

    actual = _relion_cuda_fine_diff2_sum(
        jnp.zeros(reference_shape, dtype=jnp.complex64),
        jnp.zeros(shifted_shape, dtype=jnp.complex64),
        jnp.ones(weight_shape, dtype=jnp.float32),
        jnp.arange(7, dtype=jnp.int32),
    )

    assert actual.shape == expected_shape
    assert routes[0][0] == expected_route
    assert routes[0][-1] == (7,)


def test_sparse_pass2_fused_flag_routes_float64_to_f64_ffi(monkeypatch):
    from relax.cuda import kernels as em_cuda_kernels
    from relax.sparse_pass2.sparse_pass2_scoring import _relion_cuda_fine_diff2_sum

    calls = []

    def rectangular(reference, shifted_image, weight, full_to_compact):
        calls.append((reference.dtype, shifted_image.dtype, weight.dtype))
        return jnp.zeros(
            (reference.shape[0], reference.shape[1], shifted_image.shape[1]),
            dtype=jnp.float64,
        )

    monkeypatch.setenv("RELAX_RELION_FINE_DIFF2_FUSED_FFI", "1")
    monkeypatch.setattr(
        em_cuda_kernels,
        "relion_fine_diff2_rectangular_f64",
        rectangular,
    )
    actual = _relion_cuda_fine_diff2_sum(
        jnp.zeros((2, 3, 1, 7), dtype=jnp.complex128),
        jnp.zeros((2, 1, 4, 7), dtype=jnp.complex128),
        jnp.ones((2, 1, 1, 7), dtype=jnp.float64),
        jnp.arange(7, dtype=jnp.int32),
    )

    assert actual.shape == (2, 3, 4)
    assert actual.dtype == jnp.float64
    assert calls == [(jnp.complex128, jnp.complex128, jnp.float64)]


def test_relion_runtime_cutoff_fine_diff2_fails_closed_without_gpu(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject

    from relax.cuda import kernels as em_cuda_kernels

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        em_cuda_kernels.relion_fine_diff2_fused_translate_runtime_rectangular_f32.__wrapped__(
            jnp.zeros((1, 1, 1), dtype=jnp.complex64),
            jnp.zeros((1, 1), dtype=jnp.complex64),
            jnp.zeros((1, 2), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.asarray([0], dtype=jnp.int32),
            jnp.asarray(2, dtype=jnp.int32),
        )


def test_relion_runtime_flat_rows_fails_closed_without_gpu(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject

    from relax.cuda import kernels as em_cuda_kernels

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        em_cuda_kernels.relion_fine_diff2_fused_translate_runtime_flat_rows_f32.__wrapped__(
            jnp.zeros((1, 1), dtype=jnp.complex64),
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.zeros((1, 1), dtype=jnp.complex64),
            jnp.zeros((1, 2), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.asarray([0], dtype=jnp.int32),
            jnp.asarray(2, dtype=jnp.int32),
        )


@pytest.mark.parametrize(
    "bad_indices",
    [
        np.asarray([-1], dtype=np.int32),
        np.asarray([12], dtype=np.int32),
        np.asarray([1.5], dtype=np.float64),
        np.asarray([[1]], dtype=np.int32),
        np.asarray([True], dtype=bool),
    ],
)
def test_exact_ctf_compact_indices_reject_invalid_host_geometry(monkeypatch, tmp_path, bad_indices):
    from types import SimpleNamespace

    source = (tmp_path / "particles.star").resolve()
    monkeypatch.setattr(relion_ctf, "_relion_exact_ctf_source_star", lambda _: source)
    monkeypatch.setitem(
        relion_ctf._RELION_EXACT_CTF_SOURCE_CACHE,
        (str(source), (4, 4)),
        {
            "slots": np.asarray([0], dtype=np.int64),
            "rows": np.ones((1, 12), dtype=np.float64),
            "n_cached": 1,
        },
    )
    dataset = SimpleNamespace(original_image_indices_from_local=lambda indices: indices)
    with pytest.raises(ValueError):
        relion_ctf._relion_exact_ctf_half_from_source_star_host(
            dataset,
            np.asarray([0]),
            (4, 4),
            pixel_indices=bad_indices,
        )


def test_exact_ctf_compact_indices_never_materialize_device_inputs(monkeypatch, tmp_path):
    from types import SimpleNamespace

    class DeviceOnly:
        def __array__(self, *args, **kwargs):
            raise AssertionError("Unexpected device-to-host materialization")

    source = (tmp_path / "particles.star").resolve()
    monkeypatch.setattr(relion_ctf, "_relion_exact_ctf_source_star", lambda _: source)
    monkeypatch.setitem(
        relion_ctf._RELION_EXACT_CTF_SOURCE_CACHE,
        (str(source), (4, 4)),
        {
            "slots": np.asarray([0], dtype=np.int64),
            "rows": np.ones((1, 12), dtype=np.float64),
            "n_cached": 1,
        },
    )
    dataset = SimpleNamespace(original_image_indices_from_local=lambda indices: indices)
    for indices in (DeviceOnly(), jnp.asarray([0], dtype=jnp.int32)):
        with pytest.raises(TypeError, match="host NumPy array"):
            relion_ctf._relion_exact_ctf_half_from_source_star_host(
                dataset,
                np.asarray([0]),
                (4, 4),
                pixel_indices=indices,
            )


@pytest.mark.gpu
def test_relion_half_texture_projection_matches_legacy_full_staging(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    """The compact production projector must preserve every projected value."""

    import recovar.cuda_backproject as cuda_backproject

    from relax.cuda import kernels as em_cuda_kernels
    from relax.helpers.projection import (
        compute_relion_projector_projections_block,
        relion_projector_half_to_texture_full,
    )

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    rng = np.random.default_rng(191)
    current_size = 16
    padding_factor = 2
    projector_max_r = 7
    projector_size = 31
    projector_half = (
        rng.normal(0, 0.02, (projector_size, projector_size, 16))
        + 1j * rng.normal(0, 0.02, (projector_size, projector_size, 16))
    ).astype(np.complex64)
    rotations = _off_grid_so3_rotations()
    image_coordinates = np.stack(
        np.meshgrid(
            np.arange(-current_size // 2 + 1, current_size // 2 + 1),
            np.arange(current_size // 2 + 1),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 2)
    model_coordinates = np.einsum(
        "rij,pj->rpi",
        rotations[:, :, :2],
        image_coordinates[:, ::-1],
    ) * np.float32(padding_factor)
    assert np.any(model_coordinates[..., 0] < 0)
    assert np.any(model_coordinates[..., 0] > 0)
    assert np.any(np.abs(model_coordinates[..., 2]) > 0.25)
    assert np.any(np.abs(model_coordinates[..., 2] - np.rint(model_coordinates[..., 2])) > 0.05)
    assert np.any(np.sum(model_coordinates * model_coordinates, axis=-1) > (projector_max_r * padding_factor) ** 2)

    with jax.default_device(gpu_device):
        projector_half_jax = jnp.asarray(projector_half)
        rotations_jax = jnp.asarray(rotations)
        projector_full = relion_projector_half_to_texture_full(projector_half_jax)
        legacy = cuda_backproject.project(
            projector_full.reshape(-1),
            rotations_jax,
            image_shape=(current_size, current_size),
            volume_shape=(projector_size,) * 3,
            order=1,
            half_volume=False,
            half_image=True,
            max_r=float(projector_max_r),
            relion_texture_interp=True,
        )
        compact = em_cuda_kernels.relion_projector_half_texture_f32(
            projector_half_jax,
            rotations_jax,
            current_size=current_size,
            padding_factor=padding_factor,
            projector_max_r=projector_max_r,
        )
        native_scale = np.float32(-(current_size**2))
        legacy_native_scaled = cuda_backproject.project(
            (projector_full * native_scale).reshape(-1),
            rotations_jax,
            image_shape=(current_size, current_size),
            volume_shape=(projector_size,) * 3,
            order=1,
            half_volume=False,
            half_image=True,
            max_r=float(projector_max_r),
            relion_texture_interp=True,
        )
        compact_native_scaled = em_cuda_kernels.relion_projector_half_texture_f32(
            projector_half_jax,
            rotations_jax,
            current_size=current_size,
            padding_factor=padding_factor,
            projector_max_r=projector_max_r,
            projector_scale=float(native_scale),
        )
        production, production_abs2 = compute_relion_projector_projections_block(
            projector_half_jax,
            rotations_jax,
            (current_size, current_size),
            r_max=projector_max_r,
            padding_factor=padding_factor,
            centered_rows=True,
            dense_scale=True,
            projector_output_size=current_size,
            relion_texture_interp=True,
        )
        legacy_scaled = legacy * np.float32(-(current_size**2))
        legacy_abs2 = jnp.abs(legacy_scaled) ** 2

    assert_matches(
        np.asarray(compact),
        np.asarray(legacy),
    )
    assert_matches(
        np.asarray(compact_native_scaled),
        np.asarray(legacy_native_scaled),
    )
    assert_matches(
        np.asarray(production),
        np.asarray(legacy_scaled),
    )
    assert_matches(
        np.asarray(production_abs2),
        np.asarray(legacy_abs2),
    )


@pytest.mark.gpu
def test_relion_half_texture_full_even_indexed_projection_matches_full_scatter(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    """The full-box Nyquist alias must survive compact indexed projection."""

    import recovar.cuda_backproject as cuda_backproject

    from relax.helpers.projection import (
        compute_relion_projector_projections_block,
    )

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    image_size = 16
    padding_factor = 1
    projector_max_r = image_size // 2
    padded_r_max = projector_max_r * padding_factor
    projector_size = 2 * (padded_r_max + 1) + 1
    rng = np.random.default_rng(194)
    projector = (
        rng.normal(0, 0.02, (projector_size, projector_size, padded_r_max + 2))
        + 1j
        * rng.normal(0, 0.02, (projector_size, projector_size, padded_r_max + 2))
    ).astype(np.complex64)
    rotations = _off_grid_so3_rotations()
    pixel_indices = np.arange(
        image_size * (image_size // 2 + 1),
        dtype=np.int32,
    )

    common = dict(
        image_shape=(image_size, image_size),
        r_max=projector_max_r,
        padding_factor=padding_factor,
        return_abs2=True,
        centered_rows=True,
        dense_scale=True,
        projector_output_size=image_size,
        relion_texture_interp=True,
    )
    with jax.default_device(gpu_device):
        full_projection, full_abs2 = compute_relion_projector_projections_block(
            jnp.asarray(projector),
            jnp.asarray(rotations),
            **common,
        )
        indexed_projection, indexed_abs2 = compute_relion_projector_projections_block(
            jnp.asarray(projector),
            jnp.asarray(rotations),
            pixel_indices=pixel_indices,
            **common,
        )

    assert_matches(
        np.asarray(indexed_projection),
        np.asarray(full_projection),
    )
    assert_matches(
        np.asarray(indexed_abs2),
        np.asarray(full_abs2),
    )


def _off_grid_so3_rotations() -> np.ndarray:
    """Deterministic proper rotations with all three model axes active."""

    matrices = []
    for angle_x, angle_y, angle_z in (
        (0.37, -0.52, 0.19),
        (-0.91, 0.43, 1.17),
        (1.20, -0.73, -0.44),
        (-0.28, -1.01, 0.66),
    ):
        cosine_x, sine_x = np.cos(angle_x), np.sin(angle_x)
        cosine_y, sine_y = np.cos(angle_y), np.sin(angle_y)
        cosine_z, sine_z = np.cos(angle_z), np.sin(angle_z)
        rotation_x = np.asarray(
            [[1, 0, 0], [0, cosine_x, -sine_x], [0, sine_x, cosine_x]],
            dtype=np.float64,
        )
        rotation_y = np.asarray(
            [[cosine_y, 0, sine_y], [0, 1, 0], [-sine_y, 0, cosine_y]],
            dtype=np.float64,
        )
        rotation_z = np.asarray(
            [[cosine_z, -sine_z, 0], [sine_z, cosine_z, 0], [0, 0, 1]],
            dtype=np.float64,
        )
        matrices.append(np.asarray(rotation_z @ rotation_y @ rotation_x, np.float32))
    return np.stack(matrices)


@pytest.mark.gpu
def test_relion_half_texture_projection_uses_native_rotated_image_radius_cutoff(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    """The rounded outer shell must use RELION's float32/int cutoff."""

    import recovar.cuda_backproject as cuda_backproject

    from relax.cuda import kernels as em_cuda_kernels

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    projector_max_r = 23
    projector_size = 2 * projector_max_r + 3
    projector = np.ones(
        (projector_size, projector_size, projector_max_r + 2),
        dtype=np.complex64,
    )
    # These two valid float32 rotations straddle RELION's integer-truncated
    # cutoff for source (ky, kx)=(10, 1), whose exact radius squared is 101.
    # They are frozen from the admitted EMPIAR-10076 K=4 native operand panel.
    rotations = np.asarray(
        [
            [
                [-0.60339195, -0.20407803, -0.7708893],
                [0.3038262, -0.952619, 0.014376025],
                [-0.7372976, -0.22554199, 0.6368069],
            ],
            [
                [0.3366111, 0.59114784, -0.73296463],
                [-0.88786924, 0.45852897, -0.037939373],
                [0.31365776, 0.6635476, 0.6792079],
            ],
        ],
        dtype=np.float32,
    )

    with jax.default_device(gpu_device):
        projected = em_cuda_kernels.relion_projector_half_texture_f32(
            jnp.asarray(projector),
            jnp.asarray(rotations),
            current_size=20,
            padding_factor=1,
            projector_max_r=projector_max_r,
        )
    projected = np.asarray(projected).reshape(2, 20, 11)
    assert projected[0, 0, 1] != 0
    assert projected[1, 0, 1] == 0


@pytest.mark.gpu
def test_relion_half_texture_projection_is_invariant_to_host_support_crop(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    """Compacting PPref to every consumed square pixel must preserve values."""

    import recovar.cuda_backproject as cuda_backproject

    from relax.helpers.fourier_window import (
        make_fourier_window_spec,
    )
    from relax.helpers.projection import (
        compact_relion_projector_half_for_centered_indices,
        compute_relion_projector_projections_block,
    )

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    image_shape = (16, 16)
    current_size = 6
    padding_factor = 2
    projector_r_max = 7
    padded_r_max = projector_r_max * padding_factor
    projector_size = 2 * (padded_r_max + 1) + 1
    rng = np.random.default_rng(193)
    projector = (
        rng.normal(0, 0.02, (projector_size, projector_size, padded_r_max + 2))
        + 1j * rng.normal(0, 0.02, (projector_size, projector_size, padded_r_max + 2))
    ).astype(np.complex64)
    window = make_fourier_window_spec(
        image_shape,
        current_size,
        image_shape[0] * (image_shape[1] // 2 + 1),
        include_recon_window=False,
        score_square=True,
        score_include_dc=True,
    )
    compact, compact_r_max = compact_relion_projector_half_for_centered_indices(
        projector,
        window.score_indices_np,
        image_shape,
        r_max=projector_r_max,
        padding_factor=padding_factor,
    )
    assert compact_r_max == 5
    assert compact.shape == (23, 23, 12)
    rotations = _off_grid_so3_rotations()

    common = dict(
        image_shape=image_shape,
        padding_factor=padding_factor,
        centered_rows=True,
        dense_scale=True,
        projector_output_size=current_size,
        pixel_indices=window.score_indices_np,
        relion_texture_interp=True,
    )
    with jax.default_device(gpu_device):
        full_projection, full_abs2 = compute_relion_projector_projections_block(
            jnp.asarray(projector),
            jnp.asarray(rotations),
            r_max=projector_r_max,
            **common,
        )
        compact_projection, compact_abs2 = compute_relion_projector_projections_block(
            jnp.asarray(compact),
            jnp.asarray(rotations),
            r_max=compact_r_max,
            **common,
        )

    assert_matches(
        np.asarray(compact_projection),
        np.asarray(full_projection),
    )
    assert_matches(
        np.asarray(compact_abs2),
        np.asarray(full_abs2),
    )


def test_translation_chunk_live_marks_chunks_of_four_with_a_candidate():
    from relax.cuda import kernels as em_cuda_kernels

    mask = np.zeros((3, 10), dtype=bool)
    mask[0, 1] = True
    mask[1, 9] = True
    mask[2, [4, 7]] = True
    live = np.asarray(em_cuda_kernels.relion_fine_diff2_translation_chunk_live(jnp.asarray(mask)))
    assert live.shape == (3, 3)
    assert live.tolist() == [[True, False, False], [False, False, True], [False, True, False]]


@pytest.mark.gpu
def test_relion_runtime_masked_flat_rows_skip_chunks_without_candidates(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    """Live translation chunks score as the unmasked kernel does; skipped ones are +inf."""

    import recovar.cuda_backproject as cuda_backproject

    from relax.cuda import kernels as em_cuda_kernels

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    rng = np.random.default_rng(1933)
    size = 32
    pixels = size * (size // 2 + 1)
    n_rows, n_translations = 6, 10
    row_image_ids = np.asarray([0, 0, 1, 1, -1, 0], dtype=np.int32)
    reference = (rng.normal(0, 0.02, (n_rows, pixels)) + 1j * rng.normal(0, 0.02, (n_rows, pixels))).astype(np.complex64)
    image = (rng.normal(0, 0.02, (2, pixels)) + 1j * rng.normal(0, 0.02, (2, pixels))).astype(np.complex64)
    weight = rng.uniform(0, 150_000, (2, pixels)).astype(np.float32)
    translation_angles = rng.normal(0, 0.2, (n_translations, 2)).astype(np.float32)
    lookup = np.arange(pixels, dtype=np.int32)
    initial_diff2 = np.asarray([0.022644043, 0.03125], dtype=np.float32)
    candidate = rng.random((n_rows, n_translations)) < 0.2
    candidate[1] = False
    live = em_cuda_kernels.relion_fine_diff2_translation_chunk_live(jnp.asarray(candidate))

    with jax.default_device(gpu_device):
        args = (
            jnp.asarray(reference),
            jnp.asarray(row_image_ids),
            jnp.asarray(image),
            jnp.asarray(translation_angles),
            jnp.asarray(weight),
            jnp.asarray(lookup),
            jnp.asarray(size, dtype=jnp.int32),
            jnp.asarray(initial_diff2),
        )
        full = em_cuda_kernels.relion_fine_diff2_fused_translate_runtime_flat_rows_f32(*args)
        masked = em_cuda_kernels.relion_fine_diff2_fused_translate_runtime_flat_rows_f32(
            *args, translation_chunk_live=live
        )
        full, masked = (np.asarray(v) for v in jax.block_until_ready((full, masked)))

    live_cells = np.repeat(np.asarray(live), 4, axis=1)[:, :n_translations]
    assert live_cells.any() and not live_cells.all()
    assert_matches(masked[live_cells], full[live_cells])
    assert np.all(np.isposinf(masked[~live_cells]))
