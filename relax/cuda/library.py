"""The EM CUDA library, librelax_cuda.so: its build inputs, FFI target names, registration and handle.

This module is the one owner of the library's module-level state: the ``NativeLibrary`` (loaded handle and
registration flags), its lock, every ``TARGET_*`` XLA FFI target name and the registration tables built from
them. The kernel families (``relax.cuda.preprocess``, ``coarse``, ``fine`` and the rest) reach it as
``library.<name>``: each wrapper calls ``library.ensure_ffi()`` (or ``ensure_optional_ffi(target)``) before
its first ``jax.ffi.ffi_call``, which registers every target of both libraries once, under the library lock,
whichever family runs first. Importing a family registers nothing.
"""

from __future__ import annotations

import pathlib

from recovar import cuda_backproject
from recovar.cuda_build import NativeLibrary, include_dir

# XLA FFI target names; the registration tables below map each to its handler symbol.
TARGET_PROJECT_RELION_HALF_RUNTIME = "cuda_project_relion_half_runtime"
TARGET_PROJECT_RELION_HALF_IMAGE_RADIUS = "cuda_project_relion_half_image_radius"
TARGET_PROJECT_RELION_HALF_CAPACITY_TEXTURE = "cuda_project_relion_half_capacity_texture"
TARGET_PROJECT_RELION_HALF_CAPACITY_TEXTURE_COMPACT_PACKED = "cuda_project_relion_half_capacity_texture_compact_packed"
TARGET_RELION_PROJECTOR_HALF_TEXTURE_F32 = "cuda_relion_projector_half_texture_f32"
TARGET_RELION_PROJECTOR_PERSISTENT_HALF_TEXTURE_F32 = "cuda_relion_projector_persistent_half_texture_f32"
TARGET_RELION_FUSED_X_HALF_BP = "cuda_relion_fused_x_half_bp"
TARGET_RELION_FUSED_X_HALF_BP_PARTICLE_GRID = (
    "cuda_relion_fused_x_half_bp_particle_grid"
)
TARGET_RELION_FUSED_X_HALF_BP_SIGNATURE = "cuda_relion_fused_x_half_bp_signature"
TARGET_RELION_POINT_GROUP_SYMMETRISE_BPREF_COMPLEX_RANGE = (
    "cuda_relion_point_group_symmetrise_bpref_complex_range"
)
TARGET_RELION_POINT_GROUP_SYMMETRISE_BPREF_SPLIT_RANGE = (
    "cuda_relion_point_group_symmetrise_bpref_split_range"
)
TARGET_RELION_POINT_GROUP_SYMMETRISE_BPREF = (
    "cuda_relion_point_group_symmetrise_bpref"
)
TARGET_RELION_PREPROCESS_REAL_F32 = "cuda_relion_preprocess_real_f32"
TARGET_RELION_PREPROCESS_REAL_F32_NATIVE_LANE = (
    "cuda_relion_preprocess_real_f32_native_lane"
)
TARGET_RELION_PREPROCESS_REAL_F32_NATIVE_ATOMIC = (
    "cuda_relion_preprocess_real_f32_native_atomic"
)
TARGET_RELION_MAKE_SCORING_ROTATIONS_F32 = "cuda_relion_make_scoring_rotations_f32"
TARGET_RELION_MAKE_SCORING_ROTATIONS_LEFT_F32 = "cuda_relion_make_scoring_rotations_left_f32"
TARGET_RELION_MAKE_SCORING_ROTATIONS_F64 = "cuda_relion_make_scoring_rotations_f64"
TARGET_RELION_TRANSLATE_SCORE_F32 = "cuda_relion_translate_score_f32"
TARGET_RELION_TRANSLATE_SCORE_F64 = "cuda_relion_translate_score_f64"
TARGET_RELION_TRANSLATE_BPREF_F32 = "cuda_relion_translate_bpref_f32"
TARGET_RELION_TRANSLATE_BPREF_F64 = "cuda_relion_translate_bpref_f64"
TARGET_BPREF_PARTICLE_PACK = "cuda_bpref_particle_pack"
TARGET_DEFERRED_VDAM_HOST_PACK = "cuda_deferred_vdam_host_pack"
TARGET_NOISE_PIXEL_PACK = "cuda_noise_pixel_pack"
TARGET_NOISE_RESIDUAL_STATISTICS = "recovar_noise_residual_statistics"
TARGET_SPARSE_PASS2_LOG_Z_F64 = "recovar_sparse_pass2_log_z_f64"
TARGET_SPARSE_PASS2_POSTERIOR_F32 = "recovar_sparse_pass2_posterior_f32"
TARGET_SPARSE_PASS2_SEGMENTED_LOG_Z_F64 = (
    "cuda_sparse_pass2_segmented_log_z_f64"
)
TARGET_SPARSE_PASS2_SEGMENTED_POSTERIOR_F32 = (
    "cuda_sparse_pass2_segmented_posterior_f32"
)
TARGET_SPARSE_PASS2_SEGMENTED_RECONSTRUCTION_F32 = (
    "cuda_sparse_pass2_segmented_reconstruction_f32"
)
TARGET_RELION_VDAM_MSTEP_SUMS_F32 = "cuda_relion_vdam_mstep_sums_f32"
TARGET_RELION_VDAM_MSTEP_DENOMINATOR_F32 = (
    "cuda_relion_vdam_mstep_denominator_f32"
)
TARGET_RELION_VDAM_MSTEP_FUSED_X_HALF = "cuda_relion_vdam_mstep_fused_x_half"
TARGET_RELION_VDAM_MSTEP_FUSED_PROJECTOR_X_HALF = (
    "cuda_relion_vdam_mstep_fused_projector_x_half"
)
TARGET_RELION_VDAM_MSTEP_FUSED_PROJECTOR_RUNTIME_X_HALF = (
    "cuda_relion_vdam_mstep_fused_projector_runtime_x_half"
)
TARGET_RELION_COARSE_DIFF2_RECTANGULAR_F32 = (
    "cuda_relion_coarse_diff2_rectangular_f32"
)
TARGET_RELION_COARSE_DIFF2_PROJECTOR_F32 = (
    "cuda_relion_coarse_diff2_projector_f32"
)
TARGET_RELION_COARSE_DIFF2_PROJECTOR_PER_IMAGE_F32 = "cuda_relion_coarse_diff2_projector_per_image_f32"
TARGET_RELION_COARSE_NORMALIZED_CC_PAIRS_F32 = (
    "cuda_relion_coarse_normalized_cc_pairs_f32"
)
TARGET_RELION_COARSE_NORMALIZED_CC_NATIVE_TEXTURE_PAIRS_F32 = (
    "cuda_relion_coarse_normalized_cc_native_texture_pairs_f32"
)
TARGET_RELION_FINE_DIFF2_RECTANGULAR_F32 = (
    "cuda_relion_fine_diff2_rectangular_f32"
)
TARGET_RELION_FINE_DIFF2_RECTANGULAR_MASKED_F32 = (
    "cuda_relion_fine_diff2_rectangular_masked_f32"
)
TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_FLAT_ROWS_F32 = (
    "cuda_relion_fine_diff2_fused_translate_flat_rows_f32"
)
TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_RUNTIME_FLAT_ROWS_F32 = (
    "cuda_relion_fine_diff2_fused_translate_runtime_flat_rows_f32"
)
TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_RUNTIME_MASKED_FLAT_ROWS_F32 = (
    "cuda_relion_fine_diff2_fused_translate_runtime_masked_flat_rows_f32"
)
TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_PAIRS_F32 = (
    "cuda_relion_fine_diff2_fused_translate_pairs_f32"
)
TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_RUNTIME_PAIRS_F32 = (
    "cuda_relion_fine_diff2_fused_translate_runtime_pairs_f32"
)
TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_JOBS_F32 = (
    "cuda_relion_fine_diff2_fused_translate_jobs_f32"
)
TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_RUNTIME_JOBS_F32 = (
    "cuda_relion_fine_diff2_fused_translate_runtime_jobs_f32"
)
TARGET_RELION_FINE_DIFF2_PAIRS_F32 = "cuda_relion_fine_diff2_pairs_f32"
TARGET_RELION_POWERCLASS_SPECTRUM_HIGHRES_F32 = (
    "cuda_relion_powerclass_spectrum_highres_f32"
)
TARGET_RELION_POWERCLASS_SPECTRUM_HIGHRES_RUNTIME_F32 = (
    "cuda_relion_powerclass_spectrum_highres_runtime_f32"
)
TARGET_RELION_EXPONENTIATE_F32 = "cuda_relion_exponentiate_f32"
TARGET_RELION_EXPONENTIATE_BATCHED_F32 = "cuda_relion_exponentiate_batched_f32"
TARGET_RELION_DIVIDE_F32 = "cuda_relion_divide_f32"
TARGET_RELION_FINE_DIFF2_RECTANGULAR_F64 = (
    "cuda_relion_fine_diff2_rectangular_f64"
)
TARGET_RELION_FINE_DIFF2_PAIRS_F64 = "cuda_relion_fine_diff2_pairs_f64"
TARGET_RELION_DIVIDE_BATCHED_F32 = "cuda_relion_divide_batched_f32"
TARGET_RELION_CUB_SORT_SCAN_F32 = "cuda_relion_cub_sort_scan_f32"
TARGET_RELION_CUB_SORT_SCAN_BATCHED_F32 = "cuda_relion_cub_sort_scan_batched_f32"
TARGET_RELION_COARSE_CUT_F32 = "cuda_relion_coarse_cut_f32"
TARGET_RELION_WAVG_ROTATION_ATOMIC_TRIPLET_ADD_F32 = (
    "cuda_relion_wavg_rotation_atomic_triplet_add_f32"
)
TARGET_RELION_WAVG_ROTATION_ATOMIC_RUNTIME_TRIPLET_ADD_F32 = (
    "cuda_relion_wavg_rotation_atomic_runtime_triplet_add_f32"
)
TARGET_RELION_WAVG_SEQUENTIAL_TRIPLET_F32 = (
    "cuda_relion_wavg_sequential_triplet_f32"
)
TARGET_RELION_WAVG_SEQUENTIAL_RUNTIME_TRIPLET_F32 = (
    "cuda_relion_wavg_sequential_runtime_triplet_f32"
)
TARGET_RELION_WAVG_SEQUENTIAL_RUNTIME_FLAT_ROWS_TRIPLET_F32 = (
    "cuda_relion_wavg_sequential_runtime_flat_rows_triplet_f32"
)
TARGET_RELION_WAVG_ROTATION_ATOMIC_RUNTIME_FLAT_ROWS_TRIPLET_ADD_F32 = (
    "cuda_relion_wavg_rotation_atomic_runtime_flat_rows_triplet_add_f32"
)
TARGET_RELION_WAVG_EXACT_ATOMIC_FLAT_ROWS_TRIPLET_ADD_F32 = (
    "cuda_relion_wavg_exact_atomic_flat_rows_triplet_add_f32"
)
TARGET_RELION_TRANSLATE_SUM_FLAT_ROWS_F32 = (
    "cuda_relion_translate_sum_flat_rows_f32"
)
TARGET_PPCA_MOMENT_SCATTER_F32 = "relax_ppca_moment_scatter_f32"
TARGET_PPCA_WINDOW_PROJECT_F32 = "relax_ppca_window_project_f32"
TARGET_PPCA_LATENT_EPILOGUE_F32 = "relax_ppca_latent_epilogue_f32"
TARGET_PPCA_POSTERIOR_PREP_F32 = "relax_ppca_posterior_prep_f32"
TARGET_DUAL_WEIGHTED_SUMS_PAIRS_F32 = "cuda_dual_weighted_sums_pairs_f32"
TARGET_DUAL_WEIGHTED_SUMS_PAIRS_ROWS_F32 = "cuda_dual_weighted_sums_pairs_rows_f32"
TARGET_RELION_VDAM_MSTEP_FUSED_PROJECTOR_CAPACITY_X_HALF = (
    "cuda_relion_vdam_mstep_fused_projector_capacity_x_half"
)


_RELAX_CUDA_LIB_ENV = "RELAX_CUDA_LIB"


_RELAX_CUDA_BUILD_SOURCE_NAMES = (
    "relax_kernels.cu",
    "noise_residual.cuh",
    "vdam_trace.cuh",
    "relion_preprocess.cuh",
    "relion_vdam_mstep.cuh",
    "relion_scoring.cuh",
    "relion_posterior.cuh",
    "sparse_pass2_posterior.cuh",
    "relion_translate_sum.cuh",
    "tiled_translations.cuh",
    "relion_capacity_texture.cuh",
    "ppca_moment_backproject.cuh",
    "ppca_stream.cuh",
    "relion_coarse_diff2_projector_body.inc",
    str(include_dir() / "recovar_cuda_common.cuh"),
    str(include_dir() / "device_scratch.cuh"),
    "Makefile",
)


FFI_REGISTRATIONS: tuple[tuple[str, str], ...] = (
    (TARGET_RELION_FUSED_X_HALF_BP, "RelionFusedXHalfBackproject"),
    (TARGET_RELION_PROJECTOR_HALF_TEXTURE_F32, "RelionProjectorHalfTextureF32"),
    (TARGET_RELION_PROJECTOR_PERSISTENT_HALF_TEXTURE_F32, "RelionProjectorPersistentHalfTextureF32"),
    (
        TARGET_RELION_FUSED_X_HALF_BP_PARTICLE_GRID,
        "RelionFusedXHalfBackprojectParticleGrid",
    ),
    (
        TARGET_RELION_FUSED_X_HALF_BP_SIGNATURE,
        "RelionFusedXHalfBackprojectSignature",
    ),
    (
        TARGET_RELION_POINT_GROUP_SYMMETRISE_BPREF,
        "RelionPointGroupSymmetriseBpref",
    ),
    (
        TARGET_RELION_POINT_GROUP_SYMMETRISE_BPREF_SPLIT_RANGE,
        "RelionPointGroupSymmetriseBprefSplitRange",
    ),
    (
        TARGET_RELION_POINT_GROUP_SYMMETRISE_BPREF_COMPLEX_RANGE,
        "RelionPointGroupSymmetriseBprefComplexRange",
    ),
    (TARGET_RELION_PREPROCESS_REAL_F32, "RelionPreprocessRealF32"),
    (
        TARGET_RELION_PREPROCESS_REAL_F32_NATIVE_LANE,
        "RelionPreprocessRealF32NativeLane",
    ),
    (
        TARGET_RELION_PREPROCESS_REAL_F32_NATIVE_ATOMIC,
        "RelionPreprocessRealF32NativeAtomic",
    ),
    (
        TARGET_RELION_MAKE_SCORING_ROTATIONS_F32,
        "RelionMakeScoringRotationsF32",
    ),
    (
        TARGET_RELION_MAKE_SCORING_ROTATIONS_LEFT_F32,
        "RelionMakeScoringRotationsLeftF32",
    ),
    (
        TARGET_RELION_MAKE_SCORING_ROTATIONS_F64,
        "RelionMakeScoringRotationsF64",
    ),
    (TARGET_RELION_TRANSLATE_SCORE_F32, "RelionTranslateScoreF32"),
    (TARGET_RELION_TRANSLATE_SCORE_F64, "RelionTranslateScoreF64"),
    (TARGET_RELION_TRANSLATE_BPREF_F32, "RelionTranslateBprefF32"),
    (TARGET_RELION_TRANSLATE_BPREF_F64, "RelionTranslateBprefF64"),
    (TARGET_RELION_VDAM_MSTEP_SUMS_F32, "RelionVdamMstepSumsF32"),
    (
        TARGET_RELION_VDAM_MSTEP_DENOMINATOR_F32,
        "RelionVdamMstepDenominatorF32",
    ),
    (TARGET_RELION_VDAM_MSTEP_FUSED_X_HALF, "RelionVdamMstepFusedXHalf"),
    (
        TARGET_RELION_VDAM_MSTEP_FUSED_PROJECTOR_X_HALF,
        "RelionVdamMstepFusedProjectorXHalf",
    ),
    (
        TARGET_RELION_VDAM_MSTEP_FUSED_PROJECTOR_RUNTIME_X_HALF,
        "RelionVdamMstepFusedProjectorRuntimeXHalf",
    ),
    (
        TARGET_RELION_COARSE_DIFF2_RECTANGULAR_F32,
        "RelionCoarseDiff2RectangularF32",
    ),
    (
        TARGET_RELION_COARSE_DIFF2_PROJECTOR_F32,
        "RelionCoarseDiff2ProjectorF32",
    ),
    (
        TARGET_RELION_COARSE_DIFF2_PROJECTOR_PER_IMAGE_F32,
        "RelionCoarseDiff2ProjectorPerImageF32",
    ),
    (
        TARGET_RELION_COARSE_NORMALIZED_CC_PAIRS_F32,
        "RelionCoarseNormalizedCcPairsF32",
    ),
    (
        TARGET_RELION_COARSE_NORMALIZED_CC_NATIVE_TEXTURE_PAIRS_F32,
        "RelionCoarseNormalizedCcNativeTexturePairsF32",
    ),
    (
        TARGET_RELION_FINE_DIFF2_RECTANGULAR_F32,
        "RelionFineDiff2RectangularF32",
    ),
    (
        TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_FLAT_ROWS_F32,
        "RelionFineDiff2FusedTranslateFlatRowsF32",
    ),
    (
        TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_RUNTIME_FLAT_ROWS_F32,
        "RelionFineDiff2FusedTranslateRuntimeFlatRowsF32",
    ),
    (
        TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_PAIRS_F32,
        "RelionFineDiff2FusedTranslatePairsF32",
    ),
    (
        TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_RUNTIME_PAIRS_F32,
        "RelionFineDiff2FusedTranslateRuntimePairsF32",
    ),
    (
        TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_JOBS_F32,
        "RelionFineDiff2FusedTranslateJobsF32",
    ),
    (
        TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_RUNTIME_JOBS_F32,
        "RelionFineDiff2FusedTranslateRuntimeJobsF32",
    ),
    (TARGET_RELION_FINE_DIFF2_PAIRS_F32, "RelionFineDiff2PairsF32"),
    (
        TARGET_RELION_POWERCLASS_SPECTRUM_HIGHRES_F32,
        "RelionPowerClassSpectrumHighresF32",
    ),
    (
        TARGET_RELION_POWERCLASS_SPECTRUM_HIGHRES_RUNTIME_F32,
        "RelionPowerClassSpectrumHighresRuntimeF32",
    ),
    (TARGET_RELION_EXPONENTIATE_F32, "RelionExponentiateF32"),
    (TARGET_RELION_EXPONENTIATE_BATCHED_F32, "RelionExponentiateBatchedF32"),
    (TARGET_RELION_DIVIDE_F32, "RelionDivideF32"),
    (
        TARGET_RELION_FINE_DIFF2_RECTANGULAR_F64,
        "RelionFineDiff2RectangularF64",
    ),
    (TARGET_RELION_FINE_DIFF2_PAIRS_F64, "RelionFineDiff2PairsF64"),
    (TARGET_RELION_DIVIDE_BATCHED_F32, "RelionDivideBatchedF32"),
    (TARGET_RELION_CUB_SORT_SCAN_F32, "RelionCubSortScanF32"),
    (TARGET_RELION_CUB_SORT_SCAN_BATCHED_F32, "RelionCubSortScanBatchedF32"),
    (TARGET_RELION_COARSE_CUT_F32, "RelionCoarseCutF32"),
    (
        TARGET_RELION_WAVG_ROTATION_ATOMIC_TRIPLET_ADD_F32,
        "RelionWavgRotationAtomicTripletAddF32",
    ),
    (
        TARGET_RELION_WAVG_ROTATION_ATOMIC_RUNTIME_TRIPLET_ADD_F32,
        "RelionWavgRotationAtomicRuntimeTripletAddF32",
    ),
    (
        TARGET_RELION_WAVG_SEQUENTIAL_TRIPLET_F32,
        "RelionWavgSequentialTripletF32",
    ),
    (
        TARGET_RELION_WAVG_SEQUENTIAL_RUNTIME_TRIPLET_F32,
        "RelionWavgSequentialRuntimeTripletF32",
    ),
    (TARGET_PPCA_MOMENT_SCATTER_F32, "PpcaMomentScatterF32"),
    (TARGET_PPCA_WINDOW_PROJECT_F32, "PpcaWindowProjectF32"),
    (TARGET_PPCA_LATENT_EPILOGUE_F32, "PpcaLatentEpilogueF32"),
    (TARGET_PPCA_POSTERIOR_PREP_F32, "PpcaPosteriorPrepF32"),
    (TARGET_DUAL_WEIGHTED_SUMS_PAIRS_F32, "DualWeightedSumsPairsF32"),
    (TARGET_DUAL_WEIGHTED_SUMS_PAIRS_ROWS_F32, "DualWeightedSumsPairsRowsF32"),
)


OPTIONAL_FFI_REGISTRATIONS = {
    TARGET_NOISE_RESIDUAL_STATISTICS: (
        "NoiseResidualStatistics",
        "Explicit CUDA build with NoiseResidualStatistics required",
    ),
    TARGET_PROJECT_RELION_HALF_RUNTIME: (
        "ProjectRelionHalfRuntime",
        "Projector capacity was requested but the loaded CUDA library lacks ProjectRelionHalfRuntime; explicitly rebuild the custom CUDA library",
    ),
    TARGET_PROJECT_RELION_HALF_IMAGE_RADIUS: (
        "ProjectRelionHalfImageRadius",
        "Image-radius projection requires ProjectRelionHalfImageRadius; explicitly rebuild the custom CUDA library",
    ),
    TARGET_PROJECT_RELION_HALF_CAPACITY_TEXTURE: (
        "ProjectRelionHalfCapacityTexture",
        "A persistent capacity projector texture requires ProjectRelionHalfCapacityTexture; explicitly rebuild the custom CUDA library",
    ),
    TARGET_PROJECT_RELION_HALF_CAPACITY_TEXTURE_COMPACT_PACKED: (
        "ProjectRelionHalfCapacityTextureCompactPacked",
        "Packed coarse GEMM rows require ProjectRelionHalfCapacityTextureCompactPacked; explicitly rebuild the custom CUDA library",
    ),
    TARGET_RELION_VDAM_MSTEP_FUSED_PROJECTOR_CAPACITY_X_HALF: (
        "RelionVdamMstepFusedProjectorCapacityXHalf",
        "BPref projector capacity requires RelionVdamMstepFusedProjectorCapacityXHalf; explicitly rebuild the custom CUDA library",
    ),
    TARGET_NOISE_PIXEL_PACK: (
        "NoisePixelPack",
        "CUDA noise pixel packing requires an explicit build with NoisePixelPack",
    ),
    TARGET_DEFERRED_VDAM_HOST_PACK: (
        "DeferredVdamHostPack",
        "CUDA host-plan packing requires an explicit build with DeferredVdamHostPack",
    ),
    TARGET_BPREF_PARTICLE_PACK: (
        "BprefParticlePack",
        "CUDA BPref packing requires an explicit build with BprefParticlePack",
    ),
    TARGET_RELION_FINE_DIFF2_RECTANGULAR_MASKED_F32: (
        "RelionFineDiff2RectangularMaskedF32",
        "Masked rectangular fine diff2 requires an explicit CUDA build with RelionFineDiff2RectangularMaskedF32",
    ),
    TARGET_SPARSE_PASS2_LOG_Z_F64: (
        "SparsePass2LogZF64",
        "The fused sparse pass-2 log-Z requires an explicit CUDA build with SparsePass2LogZF64",
    ),
    TARGET_SPARSE_PASS2_POSTERIOR_F32: (
        "SparsePass2PosteriorF32",
        "The fused sparse pass-2 posterior requires an explicit CUDA build with SparsePass2PosteriorF32",
    ),
    TARGET_SPARSE_PASS2_SEGMENTED_LOG_Z_F64: (
        "SparsePass2SegmentedLogZF64",
        "The segmented sparse pass-2 log-Z requires an explicit CUDA build with SparsePass2SegmentedLogZF64",
    ),
    TARGET_SPARSE_PASS2_SEGMENTED_POSTERIOR_F32: (
        "SparsePass2SegmentedPosteriorF32",
        "The segmented sparse pass-2 posterior requires an explicit CUDA build with SparsePass2SegmentedPosteriorF32",
    ),
    TARGET_SPARSE_PASS2_SEGMENTED_RECONSTRUCTION_F32: (
        "SparsePass2SegmentedReconstructionF32",
        "The reconstruction-only segmented sparse pass-2 posterior requires an explicit CUDA build with "
        "SparsePass2SegmentedReconstructionF32",
    ),
    TARGET_RELION_WAVG_SEQUENTIAL_RUNTIME_FLAT_ROWS_TRIPLET_F32: (
        "RelionWavgSequentialRuntimeFlatRowsTripletF32",
        "Flat-row RELION Wavg requires an explicit CUDA build with RelionWavgSequentialRuntimeFlatRowsTripletF32",
    ),
    TARGET_RELION_WAVG_ROTATION_ATOMIC_RUNTIME_FLAT_ROWS_TRIPLET_ADD_F32: (
        "RelionWavgRotationAtomicRuntimeFlatRowsTripletAddF32",
        "Flat-row RELION Wavg atomics require an explicit CUDA build with RelionWavgRotationAtomicRuntimeFlatRowsTripletAddF32",
    ),
    TARGET_RELION_WAVG_EXACT_ATOMIC_FLAT_ROWS_TRIPLET_ADD_F32: (
        "RelionWavgExactAtomicFlatRowsTripletAddF32",
        "Exact-position flat-row RELION Wavg atomics require an explicit CUDA build with RelionWavgExactAtomicFlatRowsTripletAddF32",
    ),
    TARGET_RELION_TRANSLATE_SUM_FLAT_ROWS_F32: (
        "RelionTranslateSumFlatRowsF32",
        "Flat-row translate-and-sum requires an explicit CUDA build with RelionTranslateSumFlatRowsF32",
    ),
    TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_RUNTIME_MASKED_FLAT_ROWS_F32: (
        "RelionFineDiff2FusedTranslateRuntimeMaskedFlatRowsF32",
        "Candidate-chunk flat-row scoring requires an explicit CUDA build with "
        "RelionFineDiff2FusedTranslateRuntimeMaskedFlatRowsF32",
    ),
}


LIBRARY = NativeLibrary(
    name="relax (EM)",
    filename="librelax_cuda.so",
    make_dir=pathlib.Path(__file__).resolve().parent,
    source_names=_RELAX_CUDA_BUILD_SOURCE_NAMES,
    lib_env=_RELAX_CUDA_LIB_ENV,
    registrations=FFI_REGISTRATIONS,
    optional_registrations=OPTIONAL_FFI_REGISTRATIONS,
)


_ffi_lock = LIBRARY.ffi_lock


def get_lib():
    """ctypes handle of the EM library (C-ABI exports such as the persistent-texture registry)."""

    return LIBRARY.get_lib()


def ensure_ffi():
    """Register recovar's pipeline FFI targets, which EM wrappers also call, then the EM library's."""

    cuda_backproject.ensure_ffi()
    LIBRARY.ensure_ffi()


def ensure_optional_ffi(target):
    """Register one optional target: EM targets from the EM library, pipeline ones through recovar."""

    if target in OPTIONAL_FFI_REGISTRATIONS:
        ensure_ffi()
        LIBRARY.ensure_optional_ffi(target)
    else:
        cuda_backproject.ensure_optional_ffi(target)
