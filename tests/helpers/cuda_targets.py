"""Probes of the loaded CUDA library's optional targets and of the deferred preprocess-check queue, for tests.

Moved from relax/cuda, where only tests called them: relax calls its optional targets without probing, and
the probes let a test skip on a library built before a target existed.
"""

from __future__ import annotations

from relax.cuda import library
from relax.cuda import preprocess as cuda_preprocess


def optional_target_supported(target: str) -> bool:
    """Return whether the loaded library exports an optional target's symbol."""

    try:
        library.ensure_ffi()
        symbol_name = library.OPTIONAL_FFI_REGISTRATIONS[target][0]
        return getattr(library.get_lib(), symbol_name, None) is not None
    except Exception:
        return False


def relion_wavg_sequential_runtime_flat_rows_triplet_f32_supported() -> bool:
    """Return whether the loaded library exports the flat-row Wavg target."""

    return optional_target_supported(
        library.TARGET_RELION_WAVG_SEQUENTIAL_RUNTIME_FLAT_ROWS_TRIPLET_F32
    )


def relion_wavg_rotation_atomic_runtime_flat_rows_triplet_add_f32_supported() -> bool:
    """Return whether the loaded library exports the flat-row Wavg atomics."""

    return optional_target_supported(
        library.TARGET_RELION_WAVG_ROTATION_ATOMIC_RUNTIME_FLAT_ROWS_TRIPLET_ADD_F32
    )


def sparse_pass2_segmented_supported() -> bool:
    """Return whether the loaded library exports both segmented pass-2 targets.

    The segmented handlers are an optional ABI, so a library built before them
    stays loadable and callers fall back to the rectangular handlers.
    """

    try:
        library.ensure_ffi()
        lib = library.get_lib()
        return all(
            getattr(lib, library.OPTIONAL_FFI_REGISTRATIONS[target][0], None) is not None
            for target in (
                library.TARGET_SPARSE_PASS2_SEGMENTED_LOG_Z_F64,
                library.TARGET_SPARSE_PASS2_SEGMENTED_POSTERIOR_F32,
            )
        )
    except Exception:
        return False


def pending_relion_preprocess_checks() -> int:
    """Number of queued deferred soft-mask checks (test and diagnostics hook)."""

    with cuda_preprocess._RELION_PREPROCESS_PENDING_LOCK:
        return len(cuda_preprocess._RELION_PREPROCESS_PENDING_CHECKS)
