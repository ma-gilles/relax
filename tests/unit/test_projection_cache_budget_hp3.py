"""Fine-projection cache admission for the HEALPix-3 / current_size-92 regime.

The 10k EMPIAR-10097 convergence run (job 14031616) skipped the sparse pass-2
projection cache in every hp3 iteration ("estimated transient 18.36 GiB
exceeds cap 7.96 GiB") and recomputed fine projections per image chunk.
These checks pin the admission rule that lets an 80 GB device cache the
294912-rotation hp3 projections while a 40 GB device still declines.
"""

import numpy as np
import pytest

from relax.sparse_pass2.sparse_pass2_budget import (
    _projection_cache_fits_budget,
    _projection_cache_max_bytes_for_pass,
    _projection_cache_transient_bytes,
    _projection_call_max_bytes_for_pass,
)
from helpers.float_compare import assert_matches

_HP3_FINE_ROTATIONS = 294912
_CS92_HALF_PIXELS = 3386  # windowed half-spectrum score pixels at current_size 92, 256^2
_H100_BYTES = int(79.65 * 1024**3)
_A100_40_BYTES = 40 * 1024**3


def _clear_env(monkeypatch):
    monkeypatch.delenv("RELAX_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES", raising=False)
    monkeypatch.delenv("RELAX_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS", raising=False)


def test_hp3_cs92_cache_estimate_matches_logged_value():
    transient = _projection_cache_transient_bytes(
        _HP3_FINE_ROTATIONS, _CS92_HALF_PIXELS, projection_complex_dtype=np.complex64, include_abs2=True,
    )
    # score (complex64) + recon (complex64 shares the score cache when not
    # windowed) is not double counted here; the logged 18.36 GiB includes the
    # windowed score+recon+abs2 triple, so only check the single-cache term.
    assert transient == _HP3_FINE_ROTATIONS * _CS92_HALF_PIXELS * (8 + 4)


@pytest.mark.parametrize(
    "logged_gib",
    [18.36, 20.00],  # before and after the stable windows' physical-class padding (job 14514823)
)
def test_hp3_cache_admitted_on_80gb_rejected_on_40gb(monkeypatch, logged_gib):
    _clear_env(monkeypatch)
    logged_hp3_estimate = int(logged_gib * 1024**3)
    assert _projection_cache_fits_budget(logged_hp3_estimate, _projection_cache_max_bytes_for_pass(_H100_BYTES))
    assert not _projection_cache_fits_budget(logged_hp3_estimate, _projection_cache_max_bytes_for_pass(_A100_40_BYTES))


def test_cache_cap_is_thirty_percent_of_device_memory_without_override(monkeypatch):
    _clear_env(monkeypatch)
    assert _projection_cache_max_bytes_for_pass(_H100_BYTES) == int(_H100_BYTES * 0.30)


def test_cache_cap_change_leaves_per_call_rotation_budget_alone(monkeypatch):
    _clear_env(monkeypatch)
    # The per-call projected-rotation budget (809 rotations per call at cs 92 on
    # an H100 in the measured runs) must not grow with the cache admission cap.
    assert _projection_call_max_bytes_for_pass(_H100_BYTES) == int(_H100_BYTES * 0.040)
    assert _projection_call_max_bytes_for_pass(_H100_BYTES) < _projection_cache_max_bytes_for_pass(_H100_BYTES)


def test_env_override_still_wins(monkeypatch):
    monkeypatch.setenv("RELAX_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES", "654321")
    assert _projection_cache_max_bytes_for_pass(_H100_BYTES) == 654321


def test_cache_build_rotations_per_call_scales_scoring_budget(monkeypatch):
    from relax.sparse_pass2.sparse_pass2_budget import _projection_cache_build_max_rotations_per_call

    _clear_env(monkeypatch)
    assert _projection_cache_build_max_rotations_per_call(809, 294912) == 4 * 809
    # never more rotations than the fine grid holds, never below one
    assert _projection_cache_build_max_rotations_per_call(809, 1000) == 1000
    assert _projection_cache_build_max_rotations_per_call(None, 294912) is None
    monkeypatch.setenv("RELAX_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS", "4096")
    assert _projection_cache_build_max_rotations_per_call(809, 294912) == 4096


def test_flatten_bucket_rotations_is_host_side_for_numpy_and_not_jitted():
    from relax.local import local_backprojection as lb

    rots = np.arange(2 * 3 * 9, dtype=np.float32).reshape(2, 3, 3, 3)
    out = lb.flatten_bucket_rotations(rots)
    assert isinstance(out, np.ndarray) and out.shape == (6, 3, 3)
    assert_matches(out, rots.reshape(6, 3, 3))
    assert not hasattr(lb.flatten_bucket_rotations, "lower")  # plain function, not a jit wrapper
    import jax.numpy as jnp
    dev = lb.flatten_bucket_rotations(jnp.asarray(rots))
    assert dev.shape == (6, 3, 3)
    assert_matches(np.asarray(dev), rots.reshape(6, 3, 3))


def test_large_bucket_pow2_rung_is_opt_in(monkeypatch):
    from relax.scoring import sparse_bucket_arrays as sba

    monkeypatch.delenv("RELAX_SPARSE_PASS2_LARGE_BUCKET_POW2", raising=False)
    monkeypatch.delenv("RELAX_LOCAL_BUCKET_QUANTUM", raising=False)
    default = [sba._pass2_bucket_rotation_size(c, 5000) for c in (7, 100, 900, 5000, 9000, 20000, 100000, 217000)]
    assert default[:3] == [16, 128, 1024]  # small supports: shared power-of-two rule, unchanged
    monkeypatch.setenv("RELAX_SPARSE_PASS2_LARGE_BUCKET_POW2", "1")
    pow2 = [sba._pass2_bucket_rotation_size(c, 5000) for c in (7, 100, 900, 5000, 9000, 20000, 100000, 217000)]
    assert pow2[:3] == default[:3]
    assert pow2[3:] == [8192, 16384, 32768, 131072, 262144]
    assert all(p >= d for p, d in zip(pow2, default))  # never smaller than the shared quantiser


def test_texture_projector_fallback_is_reported_once_per_reason(monkeypatch, caplog):
    """A silent fallback to the vmapped JAX projector costs pass-2 host time."""
    import logging

    import jax.numpy as jnp

    from relax.helpers import projection

    monkeypatch.setattr(projection, "_cuda_projection_available", lambda: True)
    projection._TEXTURE_FALLBACK_REPORTED.clear()
    good = jnp.zeros((187, 187, 94), jnp.complex64)
    assert projection._relion_projector_texture_enabled(good, r_max=46, padding_factor=2) is True

    wrong_dtype = jnp.zeros((187, 187, 94), jnp.complex128)
    with caplog.at_level(logging.WARNING, logger=projection.logger.name):
        assert (
            projection._relion_projector_texture_enabled(wrong_dtype, r_max=46, padding_factor=2)
            is False
        )
        assert (
            projection._relion_projector_texture_enabled(wrong_dtype, r_max=46, padding_factor=2)
            is False
        )
    messages = [r.message for r in caplog.records if "texture projector unavailable" in r.message]
    assert len(messages) == 1, messages
    assert "complex128" in messages[0] and "want complex64" in messages[0]


def test_pass2_projector_cast_unblocks_the_texture_projector(monkeypatch):
    """complex128 is exactly what makes the texture path reject the slab."""
    import jax.numpy as jnp

    from relax.helpers import projection

    monkeypatch.setattr(projection, "_cuda_projection_available", lambda: True)
    projection._TEXTURE_FALLBACK_REPORTED.clear()
    slab = jnp.zeros((187, 187, 94), jnp.complex128)
    assert projection._relion_projector_texture_enabled(slab, r_max=46, padding_factor=2) is False
    assert (
        projection._relion_projector_texture_enabled(
            slab.astype(jnp.complex64), r_max=46, padding_factor=2
        )
        is True
    )


def test_projector_build_log_reports_the_slab_dtype(monkeypatch, caplog):
    """The slab dtype decides texture versus JAX-fallback projection; log it."""
    import logging

    from helpers.tiny_refinement import CallTrace, run_tiny_refinement

    from relax.refinement import iteration_loop

    trace = CallTrace(monkeypatch).wrap(iteration_loop, "prepare_scoring_projector", "build")
    with caplog.at_level(logging.INFO, logger="relax.refinement.iteration_loop"):
        run_tiny_refinement(monkeypatch, max_iter=1, final_after_max_iter=False)
    built = [record for record in caplog.records if "built exact Projector::data for scoring" in record.getMessage()]
    assert len(built) == 1
    slab = trace.calls("build")[0].result
    assert built[0].getMessage().startswith(
        f"RELION mode: built exact Projector::data for scoring at current_size={built[0].args[0]} "
        f"r_max={slab.r_max} dtype={slab.data.dtype} in "
    )
