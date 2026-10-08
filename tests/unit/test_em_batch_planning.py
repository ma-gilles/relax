"""EM batch sizing and microbatch memory-policy checks."""


import numpy as np
import pytest
from recovar import utils as recovar_utils

from relax.helpers.batch_planning import _estimate_relion_em_batch_sizes
from relax.local.local_layout import (
    LocalHypothesisLayout,
    _local_search_engine_rotation_block_size,
)

IMAGE_SHAPE = (8, 8)
VOLUME_SHAPE = (8, 8, 8)


def _identity_layout(rotation_counts, *, n_trans):
    """Build identity poses for the requested per-image neighborhood sizes."""
    rotation_counts = np.asarray(rotation_counts, dtype=np.int32)
    offsets = np.concatenate(([0], np.cumsum(rotation_counts))).astype(np.int64)
    total = int(offsets[-1])
    return LocalHypothesisLayout(
        n_global_rotations=total,
        n_pixels=1,
        n_psi=1,
        rotation_offsets=offsets,
        rotation_ids_flat=np.arange(total, dtype=np.int32),
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (total, 3, 3)).copy(),
        rotation_log_priors_flat=np.zeros(total, dtype=np.float32),
        rotation_counts=rotation_counts,
        translation_grid=np.zeros((n_trans, 2), dtype=np.float32),
        translation_log_priors=np.zeros((len(rotation_counts), n_trans), dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Helpers (same as test_fsc_resolution_loop.py)
# ---------------------------------------------------------------------------


def test_local_search_engine_rotation_block_size_caps_dense_tiles():
    assert _local_search_engine_rotation_block_size(64) == 64
    assert _local_search_engine_rotation_block_size(1024) == 1024
    assert _local_search_engine_rotation_block_size(5000) == 1024


def test_relion_em_batch_sizing_preserves_small_safe_requests():
    plan = _estimate_relion_em_batch_sizes(
        requested_image_batch_size=4,
        requested_rotation_block_size=8,
        n_rot=8,
        n_trans=5,
        image_shape=IMAGE_SHAPE,
        volume_shape=VOLUME_SHAPE,
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=80.0,
    )

    assert plan.image_batch_size == 4
    assert plan.rotation_block_size == 8


def test_relion_em_batch_sizing_clamps_highres_projection_tiles():
    plan = _estimate_relion_em_batch_sizes(
        requested_image_batch_size=250,
        requested_rotation_block_size=20000,
        n_rot=294912,
        n_trans=137,
        image_shape=(384, 384),
        volume_shape=(384, 384, 384),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=80.0,
    )

    assert plan.image_batch_size == 4
    assert plan.rotation_block_size == 13
    assert plan.projection_block_gb <= plan.projection_budget_gb
    assert plan.pose_pixel_tile_gb < 3.0
    assert plan.active_score_tile_gb <= plan.active_score_tile_budget_gb
    assert plan.translation_tile_gb <= plan.translation_tile_budget_gb


def test_relion_em_batch_sizing_does_not_pad_beyond_actual_rotation_grid():
    plan = _estimate_relion_em_batch_sizes(
        requested_image_batch_size=250,
        requested_rotation_block_size=20000,
        n_rot=4608,
        n_trans=29,
        image_shape=(384, 384),
        volume_shape=(384, 384, 384),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=80.0,
    )

    assert plan.image_batch_size == 46
    assert plan.rotation_block_size == 65
    assert plan.projection_block_gb <= plan.projection_budget_gb
    assert plan.active_score_tile_gb <= plan.active_score_tile_budget_gb


def test_relion_em_batch_sizing_uses_runtime_gpu_occupancy(monkeypatch):
    monkeypatch.setattr(recovar_utils, "get_gpu_memory_total", lambda: 80.0)
    monkeypatch.setattr(recovar_utils, "get_gpu_memory_used", lambda: 60.0)

    plan = _estimate_relion_em_batch_sizes(
        requested_image_batch_size=250,
        requested_rotation_block_size=20000,
        n_rot=4608,
        n_trans=29,
        image_shape=(384, 384),
        volume_shape=(384, 384, 384),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=None,
    )

    assert plan.gpu_used_estimate_gb == pytest.approx(60.0)
    assert plan.rotation_block_size < 4608
    assert plan.projection_block_gb <= plan.projection_budget_gb


def test_relion_em_batch_sizing_caps_runtime_highres_local_translation_tile(monkeypatch):
    monkeypatch.setattr(recovar_utils, "get_gpu_memory_total", lambda: 80.0)
    monkeypatch.setattr(recovar_utils, "get_gpu_memory_used", lambda: 24.0)

    plan = _estimate_relion_em_batch_sizes(
        requested_image_batch_size=250,
        requested_rotation_block_size=4608,
        n_rot=4608,
        n_trans=116,
        image_shape=(384, 384),
        volume_shape=(384, 384, 384),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=None,
    )

    assert plan.image_batch_size <= 56
    assert plan.translation_tile_gb <= plan.translation_tile_budget_gb
    assert plan.gpu_used_estimate_gb == pytest.approx(24.0)


def test_relion_em_batch_sizing_caps_dense_big_jit_score_workspace(monkeypatch):
    monkeypatch.setattr(recovar_utils, "get_gpu_memory_total", lambda: 80.0)
    monkeypatch.setattr(recovar_utils, "get_gpu_memory_used", lambda: 41.0)

    plan = _estimate_relion_em_batch_sizes(
        requested_image_batch_size=64,
        requested_rotation_block_size=8192,
        n_rot=36864,
        n_trans=29,
        image_shape=(256, 256),
        volume_shape=(256, 256, 256),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=None,
    )

    assert plan.image_batch_size == 50
    assert plan.rotation_block_size == 146
    assert plan.projection_block_gb <= plan.projection_budget_gb
    assert plan.pose_pixel_tile_gb < 3.0
    assert plan.active_score_tile_gb <= plan.active_score_tile_budget_gb
    assert plan.gpu_used_estimate_gb == pytest.approx(41.0)


def test_relion_em_batch_sizing_uses_active_window_for_dense_score_workspace(monkeypatch):
    monkeypatch.setattr(recovar_utils, "get_gpu_memory_total", lambda: 80.0)
    monkeypatch.setattr(recovar_utils, "get_gpu_memory_used", lambda: 0.0)

    common = dict(
        requested_image_batch_size=64,
        requested_rotation_block_size=8192,
        n_rot=36864,
        n_trans=29,
        image_shape=(256, 256),
        volume_shape=(256, 256, 256),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=None,
    )
    low_res = _estimate_relion_em_batch_sizes(**common, current_size=56)
    high_res = _estimate_relion_em_batch_sizes(**common, current_size=248)

    assert low_res.rotation_block_size == 8192
    assert high_res.rotation_block_size < 8192
    assert low_res.score_pixel_count < high_res.score_pixel_count


def test_relion_em_batch_sizing_allows_larger_adaptive_pass1_blocks():
    common = dict(
        requested_image_batch_size=500,
        requested_rotation_block_size=5000,
        n_rot=36864,
        n_trans=29,
        image_shape=(256, 256),
        volume_shape=(256, 256, 256),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=80.0,
    )

    pass1_coarse = _estimate_relion_em_batch_sizes(**common, current_size=100)
    pass2_fine = _estimate_relion_em_batch_sizes(**common, current_size=154)

    assert pass1_coarse.score_pixel_count < pass2_fine.score_pixel_count
    assert pass1_coarse.rotation_block_size > pass2_fine.rotation_block_size
    assert pass1_coarse.pose_pixel_tile_gb <= pass1_coarse.projection_budget_gb
    assert pass2_fine.pose_pixel_tile_gb <= pass2_fine.projection_budget_gb


def test_relion_em_batch_sizing_projection_budget_override_expands_pass1_blocks(monkeypatch):
    common = dict(
        requested_image_batch_size=64,
        requested_rotation_block_size=8192,
        n_rot=36864,
        n_trans=29,
        image_shape=(256, 256),
        volume_shape=(256, 256, 256),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=80.0,
        current_size=100,
    )

    monkeypatch.delenv("RELAX_RELION_EM_BATCH_PROJECTION_FRACTION", raising=False)
    default = _estimate_relion_em_batch_sizes(**common)

    monkeypatch.setenv("RELAX_RELION_EM_BATCH_PROJECTION_FRACTION", "0.40")
    expanded = _estimate_relion_em_batch_sizes(**common)

    assert expanded.rotation_block_size > default.rotation_block_size
    assert expanded.rotation_block_size <= common["requested_rotation_block_size"]
    assert expanded.pose_pixel_tile_gb <= expanded.projection_budget_gb


def test_relion_em_batch_sizing_projection_budget_override_rejects_invalid(monkeypatch):
    monkeypatch.setenv("RELAX_RELION_EM_BATCH_PROJECTION_FRACTION", "0")

    with pytest.raises(ValueError, match="RELAX_RELION_EM_BATCH_PROJECTION_FRACTION"):
        _estimate_relion_em_batch_sizes(
            requested_image_batch_size=64,
            requested_rotation_block_size=8192,
            n_rot=36864,
            n_trans=29,
            image_shape=(256, 256),
            volume_shape=(256, 256, 256),
            padding_factor=2,
            n_classes=1,
            gpu_memory_gb=80.0,
            current_size=100,
        )


def test_relion_em_batch_sizing_clamps_highres_translation_tiles():
    plan = _estimate_relion_em_batch_sizes(
        requested_image_batch_size=250,
        requested_rotation_block_size=1024,
        n_rot=1024,
        n_trans=137,
        image_shape=(384, 384),
        volume_shape=(384, 384, 384),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=80.0,
    )

    assert plan.image_batch_size == 9
    assert plan.rotation_block_size == 13
    assert plan.pose_pixel_tile_gb < 3.0
    assert plan.translation_tile_gb <= plan.translation_tile_budget_gb


def test_relion_em_batch_sizing_accounts_for_k_classes():
    single = _estimate_relion_em_batch_sizes(
        requested_image_batch_size=250,
        requested_rotation_block_size=20000,
        n_rot=294912,
        n_trans=137,
        image_shape=(384, 384),
        volume_shape=(384, 384, 384),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=80.0,
    )
    k4 = _estimate_relion_em_batch_sizes(
        requested_image_batch_size=250,
        requested_rotation_block_size=20000,
        n_rot=294912,
        n_trans=137,
        image_shape=(384, 384),
        volume_shape=(384, 384, 384),
        padding_factor=2,
        n_classes=4,
        gpu_memory_gb=80.0,
    )

    assert k4.image_batch_size <= single.image_batch_size
    assert k4.rotation_block_size <= single.rotation_block_size
