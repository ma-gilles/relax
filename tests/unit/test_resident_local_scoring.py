"""Projection-by-rows and the projected-row scoring program (T12 stage 2).

Ticket: ``em_parity_tickets_20260918/T12_resident_local_search.md``.

Local search cannot gather a per-iteration projection cache: its fine grid at
HEALPix order 5 holds 2.4M rotations. The chunk's own rows are projected
instead. Two properties make that a layout change rather than an arithmetic
change, and both are measured here:

1. scoring a chunk from an explicitly supplied reference is bitwise identical
   to scoring it from the same values gathered out of a cache, so the two
   routes share one arithmetic path;
2. projecting rows in blocks is bitwise identical to projecting them in one
   call, so the byte budget that sizes the blocks cannot move a number.

The first test reuses the T6 fixture builder, so the operands, the window and
the CUDA kernel are the production ones rather than a second-hand copy.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers.float_compare import assert_matches

pytest.importorskip("jax")
import jax
import jax.numpy as jnp
from test_resident_scoring import N_FINE_TRANS, _build_case
from helpers.sparse_pass2_mock import IMAGE_SHAPE, VOLUME_SHAPE

from relax.sparse_pass2.resident_candidates import expand_mask_rows, materialize_chunk
from relax.sparse_pass2.resident_scoring import (
    live_projection_block_rows,
    project_resident_live_rows,
    project_resident_rows,
    resident_projection_block_rows,
    resident_row_projection_bytes,
    score_resident_chunk,
    score_resident_projected_chunk,
)

pytestmark = pytest.mark.unit


def _packed_row_mask(tables, chunk, n_fine_trans):
    """The chunk's candidate mask in the local layout's packing.

    The resident candidate table (T5) stores a per-parent uint32 bitset over
    coarse translations; the local layout stores per-row bytes over fine
    translations. Round-trip the T5 mask through its dense expansion so the two
    programs are handed the same candidate set in their own spellings.
    """

    dense = np.zeros((int(chunk.row_capacity), int(n_fine_trans)), dtype=bool)
    n_valid = int(chunk.n_valid_rows)
    if n_valid:
        rows = []
        for image in range(chunk.image_start, chunk.image_stop):
            rows.append(expand_mask_rows(tables, image, FINE_TRANS_PARENT))
        dense[:n_valid] = np.concatenate(rows, axis=0)
    return np.packbits(dense, axis=1, bitorder="little")


FINE_TRANS_PARENT = np.repeat(np.arange(4, dtype=np.int32), 2)


def test_projection_block_budget_is_at_least_one_row():
    per_row = resident_row_projection_bytes(n_score_pixels=3386, n_recon_pixels=4324)
    assert per_row == 3386 * 8 + 4324 * 12
    assert resident_projection_block_rows(
        n_score_pixels=3386, n_recon_pixels=4324, max_block_bytes=1
    ) == 1
    assert (
        resident_projection_block_rows(
            n_score_pixels=3386, n_recon_pixels=4324, max_block_bytes=10 * per_row
        )
        == 10
    )


@pytest.mark.gpu
def test_projected_scoring_equals_the_cached_gather_bitwise(
    monkeypatch, custom_cuda_lib, gpu_device
):
    """One arithmetic path: the reference's source cannot change a score bit."""

    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    with jax.default_device(gpu_device):
        case = _build_case(
            6, row_capacity_ladder=(16, 64, 256), image_capacity_ladder=(2, 4, 8)
        )
        tables = case["tables"]
        operands = case["operands"]
        cache = case["projection_cache"]
        compared = 0
        for chunk in case["chunks"]:
            host = materialize_chunk(tables, chunk)
            cached = score_resident_chunk(
                jnp.asarray(host["row_image_local"]),
                jnp.asarray(host["row_fine_rot"]),
                jnp.asarray(host["row_log_prior"]),
                jnp.asarray(host["row_mask_bits"]),
                jnp.asarray(host["row_mask_mode"]),
                jnp.asarray(host["n_valid_rows"]),
                jnp.asarray(host["image_ids"]),
                cache,
                operands.score_input,
                operands.corr_img_score,
                operands.highres_xi2_half,
                operands.translation_prior,
                half_weights=operands.half_weights,
                translation_angles=operands.translation_angles,
                full_to_compact=operands.full_to_compact,
                fine_translation_parent=jnp.asarray(FINE_TRANS_PARENT),
                logical_current_size=jnp.asarray(operands.current_size, dtype=jnp.int32),
                row_capacity=int(chunk.row_capacity),
                image_capacity=int(chunk.image_capacity),
                n_fine_trans=N_FINE_TRANS,
                n_score_pixels=int(operands.n_score_pixels),
            )
            reference = cache[jnp.asarray(host["row_fine_rot"], dtype=jnp.int32)]
            projected = score_resident_projected_chunk(
                reference,
                jnp.asarray(host["row_image_local"]),
                jnp.asarray(host["row_log_prior"]),
                jnp.asarray(_packed_row_mask(tables, chunk, N_FINE_TRANS)),
                jnp.asarray(host["n_valid_rows"]),
                jnp.asarray(host["image_ids"]),
                operands.score_input,
                operands.corr_img_score,
                operands.highres_xi2_half,
                operands.translation_prior,
                half_weights=operands.half_weights,
                translation_angles=operands.translation_angles,
                full_to_compact=operands.full_to_compact,
                logical_current_size=jnp.asarray(operands.current_size, dtype=jnp.int32),
                row_capacity=int(chunk.row_capacity),
                image_capacity=int(chunk.image_capacity),
                n_fine_trans=N_FINE_TRANS,
                n_score_pixels=int(operands.n_score_pixels),
            )
            for field in ("raw_diff2", "scores", "min_diff2"):
                assert_matches(
                    np.asarray(getattr(cached, field)),
                    np.asarray(getattr(projected, field)),
                    err_msg=f"{field} on chunk {chunk}",
                )
            compared += 1
        assert compared, "the fixture must produce at least one chunk"


@pytest.mark.gpu
def test_full_support_mask_equals_an_all_ones_packed_mask(
    monkeypatch, custom_cuda_lib, gpu_device
):
    """``row_mask_bits=None`` is the layout's compact spelling of full support."""

    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    with jax.default_device(gpu_device):
        case = _build_case(
            6, row_capacity_ladder=(16, 64, 256), image_capacity_ladder=(2, 4, 8)
        )
        operands = case["operands"]
        cache = case["projection_cache"]
        chunk = case["chunks"][0]
        host = materialize_chunk(case["tables"], chunk)
        reference = cache[jnp.asarray(host["row_fine_rot"], dtype=jnp.int32)]
        ones = np.packbits(
            np.ones((int(chunk.row_capacity), N_FINE_TRANS), dtype=bool),
            axis=1,
            bitorder="little",
        )

        def run(mask):
            return score_resident_projected_chunk(
                reference,
                jnp.asarray(host["row_image_local"]),
                jnp.asarray(host["row_log_prior"]),
                mask,
                jnp.asarray(host["n_valid_rows"]),
                jnp.asarray(host["image_ids"]),
                operands.score_input,
                operands.corr_img_score,
                operands.highres_xi2_half,
                operands.translation_prior,
                half_weights=operands.half_weights,
                translation_angles=operands.translation_angles,
                full_to_compact=operands.full_to_compact,
                logical_current_size=jnp.asarray(operands.current_size, dtype=jnp.int32),
                row_capacity=int(chunk.row_capacity),
                image_capacity=int(chunk.image_capacity),
                n_fine_trans=N_FINE_TRANS,
                n_score_pixels=int(operands.n_score_pixels),
            )

        none_mask = run(None)
        ones_mask = run(jnp.asarray(ones))
        for field in ("raw_diff2", "scores", "min_diff2"):
            assert_matches(
                np.asarray(getattr(none_mask, field)),
                np.asarray(getattr(ones_mask, field)),
                err_msg=field,
            )


@pytest.mark.gpu
@pytest.mark.parametrize("block_rows", [1, 3, 7])
def test_row_projection_blocking_is_bitwise(monkeypatch, custom_cuda_lib, gpu_device, block_rows):
    """The byte budget that sizes a projector call cannot move a projection bit."""

    import recovar.cuda_backproject as cuda_backproject
    from helpers.em_arrays import _hermitian_volume

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    rng = np.random.default_rng(20260919)
    n_rows = 11
    angles = rng.uniform(0.0, 2 * np.pi, n_rows)
    rotations = np.stack(
        [
            np.array(
                [[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            for a in angles
        ]
    )
    volume = _hermitian_volume(VOLUME_SHAPE, seed=5)
    n_half = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)
    score_indices = np.arange(0, n_half, 2, dtype=np.int32)
    recon_indices = np.arange(n_half, dtype=np.int32)

    with jax.default_device(gpu_device):
        whole = project_resident_rows(
            jnp.asarray(volume),
            jnp.asarray(rotations),
            IMAGE_SHAPE,
            VOLUME_SHAPE,
            "linear_interp",
            score_indices=score_indices,
            recon_indices=recon_indices,
            max_projected_rotations=n_rows,
            output_complex_dtype=jnp.complex64,
            output_abs2_dtype=jnp.float32,
            relion_texture_interp=False,
        )
        blocked = project_resident_rows(
            jnp.asarray(volume),
            jnp.asarray(rotations),
            IMAGE_SHAPE,
            VOLUME_SHAPE,
            "linear_interp",
            score_indices=score_indices,
            recon_indices=recon_indices,
            max_projected_rotations=block_rows,
            output_complex_dtype=jnp.complex64,
            output_abs2_dtype=jnp.float32,
            relion_texture_interp=False,
        )
    for name, a, b in zip(("score", "recon", "recon_abs2"), whole, blocked, strict=True):
        assert_matches(np.asarray(a), np.asarray(b), err_msg=name)
    assert np.asarray(whole[0]).shape == (n_rows, score_indices.size)
    assert np.asarray(whole[1]).shape == (n_rows, recon_indices.size)


def test_live_projection_blocks_divide_the_capacity():
    assert live_projection_block_rows(65536, 16256) == 4096
    assert live_projection_block_rows(16384, 16256) == 1024
    assert live_projection_block_rows(1024, 16256) == 256
    assert live_projection_block_rows(65536, 3000) == 2048
    assert live_projection_block_rows(256, 16256) == 256
    assert live_projection_block_rows(100, 16256) == 100
    # An overflow chunk's capacity past the projector call is cut into power-of-two
    # blocks that divide it (bigbox 14684083: 1536 rows in one 2.88 GiB call).
    assert live_projection_block_rows(1536, 609) == 512
    assert live_projection_block_rows(1536, 300) == 256
    assert 1536 % live_projection_block_rows(1536, 609) == 0


@pytest.mark.parametrize("n_valid_rows", [1, 9, 16])
def test_live_row_projection_matches_whole_chunk_projection(monkeypatch, n_valid_rows):
    """Projecting only a chunk's valid rows gives their projections, and zeros past them.

    ``live_projection_block_rows(16, ...)`` is 16, so a patched quantum of four
    rows exercises several blocks and a partial last block.
    """

    from helpers.em_arrays import _hermitian_volume

    import relax.sparse_pass2.resident_scoring as resident_scoring

    capacity = 16
    rng = np.random.default_rng(20260927)
    angles = rng.uniform(0.0, 2 * np.pi, capacity)
    rotations = np.stack(
        [
            np.array(
                [[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            for a in angles
        ]
    )
    rotations[n_valid_rows:] = np.eye(3, dtype=np.float32)
    volume = _hermitian_volume(VOLUME_SHAPE, seed=5)
    n_half = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)
    score_indices = np.arange(0, n_half, 2, dtype=np.int32)
    recon_indices = np.arange(1, n_half, 3, dtype=np.int32)
    kwargs = dict(
        score_indices=score_indices,
        recon_indices=recon_indices,
        output_complex_dtype=jnp.complex64,
        output_abs2_dtype=jnp.float32,
        relion_texture_interp=False,
    )

    whole = project_resident_rows(
        jnp.asarray(volume),
        jnp.asarray(rotations),
        IMAGE_SHAPE,
        VOLUME_SHAPE,
        "linear_interp",
        max_projected_rotations=capacity,
        **kwargs,
    )
    monkeypatch.setattr(resident_scoring, "live_projection_block_rows", lambda row_capacity, max_rows: 4)
    *live, n_projected = project_resident_live_rows(
        jnp.asarray(volume),
        jnp.asarray(rotations),
        IMAGE_SHAPE,
        VOLUME_SHAPE,
        "linear_interp",
        n_valid_rows=n_valid_rows,
        max_projected_rotations=capacity,
        **kwargs,
    )

    assert n_projected == min(-(-n_valid_rows // 4) * 4, capacity)
    for name, a, b in zip(("score", "recon", "recon_abs2"), whole, live, strict=True):
        a, b = np.asarray(a), np.asarray(b)
        assert b.shape == a.shape and b.dtype == a.dtype, name
        assert_matches(b[:n_valid_rows], a[:n_valid_rows], err_msg=name)
        assert not np.any(b[n_projected:]), name


def _relion_half_case(seed=11):
    """A random RELION ``PPref`` half slab (r_max 7, padding 2) and 12 rotations."""

    r_max, pf = 7, 2
    rng = np.random.default_rng(seed)
    shape = (2 * r_max * pf + 3, 2 * r_max * pf + 3, r_max * pf + 2)
    half = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    angles = rng.uniform(0.0, 2 * np.pi, (12, 3))
    rotations = []
    for a, b, c in angles:
        rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
        rz2 = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
        rotations.append(rz @ ry @ rz2)
    return jnp.asarray(half), jnp.asarray(np.stack(rotations).astype(np.float32)), r_max, pf


def _window_union_projection_matches(texture_interp: bool, crop: int = 16):
    from relax.sparse_pass2.sparse_pass2_projection_blocks import projection_window_union

    half, rotations, r_max, pf = _relion_half_case()
    image_shape = (16, 16)
    n_half = image_shape[0] * (image_shape[1] // 2 + 1)
    score_indices = np.arange(3, n_half - 5, 2, dtype=np.int32)
    recon_indices = np.arange(1, n_half, 3, dtype=np.int32)
    kwargs = dict(
        n_valid_rows=9,
        score_indices=score_indices,
        recon_indices=recon_indices,
        max_projected_rotations=12,
        output_complex_dtype=jnp.complex64,
        output_abs2_dtype=jnp.float32,
        relion_projector_half=half,
        relion_projector_r_max=r_max,
        projection_padding_factor=pf,
        relion_texture_interp=texture_interp,
        projector_output_size=crop,
    )
    args = (None, rotations, image_shape, (16, 16, 16), "linear_interp")
    full = project_resident_live_rows(*args, **kwargs)
    union = project_resident_live_rows(
        *args, window_union=projection_window_union(
            score_indices, recon_indices, image_shape=image_shape, projector_output_size=crop
        ), **kwargs
    )
    assert full[3] == union[3]
    for name, a, b in zip(("score", "recon", "recon_abs2"), full[:3], union[:3], strict=True):
        assert np.any(np.asarray(a)), name
        assert_matches(np.asarray(b), np.asarray(a), err_msg=name)


def test_window_union_projection_matches_the_full_row_windows():
    """The RELION projector computing only the windows' pixels gives the windows of the full rows."""

    _window_union_projection_matches(texture_interp=False)
    # A crop smaller than the box: window pixels outside it are zero in both.
    _window_union_projection_matches(texture_interp=False, crop=10)


@pytest.mark.gpu
def test_window_union_texture_projection_matches_the_full_row_windows(monkeypatch, custom_cuda_lib, gpu_device):
    """The same with RELION's CUDA texture projector (the production path)."""

    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    with jax.default_device(gpu_device):
        _window_union_projection_matches(texture_interp=True)
        _window_union_projection_matches(texture_interp=True, crop=10)
