"""Projection blocks of the sparse pass 2.

The projection of hypothesis rotations (full and windowed) with their chunking and
finalization. The resident drivers project their fine rotations through these owners.

Radius propagation: ``docs/math/sparse_projection_radius.md``.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from relax.helpers.projection import compute_projections_block as _compute_projections_block
from relax.helpers.projection import (
    compute_relion_projector_projections_block as _compute_relion_projector_projections_block,
)
from relax.sparse_pass2.sparse_pass2_budget import _MAX_PROJECTED_ROTATIONS_ENV, _optional_positive_int_env

logger = logging.getLogger(__name__)


def _compute_sparse_pass2_projections_block(
    mean_for_proj,
    rotations_block,
    image_shape,
    proj_volume_shape,
    disc_type,
    *,
    max_projected_rotations: int | None = None,
    output_complex_dtype=None,
    output_abs2_dtype=None,
    relion_projector_half=None,
    relion_projector_texture=None,
    relion_projector_r_max: int | None = None,
    projection_padding_factor: int = 1,
    projector_output_size: int | None = None,
    relion_projector_capacity_texture=None,
    pixel_indices=None,
    relion_projector_runtime_r_max=None,
    relion_projector_image_size=None,
    **projection_kwargs,
):
    """One rotation block's projections, full half-spectrum rows or, with ``pixel_indices``, those pixels.

    ``pixel_indices`` (RELION projector only) are centered-row half-spectrum
    pixels: the projector gathers them from its crop instead of expanding the
    crop to the full half, the same values the full row holds there.

    ``relion_projector_runtime_r_max`` and ``relion_projector_image_size`` (int32
    scalars, with ``relion_projector_r_max`` the static sentinel 0) project a
    center-padded capacity slab into a ``projector_output_size`` capacity crop with
    the logical model and image radii as runtime values
    (:func:`~relax.cuda.kernels.project_relion_half_capacity`), so the programs
    keep one shape across logical current sizes; the logical pixels hold the
    logical projection (``relax/relion/relion_project.py``).
    """

    projection_kwargs = dict(projection_kwargs)
    return_abs2 = projection_kwargs.pop("return_abs2", True)
    # Pass 2 and the weighted sums use RELION's fine and wavg kernels; a local parent pass
    # (RELION's pass 1) names the coarse kernel.
    relion_kernel = projection_kwargs.pop("relion_kernel", "fine")
    # The generic projector also needs this cutoff; consuming it here made
    # sparse pass 2 silently use a different radius from coarse scoring.
    projection_max_r = projection_kwargs.get("max_r", None)
    projection_relion_texture_interp = projection_kwargs.get("relion_texture_interp")
    projection_mask_current_image_disk = bool(
        projection_kwargs.pop("mask_current_image_disk", False)
    )
    if projector_output_size is None and projection_max_r is not None:
        projector_output_size = int(2 * float(projection_max_r))
    use_relion_projector = relion_projector_half is not None or relion_projector_texture is not None
    if use_relion_projector and relion_projector_r_max is None:
        raise ValueError("relion_projector_r_max is required when relion_projector_half is provided")
    if relion_projector_capacity_texture is not None and relion_projector_half is None:
        raise ValueError("a capacity projector texture projects relion_projector_half; none was given")
    if pixel_indices is not None and not use_relion_projector:
        raise ValueError("pixel_indices select the RELION projector's pixels; no RELION projector was given")

    def _project(rotations):
        if (
            relion_projector_runtime_r_max is not None
            and relion_kernel == "fine"
            and relion_projector_texture is None
            and relion_projector_capacity_texture is None
        ):
            return _capacity_projections_block(
                relion_projector_half,
                rotations,
                pixel_indices,
                relion_projector_runtime_r_max,
                relion_projector_image_size,
                image_shape=tuple(int(v) for v in image_shape),
                r_max=int(relion_projector_r_max),
                padding_factor=int(projection_padding_factor),
                return_abs2=bool(return_abs2),
                relion_texture_interp=projection_relion_texture_interp,
                projector_output_size=projector_output_size,
                mask_current_image_disk=projection_mask_current_image_disk,
            )
        if use_relion_projector:
            return _compute_relion_projector_projections_block(
                relion_projector_half,
                rotations,
                image_shape,
                r_max=int(relion_projector_r_max),
                padding_factor=int(projection_padding_factor),
                return_abs2=bool(return_abs2),
                centered_rows=True,
                dense_scale=True,
                relion_texture_interp=projection_relion_texture_interp,
                projector_output_size=projector_output_size,
                relion_kernel=relion_kernel,
                mask_current_image_disk=projection_mask_current_image_disk,
                pixel_indices=pixel_indices,
                **(
                    {
                        "projector_capacity": True,
                        "runtime_r_max": relion_projector_runtime_r_max,
                        "current_image_mask_size": relion_projector_image_size,
                    }
                    if relion_projector_runtime_r_max is not None
                    else {}
                ),
                **({"persistent_texture": relion_projector_texture} if relion_projector_texture is not None else {}),
                **(
                    {"capacity_texture": relion_projector_capacity_texture}
                    if relion_projector_capacity_texture is not None
                    else {}
                ),
            )
        return _compute_projections_block(
            mean_for_proj,
            rotations,
            image_shape,
            proj_volume_shape,
            disc_type,
            return_abs2=bool(return_abs2),
            **projection_kwargs,
        )

    if max_projected_rotations is None:
        max_projected_rotations = _optional_positive_int_env(_MAX_PROJECTED_ROTATIONS_ENV)
    if max_projected_rotations is None:
        proj_half, proj_abs2 = _project(rotations_block)
        if output_complex_dtype is not None:
            proj_half = proj_half.astype(output_complex_dtype)
        if proj_abs2 is not None and output_abs2_dtype is not None:
            proj_abs2 = proj_abs2.astype(output_abs2_dtype)
        return proj_half, proj_abs2

    n_rotations = int(rotations_block.shape[0])
    max_projected_rotations = max(1, int(max_projected_rotations))
    if n_rotations <= max_projected_rotations:
        proj_half, proj_abs2 = _project(rotations_block)
        if output_complex_dtype is not None:
            proj_half = proj_half.astype(output_complex_dtype)
        if proj_abs2 is not None and output_abs2_dtype is not None:
            proj_abs2 = proj_abs2.astype(output_abs2_dtype)
        return proj_half, proj_abs2

    proj_chunks = []
    abs2_chunks = []
    for start in range(0, n_rotations, max_projected_rotations):
        stop = min(start + max_projected_rotations, n_rotations)
        proj_chunk, abs2_chunk = _project(rotations_block[start:stop])
        if output_complex_dtype is not None:
            proj_chunk = proj_chunk.astype(output_complex_dtype)
        if abs2_chunk is not None and output_abs2_dtype is not None:
            abs2_chunk = abs2_chunk.astype(output_abs2_dtype)
        proj_chunks.append(proj_chunk)
        abs2_chunks.append(abs2_chunk)

    proj_half = jnp.concatenate(proj_chunks, axis=0)
    if all(abs2_chunk is None for abs2_chunk in abs2_chunks):
        return proj_half, None
    if any(abs2_chunk is None for abs2_chunk in abs2_chunks):
        raise RuntimeError("Inconsistent projection abs2 chunks")
    return proj_half, jnp.concatenate(abs2_chunks, axis=0)


@partial(
    jax.jit,
    static_argnames=(
        "image_shape", "r_max", "padding_factor", "return_abs2", "relion_texture_interp",
        "projector_output_size", "mask_current_image_disk",
    ),
)
def _capacity_projections_block(
    relion_projector_half,
    rotations,
    pixel_indices,
    runtime_r_max,
    image_size,
    *,
    image_shape,
    r_max,
    padding_factor,
    return_abs2,
    relion_texture_interp,
    projector_output_size,
    mask_current_image_disk,
):
    """The capacity route's fine-kernel block as one program per shape.

    Called eagerly, its radius masks, kernel call, crop gather and row zeroing
    dispatched about twenty small programs per block; a VDAM pass builds its
    projection cache in a dozen blocks every iteration. The operations are the
    same (``_compute_relion_projector_projections_block``); only the coarse
    kernel's row relabel reads the rotations on the host, so it stays eager.
    """

    return _compute_relion_projector_projections_block(
        relion_projector_half,
        rotations,
        image_shape,
        r_max=r_max,
        padding_factor=padding_factor,
        return_abs2=return_abs2,
        centered_rows=True,
        dense_scale=True,
        relion_texture_interp=relion_texture_interp,
        projector_output_size=projector_output_size,
        relion_kernel="fine",
        mask_current_image_disk=mask_current_image_disk,
        pixel_indices=pixel_indices,
        projector_capacity=True,
        runtime_r_max=runtime_r_max,
        current_image_mask_size=image_size,
    )


def _projection_kwargs_for_relion_score_window(
    projection_kwargs,
    *,
    use_relion_projector: bool,
    current_size: int | None,
):
    """Keep the RELION projector crop large enough for the particle image.

    ``r_max`` describes the model sphere, but it does not always describe the
    particle-image crop.  In particular, fresh first-iteration CC can score a
    size-58 particle image from a projector whose model ``r_max`` is 28.  A
    crop inferred as ``2 * r_max == 56`` drops the valid ``ky=-28`` row before
    the score window gathers it.  RELION projects into the particle-image box
    and clips samples independently to the model sphere, so preserve that
    distinction here.
    """

    kwargs = dict(projection_kwargs)
    if use_relion_projector:
        if current_size is None:
            raise ValueError("windowed RELION projection requires current_size")
        kwargs["projector_output_size"] = int(current_size)
    return kwargs


def _compute_sparse_pass2_windowed_projections_block(
    mean_for_proj,
    rotations_block,
    image_shape,
    proj_volume_shape,
    disc_type,
    *,
    score_indices,
    recon_indices=None,
    max_projected_rotations: int | None = None,
    output_complex_dtype=None,
    output_abs2_dtype=None,
    relion_projector_half=None,
    relion_projector_texture=None,
    relion_projector_r_max: int | None = None,
    projection_padding_factor: int = 1,
    window_union: "ProjectionWindowUnion | None" = None,
    output_rows: int | None = None,
    outputs=None,
    output_row_ids=None,
    **projection_kwargs,
):
    """Project in capped chunks and retain only score/reconstruction windows.

    ``output_rows`` (at least the number of rotations) makes the outputs that
    many rows long, zero past the projected rotations, so a caller that needs
    a fixed-length array never pads a second copy.

    ``output_row_ids`` (host int ``[n_rotations]``, ascending and distinct) puts
    rotation ``i`` at that output row instead of row ``i``; an id at or past the
    output length is dropped. ``outputs`` (``(score, recon, |recon|^2)`` from an
    earlier call, donated) are written into instead of zero-filled new arrays,
    so several calls fill one set of arrays (:func:`project_rows_by_class`).

    With ``window_union`` (:func:`projection_window_union` of these windows) a
    RELION projector projects only the union's pixels and the windows are
    taken from it, instead of expanding every row to the full half spectrum
    first: at EMPIAR-10202 box 800 a row is 320,800 pixels for a ~155,000-pixel
    window (bigbox 14558025).
    """

    if max_projected_rotations is None:
        max_projected_rotations = _optional_positive_int_env(_MAX_PROJECTED_ROTATIONS_ENV)

    projection_kwargs = dict(projection_kwargs)
    projection_kwargs["return_abs2"] = False
    pixel_indices = None
    if window_union_applies(
        window_union,
        relion_projector=relion_projector_half is not None or relion_projector_texture is not None,
        projection_kwargs=projection_kwargs,
    ):
        if (window_union.recon_take is None) != (recon_indices is None):
            raise ValueError("the window union does not describe these windows")
        pixel_indices = window_union.indices
        score_indices, recon_indices = window_union.score_take, window_union.recon_take
    score_indices = jnp.asarray(score_indices, dtype=jnp.int32)
    recon_indices = None if recon_indices is None else jnp.asarray(recon_indices, dtype=jnp.int32)

    n_rotations = int(rotations_block.shape[0])
    if outputs is not None:
        n_output_rows = int(outputs[0].shape[0])
    else:
        n_output_rows = n_rotations if output_rows is None else int(output_rows)
    if output_row_ids is None:
        if n_output_rows < n_rotations:
            raise ValueError(f"output_rows={n_output_rows} is fewer than the {n_rotations} rotations")
        output_row_ids = np.arange(n_rotations, dtype=np.int32)
    else:
        output_row_ids = np.asarray(output_row_ids, dtype=np.int32)
        if output_row_ids.shape != (n_rotations,) or np.any(np.diff(output_row_ids) <= 0):
            raise ValueError("output_row_ids must be one ascending, distinct row per rotation")
    if max_projected_rotations is None:
        chunk_ranges = [(0, n_rotations)]
    else:
        max_projected_rotations = max(1, int(max_projected_rotations))
        chunk_ranges = [
            (start, min(start + max_projected_rotations, n_rotations))
            for start in range(0, n_rotations, max_projected_rotations)
        ]

    # Each chunk's windows are written into the outputs in place: concatenating
    # the chunks held a streamed chunk's projections twice, outside every plan
    # (bench 14572645, K=1 50k/256 at current size 178: 7.70 GiB).
    score_proj, recon_proj, recon_abs2 = (None, None, None) if outputs is None else outputs
    for start, stop in chunk_ranges:
        proj_chunk, _ = _compute_sparse_pass2_projections_block(
            mean_for_proj,
            rotations_block[start:stop],
            image_shape,
            proj_volume_shape,
            disc_type,
            max_projected_rotations=None,
            relion_projector_half=relion_projector_half,
            relion_projector_texture=relion_projector_texture,
            relion_projector_r_max=relion_projector_r_max,
            projection_padding_factor=projection_padding_factor,
            pixel_indices=pixel_indices,
            **projection_kwargs,
        )
        if pixel_indices is not None:
            proj_chunk = with_zero_column(proj_chunk)
        if score_proj is not None:
            complex_dtype = score_proj.dtype
            abs2_dtype = None if recon_abs2 is None else recon_abs2.dtype
        else:
            complex_dtype = jnp.dtype(proj_chunk.dtype if output_complex_dtype is None else output_complex_dtype)
            abs2_dtype = jnp.dtype(
                jnp.finfo(complex_dtype).dtype if output_abs2_dtype is None else output_abs2_dtype
            )
            score_proj = jnp.zeros((n_output_rows, int(score_indices.shape[0])), dtype=complex_dtype)
            if recon_indices is not None:
                recon_proj = jnp.zeros((n_output_rows, int(recon_indices.shape[0])), dtype=complex_dtype)
                recon_abs2 = jnp.zeros((n_output_rows, int(recon_indices.shape[0])), dtype=abs2_dtype)
        rows = jnp.asarray(output_row_ids[start:stop])
        if recon_indices is None:
            score_proj = _place_score_window_block(
                score_proj, proj_chunk, score_indices, rows, output_complex_dtype=complex_dtype
            )
        else:
            score_proj, recon_proj, recon_abs2 = _place_windowed_projection_block(
                score_proj,
                recon_proj,
                recon_abs2,
                proj_chunk,
                score_indices,
                recon_indices,
                rows,
                output_complex_dtype=complex_dtype,
                output_abs2_dtype=abs2_dtype,
            )
        del proj_chunk
    return score_proj, recon_proj, recon_abs2


def class_call_length(n_rows: int) -> int:
    """Rotations one class's projection call projects for its ``n_rows`` rows.

    The rows rounded up to a multiple of an eighth of the power of two at or above them, at
    least 256: the programs see eight lengths per octave and a call of more than 256 rows
    projects at most a quarter more than its rows. A whole 8192-rotation stream quantum per class projected up to
    ``K * 8191`` padding rotations per K>1 chunk (K15 VDAM it150: 19-21 s of every 120 s
    of the main thread in the streamed projections, e2e job 14881673).
    """

    n_rows = int(n_rows)
    quantum = max(256, (1 << int(n_rows - 1).bit_length()) // 8)
    return -(-n_rows // quantum) * quantum


def project_rows_by_class(project, matrices, row_class, *, n_rows: int):
    """``(score, recon, |recon|^2)`` of every row, each projected from its own class's reference.

    ``project(matrices, class_index=, n_rows=, outputs=, output_row_ids=, finalize=)`` is a
    windowed projector (:func:`_compute_sparse_pass2_windowed_projections_block` behind the
    caller's units). Each class projects its own rows in one call, padded with copies of its
    first rotation to :func:`class_call_length` rotations so the programs see few distinct
    lengths; every call writes its rows in place into one set of ``n_rows``-long arrays and
    drops the padding. So the K>1 projections of a chunk are one ``n_rows`` copy, as at K=1: the
    classes' padded calls were concatenated and gathered before, up to ``K`` padded calls
    next to the gather, outside the chunk plan (K15 VDAM OOM at it151, 62ab6ae).
    ``finalize`` is true on the last call only, so a caller's per-array finishing step
    sees every row exactly once.
    """

    row_class = np.asarray(row_class, dtype=np.int64)
    classes = np.unique(row_class)
    outputs = None
    for call, class_index in enumerate(classes):
        positions = np.flatnonzero(row_class == class_index)
        n_call = class_call_length(positions.size)
        n_pad = n_call - positions.size
        padded = np.concatenate([positions, np.full(n_pad, positions[0], dtype=np.int64)])
        row_ids = np.concatenate([positions, int(n_rows) + np.arange(n_pad, dtype=np.int64)])
        outputs = project(
            matrices[padded],
            class_index=int(class_index),
            n_rows=int(n_rows),
            outputs=outputs,
            output_row_ids=row_ids,
            finalize=call == classes.size - 1,
        )
    return outputs


class ProjectionWindowUnion(NamedTuple):
    """The pixels a RELION projector computes for two windows, and where each window sits in them.

    ``indices`` are the windows' pixels inside the projector crop of
    ``projector_output_size``; a window pixel outside the crop is zero in the
    full row, so its take position is ``len(indices)``, a zero column appended
    to the projected pixels.
    """

    indices: jax.Array  # int32 [U]: sorted in-crop pixels of the score and reconstruction windows
    score_take: jax.Array  # int32 [N_score]: positions in ``indices`` + one zero column
    recon_take: jax.Array | None  # int32 [N_recon], or None for a score-only window
    projector_output_size: int


def projection_window_union(
    score_indices, recon_indices=None, *, image_shape, projector_output_size
) -> ProjectionWindowUnion:
    """:class:`ProjectionWindowUnion` of host pixel windows; computed once per pass, not per call."""

    from relax.helpers.projection import centered_relion_projector_crop_mask

    score_np = np.asarray(score_indices, dtype=np.int64)
    recon_np = None if recon_indices is None else np.asarray(recon_indices, dtype=np.int64)
    union = np.unique(score_np) if recon_np is None else np.union1d(score_np, recon_np)
    union = union[
        centered_relion_projector_crop_mask(
            union, image_shape=image_shape, projector_output_size=int(projector_output_size)
        )
    ]

    def take(window):
        position = np.searchsorted(union, window)
        clipped = np.minimum(position, max(union.size - 1, 0))
        inside = (position < union.size) & (union[clipped] == window) if union.size else np.zeros(window.shape, bool)
        # Cast on the host: a device cast of the int64 array compiled once per window size.
        return jnp.asarray(np.where(inside, position, union.size).astype(np.int32))

    return ProjectionWindowUnion(
        indices=jnp.asarray(union.astype(np.int32)),
        score_take=take(score_np),
        recon_take=None if recon_np is None else take(recon_np),
        projector_output_size=int(projector_output_size),
    )


def capacity_projection_window_union(
    score_indices, recon_indices=None, *, image_shape, logical_output_size, physical_output_size
) -> ProjectionWindowUnion:
    """The window union of a capacity projection: logical pixels, physical-class shapes.

    A projection into the physical crop holds pixels the logical crop does not
    have, RELION's ky = -L/2 row among them (its crop keeps the positive Nyquist
    row), so membership stays the logical crop's: every other window pixel takes
    the zero column. The in-crop index list is padded, by repeating its last
    pixel, to the physical crop's union size, so the projection programs keep one
    shape per physical class. The padded pixels are projected and never taken.
    """

    logical = projection_window_union(
        score_indices, recon_indices, image_shape=image_shape, projector_output_size=int(logical_output_size)
    )
    physical = projection_window_union(
        score_indices, recon_indices, image_shape=image_shape, projector_output_size=int(physical_output_size)
    )
    indices = np.asarray(logical.indices, dtype=np.int64)
    n_padded = int(physical.indices.shape[0])
    if indices.size == 0 or indices.size > n_padded:
        raise ValueError(
            f"a capacity window union needs a nonempty logical union inside the physical one "
            f"({indices.size} logical, {n_padded} physical pixels)"
        )
    padded = np.pad(indices, (0, n_padded - indices.size), mode="edge")

    def retake(take):
        take = np.asarray(take, dtype=np.int64)
        return jnp.asarray(np.where(take == indices.size, n_padded, take), dtype=jnp.int32)

    return ProjectionWindowUnion(
        indices=jnp.asarray(padded, dtype=jnp.int32),
        score_take=retake(logical.score_take),
        recon_take=None if logical.recon_take is None else retake(logical.recon_take),
        projector_output_size=int(physical_output_size),
    )


def window_union_applies(window_union, *, relion_projector: bool, projection_kwargs) -> bool:
    """Whether ``window_union`` describes this RELION projector's crop."""

    return (
        window_union is not None
        and relion_projector
        and projection_kwargs.get("projector_output_size") is not None
        and int(projection_kwargs["projector_output_size"]) == int(window_union.projector_output_size)
    )


@jax.jit
def with_zero_column(proj_block):
    """``proj_block`` with one zero column appended, the value of every out-of-crop window pixel."""

    return jnp.pad(proj_block, ((0, 0), (0, 1)))


def _place_rows(values, block, rows):
    """``values`` with ``block``'s rows written at ``rows`` (ascending, distinct); rows past the end are dropped."""

    return values.at[rows].set(block, mode="drop", indices_are_sorted=True, unique_indices=True)


@partial(jax.jit, static_argnames=("output_complex_dtype",), donate_argnums=(0,))
def _place_score_window_block(score_proj, proj_block, score_indices, rows, *, output_complex_dtype):
    """Write one projector block's score window into ``score_proj`` (donated) at ``rows``."""

    return _place_rows(score_proj, proj_block[:, score_indices].astype(output_complex_dtype), rows)


@partial(jax.jit, static_argnames=("output_complex_dtype", "output_abs2_dtype"), donate_argnums=(0, 1, 2))
def _place_windowed_projection_block(
    score_proj,  # [C_R, N_score], donated
    recon_proj,  # [C_R, N_recon], donated
    recon_abs2,  # [C_R, N_recon], donated
    proj_block,  # [Q, N_half]
    score_indices,
    recon_indices,
    rows,  # int32 [Q], runtime: ascending output rows, past the end dropped
    *,
    output_complex_dtype,
    output_abs2_dtype,
):
    """Window one projector block and write it, with ``|recon|^2``, into ``rows``."""

    score_block = proj_block[:, score_indices].astype(output_complex_dtype)
    recon_block = proj_block[:, recon_indices].astype(output_complex_dtype)
    abs2_block = (jnp.abs(recon_block) ** 2).astype(output_abs2_dtype)
    return (
        _place_rows(score_proj, score_block, rows),
        _place_rows(recon_proj, recon_block, rows),
        _place_rows(recon_abs2, abs2_block, rows),
    )


