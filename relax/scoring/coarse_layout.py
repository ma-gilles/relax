"""Where the coarse Gaussian pass keeps its scored Fourier rows: the square layout and its lookups.

``plan_coarse_gaussian_square_layout`` plans the compact table the coarse GEMMs read (RELION's square crop of the
current size, optionally inside a stable physical capacity); pass 1
(:mod:`relax.scoring.significance`) and the subtomogram coarse pass (:mod:`relax.scoring.tomo_coarse`) both read it.
"""

import operator
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class CoarseGaussianSquareLayout(NamedTuple):
    """Logical RELION square issue stream inside a stable physical capacity."""

    logical_current_size: int
    physical_current_size: int
    logical_square_count: int
    physical_square_count: int
    score_indices_np: np.ndarray
    score_active_mask_np: np.ndarray
    logical_projector_mask_np: np.ndarray
    full_to_compact_np: np.ndarray


def plan_coarse_gaussian_square_layout(
    image_shape,
    logical_current_size: int,
    active_score_indices,
    *,
    stable_fourier_window_shapes: bool,
) -> CoarseGaussianSquareLayout:
    """Plan coarse storage without changing RELION's logical pixel traversal.

    The physical compact table is ordered as the complete logical square
    followed by physical-only capacity rows.  The direct CUDA lookup retains
    the old logical full-pixel permutation as its prefix; its tail visits only
    zero-weight capacity rows.  Consequently exact rescoring keeps every
    logical lane assignment and appends only fused multiply-adds by zero.
    """

    from relax.helpers.fourier_window import (
        make_fourier_window_indices_np,
        make_frequency_coords_half_np,
        stable_fourier_window_current_size,
        stable_fourier_window_quantum,
    )
    from relax.sparse_pass2.sparse_pass2_scoring import _relion_cuda_fine_full_to_compact_lookup

    image_shape = tuple(int(value) for value in image_shape)
    physical_current_size = (
        stable_fourier_window_current_size(
            logical_current_size,
            image_shape[0],
            quantum=stable_fourier_window_quantum(),
        )
        if stable_fourier_window_shapes
        else logical_current_size
    )
    logical_indices, logical_count = make_fourier_window_indices_np(
        image_shape,
        logical_current_size,
        square=True,
        include_dc=True,
    )
    physical_indices, physical_count = make_fourier_window_indices_np(
        image_shape,
        physical_current_size,
        square=True,
        include_dc=True,
    )
    logical_indices = np.asarray(logical_indices, dtype=np.int32)
    physical_indices = np.asarray(physical_indices, dtype=np.int32)
    expected_logical_count = logical_current_size * (logical_current_size // 2 + 1)
    expected_physical_count = physical_current_size * (physical_current_size // 2 + 1)
    if logical_count != expected_logical_count or physical_count != expected_physical_count:
        raise ValueError(
            "RELION coarse Gaussian square crop has an unexpected size: "
            f"logical={logical_count}/{expected_logical_count}, "
            f"physical={physical_count}/{expected_physical_count}"
        )
    if np.setdiff1d(logical_indices, physical_indices, assume_unique=True).size:
        raise ValueError("stable coarse physical square does not contain logical support")

    physical_tail = np.setdiff1d(
        physical_indices,
        logical_indices,
        assume_unique=True,
    ).astype(np.int32, copy=False)
    score_indices_np = np.concatenate((logical_indices, physical_tail)).astype(
        np.int32,
        copy=False,
    )
    if score_indices_np.size != physical_count or np.unique(score_indices_np).size != physical_count:
        raise ValueError("stable coarse compact score rows are not a unique physical square")

    logical_lookup = np.asarray(
        _relion_cuda_fine_full_to_compact_lookup(
            image_shape,
            logical_current_size,
            logical_indices,
        ),
        dtype=np.int32,
    )
    full_to_compact_np = np.concatenate(
        (
            logical_lookup,
            np.arange(logical_count, physical_count, dtype=np.int32),
        )
    )
    active_score_indices = np.asarray(active_score_indices, dtype=np.int32).reshape(-1)
    score_active_mask_np = np.isin(score_indices_np, active_score_indices)
    if physical_count > logical_count:
        score_active_mask_np[logical_count:] = False
    compact_coords = np.rint(
        make_frequency_coords_half_np(image_shape)[score_indices_np],
    ).astype(np.int64)
    logical_projector_mask_np = np.sum(compact_coords**2, axis=1) <= (
        logical_current_size // 2
    ) ** 2
    if physical_count > logical_count:
        logical_projector_mask_np[logical_count:] = False

    return CoarseGaussianSquareLayout(
        logical_current_size=logical_current_size,
        physical_current_size=physical_current_size,
        logical_square_count=logical_count,
        physical_square_count=physical_count,
        score_indices_np=score_indices_np,
        score_active_mask_np=np.asarray(score_active_mask_np, dtype=np.bool_),
        logical_projector_mask_np=np.asarray(
            logical_projector_mask_np,
            dtype=np.bool_,
        ),
        full_to_compact_np=full_to_compact_np,
    )


def coarse_gaussian_fused_logical_lookup(
    full_to_compact,
    square_layout: CoarseGaussianSquareLayout,
    *,
    current_size: int,
):
    """Return the logical fused-ABI prefix from a stable physical lookup."""

    current_size = operator.index(current_size)
    logical_count = current_size * (current_size // 2 + 1)
    if (
        int(square_layout.logical_current_size) != current_size
        or int(square_layout.logical_square_count) != logical_count
    ):
        raise ValueError(
            "fused coarse logical lookup does not match current_size: "
            f"layout={square_layout.logical_current_size}/"
            f"{square_layout.logical_square_count}, expected={current_size}/"
            f"{logical_count}",
        )
    lookup = jnp.asarray(full_to_compact)
    if lookup.ndim != 1 or lookup.dtype != jnp.int32:
        raise TypeError("fused coarse full_to_compact lookup must be rank-1 int32")
    if int(lookup.shape[0]) < logical_count:
        raise ValueError(
            "fused coarse full_to_compact lookup is shorter than its logical prefix",
        )
    return lookup[:logical_count]
