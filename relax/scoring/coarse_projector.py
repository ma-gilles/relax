"""Projecting the references of pass 1: RELION's coarse texture projector, class by class and rotation block by block."""

from typing import NamedTuple

import jax.numpy as jnp

# Byte cap on the coarse projections one pass-1 call keeps for reuse across image batches (see
# ``CoarseProjector.block_once``).
_PASS1_PROJECTION_MEMO_MAX_BYTES = 2 * 1024**3


class CompactRows(NamedTuple):
    """The rows a projection returns instead of the full crop: their half-spectrum indices (a host table, in the order
    of the projected rows) and the side of the centred crop they are read from."""

    indices_np: object
    output_size: int


class CoarseProjector:
    """The class-stacked RELION projector of one pass 1 and the rows it projects.

    ``relion_projector_half`` is ``[K, z, y, x_half]``. With ``compact`` the projection returns only those rows of
    the half spectrum (``[R, len(compact.indices_np)]``, in the order of that table, from a centred crop of side
    ``compact.output_size``) instead of scattering the crop into a full image and gathering the same rows again;
    ``None`` projects the full ``current_size`` crop. ``rotated_radius`` clips each projection at the
    canonical rotated float32 radius of ``score_size`` (the score window), as the exact scorer does; ``False`` keeps
    the source-pixel disk of the legacy diagnostics. ``stable_fourier_window_shapes`` hands the kernel the runtime
    current size, so one compiled program serves every window of one physical size.

    The references and the rotation blocks are fixed for the whole pass, so a block's projections are the same for every
    image batch. :meth:`block_once` keeps them (up to ``_PASS1_PROJECTION_MEMO_MAX_BYTES``) instead of recomputing them
    per batch and per scoring sweep, which at K4 100k/256 was 11 s of the pass per iteration. The memo lives as long as
    this object, which the call drops on return.
    """

    def __init__(
        self,
        *,
        relion_projector_half,
        relion_projector_r_max: int,
        image_shape,
        projection_padding_factor: int,
        current_size: int | None,
        score_size: int,
        stable_fourier_window_shapes: bool,
        compact: CompactRows | None,
        rotated_radius: bool,
    ):
        self.relion_projector_half = relion_projector_half
        self.relion_projector_r_max = relion_projector_r_max
        self.image_shape = image_shape
        self.projection_padding_factor = projection_padding_factor
        self.current_size = current_size
        self.score_size = score_size
        self.stable_fourier_window_shapes = stable_fourier_window_shapes
        self.compact = compact
        self.rotated_radius = rotated_radius
        self._memo: dict = {}
        self._memo_bytes = 0

    @property
    def returns_compact(self) -> bool:
        """Whether projections come as the scored rows only."""

        return self.compact is not None

    def compact_rows(self, class_index, rots_b, *, return_abs2: bool):
        """Project the exact compact rows shared by direct and GEMM scoring: ``(projected, abs2 or None)``."""

        from relax.helpers import projection as projection_helpers

        projected, projected_abs2 = projection_helpers.compute_relion_projector_projections_block(
            self.relion_projector_half[class_index],
            rots_b,
            self.image_shape,
            r_max=int(self.relion_projector_r_max),
            padding_factor=int(self.projection_padding_factor),
            return_abs2=return_abs2,
            centered_rows=True,
            dense_scale=True,
            projector_output_size=int(self.compact.output_size),
            # Keep the already-host-resident table on the host so validation
            # cannot materialize its JAX mirror once per score block.
            pixel_indices=self.compact.indices_np,
            relion_texture_interp=True,
            relion_kernel="coarse",
            # Certificate and exact scorer share the canonical rotated
            # float32 radius. Explicit legacy diagnostics may retain the
            # source-pixel disk without changing projection storage.
            mask_current_image_disk=not self.rotated_radius,
            image_r_max=(jnp.asarray(self.score_size // 2, dtype=jnp.int32) if self.rotated_radius else None),
            current_image_mask_size=(
                jnp.asarray(self.score_size, dtype=jnp.int32) if self.stable_fourier_window_shapes else None
            ),
        )
        return projected, projected_abs2

    def block(self, class_index, rots_b):
        """One class's projection of a rotation block with its squared magnitudes: the compact rows, or the crop."""

        if self.returns_compact:
            return self.compact_rows(class_index, rots_b, return_abs2=True)
        from relax.helpers import projection as projection_helpers

        projector_kwargs = {}
        if self.current_size is not None:
            projector_kwargs["projector_output_size"] = int(self.current_size)
        return projection_helpers.compute_relion_projector_projections_block(
            self.relion_projector_half[class_index],
            rots_b,
            self.image_shape,
            r_max=int(self.relion_projector_r_max),
            padding_factor=int(self.projection_padding_factor),
            centered_rows=True,
            dense_scale=True,
            relion_texture_interp=True,
            relion_kernel="coarse",
            **projector_kwargs,
        )

    def block_once(self, class_index, rots_b, *, rotation_start):
        """:meth:`block`, kept for the next image batch while the kept projections fit under the byte cap."""

        key = (int(class_index), int(rotation_start), int(rots_b.shape[0]))
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        projected = self.block(class_index, rots_b)
        block_bytes = sum(int(value.size) * int(value.dtype.itemsize) for value in projected)
        if self._memo_bytes + block_bytes <= _PASS1_PROJECTION_MEMO_MAX_BYTES:
            self._memo[key] = projected
            self._memo_bytes += block_bytes
        return projected

    def cache_block(self, rotations, table_index, start, stop):
        """The rows of class ``table_index`` for ``rotations[start:stop]`` without squared magnitudes, for the cache build."""

        projected_reference, projected_reference_abs2 = self.compact_rows(
            table_index,
            rotations[start:stop],
            return_abs2=False,
        )
        if projected_reference_abs2 is not None:
            raise RuntimeError(
                "coarse GEMM cache build unexpectedly materialized abs2",
            )
        return projected_reference
