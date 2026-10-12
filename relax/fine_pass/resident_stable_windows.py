"""Stable (physical) window classes of the resident pass 2.

The passes of one refinement reuse each other's physical Fourier-window classes
(:func:`stable_window_class_history`, a context variable with a refinement's lifetime), so a logical current
size runs inside a quantized physical class instead of compiling its own programs. This module owns that
history, the pass's stable-window plan, the reference-model class and radius it implies, and the capacity
projection that serves a logical projector from its physical class.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging

import jax
import jax.numpy as jnp
import numpy as np

from relax.fine_pass.projection_blocks import capacity_projection_window_union, projection_window_union
from relax.fourier.fourier_window import (
    make_stable_fourier_window_shape_plan,
    stable_fourier_window_current_size,
    stable_fourier_window_quantum,
    stable_fourier_window_quantum_scope,
)
from relax.reconstruction.volume_solver import _RESIDENT_STABLE_WINDOWS_ENV
from relax.runtime.env_flags import parse_env_flag

logger = logging.getLogger(__name__)


def resident_stable_windows_requested() -> bool:
    # On by default for Class3D, auto-refine and VDAM (VDAM gate 14502014).
    return parse_env_flag(_RESIDENT_STABLE_WINDOWS_ENV, default=True)


# Physical window classes this refinement has already run, by (box, quantum).
# None outside a refinement: every pass then takes its own quantized class.
STABLE_WINDOW_CLASSES_RUN: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "relax_stable_window_classes_run", default=None
)


@contextlib.contextmanager
def stable_window_class_history(quantum: int | None = None):
    """Let the passes of one refinement reuse each other's physical window classes.

    RELION's current size settles by steps of two around its final value (80,
    82, 84 on the 5k K=1 run), and a quantum-8 ladder puts 80 and 82 in
    different classes, so each crossing compiled the resident chunk programs
    again. Inside this scope a pass whose quantized class is new takes the
    smallest class the refinement already ran that is at most one quantum
    larger. The class is capacity only (the logical window is unchanged), so
    this changes shapes, not RELION's cutoffs; the history is per refinement,
    so a run's shapes do not depend on what else the process ran.

    ``quantum`` is the refinement's class step
    (:func:`~relax.fourier.fourier_window.stable_fourier_window_quantum_scope`);
    None keeps the default.
    """

    token = STABLE_WINDOW_CLASSES_RUN.set({})
    try:
        with stable_fourier_window_quantum_scope(quantum):
            yield
    finally:
        STABLE_WINDOW_CLASSES_RUN.reset(token)


def _stable_window_physical_class(box_size: int, current_size: int, quantum: int) -> int | None:
    """The physical class to run ``current_size`` in, or None for its quantized class."""

    history = STABLE_WINDOW_CLASSES_RUN.get()
    if history is None:
        return None
    quantized = stable_fourier_window_current_size(int(current_size), int(box_size), quantum=int(quantum))
    classes = history.setdefault((int(box_size), int(quantum)), set())
    reusable = [size for size in classes if quantized < size <= quantized + int(quantum)]
    chosen = min(reusable) if reusable else quantized
    classes.add(chosen)
    return chosen


def stable_reference_volume_class(volume_size: int, volume_current_size: int) -> int:
    """The physical reference-model current size of a pass on another grid.

    The backprojector cube of images on another grid is keyed on the reference
    model's current size, not the image's; it takes the same quantized classes,
    with the run's class history, as a pass on the model grid would.
    """

    quantum = stable_fourier_window_quantum()
    physical = _stable_window_physical_class(int(volume_size), int(volume_current_size), quantum)
    if physical is None:
        physical = stable_fourier_window_current_size(int(volume_current_size), int(volume_size), quantum=quantum)
    return int(physical)


def reference_image_radius_at(image_radius, volume_current_size: int, physical_volume_current_size: int):
    """The image-space radius of a physical reference-model size, or None on one grid.

    :func:`relax.steps.shape_class_scoring.reconstruction_image_radius` is
    ``(current_size // 2) * s``; the physical class keeps the scale ``s``.
    """

    if image_radius is None:
        return None
    logical_r_max = int(volume_current_size) // 2
    if logical_r_max <= 0:
        raise ValueError(f"a reference-model current size of {volume_current_size} has no radius")
    return float(image_radius) / float(logical_r_max) * float(int(physical_volume_current_size) // 2)


def capacity_projection_applies(
    *,
    use_relion_projector,
    class_projector_halves,
    relion_projector_r_max,
    projection_padding_factor,
    projection_kwargs,
    physical_current_size,
) -> bool:
    """Whether a stable-window pass can project its fine rotations at the physical class.

    The capacity projection (``project_relion_half_capacity``, the half-storage
    texture kernel with runtime radii) serves complex64 RELION projector slabs
    at padding 1 or 2, through the texture interpolator, without RELION's exact
    image-disk mask (the kernel applies the rotated radius test itself). The
    logical crop and model radius must fit the physical class.
    """

    if not use_relion_projector or jax.default_backend() != "gpu":
        return False
    if projection_kwargs.get("mask_current_image_disk") or projection_kwargs.get("relion_texture_interp") is False:
        return False
    output_size = projection_kwargs.get("projector_output_size")
    padding = int(projection_padding_factor)
    if output_size is None or padding not in (1, 2) or int(physical_current_size) % 2:
        return False
    r_max = int(relion_projector_r_max)
    if not (0 < r_max <= int(physical_current_size) // 2 and int(output_size) <= int(physical_current_size)):
        return False
    logical_shape = (2 * padding * r_max + 3, 2 * padding * r_max + 3, padding * r_max + 2)
    return all(
        half is not None and half.dtype == jnp.complex64 and tuple(half.shape) == logical_shape
        for half in class_projector_halves
    )


def pass_window_union(
    score_indices, recon_indices=None, *, image_shape, projector_output_size, capacity_logical_output_size
):
    """The pass's projection window union: the logical crop's, or its capacity form for a capacity projection."""

    if capacity_logical_output_size is None:
        return projection_window_union(
            score_indices, recon_indices, image_shape=image_shape, projector_output_size=projector_output_size
        )
    return capacity_projection_window_union(
        score_indices,
        recon_indices,
        image_shape=image_shape,
        logical_output_size=capacity_logical_output_size,
        physical_output_size=projector_output_size,
    )


def center_pad_relion_projector_half(half, *, logical_r_max: int, physical_size: int, padding_factor: int):
    """Center-pad a logical RELION projector slab, ghost planes included, to the physical class.

    ``project_relion_half_capacity`` takes a ``(pf Q + 3, pf Q + 3, pf Q // 2 + 2)``
    slab for a physical crop of ``Q`` pixels; the logical slab sits centered on
    z and y and at the start of x, and the zeros around it are never sampled
    within the logical radius.
    """

    size = int(padding_factor) * int(physical_size) + 3
    logical = 2 * int(padding_factor) * int(logical_r_max) + 3
    offset = (size - logical) // 2
    x_pad = size // 2 + 1 - (int(padding_factor) * int(logical_r_max) + 2)
    widths = ((offset, offset), (offset, offset), (0, x_pad))
    if not isinstance(half, jax.Array):
        # A host slab is padded on the host: the device pad compiled once per logical size.
        return jnp.asarray(np.pad(np.asarray(half), widths))
    return jnp.pad(half, widths)


def resident_stable_window_plan(
    image_shape,
    *,
    current_size,
    mstep_current_size,
    n_half,
    square_window,
    window_spec_kwargs,
    firstiter_cc,
):
    """The pass's stable-window plan, or None to keep the logical window.

    The plan applies when the flag is on and the current size is below the box,
    with one current size for scoring and reconstruction on the model grid.
    The --firstiter_cc iteration (normalized-CC tiles) and a split
    score/reconstruction current size keep RELION's logical window, as the
    compact and local engines do. Images on another grid take the plan on their
    own image grid; their reference-model volume gets its own physical class in
    the caller (:func:`stable_reference_volume_class`).
    """

    if not resident_stable_windows_requested():
        return None
    reasons = []
    if int(current_size) >= int(image_shape[0]):
        reasons.append("current size at the box")
    if firstiter_cc:
        reasons.append("--firstiter_cc")
    if int(mstep_current_size) != int(current_size):
        reasons.append("reconstruction current size differs from the score's")
    if reasons:
        logger.info(
            "Resident pass-2 stable windows requested but not applicable (%s); using the logical window",
            ", ".join(reasons),
        )
        return None
    quantum = stable_fourier_window_quantum()
    return make_stable_fourier_window_shape_plan(
        image_shape,
        int(current_size),
        n_half,
        reconstruction_current_size=int(mstep_current_size),
        enabled=True,
        quantum=quantum,
        square=square_window,
        physical_current_size=_stable_window_physical_class(image_shape[0], current_size, quantum),
        **window_spec_kwargs,
    )
