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
from dataclasses import replace as dataclass_replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from relax.fine_pass import resident_chunk_records
from relax.fine_pass.projection_blocks import capacity_projection_window_union, projection_window_union
from relax.fine_pass.scoring import _relion_cuda_fine_full_to_compact_lookup
from relax.fine_pass.wavg import make_relion_wavg_rectangle, make_stable_relion_wavg_rectangle
from relax.fourier.fourier_window import (
    make_stable_fourier_window_shape_plan,
    stable_fourier_window_current_size,
)
from relax.projection.adjoint import ReferenceSphereClip, mstep_adjoint_max_r
from relax.reconstruction.half_volume_mstep import (
    crop_public_full_volume,
    crop_relion_x_half_accumulator,
    finalize_half_volume_bpref,
    physical_bpref_finalize_applies,
    relion_backprojector_volume_shape,
    relion_x_half_accumulators_to_public_layout,
)

logger = logging.getLogger(__name__)


# Physical window classes this refinement has already run, by (box, quantum).
# None outside a refinement: every pass then takes its own quantized class.
STABLE_WINDOW_CLASSES_RUN: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "relax_stable_window_classes_run", default=None
)


@contextlib.contextmanager
def stable_window_class_history():
    """Let the passes of one refinement reuse each other's physical window classes.

    RELION's current size settles by steps of two around its final value (80,
    82, 84 on the 5k K=1 run), and a quantum-8 ladder puts 80 and 82 in
    different classes, so each crossing compiled the resident chunk programs
    again. Inside this scope a pass whose quantized class is new takes the
    smallest class the refinement already ran that is at most one quantum
    larger. The class is capacity only (the logical window is unchanged), so
    this changes shapes, not RELION's cutoffs; the history is per refinement,
    so a run's shapes do not depend on what else the process ran. The history
    is keyed by box and quantum (``EngineTuning.stable_window_quantum``, which
    the passes receive).
    """

    token = STABLE_WINDOW_CLASSES_RUN.set({})
    try:
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


def stable_reference_volume_class(volume_size: int, volume_current_size: int, *, quantum: int) -> int:
    """The physical reference-model current size of a pass on another grid.

    The backprojector cube of images on another grid is keyed on the reference
    model's current size, not the image's; it takes the same quantized classes,
    with the run's class history, as a pass on the model grid would
    (``quantum``: ``EngineTuning.stable_window_quantum``).
    """

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
    requested: bool,
    quantum: int,
):
    """The pass's stable-window plan, or None to keep the logical window.

    The plan applies when ``requested`` (``EngineTuning.stable_windows``) and the current size is below the box,
    in classes of ``quantum`` (``EngineTuning.stable_window_quantum``),
    with one current size for scoring and reconstruction on the model grid.
    The --firstiter_cc iteration (normalized-CC tiles) and a split
    score/reconstruction current size keep RELION's logical window, in the
    global and the local resident passes alike. Images on another grid take the plan on their
    own image grid; their reference-model volume gets its own physical class in
    the caller (:func:`stable_reference_volume_class`).
    """

    if not requested:
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


# ---------------------------------------------------------------------------
# What a stable-window pass sets up from its plan (the global and the local resident passes alike)
# ---------------------------------------------------------------------------


def stable_window_spec(plan):
    """The pass's window spec at the physical class, with RELION's logical cutoffs.

    Capacity changes storage, not the projection or reconstruction cutoffs: the projector keeps the
    logical radius and crop, so the tail's projections never enter a logical pixel.
    """

    return dataclass_replace(
        plan.packed_physical_spec(),
        max_r=plan.logical_spec.max_r,
        projection_max_r=plan.logical_spec.projection_max_r,
        image_current_size=plan.logical_spec.image_current_size,
    )


def log_stable_window_plan(label: str, plan) -> None:
    logger.info(
        "%s stable windows: logical current_size %d -> physical class %d "
        "(score pixels %d -> %d, recon pixels %d -> %d, Wavg rectangle %d -> %d)",
        label,
        plan.logical_current_size,
        plan.physical_current_size,
        plan.logical_score_pixels,
        plan.physical_score_pixels,
        plan.logical_reconstruction_pixels,
        plan.physical_reconstruction_pixels,
        plan.logical_rectangle_pixels,
        plan.physical_rectangle_pixels,
    )


def score_weights_and_lookup(half_weights_windowed, *, image_shape, current_size, window_indices_np, plan):
    """The score window's weights and the fine scorer's full-to-compact lookup, at the plan's class.

    Without a plan: the weights as given and the lookup of the whole window. With one, the score tail gets
    no weight, and the lookup covers the logical rectangle only (the runtime kernel never reads past it);
    its -1 pad keeps the buffer at the physical class's size.
    """

    if plan is None:
        return half_weights_windowed, jnp.asarray(
            _relion_cuda_fine_full_to_compact_lookup(image_shape, current_size, window_indices_np),
            dtype=jnp.int32,
        )
    half_weights_windowed = times_logical_mask(half_weights_windowed, jnp.int32(int(plan.logical_score_pixels)))
    logical_lookup = _relion_cuda_fine_full_to_compact_lookup(
        image_shape,
        current_size,
        window_indices_np[: int(plan.logical_score_pixels)],
    )
    return half_weights_windowed, jnp.asarray(
        np.pad(
            logical_lookup,
            (0, int(plan.physical_rectangle_pixels) - logical_lookup.size),
            constant_values=-1,
        ),
        dtype=jnp.int32,
    )


def logical_score_indices(window_indices, window_indices_np, plan):
    """The score window's logical pixels: the whole window, or the plan's logical prefix."""

    return window_indices if plan is None else window_indices_np[: int(plan.logical_score_pixels)]


def stable_noise_tables(
    shell_indices_noise,
    noise_variance_for_noise,
    *,
    plan,
    image_shape,
    wsum_current_size,
    recon_window_indices,
    mstep_current_size,
    n_recon_windowed,
) -> tuple:
    """``(shell_indices_noise, noise_variance_for_noise, relion_wavg_rectangle, logical_rect_pixels,
    logical_recon_pixels)``: the noise tables over the reconstruction window, the Wavg rectangle and the
    logical pixel counts.

    With a plan, the recon tail holds real pixels above RELION's cutoff. They take the shell sentinel and
    zero noise variance, so every residual, power and scale term they could enter is exactly zero; the
    M-step masks their sums (``resident_mstep_blocks._resident_mstep_block``) and the Wavg kernels stop at
    the logical rectangle, whose layout is RELION's byte for byte.
    """

    if plan is None:
        rectangle = make_relion_wavg_rectangle(
            image_shape,
            wsum_current_size,
            recon_window_indices,
            reconstruction_current_size=mstep_current_size,
        )
        return (
            shell_indices_noise,
            noise_variance_for_noise,
            rectangle,
            int(rectangle.centered_indices.size),
            int(n_recon_windowed),
        )
    shell_indices_noise, noise_variance_for_noise = mask_recon_tail(
        shell_indices_noise,
        noise_variance_for_noise,
        jnp.int32(int(plan.logical_reconstruction_pixels)),
        shell_sentinel=int(image_shape[0] // 2 + 1),
    )
    return (
        shell_indices_noise,
        noise_variance_for_noise,
        make_stable_relion_wavg_rectangle(image_shape, plan),
        int(plan.logical_rectangle_pixels),
        int(plan.logical_reconstruction_pixels),
    )


def physical_reconstruction_volume(
    plan,
    *,
    volume_shape,
    volume_current_size,
    reconstruction_volume_current_size,
    recon_volume_shape,
    mstep_max_r,
    reconstruction_image_radius,
    reconstruction_padding_factor,
    anisotropic_magnification,
    quantum,
) -> tuple:
    """``(volume_current_size, recon_volume_shape, mstep_max_r)`` of the BPref cube the pass accumulates
    into: RELION's, or the plan's physical current-size cube.

    The pass accumulates into the physical cube and crops it to RELION's logical cube before finalizing
    (:func:`finalize_stable_bpref`). The logical pixels land on the same Fourier voxels of either cube,
    and the M-step gives the tail no weight. ``reconstruction_volume_current_size`` (None on one grid)
    is the reference model's size for images on another grid, whose cube takes its own class
    (:func:`stable_reference_volume_class`).
    """

    if plan is None:
        return int(volume_current_size), tuple(int(v) for v in recon_volume_shape), mstep_max_r
    program_volume_current_size = (
        int(plan.physical_reconstruction_current_size)
        if reconstruction_volume_current_size is None
        else stable_reference_volume_class(int(volume_shape[0]), int(volume_current_size), quantum=quantum)
    )
    return (
        program_volume_current_size,
        tuple(
            int(v)
            for v in relion_backprojector_volume_shape(
                volume_shape, reconstruction_padding_factor, current_size=program_volume_current_size
            )
        ),
        mstep_adjoint_max_r(
            program_volume_current_size,
            reference_image_radius_at(reconstruction_image_radius, volume_current_size, program_volume_current_size),
            reconstruction_padding_factor,
            anisotropic_magnification=anisotropic_magnification,
        ),
    )


def pass_capacity_projection(
    plan,
    *,
    use_relion_projector,
    class_projector_halves,
    relion_projector_r_max,
    projection_padding_factor,
    projection_kwargs,
) -> tuple:
    """``(halves, r_max, kwargs, logical_output_size)``: how a pass projects, at the physical class when the
    plan's capacity projection applies, else as given.

    ``logical_output_size`` is None for a fixed-radius pass (the slabs as given); for a capacity pass the
    slabs are center-padded to the physical class, the static radius is the sentinel 0 and ``kwargs`` carry
    RELION's logical model and image radii as runtime values. The caller then projects into the physical
    crop (``projector_output_size`` the plan's ``physical_current_size``).

    A center-padded capacity slab into the physical crop, with RELION's logical model and image radii as
    runtime values (``project_relion_half_capacity``). The projection programs then keep one shape per
    class instead of compiling at every logical current size.
    """

    if plan is None or not capacity_projection_applies(
        use_relion_projector=use_relion_projector,
        class_projector_halves=class_projector_halves,
        relion_projector_r_max=relion_projector_r_max,
        projection_padding_factor=projection_padding_factor,
        projection_kwargs=projection_kwargs,
        physical_current_size=int(plan.physical_current_size),
    ):
        return list(class_projector_halves), relion_projector_r_max, {}, None
    physical_size = int(plan.physical_current_size)
    return (
        [
            center_pad_relion_projector_half(
                half,
                logical_r_max=int(relion_projector_r_max),
                physical_size=physical_size,
                padding_factor=int(projection_padding_factor),
            )
            for half in class_projector_halves
        ],
        # The capacity route takes the model radius at runtime; its static radius is the sentinel 0.
        0,
        {
            "relion_projector_runtime_r_max": jnp.asarray(int(relion_projector_r_max), dtype=jnp.int32),
            "relion_projector_image_size": jnp.asarray(int(projection_kwargs["projector_output_size"]), dtype=jnp.int32),
        },
        int(projection_kwargs["projector_output_size"]),
    )


def runtime_mstep_radius(mstep_max_r) -> float:
    """The adjoint's runtime radius in image Fourier pixels: ``max_r``, or a sphere clip's image radius."""

    return float(mstep_max_r.image_radius) if isinstance(mstep_max_r, ReferenceSphereClip) else float(mstep_max_r)


def window_logical_sizes(
    *, current_size, recon_pixels, rect_pixels, place, mstep_max_r=None, wsum_current_size=None
) -> resident_chunk_records.WindowLogicalSizes:
    """The chunk programs' runtime logical sizes (``ChunkStageTables.window_logical``)."""

    return resident_chunk_records.WindowLogicalSizes(
        current_size=place.scalar(int(current_size), jnp.int32),
        recon_pixels=place.scalar(int(recon_pixels), jnp.int32),
        rect_pixels=place.scalar(int(rect_pixels), jnp.int32),
        mstep_max_r=None if mstep_max_r is None else place.scalar(float(mstep_max_r), jnp.float32),
        wsum_current_size=None if wsum_current_size is None else place.scalar(int(wsum_current_size), jnp.int32),
    )


@jax.jit
def times_logical_mask(values, n_logical):
    """``values`` times the mask of its first ``n_logical`` pixels (last axis); one program per physical class.

    The count is traced, so the logical sizes of a physical class share the program.
    """

    mask = jnp.arange(values.shape[-1], dtype=jnp.int32) < n_logical
    return values * mask.astype(values.dtype)


@partial(jax.jit, static_argnames=("shell_sentinel",))
def mask_recon_tail(shell_indices_noise, noise_variance_for_noise, n_logical, *, shell_sentinel: int):
    """The recon tail past the ``n_logical`` pixels: the shell sentinel and zero noise variance.

    ``noise_variance_for_noise`` is ``[P]`` or, with optics groups, ``[G, P]``; the tail
    mask broadcasts over groups. One program per physical class (the count is traced).
    """

    tail = ~(jnp.arange(shell_indices_noise.shape[-1], dtype=jnp.int32) < n_logical)
    return (
        jnp.where(tail, jnp.int32(shell_sentinel), shell_indices_noise),
        jnp.where(tail, jnp.zeros((), noise_variance_for_noise.dtype), noise_variance_for_noise),
    )


def finalize_stable_bpref(
    Ft_y,
    Ft_ctf,
    *,
    program_recon_volume_shape,
    recon_volume_shape,
    symmetry_label,
    engine_tuning,
    logger: logging.Logger,
    label: str,
    keep_physical_bpref: bool = False,
):
    """Finalize one BPref pair accumulated in ``program_recon_volume_shape`` and return its public layout.

    RELION symmetriseReconstructions (ml_optimiser.cpp:5541-5575): x=0 Hermitian enforcement, then
    applyPointGroupSymmetry on BPref; C1 is the x=0 enforcement alone. A stable-window C1 pass finalizes at
    its physical cube and crops the public volume, so the finalize programs keep the physical class's shapes
    across logical current sizes (they compiled at every logical size before); otherwise the physical cube
    is cropped to RELION's first. ``keep_physical_bpref``: the caller slices the logical cube out of the
    physical one itself (VDAM's accumulator adapter, on the host), so no crop program compiles per logical
    size. ``logger`` and ``label`` are the pass's own (its finalize log lines).
    """

    logical_recon_volume_shape = tuple(int(v) for v in recon_volume_shape)
    program_recon_volume_shape = tuple(int(v) for v in program_recon_volume_shape)
    cropped = program_recon_volume_shape != logical_recon_volume_shape
    finalize_physical = cropped and physical_bpref_finalize_applies(
        program_recon_volume_shape, logical_recon_volume_shape, symmetry_label,
        large_grid=engine_tuning.x_half_large_grid,
    )
    finalize_shape = program_recon_volume_shape if finalize_physical else logical_recon_volume_shape
    if cropped and not finalize_physical:
        Ft_y = crop_relion_x_half_accumulator(Ft_y, program_recon_volume_shape, recon_volume_shape)
        Ft_ctf = crop_relion_x_half_accumulator(Ft_ctf, program_recon_volume_shape, recon_volume_shape)
    Ft_y, Ft_ctf = finalize_half_volume_bpref(
        Ft_y,
        Ft_ctf,
        finalize_shape,
        logger=logger,
        label=label,
        symmetry_label=symmetry_label,
        relion_x_half=True,
        symmetry_chunk_voxels=engine_tuning.bpref_symmetry_chunk_voxels,
        large_grid=engine_tuning.x_half_large_grid,
    )
    Ft_y, Ft_ctf = relion_x_half_accumulators_to_public_layout(
        Ft_y,
        Ft_ctf,
        finalize_shape,
        large_grid=engine_tuning.x_half_large_grid,
    )
    if finalize_physical and not keep_physical_bpref:
        Ft_y = crop_public_full_volume(Ft_y, program_recon_volume_shape, logical_recon_volume_shape)
        Ft_ctf = crop_public_full_volume(Ft_ctf, program_recon_volume_shape, logical_recon_volume_shape)
    return Ft_y, Ft_ctf
