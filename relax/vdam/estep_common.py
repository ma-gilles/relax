"""E-step records and accumulator helpers shared by the single-particle and subtomogram InitialModel E-steps.

The E-step configuration and result records, the accumulator packing, the per-image row selection and the
metadata assembly live here so ``adaptive_estep`` and ``tomo_estep`` import them without importing each
other. The layout bridge from the shared RELION-x-half M-step's public Fourier cubes to RELION
BackProjector centered half-complex slabs comes first.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from relax.vdam.state import InitialModelState, VdamAccumulator


def _bp_slab(arr: np.ndarray, r_max: int, c: int) -> np.ndarray:
    """Slice a centered full volume into a RELION BPref slab (full half-complex or cropped)."""
    if r_max >= c:
        return np.concatenate([arr[:, :, c:], arr[:, :, :1]], axis=2)
    half_ps = r_max + 1
    return arr[c - half_ps : c + half_ps + 1, c - half_ps : c + half_ps + 1, c : c + half_ps + 1]


def _as_centered_bpref_source(
    values: np.ndarray,
    *,
    box_size: int,
    r_max: int,
    padding_factor: int,
) -> tuple[np.ndarray, int, int]:
    """Return ``(centered cube, center, effective radius)`` for BPref slicing.

    ``values`` is an original-box full cube or the shared RELION x-half M-step's current-size odd
    BackProjector cube in the public full layout. Both encode the same centered support and feed the one
    BPref slab conversion below.
    """
    arr = np.asarray(values)
    full_size = int(box_size) * int(padding_factor)
    if arr.size == full_size**3:
        return arr.reshape(full_size, full_size, full_size), full_size // 2, int(r_max)

    effective_radius = int(float(padding_factor) * float(r_max) + 0.5)
    compact_size = 2 * (effective_radius + 1) + 1
    if arr.size == compact_size**3:
        return (
            arr.reshape(compact_size, compact_size, compact_size),
            compact_size // 2,
            effective_radius,
        )
    # A stable-window pass may leave its BPref at the odd cube of its physical window class
    # (``keep_physical_bpref``): the logical cube is its centre, and the slab is cut from there.
    edge = round(arr.size ** (1.0 / 3.0))
    if edge**3 == arr.size and edge % 2 == 1 and edge > compact_size:
        return arr.reshape(edge, edge, edge), edge // 2, effective_radius
    raise ValueError(
        "expected either an original-box centered Fourier cube of size "
        f"{full_size**3} or a current-size BackProjector cube of size {compact_size**3}; got shape {arr.shape}"
    )


def _centered_bpref_sources(Ft_y, Ft_ctf, *, box_size: int, r_max: int, padding_factor: int):
    """Return ``(data cube, weight cube, center, radius)`` for the RELION-x-half BPref converter.

    Both accumulators must encode the same centered support.
    """
    if padding_factor not in (1, 2):
        raise NotImplementedError(f"padding_factor must be 1 or 2, got {padding_factor}")
    if r_max < 0:
        raise ValueError(f"r_max must be non-negative, got {r_max}")

    data_cube, data_center, data_radius = _as_centered_bpref_source(
        Ft_y,
        box_size=box_size,
        r_max=r_max,
        padding_factor=padding_factor,
    )
    weight_cube, weight_center, weight_radius = _as_centered_bpref_source(
        Ft_ctf,
        box_size=box_size,
        r_max=r_max,
        padding_factor=padding_factor,
    )
    if (data_center, data_radius) != (weight_center, weight_radius):
        raise ValueError("data and weight accumulators use different centered layouts")
    return data_cube, weight_cube, data_center, data_radius


def _bpref_slab_outputs(bp_data: np.ndarray, bp_weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cast a BPref slab pair to RELION double precision; clamp denormal weights (``updateSSNRarrays`` aborts on (0, 1e-20])."""
    bp_weight_f64 = np.asarray(bp_weight.real, dtype=np.float64).copy()
    bp_weight_f64[np.abs(bp_weight_f64) < 1e-15] = 0.0
    return np.asarray(bp_data, dtype=np.complex128).copy(), bp_weight_f64


def relion_bpref_frame_scales(box_size: int) -> tuple[float, float]:
    """``(-N², N⁴)`` — RECOVAR unnormalised-FFT → RELION BPref frame."""
    n = float(box_size)
    return -(n**2), n**4


def relion_x_public_output_to_bpref(
    Ft_y: np.ndarray,
    Ft_ctf: np.ndarray,
    box_size: int,
    r_max: int,
    padding_factor: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert the shared RELION-x-half public-layout conversion.

    The shared M-step expands native RELION ``(z, y, xhalf)`` storage to a
    full cube and transposes it to RECOVAR's public ``(x, y, z)`` order.
    InitialModel consumes a native BPref again, so undo that transpose before
    selecting the positive-x slab.
    """

    data_cube, weight_cube, center, radius = _centered_bpref_sources(
        Ft_y,
        Ft_ctf,
        box_size=box_size,
        r_max=r_max,
        padding_factor=padding_factor,
    )
    return _bpref_slab_outputs(
        _bp_slab(data_cube.transpose(2, 1, 0), radius, center),
        _bp_slab(weight_cube.transpose(2, 1, 0), radius, center),
    )

# The engine result's per-image fields an E-step copies into its meta, with their meta dtypes. Neither route
# copies best_pose_rotation_ids: they index the route's RECOVAR-order fine grid, while VDAM reads rotation ids as
# RELION-order rows (the source Euler rows and the matrices carry the pose).
SPA_META_PARTICLE_FIELDS: tuple[tuple[str, type], ...] = (
    ("pose_assignments", np.int32),
    ("class_assignments", np.int32),
    ("best_pose_rotations", np.float32),
    ("best_pose_translations", np.float32),
    ("significant_counts", np.int32),
)
# Subtomograms: the 2D winning translation and pose id become the particle's 3D offset (tomo_offsets_px), and
# the significant counts are the tilt pass's own.
TOMO_META_PARTICLE_FIELDS: tuple[tuple[str, type], ...] = (
    ("class_assignments", np.int32),
    ("best_pose_rotations", np.float32),
)
# The engine's interpolation for every InitialModel E-step call.
ENGINE_DISC_TYPE = "linear_interp"

@dataclass(frozen=True, kw_only=True)
class MultiShapeTranslations:
    """What each image shape (optics groups on several shapes) rebuilds its pre-shifts and coarse ``pdf_offset``
    from, in its own pixels: every particle's offset ``offsets_px`` ``[N, 2]`` (float64, reference pixels), the
    coarse prior grid ``coarse_prior_translations`` (float32, reference pixels) and the offset sigma in Angstrom."""

    offsets_px: np.ndarray
    coarse_prior_translations: np.ndarray
    sigma_angstrom: float


@dataclass(frozen=True, kw_only=True)
class EstepSampling:
    """One E-step's pass controls, from the iteration's sampling plan, state and options (built once by
    :func:`relax.vdam.estep_setup.initial_model_estep_config`). The adaptive route builds its grids and
    coarse sizes from them; the engine receives what the route derives, never these fields themselves.

    ``coarse_translations`` ``[T, 2]`` float32 are the pass-1 grid (perturbed), ``coarse_base_translations``
    float64 RELION's unperturbed host grid, ``coarse_translation_log_prior`` ``[N, T]`` the coarse ``pdf_offset``
    of every image; ``translation_step`` is in pixels. ``pass1_healpix_order`` is the order before this
    iteration's sampling update; ``adaptive_fraction`` None leaves the route's 0.999.
    ``multi_shape_translations`` (optics groups on several image shapes): each particle's offset in
    reference pixels, the coarse prior grid and the offset sigma, from which each shape rebuilds its own.
    """

    healpix_order: int
    oversampling_order: int
    translation_step: float
    random_perturbation: float
    coarse_translations: np.ndarray
    coarse_base_translations: np.ndarray
    coarse_translation_log_prior: np.ndarray
    particle_diameter_ang: float
    pass1_healpix_order: int
    max_significants: int
    adaptive_fraction: float | None
    multi_shape_translations: MultiShapeTranslations | None


@dataclass(frozen=True, kw_only=True)
class InitialModelEstepConfig:
    """Configuration for one InitialModel dense K-class E-step."""

    noise_variance: Any
    translations: Any
    sampling: EstepSampling
    image_batch_size: int = 500
    rotation_block_size: int = 5000
    coarse_engine: str = "auto"
    padding_factor: int = 1
    relion_projector_half_by_class: Any | None = None
    relion_projector_r_max: int | None = None
    engine_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class InitialModelEstepResult:
    """Output consumed by ``iteration_loop.run_vdam_iterations``."""

    accumulators: list[VdamAccumulator]
    meta: dict[str, Any]


@dataclass(frozen=True)
class EstepNoiseSums:
    """One E-step's weighted noise sums (engine units): ``[n]`` with a 0-d weight for one optics group,
    ``[G, n]`` with ``[G]`` weights for several. ``a2``/``xa``: the boundary dump's extra sums, or None."""

    wsum_sigma2_noise: np.ndarray
    wsum_img_power: np.ndarray
    sumw: np.ndarray
    wsum_noise_a2: np.ndarray | None
    wsum_noise_xa: np.ndarray | None


@dataclass(frozen=True)
class EstepSums:
    """What the model update reads from one E-step; the rest of the E-step's ``meta`` is its report.

    Built once by :func:`estep_sums` where the E-step returns. ``class_mass`` ([K], float64) is the retained
    (M-step) class posterior mass, ``direction_mass`` [K, D] the class/direction posterior, ``pmax`` [n]
    (float32) each image's maximum posterior; ``offset_wsum``/``offset_sumw`` the offset sums over the
    significant-pruned weights and ``offset_dims`` 2 for particles, 3 for subtomograms; ``average_ctf2``
    the subset's average CTF^2 of CTF-premultiplied images. A None group: the E-step has no such sums
    (an empty subset has none), and the update that reads it keeps the model.
    """

    class_mass: np.ndarray | None
    direction_mass: np.ndarray | None
    pmax: np.ndarray | None
    offset_wsum: float | None
    offset_sumw: float | None
    offset_dims: int
    noise: EstepNoiseSums | None
    average_ctf2: Any | None


def estep_sums(meta: dict[str, Any]) -> EstepSums:
    """Read the model update's operands from an E-step's ``meta``, once; refuse a partial set of sums."""

    def array(key, dtype=np.float64):
        return None if meta.get(key) is None else np.asarray(meta[key], dtype=dtype)

    noise_keys = ("wsum_sigma2_noise", "wsum_img_power", "noise_sumw")
    present = [key for key in noise_keys if meta.get(key) is not None]
    if present and len(present) != len(noise_keys):
        raise ValueError(f"the E-step meta has {present} but not all of {list(noise_keys)}")
    if meta.get("wsum_sigma2_offset") is not None and meta.get("sigma2_offset_sumw") is None:
        raise ValueError("the E-step meta has wsum_sigma2_offset but no sigma2_offset_sumw")
    pmax = array("max_posterior_per_image", np.float32)
    if pmax is not None and pmax.size and meta.get("class_posterior_sums") is None:
        raise ValueError("the E-step meta has per-image Pmax but no class posterior mass to normalise it")
    return EstepSums(
        class_mass=array("class_posterior_sums"),
        direction_mass=array("class_direction_posterior_sums"),
        pmax=pmax,
        offset_wsum=None if meta.get("wsum_sigma2_offset") is None else float(meta["wsum_sigma2_offset"]),
        offset_sumw=None if meta.get("sigma2_offset_sumw") is None else float(meta["sigma2_offset_sumw"]),
        offset_dims=int(meta.get("offset_dims", 2)),  # the subtomogram E-step writes 3
        noise=None if not present else EstepNoiseSums(
            *(array(key) for key in noise_keys), wsum_noise_a2=array("wsum_noise_a2"), wsum_noise_xa=array("wsum_noise_xa")
        ),
        average_ctf2=meta.get("premultiplied_average_ctf2"),
    )


def select_image_rows(value, image_indices: np.ndarray, *, n_images: int, name: str):
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim == 0:
        return value
    image_indices = np.asarray(image_indices, dtype=np.int64)
    if array.shape[0] == int(n_images):
        return array[image_indices]
    if array.shape[0] == int(image_indices.size):
        return value
    raise ValueError(
        f"{name} must be shared, selected-image, or full-dataset shaped; "
        f"got first axis {array.shape[0]} for {image_indices.size} selected images and {n_images} total images"
    )


def group_local_kwargs(
    engine_kwargs: dict[str, Any],
    image_indices: np.ndarray,
    *,
    n_images: int,
) -> dict[str, Any]:
    """Return dense/local kwargs in the compact row space of a dataset subset."""

    out = dict(engine_kwargs)
    prior = out.get("translation_log_prior")
    # A shared translation prior is 1D; only 2D priors have an image axis.
    prior_np = None if prior is None else np.asarray(prior)
    if prior_np is not None and prior_np.ndim == 2:
        out["translation_log_prior"] = select_image_rows(
            prior, image_indices, n_images=n_images, name="translation_log_prior"
        )
    for name in (
        "image_pre_shifts",
        "image_corrections",
        "scale_corrections",
        "translation_prior_centers",
        "optics_group_ids",
    ):
        out[name] = select_image_rows(out.get(name), image_indices, n_images=n_images, name=name)
    return out


def add_accumulator_weight_meta(meta: dict[str, Any], accumulators: list[VdamAccumulator], K: int) -> None:
    """Record each class's RELION BPref weight total, and per half-set, in ``meta`` (a report; nothing reads it)."""

    sums = np.zeros(int(K), dtype=np.float64)
    halfset_sums: dict[int, np.ndarray] = {}
    for accum in accumulators:
        value = float(np.sum(np.asarray(accum.weight, dtype=np.float64)))
        sums[accum.class_idx] += value
        halfset_sums.setdefault(accum.halfset_idx, np.zeros(int(K), dtype=np.float64))[accum.class_idx] += value
    meta["class_bpref_weight_sums"] = sums
    for halfset_idx, values in sorted(halfset_sums.items()):
        meta[f"halfset_{halfset_idx}_class_bpref_weight_sums"] = values


def empty_accumulator(state: InitialModelState, class_idx: int, halfset_idx: int) -> VdamAccumulator:
    r_max = state.effective_current_size // 2
    if r_max >= state.box_size // 2:
        shape = (state.box_size, state.box_size, state.box_size // 2 + 1)
    else:
        half_ps = r_max + 1
        shape = (2 * half_ps + 1, 2 * half_ps + 1, half_ps + 1)
    return VdamAccumulator(
        data=np.zeros(shape, dtype=np.complex128),
        weight=np.zeros(shape, dtype=np.float64),
        class_idx=class_idx,
        halfset_idx=halfset_idx,
    )


def arrays_to_accumulators(
    Ft_y_by_class,
    Ft_ctf_by_class,
    state: InitialModelState,
    *,
    halfset_idx: int | None,
    reconstruction_group_count: int | None = None,
    padding_factor: int,
) -> list[VdamAccumulator]:
    # Refuse a class axis other than K on either array before building a partially aligned list.
    try:
        data_class_count = len(Ft_y_by_class)
        weight_class_count = len(Ft_ctf_by_class)
    except TypeError as error:
        raise ValueError("K-class accumulators must expose a leading class axis") from error
    if data_class_count != int(state.K) or weight_class_count != int(state.K):
        raise ValueError(
            "K-class accumulator class axes must each contain exactly "
            f"{int(state.K)} classes, got data={data_class_count} and weight={weight_class_count}",
        )
    r_max = state.effective_current_size // 2
    data_scale, weight_scale = relion_bpref_frame_scales(state.box_size)
    dump_dir = os.environ.get("RELAX_INITIAL_MODEL_ACCUM_DUMP_DIR")

    grouped = halfset_idx is None
    if grouped:
        if reconstruction_group_count is None or int(reconstruction_group_count) <= 0:
            raise ValueError(
                "reconstruction_group_count is required for grouped accumulators"
            )
        output_halfsets = range(int(reconstruction_group_count))
    else:
        if reconstruction_group_count not in (None, 1):
            raise ValueError(
                "reconstruction_group_count is only valid for grouped accumulators"
            )
        output_halfsets = (int(halfset_idx),)

    accumulators: list[VdamAccumulator] = []
    for k in range(state.K):
        class_data = np.asarray(Ft_y_by_class[k])
        class_weight = np.asarray(Ft_ctf_by_class[k])
        if grouped and (
            class_data.shape[0] != int(reconstruction_group_count)
            or class_weight.shape[0] != int(reconstruction_group_count)
        ):
            raise ValueError(
                "grouped accumulator arrays do not match reconstruction_group_count"
            )
        for output_halfset in output_halfsets:
            public_data = class_data[output_halfset] if grouped else class_data
            public_weight = class_weight[output_halfset] if grouped else class_weight
            bp_data, bp_weight = relion_x_public_output_to_bpref(
                public_data,
                public_weight,
                state.box_size,
                r_max,
                padding_factor=padding_factor,
            )
            if dump_dir:
                path = Path(dump_dir)
                path.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    path / f"accum_h{int(output_halfset)}_k{int(k)}.npz",
                    Ft_y=np.asarray(public_data),
                    Ft_ctf=np.asarray(public_weight),
                    bp_data_unscaled=np.asarray(bp_data),
                    bp_weight_unscaled=np.asarray(bp_weight),
                    bp_data_scaled=np.asarray(bp_data * data_scale),
                    bp_weight_scaled=np.asarray(bp_weight * weight_scale),
                    data_scale=np.float64(data_scale),
                    weight_scale=np.float64(weight_scale),
                    padding_factor=np.int32(padding_factor),
                    ori_size=np.int32(state.box_size),
                    current_size=np.int32(state.current_size),
                )
            accumulators.append(
                VdamAccumulator(
                    data=bp_data * data_scale,
                    weight=bp_weight * weight_scale,
                    class_idx=k,
                    halfset_idx=int(output_halfset),
                )
            )
    return accumulators


def estep_meta(halfset_results: dict[int, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {"halfset_ids": tuple(sorted(halfset_results))}
    class_totals: dict[str, np.ndarray] = {}

    def add_class_total(name, values):
        previous = class_totals.get(name)
        class_totals[name] = values if previous is None else previous + values

    noise_totals: dict[str, Any] | None = None
    for h, result in halfset_results.items():
        if getattr(result, "class_posterior_sums", None) is not None:
            full_sums = np.asarray(result.class_posterior_sums, dtype=np.float64)
            # The engine always resolves the M-step (retained) class mass (k_class_results); a result
            # without it cannot normalise the class and offset updates.
            if getattr(result, "class_mstep_posterior_sums", None) is None:
                raise ValueError(f"the E-step result of half {h} has class posterior sums but no M-step class mass")
            sums = np.asarray(result.class_mstep_posterior_sums, dtype=np.float64)
            meta[f"halfset_{h}_class_posterior_sums"] = sums
            meta[f"halfset_{h}_class_posterior_sums_full"] = full_sums
            add_class_total("class_posterior_sums", sums)
            add_class_total("class_posterior_sums_full", full_sums)
        per_class_noise = getattr(result, "noise_stats", None)
        if per_class_noise is not None:
            # A class's support over all its optics groups (sumw is [G] with several groups).
            support = np.asarray([float(np.sum(stats.sumw)) for stats in per_class_noise], dtype=np.float64)
            meta[f"halfset_{h}_class_reconstruction_support_sums"] = support
            add_class_total("class_reconstruction_support_sums", support)
        if getattr(result, "class_assignments", None) is not None:
            meta[f"halfset_{h}_class_assignments"] = np.asarray(result.class_assignments, dtype=np.int32)
        profile_summary = getattr(result, "profile_summary", None)
        if profile_summary is not None:
            meta[f"halfset_{h}_profile_summary"] = dict(profile_summary)
        noise_stats = getattr(result, "aggregate_noise_stats", None)
        if noise_stats is not None:
            half = {
                "wsum_sigma2_offset": float(noise_stats.wsum_sigma2_offset),
                # Several optics groups give [G, n] noise sums and [G] weights (sumw_group).
                "sigma2_offset_sumw": float(np.sum(np.asarray(noise_stats.sumw, dtype=np.float64))),
                "wsum_sigma2_noise": np.asarray(noise_stats.wsum_sigma2_noise, dtype=np.float64),
                "wsum_img_power": np.asarray(noise_stats.wsum_img_power, dtype=np.float64),
                "noise_sumw": (
                    float(noise_stats.sumw)
                    if np.ndim(noise_stats.sumw) == 0
                    else np.asarray(noise_stats.sumw, dtype=np.float64)
                ),
            }
            if getattr(noise_stats, "wsum_noise_a2", None) is not None:
                half["wsum_noise_a2"] = np.asarray(noise_stats.wsum_noise_a2, dtype=np.float64)
            if getattr(noise_stats, "wsum_noise_xa", None) is not None:
                half["wsum_noise_xa"] = np.asarray(noise_stats.wsum_noise_xa, dtype=np.float64)
            for k, v in half.items():
                meta[f"halfset_{h}_{k}"] = v
            if noise_totals is None:
                noise_totals = dict(half)
            else:
                for k, v in half.items():
                    noise_totals[k] = noise_totals.get(k, 0.0) + v
        per_class_stats = getattr(result, "per_class_stats", None)
        if per_class_stats is not None:
            direction_sums = np.stack(
                [np.asarray(cs.rotation_posterior_sums, dtype=np.float64) for cs in per_class_stats],
                axis=0,
            )
            add_class_total("class_direction_posterior_sums", direction_sums)
    for name in (
        "class_posterior_sums", "class_posterior_sums_full",
        "class_reconstruction_support_sums", "class_direction_posterior_sums",
    ):
        if name in class_totals:
            meta[name] = class_totals[name]
    if noise_totals is not None:
        meta.update(noise_totals)
    return meta
