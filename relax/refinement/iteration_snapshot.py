"""The state a numbered refinement iteration hands to the next one.

RELION writes this state after every iteration as ``run_itNNN_*`` files and
``--continue`` reads it back (``MlOptimiser::write``/``read``,
ml_optimiser.cpp:1359-1557 and 1089-1357 at RELION 5.0.1 f2c1a38).
``refine_single_volume`` captures an :class:`IterationSnapshot` at the end of
each numbered iteration and, given one through
``options.checkpoint.resume``, starts from it. ``relax.refinement.run_files``
maps a snapshot onto RELION's files and back.

Arrays are host NumPy arrays in the loop's own layouts and frames:

* ``means`` are centered Fourier volumes, one per half (K=1, ``(V,)``) or one
  class stack per half (K>1, ``(K, V)``; Class3D keeps one stack, so half 2
  is half 1).
* Spectra are in relax's frame, which is RELION's times ``ori_size**4``
  (``relion_replay._build_replay_iteration_overrides`` reads RELION's model
  STAR with the same factor).
* Per-particle arrays are per half, in the half's local particle order.

``incr_size`` and ``has_high_fsc_at_limit`` are RELION's values after the
iteration's FSC update, as ``run_itNNN_optimiser.star`` holds them; the loop
applies that idempotent update again at the top of the next iteration.

See ``docs/math/relion_refinement_algorithm.md`` section 8.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from relax.helpers.convergence import RefinementState

# RefinementState fields that hold per-image arrays rather than scalars; the
# snapshot carries per-image state separately.
_STATE_ARRAY_FIELDS = frozenset({"best_rotations", "best_translations"})
REFINEMENT_STATE_SCALAR_FIELDS = tuple(
    f.name for f in dataclasses.fields(RefinementState) if f.name not in _STATE_ARRAY_FIELDS
)


@dataclass
class IterationSnapshot:
    """Loop state at the end of numbered RELION iteration ``relion_iteration``."""

    relion_iteration: int
    n_classes: int
    ori_size: int
    pixel_size: float
    tau2_fudge: float

    means: list
    tau2_shells: np.ndarray
    data_vs_prior: np.ndarray
    noise_shells: list
    sigma_offset_angstrom: tuple

    current_size: int
    incr_size: int
    has_high_fsc_at_limit: bool
    random_perturbation: float
    state_fields: dict

    rotation_eulers: list
    translations: list
    image_corrections: list
    scale_corrections: list
    group_ids: list

    fsc: np.ndarray | None = None
    fsc_for_growth: np.ndarray | None = None
    class_weights: np.ndarray | None = None
    direction_prior: list | None = None
    class_assignments: list | None = None
    max_posterior: list | None = None
    significant_counts: list | None = None
    avg_norm_correction: tuple = (1.0, 1.0)
    unfiltered_means: list | None = None
    extra: dict = field(default_factory=dict)

    @property
    def k_class(self) -> bool:
        return int(self.n_classes) > 1

    def refinement_state(self, base: RefinementState) -> RefinementState:
        """``base`` with every scalar field replaced by the snapshot's value."""

        unknown = sorted(set(self.state_fields) - set(REFINEMENT_STATE_SCALAR_FIELDS))
        missing = sorted(set(REFINEMENT_STATE_SCALAR_FIELDS) - set(self.state_fields))
        if unknown or missing:
            raise ValueError(
                f"snapshot RefinementState fields do not match: unknown={unknown} missing={missing}"
            )
        return dataclasses.replace(base, **self.state_fields)


def refinement_state_fields(state: RefinementState) -> dict:
    """The scalar fields of ``state`` as plain Python values."""

    values = {}
    for name in REFINEMENT_STATE_SCALAR_FIELDS:
        value = getattr(state, name)
        if isinstance(value, (bool, np.bool_)):
            values[name] = bool(value)
        elif isinstance(value, (int, np.integer)):
            values[name] = int(value)
        else:
            values[name] = float(value)
    return values


def radial_shell_volume(shells, volume_shape, *, dtype):
    """Expand radial shells onto a centered volume the way the M-step does.

    ``compute_relion_tau2_from_weights`` and
    ``compute_relion_tau2_from_iref_power_spectrum`` both index their shells by
    the truncated integer radius, clamped at the last shell.
    """

    from recovar.core import fourier_transform_utils

    shells = jnp.asarray(shells, dtype=dtype)
    radius = (
        fourier_transform_utils.get_grid_of_radial_distances(volume_shape, scaled=False, frequency_shift=0)
        .astype(int)
        .reshape(-1)
    )
    radius = jnp.minimum(radius, shells.shape[-1] - 1)
    return shells[..., radius]


def tau2_mean_variance(snapshot: IterationSnapshot, volume_shape, *, dtype):
    """The ``mean_variance`` volume the loop scores the next iteration with.

    K=1 averages the two halves' tau2 volumes (``iteration_loop``: ``0.5 *
    (msv_half1 + msv_half2)``); Class3D stacks one tau2 volume per class.
    """

    shells = np.asarray(snapshot.tau2_shells)
    if snapshot.k_class:
        return radial_shell_volume(shells, volume_shape, dtype=dtype)
    per_half = [radial_shell_volume(shells[h], volume_shape, dtype=dtype) for h in range(2)]
    return 0.5 * (per_half[0] + per_half[1])


def noise_pixel_rows(noise_shells, image_shape):
    """Expand one half's noise shells (``(S,)`` or ``(G, S)``) to the loop's pixel rows."""

    from recovar.reconstruction import noise

    shells = np.asarray(noise_shells, dtype=np.float64)
    if shells.ndim == 1:
        return jnp.asarray(noise.make_radial_noise(shells, image_shape)).reshape(-1)
    return jnp.stack(
        [jnp.asarray(noise.make_radial_noise(row, image_shape)).reshape(-1) for row in shells]
    )


def host_array(value, dtype=None):
    """A host copy of a device or host array (``None`` stays ``None``)."""

    if value is None:
        return None
    return np.array(value, dtype=dtype, copy=True)


def host_half_pair(values, dtype=None):
    if values is None:
        return None
    return [host_array(v, dtype) for v in values]


def validate_resume_snapshot(snapshot: IterationSnapshot, *, init_relion_iteration, n_classes, grid_size, options):
    """Refuse a continuation the loop cannot start exactly from ``snapshot``."""

    problems = []
    if int(snapshot.relion_iteration) != int(init_relion_iteration):
        problems.append(
            f"snapshot iteration {snapshot.relion_iteration} != init_relion_iteration {init_relion_iteration}"
        )
    if int(snapshot.n_classes) != int(n_classes):
        problems.append(f"snapshot has {snapshot.n_classes} classes, the run {n_classes}")
    if int(snapshot.ori_size) != int(grid_size):
        problems.append(f"snapshot box {snapshot.ori_size} != image box {grid_size}")
    if options.parity.perturb_replay_relion_dir is not None or options.replay.replay_iteration_overrides is not None:
        problems.append("a continuation cannot replay a RELION trajectory")
    if options.debug.sealed_sampling_state is not None or options.replay.init_refinement_state_fields is not None:
        problems.append("a continuation cannot start from a frozen boundary")
    if options.parity.use_per_half_mean_variance:
        problems.append("per-half scoring tau2 is a frozen-boundary diagnostic")
    if options.debug.state_swap_probe is not None:
        problems.append("state-swap probes need a replayed trajectory")
    if options.adaptive.relion_current_sizes is not None or options.adaptive.relion_healpix_orders is not None:
        problems.append("sampling oracles do not apply to a continuation")
    if options.replay.init_reference_real is not None:
        problems.append("a continuation projects its Fourier references, not initial real maps")
    if problems:
        raise ValueError("cannot continue from the run files: " + "; ".join(problems))


def capture_iteration_snapshot(
    *,
    relion_iteration,
    n_classes,
    grid_size,
    voxel_size,
    tau2_fudge,
    means,
    unfiltered_means,
    tau2_shells,
    data_vs_prior,
    fsc,
    fsc_for_growth,
    noise_shells,
    sigma_offset_angstrom_per_half,
    current_size,
    incr_size,
    has_high_fsc_at_limit,
    random_perturbation,
    state,
    half_inputs,
    class_weights,
    direction_prior,
    direction_prior_order,
    class_assignments,
    max_posterior,
    significant_counts,
    avg_norm_correction,
) -> IterationSnapshot:
    """Copy the loop's end-of-iteration state to the host (see the module docstring)."""

    k_class = int(n_classes) > 1
    if k_class:
        tau2 = np.asarray(tau2_shells, dtype=np.float64)
    else:
        tau2 = np.stack([np.asarray(shells, dtype=np.float64) for shells in tau2_shells])
    eulers = host_half_pair(half_inputs.previous_best_rotation_eulers)
    translations = host_half_pair(half_inputs.previous_best_translations)
    image_corrections = host_half_pair(half_inputs.image_corrections)

    def _dtype_name(values):
        present = [v for v in values if v is not None]
        return str(present[0].dtype) if present else "float32"

    avg_norm = tuple(1.0 if value is None else float(value) for value in avg_norm_correction)
    return IterationSnapshot(
        relion_iteration=int(relion_iteration),
        n_classes=int(n_classes),
        ori_size=int(grid_size),
        pixel_size=float(voxel_size),
        tau2_fudge=float(tau2_fudge),
        means=[host_array(means[0])] * 2 if k_class else host_half_pair(means),
        tau2_shells=tau2,
        data_vs_prior=np.asarray(data_vs_prior, dtype=np.float64),
        noise_shells=[np.asarray(shells, dtype=np.float64) for shells in noise_shells],
        sigma_offset_angstrom=tuple(float(v) for v in sigma_offset_angstrom_per_half),
        current_size=int(current_size),
        incr_size=int(incr_size),
        has_high_fsc_at_limit=bool(has_high_fsc_at_limit),
        random_perturbation=float(random_perturbation),
        state_fields=refinement_state_fields(state),
        rotation_eulers=eulers,
        translations=translations,
        image_corrections=image_corrections,
        scale_corrections=host_half_pair(half_inputs.scale_corrections),
        group_ids=[
            np.zeros(0 if e is None else len(e), dtype=np.int64) if g is None else np.asarray(g, dtype=np.int64)
            for g, e in zip(half_inputs.group_ids, eulers)
        ],
        fsc=None if fsc is None or k_class else np.asarray(fsc, dtype=np.float64),
        fsc_for_growth=None if fsc_for_growth is None or k_class else np.asarray(fsc_for_growth, dtype=np.float64),
        class_weights=None if not k_class else np.asarray(class_weights, dtype=np.float64),
        direction_prior=host_half_pair(direction_prior)
        if direction_prior is not None and any(p is not None for p in direction_prior)
        else None,
        class_assignments=host_half_pair(class_assignments) if k_class else None,
        max_posterior=host_half_pair(max_posterior),
        significant_counts=host_half_pair(significant_counts),
        avg_norm_correction=avg_norm,
        unfiltered_means=None
        if unfiltered_means is None or all(m is None for m in unfiltered_means)
        else host_half_pair(unfiltered_means),
        extra={
            "euler_dtype": _dtype_name(eulers),
            "translation_dtype": _dtype_name(translations),
            "correction_dtype": _dtype_name(image_corrections),
            # The HEALPix order a direction prior indexes; its length alone is ambiguous
            # under symmetry. -1: no prior.
            **{
                f"direction_prior_order_half{h + 1}": -1 if order is None else int(order)
                for h, order in enumerate(direction_prior_order)
            },
        },
    )
