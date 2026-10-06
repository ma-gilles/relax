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
* Norm corrections are in relax's frame too, RELION's times ``ori_size**2``
  (relax's forward FFT is unnormalized).
* ``direction_prior`` rows of a Class3D snapshot are per-class conditionals that
  sum to one; RELION's ``pdf_direction[class]`` sums to the class fraction.
* ``acc_rot_per_class``/``acc_trans_per_class_angstrom`` are RELION's
  ``MlModel::acc_rot``/``acc_trans``: zero until the first expected-accuracy
  estimate (ml_model.cpp:68), then that estimate's per-class values.

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
# Limits the command line sets rather than state the iterations evolve; a continued run keeps its own.
RUN_LIMIT_FIELDS = ("max_healpix_order",)


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
    acc_rot_per_class: np.ndarray | None = None
    acc_trans_per_class_angstrom: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    @property
    def k_class(self) -> bool:
        return int(self.n_classes) > 1

    def refinement_state(self, base: RefinementState) -> RefinementState:
        """``base`` with every scalar field replaced by the snapshot's value.

        Except the configured limits in :data:`RUN_LIMIT_FIELDS`, which come from
        the continuing run's command line: a continuation from a run capped at
        HEALPix 7 kept that cap over ``--max_healpix_order 12`` (bigbox 14557970).
        """

        unknown = sorted(set(self.state_fields) - set(REFINEMENT_STATE_SCALAR_FIELDS))
        missing = sorted(set(REFINEMENT_STATE_SCALAR_FIELDS) - set(self.state_fields))
        if unknown or missing:
            raise ValueError(
                f"snapshot RefinementState fields do not match: unknown={unknown} missing={missing}"
            )
        restored = {name: value for name, value in self.state_fields.items() if name not in RUN_LIMIT_FIELDS}
        return dataclasses.replace(base, **restored)


def refinement_state_fields(state: RefinementState) -> dict:
    """The scalar fields of ``state`` as plain Python values."""

    values = {}
    for name in REFINEMENT_STATE_SCALAR_FIELDS:
        value = getattr(state, name)
        if value is None:
            values[name] = None
        elif isinstance(value, (bool, np.bool_)):
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
    written_with = {
        key[len("consistency_") :]: str(value) for key, value in snapshot.extra.items() if key.startswith("consistency_")
    }
    if written_with != options.consistency.non_default():
        problems.append(
            f"the run files were written with RELION-consistency options {written_with}, "
            f"this run asks for {options.consistency.non_default()}"
        )
    if problems:
        raise ValueError("cannot continue from the run files: " + "; ".join(problems))


def _host_spectra(data_vs_prior, noise_shells) -> dict:
    """The spectra every checkpoint holds in one layout: the scheduling curve and each half's noise shells."""

    return dict(
        data_vs_prior=np.array(data_vs_prior, dtype=np.float64),
        noise_shells=[np.array(shells, dtype=np.float64) for shells in noise_shells],
    )


def _host_unfiltered_means(unfiltered_means):
    """Host copies of the unregularized maps, or ``None`` when no half has one."""

    if unfiltered_means is None or all(m is None for m in unfiltered_means):
        return None
    return host_half_pair(unfiltered_means)


def _host_direction_prior(direction_priors):
    """Host copies of the two halves' direction priors, or ``None`` when neither half has one."""

    direction_prior = [p.values for p in direction_priors]
    if any(prior is not None for prior in direction_prior):
        return host_half_pair(direction_prior)
    return None


def _host_particle_state(
    half_inputs, max_posterior, significant_counts, avg_norm_correction, *, direction_priors, grid_size, consistency
) -> dict:
    """Per-particle arrays of both halves, the norm corrections, and the dtype, prior-order and consistency tags.

    ``consistency`` is the run's non-default RELION-consistency options (``SnapshotCapture.consistency``).
    """

    direction_prior_order = [p.healpix_order for p in direction_priors]
    eulers = host_half_pair([particle_half.rotation_eulers for particle_half in half_inputs])
    translations = host_half_pair([particle_half.translations for particle_half in half_inputs])
    image_corrections = host_half_pair([particle_half.image_corrections for particle_half in half_inputs])

    def dtype_name(values):
        present = [value for value in values if value is not None]
        return str(present[0].dtype) if present else "float32"

    return dict(
        rotation_eulers=eulers,
        translations=translations,
        image_corrections=image_corrections,
        scale_corrections=host_half_pair([particle_half.scale_corrections for particle_half in half_inputs]),
        group_ids=[
            np.zeros(0 if euler is None else len(euler), dtype=np.int64)
            if group_ids is None
            else np.array(group_ids, dtype=np.int64)
            for group_ids, euler in zip([particle_half.group_ids for particle_half in half_inputs], eulers)
        ],
        max_posterior=host_half_pair(max_posterior),
        significant_counts=host_half_pair(significant_counts),
        # No norm correction (a subtomogram run, --no_norm): RELION's 1.0 in relax's frame.
        avg_norm_correction=tuple(
            float(grid_size) ** 2 if value is None else float(value) for value in avg_norm_correction
        ),
        extra={
            "euler_dtype": dtype_name(eulers),
            "translation_dtype": dtype_name(translations),
            "correction_dtype": dtype_name(image_corrections),
            **{
                f"direction_prior_order_half{h + 1}": (
                    -1 if order is None else int(order)
                )
                for h, order in enumerate(direction_prior_order or [])
            },
            **{f"consistency_{name}": value for name, value in sorted(consistency.items())},
        },
    )


@dataclass
class _SnapshotAssembly:
    """Host-owned fields accumulated before one complete snapshot is published."""

    values: dict


@dataclass(frozen=True, kw_only=True)
class SnapshotCapture:
    """Run-level snapshot settings reused for every numbered checkpoint."""

    n_classes: int
    grid_size: int
    voxel_size: float
    tau2_fudge: float
    # The run's non-default RELION-consistency options (RelionConsistencyOptions.non_default),
    # recorded so a continuation cannot silently mix two rules.
    consistency: dict = field(default_factory=dict)

    def begin(
        self,
        relion_iteration,
        state,
        *,
        sigma_offset_angstrom_per_half,
        current_size,
        incr_size,
        has_high_fsc_at_limit,
        random_perturbation,
        acc_rot_per_class,
        acc_trans_per_class_angstrom,
    ) -> _SnapshotAssembly:
        """Capture scalar run and sampling state for one numbered iteration."""

        return _SnapshotAssembly(
            values={
                "relion_iteration": int(relion_iteration),
                "n_classes": int(self.n_classes),
                "ori_size": int(self.grid_size),
                "pixel_size": float(self.voxel_size),
                "tau2_fudge": float(self.tau2_fudge),
                "sigma_offset_angstrom": tuple(
                    float(v) for v in sigma_offset_angstrom_per_half
                ),
                "current_size": int(current_size),
                "incr_size": int(incr_size),
                "has_high_fsc_at_limit": bool(has_high_fsc_at_limit),
                "random_perturbation": float(random_perturbation),
                "state_fields": refinement_state_fields(state),
                "acc_rot_per_class": np.array(
                    acc_rot_per_class,
                    dtype=np.float64,
                ),
                "acc_trans_per_class_angstrom": np.array(
                    acc_trans_per_class_angstrom,
                    dtype=np.float64,
                ),
            }
        )

    def finish(
        self,
        assembly,
        means,
        unfiltered_means,
        tau2_shells,
        data_vs_prior,
        noise_shells,
        *,
        fsc,
        fsc_for_growth,
        class_weights,
        direction_priors,
        half_inputs,
        class_assignments,
        max_posterior,
        significant_counts,
        avg_norm_correction,
    ) -> IterationSnapshot:
        """Complete a checkpoint in the layout of the run's mode.

        This is the one remaining mode decision of checkpoint capture, to be
        removed when the K=1 and Class3D trajectories call ``finish_k1`` and
        ``finish_class`` directly. Each form takes its own mode's operands: the
        caller passes ``None`` for the two FSC curves in Class3D and for the
        two class operands in K=1, and those are not forwarded.
        """
        if int(self.n_classes) > 1:
            return self.finish_class(
                assembly,
                means,
                unfiltered_means,
                tau2_shells,
                data_vs_prior,
                noise_shells,
                class_weights=class_weights,
                direction_priors=direction_priors,
                half_inputs=half_inputs,
                class_assignments=class_assignments,
                max_posterior=max_posterior,
                significant_counts=significant_counts,
                avg_norm_correction=avg_norm_correction,
            )
        return self.finish_k1(
            assembly,
            means,
            unfiltered_means,
            tau2_shells,
            data_vs_prior,
            noise_shells,
            fsc=fsc,
            fsc_for_growth=fsc_for_growth,
            direction_priors=direction_priors,
            half_inputs=half_inputs,
            max_posterior=max_posterior,
            significant_counts=significant_counts,
            avg_norm_correction=avg_norm_correction,
        )

    def finish_k1(
        self,
        assembly,
        means,
        unfiltered_means,
        tau2_shells_per_half,
        data_vs_prior,
        noise_shells,
        *,
        fsc,
        fsc_for_growth,
        direction_priors,
        half_inputs,
        max_posterior,
        significant_counts,
        avg_norm_correction,
    ) -> IterationSnapshot:
        """Copy a K=1 iteration's maps, spectra, priors and particles into a complete checkpoint.

        One map and one tau2 curve per half, the split-half FSC and the curve
        that drives image-size growth; no class operands.

        Fills ``assembly`` (from ``begin``) in place. ``begin`` precedes array capture so replacing the prior
        header releases
        the preceding checkpoint's retained arrays before new maps are copied.
        See ``docs/math/relion_refinement_algorithm.md#checkpoint-capture``.
        """
        tau2 = np.stack([np.asarray(shells, dtype=np.float64) for shells in tau2_shells_per_half])
        assembly.values.update(
            means=host_half_pair(means),
            tau2_shells=tau2,
            **_host_spectra(data_vs_prior, noise_shells),
            fsc=host_array(fsc, np.float64),
            fsc_for_growth=host_array(fsc_for_growth, np.float64),
            unfiltered_means=_host_unfiltered_means(unfiltered_means),
        )
        assembly.values.update(
            class_weights=None,
            direction_prior=_host_direction_prior(direction_priors),
        )
        assembly.values.update(
            **_host_particle_state(
                half_inputs,
                max_posterior,
                significant_counts,
                avg_norm_correction,
                direction_priors=direction_priors,
                grid_size=self.grid_size,
                consistency=self.consistency,
            ),
            class_assignments=None,
        )
        return IterationSnapshot(**assembly.values)

    def finish_class(
        self,
        assembly,
        means,
        unfiltered_means,
        tau2_shells,
        data_vs_prior,
        noise_shells,
        *,
        class_weights,
        direction_priors,
        half_inputs,
        class_assignments,
        max_posterior,
        significant_counts,
        avg_norm_correction,
    ) -> IterationSnapshot:
        """Copy a Class3D iteration's maps, spectra, priors and particles into a complete checkpoint.

        One class stack that both half slots hold, one tau2 curve per class,
        the class weights and each particle's class; no FSC curves.

        Fills ``assembly`` (from ``begin``) in place. ``begin`` precedes array capture so replacing the prior
        header releases
        the preceding checkpoint's retained arrays before new maps are copied.
        See ``docs/math/relion_refinement_algorithm.md#checkpoint-capture``.
        """
        tau2 = np.array(tau2_shells, dtype=np.float64)
        assembly.values.update(
            means=[host_array(means[0])] * 2,
            tau2_shells=tau2,
            **_host_spectra(data_vs_prior, noise_shells),
            fsc=None,
            fsc_for_growth=None,
            unfiltered_means=_host_unfiltered_means(unfiltered_means),
        )
        assembly.values.update(
            class_weights=host_array(class_weights, np.float64),
            direction_prior=_host_direction_prior(direction_priors),
        )
        assembly.values.update(
            **_host_particle_state(
                half_inputs,
                max_posterior,
                significant_counts,
                avg_norm_correction,
                direction_priors=direction_priors,
                grid_size=self.grid_size,
                consistency=self.consistency,
            ),
            class_assignments=host_half_pair(class_assignments),
        )
        return IterationSnapshot(**assembly.values)
