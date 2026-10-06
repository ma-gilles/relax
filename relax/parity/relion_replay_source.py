"""A RELION run as the input source of the numbered iterations (code rule 15).

Each numbered iteration takes from RELION what the run supplies: its override slot (the numbered STAR
replay's per-iteration state, ``--relion_init_dir``'s run_it000 state, captured projectors) and, while the
STAR replay is live (``--perturb_replay_relion_dir`` up to ``perturb_replay_max_iter``), the sampling
controls and direction priors of its numbered STAR files. ``relax.diagnostics.relion_replay`` reads and
installs them.

While the replay's settings are still fields of the run's options, the source reads them there
(``from_options``); they are not copied.
"""

from __future__ import annotations

import logging
import os

import jax.numpy as jnp
import numpy as np

from relax import sampling
from relax.dense.scoring_policy import _dense_global_scoring_dtype
from relax.diagnostics.relion_replay import (
    _class_tau2_replay,
    _has_numbered_replay_iteration_overrides,
    _past_perturb_replay_max_iter,
    _perturbation_restart_state_iteration,
    _prepare_final_replay_references,
    _resolve_replay_random_perturbation,
    _restore_convergence_state_from_replay_restart,
    _select_final_replay_override,
    apply_final_replay_state,
    apply_iter_replay_overrides,
    apply_optimiser_convergence_replay,
    read_optimiser_accuracy_replay,
    select_final_sampling_star,
)
from relax.refinement.final_sampling import FinalSamplingSettings, native_final_sampling_settings
from relax.refinement.half_inputs import SigmaOffset
from relax.refinement.mean_helpers import class_mixture_from_weights
from relax.refinement.ports import ClassTau2, FinalState, InputSource, NumberedState
from relax.refinement.refinement_options import FINAL_ALL_DATA_DISABLE_REPLAY_LAST_NUMBERED_STATE_ENV
from relax.relion.relion_metadata import read_relion_sampling_metadata

# The controller's log: what the replay installs is logged under its name, as before.
logger = logging.getLogger("relax.refinement.iteration_loop")


class RelionReplaySource(InputSource):
    """Replays a RELION run's per-iteration state into the numbered iterations of ``options``' run."""

    def __init__(self, options):
        self.options = options
        self._cutoff_announced = False
        # The STAR directory of the last iteration asked about: the one the final pass replays from.
        self._live_directory = options.parity.perturb_replay_relion_dir

    @classmethod
    def from_options(cls, options) -> InputSource:
        """The source the run's options ask for: this replay when they name override slots, a STAR replay
        directory or a final-pass replay, else the native source."""
        return cls(options) if replays(options) else InputSource()

    def _slot(self, iteration: int):
        slots = self.options.replay.replay_iteration_overrides
        return slots[iteration] if slots is not None and iteration < len(slots) else None

    def _star_directory(self, iteration: int):
        parity = self.options.parity
        if parity.perturb_replay_relion_dir is None or _past_perturb_replay_max_iter(iteration, parity.perturb_replay_max_iter):
            return None
        return parity.perturb_replay_relion_dir

    def relion_run_directory(self, iteration):
        """The STAR replay's directory up to ``perturb_replay_max_iter`` (announced once when it ends), else None."""
        parity = self.options.parity
        directory = self._live_directory = self._star_directory(iteration)
        if directory is None and parity.perturb_replay_relion_dir is not None and not self._cutoff_announced:
            self._cutoff_announced = True
            logger.info(
                "Replay override: disabling RELION per-iteration STAR replay from "
                "iteration %d onward (--replay-override-max-iter %d)",
                iteration + 1,
                parity.perturb_replay_max_iter,
            )
        return directory

    def final_state(self, inputs, *, means, numbered_iteration_count, halves, direction_priors, healpix_order,
                    image_geometry):
        """The final pass's RELION state: a final-only replay's references and state, else the last numbered
        slot's (when the replay has numbered slots, or RELAX_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE asks and
        RELAX_FINAL_ALL_DATA_DISABLE_REPLAY_LAST_NUMBERED_STATE does not forbid it). Installs the particle state
        in the ``halves`` and the priors in ``direction_priors`` in place, as apply_final_replay_state does."""
        options, replay = self.options, self.options.replay
        environment = options.debug.environment
        n_classes = int(options.k_class.n_classes)
        k_class_enabled = n_classes > 1
        final_replay_override = replay.final_replay_override
        sigma_offset, noise_model = inputs.sigma_offset, inputs.noise_model
        disabled = environment.final_all_data_disable_replay_last_numbered_state
        has_overrides = replay.replay_iteration_overrides is not None and len(replay.replay_iteration_overrides) > 0
        join_means = _prepare_final_replay_references(
            replay=replay,
            diagnostic_override=final_replay_override,
            numbered_iteration_count=numbered_iteration_count,
            means=means,
            final_join_means=inputs.join_means,
            k_class_enabled=k_class_enabled,
            logger=logger,
        )
        replay_last_numbered_state = final_replay_override is not None or (
            not disabled
            and (
                environment.final_all_data_replay_last_numbered_state
                or _has_numbered_replay_iteration_overrides(replay.replay_iteration_overrides)
            )
        )
        if replay_last_numbered_state:
            override_index, override = _select_final_replay_override(
                requested_index=numbered_iteration_count,
                diagnostic_override=final_replay_override,
                replay_overrides=replay.replay_iteration_overrides,
                has_overrides=has_overrides,
                logger=logger,
            )
            if override is not None:
                sigma_offset, noise_model = apply_final_replay_state(
                    override,
                    halves,
                    direction_priors,
                    sigma_offset=sigma_offset,
                    noise_model=noise_model,
                    n_classes=n_classes,
                    healpix_order=healpix_order,
                    image_shape=image_geometry.image_shape,
                    symmetry=options.symmetry.point_group,
                    dtype=_dense_global_scoring_dtype(),
                    override_index=override_index,
                    log=logger,
                )
        elif not k_class_enabled and disabled and has_overrides:
            logger.info(
                "Diagnostic %s=1: final all-data skips automatic last-numbered RELION state replay",
                FINAL_ALL_DATA_DISABLE_REPLAY_LAST_NUMBERED_STATE_ENV,
            )
        return FinalState(join_means, sigma_offset, noise_model)

    def final_sampling_settings(self, state, image_geometry, *, grid_order, relion_iteration, native):
        """RELION's final (or last numbered) sampling STAR of the final-pass replay directory, else of the live
        STAR replay; with neither, the run's own. A replay without the STAR leaves the final grid unperturbed."""
        options, parity = self.options, self.options.parity
        n_classes = int(options.k_class.n_classes)
        last_numbered_iteration = relion_iteration - 1
        active_replay_dir = self._live_directory
        replay_dir = (
            parity.final_sampling_replay_relion_dir
            if parity.final_sampling_replay_relion_dir is not None
            else active_replay_dir
        )
        if replay_dir is None:
            return native()
        star, source, candidates = select_final_sampling_star(
            replay_dir, parity.perturb_replay_relion_prefix,
            final_iteration=relion_iteration,
            previous_iteration=last_numbered_iteration,
            require_final_state=options.replay.replay_iteration_overrides is not None,
        )
        if star is not None:
            metadata = read_relion_sampling_metadata(star)
            replay_iteration = last_numbered_iteration if source == "last-numbered" else relion_iteration
            perturbation, perturbation_source = _resolve_replay_random_perturbation(
                star_value=metadata["random_perturbation"],
                perturbation_factor=metadata["perturbation_factor"],
                relion_iteration=replay_iteration,
                replay_dir=str(replay_dir),
                replay_prefix=parity.perturb_replay_relion_prefix,
                explicit_seed=parity.perturb_seed,
                precision_mode=parity.perturb_replay_precision,
                restart_state_iteration=_perturbation_restart_state_iteration(
                    parity.perturb_replay_restart_state_iterations, replay_iteration,
                ),
            )
            pixel_size = image_geometry.pixel_size_angstrom
            settings = FinalSamplingSettings(
                relion_iteration=relion_iteration,
                grid_order=grid_order,
                perturbation_order=metadata["healpix_order"],
                translation_range=metadata["offset_range"] / pixel_size,
                translation_step=metadata["offset_step"] / pixel_size,
                pixel_size_angstrom=pixel_size,
                perturbation_factor=metadata["perturbation_factor"],
                perturbation=perturbation,
                sampling_star=star,
                sampling_star_source=source,
            )
            _log_replayed_translation_grid_change(
                settings, replay_dir=replay_dir,
                replay_prefix=parity.perturb_replay_relion_prefix, n_classes=n_classes,
            )
            logger.info(
                "Perturbation replay: final all-data relion_iter=%d rp=%+.12g pf=%.3f "
                "relion_hp_order=%d offset_range=%.3f px offset_step=%.3f px source=%s/%s",
                replay_iteration, settings.random_perturbation, settings.perturbation_factor,
                settings.perturbation_order, settings.translation_range, settings.translation_step,
                source, perturbation_source,
            )
            return settings
        logger.info(
            "Perturbation replay: final all-data sampling STAR missing for relion_iter=%d (%s); "
            "leaving final trial grid unperturbed",
            relion_iteration, ", ".join(path for path, _source in candidates),
        )
        # A final-only replay directory historically applies a zero perturbation
        # when native perturbation is enabled, without advancing its RNG.
        return native_final_sampling_settings(
            state, image_geometry, options, grid_order=grid_order, relion_iteration=relion_iteration,
            perturbation=0.0 if active_replay_dir is None and parity.perturb_factor > 0 else None,
        )

    def class_tau2(self, iteration, n_classes):
        """The slot's captured Class3D prior, used when RELAX_KCLASS_REPLAY_TAU2 asks for it (validated always)."""
        shells, enabled, label = _class_tau2_replay(
            iteration=iteration, n_classes=n_classes, iter_replay_override=self._slot(iteration),
            replay=self.options.replay, logger=logger,
        )
        return ClassTau2(shells if enabled else None, label)

    def restore_convergence_state(self, state):
        """A replay restart (init_relion_iteration > 0, no sealed sampling state) resumes RELION's convergence
        counters from its optimiser and model STAR files."""
        options = self.options
        if (
            options.debug.sealed_sampling_state is not None
            or options.parity.perturb_replay_relion_dir is None
            or int(options.schedule.init_relion_iteration) <= 0
        ):
            return False
        _restore_convergence_state_from_replay_restart(state, options)
        return True

    def convergence_accuracy(self, iteration, accuracy):
        """RELION's numbered optimiser accuracies, while the STAR replay is live and no sealed state is."""
        return read_optimiser_accuracy_replay(
            replay_dir=self._star_directory(iteration),
            replay_prefix=self.options.parity.perturb_replay_relion_prefix,
            init_relion_iteration=self.options.schedule.init_relion_iteration,
            iteration=iteration,
            sealed_sampling_state=self.options.debug.sealed_sampling_state,
            acc_rot=accuracy.acc_rot,
            acc_trans=accuracy.acc_trans,
            convergence_acc_rot=accuracy.convergence_acc_rot,
            convergence_acc_trans=accuracy.convergence_acc_trans,
            logger=logger,
        )

    def apply_optimiser_controls(self, iteration, state, accuracy):
        """The numbered optimiser STAR's counters, changes and convergence flag, installed in ``state``."""
        if accuracy.metadata is not None:
            apply_optimiser_convergence_replay(
                state,
                metadata=accuracy.metadata,
                optimiser_star=accuracy.optimiser_star,
                optimiser_iteration=accuracy.optimiser_iteration,
                replay_dir=self._star_directory(iteration),
                replay_prefix=self.options.parity.perturb_replay_relion_prefix,
                logger=logger,
            )

    def random_perturbation(self, iteration, sampling_meta, native):
        """The perturbation of RELION's sampling STAR (exact from its seed where the precision allows), when the
        STAR set this iteration's sampling; otherwise the run's own."""
        if sampling_meta is None or sampling_meta.get("sealed_v3", False):
            return native()
        parity = self.options.parity
        relion_iteration = self.options.schedule.init_relion_iteration + iteration + 1
        restart_iteration = _perturbation_restart_state_iteration(
            parity.perturb_replay_restart_state_iterations, relion_iteration,
        )
        perturbation, source = _resolve_replay_random_perturbation(
            star_value=float(sampling_meta["random_perturbation"]),
            perturbation_factor=float(sampling_meta["perturbation_factor"]),
            relion_iteration=relion_iteration,
            replay_dir=str(self._star_directory(iteration)),
            replay_prefix=parity.perturb_replay_relion_prefix,
            explicit_seed=parity.perturb_seed,
            precision_mode=str(parity.perturb_replay_precision),
            restart_state_iteration=restart_iteration,
        )
        logger.info(
            "Perturbation replay: iter=%d rp=%+.12g pf=%.3f relion_hp_order=%d source=%s",
            iteration + 1, perturbation,
            float(sampling_meta["perturbation_factor"]),
            int(sampling_meta["healpix_order"]), source,
        )
        return perturbation

    def numbered_state(self, iteration, inputs, *, state, halves, direction_priors, image_geometry, sampling_sealed):
        """RELION's sampling controls, priors, particle state, noise, tau2 and class weights for this iteration.

        Updates ``state``'s sampling controls, the ``halves``' poses and corrections and the
        ``direction_priors`` list in place, as ``apply_iter_replay_overrides`` does.
        """
        options = self.options
        slot = self._slot(iteration)
        result = apply_iter_replay_overrides(
            iter_replay_override=slot,
            perturb_replay_relion_dir=None if sampling_sealed else self._star_directory(iteration),
            perturb_replay_relion_prefix=options.parity.perturb_replay_relion_prefix,
            init_relion_iteration=options.schedule.init_relion_iteration,
            iteration=iteration,
            state=state,
            cs=inputs.current_size,
            image_geometry=image_geometry,
            n_classes=int(options.k_class.n_classes),
            relion_half_inputs=halves,
            previous_best_rotations=inputs.previous_best_rotations,
            noise_model=inputs.noise_model,
            current_sigma_offset_angstrom=inputs.sigma_offset.shared_angstrom,
            current_sigma_offset_angstrom_per_half=inputs.sigma_offset.per_half_angstrom,
            direction_priors=direction_priors,
            preserve_existing_direction_prior=options.replay.preserve_initial_direction_prior,
            dtype=_dense_global_scoring_dtype(),
            symmetry=options.symmetry.point_group,
        )
        mean_variance = inputs.mean_variance
        replay_mean_variance = None if slot is None else slot.get("mean_variance")
        if replay_mean_variance is not None:
            replay_mean_variance = np.asarray(replay_mean_variance, dtype=np.float64).reshape(-1)
            expected_shape = tuple(mean_variance.shape)
            if replay_mean_variance.shape != expected_shape:
                raise ValueError(
                    f"K=1 replay mean_variance shape mismatch: expected {expected_shape}, got {replay_mean_variance.shape}"
                )
            mean_variance = jnp.asarray(replay_mean_variance)
            logger.info("Replay override: K=1 tau2/mean_variance <- model.star")
        class_mixture = inputs.class_mixture
        if int(options.k_class.n_classes) > 1 and result.class_weights is not None:
            class_mixture = class_mixture_from_weights(np.asarray(result.class_weights, dtype=np.float64))
            logger.info(
                "Replay override: class priors <- direction-prior row sums (%s)",
                ", ".join(f"class {idx + 1}={weight:.4f}" for idx, weight in enumerate(class_mixture.weights)),
            )
        return NumberedState(
            current_size=result.cs,
            noise_model=result.noise_model,
            sigma_offset=SigmaOffset(result.current_sigma_offset_angstrom, result.current_sigma_offset_angstrom_per_half),
            previous_best_rotations=result.previous_best_rotations,
            mean_variance=mean_variance,
            class_mixture=class_mixture,
            prior_translations=inputs.prior_translations if sampling_sealed else result.prior_translations,
            sampling_meta=inputs.sampling_meta if sampling_sealed else result.replay_meta,
            projector_state=result.relion_projector_state,
        )


def replays(options) -> bool:
    """Whether ``options`` name RELION state to replay: override slots, a STAR replay directory, a final-pass
    replay (its override, reference maps or sampling directory) or the last numbered state's replay in the
    final pass (RELAX_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE)."""
    replay, parity = options.replay, options.parity
    return any(
        value is not None
        for value in (
            replay.replay_iteration_overrides, parity.perturb_replay_relion_dir, replay.final_replay_override,
            replay.final_replay_reference_maps, parity.final_sampling_replay_relion_dir,
        )
    ) or options.debug.environment.final_all_data_replay_last_numbered_state


def _log_replayed_translation_grid_change(settings, *, replay_dir, replay_prefix, n_classes):
    if settings.sampling_star_source != "final":
        return
    numbered_path = os.path.join(
        replay_dir, f"{replay_prefix}_it{settings.relion_iteration - 1:03d}_sampling.star",
    )
    if not os.path.exists(numbered_path):
        return
    numbered = read_relion_sampling_metadata(numbered_path)
    numbered_range = numbered["offset_range"] / settings.pixel_size_angstrom
    numbered_step = numbered["offset_step"] / settings.pixel_size_angstrom
    numbered_grid = sampling._relion_base_translation_grid(
        numbered_range, numbered_step,
        n_classes=n_classes, voxel_size=settings.pixel_size_angstrom,
    ).astype(np.float32)
    final_grid = sampling._relion_base_translation_grid(
        settings.translation_range, settings.translation_step,
        n_classes=n_classes, voxel_size=settings.pixel_size_angstrom,
    ).astype(np.float32)
    if numbered_grid.shape != final_grid.shape or not np.allclose(
        numbered_grid, final_grid, rtol=0.0, atol=1e-6,
    ):
        logger.info(
            "RELION final all-data sampling grid differs from last numbered sampling: "
            "numbered n=%d range=%.9g step=%.9g hp=%d; final n=%d range=%.9g step=%.9g hp=%d",
            numbered_grid.shape[0], numbered_range, numbered_step, numbered["healpix_order"],
            final_grid.shape[0], settings.translation_range, settings.translation_step,
            settings.perturbation_order,
        )
