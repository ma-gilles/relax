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

import jax.numpy as jnp
import numpy as np

from relax.dense.scoring_policy import _dense_global_scoring_dtype
from relax.diagnostics.relion_replay import (
    _past_perturb_replay_max_iter,
    _perturbation_restart_state_iteration,
    _resolve_replay_random_perturbation,
    apply_iter_replay_overrides,
)
from relax.refinement.half_inputs import SigmaOffset
from relax.refinement.mean_helpers import class_mixture_from_weights
from relax.refinement.ports import InputSource, NumberedState

# The controller's log: what the replay installs is logged under its name, as before.
logger = logging.getLogger("relax.refinement.iteration_loop")


class RelionReplaySource(InputSource):
    """Replays a RELION run's per-iteration state into the numbered iterations of ``options``' run."""

    def __init__(self, options):
        self.options = options
        self._cutoff_announced = False

    @classmethod
    def from_options(cls, options) -> InputSource:
        """The source the run's options ask for: this replay when they name override slots or a STAR replay
        directory, else the native source."""
        if options.replay.replay_iteration_overrides is None and options.parity.perturb_replay_relion_dir is None:
            return InputSource()
        return cls(options)

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
        directory = self._star_directory(iteration)
        if directory is None and parity.perturb_replay_relion_dir is not None and not self._cutoff_announced:
            self._cutoff_announced = True
            logger.info(
                "Replay override: disabling RELION per-iteration STAR replay from "
                "iteration %d onward (--replay-override-max-iter %d)",
                iteration + 1,
                parity.perturb_replay_max_iter,
            )
        return directory

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
