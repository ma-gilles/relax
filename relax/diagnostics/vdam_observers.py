"""The InitialModel (VDAM) run observers the command chooses (code rule 15; the port is
``relax.vdam.ports.VdamObserver``).

``VdamDumpObserver`` writes the diagnostic dumps the environment asks for. ``vdam_command_observer`` reads the
variables once, when the command builds the run; the algorithm's modules never import this module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from relax.diagnostics.vdam_noise import dump_noise_failure_meta, dump_noise_update_boundary
from relax.vdam.ports import VdamObserver

EXPECTED_ACCURACY_DUMP_DIR_ENV = "RELAX_INITIALMODEL_EXPECTED_ACCURACY_DUMP_DIR"
EXPECTED_ACCURACY_DUMP_ITERATIONS_ENV = "RELAX_INITIALMODEL_EXPECTED_ACCURACY_DUMP_ITERATIONS"
NOISE_UPDATE_DUMP_DIR_ENV = "RELAX_INITIALMODEL_NOISE_UPDATE_DUMP_DIR"
NOISE_UPDATE_DUMP_ITERATION_ENV = "RELAX_INITIALMODEL_NOISE_UPDATE_DUMP_ITERATION"
NOISE_FAILURE_DUMP_DIR_ENV = "RELAX_INITIALMODEL_NOISE_FAILURE_DUMP_DIR"


def _iterations(value: str) -> frozenset[int]:
    """A comma-separated iteration list (empty: every iteration)."""
    return frozenset(int(token.strip()) for token in value.split(",") if token.strip())


@dataclass(frozen=True)
class VdamDumpObserver(VdamObserver):
    """The InitialModel's diagnostic dumps; a directory None turns its dump off, an empty iteration set dumps
    every iteration. ``expected_accuracy_dir`` receives each single-particle expected-accuracy estimate's inputs
    of ``expected_accuracy_iterations``; ``noise_update_dir`` each VDAM noise update's sums and spectra of
    ``noise_update_iterations`` (refusing to overwrite a file); ``noise_failure_dir`` the noise sums of the
    E-step meta when they are not finite."""

    expected_accuracy_dir: str | None = None
    expected_accuracy_iterations: frozenset[int] = frozenset()
    noise_update_dir: str | None = None
    noise_update_iterations: frozenset[int] = frozenset()
    noise_failure_dir: str | None = None

    def noise_updated(self, previous, updated, sums) -> None:
        if self.noise_update_dir is None:
            return
        if self.noise_update_iterations and int(previous.iter) not in self.noise_update_iterations:
            return
        noise = sums.noise
        per_group = noise.wsum_sigma2_noise.ndim == 2
        dump_noise_update_boundary(
            self.noise_update_dir,
            previous,
            updated,
            wsum_sigma2_noise=noise.wsum_sigma2_noise,
            wsum_img_power=noise.wsum_img_power,
            noise_sumw=noise.sumw if per_group else float(np.sum(noise.sumw)),
            wsum_noise_a2=noise.wsum_noise_a2,
            wsum_noise_xa=noise.wsum_noise_xa,
        )

    def noise_sums_nonfinite(self, state, meta, summaries) -> str | None:
        if self.noise_failure_dir is None:
            return None
        return dump_noise_failure_meta(self.noise_failure_dir, state, meta, summaries)

    def expected_accuracy_estimated(self, inputs, accuracy) -> None:
        state, particle_state, optics_state = inputs.state, inputs.particle_state, inputs.optics_state
        if self.expected_accuracy_dir is None or optics_state is None:
            return
        if self.expected_accuracy_iterations and int(state.iter) not in self.expected_accuracy_iterations:
            return
        n_trials = int(inputs.trial_particle_ids.size)
        trial_particle_ids = inputs.trial_particle_ids
        dump_path = Path(self.expected_accuracy_dir)
        dump_path.mkdir(parents=True, exist_ok=True)
        np.savez(
            dump_path / f"iter{int(state.iter):03d}_expected_accuracy_inputs.npz",
            refs_relion=inputs.references_relion,
            eulers=inputs.eulers,
            source_eulers_valid=(
                np.zeros(n_trials, dtype=bool)
                if particle_state.best_pose_eulers_valid is None
                else np.asarray(particle_state.best_pose_eulers_valid)[trial_particle_ids]
            ),
            source_eulers_deg=(
                np.zeros((n_trials, 3), dtype=np.float64)
                if particle_state.best_pose_eulers_deg is None
                else np.asarray(particle_state.best_pose_eulers_deg)[trial_particle_ids]
            ),
            trial_particle_ids=trial_particle_ids,
            class_ids=inputs.class_ids,
            pdf_class=np.asarray(state.pdf_class, dtype=np.float64),
            sigma2_noise=np.asarray(state.sigma2_noise[0], dtype=np.float64),
            defU=np.asarray(optics_state.defU, dtype=np.float64),
            defV=np.asarray(optics_state.defV, dtype=np.float64),
            defAngle=np.asarray(optics_state.defAngle, dtype=np.float64),
            phase_shift=np.asarray(optics_state.phase_shift, dtype=np.float64),
            voltage=np.asarray(optics_state.voltage, dtype=np.float64),
            Cs=np.asarray(optics_state.Cs, dtype=np.float64),
            Q0=np.asarray(optics_state.Q0, dtype=np.float64),
            pixel_size=np.asarray(float(optics_state.pixel_size), dtype=np.float64),
            ori_size=np.asarray(int(state.box_size), dtype=np.int64),
            current_image_size=np.asarray(inputs.current_image_size, dtype=np.int64),
            padding_factor=np.asarray(int(inputs.padding_factor), dtype=np.int64),
            sigma2_fudge=np.asarray(float(inputs.sigma2_fudge), dtype=np.float64),
            random_seed=np.asarray(int(inputs.random_seed), dtype=np.int64),
            random_seed_particle_ids=inputs.random_seed_particle_ids,
            acc_rot=np.asarray(accuracy.acc_rot, dtype=np.float64),
            acc_trans=np.asarray(accuracy.acc_trans_angstrom, dtype=np.float64),
        )


def vdam_command_observer(environ=None) -> VdamObserver:
    """The observer the environment of a ``relax initial_model`` command asks for (``VdamObserver()``: none)."""

    env = os.environ if environ is None else environ
    dumps = VdamDumpObserver(
        expected_accuracy_dir=env.get(EXPECTED_ACCURACY_DUMP_DIR_ENV, "").strip() or None,
        expected_accuracy_iterations=_iterations(env.get(EXPECTED_ACCURACY_DUMP_ITERATIONS_ENV, "").strip()),
        noise_update_dir=env.get(NOISE_UPDATE_DUMP_DIR_ENV) or None,
        noise_update_iterations=_iterations(env.get(NOISE_UPDATE_DUMP_ITERATION_ENV) or ""),
        noise_failure_dir=env.get(NOISE_FAILURE_DUMP_DIR_ENV) or None,
    )
    if dumps == VdamDumpObserver():
        return VdamObserver()
    return dumps
