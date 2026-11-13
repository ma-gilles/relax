"""The VDAM shrinkage gate's noise reference: RELION's, and the power gates.

The gate of a gradient iteration is rho / (1 + rho), rho = 2 * fudge * signal / disagreement, per Fourier shell
(docs/math/vdam_gate_derivation.md, section 4). It stands in for the FSC of two half maps of the full data set.
:class:`RelionGate` is RELION's: shell means of amplitudes, with the disagreement of the two half-batch first moments
taken as measured. The other gates take shell means of power (section 7.2 of the note) and differ in the noise the
disagreement is rescaled to, a factor the M-step multiplies onto it
(:func:`relax.vdam.relion_mstep.relion_vdam_m_step_device`, ``disagreement_scale``):

- :class:`PowerGate`: none, the disagreement as measured (the half-batch noise after the running average);
- :class:`FullDataPowerGate`: two full-data half maps, with the running average's noise reduction of independent batches
  (sections 7.3 and 7.4: :func:`full_data_disagreement_scale`, (b / N) / e for a constant batch);
- :class:`FullDataOverlapPowerGate`: two full-data half maps, from each particle's weight in its half's first moment, so
  that a particle in several batches brings the same noise to each (section 7.6:
  :func:`overlap_disagreement_scale`; 1 when every batch is the whole data set).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from relax.vdam.relion_mstep import FIRST_MOMENT_DECAY
from relax.vdam.state import InitialModelState


def advance_noise_weight(weight: float | None, count: int) -> float:
    """The noise weight of the half first moments after one more iteration whose mini-batch holds ``count`` particles.

    A direction's noise power is inversely proportional to the particles of its half (count / 2 each), so the two
    halves' first moments differ by noise of power s^2 * ``weight``, with s^2 the noise power of one particle and
    ``weight`` the sum, over the iterations averaged into the moments, of (their coefficient)^2 * 4 / count.
    ``weight`` is None before the first iteration, whose direction is the moment itself. Batches are taken as
    independent: a particle drawn again brings new noise.
    """
    inverse = 4.0 / count
    if weight is None:
        return inverse
    return FIRST_MOMENT_DECAY**2 * weight + (1.0 - FIRST_MOMENT_DECAY) ** 2 * inverse


def full_data_disagreement_scale(weight: float, n_particles: int) -> float:
    """The factor on the halves' disagreement power that gives the noise power of two full-data half maps.

    The full-data half map of n_particles / 2 particles has noise power 2 s^2 / n_particles, and the disagreement of
    two such maps is twice that; the measured disagreement is s^2 * ``weight``. For a constant batch of b particles
    ``weight`` tends to 4 e / b with e = (1 - decay) / (1 + decay) the running average's noise reduction, so the factor
    is (b / n_particles) / e.
    """
    return 4.0 / (n_particles * weight)


def advance_noise_coefficients(
    coefficients: np.ndarray | None, particle_ids: np.ndarray, halfset_ids: np.ndarray, n_particles: int
) -> np.ndarray:
    """Each particle's weight in its half's first moment after one more iteration (float64, one per particle).

    ``particle_ids`` and ``halfset_ids`` are the iteration's batch: a half's direction averages the noise of its own
    particles, 1 / n_h each. ``coefficients`` are the values after the previous iteration, None before the first,
    whose direction is the moment itself. The moment keeps a particle's noise from every batch it was in, so its
    weight adds up over them instead of entering as new noise.
    """
    added = np.zeros(n_particles)
    for half in (0, 1):
        members = particle_ids[halfset_ids == half]
        if members.size:
            added[members] = 1.0 / members.size
    if coefficients is None:
        return added
    return FIRST_MOMENT_DECAY * coefficients + (1.0 - FIRST_MOMENT_DECAY) * added


def overlap_disagreement_scale(coefficients: np.ndarray, n_particles: int) -> float:
    """The factor on the halves' disagreement power that gives the noise power of two full-data half maps.

    The halves' first moments differ by noise of power s^2 * sum(coefficients^2) (each particle is in one half);
    two full-data half maps of ceil(N / 2) and floor(N / 2) particles differ by s^2 * (1 / ceil(N / 2) + 1 / floor(N / 2)).
    The factor is the ratio: (b / N) / e for batches that never repeat a particle, 1 when every batch is all of them.
    """
    full_data = 1.0 / ((n_particles + 1) // 2) + 1.0 / (n_particles // 2)
    return full_data / float(np.sum(coefficients**2))


@dataclass(frozen=True)
class RelionGate:
    """RELION's gate: the M-step takes no scale."""

    name = "relion"

    def advance(self, state: InitialModelState) -> tuple[InitialModelState, float | None]:
        """``(state, None)``: nothing to carry and no scale."""
        return state, None


@dataclass(frozen=True)
class PowerGate:
    """Shell means of power, with the disagreement as measured: the M-step's scale is 1."""

    name = "power"

    def advance(self, state: InitialModelState) -> tuple[InitialModelState, float]:
        """``(state, 1.0)``: nothing to carry."""
        return state, 1.0


@dataclass(frozen=True)
class FullDataPowerGate:
    """The full-data power gate with independent batches; ``n_particles`` is the size of the data set.

    The state carries the noise weight of the first moments (``InitialModelState.gate_noise_weight``). A voxel that
    enters the Fourier sphere after the first iteration starts its moment from zero, not from a raw direction; the
    weight counts every voxel as averaged since iteration 1.
    """

    n_particles: int
    name = "full_data_power"

    def advance(self, state: InitialModelState) -> tuple[InitialModelState, float]:
        """The state after this iteration's mini-batch joined the moments, and the scale for its disagreement.

        An iteration without pseudo-halfsets (RELION's EM tail) has no disagreement; the scale is then 1 and the
        weight unchanged.
        """
        if not state.pseudo_halfsets:
            return state, 1.0
        weight = advance_noise_weight(state.gate_noise_weight, int(state.subset_particle_ids.size))
        return replace(state, gate_noise_weight=weight), full_data_disagreement_scale(weight, self.n_particles)


@dataclass(frozen=True)
class FullDataOverlapPowerGate:
    """The full-data power gate from per-particle noise coefficients; ``n_particles`` is the size of the data set.

    The state carries each particle's weight in its half's first moment (``InitialModelState.gate_noise_coefficients``).
    Equal information per particle is assumed, and the same voxel caveat as :class:`FullDataPowerGate` applies.
    """

    n_particles: int
    name = "full_data_power_overlap"

    def advance(self, state: InitialModelState) -> tuple[InitialModelState, float]:
        """The state after this iteration's mini-batch joined the moments, and the scale for its disagreement.

        An iteration without pseudo-halfsets (RELION's EM tail) has no disagreement; the scale is then 1 and the
        coefficients unchanged.
        """
        if not state.pseudo_halfsets:
            return state, 1.0
        coefficients = advance_noise_coefficients(
            state.gate_noise_coefficients,
            np.asarray(state.subset_particle_ids),
            np.asarray(state.subset_halfset_ids),
            self.n_particles,
        )
        scale = overlap_disagreement_scale(coefficients, self.n_particles)
        return replace(state, gate_noise_coefficients=coefficients), scale


VdamGate = RelionGate | PowerGate | FullDataPowerGate | FullDataOverlapPowerGate


def vdam_gate(name: str, n_particles: int) -> VdamGate:
    """The gate of the option ``--vdam-gate`` (``InitialModelDefaults.vdam_gate``), for a data set of ``n_particles``."""
    if name == "relion":
        return RelionGate()
    if name == "power":
        return PowerGate()
    if name == "full_data_power":
        return FullDataPowerGate(n_particles=n_particles)
    if name == "full_data_power_overlap":
        return FullDataOverlapPowerGate(n_particles=n_particles)
    raise ValueError(f"unknown VDAM gate: {name!r}")
