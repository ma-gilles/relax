"""The VDAM shrinkage gate's noise reference: RELION's, and the power gates.

The gate of a gradient iteration is rho / (1 + rho), rho = 2 * fudge * signal / disagreement, per Fourier shell
(docs/math/vdam_gate_derivation.md, section 4). It stands in for the FSC of two half maps of the full data set.
:class:`RelionGate` is RELION's: shell means of amplitudes, with the disagreement of the two half-batch first moments
taken as measured. The other gates take shell means of power (section 7.2 of the note) and differ in the noise the
disagreement is rescaled to, a factor the M-step multiplies onto it (:class:`relax.vdam.relion_mstep.MStepGate`).
:class:`CrossWindowGate` is not a power ratio but the Wiener weight from the halves' cross power (section 10).

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

from relax.vdam.relion_mstep import FIRST_MOMENT_DECAY, MStepGate
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

    def advance(self, state: InitialModelState) -> tuple[InitialModelState, MStepGate]:
        """``state`` and the scale 1: nothing to carry."""
        return state, MStepGate(1.0, cross_power=False)


@dataclass(frozen=True)
class FullDataPowerGate:
    """The full-data power gate with independent batches; ``n_particles`` is the size of the data set.

    The state carries the noise weight of the first moments (``InitialModelState.gate_noise_weight``). A voxel that
    enters the Fourier sphere after the first iteration starts its moment from zero, not from a raw direction; the
    weight counts every voxel as averaged since iteration 1.
    """

    n_particles: int
    name = "full_data_power"

    def advance(self, state: InitialModelState) -> tuple[InitialModelState, MStepGate]:
        """The state after this iteration's mini-batch joined the moments, and the scale for its disagreement.

        An iteration without pseudo-halfsets (RELION's EM tail) has no disagreement; the scale is then 1 and the
        weight unchanged.
        """
        if not state.pseudo_halfsets:
            return state, MStepGate(1.0, cross_power=False)
        weight = advance_noise_weight(state.gate_noise_weight, int(state.subset_particle_ids.size))
        scale = full_data_disagreement_scale(weight, self.n_particles)
        return replace(state, gate_noise_weight=weight), MStepGate(scale, cross_power=False)


@dataclass(frozen=True)
class FullDataOverlapPowerGate:
    """The full-data power gate from per-particle noise coefficients; ``n_particles`` is the size of the data set.

    The state carries each particle's weight in its half's first moment (``InitialModelState.gate_noise_coefficients``).
    Equal information per particle is assumed, and the same voxel caveat as :class:`FullDataPowerGate` applies.
    """

    n_particles: int
    name = "full_data_power_overlap"

    def advance(self, state: InitialModelState) -> tuple[InitialModelState, MStepGate]:
        """The state after this iteration's mini-batch joined the moments, and the scale for its disagreement.

        An iteration without pseudo-halfsets (RELION's EM tail) has no disagreement; the scale is then 1 and the
        coefficients unchanged.
        """
        if not state.pseudo_halfsets:
            return state, MStepGate(1.0, cross_power=False)
        coefficients = advance_noise_coefficients(
            state.gate_noise_coefficients,
            np.asarray(state.subset_particle_ids),
            np.asarray(state.subset_halfset_ids),
            self.n_particles,
        )
        scale = overlap_disagreement_scale(coefficients, self.n_particles)
        return replace(state, gate_noise_coefficients=coefficients), MStepGate(scale, cross_power=False)


def seen_disagreement_scale(coefficients: np.ndarray) -> float:
    """:func:`overlap_disagreement_scale` with the data set replaced by the particles seen so far.

    The particles seen are those with a nonzero weight in the first moments (a weight decays by 0.9 per iteration and
    reaches the float64 underflow only after about 7000 iterations). The reference holds no particle it has not seen,
    so its noise is at least that of two half maps of the particles seen: 1 at the first iteration (the batch itself),
    the full-data scale once every particle has been seen.
    """
    return overlap_disagreement_scale(coefficients, int(np.count_nonzero(coefficients)))


def advance_window_coefficients(
    moment: np.ndarray | None,
    reference: np.ndarray | None,
    particle_ids: np.ndarray,
    halfset_ids: np.ndarray,
    n_particles: int,
    step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Each particle's weight in the noise of the mean first moment and of the reference after one more iteration.

    VDAM's direction is the halves' reconstruction minus the reference, so with the gate open (1) and the second moment
    at 1 the noise follows mu <- 0.9 mu + 0.1 (delta - c) and c <- c + step mu, with delta the particle's weight in the
    mean of the two half reconstructions (1 / (2 n_h) in half h) and mu = delta - c at the first iteration (the moment
    is then the direction itself). ``moment`` and ``reference`` are mu and c after the previous iteration (None before
    the first; the initial reference holds no particle noise). Returns the new (mu, c), float64 per particle.
    """
    delta = np.zeros(n_particles)
    for half in (0, 1):
        members = particle_ids[halfset_ids == half]
        if members.size:
            delta[members] = 0.5 / members.size
    if moment is None:
        reference = np.zeros(n_particles)
        moment = delta
    else:
        moment = FIRST_MOMENT_DECAY * moment + (1.0 - FIRST_MOMENT_DECAY) * (delta - reference)
    return moment, reference + step * moment


def window_disagreement_scale(reference: np.ndarray, coefficients: np.ndarray) -> float:
    """The factor on the halves' disagreement power that gives the noise power of the reference itself.

    The reference's noise power is s^2 * sum(reference^2); in the convention of the half-map FSC (two halves, each with
    twice the noise of their mean) the gate's disagreement must stand for 4 s^2 * sum(reference^2), and the measured
    one is s^2 * sum(coefficients^2). The factor is 1 when every batch is all particles and the reference has settled.
    """
    return 4.0 * float(np.sum(reference**2)) / float(np.sum(coefficients**2))


@dataclass(frozen=True)
class SeenOverlapPowerGate:
    """:class:`FullDataOverlapPowerGate` with the particles seen so far in place of the data set (study variant)."""

    n_particles: int
    name = "seen_power_overlap"

    def advance(self, state: InitialModelState) -> tuple[InitialModelState, MStepGate]:
        """The state after this iteration's mini-batch joined the moments, and the scale for its disagreement."""
        if not state.pseudo_halfsets:
            return state, MStepGate(1.0, cross_power=False)
        coefficients = advance_noise_coefficients(
            state.gate_noise_coefficients,
            np.asarray(state.subset_particle_ids),
            np.asarray(state.subset_halfset_ids),
            self.n_particles,
        )
        scale = seen_disagreement_scale(coefficients)
        return replace(state, gate_noise_coefficients=coefficients), MStepGate(scale, cross_power=False)


@dataclass(frozen=True)
class WindowPowerGate:
    """The power gate with the reference's own noise, from the step and the moments (study variant).

    The state carries each particle's weight in its half's first moment (``gate_noise_coefficients``) and in the noise of
    the mean moment and of the reference (``gate_window_moment``, ``gate_window_reference``). The noise is that of an
    open gate (:func:`advance_window_coefficients`); a shell the gate partly closes holds less.
    """

    n_particles: int
    name = "window_power"

    def advance(self, state: InitialModelState) -> tuple[InitialModelState, MStepGate]:
        """The state after this iteration's mini-batch and step, and the scale for its disagreement."""
        if not state.pseudo_halfsets:
            return state, MStepGate(1.0, cross_power=False)
        ids, halves = np.asarray(state.subset_particle_ids), np.asarray(state.subset_halfset_ids)
        coefficients = advance_noise_coefficients(state.gate_noise_coefficients, ids, halves, self.n_particles)
        moment, reference = advance_window_coefficients(
            state.gate_window_moment,
            state.gate_window_reference,
            ids,
            halves,
            self.n_particles,
            float(state.grad_current_stepsize),
        )
        state = replace(
            state, gate_noise_coefficients=coefficients, gate_window_moment=moment, gate_window_reference=reference
        )
        return state, MStepGate(window_disagreement_scale(reference, coefficients), cross_power=False)


@dataclass(frozen=True)
class CrossWindowGate(WindowPowerGate):
    """The Wiener weight from the halves' cross power, with :class:`WindowPowerGate`'s noise of the reference itself
    (section 10 of the note): signal and noise are both measured from the halves, not from the gated reference."""

    name = "cross_window"

    def advance(self, state: InitialModelState) -> tuple[InitialModelState, MStepGate]:
        """:class:`WindowPowerGate`'s state and scale, for the cross-power gate."""
        state, gate = WindowPowerGate.advance(self, state)
        return state, replace(gate, cross_power=True)


VdamGate = (
    RelionGate | PowerGate | FullDataPowerGate | FullDataOverlapPowerGate | SeenOverlapPowerGate | WindowPowerGate
    | CrossWindowGate
)


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
    if name == "seen_power_overlap":
        return SeenOverlapPowerGate(n_particles=n_particles)
    if name == "window_power":
        return WindowPowerGate(n_particles=n_particles)
    if name == "cross_window":
        return CrossWindowGate(n_particles=n_particles)
    raise ValueError(f"unknown VDAM gate: {name!r}")
