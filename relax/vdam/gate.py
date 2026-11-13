"""The VDAM shrinkage gate's noise reference: RELION's, and the full-data power gate.

The gate of a gradient iteration is rho / (1 + rho), rho = 2 * fudge * signal / disagreement, per Fourier shell
(docs/math/vdam_gate_derivation.md, section 4). It stands in for the FSC of two half maps of the full data set.
:class:`RelionGate` is RELION's: shell means of amplitudes, with the disagreement of the two half-batch first moments
taken as measured. :class:`FullDataPowerGate` changes how rho is computed in three places (section 7 of the note): the
statistic is shell means of power, the disagreement is rescaled from the noise of the mini-batch to that of two
full-data half maps (b / N), and the running average of the first moments is divided out (1 / e). The last two are one
factor, :func:`full_data_disagreement_scale`, which the M-step multiplies onto the disagreement
(:func:`relax.vdam.relion_mstep.relion_vdam_m_step_device`, ``disagreement_scale``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from relax.vdam.relion_mstep import FIRST_MOMENT_DECAY
from relax.vdam.state import InitialModelState


def advance_noise_weight(weight: float | None, count: int) -> float:
    """The noise weight of the half first moments after one more iteration whose mini-batch holds ``count`` particles.

    A direction's noise power is inversely proportional to the particles of its half (count / 2 each), so the two
    halves' first moments differ by noise of power s^2 * ``weight``, with s^2 the noise power of one particle and
    ``weight`` the sum, over the iterations averaged into the moments, of (their coefficient)^2 * 4 / count.
    ``weight`` is None before the first iteration, whose direction is the moment itself.
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


@dataclass(frozen=True)
class RelionGate:
    """RELION's gate: the M-step takes no scale."""

    name = "relion"

    def advance(self, state: InitialModelState) -> tuple[InitialModelState, float | None]:
        """``(state, None)``: nothing to carry and no scale."""
        return state, None


@dataclass(frozen=True)
class FullDataPowerGate:
    """The full-data power gate; ``n_particles`` is the size of the data set the two half maps would hold.

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


def vdam_gate(name: str, n_particles: int) -> RelionGate | FullDataPowerGate:
    """The gate of the option ``--vdam-gate`` (``InitialModelDefaults.vdam_gate``), for a data set of ``n_particles``."""
    if name == "full_data_power":
        return FullDataPowerGate(n_particles=n_particles)
    if name == "relion":
        return RelionGate()
    raise ValueError(f"unknown VDAM gate: {name!r}")
