"""The port through which comparison enters an InitialModel run (code rule 15).

A :class:`VdamInputSource` supplies what a comparison run takes from elsewhere instead of what the run
computes, at the call sites the driver assigns: the M-step of each class, and the references after each
iteration's M-step. This base class is the native source. The command chooses the source once
(``relax.parity.vdam_replay``); the algorithm never imports an implementation.
"""

from __future__ import annotations

from relax.vdam.m_step import vdam_m_step_single_class
from relax.vdam.state import InitialModelState, VdamAccumulator


class VdamInputSource:
    """The native source: relax's own M-step, and the references it made."""

    def single_class_m_step(
        self, state: InitialModelState, k: int, accum_h0: VdamAccumulator, accum_h1: VdamAccumulator | None, **settings
    ) -> InitialModelState:
        """Class ``k``'s M-step (:func:`relax.vdam.m_step.vdam_m_step_single_class`, which takes ``settings``)."""
        return vdam_m_step_single_class(state, k, accum_h0, accum_h1, **settings)

    def iteration_references(self, state: InitialModelState, *, iteration: int, meta: dict) -> InitialModelState:
        """The model after iteration ``iteration``'s M-step and solvent flattening (the native source: ``state``).

        A replaying source returns a copy with other references and records what it read in ``meta``.
        """
        return state
