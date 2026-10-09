"""The ports through which comparison enters, and observation leaves, an InitialModel run (code rule 15).

A :class:`VdamInputSource` supplies what a comparison run takes from elsewhere instead of what the run
computes, at the call sites the driver assigns: the start-up references, the M-step of each class, and the
references after each iteration's M-step. This base class is the native source. The command chooses the source once
(``relax.parity.vdam_replay``); the algorithm never imports an implementation.

A :class:`VdamObserver` watches the run at named moments and never steers it: dumps, captures and timings.
This base class observes nothing; the command chooses the observer once
(``relax.diagnostics.vdam_observers.vdam_command_observer``).
"""

from __future__ import annotations

from relax.vdam.m_step import vdam_m_step_single_class
from relax.vdam.state import InitialModelState, VdamAccumulator


class VdamInputSource:
    """The native source: relax's own bootstrap and M-step, and the references it made."""

    def startup_references(self, *, n_classes: int, box_size: int):
        """References ``[K, N, N, N]`` (RELION layout, float64) that replace the start-up bootstrap, or None (the
        native source: the run bootstraps its own)."""
        return None

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


class VdamObserver:
    """Watches an InitialModel run and never steers it; every hook does nothing by default."""

    def expected_accuracy_estimated(self, inputs, accuracy) -> None:
        """An expected-accuracy estimate is made: ``inputs`` (``native_sampling.AccuracyEstimateInputs``) is what
        it read, ``accuracy`` (``helpers.expected_accuracy.ExpectedAccuracy``) what it found."""
