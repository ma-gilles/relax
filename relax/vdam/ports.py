"""The ports through which comparison enters, and observation leaves, an InitialModel run (code rule 15).

A :class:`VdamInputSource` supplies what a comparison run takes from elsewhere instead of what the run computes,
at the call sites the driver assigns: the start-up references, each class's M-step and the references after each
M-step. A :class:`VdamObserver` watches the run and never steers it: dumps, captures and timings. The base classes
are the native source and no observation; the command chooses both once (``relax.parity.vdam_replay``,
``relax.diagnostics.vdam_observers.vdam_command_observer``), and the algorithm never imports an implementation.
Only the command and the driver hold the observer; a step that calls a hook mid-step takes a narrow probe the
driver builds from it once per run (:class:`ExpectationProbe`, :class:`MaximizationProbe`), as refinement's steps do.
"""

from __future__ import annotations

from typing import Protocol

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


class NoStageProfile:
    """The timer of untimed stages: ``record(name)`` ends a stage, ``report(label)`` the profile."""

    def record(self, name: str) -> None: ...

    def report(self, label: str) -> None: ...


class NoIterationProfile:
    """The timer of an untimed iteration: ``stage(name)`` ends a stage, ``before_artifacts(meta)`` precedes the
    artifact writes, ``finish()`` follows them."""

    def stage(self, name: str) -> None: ...

    def before_artifacts(self, meta: dict) -> None: ...

    def finish(self) -> None: ...


class ExpectationProbe(Protocol):
    """The hook the E-step calls mid-step with the expected-accuracy estimate's inputs
    (``native_sampling.AccuracyEstimateInputs``) and result. Steps receive this, never the observer."""

    def expected_accuracy_estimated(self, inputs, accuracy) -> None: ...


class MaximizationProbe(Protocol):
    """The hooks the VDAM model update calls mid-step: the noise update made ``updated`` from ``previous`` with the
    E-step's sums (not called when there was nothing to update), or the sums are not finite and the run stops
    (returns a dump file for the error message, or None). Steps receive this, never the observer."""

    def noise_updated(self, previous: InitialModelState, updated: InitialModelState, sums) -> None: ...

    def noise_sums_nonfinite(self, state: InitialModelState, meta: dict, summaries) -> str | None: ...


class NoProbe:
    """The no-op :class:`ExpectationProbe` and :class:`MaximizationProbe`."""

    def expected_accuracy_estimated(self, inputs, accuracy) -> None: ...

    def noise_updated(self, previous, updated, sums) -> None: ...

    def noise_sums_nonfinite(self, state, meta, summaries) -> str | None:
        return None


class VdamObserver:
    """Watches an InitialModel run and never steers it; observes nothing by default. ``stage_profile()`` times one
    part of the run (start-up, driver, one iteration's artifacts) and ``iteration_profile(it)`` an iteration's
    stages, from now; the driver builds the steps' probes once per run."""

    def stage_profile(self) -> NoStageProfile:
        return NoStageProfile()

    def iteration_profile(self, iteration: int) -> NoIterationProfile:
        return NoIterationProfile()

    def expectation_probe(self) -> ExpectationProbe:
        return NoProbe()

    def maximization_probe(self) -> MaximizationProbe:
        return NoProbe()
