"""RELION's references and RELION's own M-step as the input source of an InitialModel run (code rule 15).

``RELAX_INITIALMODEL_IREF_REPLAY_TEMPLATE`` replaces the references after each iteration's M-step with RELION's
maps of that iteration (causal trajectory boundaries). The M-step oracle
(``python -m relax.diagnostics.vdam_native_mstep``) replaces relax's single-class M-step with RELION's
step-by-step primitives, which read the native replays (``NATIVE_MSTEP_REPLAY_ENVS``) themselves. The command
reads the environment once and builds the source here (:func:`vdam_input_source`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from relax.vdam.m_step import vdam_m_step_single_class
from relax.vdam.ports import VdamInputSource
from relax.vdam.state import InitialModelState

INITIAL_MODEL_IREF_REPLAY_TEMPLATE_ENV = "RELAX_INITIALMODEL_IREF_REPLAY_TEMPLATE"
# RELION's start-up references in place of the bootstrap (VdamReplaySource.startup_references).
INITIAL_IREF_OVERRIDE_ENV = "RELAX_INITIAL_IREF_OVERRIDE"

# The native replays of RELION's M-step intermediates (relax.parity.vdam_mstep_replay); only the oracle
# reads them.
NATIVE_MSTEP_REPLAY_ENVS = (
    "RELAX_VDAM_NATIVE_SECOND_MOMENT_REPLAY_BIN",
    "RELAX_VDAM_NATIVE_FIRST_MOMENT_REPLAY_BIN",
    "RELAX_VDAM_NATIVE_BPREF_DATA_REPLAY_BIN",
    "RELAX_VDAM_NATIVE_BPREF_WEIGHT_REPLAY_BIN",
    "RELAX_VDAM_NATIVE_IREF_INPUT_REPLAY_BIN",
)


@dataclass(frozen=True)
class VdamReplaySource(VdamInputSource):
    """``reference_template`` (or None): RELION's references after each M-step. It may contain ``{iteration}``
    and ``{k}`` (RELION's one-based class number); a comma-separated list gives one path per class; a single path
    without ``{k}`` serves K=1 only. ``m_step``: the single-class M-step (relax's, or the oracle's)."""

    reference_template: str | None = None
    m_step: Callable[..., InitialModelState] = vdam_m_step_single_class
    startup_reference_paths: str | None = None

    def startup_references(self, *, n_classes: int, box_size: int):
        """RELION's start-up references (``RELAX_INITIAL_IREF_OVERRIDE``) in place of the bootstrap: one path for
        every class, one per class (comma-separated), or a template with ``{k}`` (the one-based class)."""
        if self.startup_reference_paths is None:
            return None
        from recovar.utils.helpers import load_relion_volume

        K = n_classes
        paths = [p.strip() for p in self.startup_reference_paths.split(",") if p.strip()]
        if len(paths) == 1 and "{k" in paths[0]:
            paths = [paths[0].format(k=k + 1) for k in range(K)]
        if len(paths) not in (1, K):
            raise ValueError(f"{INITIAL_IREF_OVERRIDE_ENV} expects 1 or K={K} paths, got {len(paths)}")
        vols = np.stack(
            [np.asarray(load_relion_volume(p), dtype=np.float64) for p in paths],
            axis=0,
        )
        if vols.shape[1:] != (box_size, box_size, box_size):
            raise ValueError(f"{INITIAL_IREF_OVERRIDE_ENV} volume shape {vols.shape[1:]} != {(box_size,) * 3}")
        return np.broadcast_to(vols, (K, box_size, box_size, box_size)).copy() if len(paths) == 1 else vols

    def single_class_m_step(self, state, k, accum_h0, accum_h1, **settings) -> InitialModelState:
        return self.m_step(state, k, accum_h0, accum_h1, **settings)

    def iteration_references(self, state: InitialModelState, *, iteration: int, meta: dict) -> InitialModelState:
        if self.reference_template is None:
            return state
        tokens = [token.strip() for token in self.reference_template.split(",") if token.strip()]
        if len(tokens) == 1 and "{k" in tokens[0]:
            paths = [
                tokens[0].format(iteration=int(iteration), k=class_index + 1)
                for class_index in range(int(state.K))
            ]
        elif len(tokens) == 1 and int(state.K) == 1:
            paths = [tokens[0].format(iteration=int(iteration), k=1)]
        elif len(tokens) == int(state.K):
            paths = [
                token.format(iteration=int(iteration), k=class_index + 1)
                for class_index, token in enumerate(tokens)
            ]
        else:
            raise ValueError(
                f"{INITIAL_MODEL_IREF_REPLAY_TEMPLATE_ENV} expects one path for K=1 "
                f"or K={int(state.K)} comma-separated paths, got {len(tokens)}"
            )

        from recovar.utils.helpers import load_relion_volume

        references = np.stack(
            [np.asarray(load_relion_volume(path), dtype=np.float64) for path in paths],
            axis=0,
        )
        expected_shape = (int(state.K), int(state.box_size), int(state.box_size), int(state.box_size))
        if references.shape != expected_shape:
            raise ValueError(
                f"iteration reference replay shape {references.shape} != {expected_shape}"
            )
        if not np.all(np.isfinite(references)):
            raise ValueError("iteration reference replay contains non-finite values")

        out = replace(state)
        out.Iref = references
        meta["diagnostic_iref_replay_paths"] = paths
        meta["diagnostic_iref_replay_iteration"] = int(iteration)
        return out


def vdam_input_source(
    *,
    reference_template: str,
    startup_references: str,
    native_mstep_replays: Sequence[str],
    mstep_compute_dtype: str,
    oracle_m_step: Callable[..., InitialModelState] | None = None,
) -> VdamInputSource:
    """The run's source, from the values of ``RELAX_INITIALMODEL_IREF_REPLAY_TEMPLATE`` and
    ``RELAX_INITIAL_IREF_OVERRIDE`` (``""``: unset) and the names of the set ``NATIVE_MSTEP_REPLAY_ENVS``, which only
    ``oracle_m_step`` (RELION's M-step) takes."""
    if native_mstep_replays and oracle_m_step is None:
        raise ValueError(
            f"{list(native_mstep_replays)} replay RELION's M-step; run python -m relax.diagnostics.vdam_native_mstep instead"
        )
    if reference_template and mstep_compute_dtype == "float32":
        raise ValueError("float32 M-step is incompatible with iteration reference replay")
    if not reference_template and not startup_references and oracle_m_step is None:
        return VdamInputSource()
    return VdamReplaySource(
        reference_template=reference_template or None,
        m_step=vdam_m_step_single_class if oracle_m_step is None else oracle_m_step,
        startup_reference_paths=startup_references or None,
    )
