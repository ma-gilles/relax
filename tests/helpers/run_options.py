"""The one factory of refinement options for tests.

The option records' defaults are the command line's (``relax refine``, PLAN d1). ``cli`` builds records with
those defaults. ``stand_in`` builds them with the values the CPU stand-in tests are written against, named in
``STAND_IN``: the command line's backend and first-iteration CC need a GPU, and the stand-in engines' small
fixtures were sized for the rest. A field a test passes wins over either preset.

    from helpers.run_options import stand_in

    options = stand_in.options(schedule=stand_in.schedule(max_iter=1, init_current_size=4))
"""

from __future__ import annotations

from dataclasses import dataclass, field

from relax.refinement.refinement_options import (
    AdaptiveOptions,
    ExecutionOptions,
    RefinementOptions,
    RefinementSchedule,
    RelionParityOptions,
)

# What the CPU stand-in tests run instead of the command line's defaults (see the module docstring).
STAND_IN = {
    "schedule": {
        "max_iter": 10,  # the CLI's 999 runs to convergence
        "init_current_size": 32,  # computed from the data by the command
        "init_translation_range": 10.0,  # the CLI's --offset_range is 5.0
    },
    "adaptive": {
        "adaptive_oversampling": 0,  # the CLI's 1
        "max_significants": 500,  # the CLI's -1 (uncapped)
    },
    "parity": {
        "perturb_factor": 0.0,  # the CLI's 0.5
        "emulate_relion_firstiter_cc": False,  # the CLI's --firstiter_cc (a GPU scorer)
        "firstiter_cc_tree_rescore_max_margin": None,  # the CLI's K=1 --firstiter_cc margin
        "image_fourier_backend": "host_numpy",  # the CLI's relion_cuda (the RELION CUDA library)
    },
    "execution": {
        "rotation_block_size": 5000,  # the CLI's 40000
    },
}


@dataclass(frozen=True)
class OptionsFactory:
    """Option records with ``preset``'s values under the fields a caller passes."""

    preset: dict = field(default_factory=dict)

    def _fields(self, group: str, fields: dict) -> dict:
        return {**self.preset.get(group, {}), **fields}

    def schedule(self, **fields) -> RefinementSchedule:
        return RefinementSchedule(**self._fields("schedule", fields))

    def adaptive(self, **fields) -> AdaptiveOptions:
        return AdaptiveOptions(**self._fields("adaptive", fields))

    def parity(self, **fields) -> RelionParityOptions:
        return RelionParityOptions(**self._fields("parity", fields))

    def execution(self, **fields) -> ExecutionOptions:
        return ExecutionOptions(**self._fields("execution", fields))

    def options(self, **groups) -> RefinementOptions:
        """``RefinementOptions`` with every preset group the caller did not pass built from the preset."""
        for group in self.preset:
            if group not in groups:
                groups[group] = getattr(self, group)()
        return RefinementOptions(**groups)


cli = OptionsFactory()
stand_in = OptionsFactory(STAND_IN)
