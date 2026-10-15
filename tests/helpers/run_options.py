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

from relax.reconstruction import regularization
from relax.reconstruction.volume_solver import ReconstructionPrograms
from relax.refinement.refinement_options import (
    AdaptiveOptions,
    ExecutionOptions,
    LocalAdaptivePass2Support,
    RefinementOptions,
    RefinementSchedule,
    RelionParityOptions,
    ScoringVariants,
)
from relax.steps import scoring_policy

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

    def variants(self, **fields) -> ScoringVariants:
        """``ScoringVariants`` as ``from_environ`` builds them with none of its variables set, under ``fields``.

        K=1's x-half M-step default is the device's (``scoring_policy.k1_relion_x_half_mstep_default_available``:
        on where custom CUDA runs on a GPU), as the command's; the reconstruction runs its programs on one host FFT
        thread (``host_fft_workers`` changes no value).
        """
        defaults = dict(
            k1_relion_x_half_mstep=scoring_policy.k1_relion_x_half_mstep_default_available(),
            k_class_relion_x_half_mstep=True,
            local_adaptive_pass2=LocalAdaptivePass2Support(full_parent=False, rotation_only=False, denominator_mode=None),
            approx_acc_rot_for_convergence=False,
            reconstruction=reconstruction_programs(),
        )
        return ScoringVariants(**{**defaults, **fields})

    def options(self, **groups) -> RefinementOptions:
        """``RefinementOptions`` with every preset group the caller did not pass built from the preset, the
        unset environment's scoring variants (:meth:`variants`) and the process's dense precision."""
        for group in self.preset:
            if group not in groups:
                groups[group] = getattr(self, group)()
        groups.setdefault("variants", self.variants())
        groups.setdefault("precision", scoring_policy.DENSE_PRECISION)
        return RefinementOptions(**groups)


def reconstruction_programs(**fields) -> ReconstructionPrograms:
    """``ReconstructionPrograms`` as ``from_environ`` builds them with none of its variables set (one host FFT
    thread), under ``fields``."""
    defaults = dict(
        stable_windows=True,
        host_irfft=None,
        host_fft_workers=1,
        fsc_packed_stream_min_elements=regularization.RELION_FSC_PACKED_STREAM_MIN_ELEMENTS,
        lowres_join=lowres_join(),
        fsc_dump=None,
    )
    return ReconstructionPrograms(**{**defaults, **fields})


def lowres_join(**fields) -> regularization.LowResolutionJoinRoute:
    """The low-resolution join route ``ReconstructionPrograms.from_environ`` reads with none of its variables set
    (by size, above the default threshold), under ``fields``."""
    defaults = dict(forced_host=None, host_min_elements=regularization.LOW_RESOLUTION_JOIN_HOST_FALLBACK_MIN_ELEMENTS)
    return regularization.LowResolutionJoinRoute(**{**defaults, **fields})


cli = OptionsFactory()
stand_in = OptionsFactory(STAND_IN)
