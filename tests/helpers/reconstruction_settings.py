"""The builders of ``ReconstructionSettings`` and of the numbered iteration a reconstruction reads, for unit tests."""

from relax.refinement.iteration_planning import ClassSeeding, FirstIterationPolicy, NumberedIteration
from relax.refinement.numbered_reconstruction import ReconstructionSettings


def reconstruction_settings(**fields) -> ReconstructionSettings:
    """``ReconstructionSettings`` with RELION's values for the consistency and solvent fields a test leaves out.

    The production builder (``setup_checks.reconstruction_settings_for_run``) passes every field from the
    run's options; these are the options' defaults: the radial gridding window, RELION's shell-pair counting
    and no user solvent mask.
    """
    return ReconstructionSettings(
        **{
            "gridding_kernel": "radial",
            "shell_pair_counting": "relion",
            "solvent_mask": None,
            "solvent_correct_fsc": False,
            "solvent_phase_stream": None,
            **fields,
        }
    )


def numbered_iteration(*, iteration: int, current_size: int, relion_firstiter_cc: bool = False) -> NumberedIteration:
    """The ``NumberedIteration`` of a run started at RELION iteration 0 (no Class3D seeding): its 0-based
    ``iteration``, its ``current_size`` and whether it is the first-iteration CC (``first_iteration_policy``)."""
    return NumberedIteration(
        iteration=iteration,
        numbered_relion_iteration=iteration + 1,
        has_previous_iteration=iteration > 0,
        first_iteration=FirstIterationPolicy(
            relion_firstiter_cc, "normalized_cc" if relion_firstiter_cc else "gaussian", relion_firstiter_cc,
        ),
        seeding=ClassSeeding(single_class_iteration=False, seed_classes=None),
        current_size=current_size,
    )
