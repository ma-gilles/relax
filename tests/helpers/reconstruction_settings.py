"""One builder of ``ReconstructionSettings`` for unit tests."""

from relax.refinement.mean_helpers import ReconstructionSettings


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
            "solvent_fsc_seed": 0,
            **fields,
        }
    )
