"""Final all-data pass admission."""

from relax.helpers.env_flags import parse_env_flag_or_false

_FINAL_ALL_DATA_AFTER_MAX_ITER_ENV = "RELAX_FINAL_ALL_DATA_AFTER_MAX_ITER"


def _should_run_final_all_data_iteration(
    *,
    logger,
    has_converged: bool,
    iteration: int,
    max_iter: int,
    force_max_iter_after_convergence: bool,
    k_class_enabled: bool = False,
) -> bool:
    """Return whether to run RELION's final all-data reconstruction pass."""

    if force_max_iter_after_convergence:
        return False
    if bool(has_converged):
        return True
    if not (
        parse_env_flag_or_false(_FINAL_ALL_DATA_AFTER_MAX_ITER_ENV, logger=logger)
        and int(iteration) >= int(max_iter)
    ):
        return False
    if bool(k_class_enabled):
        logger.warning(
            "Ignoring %s=1 for K-class after max_iter exhaustion; final all-data "
            "is only valid for K-class after convergence",
            _FINAL_ALL_DATA_AFTER_MAX_ITER_ENV,
        )
        return False
    return True
