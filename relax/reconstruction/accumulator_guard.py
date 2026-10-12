"""The finite guard of the native half accumulators, before any cross-half join.

A check, not an observer: in ``raise`` mode it stops the run on the first non-finite accumulator (rule 12), so
the controller calls it with the mode its options resolved (``RefinementEnvironment.half_accumulator_guard``,
from ``RELAX_EM_BPREF_FINITE_GUARD``).
"""

from __future__ import annotations

from relax.numerics import finite_check

HALF_ACCUMULATOR_GUARD_ENV = "RELAX_EM_BPREF_FINITE_GUARD"


def check_half_accumulators_before_join(
    numerators, denominators, *, guard_mode: str, iteration, init_relion_iteration, log
) -> None:
    """The finite guard of the native half accumulators, before any cross-half join.

    ``guard_mode`` (``RELAX_EM_BPREF_FINITE_GUARD``, ``RefinementEnvironment.half_accumulator_guard``) is off, warn or raise;
    the mode is logged at the first iteration. Run before the join, so a report names the half that is
    actually damaged rather than the one the join copied it into.
    """
    if iteration == 0:
        log.info(
            "BPref accumulator guard (%s): mode=%s",
            HALF_ACCUMULATOR_GUARD_ENV,
            guard_mode,
        )
    if guard_mode != "off":
        finite_check.check_half_accumulators(
            {
                "Ft_y_0": numerators[0],
                "Ft_y_1": numerators[1],
                "Ft_ctf_0": denominators[0],
                "Ft_ctf_1": denominators[1],
            },
            mode=guard_mode,
            context=finite_check.describe_context(
                iteration=iteration,
                relion_iteration=int(init_relion_iteration) + int(iteration) + 1,
            ),
        )
