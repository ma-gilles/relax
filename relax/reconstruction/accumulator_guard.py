"""The finite guard of the native half accumulators, before any cross-half join.

A check, not an observer: in ``raise`` mode it stops the run on the first non-finite accumulator (rule 12), so
the controller calls it with the mode its options resolved (``RefinementEnvironment.half_accumulator_guard``,
from ``RELAX_EM_BPREF_FINITE_GUARD``).
"""

from __future__ import annotations

from relax.numerics import finite_check

HALF_ACCUMULATOR_GUARD_ENV = "RELAX_EM_BPREF_FINITE_GUARD"
_GUARD_OFF_VALUES = frozenset({"0", "false", "no", "off"})
_GUARD_RAISE_VALUES = frozenset({"", "1", "true", "yes", "on", "raise"})


def half_accumulator_guard_mode(raw: str | None, *, warn_only: bool) -> str:
    """How the once-per-iteration accumulator guard behaves: raise, warn or off.

    ``raw`` is the text of ``RELAX_EM_BPREF_FINITE_GUARD`` (None when unset) and ``warn_only`` the flag
    ``RELAX_EM_FINITE_CHECK_WARN`` (``env_flags.parse_env_true_flag``, as ``EngineTuning.finite_check`` reads it), both read once
    by ``RefinementEnvironment.from_environ``; ``finite_check.check_half_accumulators`` runs in this mode.

    The guard is on by default in ``raise`` mode. Job 14193291 (P4-D,
    2026-09-20) measured its cost on an interleaved off/on pair of twelve
    one-iteration arms at current size 74 on one H100: in the eight arms that
    ran without a co-located cold compile the medians were 218.0 s off and
    218.3 s on, inside the 217.4 to 219.1 s same-source spread, so production
    pays nothing measurable for stopping where the damage is instead of four
    stages later.

    ``RELAX_EM_BPREF_FINITE_GUARD=0`` (also ``off``, ``false``, ``no``) opts
    out; ``warn`` logs the offending accumulators and continues;
    ``RELAX_EM_FINITE_CHECK_WARN=1`` demotes it to ``warn`` as well, as it
    does the per-bucket checks. Any other value is a configuration error and
    raises, rather than silently running with a guard the caller did not ask
    for.
    """

    raw = (raw or "").strip().lower()
    if raw in _GUARD_OFF_VALUES:
        return "off"
    if raw == "warn":
        return "warn"
    if raw not in _GUARD_RAISE_VALUES:
        raise ValueError(
            f"{HALF_ACCUMULATOR_GUARD_ENV}={raw!r}: expected one of 1/on/raise "
            "(the default), warn, or 0/off"
        )
    return "warn" if warn_only else "raise"


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
