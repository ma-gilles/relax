"""Which engine each K=1 E-step pass of the current iteration ran, for the run's results.

The resident drivers are relax's one pass-2 engine; the pass-1 local parent probe still runs
on the deprecated exact-local engine with a logged reason. The log alone makes that easy to
miss on a real dataset, so the routing also appends one entry per pass here, and the
iteration loop moves the entries into the per-iteration ``pass2_engine_trajectory`` of the
results. Entries read ``"<pass>:<engine>"`` or ``"<pass>:<engine> (<reason>)"``.

Every engine other than the resident one is deprecated (docs/development/em_status.md,
"One engine: removal TODO"). :func:`warn_deprecated_engine` logs one warning per engine,
pass kind and reason the first time a run routes a pass to one of them.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_entries: list[str] = []
_coarse_entries: list[dict] = []

# Record labels of the deprecated engines, and what they are.
_DEPRECATED_ENGINES = {
    "exact_local": "the exact local engine",
    "local": "the VDAM exact-local E-step (exact local engine)",
}

_warned: set[tuple[str, str, str]] = set()


def warn_deprecated_engine(engine: str, pass_kind: str, reason: str) -> None:
    """Log once per engine, pass kind and reason that a pass ran on a deprecated engine.

    Numbers in the reason (sizes, budgets) do not make a new warning, so a refusal that
    repeats every iteration with other byte counts is logged once.
    """

    key = (engine, pass_kind, re.sub(r"\d+(\.\d+)?", "#", reason))
    if key in _warned:
        return
    _warned.add(key)
    logger.warning(
        "DEPRECATED engine: a %s pass runs on %s because %s. relax keeps one pass-2 engine, the "
        "resident one; this engine is removed once the resident engine covers this case "
        "(docs/development/em_status.md, 'One engine: removal TODO').",
        pass_kind,
        _DEPRECATED_ENGINES.get(engine, engine),
        reason,
    )


def record_pass_engine(pass_kind: str, engine: str, reason: str | None = None) -> None:
    """Append the engine one pass ran on (``pass_kind`` is ``global`` or ``local``).

    A non-resident engine with a reason also logs the deprecation warning; a route that
    records no reason warns at its call site.
    """

    if engine != "resident" and reason:
        warn_deprecated_engine(engine, pass_kind, reason)
    _entries.append(f"{pass_kind}:{engine}" if not reason else f"{pass_kind}:{engine} ({reason})")


def take_pass_engines() -> list[str]:
    """Return the entries recorded since the last call and clear them."""

    entries = list(_entries)
    _entries.clear()
    return entries


def record_coarse_engine_call(
    *,
    requested: str,
    resolved: str,
    strategy: str,
    fine_engine: str | None,
    images: int,
    classes: int,
    rotations: int,
    translations: int,
    coarse_candidates_per_image: int,
    fine_candidates_per_image: int,
    evaluated_fine_candidates_total: int,
    selected_fine_candidates_total: int | None,
    current_size: int,
    reconstruction_current_size: int,
    score_mode: str,
    posterior_policy: str,
    pruned: bool,
    precision: dict[str, str],
) -> None:
    """Record one executed global E/M call, with its actual candidate counts.

    Call after the engine returns successfully. A parsed CLI option, JIT trace
    or planned pass is not execution evidence. ``take_coarse_engine_calls``
    transfers the records at the existing iteration/finalization boundaries.
    """
    if requested not in {"auto", "gemm_hybrid", "gemm_dense"}:
        raise ValueError(f"unknown requested coarse engine {requested!r}")
    if strategy not in {"hybrid", "dense"}:
        raise ValueError(f"unknown coarse strategy {strategy!r}")
    counts = {
        "images": images,
        "classes": classes,
        "rotations": rotations,
        "translations": translations,
        "coarse_candidates_per_image": coarse_candidates_per_image,
        "fine_candidates_per_image": fine_candidates_per_image,
    }
    if any(int(value) <= 0 for value in counts.values()):
        raise ValueError("an executed global pass must have positive image and candidate counts")
    if int(rotations) * int(translations) * int(classes) != (
        int(coarse_candidates_per_image) if strategy == "hybrid" else int(fine_candidates_per_image)
    ):
        raise ValueError("scored grid dimensions disagree with the selected strategy")
    total = int(images) * int(fine_candidates_per_image)
    evaluated = int(evaluated_fine_candidates_total)
    selected = None if selected_fine_candidates_total is None else int(selected_fine_candidates_total)
    if evaluated < 0 or evaluated > total or (selected is not None and (selected < 0 or selected > evaluated)):
        raise ValueError("selected fine candidate count must fit the complete expanded grid")
    if posterior_policy not in {"gaussian", "cc_winner"} or (
        (score_mode == "gaussian") != (posterior_policy == "gaussian")
    ):
        raise ValueError("score mode and posterior policy disagree")
    if strategy == "dense" and (evaluated != total or selected != (int(images) if posterior_policy == "cc_winner" else total)):
        raise ValueError("dense GEMM must score every expanded-grid candidate and apply its stated posterior")
    if strategy == "dense" and fine_engine is not None:
        raise ValueError("dense GEMM has no separate fine engine")
    if strategy == "hybrid" and not fine_engine:
        raise ValueError("hybrid requires the executed fine engine name")
    if strategy == "dense" and pruned:
        raise ValueError("dense GEMM cannot report adaptive support pruning")
    if set(precision) != {"score", "projection", "mstep"} or any(
        not isinstance(value, str) or not value for value in precision.values()
    ):
        raise ValueError("precision must name score, projection and M-step dtypes")
    _coarse_entries.append(
        {
            "requested": requested,
            "resolved": str(resolved),
            "strategy": strategy,
            "fine_engine": fine_engine,
            "images": int(images),
            "classes": int(classes),
            "rotations": int(rotations),
            "translations": int(translations),
            "hypotheses_per_image": int(classes) * int(rotations) * int(translations),
            "coarse_candidates_per_image": int(coarse_candidates_per_image),
            "fine_candidates_per_image": int(fine_candidates_per_image),
            "evaluated_fine_candidates_total": evaluated,
            "selected_fine_candidates_total": selected,
            "selected_fine_candidates_known": selected is not None,
            "current_size": int(current_size),
            "reconstruction_current_size": int(reconstruction_current_size),
            "score_mode": str(score_mode),
            "posterior_policy": posterior_policy,
            "pruned": bool(pruned),
            "precision": dict(precision),
        }
    )


def take_coarse_engine_calls() -> list[dict]:
    """Return and clear executed global-call records since the last boundary."""
    entries = list(_coarse_entries)
    _coarse_entries.clear()
    return entries
