#!/usr/bin/env python
"""Fingerprint what the refinement controller does, to check that a refactor moved code and nothing else.

    pixi run fingerprint run OUT.json [--rev REV | --source DIR] [CASE ...]
    pixi run fingerprint diff A.json B.json          # exit 1 unless only log rows differ
    pixi run fingerprint check BASE_REV [HEAD_REV]   # run both (HEAD defaults to the worktree) and diff
    pixi run fingerprint cases                       # the case list, one line each
    pixi run fingerprint selftest [MUTATION ...]     # perturb the controller; every mutation must show

``run`` calls the real ``refine_single_volume`` of one source tree on small CPU cases (8-cubed volume,
four images) and records, per case: the result mapping, every file the run wrote, every checkpoint
snapshot, and one ordered trace of log records and selected calls. The ``main_*`` cases run the command
entry (``full_refinement.main``, ``relax refine`` / ``relax class3d``) on a written 12-image, 16-pixel data
directory with the device check stood in for, and record the operands main hands ``refine_single_volume``,
its result and every file main writes. Arrays are recorded as module,
dtype, shape and SHA-256 of their bytes, so a changed value, dtype, host/device placement, write or
log line is a difference. The source tree is a worktree (``--source``, default this checkout with
its uncommitted edits) or a ``git archive`` export of any commit (``--rev``); nothing is written into
it. Scratch goes under ``--work-dir`` (default: a new directory under the system temp directory).

The E-step engine cannot run on a CPU, so ``run_dense_k_class_em_adaptive`` is replaced on both sides by
one stand-in whose output is seeded by a hash of every operand it receives: a changed operand anywhere
upstream changes every later array of the run. The comparison is exact because both sides run the same
arithmetic on the same CPU; it is a check for move-only commits, not a merge gate for numerical changes
(``tests/CLAUDE.md``: no bitwise float asserts).

``diff`` and ``check`` count differences in five classes: outputs (cases, status, results, files and
checkpoints), added controller inputs, retired controller inputs, non-log trace rows (selected calls), and log
trace rows. Log order is not behaviour (code rule 2 in ``docs/development/refactor_rules.md``), so when only
log rows differ they print "only log rows differ (N); accepted under rule 2" and exit 0. A controller input
present only in B (a leaf under ``refine<N>/inputs``: a new option field the command hands the controller) is
listed and accepted: a value that changed what the controller does would change its results, which are
compared. So is one present only in A whose value there was ``None``, ``False`` or the field's declared default
(recorded with a ``=default`` mark; an option field that was off in the case and is retired, its behaviour now
chosen through a port: code rule 15), and the length of the
container that holds only such added or retired members; a removed input that was on, or a changed one, is an
output difference. Any other difference exits 1.

NOT covered (use the GPU test tiers): the real E-step engines and their numbers; local search and the
profile-only return; symmetry other than C1; tomography; multi-shape optics halves; follower-scale
emulation; captured RELION projectors; a sealed sampling state beyond one global iteration; GPU
operation order, peak memory and array lifetimes.
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 1
NOT_COVERED = (
    "in full_refinement.main: inputs from RELION run directories (--relion_init_dir, --relion_half_sets, the "
    "STAR replays, --final-replay-relion-dir), frozen boundaries, state-swap probes, captured projectors, "
    "follower-scale topologies, noise or poses from an earlier archive, several optics groups, subtomograms",
    "the real E-step engines and their numbers (a stand-in seeded by its operands replaces the dense adaptive engine)",
    "local search, local sampling and the profile-only return",
    "symmetry other than C1",
    "tomography",
    "multi-shape optics halves",
    "follower-scale emulation, captured RELION projectors; a sealed sampling state beyond one global iteration",
    "GPU operation order, peak memory and array lifetimes",
)
# A log template containing one of these formats a wall time: its template is kept, its text dropped.
TIMING_WORDS = ("wall", "elapsed", "%.2fs", "%.1fs", "%.3fs", "seconds", " in %")
# Rows whose text depends on the node (free host or device memory), not on the source.
NODE_DEPENDENT_TEMPLATES = (
    "timing", "batch planner memory inputs", "RELION EM batch sizing", "Persistent JAX compilation cache",
)
DROPPED_RESULT_KEYS = frozenset({"wall_times", "setup_phase_seconds"})
SECTIONS = ("status", "result", "files", "checkpoints")
# The classes a difference is counted in; only a difference confined to "log" is accepted.
DIFFERENCE_CLASSES = ("outputs", "added", "retired", "trace", "log")
# A controller input B no longer has is accepted only at one of these off values in A, or at its declared
# default (a leaf ending in DEFAULT_MARK): an option field that was off in the case is retired (its non-default
# behaviour moved behind a port, code rule 15).
OFF_VALUES = frozenset({"None", "False"})
DEFAULT_MARK = "=default"
# Leaves of what full_refinement.main hands the controller (fingerprints of the main_* cases).
CONTROLLER_INPUT = re.compile(r"^refine\d+/inputs\[")
TMP_TOKEN = "<TMP>"


# ----------------------------------------------------------------------------- pure logic


def _array_leaf(value) -> str | None:
    """``module dtype shape sha256`` of an array-like, or None for an object array."""
    import numpy as np

    host = np.asarray(value)
    if host.dtype == object:
        return None
    digest = hashlib.sha256(np.ascontiguousarray(host).tobytes()).hexdigest()
    return f"{type(value).__module__.split('.')[0]} {host.dtype} {tuple(host.shape)} sha256:{digest}"


def _is_field_default(field, value) -> bool:
    """Whether a dataclass field holds its declared plain default (None, a bool, number, string or empty tuple)."""
    if field.default is not dataclasses.MISSING:
        default = field.default
    else:
        return False
    if default is None or isinstance(default, (bool, int, float, str)) or default == ():
        return type(value) is type(default) and value == default
    return False


def flatten(value, path: str = "", out: dict | None = None, *, scrub=lambda text: text) -> dict:
    """``{path: leaf}`` for a nested result: scalars by ``repr`` (floats by hex), arrays by hash.

    Sequences also record their type and length, so an empty list and a tuple that became a list differ.
    """
    out = {} if out is None else out
    if value is None or isinstance(value, (bool, int, bytes)):
        out[path] = repr(value)
    elif isinstance(value, str):
        out[path] = repr(scrub(value))
    elif isinstance(value, float):
        out[path] = f"float:{value.hex()}"
    elif isinstance(value, dict):
        out[path + "#"] = f"dict[{len(value)}]"
        for key, item in value.items():
            if str(key) not in DROPPED_RESULT_KEYS:
                flatten(item, f"{path}/{key}", out, scrub=scrub)
    elif isinstance(value, (list, tuple)):
        out[path + "#"] = f"{type(value).__name__}[{len(value)}]"
        for index, item in enumerate(value):
            flatten(item, f"{path}[{index}]", out, scrub=scrub)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            item = getattr(value, field.name)
            item_path = f"{path}<{type(value).__name__}>/{field.name}"
            if _is_field_default(field, item):
                # A field at its declared default: marked, so removing it can be told from removing a set value.
                leaves = flatten(item, item_path, {}, scrub=scrub)
                out.update({key: f"{leaf} {DEFAULT_MARK}" for key, leaf in leaves.items()})
            else:
                flatten(item, item_path, out, scrub=scrub)
    elif hasattr(value, "__dict__") and not hasattr(value, "shape"):
        flatten(vars(value), f"{path}<{type(value).__name__}>", out, scrub=scrub)
    else:
        leaf = _array_leaf(value)
        if leaf is not None:
            out[path] = leaf
        else:
            import numpy as np

            host = np.asarray(value)
            if host.ndim == 0:
                out[path] = f"object:{type(host.item()).__name__}"
            else:
                flatten(host.tolist(), path, out, scrub=scrub)
    return out


def result_fields(result):
    """The controller's result as its flat archive mapping.

    A ``RefinementResult`` is recorded through ``archive_fields()`` (the saved keys and values); a dict, which
    an older tree returns, is recorded as it is. Both trees of a check are then recorded the same way.
    """
    archive_fields = getattr(result, "archive_fields", None)
    return result if archive_fields is None else archive_fields()


def digest_operands(*values) -> int:
    """A seed from every array and scalar handed to a stand-in, so a changed operand changes its result."""
    import numpy as np

    h = hashlib.sha256()

    def feed(value, depth=0):
        if value is None or isinstance(value, (bool, int, float, str)):
            h.update(repr(value).encode())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                h.update(str(key).encode())
                feed(value[key], depth + 1)
        elif isinstance(value, (list, tuple)):
            h.update(f"<{len(value)}>".encode())
            for item in value:
                feed(item, depth + 1)
        elif hasattr(value, "shape") and hasattr(value, "dtype"):
            host = np.asarray(value)
            h.update(str(host.dtype).encode())
            h.update(str(host.shape).encode())
            h.update(np.ascontiguousarray(host).tobytes())
        elif callable(value):
            h.update(b"<callable>")
        elif dataclasses.is_dataclass(value) and not isinstance(value, type) and depth < 4:
            feed({f.name: getattr(value, f.name) for f in dataclasses.fields(value)}, depth + 1)
        else:
            h.update(type(value).__name__.encode())

    for value in values:
        feed(value)
    return int.from_bytes(h.digest()[:8], "little")


def log_row(logger_name: str, level: str, template: str, text: str, *, scrub=lambda text: text) -> list:
    """One trace row of a log record; wall times and node-dependent text are dropped, the template kept."""
    timed = any(word in template for word in TIMING_WORDS)
    node_dependent = any(word in template for word in NODE_DEPENDENT_TEMPLATES)
    return ["log", logger_name, level, template, "" if timed or node_dependent else scrub(text)]


def ordered_trace(rows: list) -> list:
    """Trace rows ``(thread_name, row)`` in a thread-independent order.

    The controller's own thread is the spine. Rows of worker threads (the overlapped halves) are grouped
    per thread, in thread-name order, at the point where the controller waited for them: which worker
    logged first is a race, what each worker did in order is not. Rows of any other thread (the run-file
    writer, which the command runs in the background) go after the spine, per thread.
    """
    keyed, main_rows = [], 0
    for sequence, (thread_name, row) in enumerate(rows):
        if thread_name == "MainThread":
            keyed.append(((main_rows, "", sequence), row))
            main_rows += 1
        elif thread_name.startswith("em-half-"):
            keyed.append(((main_rows - 1, thread_name, sequence), row))
        else:
            keyed.append(((float("inf"), thread_name, sequence), row))
    return [row for _, row in sorted(keyed, key=lambda item: item[0])]


def _changed_rows(rows_a: list[str], rows_b: list[str]) -> list[str]:
    """The removed (``-``) and added (``+``) rows of a line diff from ``rows_a`` to ``rows_b``."""
    return [
        line
        for line in difflib.unified_diff(rows_a, rows_b, lineterm="", n=0)
        if line[:1] in "+-" and line[:3] not in ("+++", "---")
    ]


def diff_fingerprints(a: dict, b: dict, *, shown_per_section: int = 12) -> tuple[dict[str, int], list[str]]:
    """``(differences per class of DIFFERENCE_CLASSES, report lines)`` between two fingerprints written by ``run``.

    ``outputs`` counts missing cases and changed status, result, file and checkpoint leaves; ``trace`` the
    removed or added trace rows that are not log records; ``log`` those that are.
    """
    lines = []
    counts = dict.fromkeys(DIFFERENCE_CLASSES, 0)
    cases_a, cases_b = a["cases"], b["cases"]
    for name in sorted(set(cases_a) ^ set(cases_b)):
        counts["outputs"] += 1
        lines.append(f"CASE {name}: only in {'A' if name in cases_a else 'B'}")
    trace_lines: dict[str, list[str]] = {}
    totals = dict.fromkeys((*SECTIONS, "trace"), 0)
    for name in sorted(set(cases_a) & set(cases_b)):
        case_a, case_b = cases_a[name], cases_b[name]
        for section in SECTIONS:
            flat_a, flat_b = case_a[section], case_b[section]
            totals[section] += len(flat_a)
            changed = [key for key in sorted(set(flat_a) | set(flat_b)) if flat_a.get(key) != flat_b.get(key)]
            added = [key for key in changed if key not in flat_a and CONTROLLER_INPUT.match(key)]
            counts["added"] += len(added)
            for key in added[:shown_per_section]:
                lines.append(f"ADDED {name} {section} {key}: {flat_b[key]}")
            retired = [
                key for key in changed
                if key not in flat_b and CONTROLLER_INPUT.match(key)
                and (flat_a[key] in OFF_VALUES or flat_a[key].endswith(DEFAULT_MARK))
            ]
            counts["retired"] += len(retired)
            for key in retired[:shown_per_section]:
                lines.append(f"RETIRED {name} {section} {key}: {flat_a[key]}")
            # A controller-input container whose length changed only through added or retired members.
            moved = set(added) | set(retired)
            resized = [
                key for key in changed
                if key.endswith("#") and CONTROLLER_INPUT.match(key) and key in flat_a and key in flat_b
                and any(member.startswith(key[:-1]) for member in moved)
                and all(member in moved for member in changed if member.startswith(key[:-1]) and member != key)
            ]
            changed = [key for key in changed if key not in moved and key not in resized]
            counts["outputs"] += len(changed)
            for key in changed[:shown_per_section]:
                lines.append(
                    f"DIFF {name} {section} {key}: {flat_a.get(key, '<absent>')} != {flat_b.get(key, '<absent>')}"
                )
            if len(changed) > shown_per_section:
                lines.append(f"DIFF {name} {section}: {len(changed) - shown_per_section} more")
        text_a = [" | ".join(row) for row in case_a["trace"]]
        text_b = [" | ".join(row) for row in case_b["trace"]]
        totals["trace"] += len(text_a)
        changed_rows = _changed_rows(text_a, text_b)
        for line in changed_rows:
            trace_lines.setdefault(line, []).append(name)
        # The non-log rows are compared on their own, so a log record that moved past a call is not counted
        # as a changed call; when they agree, every changed row of the full trace is a log row.
        non_log = len(
            _changed_rows(*([row for row in text if not row.startswith("log | ")] for text in (text_a, text_b)))
        )
        counts["trace"] += non_log
        counts["log"] += (
            len(_changed_rows(*([row for row in text if row.startswith("log | ")] for text in (text_a, text_b))))
            if non_log
            else len(changed_rows)
        )
    for line, names in trace_lines.items():
        lines.append(f"TRACE [{len(names)} cases, e.g. {names[0]}] {line[:400]}")
    lines.append(
        f"{len(set(cases_a) & set(cases_b))} cases compared; {sum(counts.values())} differences "
        f"(outputs {counts['outputs']}, added inputs {counts['added']}, retired inputs {counts['retired']}, "
        f"trace rows {counts['trace']}, log rows {counts['log']}); leaves compared: "
        + ", ".join(f"{section} {totals[section]}" for section in SECTIONS)
        + f"; trace rows {totals['trace']}"
    )
    if counts["log"] and not counts["outputs"] and not counts["trace"]:
        lines.append(f"only log rows differ ({counts['log']}); accepted under rule 2")
    if counts["added"] and not counts["outputs"] and not counts["trace"]:
        lines.append(f"controller inputs added ({counts['added']}); accepted: new option fields")
    if counts["retired"] and not counts["outputs"] and not counts["trace"]:
        lines.append(f"controller inputs retired ({counts['retired']}); accepted: option fields removed at their off value")
    return counts, lines


def accepted(counts: dict[str, int]) -> bool:
    """Whether a difference of ``counts`` passes: nothing but log rows differs."""
    return not counts["outputs"] and not counts["trace"]


def differing_cases(a: dict, b: dict) -> list[str]:
    """Names of the cases whose fingerprints differ in any section or trace row."""
    return sorted(
        name
        for name in set(a["cases"]) | set(b["cases"])
        if a["cases"].get(name) != b["cases"].get(name)
    )


def apply_mutation(text: str, old: str, new: str) -> tuple[str, int]:
    """``text`` with ``old`` replaced by ``new`` line for line, and the number of places replaced.

    ``old`` and ``new`` have the same number of lines. A place matches when each consecutive source
    line contains the corresponding ``old`` line; indentation is not part of the match, so a statement
    keeps matching after an extraction re-indents it.
    """
    old_lines = [line.strip() for line in old.strip().splitlines()]
    new_lines = [line.strip() for line in new.strip().splitlines()]
    if len(old_lines) != len(new_lines):
        raise ValueError("a mutation replaces line for line")
    lines = text.splitlines(keepends=True)
    index, replaced = 0, 0
    while index + len(old_lines) <= len(lines):
        if all(wanted in lines[index + offset] for offset, wanted in enumerate(old_lines)):
            for offset, (wanted, replacement) in enumerate(zip(old_lines, new_lines, strict=True)):
                lines[index + offset] = lines[index + offset].replace(wanted, replacement, 1)
            index += len(old_lines)
            replaced += 1
        else:
            index += 1
    return "".join(lines), replaced


# ----------------------------------------------------------------------------- cases


def _cases() -> dict[str, tuple[str, dict]]:
    """``{name: (description, run_case keywords)}``; every case runs numbered iterations of the real controller."""
    cases: dict[str, tuple[str, dict]] = {}

    def add(name, description, **keywords):
        assert name not in cases, name
        cases[name] = (description, keywords)

    for k in (1, 2, 4):
        for cc in (False, True):
            for join in (0.0, 40.0):
                add(
                    f"k{k}_{'cc' if cc else 'gauss'}_join{int(join)}",
                    f"K={k}, {'first-iteration CC' if cc else 'Gaussian'} start, low-resolution join {join:g} A, "
                    "two iterations, final pass, intermediates dumped",
                    n_classes=k, join=join, cc=cc,
                )
    add("k1_gauss_join_none", "K=1 with no low-resolution join option", n_classes=1, join=None)
    add("k1_gauss_join_1e6", "K=1 joining every shell", n_classes=1, join=1.0e6)
    add("k1_gauss_join40_nodump", "K=1 without the intermediates dump", n_classes=1, join=40.0, dump=False)
    add(
        "k1_join40_after_cap", "K=1 final pass after the iteration cap (RELAX_FINAL_ALL_DATA_AFTER_MAX_ITER)",
        n_classes=1, join=40.0, max_iter=2, converge_after=None, env={"RELAX_FINAL_ALL_DATA_AFTER_MAX_ITER": "1"},
    )
    add("k1_no_final", "K=1 stopping at the cap with no final pass", n_classes=1, join=40.0, max_iter=2, converge_after=None)
    add("k4_no_final", "K=4 stopping at the cap", n_classes=4, join=40.0, max_iter=2, converge_after=None)
    add("k1_os1_join40", "K=1 with adaptive oversampling 1", n_classes=1, join=40.0, oversampling=1)
    add("k4_os1_join40", "K=4 with adaptive oversampling 1", n_classes=4, join=40.0, oversampling=1)
    continued = dict(max_iter=2, converge_after=None, dump=False, continued=True)
    add("k1_continued", "K=1 resumed from the snapshot of a one-iteration run", n_classes=1, join=40.0, writer=True, **continued)
    add("k1_cc_continued", "K=1 resumed after a first-iteration CC run", n_classes=1, join=40.0, cc=True, **continued)
    add("k2_continued", "K=2 resumed from a snapshot", n_classes=2, join=0.0, writer=True, **continued)
    add("k1_continued_perturb", "K=1 resumed with sampling perturbation", n_classes=1, join=40.0, writer=True, perturb=0.5, **continued)
    add("k2_continued_perturb", "K=2 resumed with sampling perturbation", n_classes=2, join=0.0, perturb=0.5, **continued)
    add("k1_fresh_perturb", "K=1 fresh start with sampling perturbation", n_classes=1, join=40.0, max_iter=2, converge_after=None, dump=False, perturb=0.5)
    for k in (1, 2, 4):
        common = dict(n_classes=k, join=0.0)
        add(f"k{k}_init_prior", f"K={k} direction prior installed at start-up, checkpoints written", init_prior=2, writer=True, **common)
        add(f"k{k}_init_prior_other_order", f"K={k} start-up prior at another HEALPix order", init_prior=1, dump=False, writer=True, **common)
        add(f"k{k}_init_prior_one_half", f"K={k} start-up prior for half 2 only", init_prior=(None, 2), dump=False, writer=True, **common)
        add(f"k{k}_iter_prior", f"K={k} per-iteration replay priors; the final pass reuses the last slot", iter_prior=[2, 2, 1], dump=False, writer=True, **common)
        add(f"k{k}_iter_prior_final_same_order", f"K={k} per-iteration replay priors, one half missing in the last slot", iter_prior=[None, 2, (2, None)], dump=False, **common)
        add(f"k{k}_final_prior_remap", f"K={k} final-only replay prior remapped from order 1", final_prior=1, dump=False, **common)
        add(f"k{k}_final_prior", f"K={k} final-only replay prior at the run's order", final_prior=2, dump=False, **common)
        add(f"k{k}_star_prior", f"K={k} priors and sampling replayed from RELION STAR files", star_prior=[2, 1, 2, 2], writer=True, **common)
        add(f"k{k}_star_prior_preserved", f"K={k} STAR replay with a start-up prior", star_prior=[2, 2, 2, 2], init_prior=2, dump=False, **common)
        add(f"k{k}_state_swap", f"K={k} state-swap probe restoring the run's own direction prior", iter_prior=[None, 2, 2], dump=False, swap="recovar_direction_prior", **common)
        add(f"k{k}_continued_writer", f"K={k} resumed run that writes checkpoints", max_iter=3, converge_after=None, dump=False, continued=True, writer=True, **common)
    add("k1_star_optimiser", "K=1 STAR replay with optimiser files", n_classes=1, join=0.0, star_prior=[2, 2, 2, 2], star_optimiser=True, dump=False, writer=True)
    add("k2_star_optimiser", "K=2 STAR replay with optimiser files", n_classes=2, join=0.0, star_prior=[2, 2, 2, 2], star_optimiser=True, dump=False)
    add("k1_frozen", "K=1 frozen initial scoring-state assertion", n_classes=1, join=0.0, init_prior=2, frozen=True, dump=False)
    add("k2_frozen", "K=2 frozen assertion (the controller refuses it)", n_classes=2, join=0.0, init_prior=2, frozen=True, dump=False)
    add("k4_writer_dump", "K=4 with checkpoints and the intermediates dump", n_classes=4, join=0.0, writer=True)
    add("k1_cc_writer", "K=1 first-iteration CC with checkpoints", n_classes=1, join=40.0, cc=True, writer=True)
    add("k4_cc_writer", "K=4 first-iteration CC with checkpoints", n_classes=4, join=0.0, cc=True, writer=True)
    # Cases added with the in-repository tool.
    for k in (2, 4):
        add(f"k{k}_seed_start", f"K={k} from one reference: random seed classes in iteration 1", n_classes=k, join=0.0, seed=True, max_iter=3, converge_after=3, writer=True)
        add(f"k{k}_seed_start_cc", f"K={k} from one reference with CC: class 1 copied to every class, seeds in iteration 2", n_classes=k, join=0.0, seed=True, cc=True, max_iter=3, converge_after=3, writer=True)
    for k in (1, 2):
        add(f"k{k}_order_oracle", f"K={k} HEALPix-order schedule 1,2,2: the coarse grids are rebuilt in iteration 2", n_classes=k, join=0.0, orders=(1, 2, 2), max_iter=3, converge_after=3, dump=False)
        add(f"k{k}_overlap", f"K={k} with the two halves' E-steps in one thread each", n_classes=k, join=0.0, overlap=True)
        add(f"k{k}_accuracy", f"K={k} with a stand-in expected-accuracy estimate that succeeds", n_classes=k, join=0.0, accuracy=True, max_iter=3, converge_after=3, dump=False, writer=True)
    add("k1_accuracy_order1", "K=1 from HEALPix order 1 with the stand-in accuracy estimate (native sampling may advance)", n_classes=1, join=0.0, accuracy=True, init_order=1, max_iter=4, converge_after=4, dump=False)
    add("k1_replay_cutoff", "K=1 STAR replay disabled after iteration 1 (--replay-override-max-iter)", n_classes=1, join=0.0, star_prior=[2, 2, 2, 2], replay_max_iter=1, max_iter=3, converge_after=3, dump=False)
    # Non-default RELION-consistency options: the loop forwards them only when one is set.
    counting = dict(noise_shell_count="summed", shell_pair_counting="once", nyquist_column_counting="once")
    for k in (1, 2):
        add(f"k{k}_consistency_counting", f"K={k} with the summed noise count and both pair countings set to once, "
            "current sizes 4, 6, 8 of a box of 8", n_classes=k, join=0.0, consistency=counting, current_sizes=(4, 6, 8),
            max_iter=3, converge_after=3, dump=False, writer=True)
    # --mode relax: the bundle the command resolves for this route (K=1 sets all six options; Class3D keeps
    # the radial gridding window), with the CC iteration its CC-support option needs.
    for k in (1, 2):
        add(f"k{k}_mode_relax", f"K={k} with the consistency options --mode relax resolves to, first-iteration CC, "
            "current sizes 4, 6, 8 of a box of 8", n_classes=k, join=0.0, cc=True, consistency="mode_relax",
            current_sizes=(4, 6, 8), max_iter=3, converge_after=3, dump=False, writer=True)
    for k in (1, 2):
        add(f"k{k}_sealed_sampling", f"K={k}, one iteration on a sealed sampling state (captured directions, psi "
            "angles, translations and sizes), adaptive oversampling 1", n_classes=k, join=0.0, sealed=True,
            oversampling=1, max_iter=1, converge_after=None, dump=False)
    add("k1_sealed_final", "K=1 on a sealed sampling state that converges after its iteration: the final pass "
        "reuses the sealed grid", n_classes=1, join=0.0, sealed=True, max_iter=1, converge_after=1, dump=False)
    # The command entry (full_refinement.main) on a 12-image, 16-pixel data directory.
    seed = ["--seed", "42"]  # a seed whose random halves are both non-empty for twelve particles

    def add_main(name, description, command, arguments, **keywords):
        add(name, description, main=dict(command=command, arguments=[*arguments, *seed], **keywords))

    add_main("main_k1_refine", "relax refine: K=1 from the data directory's start-up map, two iterations",
             "refine", ["--max_iter", "2"])
    add_main("main_k1_os0_options", "relax refine: oversampling 0, no CC iteration, explicit sampling, resolution, "
             "tau2 fudge and particle diameter", "refine",
             ["--max_iter", "2", "--adaptive_oversampling", "0", "--no-firstiter_cc", "--healpix_order", "1",
              "--offset_range", "4", "--offset_step", "2", "--init_resolution", "30", "--tau2_fudge", "2",
              "--particle_diameter_ang", "50"])
    add_main("main_k1_mode_relax", "relax refine --mode relax", "refine", ["--max_iter", "2", "--mode", "relax"])
    add_main("main_k1_schedule", "relax refine with RELION's current sizes and HEALPix orders and a perturbation",
             "refine", ["--max_iter", "2", "--relion_current_sizes", "8,12", "--relion_healpix_orders", "2,2",
                        "--perturb_factor", "0.5", "--perturb_seed", "3"])
    add_main("main_k1_run_files", "relax refine writing every iteration, keeping one, unfiltered half maps, no "
             "final pass", "refine", ["--max_iter", "2", "--write-iteration-every", "1", "--keep-iterations", "1",
                                      "--write-unfiltered-half-maps", "--skip_final_iteration"])
    add_main("main_k1_half_sets_off", "relax refine without the input-STAR half sets", "refine",
             ["--max_iter", "2", "--no-relion-half-sets-from-input"])
    add_main("main_k1_overlap", "relax refine with the halves' E-steps overlapped", "refine",
             ["--max_iter", "2", "--overlap_halves"])
    add_main("main_k1_ledger", "relax refine writing a benchmark ledger", "refine",
             ["--max_iter", "2", "--benchmark_ledger_json", "<OUTDIR>/ledger.json"])
    add_main("main_k1_init_noise", "relax refine starting from an earlier run's archived noise (--init_noise_from_npz)",
             "refine", ["--max_iter", "2", "--init_noise_from_npz", "<FIRST>/refinement_results.npz"],
             first=["--max_iter", "1", "--seed", "42"])
    add_main("main_k1_continue", "relax refine --continue from the run files of one iteration", "refine",
             ["--max_iter", "2"], continue_after=1)
    add_main("main_k2_class3d", "relax class3d: K=2 from the per-class start-up maps", "class3d",
             ["--max_iter", "2", "--n_classes", "2"], n_classes=2)
    add_main("main_k2_one_reference", "relax class3d: K=2 from one start-up map (random seed classes)", "class3d",
             ["--max_iter", "2", "--n_classes", "2", "--init_volume", "<DATA>/reference_init_relion.mrc"],
             n_classes=2)
    add_main("main_k2_os0_cc", "relax class3d: K=2, oversampling 0, first-iteration CC", "class3d",
             ["--max_iter", "2", "--n_classes", "2", "--adaptive_oversampling", "0", "--firstiter_cc"],
             n_classes=2)
    add_main("main_k2_continue", "relax class3d --continue from the run files of one iteration", "class3d",
             ["--max_iter", "2", "--n_classes", "2"], n_classes=2, continue_after=1)
    add_main("main_refused_command", "relax refine asked for two classes (refused)", "refine",
             ["--max_iter", "2", "--n_classes", "2"], n_classes=2)
    add_main("main_refused_single_half", "--diagnostic_single_half without a local-search stop (refused)", "refine",
             ["--max_iter", "2", "--diagnostic_single_half"])
    return cases


CASES = _cases()


# A deliberately wrong controller for the self-test: (name, old lines, new lines, what breaks, detected).
# The first thirteen are the set the refactor workers used; a False marks a known blind spot of the cases.
# A target must occur in exactly one place of the tree: when a statement is rewritten, update its entry.
MUTATIONS = (
    ("final_class_prior_not_normalised", "_prior_k = normalize_class_direction_prior(_prior_k, n_classes, dtype=dtype)", "pass",
     "the final-pass replay prior of a class run is installed unnormalised", True),
    ("resumed_prior_order_dropped", "DirectionPrior(prior.values, order)", "DirectionPrior(prior.values, None)",
     "a resumed direction prior loses its saved HEALPix order", True),
    ("state_swap_prior_order_dropped", 'recovar_snapshot["direction_prior_order_per_half"][k],', "None,",
     "the state-swap probe restores a prior without its order", True),
    ("replay_prior_reversed", "direction_priors[_half_idx] = DirectionPrior(prior_k, prior_order_k)", "direction_priors[_half_idx] = DirectionPrior(prior_k[::-1], prior_order_k)",
     "a replayed K1 prior is installed reversed", True),
    ("snapshot_prior_order_dropped", "direction_prior_order = [p.healpix_order for p in direction_priors]", "direction_prior_order = [None for p in direction_priors]",
     "the checkpoint drops the prior orders", True),
    ("star_class_prior_reversed", "normalize_class_direction_prior(_relion_direction_prior, n_classes, dtype=np.float32),", "normalize_class_direction_prior(_relion_direction_prior[::-1], n_classes, dtype=np.float32),",
     "a STAR class prior is installed reversed", True),
    ("replay_reads_k1_layout", "if n_classes > 1:\nreturn apply_class_iter_replay_overrides(", "if False:\nreturn apply_class_iter_replay_overrides(",
     "the per-iteration replay treats every run as K1", True),
    ("intermediates_one_class", "range(n_classes) if n_classes > 1 else (None,)", "(None,)",
     "the intermediates writer saves one class", True),
    ("shell_cutoff_off_by_one", "first_unavailable_shell = min(truncated.shape[-1], int(current_size) // 2 + 1)", "first_unavailable_shell = min(truncated.shape[-1], int(current_size) // 2)",
     "the current-size shell cut-off moves by one", True),
    ("noise_rows_doubled", "return jnp.asarray(noise.make_radial_noise(shell_profile, image_shape)).reshape(-1)", "return jnp.asarray(noise.make_radial_noise(shell_profile, image_shape)).reshape(-1) * 2",
     "the shells-to-pixel-row noise expansion is doubled", True),
    ("noise_summed_count_dropped", 'if consistency.noise_shell_count == "summed" else None,', 'if consistency.noise_shell_count == "never" else None,',
     "the summed noise count is never forwarded to the noise update", True),
    ("mode_relax_class_gridding_not_skipped", 'not_honoured["gridding_kernel"] = "Class3D\'s tau2 is the power of the radially corrected reference"', "pass",
     "--mode relax sets the separable gridding window in Class3D, which the loop refuses", True),
    ("sealed_rotation_ids_dropped", "if sealed_sampling_state is None or use_local:", "if True:",
     "a sealed capture's rotation ids never reach the scorer", True),
    ("shared_tau2_half2_doubled", "return [tau2, tau2]", "return [tau2, tau2 * 2]",
     "the shared tau2 pair differs between halves", True),
    ("next_sampling_star_skipped", "if not state.has_converged:\n_next_sampling_star", "if False:\n_next_sampling_star",
     "the optimiser replay never reads the next sampling STAR", False),
    ("previous_data_vs_prior_doubled", "data_vs_prior_prev_raw = np.asarray(", "data_vs_prior_prev_raw = 2 * np.asarray(",
     "the half-map size plan reads a doubled previous curve", False),
    # One per operation of the numbered iteration that is, or is to become, a function of its own.
    ("accuracy_projector_size", "shared_projector_size = min(int(current_size), int(image_box_size))", "shared_projector_size = min(int(current_size) - 2, int(image_box_size))",
     "the accuracy estimate's shared projector is built two pixels small", True),
    ("accuracy_status", 'exact_accuracy_status_this_iter = "ok"', 'exact_accuracy_status_this_iter = "okay"',
     "a successful accuracy estimate reports another status", True),
    ("coarse_translation_range", "base_translations = sampling._relion_base_translation_grid(\nstate.translation_range,", "base_translations = sampling._relion_base_translation_grid(\nstate.translation_range * 2,",
     "a rebuilt coarse translation grid has twice the range", True),
    ("replay_translations_reversed", "base_translations = _new_t_source", "base_translations = _new_t_source[::-1]",
     "a translation grid rebuilt under replay is installed reversed", True),
    ("trial_grid_step", "translation_step=float(state.translation_step),\nrandom_perturbation=random_perturbation,\nangular_sampling_deg=angsamp_deg,", "translation_step=2 * float(state.translation_step),\nrandom_perturbation=random_perturbation,\nangular_sampling_deg=angsamp_deg,",
     "the perturbed trial grid shifts translations by twice the step", True),
    ("pass1_angular_sampling", "sampling.relion_angular_sampling_deg(adaptive_pass1_order, adaptive_oversampling=0),", "sampling.relion_angular_sampling_deg(adaptive_pass1_order, adaptive_oversampling=1),",
     "the pass-1 coarse rotations are perturbed by the oversampled step", True),
    ("projector_not_reused", "reusable=reusable_half1 if half.index == 0 else None,", "reusable=None,",
     "half 1's scoring projector is rebuilt instead of reusing the accuracy projector", True),
    ("significance_not_combined", "significance.combine()", "pass",
     "the halves' significant-sample counts are never combined", True),
    ("final_replay_prior_order_dropped", "direction_priors[_half_idx] = DirectionPrior(_prior_k, _prior_order_k)", "direction_priors[_half_idx] = DirectionPrior(_prior_k, None)",
     "the Class3D final-pass replay installs a prior without its HEALPix order", True),
    ("k1_mstep_join_radius", "low_resolution_angstrom=parity.low_resol_join_halves_angstrom,\npixel_resolutions=pixel_resolutions,", "low_resolution_angstrom=0.5 * parity.low_resol_join_halves_angstrom,\npixel_resolutions=pixel_resolutions,",
     "the K=1 M-step joins the half accumulators out to twice the requested resolution (the previous "
     "resolution caps the join radius in these cases)", False),
    ("k1_mstep_details_from_half2", "tau2_update_details = tau2_update_details_per_half[0]\ndel split_prior", "tau2_update_details = tau2_update_details_per_half[1]\ndel split_prior",
     "the K=1 M-step reports half 2's tau2 details", True),
    ("class_mstep_curve_doubled", "data_vs_prior_trajectory.append(data_vs_prior_iter)", "data_vs_prior_trajectory.append(2 * data_vs_prior_iter)",
     "the Class3D M-step publishes a doubled data-vs-prior curve to the history", True),
    ("class_mstep_keeps_old_tau2", "reference_model.tau2 = mean_signal_variance\nreference_model.tau2_per_half = shared_tau2_per_half(reference_model.tau2)", "pass\nreference_model.tau2_per_half = shared_tau2_per_half(reference_model.tau2)",
     "the Class3D M-step does not install its new tau2", True),
    ("class_copy_keeps_class_weights", "np.full(n_classes, float(class_mixture.weights[0]) / n_classes, dtype=np.float64)", "np.asarray(class_mixture.weights, dtype=np.float64)",
     "the one-reference class copy leaves the class weights as they were", True),
    ("class_copy_skips_direction_priors", "_copy_first_class(prior.values), prior.healpix_order,", "prior.values, prior.healpix_order,",
     "the one-reference class copy leaves each class's own direction prior", True),
    ("first_iteration_cc_every_iteration", "parity.emulate_relion_firstiter_cc and init_relion_iteration == 0 and iteration == 0", "parity.emulate_relion_firstiter_cc and init_relion_iteration == 0",
     "the first-iteration CC emulation applies to every iteration", True),
    ("current_size_not_quantised", "current_size = quantize_current_size(image_size_plan.size, ori_size=grid_size)", "current_size = int(image_size_plan.size) + 1",
     "the planned current size is used one pixel larger and unquantised", True),
    ("numbered_result_claims_final_pass", '"final_all_data_ran": False,\n**history,', '"final_all_data_ran": True,\n**history,',
     "the result of a run without a final pass says the final pass ran", True),
    # One per per-mode operation split out of a function that received the mode, and one per dispatch left.
    ("checkpoint_reads_k1_layout", "if int(self.n_classes) > 1:\nreturn self.finish_class(", "if False:\nreturn self.finish_class(",
     "every checkpoint is completed in the K1 layout", True),
    ("k1_checkpoint_growth_fsc_dropped", "fsc_for_growth=host_array(fsc_for_growth, np.float64),", "fsc_for_growth=None,",
     "a K1 checkpoint loses the curve that drives image-size growth", True),
    ("class_checkpoint_weights_dropped", "class_weights=host_array(class_weights, np.float64),", "class_weights=None,",
     "a Class3D checkpoint loses the class weights", True),
    ("final_replay_reads_k1_layout", "if n_classes > 1:\nreturn apply_class_final_replay_state(", "if False:\nreturn apply_class_final_replay_state(",
     "the final-pass replay treats every run as K1", True),
    ("k1_final_replay_prior_order_dropped", "direction_priors[_half_idx] = DirectionPrior(_prior, _prior_order)", "direction_priors[_half_idx] = DirectionPrior(_prior, None)",
     "the K1 final-pass replay installs a prior without its HEALPix order", True),
    ("k1_final_replay_prior_not_remapped", "if _prior_order != healpix_order:", "if False:",
     "the K1 final-pass replay keeps a prior at its source order", True),
    ("final_replay_fields_log_dropped", '",".join(fields) if fields else "<none>",', '"<none>",',
     "the final-pass replay log names no installed field", True),
    ("star_k1_prior_reversed", "direction_priors[_half_idx] = DirectionPrior(_relion_direction_prior, _relion_direction_prior_order)", "direction_priors[_half_idx] = DirectionPrior(_relion_direction_prior[::-1], _relion_direction_prior_order)",
     "a STAR K1 prior is installed reversed", True),
    ("replay_class_prior_not_normalised", "normalize_class_direction_prior(prior_k, n_classes, dtype=dtype), prior_order_k", "prior_k, prior_order_k",
     "a replayed class prior is installed without the per-class normalisation", True),
    ("star_class_weights_dropped", "inferred_weights = class_weights_from_direction_prior(_relion_direction_prior, n_classes)\nif inferred_weights is not None:\n_replay_class_weights = inferred_weights", "inferred_weights = class_weights_from_direction_prior(_relion_direction_prior, n_classes)\nif inferred_weights is not None:\npass",
     "the class weights of a STAR class prior are not returned", True),
    ("replay_class_weights_dropped", "inferred_weights = class_weights_from_direction_prior(_replay_dir_prior, n_classes)\nif inferred_weights is not None:\n_replay_class_weights = inferred_weights", "inferred_weights = class_weights_from_direction_prior(_replay_dir_prior, n_classes)\nif inferred_weights is not None:\npass",
     "the class weights of a replayed class prior are not returned", True),
    ("class_star_shared_model_skipped", "# both RECOVAR halfsets.\n_prior_star = os.path.join(\nperturb_replay_relion_dir,\n_model.star", "# both RECOVAR halfsets.\n_prior_star = os.path.join(\nperturb_replay_relion_dir,\n_absent.star",
     "a Class3D STAR replay never reads the model STAR both halves share", True),
    ("star_priors_never_due", 'return iter_replay_override is None or iter_replay_override.get("direction_prior") is None', "return False",
     "a STAR replay never installs the model STAR's direction priors", True),
    ("replayed_prior_source_order_truncated", "return infer_direction_prior_healpix_order(\ndirection_row,", "return infer_direction_prior_healpix_order(\ndirection_row[:12],",
     "a replayed prior is installed with the order of a 12-direction grid", True),
    ("star_prior_translations_reversed", "prior_translations = jnp.array(replay_translations)", "prior_translations = jnp.array(replay_translations[::-1])",
     "the STAR replay returns its prior translations reversed", False),
    ("star_offset_range_doubled", '_relion_offset_range = float(_replay_meta["offset_range"]) / pixel_size_angstrom', '_relion_offset_range = 2 * float(_replay_meta["offset_range"]) / pixel_size_angstrom',
     "the STAR replay reads twice the translation range", True),
    # The command entry (the main_* cases).
    ("main_max_iter_not_continued", "max_iter=int(args.max_iter) - (continued_iterations or 0),", "max_iter=int(args.max_iter),",
     "a continued run counts --max_iter from its own first iteration", True),
    ("main_seed_classes_inverted", "if args.n_classes > 1 and args.init_volume is not None and not resumed",
     "if args.n_classes > 1 and args.init_volume is None and not resumed",
     "Class3D seeds random classes from per-class maps instead of from one reference", True),
    ("main_tau2_fudge_ignored", "parity=RelionParityOptions(\ntau2_fudge=effective_tau2_fudge,", "parity=RelionParityOptions(\ntau2_fudge=1.0,",
     "the controller is handed the default tau2 fudge instead of --tau2_fudge", True),
    ("startup_k1_lowpass_skipped", "filtered_real = _initial_lowpass_real(init_vol_real, volume_shape, model_pixel_size, ini_high)",
     "filtered_real = np.asarray(init_vol_real, dtype=np.float64)", "the K=1 start-up reference is not low-pass filtered", True),
    ("startup_class_lowpass_skipped", "filtered_real = _initial_lowpass_real(vol_real, volume_shape, voxel_size, ini_high)",
     "filtered_real = np.asarray(vol_real, dtype=np.float64)", "the Class3D start-up references are not low-pass filtered", True),
    ("startup_class_prior_last_class", "prior_source=per_class_ft[0],", "prior_source=per_class_ft[-1],",
     "the Class3D start-up tau2 is bootstrapped from the last class", True),
)


# ----------------------------------------------------------------------------- the worker (imports the tree)


def _worker(source: str, out_path: str, tmp_root: str, names: list[str]) -> None:
    """Run the cases against ``source`` in this process; the caller has set the CPU-only environment."""
    import importlib
    import importlib.util
    import logging
    import threading
    import types

    sys.dont_write_bytecode = True
    sys.path[:0] = [source, os.path.join(source, "tests"), os.path.join(source, "tests", "unit")]
    os.chdir(tmp_root)

    import jax.numpy as jnp
    import numpy as np
    import recovar
    import test_refine_relion_mode as fixtures
    from helpers.fake_adaptive_engine import adaptive_result

    import relax
    from relax.helpers import expected_accuracy
    from relax.refinement import iteration_loop, refinement_options

    assert relax.__file__.startswith(source), relax.__file__
    assert fixtures.__file__.startswith(source), fixtures.__file__

    tmp_pattern = re.compile(re.escape(tmp_root) + r"(/tmp\w+)?")
    # The command entry also records its own command line and the run's cache directories.
    run_paths = [(re.escape(out_path), "<OUT>"), (re.escape(str(Path(tmp_root).parent)), "<WORK>"),
                 (re.escape(source), "<SRC>")]

    def scrub(text):
        text = tmp_pattern.sub(TMP_TOKEN, text)
        for pattern, token in run_paths:
            text = re.sub(pattern, token, text)
        return text

    trace: list = []

    def record(*row):
        trace.append((threading.current_thread().name, list(row)))

    class Capture(logging.Handler):
        def emit(self, log_record):
            template = str(log_record.msg)
            try:
                text = log_record.getMessage()
            except Exception:  # a malformed format string is itself behaviour
                text = "<unformattable> " + template
            trace.append((
                threading.current_thread().name,
                log_row(log_record.name, log_record.levelname, template, text, scrub=scrub),
            ))

    def relax_modules_with(name):
        return [
            module for module_name, module in sorted(sys.modules.items())
            if module_name.startswith("relax") and callable(getattr(module, name, None))
        ]

    class Dataset(fixtures.MockDataset):
        """The unit-test dataset plus the row layout the Class3D seed iteration reads."""

        def __init__(self, n_images, rng, first_row):
            super().__init__(n_images, rng)
            self._index_layout = self
            self._first_row = first_row

        def original_image_indices_for_local(self, indices):
            return self._first_row + np.asarray(indices, dtype=np.int64)

    def stand_in_engine(experiment_dataset, means, mean_variance, noise_variance, coarse_rotations,
                        coarse_translations, fine_rotations, fine_translations, rot_parent_map, trans_parent_map,
                        disc_type, **kwargs):
        """CPU stand-in for run_dense_k_class_em_adaptive: a result seeded by every operand it receives."""
        seed = digest_operands(means, mean_variance, noise_variance, coarse_rotations, coarse_translations,
                               fine_rotations, fine_translations, rot_parent_map, trans_parent_map, disc_type, kwargs)
        record("call", "engine", f"seed={seed:016x}", "kwargs=" + ",".join(sorted(kwargs)),
               "priors=" + ",".join(k for k in sorted(kwargs) if "rotation_log_prior" in k and kwargs[k] is not None))
        # Overlapped halves: finish in half order, so what the controller records does not depend on a race.
        me = threading.current_thread()
        for other in threading.enumerate():
            if other is not me and other.name.startswith("em-half-") and me.name.startswith("em-half-") and other.name < me.name:
                other.join()
        rng = np.random.default_rng(seed)
        n_classes = int(np.asarray(means).shape[0]) if np.ndim(means) >= 2 else 1
        n_images = int(experiment_dataset.n_units)
        n_fine = int(np.asarray(fine_rotations).shape[0])

        def half_map(k, size):
            side = round(size ** (1.0 / 3.0))
            return fixtures._hermitian_volume((side, side, side), seed=int(rng.integers(0, 2**31 - 1)) + k)

        def weights(k, size):
            return jnp.asarray(1.0 + rng.random(size), dtype=jnp.complex64)

        result = adaptive_result(
            experiment_dataset, means, fine_rotations, kwargs,
            Ft_y=half_map, Ft_ctf=weights,
            max_posterior=0.2 + 0.8 * rng.random(n_images),
            pose_assignments=rng.integers(0, max(n_fine * int(np.asarray(fine_translations).shape[0]), 1), size=n_images),
            best_pose_translations=rng.integers(-1, 2, size=(n_images, 2)).astype(np.float32),
            significant_counts=rng.integers(1, 5, size=n_images),
            sigma2_offset=float(rng.random()),
        )
        per_class = tuple(
            stats._replace(rotation_posterior_sums=jnp.asarray(rng.random(n_fine) * (k + 1), dtype=jnp.float32))
            for k, stats in enumerate(result.per_class_stats)
        )
        classes = rng.integers(0, n_classes, size=n_images)
        resp = rng.random((n_classes, n_images))
        resp /= resp.sum(axis=0, keepdims=True)
        return result._replace(
            per_class_stats=per_class, stats=per_class[0],
            class_assignments=jnp.asarray(classes, dtype=jnp.int32),
            class_responsibilities=jnp.asarray(resp, dtype=jnp.float32),
            class_posterior_sums=jnp.asarray(resp.sum(axis=1), dtype=jnp.float32),
        )

    def stand_in_accuracy(self, **operands):
        """Stand-in for Half1AccuracyInputs.estimate: a successful estimate seeded by its operands."""
        seed = digest_operands(operands)
        record("call", "accuracy", f"seed={seed:016x}", "operands=" + ",".join(sorted(operands)),
               f"projector={'none' if operands.get('projector_data') is None else 'shared'}")
        rng = np.random.default_rng(seed)
        n_classes = int(np.asarray(operands["class_weights"]).size)
        trials = np.asarray(self.trial_order_local, dtype=np.int64)[:2]
        return expected_accuracy.ExpectedAccuracy(
            acc_rot=float(1.0 + rng.random()),
            acc_trans_angstrom=float(0.5 + rng.random()),
            acc_rot_per_class=1.0 + rng.random(n_classes),
            acc_trans_per_class_angstrom=0.5 + rng.random(n_classes),
            class_counts=rng.integers(0, 3, size=n_classes),
            trial_local_indices=trials,
            trial_particle_ids=trials + 1,
        )

    def one_prior(n_classes, order, seed):
        """A direction prior with zeros and unequal class mass: (n_pix,) for K=1, (K, n_pix) for K>1."""
        rng = np.random.default_rng(seed)
        n_pix = 12 * 4**order
        values = rng.random((n_classes, n_pix)) + 0.05
        values[:, rng.integers(0, n_pix, size=5)] = 0.0
        values *= (1.0 + np.arange(n_classes))[:, None]
        values /= values.sum()
        return values[0] / values[0].sum() if n_classes == 1 else values

    def prior_pair(n_classes, order, seed):
        if isinstance(order, tuple):  # (order of half 1, order of half 2); None leaves that half alone
            return [None if o is None else one_prior(n_classes, o, seed + h) for h, o in enumerate(order)]
        return [one_prior(n_classes, order, seed), one_prior(n_classes, order, seed + 1)]

    def write_relion_dir(n_classes, orders, max_iter, optimiser):
        """RELION sampling and model STAR files for the replay directory; orders[i] is the prior written
        after RELION iteration i+1 (None writes no model file for that iteration)."""
        root = Path(tempfile.mkdtemp(dir=tmp_root))
        for it in range(0, max_iter + 2):
            (root / f"run_it{it:03d}_sampling.star").write_text(
                "data_sampling_general\n\n_rlnHealpixOrder 2\n_rlnPsiStep 15.0\n_rlnOffsetRange 10.0\n"
                "_rlnOffsetStep 2.0\n_rlnSamplingPerturbInstance 0.25\n_rlnSamplingPerturbFactor 0.5\n"
            )
        if optimiser:
            for it in range(1, max_iter + 2):
                (root / f"run_it{it:03d}_optimiser.star").write_text(
                    "data_optimiser_general\n\n_rlnOverallAccuracyRotations 1.75\n"
                    "_rlnOverallAccuracyTranslationsAngst 0.75\n_rlnNumberOfIterWithoutResolutionGain 1\n"
                    f"_rlnNumberOfIterWithoutChangingAssignments {it}\n_rlnChangesOptimalOffsets 0.625\n"
                    "_rlnHasConverged 0\n"
                )
        for it, order in enumerate(orders, start=1):
            if order is None:
                continue
            for half in [None] if n_classes > 1 else [1, 2]:
                values = np.atleast_2d(one_prior(n_classes, order, 400 + it + (half or 0)))
                text = "data_model_general\n\n_rlnCurrentImageSize 4\n_rlnCurrentResolution 0.1\n\n"
                for c in range(n_classes):
                    text += f"data_model_pdf_orient_class_{c + 1}\n\nloop_\n_rlnOrientationDistribution #1\n"
                    text += "".join(f"{float(v)!r}\n" for v in values[c]) + "\n"
                name = f"run_it{it:03d}_model.star" if half is None else f"run_it{it:03d}_half{half}_model.star"
                (root / name).write_text(text)
        return str(root)

    class Writer:
        """Checkpoint writer that keeps and fingerprints each snapshot the controller publishes."""

        def __init__(self):
            self.snapshots, self.prints = [], []

        def due(self, relion_iteration):
            return True

        def wants_unfiltered_maps(self, relion_iteration, *, n_classes):
            return False

        def __call__(self, snapshot):
            record("call", "checkpoint", f"snapshot={len(self.snapshots)}")
            self.snapshots.append(snapshot)
            self.prints.append(flatten(snapshot, scrub=scrub))

    def install_stand_ins(patch, calls, *, converge_after, accuracy):
        """The stand-in engine and the call spies of every case; ``converge_after`` forces convergence."""
        for owner in relax_modules_with("run_dense_k_class_em_adaptive"):
            patch(owner, "run_dense_k_class_em_adaptive", stand_in_engine)
        for owner in relax_modules_with("_host_tau2_volumes"):
            original_host = owner._host_tau2_volumes

            def host_spy(*args, _original=original_host, **kwargs):
                out = _original(*args, **kwargs)
                record("call", "_host_tau2_volumes", json.dumps(flatten([list(args), kwargs, list(out)]), sort_keys=True))
                return out

            patch(owner, "_host_tau2_volumes", host_spy)
        for owner in relax_modules_with("prepare_scoring_projector"):
            original_projector = owner.prepare_scoring_projector

            def projector_spy(references, *, _original=original_projector, **kwargs):
                out = _original(references, **kwargs)
                reusable = kwargs.get("reusable")
                record("call", "prepare_scoring_projector",
                       json.dumps(flatten([references, {k: v for k, v in kwargs.items() if k != "reusable"}, out]), sort_keys=True),
                       f"reusable={'none' if reusable is None else 'given'}",
                       f"reused={reusable is not None and out is reusable.projector}")
                return out

            patch(owner, "prepare_scoring_projector", projector_spy)
        for owner in relax_modules_with("_relion_adaptive_pass1_rotations"):
            original_pass1 = owner._relion_adaptive_pass1_rotations

            def pass1_spy(*args, _original=original_pass1, **kwargs):
                out = _original(*args, **kwargs)
                record("call", "_relion_adaptive_pass1_rotations", json.dumps(flatten([list(args), kwargs, out]), sort_keys=True))
                return out

            patch(owner, "_relion_adaptive_pass1_rotations", pass1_spy)
        for owner in relax_modules_with("update_refinement_state"):
            original_update = owner.update_refinement_state

            def force(*args, _original=original_update, **kwargs):
                updated = _original(*args, **kwargs)
                calls["updates"] += 1
                if converge_after is not None and calls["updates"] >= converge_after:
                    updated.has_converged = True
                return updated

            patch(owner, "update_refinement_state", force)
        if accuracy:
            patch(expected_accuracy.Half1AccuracyInputs, "estimate", stand_in_accuracy)

    def run_case(n_classes, join, *, max_iter=4, converge_after=2, dump=True, cc=False, oversampling=0, env=None,
                 writer=False, skip_final=False, continued=False, resume=None, perturb=None, init_prior=None,
                 iter_prior=None, final_prior=None, star_prior=None, star_optimiser=False, swap=None, frozen=False,
                 seed=False, orders=None, overlap=False, accuracy=False, init_order=2, replay_max_iter=None,
                 consistency=None, current_sizes=None, sealed=False):
        first_pass = None
        if continued:
            first_pass = run_case(n_classes, join, max_iter=1, converge_after=None, dump=False, cc=cc, writer=True,
                                  skip_final=True, perturb=perturb)
            snapshots = first_pass.pop("snapshots")
            if first_pass["status"]["error"] != "None" or not snapshots:
                return {"status": {"error": "first pass failed"}, "result": {}, "files": {}, "checkpoints": {},
                        "trace": [], "first_pass": first_pass}
            resume = snapshots[-1]
        rng = np.random.default_rng(fixtures.SEED)
        n_half = fixtures.N_IMAGES // 2
        halves = [Dataset(n_half, rng, 0), Dataset(n_half, rng, n_half)]
        init_volume = fixtures._hermitian_volume(fixtures.VOLUME_SHAPE, seed=42)
        translations = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
        patches, calls = [], {"updates": 0}
        del trace[:]

        def patch(owner, name, replacement):
            patches.append((owner, name, getattr(owner, name)))
            setattr(owner, name, replacement)

        install_stand_ins(patch, calls, converge_after=converge_after, accuracy=accuracy)
        for name in ("RELAX_PARITY_DUMP_DIR", "RELAX_PARITY_TIMING_DIR", "RELAX_FINAL_ALL_DATA_AFTER_MAX_ITER"):
            os.environ.pop(name, None)
        os.environ.update(env or {})
        dump_dir = tempfile.mkdtemp(dir=tmp_root) if dump else None
        handler = Capture(level=logging.DEBUG)
        root = logging.getLogger("relax")
        old_level = root.level
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)
        extra = {}
        if n_classes > 1:
            extra["k_class"] = refinement_options.KClassOptions(
                n_classes=n_classes,
                init_class_log_priors=np.log(np.full(n_classes, 1.0 / n_classes, dtype=np.float64)),
                **({"first_iteration_seed_classes": np.arange(fixtures.N_IMAGES)[::-1] % n_classes} if seed else {}),
            )
        adaptive_fields = {}
        if oversampling:
            adaptive_fields["adaptive_oversampling"] = oversampling
        if orders is not None:
            adaptive_fields["relion_healpix_orders"] = tuple(orders)
            init_order = int(orders[0])
        if current_sizes is not None:
            adaptive_fields["relion_current_sizes"] = tuple(current_sizes)
        if adaptive_fields:
            extra["adaptive"] = refinement_options.AdaptiveOptions(**adaptive_fields)
        if overlap:
            extra["overlap"] = refinement_options.HalfOverlapOptions(overlap_halves=True)
        if consistency == "mode_relax":
            if hasattr(refinement_options, "relax_mode_consistency"):
                extra["consistency"], _ = refinement_options.relax_mode_consistency(
                    {}, n_classes=n_classes, has_cc_iteration=cc, coarse_engine="auto", ctf_premultiplied=False)
            else:
                # A source older than the mode (the base of the commit that added it): the six options by name.
                extra["consistency"] = refinement_options.RelionConsistencyOptions(
                    shell_pair_counting="once", noise_shell_count="summed", initial_noise_pair_counting="once",
                    nyquist_column_counting="once", firstiter_cc_support="gaussian",
                    **({"gridding_kernel": "separable"} if n_classes == 1 else {}))
        elif consistency:
            extra["consistency"] = refinement_options.RelionConsistencyOptions(**consistency)
        writer_object = Writer() if writer else None
        if writer or resume is not None:
            extra["checkpoint"] = refinement_options.CheckpointOptions(writer=writer_object, resume=resume)
        parity = dict(low_resol_join_halves_angstrom=join, perturb_seed=17, optimizer_random_seed=17)
        replay_fields = {}
        if init_prior is not None:
            replay_fields["init_direction_prior"] = prior_pair(n_classes, init_prior, 100)
        if iter_prior is not None:
            replay_fields["replay_iteration_overrides"] = [
                None if order is None else {"direction_prior": prior_pair(n_classes, order, 200 + i)}
                for i, order in enumerate(iter_prior)
            ]
        if final_prior is not None:
            replay_fields["final_replay_override"] = {"direction_prior": prior_pair(n_classes, final_prior, 300)}
        if frozen:
            replay_fields.update(
                init_previous_best_rotation_eulers=[np.full((n_half, 3), 10.0 * (h + 1), dtype=np.float32) for h in range(2)],
                init_previous_best_translations=[np.zeros((n_half, 2), dtype=np.float32) for _ in range(2)],
                init_image_corrections=[np.ones(n_half, dtype=np.float32) for _ in range(2)],
                init_scale_corrections=[np.ones(n_half, dtype=np.float32) for _ in range(2)],
            )
        if replay_fields:
            # The replay slots and the final-only replay are a ReplayState's on a source older than the replay
            # input source's settings (RelionReplay); run_case moves them there below otherwise.
            replay_state_fields = {f.name for f in dataclasses.fields(refinement_options.ReplayState)}
            extra["replay"] = refinement_options.ReplayState(
                **{name: value for name, value in replay_fields.items() if name in replay_state_fields}
            )
        if star_prior is not None:
            parity["perturb_replay_relion_dir"] = write_relion_dir(n_classes, star_prior, max_iter, star_optimiser)
            parity["perturb_replay_precision"] = "star"
        if replay_max_iter is not None:
            parity["perturb_replay_max_iter"] = replay_max_iter
        debug_fields = {}
        if swap is not None:
            debug_fields["state_swap_probe"] = {"iteration": 1, "variant": swap}
        if frozen:
            debug_fields["assert_initial_scoring_state_immutable"] = True
        observer = {}
        if dump and "save_intermediates_dir" in {f.name for f in dataclasses.fields(refinement_options.EngineDebugOptions)}:
            # A source older than the observer port (code rule 15): the intermediates are a debug option.
            debug_fields["save_intermediates_dir"] = dump_dir
        elif dump:
            from relax.diagnostics.observers import IntermediatesObserver

            observer["observer"] = IntermediatesObserver(dump_dir)
        if sealed:
            # A schema-v3 sealed sampling state: three order-2 directions, two psi angles, three translations.
            debug_fields["sealed_sampling_state"] = {
                "consumer_relion_iteration": 1,
                "directions_ipix": np.asarray([7, 19, 103], dtype=np.int64),
                "rot_angles_deg": np.asarray([10.0, 20.0, 30.0], dtype=np.float64),
                "tilt_angles_deg": np.asarray([40.0, 50.0, 60.0], dtype=np.float64),
                "psi_angles_deg": np.asarray([0.0, 90.0], dtype=np.float64),
                "translations_x_angstrom": np.asarray([-1.0, 0.0, 1.0], dtype=np.float64),
                "translations_y_angstrom": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
                "healpix_order_original": 2, "psi_step_deg": 90.0,
                "offset_range_angstrom": 1.0, "offset_step_angstrom": 1.0,
                "perturbation_factor": 0.5, "random_perturbation": 0.125,
                "sigma_rot_deg": 0.0, "sigma_psi_deg": 0.0,
                "coarse_size": 4, "current_size": 6,
            }
        replay_class = None
        if importlib.util.find_spec("relax.parity") is not None:
            replay_class = getattr(importlib.import_module("relax.parity.relion_replay_source"), "RelionReplay", None)
        source_fields = {}
        if replay_class is not None and "sealed_sampling_state" in {f.name for f in dataclasses.fields(replay_class)}:
            # A source whose frozen boundary and state-swap probe enter through the input source (code rule 15).
            renamed = {"assert_initial_scoring_state_immutable": "assert_scoring_state_unchanged"}
            for name in ("state_swap_probe", "assert_initial_scoring_state_immutable", "sealed_sampling_state"):
                if name in debug_fields:
                    source_fields[renamed.get(name, name)] = debug_fields.pop(name)
        if debug_fields:
            extra["debug"] = refinement_options.EngineDebugOptions(**debug_fields)
        if perturb is not None:
            parity["perturb_factor"] = perturb
        if cc:
            parity.update(emulate_relion_firstiter_cc=True, relion_firstiter_ini_high_angstrom=8.0)
        error, result = None, {}
        try:
            source = {}
            replay_module = (
                importlib.import_module("relax.parity.relion_replay_source")
                if importlib.util.find_spec("relax.parity") is not None else None
            )
            if replay_module is not None and hasattr(replay_module, "RelionReplay"):
                # A source with the input-source port (code rule 15): what the case replays enters through it.
                replay_settings = {
                    name: group.pop(name)
                    for group, names in (
                        (parity, ("perturb_replay_relion_dir", "perturb_replay_precision", "perturb_replay_max_iter")),
                        (replay_fields, ("replay_iteration_overrides", "final_replay_override")),
                    )
                    for name in names if name in group
                }
                replay_settings.update(source_fields)
            options = refinement_options.RefinementOptions(
                disc_type="linear_interp",
                schedule=refinement_options.RefinementSchedule(
                    max_iter=max_iter, init_current_size=4, init_healpix_order=init_order, max_healpix_order=2,
                    **({} if resume is None else {"init_relion_iteration": int(resume.relion_iteration)}),
                    **({"skip_final_iteration": True} if skip_final else {}),
                ),
                batching=refinement_options.RefinementBatching(
                    image_batch_size=fixtures.N_IMAGES, rotation_block_size=fixtures.N_ROTATIONS,
                ),
                parity=refinement_options.RelionParityOptions(**parity),
                **extra,
            )
            if replay_module is not None and hasattr(replay_module, "RelionReplay"):
                source["source"] = replay_module.RelionReplaySource.for_run(
                    replay_module.RelionReplay(**replay_settings), options,
                )
            elif replay_module is not None:
                # A source whose replay source still read the replay settings from the options.
                source["source"] = replay_module.RelionReplaySource.from_options(options)
            result = iteration_loop.refine_single_volume(
                halves,
                init_volume,
                jnp.ones(fixtures.IMAGE_SIZE, dtype=jnp.float32),
                jnp.ones(fixtures.VOLUME_SIZE, dtype=jnp.float32) * 100.0,
                translations,
                options=options,
                **observer,
                **source,
            )
        except Exception as exc:  # recorded: both sides must fail the same way
            error = (type(exc).__name__, scrub(str(exc)))
        finally:
            root.removeHandler(handler)
            root.setLevel(old_level)
            for owner, name, original in reversed(patches):
                setattr(owner, name, original)
            for name in env or {}:
                os.environ.pop(name, None)
        files = {}
        if dump:
            base = Path(dump_dir)
            for path in sorted(base.rglob("*")):
                if not path.is_file():
                    continue
                rel = str(path.relative_to(base))
                if path.suffix == ".npz":
                    with np.load(path, allow_pickle=True) as data:
                        for key in data.files:
                            flatten(data[key], f"{rel}/{key}", files, scrub=scrub)
                elif path.suffix == ".npy":
                    flatten(np.load(path, allow_pickle=True), rel, files, scrub=scrub)
                else:
                    files[rel] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        result = result_fields(result)
        out = {
            "status": {
                "error": repr(error),
                "final_all_data_ran": repr(result.get("final_all_data_ran")),
                "update_calls": repr(calls["updates"]),
            },
            "result": flatten(result, scrub=scrub),
            "files": files,
            "checkpoints": {} if writer_object is None else flatten(writer_object.prints),
            "trace": [row for row in ordered_trace(trace) if row[0] != "log" or row[1].startswith("relax")],
        }
        if writer_object is not None and not continued and resume is None:
            out["snapshots"] = writer_object.snapshots
        if first_pass is not None:
            out["first_pass"] = first_pass
        return out

    def write_tiny_dataset(root, *, n_images=12, box=16, n_classes=1, seed=5):
        """A RELION particle STAR with its stack and RELION-convention start-up maps, as a data directory."""
        import mrcfile
        import pandas as pd
        import starfile

        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        pixel = 4.25
        with mrcfile.new(root / f"particles.{box}.mrcs") as stack:
            stack.set_data(rng.standard_normal((n_images, box, box)).astype(np.float32))
            stack.voxel_size = pixel
        maps = ["reference_init_relion.mrc"] + [f"reference_init_class{k + 1:03d}_relion.mrc" for k in range(n_classes)]
        for name in maps:
            with mrcfile.new(root / name) as volume:
                volume.set_data(rng.standard_normal((box, box, box)).astype(np.float32))
                volume.voxel_size = pixel
        optics = pd.DataFrame({
            "rlnOpticsGroup": [1], "rlnOpticsGroupName": ["opticsGroup1"], "rlnAmplitudeContrast": [0.07],
            "rlnSphericalAberration": [2.7], "rlnVoltage": [300.0], "rlnImagePixelSize": [pixel],
            "rlnImageSize": [box], "rlnImageDimensionality": [2],
        })
        rows = np.arange(n_images)
        particles = pd.DataFrame({
            "rlnImageName": [f"{i + 1}@particles.{box}.mrcs" for i in rows],
            "rlnMicrographName": [str(i + 1) for i in rows],
            "rlnDefocusU": 15000.0 + 100.0 * rows, "rlnDefocusV": 15100.0 + 100.0 * rows,
            "rlnDefocusAngle": np.full(n_images, 10.0), "rlnPhaseShift": np.zeros(n_images),
            "rlnOpticsGroup": np.ones(n_images, dtype=int),
            "rlnAngleRot": rng.uniform(-180.0, 180.0, n_images), "rlnAngleTilt": rng.uniform(0.0, 180.0, n_images),
            "rlnAnglePsi": rng.uniform(-180.0, 180.0, n_images),
            "rlnOriginXAngst": np.zeros(n_images), "rlnOriginYAngst": np.zeros(n_images),
        })
        star = root / "particles.star"
        starfile.write({"optics": optics, "particles": particles}, star)
        # The writer's creation stamp would make the file, and the hash main records of it, differ per run.
        star.write_text("".join(line for line in star.read_text().splitlines(True) if not line.startswith("# Created")))
        return root

    def hash_output_files(base, out):
        """Every file under ``base``: archives by array, maps by data, text with its creation stamp dropped."""
        import mrcfile

        for path in sorted(Path(base).rglob("*")):
            if not path.is_file():
                continue
            rel = str(path.relative_to(base))
            if path.suffix == ".npz":
                with np.load(path, allow_pickle=True) as data:
                    for key in data.files:
                        if any(word in key for word in ("time", "wall", "seconds", "elapsed", "cumulative_s")):
                            continue
                        value = data[key]
                        if value.dtype.kind == "U":  # paths of the run's directories
                            value = np.vectorize(scrub, otypes=[str])(value) if value.size else value
                        flatten(value, f"{rel}/{key}", out, scrub=scrub)
            elif path.suffix in (".mrc", ".mrcs"):
                with mrcfile.open(path, permissive=True) as volume:
                    flatten(np.asarray(volume.data), rel, out, scrub=scrub)
            elif path.suffix == ".json":
                # A ledger: its wall times and the checkout's git state differ per run and per source tree.
                ledger = json.loads(path.read_text())
                if isinstance(ledger, dict):
                    ledger = {key: value for key, value in ledger.items()
                              if not any(word in key for word in ("time", "wall", "git", "seconds", "timing"))}
                flatten(ledger, rel, out, scrub=scrub)
            elif path.suffix in (".star", ".txt", ".log"):
                lines = [line for line in path.read_text().splitlines() if not line.startswith("# Created")]
                out[rel] = "sha256:" + hashlib.sha256(scrub("\n".join(lines)).encode()).hexdigest()
            else:
                out[rel] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        return out

    def run_main_case(command, arguments, *, n_classes=1, converge_after=None, continue_after=None, first=None):
        """``relax <command>`` (full_refinement.main) on a tiny data directory, the stand-in engine underneath.

        Records the operands main hands refine_single_volume, its result, and every file main writes.
        ``continue_after`` first runs that many iterations and then continues from their run files; ``first``
        is the arguments of an earlier run whose output directory the case's arguments name as ``<FIRST>``.
        """
        import jax

        from relax.refinement import full_refinement

        work = Path(tempfile.mkdtemp(dir=tmp_root))
        data = write_tiny_dataset(work / "data", n_classes=n_classes)
        environ = dict(os.environ)
        patches, calls = [], {"updates": 0}
        refine_inputs, refine_results = [], []
        del trace[:]

        def patch(owner, name, replacement):
            patches.append((owner, name, getattr(owner, name)))
            setattr(owner, name, replacement)

        install_stand_ins(patch, calls, converge_after=converge_after, accuracy=False)
        original_refine = iteration_loop.refine_single_volume

        def refine_spy(*args, **kwargs):
            refine_inputs.append(flatten([list(args), kwargs], scrub=scrub))
            record("call", "refine_single_volume", "inputs")
            out = original_refine(*args, **kwargs)
            refine_results.append(flatten(result_fields(out), scrub=scrub))
            return out

        patch(iteration_loop, "refine_single_volume", refine_spy)
        patch(jax, "devices", lambda *args, **kwargs: [types.SimpleNamespace(platform="gpu", id=0)])
        handler = Capture(level=logging.DEBUG)
        root = logging.getLogger("relax")
        old_level = root.level
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)
        old_argv, old_orig_argv = sys.argv, sys.orig_argv
        import contextlib
        import io

        printed = io.StringIO()
        statuses = []
        try:
            runs = [("first", work / "first", [*arguments, "--max_iter", str(continue_after)])] if continue_after else []
            if first is not None:
                runs.append(("first", work / "first", list(first)))
            runs.append(("run", work / "out", list(arguments)))
            for label, output, run_arguments in runs:
                if label == "run" and continue_after:
                    optimiser = sorted((work / "first").glob("run_it*_optimiser.star"))[-1]
                    run_arguments += ["--continue", str(optimiser)]
                run_arguments = [argument.replace("<DATA>", str(data)).replace("<OUTDIR>", str(output))
                                 .replace("<FIRST>", str(work / "first")) for argument in run_arguments]
                sys.argv = ["relax", "--data_dir", str(data), "--output", str(output), *run_arguments]
                sys.orig_argv = ["python", "-m", f"relax.commands.{command}", *sys.argv[1:]]  # run files record it
                try:
                    with contextlib.redirect_stdout(printed):
                        full_refinement.run_from_command_line(command)
                    statuses.append("ok")
                except SystemExit as exc:
                    statuses.append(f"SystemExit({scrub(str(exc.code))})")
                except Exception as exc:  # recorded: both sides must fail the same way
                    statuses.append(f"{type(exc).__name__}({scrub(str(exc))})")
                    break
        finally:
            sys.argv, sys.orig_argv = old_argv, old_orig_argv
            root.removeHandler(handler)
            root.setLevel(old_level)
            for owner, name, original in reversed(patches):
                setattr(owner, name, original)
            os.environ.clear()
            os.environ.update(environ)
        files = hash_output_files(work, {})
        # What main prints (its summary table): wall times blanked, kept as log rows.
        for line in printed.getvalue().splitlines():
            trace.append(("MainThread", ["log", "relax.stdout", "PRINT", "", scrub(re.sub(r"\d+\.\d+s?", "#", line))]))
        result = {}
        for index, (inputs, out) in enumerate(zip(refine_inputs, refine_results + [{}] * len(refine_inputs))):
            result.update({f"refine{index}/inputs{key}": value for key, value in inputs.items()})
            result.update({f"refine{index}/result{key}": value for key, value in out.items()})
        return {
            "status": {"runs": repr(statuses), "update_calls": repr(calls["updates"])},
            "result": result,
            "files": files,
            "checkpoints": {},
            "trace": [row for row in ordered_trace(trace) if row[0] != "log" or row[1].startswith("relax")],
        }

    importlib.invalidate_caches()
    results = {}
    for name in names:
        keywords = CASES[name][1]
        case = run_main_case(**keywords["main"]) if "main" in keywords else run_case(**keywords)
        case.pop("snapshots", None)
        first_pass = case.pop("first_pass", None)
        if first_pass is not None:  # the one-iteration run a continued case resumes from is a case of its own
            results[name + "/first_pass"] = first_pass
        results[name] = case
        print(
            f"{name}: error={case['status'].get('error', case['status'].get('runs'))} "
            f"final={case['status'].get('final_all_data_ran')} "
            f"result={len(case['result'])} files={len(case['files'])} checkpoints={len(case['checkpoints'])} "
            f"trace={len(case['trace'])}",
            flush=True,
        )
    fingerprint = {
        "schema": SCHEMA,
        "source": source,
        "relax": relax.__file__,
        "recovar": recovar.__file__,
        "not_covered": list(NOT_COVERED),
        "cases": results,
    }
    Path(out_path).write_text(json.dumps(fingerprint, sort_keys=True))
    print(f"{len(results)} cases from {relax.__file__} (recovar {recovar.__file__}) -> {out_path}")


# ----------------------------------------------------------------------------- commands


def _print_not_covered() -> None:
    print("NOT covered by these fingerprints:")
    for item in NOT_COVERED:
        print(f"  - {item}")


def export_rev(rev: str, work_dir: Path) -> Path:
    """A ``git archive`` export of ``rev`` (``relax`` and ``tests``) under ``work_dir``."""
    sha = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--verify", f"{rev}^{{commit}}"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    target = work_dir / f"src_{sha[:12]}"
    if not target.exists():
        staging = Path(tempfile.mkdtemp(dir=work_dir, prefix="export_"))
        archive = subprocess.Popen(["git", "-C", str(REPO_ROOT), "archive", sha, "relax", "tests"], stdout=subprocess.PIPE)
        subprocess.run(["tar", "-x", "-C", str(staging)], stdin=archive.stdout, check=True)
        if archive.wait() != 0:
            raise RuntimeError(f"git archive {sha} failed")
        staging.rename(target)
    return target


def run_tree(source: Path, out: Path, work_dir: Path, names: list[str], *, threads: int = 4, quiet: bool = False) -> int:
    """Fingerprint ``source`` in a child process with a CPU-only environment that imports from ``source``."""
    label = hashlib.sha256(f"{source}|{out}".encode()).hexdigest()[:12]
    tmp_root = work_dir / f"tmp_{label}"
    shutil.rmtree(tmp_root, ignore_errors=True)
    tmp_root.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONHOME", "CONDA_PREFIX", "VIRTUAL_ENV")}
    env.update(
        PYTHONPATH=str(source), PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", XLA_PYTHON_CLIENT_PREALLOCATE="false",
        OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
        JAX_COMPILATION_CACHE_DIR=str(work_dir / f"jax_cache_{label}"),
        RECOVAR_JAX_CACHE_DIR=str(work_dir / f"recovar_jax_cache_{label}"),
    )
    command = [sys.executable, str(Path(__file__).resolve()), "_worker", str(source), str(out), str(tmp_root), *names]
    log_path = out.with_suffix(out.suffix + ".log")
    with open(log_path, "w") as log:
        code = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if not quiet or code != 0:
        tail = log_path.read_text().splitlines()
        for line in tail if code != 0 else tail[-1:]:
            print(line)
    shutil.rmtree(tmp_root, ignore_errors=True)
    return code


def _resolve_source(args, work_dir: Path) -> Path:
    if getattr(args, "rev", None):
        return export_rev(args.rev, work_dir)
    return Path(args.source).resolve() if getattr(args, "source", None) else REPO_ROOT


def _work_dir(args) -> Path:
    if not args.work_dir:
        return Path(tempfile.mkdtemp(prefix="relax-fingerprint-"))
    path = Path(args.work_dir).resolve()
    if REPO_ROOT in path.parents or path == REPO_ROOT:
        raise SystemExit(f"--work-dir {path} is inside the checkout; choose a directory outside it")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _case_names(requested: list[str]) -> list[str]:
    unknown = [name for name in requested if name not in CASES]
    if unknown:
        raise SystemExit(f"unknown case(s): {', '.join(unknown)}; `fingerprint.py cases` lists them")
    return requested or list(CASES)


def command_run(args) -> int:
    work_dir = _work_dir(args)
    code = run_tree(_resolve_source(args, work_dir), Path(args.out).resolve(), work_dir, _case_names(args.cases))
    _print_not_covered()
    return code


def command_diff(args) -> int:
    a, b = (json.loads(Path(path).read_text()) for path in (args.a, args.b))
    counts, lines = diff_fingerprints(a, b)
    print(f"A: {args.a} ({a['source']})\nB: {args.b} ({b['source']})")
    print("\n".join(lines))
    _print_not_covered()
    return 0 if accepted(counts) else 1


def command_check(args) -> int:
    work_dir = _work_dir(args)
    names = _case_names(args.cases)
    base_source = export_rev(args.base, work_dir)
    base_out = work_dir / f"fp_{base_source.name[4:]}.json"
    if names != list(CASES) or not base_out.exists():
        if run_tree(base_source, base_out, work_dir, names) != 0:
            return 2
    if args.head:
        head_source = export_rev(args.head, work_dir)
        head_out = work_dir / f"fp_{head_source.name[4:]}.json"
    else:
        head_source, head_out = REPO_ROOT, work_dir / "fp_worktree.json"
    if run_tree(head_source, head_out, work_dir, names) != 0:
        return 2
    return command_diff(argparse.Namespace(a=str(base_out), b=str(head_out)))


def command_cases(args) -> int:
    for name, (description, _) in CASES.items():
        print(f"{name:34s} {description}")
    print(f"{len(CASES)} cases; a continued case also records the one-iteration run it resumes from")
    _print_not_covered()
    return 0


def mutated_tree(source: Path, target: Path, old: str, new: str) -> dict[str, int]:
    """Copy ``source``'s ``relax`` to ``target`` with one mutation applied; ``{file: places replaced}``."""
    shutil.copytree(source / "relax", target / "relax", ignore=shutil.ignore_patterns("__pycache__", "*.so", "build"))
    os.symlink(source / "tests", target / "tests")
    first_line = old.strip().splitlines()[0].strip()
    changed = {}
    for path in sorted((target / "relax").rglob("*.py")):
        text = path.read_text()
        if first_line not in text:
            continue
        mutated, replaced = apply_mutation(text, old, new)
        if replaced:
            path.write_text(mutated)
            changed[str(path.relative_to(target))] = replaced
    return changed


def command_selftest(args) -> int:
    work_dir = _work_dir(args)
    source = _resolve_source(args, work_dir)
    known = {mutation[0]: mutation for mutation in MUTATIONS}
    unknown = [name for name in args.mutations if name not in known]
    if unknown:
        raise SystemExit(f"unknown mutation(s): {', '.join(unknown)}")
    selected = [known[name] for name in args.mutations] if args.mutations else list(MUTATIONS)
    clean_out = work_dir / "fp_selftest_clean.json"
    if run_tree(source, clean_out, work_dir, list(CASES), quiet=True) != 0:
        print("the unmutated tree failed to run")
        return 2
    clean = json.loads(clean_out.read_text())

    def one(mutation):
        name, old, new, _, _ = mutation
        target = work_dir / f"mutant_{name}"
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir()
        changed = mutated_tree(source, target, old, new)
        if sum(changed.values()) != 1:
            return name, changed, []
        out = work_dir / f"fp_mutant_{name}.json"
        if run_tree(target, out, work_dir, list(CASES), threads=2, quiet=True) != 0:
            return name, changed, None
        cases = differing_cases(clean, json.loads(out.read_text()))
        shutil.rmtree(target, ignore_errors=True)
        return name, changed, cases

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        outcomes = list(pool.map(one, selected))
    failures = 0
    for (name, _, _, description, expected), (_, changed, cases) in zip(selected, outcomes, strict=True):
        if sum(changed.values()) != 1:
            verdict, detail = "FAIL", (
                f"the mutation's target text is in {sum(changed.values())} places of this tree, not one; update MUTATIONS"
            )
        elif cases is None:
            verdict, detail = "FAIL", "the mutated tree did not run to a fingerprint"
        elif expected and not cases:
            verdict, detail = "FAIL", "not detected"
        elif not expected and not cases:
            verdict, detail = "BLIND", "not detected (known blind spot of the cases)"
        elif not expected:
            verdict, detail = "OK", f"detected in {len(cases)} cases (was a known blind spot; mark it detected)"
        else:
            verdict, detail = "OK", f"detected in {len(cases)} cases, e.g. {cases[0]}"
        failures += verdict == "FAIL"
        print(f"{verdict:5s} {name}: {description} [{', '.join(changed)}] -> {detail}")
    print(f"{len(selected)} mutations, {failures} failed")
    _print_not_covered()
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["_worker"]:
        _worker(argv[1], argv[2], argv[3], argv[4:])
        return 0
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_tree_options(sub):
        tree = sub.add_mutually_exclusive_group()
        tree.add_argument("--source", help="a source tree with relax/ and tests/ (default: this checkout as it is)")
        tree.add_argument("--rev", help="a commit to export with git archive and fingerprint")
        sub.add_argument("--work-dir", help="scratch directory outside the checkout (default: a new temp directory)")

    run = commands.add_parser("run", help="fingerprint one source tree to a JSON file")
    run.add_argument("out")
    run.add_argument("cases", nargs="*")
    add_tree_options(run)
    run.set_defaults(function=command_run)
    diff = commands.add_parser("diff", help="compare two fingerprint files; exit 1 unless only log rows differ")
    diff.add_argument("a")
    diff.add_argument("b")
    diff.set_defaults(function=command_diff)
    check = commands.add_parser("check", help="fingerprint BASE and HEAD (default: the worktree) and diff them")
    check.add_argument("base")
    check.add_argument("head", nargs="?")
    check.add_argument("--cases", nargs="*", default=[])
    check.add_argument("--work-dir")
    check.set_defaults(function=command_check)
    cases = commands.add_parser("cases", help="list the cases")
    cases.set_defaults(function=command_cases)
    selftest = commands.add_parser("selftest", help="check that each deliberately perturbed controller is detected")
    selftest.add_argument("mutations", nargs="*")
    selftest.add_argument("--jobs", type=int, default=4)
    add_tree_options(selftest)
    selftest.set_defaults(function=command_selftest)
    args = parser.parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    sys.exit(main())
