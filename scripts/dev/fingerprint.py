#!/usr/bin/env python
"""Fingerprint what the refinement controller does, to check that a refactor moved code and nothing else.

    pixi run fingerprint run OUT.json [--rev REV | --source DIR] [CASE ...]
    pixi run fingerprint diff A.json B.json          # exit 1 on any difference
    pixi run fingerprint check BASE_REV [HEAD_REV]   # run both (HEAD defaults to the worktree) and diff
    pixi run fingerprint cases                       # the case list, one line each
    pixi run fingerprint selftest [MUTATION ...]     # perturb the controller; every mutation must show

``run`` calls the real ``refine_single_volume`` of one source tree on small CPU cases (8-cubed volume,
four images) and records, per case: the result mapping, every file the run wrote, every checkpoint
snapshot, and one ordered trace of log records and selected calls. Arrays are recorded as module,
dtype, shape and SHA-256 of their bytes, so a changed value, dtype, host/device placement, write or
log line is a difference. The source tree is a worktree (``--source``, default this checkout with
its uncommitted edits) or a ``git archive`` export of any commit (``--rev``); nothing is written into
it. Scratch goes under ``--work-dir`` (default: a new directory under the system temp directory).

The E-step engine cannot run on a CPU, so ``run_dense_k_class_em_adaptive`` is replaced on both sides by
one stand-in whose output is seeded by a hash of every operand it receives: a changed operand anywhere
upstream changes every later array of the run. The comparison is exact because both sides run the same
arithmetic on the same CPU; it is a check for move-only commits, not a merge gate for numerical changes
(``tests/CLAUDE.md``: no bitwise float asserts).

NOT covered (use the GPU test tiers): the real E-step engines and their numbers; local search and the
profile-only return; symmetry other than C1; tomography; multi-shape optics halves; follower-scale
emulation; sealed sampling state; captured RELION projectors; GPU operation order, peak memory and
array lifetimes.
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
    "the real E-step engines and their numbers (a stand-in seeded by its operands replaces the dense adaptive engine)",
    "local search, local sampling and the profile-only return",
    "symmetry other than C1",
    "tomography",
    "multi-shape optics halves",
    "follower-scale emulation, sealed sampling state, captured RELION projectors",
    "GPU operation order, peak memory and array lifetimes",
)
# A log template containing one of these formats a wall time: its template is kept, its text dropped.
TIMING_WORDS = ("wall", "elapsed", "%.2fs", "%.1fs", "%.3fs", "seconds", " in %")
# Rows whose text depends on the node (free host or device memory), not on the source.
NODE_DEPENDENT_TEMPLATES = ("timing", "batch planner memory inputs")
DROPPED_RESULT_KEYS = frozenset({"wall_times", "setup_phase_seconds"})
SECTIONS = ("status", "result", "files", "checkpoints")
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
            flatten(getattr(value, field.name), f"{path}<{type(value).__name__}>/{field.name}", out, scrub=scrub)
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
    logged first is a race, what each worker did in order is not.
    """
    keyed, main_rows = [], 0
    for sequence, (thread_name, row) in enumerate(rows):
        if thread_name == "MainThread":
            keyed.append(((main_rows, "", sequence), row))
            main_rows += 1
        else:
            keyed.append(((main_rows - 1, thread_name, sequence), row))
    return [row for _, row in sorted(keyed, key=lambda item: item[0])]


def diff_fingerprints(a: dict, b: dict, *, shown_per_section: int = 12) -> tuple[int, list[str]]:
    """``(number of differences, report lines)`` between two fingerprints written by ``run``."""
    lines, count = [], 0
    cases_a, cases_b = a["cases"], b["cases"]
    for name in sorted(set(cases_a) ^ set(cases_b)):
        count += 1
        lines.append(f"CASE {name}: only in {'A' if name in cases_a else 'B'}")
    trace_lines: dict[str, list[str]] = {}
    totals = dict.fromkeys((*SECTIONS, "trace"), 0)
    for name in sorted(set(cases_a) & set(cases_b)):
        case_a, case_b = cases_a[name], cases_b[name]
        for section in SECTIONS:
            flat_a, flat_b = case_a[section], case_b[section]
            totals[section] += len(flat_a)
            changed = [key for key in sorted(set(flat_a) | set(flat_b)) if flat_a.get(key) != flat_b.get(key)]
            count += len(changed)
            for key in changed[:shown_per_section]:
                lines.append(
                    f"DIFF {name} {section} {key}: {flat_a.get(key, '<absent>')} != {flat_b.get(key, '<absent>')}"
                )
            if len(changed) > shown_per_section:
                lines.append(f"DIFF {name} {section}: {len(changed) - shown_per_section} more")
        text_a = [" | ".join(row) for row in case_a["trace"]]
        text_b = [" | ".join(row) for row in case_b["trace"]]
        totals["trace"] += len(text_a)
        for line in difflib.unified_diff(text_a, text_b, lineterm="", n=0):
            if line[:1] in "+-" and line[:3] not in ("+++", "---"):
                count += 1
                trace_lines.setdefault(line, []).append(name)
    for line, names in trace_lines.items():
        lines.append(f"TRACE [{len(names)} cases, e.g. {names[0]}] {line[:400]}")
    lines.append(
        f"{len(set(cases_a) & set(cases_b))} cases compared; {count} differences; leaves compared: "
        + ", ".join(f"{section} {totals[section]}" for section in SECTIONS)
        + f"; trace rows {totals['trace']}"
    )
    return count, lines


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
    return cases


CASES = _cases()


# A deliberately wrong controller for the self-test: (name, old lines, new lines, what breaks, detected).
# The first thirteen are the set the refactor workers used; a False marks a known blind spot of the cases.
# A target must occur in exactly one place of the tree: when a statement is rewritten, update its entry.
MUTATIONS = (
    ("final_class_prior_not_normalised", "_prior_k = normalize_class_direction_prior(_prior_k, n_classes, dtype=scoring_dtype)", "pass",
     "the final-pass replay prior of a class run is installed unnormalised", True),
    ("resumed_prior_order_dropped", "DirectionPrior(prior.values, order)", "DirectionPrior(prior.values, None)",
     "a resumed direction prior loses its saved HEALPix order", True),
    ("state_swap_prior_order_dropped", 'recovar_snapshot["direction_prior_order_per_half"][k],', "None,",
     "the state-swap probe restores a prior without its order", True),
    ("replay_prior_reversed", "direction_priors[_half_idx] = DirectionPrior(\nprior_k,", "direction_priors[_half_idx] = DirectionPrior(\nprior_k[::-1],",
     "a replayed K1 prior is installed reversed", True),
    ("snapshot_prior_order_dropped", "direction_prior_order = [p.healpix_order for p in direction_priors]", "direction_prior_order = [None for p in direction_priors]",
     "the checkpoint drops the prior orders", True),
    ("star_class_prior_reversed", "_relion_direction_prior, n_classes,\n),", "_relion_direction_prior[::-1], n_classes,\n),",
     "a STAR class prior is installed reversed", True),
    ("replay_reads_k1_layout", "runtime_dtype = dtype\nk_class_enabled = n_classes > 1", "runtime_dtype = dtype\nk_class_enabled = False",
     "the per-iteration replay treats every run as K1", True),
    ("intermediates_one_class", "range(n_classes) if n_classes > 1 else (None,)", "(None,)",
     "the intermediates writer saves one class", True),
    ("shell_cutoff_off_by_one", "first_unavailable_shell = min(truncated.shape[-1], int(current_size) // 2 + 1)", "first_unavailable_shell = min(truncated.shape[-1], int(current_size) // 2)",
     "the current-size shell cut-off moves by one", True),
    ("noise_rows_doubled", "return jnp.asarray(noise.make_radial_noise(shell_profile, image_shape)).reshape(-1)", "return jnp.asarray(noise.make_radial_noise(shell_profile, image_shape)).reshape(-1) * 2",
     "the shells-to-pixel-row noise expansion is doubled", True),
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
    ("trial_grid_step", "translation_step=float(translation_step),\nrandom_perturbation=random_perturbation,\nangular_sampling_deg=angsamp_deg,", "translation_step=2 * float(translation_step),\nrandom_perturbation=random_perturbation,\nangular_sampling_deg=angsamp_deg,",
     "the perturbed trial grid shifts translations by twice the step", True),
    ("pass1_angular_sampling", "sampling.relion_angular_sampling_deg(adaptive_pass1_order, adaptive_oversampling=0),", "sampling.relion_angular_sampling_deg(adaptive_pass1_order, adaptive_oversampling=1),",
     "the pass-1 coarse rotations are perturbed by the oversampled step", True),
    ("projector_not_reused", "reusable=shared_projector_half1 if half.index == 0 else None,", "reusable=None,",
     "half 1's scoring projector is rebuilt instead of reusing the accuracy projector", True),
    ("significance_not_combined", "significance.combine()", "pass",
     "the halves' significant-sample counts are never combined", True),
    ("numbered_result_claims_final_pass", '"final_all_data_ran": False,\n**history.to_dict(),', '"final_all_data_ran": True,\n**history.to_dict(),',
     "the result of a run without a final pass says the final pass ran", True),
)


# ----------------------------------------------------------------------------- the worker (imports the tree)


def _worker(source: str, out_path: str, tmp_root: str, names: list[str]) -> None:
    """Run the cases against ``source`` in this process; the caller has set the CPU-only environment."""
    import importlib
    import logging
    import threading

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

    def scrub(text):
        return tmp_pattern.sub(TMP_TOKEN, text)

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

    def run_case(n_classes, join, *, max_iter=4, converge_after=2, dump=True, cc=False, oversampling=0, env=None,
                 writer=False, skip_final=False, continued=False, resume=None, perturb=None, init_prior=None,
                 iter_prior=None, final_prior=None, star_prior=None, star_optimiser=False, swap=None, frozen=False,
                 seed=False, orders=None, overlap=False, accuracy=False, init_order=2, replay_max_iter=None):
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
        if adaptive_fields:
            extra["adaptive"] = refinement_options.AdaptiveOptions(**adaptive_fields)
        if overlap:
            extra["overlap"] = refinement_options.HalfOverlapOptions(overlap_halves=True)
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
            extra["replay"] = refinement_options.ReplayState(**replay_fields)
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
        if dump:
            debug_fields["save_intermediates_dir"] = dump_dir
        if debug_fields:
            extra["debug"] = refinement_options.EngineDebugOptions(**debug_fields)
        if perturb is not None:
            parity["perturb_factor"] = perturb
        if cc:
            parity.update(emulate_relion_firstiter_cc=True, relion_firstiter_ini_high_angstrom=8.0)
        error, result = None, {}
        try:
            result = iteration_loop.refine_single_volume(
                halves,
                init_volume,
                jnp.ones(fixtures.IMAGE_SIZE, dtype=jnp.float32),
                jnp.ones(fixtures.VOLUME_SIZE, dtype=jnp.float32) * 100.0,
                translations,
                options=refinement_options.RefinementOptions(
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
                ),
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

    importlib.invalidate_caches()
    results = {}
    for name in names:
        case = run_case(**CASES[name][1])
        case.pop("snapshots", None)
        first_pass = case.pop("first_pass", None)
        if first_pass is not None:  # the one-iteration run a continued case resumes from is a case of its own
            results[name + "/first_pass"] = first_pass
        results[name] = case
        print(
            f"{name}: error={case['status']['error']} final={case['status'].get('final_all_data_ran')} "
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
    count, lines = diff_fingerprints(a, b)
    print(f"A: {args.a} ({a['source']})\nB: {args.b} ({b['source']})")
    print("\n".join(lines))
    _print_not_covered()
    return 1 if count else 0


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
    diff = commands.add_parser("diff", help="compare two fingerprint files; exit 1 on any difference")
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
