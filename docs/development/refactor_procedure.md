# How to run one refactor slice

A checklist for an agent who refactors a module of this repository without a supervisor. The rules are
[refactor_rules.md](refactor_rules.md) (the code rules); this page is the procedure around them, and
[module_template.md](module_template.md) is the shape a slice works towards. A *slice* is one reviewable
change of structure with no change of behaviour (rule 1). The scripts are in `scripts/dev/`; each takes its scratch
directory from `REFACTOR_SCRATCH` (a directory outside the checkout, under a shared scratch root).

## 1. Choose the slice: measure first

- [ ] Measure the stretch the slice would change, with a tool, before writing any code:
      `python scripts/report_refinement_structure.py --format markdown` (spans, parameter counts, the list of
      functions with ten or more parameters); for a controller, a line classifier such as
      `python scripts/dev/loop_kinds.py [--calls]` (lines per mode, shared lines by kind: installs, history
      writes, logs, wide calls). For another package, copy the report script and change its root.
- [ ] Write down the concrete benefit: an invariant isolated, a decision moved to its owner, fewer facts a
      caller must hold. Counts (lines, parameters, mode tests) are supporting evidence, not a required
      reduction. If you cannot name the benefit, the slice is not defined yet.
- [ ] Inventory before removal. For every name you will move, rename or delete: all callers, and
      `grep -rn '<name>' tests/ scripts/ docs/` (monkeypatch strings, `inspect.getsource` pins, captured
      keyword arguments, function-local names that tests execute as text). A test that reads or executes
      source text is converted to a behavioural test when the code it pins changes, keeping its assertions
      (rule 13).
- [ ] List the installs, history writes and release points (`del`, `x = None`) in the stretch. Installs
      and history writes stay statements of the controller (rule 9); if the slice needs one to leave it, it
      is a different slice. For each release, trace the lifetime it serves (rule 3): releasing an unused
      buffer earlier is allowed and stated with its peak-memory effect; extending a lifetime is not. Log
      records may move (rule 2).

## 2. Coverage before the first edit

Choose the evidence by the kind of module (rule 14). The stand-in-engine fingerprint is the controller's
harness, not a requirement for every module: a kernel is protected by small independent references and
device tests, a parser or option resolver by precedence, rejection and round-trip tests.

- [ ] Controller: `pixi run fingerprint cases`: does a case execute the stretch, in every mode and with the
      non-default options it reads (rule 13)? If not, add cases first, as their own commit, with a mutation
      in the selftest that proves the new case sees the stretch.
- [ ] If a controller has no fingerprint at all, build the harness before the refactor. It needs: a stand-in
      for whatever cannot run on a CPU (the GPU engine), whose output is seeded by a hash of every operand
      it receives; small CPU cases (seconds each) that reach every branch you will touch; an ordered trace of
      log records and selected calls; hashes of every result, file and checkpoint; and a selftest of
      deliberate mutations that must each show. `scripts/dev/fingerprint.py` is the model.
- [ ] Write down what is still not covered. That list goes into the report (rule 14).

## 3. Work: one idea per commit

- [ ] A new branch from `origin/main`, in a worktree under a shared scratch root, never under `/home`.
- [ ] One idea per commit. Before each commit, from the checkout:
      `REFACTOR_SCRATCH=<dir> scripts/dev/refactor_verify.sh <the unit files that name the module>`.
      It prints one line per check: the fingerprint of the worktree against `HEAD` (results, files and
      checkpoints: 0 differences; trace differences confined to log rows are allowed and are reported as
      such; the script exits nonzero on any difference, so read `logs/fp_vs_parent.txt`); `ruff`
      findings that `origin/main` lacks (none); `git diff --check`; `scripts/dev/check_mutation_anchors.py`
      (every selftest mutation still finds its anchor in the source); and `pytest` on
      `tests/unit/test_refinement_structure_metrics.py` plus the files given.
- [ ] The commit message says what moved, what did not, the benefit, the supporting numbers, and any
      log-only trace differences.

## 4. Gate at the end of the slice

On a frozen copy of the head (`git worktree add --detach <dir> <sha>`, with the checkout's `.pixi` linked
into it), from that directory: `REFACTOR_SCRATCH=<dir> scripts/dev/refactor_gate.sh <step>`, or all steps as
one Slurm CPU job, `sbatch --account=<account> --export=ALL scripts/dev/refactor_gate.sh all`. Use the login
node only when `uptime` shows a load under 20. Each step prints one summary line. The steps:

- [ ] `fp`: `fingerprint.py check origin/main`: results, files and checkpoints 0 differences; trace
      differences confined to log rows are allowed and are reported as such. After any rebase, run it and
      the lint again: a clean rebase can still leave a dangling name.
- [ ] `selftest`: `fingerprint.py selftest --jobs 4`, every mutation detected.
- [ ] `guard`: `bash scripts/run_em_fast_guard.sh` and `python scripts/check_agent_guides.py`.
- [ ] `tests base <base rev>`, `tests head HEAD`, `compare base head`: the CPU unit list
      (`scripts/dev/refactor_cpu_unit_list.txt`, 166 files) on a snapshot of each revision. The two lists of
      failed tests (`logs/fail_base.txt`, `logs/fail_head.txt`) must be identical. Some tests fail on both
      sides on a CPU node; compare the lists, not the counts.
- [ ] Repeat the search of all of `tests/` and `scripts/` for every moved or removed name: the CPU list does
      not run the GPU unit files.
- [ ] GPU tiers on the final head, same GPU model for every arm:
      `python scripts/run_test_tier.py submit smoke --where slurm --gpu-model h100 --run-root <dir>`, and the
      same with `medium`. Current practice, not an owner ruling: smoke alone is enough for docs, tests,
      scripts and moves the fingerprint covers completely; medium is needed when the slice touches a route
      the fingerprint does not reach, moves a name a GPU test may patch, or changes scoring, reconstruction,
      noise, priors or convergence.
      Each run writes `RECEIPT.json` in its run root.
- [ ] Ceilings, as the last commit:
      `python scripts/report_refinement_structure.py --lower-ceilings docs/development/refinement_structure_metrics.json`,
      and the span quoted in the module's `AGENTS.md`. The ceilings are review signals with slack (owner
      ruling, 2026-10-05): `--check` warns when a metric is above its ceiling but within the slack, and
      fails above the slack. Name a warning in the report with the reason for the growth. Never raise a
      ceiling to make a refactor pass; a requested feature may raise one by hand, with the value and the
      reason in its commit.

## 5. When the right result is "do not build it"

Stop and report instead of building when: the slice has no concrete benefit; the operation would need ten
or more parameters, or gets under ten only through a record whose fields are related by timing alone; it
would be a step that only renames its arguments (`dump_numbered_iteration` keeps 18 parameters because each
maps to one field of one record, so taking records would remove none); it would hide an install or a
history write, or extend a buffer's lifetime; the measured cost exceeds the measured benefit (two copies of
a sequence for one flag); or the stretch has no coverage and you cannot add it. Then do the smaller change
that still reads better, or none. A measured "not worth it" is a result.

## 6. Report and hand over

The report has: the head and its commits; the before and after of the changed stretch, quoted, 30 to 40 lines
each; the benefit, with span, parameter and mode-test numbers before and after as support; each check with
its job ID and outcome, separating what was read, executed, numerically compared and performance-measured
(rule 14); any ceiling warning; what was not verified and why; what was declined and why; and a plain answer
to "does it read better", with the cost named.

Keep `HANDOFF.json` in the scratch directory current from the first commit: task and constraints; worktree,
branch, base and head; the decision taken and the reason; the numbers; job IDs with what each checks; paths
of logs and receipts; what is next; what is not covered. A fresh agent must be able to continue from that
file alone.
