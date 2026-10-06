#!/bin/bash
# Per-commit check of a move-only refactor: the worktree (with its uncommitted edits) against HEAD.
# usage: REFACTOR_SCRATCH=<dir outside the checkout> scripts/dev/refactor_verify.sh [pytest arguments: test files, -k ...]
# Run it from the checkout. Prints one line per check: fingerprint of the worktree against HEAD (no difference
# in outputs or non-log trace rows; a log-only difference passes under rule 2), ruff findings the base lacks,
# git diff --check, stale selftest mutation anchors, and pytest on the module's structure tests plus the
# arguments. REFACTOR_MODULE (default refinement) selects the harness and the structure tests:
# scripts/dev/refactor_module.sh. REFACTOR_BASE (default origin/main) is the revision whose ruff findings are
# the baseline; REFACTOR_PYTHON defaults to the checkout's pixi environment. Logs go to $REFACTOR_SCRATCH/logs.
# Where it runs: a Slurm CPU job, or the login node only with OMP_NUM_THREADS<=4, the module's unit files as
# arguments, and a load (uptime) under 20; the full lists run in the gate (refactor_gate.sh).
set -u
S=${REFACTOR_SCRATCH:?set REFACTOR_SCRATCH to a scratch directory outside the checkout}
W=$(git rev-parse --show-toplevel) || exit 2
PY=${REFACTOR_PYTHON:-$W/.pixi/envs/default/bin/python}
BASE=$(git -C "$W" rev-parse "${REFACTOR_BASE:-origin/main}") || exit 2
# shellcheck source=scripts/dev/refactor_module.sh
. "$W/scripts/dev/refactor_module.sh"
mkdir -p "$S/logs" "$S/fp"
cd "$W" || exit 2
unset PYTHONHOME CONDA_PREFIX VIRTUAL_ENV RELAX_TEST_RECEIPTS
export PYTHONPATH=$W PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} XLA_PYTHON_CLIENT_PREALLOCATE=false
status=0
ruff_findings() { "$PY" -m ruff check relax tests scripts --output-format concise 2>/dev/null | sed "s/:[0-9]*:[0-9]*:/:/" | grep -v "^Found\|fixable\|^All checks" | sort; }

H=$(git rev-parse HEAD)
parent=$S/fp/${FPN}_${H:0:12}.json
[ -f "$parent" ] || "$PY" "$FP" run "$parent" --rev HEAD --work-dir "$S/fp" > "$S/logs/${FPN}_run_${H:0:7}.txt" 2>&1
"$PY" "$FP" run "$S/fp/${FPN}_worktree.json" --work-dir "$S/fp" > "$S/logs/${FPN}_run_worktree.txt" 2>&1 || tail -5 "$S/logs/${FPN}_run_worktree.txt"
"$PY" "$FP" diff "$parent" "$S/fp/${FPN}_worktree.json" > "$S/logs/${FPN}_vs_parent.txt" 2>&1; rc=$?
echo "$FPN worktree vs ${H:0:7}: rc=$rc $(grep -E 'cases compared|only log rows differ|controller inputs added|controller inputs retired' "$S/logs/${FPN}_vs_parent.txt" | paste -sd ' ')"; [ $rc = 0 ] || status=1

if [ ! -f "$S/logs/ruff_base_${BASE:0:12}.txt" ]; then
  rm -rf "$S/ruff_base_src"; mkdir -p "$S/ruff_base_src"; git archive "$BASE" | tar -x -C "$S/ruff_base_src"
  (cd "$S/ruff_base_src" && ruff_findings) > "$S/logs/ruff_base_${BASE:0:12}.txt"; rm -rf "$S/ruff_base_src"
fi
ruff_findings > "$S/logs/ruff_head.txt"
new=$(comm -13 "$S/logs/ruff_base_${BASE:0:12}.txt" "$S/logs/ruff_head.txt")
echo "ruff findings ${BASE:0:7} lacks: $(printf '%s' "$new" | grep -c .)"; [ -z "$new" ] || { echo "$new"; status=1; }

if git diff --check; then echo "git diff --check: clean"; else status=1; fi
"$PY" scripts/dev/check_mutation_anchors.py | tail -1; [ "${PIPESTATUS[0]}" = 0 ] || status=1
# shellcheck disable=SC2086
"$PY" -m pytest $STRUCTURE_TESTS "$@" -q -p no:cacheprovider --basetemp="$S/pytest_tmp_verify" > "$S/logs/verify_pytest.txt" 2>&1; rc=$?
echo "pytest: rc=$rc $(tail -1 "$S/logs/verify_pytest.txt")"; [ $rc = 0 ] || { grep -E "^(FAILED|ERROR) " "$S/logs/verify_pytest.txt"; status=1; }
exit $status
