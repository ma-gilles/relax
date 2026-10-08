#!/bin/bash
#SBATCH --partition=cpu,cryoem
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# CPU gate of a refactor slice, on a frozen worktree of the head (git worktree add --detach <dir> <sha>).
# usage: REFACTOR_SCRATCH=<dir> [REFACTOR_WORKTREE=<frozen worktree>] scripts/dev/refactor_gate.sh <step>
#   fp [BASE]                 fingerprint check of the worktree against BASE (default origin/main)
#   selftest                  every fingerprint mutation must be detected
#   guard                     the fast guard and the agent-guide check
#   tests <label> <rev>       the CPU unit list and the module's test directories on a detached git worktree of
#                             <rev> (the launcher tests need git); writes logs/fail_<label>.txt
#   clean <label>             the failure list of a tests run must be empty
#   compare <label> <label>   the two failure lists must be identical (to compare a base by hand)
#   all [BASE]                fp, selftest, guard, tests head HEAD, clean head (about 40 minutes; more with a
#                             module's test directories)
# Each step prints one summary line and exits nonzero on failure ("tests" does not: "clean" or "compare"
# decides). Run it from the worktree, or as a Slurm job from there
# (sbatch --account=<account> --export=ALL scripts/dev/refactor_gate.sh all). REFACTOR_WORKTREE defaults to
# the checkout of the current directory, REFACTOR_PYTHON to its pixi environment, REFACTOR_TAG (a suffix
# for log names, so two gates can share one scratch directory) to the short head. REFACTOR_MODULE (default
# refinement) selects the module's harness and test directories: scripts/dev/refactor_module.sh.
# Logs: $REFACTOR_SCRATCH/logs.
set -u
S=${REFACTOR_SCRATCH:?set REFACTOR_SCRATCH to a scratch directory outside the checkout}
W=${REFACTOR_WORKTREE:-$(git rev-parse --show-toplevel)} || exit 2
PY=${REFACTOR_PYTHON:-$W/.pixi/envs/default/bin/python}
T=${REFACTOR_TAG:-$(git -C "$W" rev-parse --short HEAD)}
# shellcheck source=scripts/dev/refactor_module.sh
. "$W/scripts/dev/refactor_module.sh"
mkdir -p "$S/logs"
unset PYTHONHOME CONDA_PREFIX VIRTUAL_ENV RELAX_TEST_RECEIPTS
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 XLA_PYTHON_CLIENT_PREALLOCATE=false

step_fp() {
  cd "$W" && PYTHONPATH=$W "$PY" "$FP" check "${1:-origin/main}" --work-dir "$S/${FPN}_gate_$T" > "$S/logs/${FPN}_check_$T.txt" 2>&1; rc=$?
  echo "$FPN check vs ${1:-origin/main}: rc=$rc $(grep -E 'cases compared|only log rows differ|controller inputs added|controller inputs retired' "$S/logs/${FPN}_check_$T.txt" | paste -sd ' ')"; return $rc
}
step_selftest() {
  cd "$W" && PYTHONPATH=$W "$PY" "$FP" selftest --jobs 4 --work-dir "$S/${FPN}_selftest_$T" > "$S/logs/${FPN}_selftest_$T.txt" 2>&1; rc=$?
  echo "$FPN selftest: rc=$rc $(grep -E '^[0-9]+ mutations, [0-9]+ failed' "$S/logs/${FPN}_selftest_$T.txt")"; return $rc
}
step_guard() {
  cd "$W" && PYTHONPATH=$W JAX_COMPILATION_CACHE_DIR=$S/jax_cache_guard_$T RECOVAR_JAX_CACHE_DIR=$S/recovar_jax_cache_guard_$T \
    PATH=$(dirname "$PY"):$PATH bash scripts/run_em_fast_guard.sh > "$S/logs/fast_guard_$T.log" 2>&1; rc=$?
  echo "fast guard: rc=$rc $(tail -1 "$S/logs/fast_guard_$T.log")"
  PYTHONPATH=$W "$PY" scripts/check_agent_guides.py > "$S/logs/agent_guides_$T.log" 2>&1; rc2=$?
  echo "agent guides: rc=$rc2 $(tail -1 "$S/logs/agent_guides_$T.log")"; return $((rc + rc2))
}
step_tests() {
  local label=${1:?tests needs a label} rev=${2:?tests needs a revision} src=$S/snap_$1
  # A real checkout: the launcher and guard tests run git (provenance, merge-base) in it.
  git -C "$W" worktree remove --force "$src" 2>/dev/null; rm -rf "$src"; git -C "$W" worktree prune
  git -C "$W" worktree add --detach "$src" "$rev" > /dev/null 2>&1 || return 2
  [ -e "$W/.pixi" ] && ln -s "$(readlink -f "$W/.pixi")" "$src/.pixi"
  cd "$src" || return 2
  local files dirs=""; for d in $MODULE_TESTS; do [ -d "$d" ] && dirs="$dirs $d"; done
  # shellcheck disable=SC2086
  files=$( { grep -v '^#' "$W/scripts/dev/refactor_cpu_unit_list.txt"; [ -z "$dirs" ] || find $dirs -name 'test_*.py'; } \
    | while read -r f; do [ -f "$f" ] && echo "$f"; done | sort -u)
  # shellcheck disable=SC2086
  # The robustness-matrix and completion launchers need an installed pixi Python as their base environment.
  EM_COMPLETION_PIXI_PY=$PY EM_K1_MATRIX_PIXI_PY=$PY EM_KCLASS_MATRIX_PIXI_PY=$PY \
    PYTHONPATH=$src JAX_COMPILATION_CACHE_DIR=$S/jax_cache_tests_$label RECOVAR_JAX_CACHE_DIR=$S/recovar_jax_cache_tests_$label \
    "$PY" -m pytest $files -p no:cacheprovider --basetemp="$S/pytest_tmp_$label" -rfEs -v > "$S/logs/tests_$label.log" 2>&1; rc=$?
  grep -E "^(FAILED|ERROR) " "$S/logs/tests_$label.log" | sed 's/ - .*//' | sort > "$S/logs/fail_$label.txt"
  echo "tests $label ($rev, $(echo "$files" | wc -l) files, module $REFACTOR_MODULE): rc=$rc $(tail -1 "$S/logs/tests_$label.log") -> $S/logs/fail_$label.txt"
  [ $rc -le 1 ]
}
step_clean() {
  local f=$S/logs/fail_${1:?clean needs a label}.txt
  [ -f "$f" ] || { echo "failure list $1: missing"; return 1; }
  if [ ! -s "$f" ]; then echo "failure list $1: empty"; return 0; fi
  echo "failure list $1: $(wc -l < "$f") failed"; cat "$f"; return 1
}
step_compare() {
  local a=$S/logs/fail_${1:?compare needs two labels}.txt b=$S/logs/fail_${2:?compare needs two labels}.txt
  if cmp -s "$a" "$b"; then echo "failure lists $1 / $2: identical ($(wc -l < "$a") failed on both)"; return 0; fi
  echo "failure lists $1 / $2: DIFFER"; diff "$a" "$b"; return 1
}

case ${1:-} in
fp) step_fp "${2:-}";;
selftest) step_selftest;;
guard) step_guard;;
tests) step_tests "${2:-}" "${3:-}";;
clean) step_clean "${2:-}";;
compare) step_compare "${2:-}" "${3:-}";;
all) status=0; base=${2:-origin/main}
  step_fp "$base" || status=1; step_selftest || status=1; step_guard || status=1
  step_tests head HEAD || status=1; step_clean head || status=1
  exit $status;;
*) sed -n '6,23p' "$0"; exit 2;;
esac
