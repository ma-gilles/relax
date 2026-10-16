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
#   smell                     scripts/dev/smell_check.py against its baseline; a warning, never a failure
#   ruff [BASE]               ruff findings of relax, tests and scripts that BASE lacks: none
#   arity [BASE]              scripts/dev/call_arity_check.py findings (rule 18) that BASE lacks: none
#   diffcheck [BASE]          git diff --check BASE HEAD: clean
#   anchors                   scripts/dev/check_mutation_anchors.py: no stale selftest mutation
#   ceilings                  scripts/report_refinement_structure.py --check of the ceilings file: none above
#                             its slack
#   static [BASE]             ruff, arity, diffcheck, anchors, ceilings (a few minutes)
#   extra BASE FILE...        the pytest FILEs on the head and on a snapshot of BASE: the failure lists must be
#                             identical (a file BASE lacks runs on the head only)
#   all [BASE [FILE...]]      fp, selftest, guard, tests head HEAD, clean head, static, smell, then extra when
#                             FILEs are given (about 40 minutes; more with a module's test directories)
# Each step prints one summary line per check and exits nonzero on failure ("tests" does not: "clean" or
# "compare" decides; "smell" always exits 0 and lists its new findings in its log). New findings are listed
# under their line. Run it from the worktree, or as a Slurm job from there
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
  echo "$FPN check vs ${1:-origin/main}: rc=$rc $(grep -E 'cases compared|only log rows differ|controller inputs added|controller inputs retired|controller inputs moved|controller inputs redefaulted' "$S/logs/${FPN}_check_$T.txt" | paste -sd ' ')"; return $rc
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
step_smell() {
  cd "$W" && "$PY" scripts/dev/smell_check.py > "$S/logs/smell_$T.txt" 2>&1
  echo "$(tail -1 "$S/logs/smell_$T.txt") -> $S/logs/smell_$T.txt (warning only)"; return 0
}
# A git archive of BASE in $S/base_src_$T (no .git: the checkers read every file of the tree), for the
# steps that diff a head report against a base report; "static" extracts it once for all of them.
base_tree() {
  local rev; rev=$(git -C "$W" rev-parse "${1:-origin/main}^{commit}") || return 2; B=$S/base_src_$T
  [ "$(cat "$B/.gate_rev" 2>/dev/null)" = "$rev" ] && return 0
  rm -rf "$B"; mkdir -p "$B" && git -C "$W" archive "$rev" | tar -x -C "$B" && echo "$rev" > "$B/.gate_rev"
}
# Lines of the head report $2 that the base report $1 lacks, under the summary line $3; fails when there are any.
new_findings() {
  local new; new=$(comm -13 "$1" "$2")
  echo "$3: $(printf '%s' "$new" | grep -c .) new -> $2"; [ -z "$new" ] || { echo "$new"; return 1; }
}
ruff_findings() { "$PY" -m ruff check relax tests scripts --output-format concise 2>/dev/null | sed "s/:[0-9]*:[0-9]*:/:/" | grep -v "^Found\|fixable\|^All checks" | sort; }
step_ruff() {
  local base=${1:-origin/main}; base_tree "$base" || return 2
  (cd "$B" && ruff_findings) > "$S/logs/ruff_base_$T.txt"; (cd "$W" && ruff_findings) > "$S/logs/ruff_head_$T.txt"
  new_findings "$S/logs/ruff_base_$T.txt" "$S/logs/ruff_head_$T.txt" "ruff findings $base lacks"
}
step_arity() {
  # The head's checker reads both trees, so a checker change does not show as findings.
  local base=${1:-origin/main}; base_tree "$base" || return 2
  "$PY" "$W/scripts/dev/call_arity_check.py" "$B" | sort > "$S/logs/arity_base_$T.txt"
  "$PY" "$W/scripts/dev/call_arity_check.py" "$W" | sort > "$S/logs/arity_head_$T.txt"
  new_findings "$S/logs/arity_base_$T.txt" "$S/logs/arity_head_$T.txt" "call arity findings $base lacks"
}
step_diffcheck() {
  local base=${1:-origin/main}
  git -C "$W" diff --check "$base" HEAD > "$S/logs/diffcheck_$T.txt" 2>&1; rc=$?
  echo "git diff --check $base HEAD: rc=$rc $(grep -vc '^+' "$S/logs/diffcheck_$T.txt") findings"; [ $rc = 0 ] || { cat "$S/logs/diffcheck_$T.txt"; return 1; }
}
step_anchors() {
  cd "$W" && PYTHONPATH=$W "$PY" scripts/dev/check_mutation_anchors.py > "$S/logs/anchors_$T.txt" 2>&1; rc=$?
  echo "$(tail -1 "$S/logs/anchors_$T.txt"): rc=$rc"; [ $rc = 0 ] || { grep '^STALE' "$S/logs/anchors_$T.txt"; return 1; }
}
step_ceilings() {
  # A metric above its ceiling within the slack is a warning (named in the report, procedure section 4).
  cd "$W" && PYTHONPATH=$W "$PY" scripts/report_refinement_structure.py --check docs/development/refinement_structure_metrics.json \
    > "$S/logs/ceilings_metrics_$T.json" 2> "$S/logs/ceilings_$T.txt"; rc=$?
  echo "structure ceilings: rc=$rc $(grep -c '^warning: ' "$S/logs/ceilings_$T.txt") warnings $(grep -vc '^warning: ' "$S/logs/ceilings_$T.txt") above the slack -> $S/logs/ceilings_$T.txt"
  grep '^warning: ' "$S/logs/ceilings_$T.txt"; [ $rc = 0 ] || { grep -v '^warning: ' "$S/logs/ceilings_$T.txt"; return 1; }
}
step_static() {
  local base=${1:-origin/main} st=0
  base_tree "$base" || { echo "base tree $base: cannot extract"; return 2; }
  step_ruff "$base" || st=1; step_arity "$base" || st=1; step_diffcheck "$base" || st=1
  step_anchors || st=1; step_ceilings || st=1; rm -rf "$S/base_src_$T"; return $st
}
step_extra() {
  local base=${1:?extra needs a base and test files}; shift; [ $# -gt 0 ] || { echo "extra: no test files"; return 2; }
  local f hx=() bx=(); for f in "$@"; do [ -f "$W/$f" ] && hx+=("$f"); done
  [ ${#hx[@]} -gt 0 ] || { echo "extra: none of $* exists on the head"; return 1; }
  cd "$W" && PYTHONPATH=$W "$PY" -m pytest "${hx[@]}" -q -p no:cacheprovider --basetemp="$S/pytest_tmp_extra_head_$T" -rfE \
    > "$S/logs/extra_head_$T.log" 2>&1
  grep -E "^(FAILED|ERROR) " "$S/logs/extra_head_$T.log" | sed 's/ - .*//' | sort > "$S/logs/fail_extra_head_$T.txt"
  base_tree "$base" || return 2; [ -e "$W/.pixi" ] && ln -sfn "$(readlink -f "$W/.pixi")" "$B/.pixi"
  for f in "${hx[@]}"; do [ -f "$B/$f" ] && bx+=("$f"); done
  if [ ${#bx[@]} -gt 0 ]; then
    (cd "$B" && PYTHONPATH=$B "$PY" -m pytest "${bx[@]}" -q -p no:cacheprovider --basetemp="$S/pytest_tmp_extra_base_$T" -rfE \
      > "$S/logs/extra_base_$T.log" 2>&1)
  else echo "no extra file exists on $base" > "$S/logs/extra_base_$T.log"; fi
  grep -E "^(FAILED|ERROR) " "$S/logs/extra_base_$T.log" | sed 's/ - .*//' | sort > "$S/logs/fail_extra_base_$T.txt"
  rm -rf "$S/base_src_$T"
  echo "extra tests head: $(tail -1 "$S/logs/extra_head_$T.log") | base: $(tail -1 "$S/logs/extra_base_$T.log")"
  if cmp -s "$S/logs/fail_extra_base_$T.txt" "$S/logs/fail_extra_head_$T.txt"; then
    echo "extra failure lists base / head: identical ($(wc -l < "$S/logs/fail_extra_head_$T.txt") failed on both)"; return 0; fi
  echo "extra failure lists base / head: DIFFER"; diff "$S/logs/fail_extra_base_$T.txt" "$S/logs/fail_extra_head_$T.txt"; return 1
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
smell) step_smell;;
ruff) step_ruff "${2:-}";;
arity) step_arity "${2:-}";;
diffcheck) step_diffcheck "${2:-}";;
anchors) step_anchors;;
ceilings) step_ceilings;;
static) step_static "${2:-}";;
extra) shift; step_extra "$@";;
all) status=0; base=${2:-origin/main}; shift $(($# < 2 ? $# : 2))
  step_fp "$base" || status=1; step_selftest || status=1; step_guard || status=1
  step_tests head HEAD || status=1; step_clean head || status=1
  step_static "$base" || status=1; step_smell
  [ $# = 0 ] || step_extra "$base" "$@" || status=1
  exit $status;;
*) sed -n '6,34p' "$0"; exit 2;;
esac
