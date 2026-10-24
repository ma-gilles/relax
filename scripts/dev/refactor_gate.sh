#!/bin/bash
#SBATCH --partition=cpu,cryoem
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# CPU gate of a refactor slice, on a frozen worktree of the head (git worktree add --detach <dir> <sha>).
# usage: REFACTOR_SCRATCH=<dir> [REFACTOR_WORKTREE=<frozen worktree>] scripts/dev/refactor_gate.sh <step>
#   fp [BASE]                 fingerprint check of the worktree against BASE (default origin/main)
#   selftest                  every fingerprint mutation must be detected, except the harness's EXPECTED_BLIND ones
#   guard                     the fast guard and the agent-guide check
#   tests <label> <rev>       the CPU unit list and every module's test directories on a detached git worktree of
#                             <rev> (the launcher tests need git); writes logs/fail_<label>.txt. A listed file
#                             the tree lacks, or a pytest run that ends other than passed/failed (a crash, an OOM
#                             kill, a timeout, a collection abort), fails the step
#   clean <label>             the failure list of a tests run must be empty
#   counts [BASE]             tests base BASE, then the head passes at least as many tests as the base (junit counts)
#                             unless REFACTOR_ACCEPT_HARNESS is the head's sha
#   compare <label> <label>   the two failure lists must be identical (to compare a base by hand)
#   smell                     scripts/dev/smell_check.py against its baseline: new findings are a warning; a
#                             checker that does not run to its summary line fails
#   harness [BASE]            the diff BASE..HEAD must not touch the gate's own inputs (HARNESS_PATTERNS) unless
#                             REFACTOR_ACCEPT_HARNESS is the head's full sha (the owner accepted the change)
#   ruff [BASE]               ruff findings of relax, tests and scripts that BASE lacks: none
#   arity [BASE]              scripts/dev/call_arity_check.py findings (rule 18) that BASE lacks: none
#   diffcheck [BASE]          git diff --check BASE HEAD: clean
#   anchors                   scripts/dev/check_mutation_anchors.py: no stale selftest mutation
#   ceilings                  scripts/report_refinement_structure.py --check of the ceilings file: none above
#                             its slack
#   static [BASE]             ruff, arity, diffcheck, anchors, ceilings (a few minutes)
#   extra BASE FILE...        the pytest FILEs on the head and on a snapshot of BASE: the failure lists must be
#                             identical (a file BASE lacks runs on the head only)
#   all [BASE [FILE...]]      harness, fp, selftest, guard, tests head HEAD, clean head, counts, static, smell, then extra
#                             when FILEs are given (about 40 minutes; more with the modules' test directories)
# Each step prints one summary line per check and exits nonzero on failure ("tests" does not: "clean" or
# "compare" decides; "smell" always exits 0 and lists its new findings in its log). New findings are listed
# under their line. Run it from the worktree, or as a Slurm job from there
# (sbatch --account=<account> --export=ALL scripts/dev/refactor_gate.sh all). REFACTOR_WORKTREE defaults to
# the checkout of the current directory, REFACTOR_PYTHON to its pixi environment, REFACTOR_TAG (a suffix
# for log names, so two gates can share one scratch directory) to the short head. REFACTOR_MODULE (default
# refinement) selects the module's harness and test directories: scripts/dev/refactor_module.sh.
# REFACTOR_HARNESS (default: the worktree) is the tree whose gate inputs judge the worktree: the fingerprint
# harnesses and their cases and mutations, the CPU unit list, the module table, the ruff configuration, the
# arity, anchor, ceiling and smell checkers, the ceilings file and the smell baseline. A refactor gate passes
# the base's checkout here, so a branch cannot change what judges it (the harness step names such changes).
# Logs: $REFACTOR_SCRATCH/logs.
set -u -o pipefail
S=${REFACTOR_SCRATCH:?set REFACTOR_SCRATCH to a scratch directory outside the checkout}
W=${REFACTOR_WORKTREE:-$(git rev-parse --show-toplevel)} || exit 2
PY=${REFACTOR_PYTHON:-$W/.pixi/envs/default/bin/python}
T=${REFACTOR_TAG:-$(git -C "$W" rev-parse --short HEAD)}
HN=$(cd "${REFACTOR_HARNESS:-$W}" && pwd) || exit 2
# shellcheck source=scripts/dev/refactor_module.sh
. "$HN/scripts/dev/refactor_module.sh"
# The gate's own inputs (scripts/dev/harness_patterns.txt of the harness tree, the one copy the refactor tools read
# too): a diff that touches one needs the owner's acceptance (step_harness).
HARNESS_PATTERNS=$(grep -v '^#' "$HN/scripts/dev/harness_patterns.txt" | grep . | paste -sd'|') || {
  echo "no harness patterns in $HN/scripts/dev/harness_patterns.txt" >&2; exit 2; }
mkdir -p "$S/logs"
unset PYTHONHOME CONDA_PREFIX VIRTUAL_ENV RELAX_TEST_RECEIPTS
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 XLA_PYTHON_CLIENT_PREALLOCATE=false

step_fp() {
  # The harness's cases on the base and the head (the worktree as it is when the harness is the worktree).
  local head=(); [ "$HN" = "$W" ] || head=("$(git -C "$W" rev-parse HEAD)")
  cd "$HN" && PYTHONPATH=$HN "$PY" "$FP" check "${1:-origin/main}" "${head[@]}" --work-dir "$S/${FPN}_gate_$T" > "$S/logs/${FPN}_check_$T.txt" 2>&1; rc=$?
  echo "$FPN check vs ${1:-origin/main}: rc=$rc $(grep -E 'cases compared|only log rows differ|controller inputs added|controller inputs retired|controller inputs moved|controller inputs redefaulted' "$S/logs/${FPN}_check_$T.txt" | paste -sd ' ')"; return $rc
}
step_selftest() {
  # The harness's mutations, applied to the worktree's relax/.
  cd "$HN" && PYTHONPATH=$HN "$PY" "$FP" selftest --jobs 4 --source "$W" --work-dir "$S/${FPN}_selftest_$T" > "$S/logs/${FPN}_selftest_$T.txt" 2>&1; rc=$?
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
  rm -f "$S/logs/fail_$label.txt"  # a step that stops early leaves no failure list, so "clean" cannot read an old one
  # A real checkout: the launcher and guard tests run git (provenance, merge-base) in it.
  git -C "$W" worktree remove --force "$src" 2>/dev/null; rm -rf "$src"; git -C "$W" worktree prune
  git -C "$W" worktree add --detach "$src" "$rev" > /dev/null 2>&1 || return 2
  [ -e "$W/.pixi" ] && ln -s "$(readlink -f "$W/.pixi")" "$src/.pixi"
  cd "$src" || return 2
  # Every module's test directories, whichever module the gate is for (a VDAM change runs the VDAM unit tests).
  local m d files missing="" dirs=""
  for m in refinement vdam ppca scoring; do
    for d in $(REFACTOR_MODULE=$m; . "$HN/scripts/dev/refactor_module.sh"; echo "$MODULE_TESTS"); do
      if [ -d "$d" ]; then dirs="$dirs $d"; else missing="$missing$d (a module's test directory)"$'\n'; fi
    done
  done
  missing=$missing$(grep -v '^#' "$HN/scripts/dev/refactor_cpu_unit_list.txt" | grep . | while read -r f; do [ -f "$f" ] || echo "$f"; done)
  if [ -n "$missing" ]; then
    echo "tests $label ($rev): listed in the CPU unit list or the module table but missing from the tree:"; echo "$missing"; return 2
  fi
  # shellcheck disable=SC2086
  files=$( { grep -v '^#' "$HN/scripts/dev/refactor_cpu_unit_list.txt" | grep .; [ -z "$dirs" ] || find $dirs -name 'test_*.py'; } | sort -u)
  # shellcheck disable=SC2086
  # The robustness-matrix and completion launchers need an installed pixi Python as their base environment.
  EM_COMPLETION_PIXI_PY=$PY EM_K1_MATRIX_PIXI_PY=$PY \
    PYTHONPATH=$src JAX_COMPILATION_CACHE_DIR=$S/jax_cache_tests_$label RECOVAR_JAX_CACHE_DIR=$S/recovar_jax_cache_tests_$label \
    "$PY" -m pytest $files -p no:cacheprovider --basetemp="$S/pytest_tmp_$label" --junitxml="$S/logs/junit_$label.xml" \
    -rfEs -v > "$S/logs/tests_$label.log" 2>&1; rc=$?
  grep -E "^(FAILED|ERROR) " "$S/logs/tests_$label.log" | sed 's/ - .*//' | sort > "$S/logs/fail_$label.txt"
  # collected passed skipped failed, from the junit report (the "counts" step compares the head's with the base's).
  "$PY" - "$S/logs/junit_$label.xml" > "$S/logs/counts_$label.txt" <<'PY' || rm -f "$S/logs/counts_$label.txt"
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
n = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
failed = n["failures"] + n["errors"]
print(n["tests"], n["tests"] - failed - n["skipped"], n["skipped"], failed)
PY
  echo "tests $label ($rev, $(echo "$files" | wc -l) files, all modules): rc=$rc $(tail -1 "$S/logs/tests_$label.log") -> $S/logs/fail_$label.txt"
  # pytest exits 0 (passed) or 1 (some failed: "clean" decides); anything else is a run that did not finish.
  [ $rc -le 1 ] || { echo "tests $label: pytest exited $rc (not a test outcome)"; return 1; }
}
step_clean() {
  local f=$S/logs/fail_${1:?clean needs a label}.txt
  [ -f "$f" ] || { echo "failure list $1: missing"; return 1; }
  if [ ! -s "$f" ]; then echo "failure list $1: empty"; return 0; fi
  echo "failure list $1: $(wc -l < "$f") failed"; cat "$f"; return 1
}
step_smell() {
  # New findings are a warning; a checker that crashes, or ends without its summary line, fails.
  cd "$HN" && "$PY" scripts/dev/smell_check.py --root "$W" --baseline "$HN/scripts/dev/smell_baseline.json" > "$S/logs/smell_$T.txt" 2>&1; rc=$?
  local last; last=$(tail -1 "$S/logs/smell_$T.txt")
  if [ $rc != 0 ] || [[ $last != "smell check: "* ]]; then echo "smell check: rc=$rc, did not finish: $last -> $S/logs/smell_$T.txt"; return 1; fi
  echo "$last -> $S/logs/smell_$T.txt (new findings: warning only)"; return 0
}
step_counts() {
  # The base's tests under the same harness, then: the head passes at least as many tests as the base, unless the
  # owner accepted the head's harness change (a deleted test, a narrowed collection).
  local base=${1:-origin/main} h b head
  step_tests base "$base"
  h=$(cat "$S/logs/counts_head.txt" 2>/dev/null) || { echo "test counts: no head counts (run tests head HEAD first)"; return 2; }
  b=$(cat "$S/logs/counts_base.txt" 2>/dev/null) || { echo "test counts: no base counts"; return 2; }
  set -- $h; local hc=$1 hp=$2 hs=$3; set -- $b; local bc=$1 bp=$2 bs=$3
  head=$(git -C "$W" rev-parse HEAD)
  echo "test counts (collected passed skipped): head $hc $hp $hs, base $bc $bp $bs"
  [ "$hp" -ge "$bp" ] && return 0
  if [ "${REFACTOR_ACCEPT_HARNESS:-}" = "$head" ]; then echo "test counts: head passes $((bp - hp)) fewer, accepted for $head"; return 0; fi
  echo "test counts: the head passes $((bp - hp)) fewer tests than $base; the owner must accept it (REFACTOR_ACCEPT_HARNESS=$head)"
  return 1
}
step_harness() {
  local base=${1:-origin/main} touched head
  touched=$(git -C "$W" diff --no-renames --name-only "$base" HEAD) || { echo "harness: git diff $base HEAD failed"; return 2; }
  touched=$(printf '%s\n' "$touched" | grep -E "$HARNESS_PATTERNS" || true)
  head=$(git -C "$W" rev-parse HEAD)
  if [ -z "$touched" ]; then echo "harness files touched since $base: none"; return 0; fi
  if [ "${REFACTOR_ACCEPT_HARNESS:-}" = "$head" ]; then
    echo "harness files touched since $base, accepted for $head: $(echo $touched)"; return 0; fi
  echo "harness files touched since $base: $(echo $touched); the owner must accept them (REFACTOR_ACCEPT_HARNESS=$head)"
  return 1
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
# ruff under the harness's configuration (a branch's own pyproject cannot ignore a rule). A run counts only when
# it ends with ruff's own verdict ("All checks passed!" or "Found N errors"): a missing ruff module exits 1 too.
ruff_findings() {
  local cfg=() out rc; [ -f "$HN/pyproject.toml" ] && cfg=(--config "$HN/pyproject.toml")
  out=$("$PY" -m ruff check "${cfg[@]}" relax tests scripts --output-format concise 2>"$S/logs/ruff_stderr_$T.txt"); rc=$?
  if [ $rc -gt 1 ] || ! grep -qE '^(All checks passed!|Found [0-9]+ error)' <<<"$out"; then
    echo "ruff did not run in $(pwd): rc=$rc $(tail -1 "$S/logs/ruff_stderr_$T.txt")" >&2; return 2
  fi
  printf '%s\n' "$out" | sed "s/:[0-9]*:[0-9]*:/:/" | { grep -v "^Found\|fixable\|^All checks\|^$" || true; } | sort
}
step_ruff() {
  local base=${1:-origin/main}; base_tree "$base" || return 2
  (cd "$B" && ruff_findings) > "$S/logs/ruff_base_$T.txt" || { echo "ruff: did not run on $base"; return 2; }
  (cd "$W" && ruff_findings) > "$S/logs/ruff_head_$T.txt" || { echo "ruff: did not run on the head"; return 2; }
  new_findings "$S/logs/ruff_base_$T.txt" "$S/logs/ruff_head_$T.txt" "ruff findings $base lacks"
}
step_arity() {
  # The harness's checker reads both trees, so a checker change does not show as findings; a crash fails.
  local base=${1:-origin/main}; base_tree "$base" || return 2
  "$PY" "$HN/scripts/dev/call_arity_check.py" "$B" | sort > "$S/logs/arity_base_$T.txt" || { echo "call arity check: crashed on $base"; return 2; }
  "$PY" "$HN/scripts/dev/call_arity_check.py" "$W" | sort > "$S/logs/arity_head_$T.txt" || { echo "call arity check: crashed on the head"; return 2; }
  new_findings "$S/logs/arity_base_$T.txt" "$S/logs/arity_head_$T.txt" "call arity findings $base lacks"
}
step_diffcheck() {
  local base=${1:-origin/main}
  git -C "$W" diff --check "$base" HEAD > "$S/logs/diffcheck_$T.txt" 2>&1; rc=$?
  echo "git diff --check $base HEAD: rc=$rc $(grep -vc '^+' "$S/logs/diffcheck_$T.txt") findings"; [ $rc = 0 ] || { cat "$S/logs/diffcheck_$T.txt"; return 1; }
}
step_anchors() {
  cd "$HN" && PYTHONPATH=$HN "$PY" scripts/dev/check_mutation_anchors.py --root "$W" > "$S/logs/anchors_$T.txt" 2>&1; rc=$?
  echo "$(tail -1 "$S/logs/anchors_$T.txt"): rc=$rc"; [ $rc = 0 ] || { grep '^STALE' "$S/logs/anchors_$T.txt"; return 1; }
}
step_ceilings() {
  # A metric above its ceiling within the slack is a warning (named in the report, procedure section 4).
  cd "$HN" && PYTHONPATH=$HN "$PY" scripts/report_refinement_structure.py --root "$W" --check "$HN/docs/development/refinement_structure_metrics.json" \
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
  local hrc brc=0
  cd "$W" && PYTHONPATH=$W "$PY" -m pytest "${hx[@]}" -q -p no:cacheprovider --basetemp="$S/pytest_tmp_extra_head_$T" -rfE \
    > "$S/logs/extra_head_$T.log" 2>&1; hrc=$?
  grep -E "^(FAILED|ERROR) " "$S/logs/extra_head_$T.log" | sed 's/ - .*//' | sort > "$S/logs/fail_extra_head_$T.txt"
  base_tree "$base" || return 2; [ -e "$W/.pixi" ] && ln -sfn "$(readlink -f "$W/.pixi")" "$B/.pixi"
  for f in "${hx[@]}"; do [ -f "$B/$f" ] && bx+=("$f"); done
  if [ ${#bx[@]} -gt 0 ]; then
    (cd "$B" && PYTHONPATH=$B "$PY" -m pytest "${bx[@]}" -q -p no:cacheprovider --basetemp="$S/pytest_tmp_extra_base_$T" -rfE \
      > "$S/logs/extra_base_$T.log" 2>&1); brc=$?
  else echo "no extra file exists on $base" > "$S/logs/extra_base_$T.log"; fi
  grep -E "^(FAILED|ERROR) " "$S/logs/extra_base_$T.log" | sed 's/ - .*//' | sort > "$S/logs/fail_extra_base_$T.txt"
  rm -rf "$S/base_src_$T"
  echo "extra tests head: rc=$hrc $(tail -1 "$S/logs/extra_head_$T.log") | base: rc=$brc $(tail -1 "$S/logs/extra_base_$T.log")"
  # Identical failure lists mean nothing when a run did not finish (a crash or timeout leaves no FAILED lines).
  [ $hrc -le 1 ] && [ $brc -le 1 ] || { echo "extra: pytest exited $hrc on the head, $brc on the base (not a test outcome)"; return 1; }
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
harness) step_harness "${2:-}";;
counts) step_counts "${2:-}";;
ruff) step_ruff "${2:-}";;
arity) step_arity "${2:-}";;
diffcheck) step_diffcheck "${2:-}";;
anchors) step_anchors;;
ceilings) step_ceilings;;
static) step_static "${2:-}";;
extra) shift; step_extra "$@";;
all) status=0; base=${2:-origin/main}; shift $(($# < 2 ? $# : 2))
  step_harness "$base" || status=1
  step_fp "$base" || status=1; step_selftest || status=1; step_guard || status=1
  step_tests head HEAD || status=1; step_clean head || status=1; step_counts "$base" || status=1
  step_static "$base" || status=1; step_smell || status=1
  [ $# = 0 ] || step_extra "$base" "$@" || status=1
  exit $status;;
*) sed -n '6,/^# Logs/p' "$0"; exit 2;;
esac
