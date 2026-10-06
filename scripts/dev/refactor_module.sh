# The module under refactor, for refactor_verify.sh and refactor_gate.sh (sourced by both; not run alone).
# This is the one place that says which harness, structure test and test directories belong to a module.
#   REFACTOR_MODULE            refinement (default) or vdam: selects the three defaults below
#   REFACTOR_FINGERPRINT       the module's fingerprint harness (fp, selftest and the per-commit check)
#   REFACTOR_STRUCTURE_TESTS   pytest arguments of the module's ceiling tests; verify runs them on every commit
#   REFACTOR_MODULE_TESTS      the module's test directories; the gate's "tests" step runs every test_*.py
#                              in them on both snapshots, on top of scripts/dev/refactor_cpu_unit_list.txt
# Each variable set in the environment wins over the module's default. A new module adds one case here and
# names REFACTOR_MODULE=<name> in its status document (docs/development/<module>_rules_status.md).
case ${REFACTOR_MODULE:=refinement} in
refinement)
  _fp=scripts/dev/fingerprint.py
  _structure=tests/unit/test_refinement_structure_metrics.py
  _tests="";;  # its unit files are all in the CPU unit list
vdam)
  _fp=scripts/dev/vdam_fingerprint.py
  _structure=tests/unit/initial_model/test_refactor_invariants.py::test_responsibility_loc_budget
  _tests="tests/unit/initial_model tests/unit/ppca_initial_model";;
*) echo "unknown REFACTOR_MODULE=$REFACTOR_MODULE (scripts/dev/refactor_module.sh lists the modules)" >&2; exit 2;;
esac
FP=${REFACTOR_FINGERPRINT:-$_fp}
STRUCTURE_TESTS=${REFACTOR_STRUCTURE_TESTS-$_structure}
MODULE_TESTS=${REFACTOR_MODULE_TESTS-$_tests}
FPN=$(basename "$FP" .py)
unset _fp _structure _tests
