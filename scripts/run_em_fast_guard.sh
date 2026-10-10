#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
# Tests share a GPU between the pytest process and the relax subprocesses it starts; see CONTRIBUTING.md.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

backend="${EM_FAST_GUARD_BACKEND:-cpu}"
if [[ "$backend" == "gpu" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "EM_FAST_GUARD_BACKEND=gpu requested, but nvidia-smi is not available" >&2
    exit 1
  fi
  nvidia-smi
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"
else
  export JAX_PLATFORMS=cpu
  unset JAX_PLATFORM_NAME
fi

PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$ROOT/.pixi/envs/default/bin/python" ]]; then
    PYTHON_BIN="$ROOT/.pixi/envs/default/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

# Catch incomplete caller migrations before importing JAX or compiling tests.
# This guard covers the dense/local package; shared/legacy EM has separate gates.
"$PYTHON_BIN" -m ruff check --select F821 "$ROOT/relax"

"$PYTHON_BIN" - <<'PY'
import pathlib
import importlib
import sys

import jax
import recovar

repo = pathlib.Path.cwd().resolve()
recovar_file = pathlib.Path(recovar.__file__).resolve()
jax_file = pathlib.Path(jax.__file__).resolve()
pixi_env = (repo / ".pixi" / "envs" / "default").resolve()
assert str(pathlib.Path(__import__("relax").__file__).resolve()).startswith(str(repo) + "/"), __import__("relax").__file__
assert str(jax_file).startswith(str(pixi_env) + "/"), (jax_file, pixi_env)
for helper in (
    "helpers.oversampling", "helpers.half_volume_mstep", "relion.relion_projector_setup",
    "parity.relion_replay", "relion.relion_normalization", "refinement.projector_preparation",
    "dense.score_outputs", "classification.k_class_results", "classification.k_class_inputs", "dense.scoring_policy", "helpers.resolution", "diagnostics.bpref_diagnostics",
    "helpers.expected_accuracy", "scoring.significant_samples", "diagnostics.coarse_score_diagnostics", "scoring.sparse_bucket_arrays", "scoring.compact_candidates", "relion.relion_ctf", "helpers.scale_groups",
    "relion.vdam_checkpoint", "local.local_layout", "diagnostics.local_debug",
):
    importlib.import_module(f"relax.{helper}")
for diagnostic in ("iteration", "reconstruction"):
    importlib.import_module(f"relax.diagnostics.{diagnostic}")
execution_modules = (
    "refinement.iteration_loop", "refinement.dense_half", "classification.k_class",
    "scoring.significance", "sparse_pass2.resident_pass2", "sparse_pass2.dispatch",
    "refinement.local_half",
)
loaded = [name for name in execution_modules
          if f"relax.{name}" in sys.modules]
assert not loaded, f"EM helper imports must not load execution modules: {loaded}"
print(f"provenance_ok recovar={recovar_file} jax={jax_file}")
print("helper_import_boundary_ok")
PY

tests=(
  tests/unit/test_em_fast_guardrail.py
  tests/unit/test_relion_replay_state.py
  tests/unit/test_healpix_order_oracle.py
  tests/unit/test_resolution_scheduling.py
)

exec "$PYTHON_BIN" -m pytest "${tests[@]}" -q "$@"
