#!/usr/bin/env bash
# Build the native libraries a GPU test tier loads into one directory and record their identity.
#
#   scripts/build_test_natives.sh <out_dir>
#
# Builds the recovar custom CUDA library (libcuda_backproject.so), relax's EM CUDA library
# (librelax_cuda.so) and the RELION binding (_relion_bind_core*.so) for A100 and H100, then
# writes <out_dir>/NATIVE.json with their sha256. cudatoolkit is loaded for the make steps
# only and unloaded afterwards, so the tests run against the pixi environment's CUDA runtime.
# Point a run at the result with:
#   RECOVAR_CUDA_LIB=<out_dir>/libcuda_backproject.so RELAX_CUDA_LIB=<out_dir>/librelax_cuda.so
#   RECOVAR_RELION_BIND_BUILD_DIR=<out_dir>/relion_bind
set -euo pipefail

OUT="${1:?usage: build_test_natives.sh <out_dir>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.pixi/envs/default/bin/python"
RELION_SRC_DIR="${RELION_SRC_DIR:-/scratch/gpfs/GILLES/mg6942/relion/src}"
CUDA_MODULE="${CUDA_MODULE:-cudatoolkit/12.8}"
ARCH='-gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90'
test -x "${PY}" || { echo "missing pixi environment at ${PY}" >&2; exit 2; }
mkdir -p "${OUT}/relion_bind"
OUT="$(cd "${OUT}" && pwd)"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1 RELION_SRC_DIR

# The RELION binding is host code; build it before the CUDA module is loaded.
RECOVAR_RELION_BIND_BUILD_DIR="${OUT}/relion_bind" "${PY}" "${ROOT}/relax/relion_bind/build.py" > "${OUT}/build_relion_bind.log" 2>&1

set +u
source /etc/profile.d/modules.sh
module load "${CUDA_MODULE}"
set -u
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
CUDA_ARCH="${ARCH}" "${PY}" -m recovar.commands.build_custom_cuda \
  --output "${OUT}/libcuda_backproject.so" --force > "${OUT}/build_recovar_cuda.log" 2>&1
env PYTHON="${PY}" CUDA_ARCH="${ARCH}" make -C "${ROOT}/relax/cuda" LIB="${OUT}/librelax_cuda.so" all \
  > "${OUT}/build_relax_cuda.log" 2>&1
set +u
module unload "${CUDA_MODULE}"
set -u

"${PY}" - "${OUT}" "${ROOT}" "${CUDA_MODULE}" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path

out, root, module = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
libs = [out / "libcuda_backproject.so", out / "librelax_cuda.so", *sorted((out / "relion_bind").glob("_relion_bind_core*.so"))]
missing = [str(p) for p in libs if not p.is_file()]
if missing or len(libs) != 3:
    sys.exit(f"native build incomplete: missing {missing or 'RELION binding'}")
record = {
    "source_head": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
    "source_dirty": bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True).strip()),
    "cuda_module": module,
    "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in libs},
    "paths": {p.name: str(p) for p in libs},
}
(out / "NATIVE.json").write_text(json.dumps(record, indent=1) + "\n")
print(json.dumps(record["sha256"], indent=1))
PY
# Record the native source tree these libraries come from; runs check it (scripts/native_sources.py).
"${PY}" "${ROOT}/scripts/native_sources.py" record "${OUT}" --root "${ROOT}"
