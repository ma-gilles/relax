#!/bin/bash
#SBATCH --job-name=relax-polar-natives
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00

set -euo pipefail
: "${POLAR_SOURCE_ROOT:?submit with scripts/polar/submit.py}"
: "${POLAR_ENV_ROOT:?submit with scripts/polar/submit.py}"
: "${POLAR_RUN_ROOT:?submit with scripts/polar/submit.py}"
: "${POLAR_CUDA_ARCHIVE:?stage the CUDA toolkit archive on universal scratch}"
: "${POLAR_CUDA_ARCHIVE_SHA256:?pass the archive SHA256}"
: "${POLAR_RELION_BIND_BUILD_DIR:?pass a checked RELION binding build directory}"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
export PATH="$POLAR_ENV_ROOT/bin:$PATH"
export PYTHONPATH="$POLAR_SOURCE_ROOT"
export TMPDIR="$POLAR_RUN_ROOT/tmp" XDG_CACHE_HOME="$POLAR_RUN_ROOT/cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"

printf '%s  %s\n' "$POLAR_CUDA_ARCHIVE_SHA256" "$POLAR_CUDA_ARCHIVE" | sha256sum -c -
toolchain_base="/scratch/universal/mg6942/relax-polar/toolchains"
toolchain="$toolchain_base/cuda-12.8-${POLAR_CUDA_ARCHIVE_SHA256:0:12}"
if [[ ! -f "$toolchain/.polar-ready" ]]; then
    staging="$toolchain.uploading-${SLURM_JOB_ID}"
    mkdir -p "$staging"
    tar -xzf "$POLAR_CUDA_ARCHIVE" -C "$staging"
    "$staging/bin/nvcc" --version
    touch "$staging/.polar-ready"
    chmod -R a-w "$staging"
    mv -T "$staging" "$toolchain"
fi
export CUDA_HOME="$toolchain"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_ARCH='-gencode arch=compute_80,code=sm_80'
"$CUDA_HOME/bin/nvcc" --version | tee "$POLAR_RUN_ROOT/nvcc.version"

natives="$POLAR_RUN_ROOT/natives"
mkdir -p "$natives/relion_bind"
cp "$POLAR_RELION_BIND_BUILD_DIR"/_relion_bind_core*.so "$natives/relion_bind/"
python -m recovar.commands.build_custom_cuda --output "$natives/libcuda_backproject.so" --force \
    > "$POLAR_RUN_ROOT/build_recovar_cuda.log" 2>&1
make -C "$POLAR_SOURCE_ROOT/relax/cuda" PYTHON="$POLAR_ENV_ROOT/bin/python" \
    LIB="$natives/librelax_cuda.so" all > "$POLAR_RUN_ROOT/build_relax_cuda.log" 2>&1

python "$POLAR_SOURCE_ROOT/scripts/native_sources.py" record "$natives" --root "$POLAR_SOURCE_ROOT"
sha256sum "$natives"/*.so "$natives"/relion_bind/*.so | tee "$natives/binaries.sha256"
python - "$natives/NATIVE.json" "$POLAR_CUDA_ARCHIVE_SHA256" "$toolchain" "$POLAR_SOURCE_ROOT" <<'PY'
import json, sys
from pathlib import Path
record_path, archive_sha, toolchain, source = sys.argv[1:]
path = Path(record_path)
record = json.loads(path.read_text())
record.update(cuda_archive_sha256=archive_sha, cuda_toolchain=toolchain, source=source)
path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY
chmod -R a-w "$natives"
echo "native build ready: $natives"
