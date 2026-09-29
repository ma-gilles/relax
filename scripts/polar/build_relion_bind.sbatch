#!/bin/bash
#SBATCH --job-name=relax-polar-bind
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00

set -euo pipefail
: "${POLAR_SOURCE_ROOT:?submit with scripts/polar/submit.py}"
: "${POLAR_ENV_ROOT:?submit with scripts/polar/submit.py}"
: "${POLAR_RUN_ROOT:?submit with scripts/polar/submit.py}"
: "${POLAR_RELION_SRC_DIR:?stage RELION source and pass its Polar path with --set}"
test -f "$POLAR_RELION_SRC_DIR/projector.h"
[[ "$(basename "$POLAR_RELION_SRC_DIR")" == src ]] || {
    echo "POLAR_RELION_SRC_DIR must end in /src for RELION's include layout" >&2
    exit 1
}
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
export PATH="$POLAR_ENV_ROOT/bin:$PATH"
export TMPDIR="$POLAR_RUN_ROOT/tmp"
export XDG_CACHE_HOME="$POLAR_RUN_ROOT/cache"
export RELION_SRC_DIR="$POLAR_RELION_SRC_DIR"
export RECOVAR_RELION_BIND_BUILD_DIR="$POLAR_RUN_ROOT/relion_bind"
export RELAX_RELION_BIND_JOBS="$SLURM_CPUS_PER_TASK"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$RECOVAR_RELION_BIND_BUILD_DIR"

# CMake looks for FFTW beneath the source checkout's .pixi directory.
cp -a "$POLAR_SOURCE_ROOT" "$POLAR_RUN_ROOT/source"
chmod -R u+w "$POLAR_RUN_ROOT/source"
mkdir -p "$POLAR_RUN_ROOT/source/.pixi/envs"
ln -s "$POLAR_ENV_ROOT" "$POLAR_RUN_ROOT/source/.pixi/envs/default"
cd "$POLAR_RUN_ROOT/source"
python relax/relion_bind/build.py
python - <<'PY'
from pathlib import Path
from relax.relion_bind import _relion_bind_core as binding
assert hasattr(binding, "get_healpix_directions")
print("binding", Path(binding.__file__).resolve())
PY
sha256sum "$RECOVAR_RELION_BIND_BUILD_DIR"/_relion_bind_core*.so | tee "$POLAR_RUN_ROOT/binding.sha256"
