#!/bin/bash
#SBATCH --job-name=relax-polar-guard
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:08:00

set -euo pipefail
: "${POLAR_SOURCE_ROOT:?submit with scripts/polar/submit.py}"
: "${POLAR_ENV_ROOT:?submit with scripts/polar/submit.py}"
: "${POLAR_RUN_ROOT:?submit with scripts/polar/submit.py}"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PATH="$POLAR_ENV_ROOT/bin:$PATH"
export PYTHON="$POLAR_ENV_ROOT/bin/python"
export RECOVAR_JAX_CACHE_DIR="$POLAR_RUN_ROOT/jax-cache"
export JAX_COMPILATION_CACHE_DIR="$RECOVAR_JAX_CACHE_DIR"
export XDG_CACHE_HOME="$POLAR_RUN_ROOT/cache"
export TMPDIR="$POLAR_RUN_ROOT/tmp"
mkdir -p "$RECOVAR_JAX_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"

# The repository guard expects its frozen environment at .pixi/envs/default.
# Keep the submitted snapshot read-only and add that link in a private run copy.
cp -a "$POLAR_SOURCE_ROOT" "$POLAR_RUN_ROOT/source"
chmod -R u+w "$POLAR_RUN_ROOT/source"
mkdir -p "$POLAR_RUN_ROOT/source/.pixi/envs"
ln -s "$POLAR_ENV_ROOT" "$POLAR_RUN_ROOT/source/.pixi/envs/default"
cd "$POLAR_RUN_ROOT/source"
bash scripts/run_em_fast_guard.sh
