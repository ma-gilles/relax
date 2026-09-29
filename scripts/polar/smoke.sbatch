#!/bin/bash
#SBATCH --job-name=relax-polar-smoke
#SBATCH --partition=main
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:10:00

set -euo pipefail
: "${POLAR_SOURCE_ROOT:?submit with scripts/polar/submit.py}"
: "${POLAR_INPUT_ROOT:?submit with scripts/polar/submit.py}"
: "${POLAR_ENV_ROOT:?submit with scripts/polar/submit.py}"
: "${POLAR_RUN_ROOT:?submit with scripts/polar/submit.py}"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu
export PATH="$POLAR_ENV_ROOT/bin:$PATH"
export PYTHONPATH="$POLAR_SOURCE_ROOT"
export RECOVAR_JAX_CACHE_DIR="$POLAR_RUN_ROOT/jax-cache"
export JAX_COMPILATION_CACHE_DIR="$RECOVAR_JAX_CACHE_DIR"
export XDG_CACHE_HOME="$POLAR_RUN_ROOT/cache"
export TMPDIR="$POLAR_RUN_ROOT/tmp"
mkdir -p "$RECOVAR_JAX_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"

printf 'job=%s node=%s visible_gpu=%s\n' "$SLURM_JOB_ID" "$(hostname)" "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
cd "$POLAR_SOURCE_ROOT"
python scripts/polar/smoke.py "$POLAR_INPUT_ROOT/residuals.json" "$POLAR_RUN_ROOT/result.json"
