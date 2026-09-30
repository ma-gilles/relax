#!/usr/bin/env bash
#SBATCH --job-name=ppca-batch-pilot
#SBATCH --partition=main
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=00:40:00
set -euo pipefail
: "${POLAR_SOURCE_ROOT:?} ${POLAR_ENV_ROOT:?} ${POLAR_RUN_ROOT:?} ${POLAR_FIXTURE_ROOT:?} ${PPCA_TEMPLATE_CP0:?} ${PPCA_MANIFEST_SHA256:?} ${CUDA_VISIBLE_DEVICES:?}"
natives=/scratch/network/mg6942/relax-polar/runs/b1846995885f/natives
unset PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE VIRTUAL_ENV
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu RECOVAR_EM_XLA_DEFAULTS=1
export PATH="$POLAR_ENV_ROOT/bin:$PATH" PYTHONPATH="$POLAR_SOURCE_ROOT"
export RECOVAR_CUDA_LIB="$natives/libcuda_backproject.so"
export RELAX_CUDA_LIB="$natives/librelax_cuda.so"
export RECOVAR_RELION_BIND_BUILD_DIR="$natives/relion_bind"
export RECOVAR_JAX_CACHE_DIR="$POLAR_RUN_ROOT/jax-cache" JAX_COMPILATION_CACHE_DIR="$POLAR_RUN_ROOT/jax-cache"
export XDG_CACHE_HOME="$POLAR_RUN_ROOT/cache" TMPDIR="$POLAR_RUN_ROOT/tmp"
export OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 MKL_NUM_THREADS=10
mkdir -p "$RECOVAR_JAX_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"
printf '%s\n' 'Disposable timing-only Polar pilot; frozen source and 20k fixture.' > "$POLAR_RUN_ROOT/SAFE_TO_DELETE"
nvidia-smi --query-gpu=name,uuid --format=csv,noheader
(cd "$POLAR_SOURCE_ROOT" && sha256sum -c .polar-sha256 >/dev/null)
(cd "$POLAR_FIXTURE_ROOT" && sha256sum -c .polar-sha256 >/dev/null)
(cd "$natives" && sha256sum -c binaries.sha256 >/dev/null)
"$POLAR_ENV_ROOT/bin/python" "$POLAR_SOURCE_ROOT/scripts/polar/ppca_batch_pilot.py"
