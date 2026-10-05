#!/bin/bash
#SBATCH --job-name=relax-gpu-matrix
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
# Cross-GPU matrix cells on one Polar GPU (docs/development/gpu_compatibility.md). Submit with
#   scripts/polar/submit.py scripts/gpu_matrix/polar_cells.sbatch --sbatch-option=--gres=gpu:<model>:1 \
#     --set POLAR_NATIVE_ROOT=... --set POLAR_CELLS='cell1 cell2' --set POLAR_FIXTURES='name=path name=path'
# POLAR_FIXTURES maps each fixture a cell uses to its staged copy on /scratch/network.

set -euo pipefail
: "${POLAR_SOURCE_ROOT:?submit with scripts/polar/submit.py}"
: "${POLAR_ENV_ROOT:?submit with scripts/polar/submit.py}"
: "${POLAR_RUN_ROOT:?submit with scripts/polar/submit.py}"
: "${POLAR_NATIVE_ROOT:?pass the natives directory built for this GPU architecture}"
: "${POLAR_CELLS:?pass the cells to run}"
# Cells run JAX's default allocator mode, as a user does (one preallocated pool; relax#20).
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV LD_PRELOAD LD_LIBRARY_PATH XLA_PYTHON_CLIENT_MEM_FRACTION
unset XLA_PYTHON_CLIENT_PREALLOCATE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PATH="$POLAR_ENV_ROOT/bin:$PATH"
export PYTHONPATH="$POLAR_SOURCE_ROOT"
export RECOVAR_JAX_CACHE_DIR="$POLAR_RUN_ROOT/jax-cache" JAX_COMPILATION_CACHE_DIR="$POLAR_RUN_ROOT/jax-cache"
export RECOVAR_CUDA_CACHE_DIR="$POLAR_RUN_ROOT/cuda-cache"
export XDG_CACHE_HOME="$POLAR_RUN_ROOT/cache" TMPDIR="$POLAR_RUN_ROOT/tmp"
export RECOVAR_CUDA_LIB="$POLAR_NATIVE_ROOT/libcuda_backproject.so"
export RELAX_CUDA_LIB="$POLAR_NATIVE_ROOT/librelax_cuda.so"
export RECOVAR_RELION_BIND_BUILD_DIR="$POLAR_NATIVE_ROOT/relion_bind"
export RELAX_ENABLE_CUSTOM_CUDA=1 RELAX_RELION_BIND_COPY_TO_PACKAGE=0
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p "$RECOVAR_JAX_CACHE_DIR" "$RECOVAR_CUDA_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"

printf 'job=%s node=%s visible_gpu=%s\n' "$SLURM_JOB_ID" "$(hostname)" "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,uuid,compute_cap,memory.total,driver_version,compute_mode --format=csv,noheader
(cd "$POLAR_NATIVE_ROOT" && sha256sum -c binaries.sha256)
python "$POLAR_SOURCE_ROOT/scripts/native_sources.py" check "$POLAR_NATIVE_ROOT" --root "$POLAR_SOURCE_ROOT"
python -c 'import relax, recovar, jax; print(relax.__file__, recovar.__file__, jax.__version__, jax.devices())'

fixture_args=()
for item in ${POLAR_FIXTURES:-}; do
    fixture_args+=(--cell-arg="--fixture=$item")
done
cd "$POLAR_RUN_ROOT"
set +e
# shellcheck disable=SC2086
python "$POLAR_SOURCE_ROOT/scripts/gpu_matrix/run_cells.py" --out-root "$POLAR_RUN_ROOT/cells" \
    "${fixture_args[@]}" $POLAR_CELLS
status=$?
set -e
rm -rf "$TMPDIR"
exit "$status"
