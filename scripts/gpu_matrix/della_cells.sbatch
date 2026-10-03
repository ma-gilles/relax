#!/bin/bash
#SBATCH --job-name=relax-gpu-matrix
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=NONE
# Cross-GPU matrix cells on Della (docs/development/gpu_compatibility.md), one worker per GPU. Submit as
#   sbatch --gres=gpu:N --cpus-per-task=$((8*N)) --mem=$((128*N))G --time=HH:MM:00 --constraint=h100|a100 \
#     --output=OUT/job-%j.log --export=NONE,MATRIX_SRC=<frozen checkout>,MATRIX_NATIVES=<natives>,\
#MATRIX_OUT=<out root>,MATRIX_CELLS='cell cell@16 ...' scripts/gpu_matrix/della_cells.sbatch
# MATRIX_SRC is a frozen checkout sharing a pixi environment; it is put first on PYTHONPATH.

set -uo pipefail
: "${MATRIX_SRC:?}" "${MATRIX_NATIVES:?}" "${MATRIX_OUT:?}" "${MATRIX_CELLS:?}"
unset PYTHONHOME CONDA_PREFIX VIRTUAL_ENV LD_PRELOAD LD_LIBRARY_PATH XLA_PYTHON_CLIENT_MEM_FRACTION
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE="${MATRIX_PREALLOCATE:-false}"
export PYTHONPATH="$MATRIX_SRC"
export RECOVAR_JAX_CACHE_DIR="$MATRIX_OUT/jax-cache" JAX_COMPILATION_CACHE_DIR="$MATRIX_OUT/jax-cache"
export RECOVAR_CUDA_CACHE_DIR="$MATRIX_OUT/cuda-cache" TMPDIR="$MATRIX_OUT/tmp"
export RECOVAR_CUDA_LIB="$MATRIX_NATIVES/libcuda_backproject.so" RELAX_CUDA_LIB="$MATRIX_NATIVES/librelax_cuda.so"
export RECOVAR_RELION_BIND_BUILD_DIR="$MATRIX_NATIVES/relion_bind"
export RELAX_ENABLE_CUSTOM_CUDA=1 RELAX_RELION_BIND_COPY_TO_PACKAGE=0
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 SLURM_CPUS_PER_TASK=8
mkdir -p "$RECOVAR_JAX_CACHE_DIR" "$RECOVAR_CUDA_CACHE_DIR" "$TMPDIR"
PY="$MATRIX_SRC/.pixi/envs/default/bin/python"

printf 'job=%s node=%s visible_gpu=%s\n' "${SLURM_JOB_ID:-}" "$(hostname)" "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,uuid,compute_cap,memory.total,driver_version,compute_mode --format=csv,noheader
git -C "$MATRIX_SRC" rev-parse HEAD
"$PY" "$MATRIX_SRC/scripts/native_sources.py" check "$MATRIX_NATIVES" --root "$MATRIX_SRC" || exit 3
"$PY" "$MATRIX_SRC/scripts/native_sources.py" imports "$MATRIX_SRC" || exit 3
cd "$MATRIX_OUT"
# shellcheck disable=SC2086
"$PY" "$MATRIX_SRC/scripts/gpu_matrix/run_cells.py" --out-root "$MATRIX_OUT/cells" $MATRIX_CELLS
status=$?
rm -rf "$TMPDIR"
exit "$status"
