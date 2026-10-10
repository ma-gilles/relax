#!/usr/bin/env bash
#SBATCH --job-name=ppca_K2q1_test
#SBATCH --partition=gpu --account=czb --qos=normal
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1
#SBATCH --cpus-per-task=8 --mem=32G --time=01:00:00
#SBATCH --no-requeue --export=NONE
#SBATCH --mail-user=yw3384@princeton.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,INVALID_DEPEND
#SBATCH --output=/home/ywang/three_species_particle_stacks/relax_tomo_abinitio_k10_D64/logs/%x-%j.log
#SBATCH --error=/home/ywang/three_species_particle_stacks/relax_tomo_abinitio_k10_D64/logs/%x-%j.log
set -euo pipefail
export PATH=/home/ywang/relax-dev/.pixi/envs/default/bin:/usr/bin:/bin
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cuda,cpu
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
cd /home/ywang/relax-dev
RUN=/home/ywang/three_species_particle_stacks/relax_tomo_abinitio_k10_D64/validation/gpu_${SLURM_JOB_ID}
mkdir -p "$RUN/natives"
git rev-parse HEAD > "$RUN/base_commit.txt"
git diff HEAD | sha256sum > "$RUN/tracked_diff.sha256"
git ls-files -z --cached --others --exclude-standard relax tests scripts pixi.toml pixi.lock |
    sort -zu | xargs -0 sha256sum > "$RUN/source_files.sha256"
trap 'sha256sum --quiet -c "$RUN/source_files.sha256"' EXIT

export RECOVAR_CUDA_LIB="$RUN/natives/libcuda_backproject.so"
export RELAX_CUDA_LIB="$RUN/natives/librelax_cuda.so"
export RECOVAR_JAX_CACHE_DIR="$RUN/jax_cache"
export JAX_COMPILATION_CACHE_DIR="$RUN/jax_compilation_cache"
export NVCC=/hpc/apps/x86_64/cuda/12.8.0_570.86.10/bin/nvcc
GPU_ARCH=$(python -c 'import jax; d=jax.devices()[0]; assert d.platform == "gpu"; print(str(d.compute_capability).replace(".", ""))')
export CUDA_ARCH="-arch=sm_${GPU_ARCH}"
python -c 'import jax; print([(str(d), d.device_kind, d.compute_capability) for d in jax.devices()])'
if [[ -n "${2:-}" ]]; then
    # Optional source-verified natives from a completed test on the same architecture.
    [[ "$GPU_ARCH" == "${3:?Specify the original native CUDA architecture}" ]]
    python scripts/native_sources.py check "$2" --root "$PWD"
    cp "$2/libcuda_backproject.so" "$RECOVAR_CUDA_LIB"
    cp "$2/librelax_cuda.so" "$RELAX_CUDA_LIB"
else
    JAX_PLATFORMS=cpu recovar build_custom_cuda --output "$RECOVAR_CUDA_LIB"
    JAX_PLATFORMS=cpu relax build_cuda --output "$RELAX_CUDA_LIB"
fi
sha256sum "$RECOVAR_CUDA_LIB" "$RELAX_CUDA_LIB" > "$RUN/natives/libraries.sha256"
python scripts/native_sources.py record "$RUN/natives" --root "$PWD"
python scripts/native_sources.py check "$RUN/natives" --root "$PWD"

# GPU qualification focuses on K>1 as requested; K=1 full-loop reduction
# remains covered by the CPU suite, not the new feature's GPU acceptance gate.
if [[ "${1:-tests}" == "diagnose" ]]; then
    export PYTHONPATH="$PWD/tests:$PWD/tests/unit/ppca_initial_model"
    python -m relax.ppca_initial_class3d.tests.diagnose_k1_gpu \
        /home/ywang/three_species_particle_stacks/relax_tomo_abinitio_k10_D64/validation/gpu_37074660/pytest/test_spa_k1_complete_updates_m0 \
        "$RUN/diagnostic"
    exit
fi
if [[ "${1:-tests}" == "simulate" ]]; then
    python -m relax.ppca_initial_class3d.tests.validate_k2_cli --output "$RUN/simulated_k2"
    exit
fi
test_status=0
python -m relax.ppca_initial_class3d.tests.run_tests --run-gpu -q \
    -k 'not rank_ten and not spa_k1_complete_updates_match_single_ppca' \
    --basetemp "$RUN/pytest" --junitxml "$RUN/results.xml" || test_status=$?
if [[ "${1:-tests}" == "all" ]]; then
    python -m relax.ppca_initial_class3d.tests.validate_k2_cli --output "$RUN/simulated_k2" || test_status=$?
fi
exit "$test_status"
