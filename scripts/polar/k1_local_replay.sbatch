#!/bin/bash
#SBATCH --job-name=relax-polar-k1-local
#SBATCH --partition=main
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00

set -euo pipefail
: "${POLAR_SOURCE_ROOT:?submit with scripts/polar/submit.py}"
: "${POLAR_ENV_ROOT:?submit with scripts/polar/submit.py}"
: "${POLAR_RUN_ROOT:?submit with scripts/polar/submit.py}"
: "${POLAR_FIXTURE_ROOT:?stage K1 fixtures on Polar network scratch}"
: "${POLAR_GIT_BUNDLE:?stage this checkout's Git history}"
: "${POLAR_GIT_BUNDLE_SHA256:?pass the Git bundle SHA256}"
: "${POLAR_NATIVE_ROOT:?pass the native build's natives directory}"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu
export PATH="$POLAR_ENV_ROOT/bin:$PATH"
export RECOVAR_JAX_CACHE_DIR="$POLAR_RUN_ROOT/jax-cache"
export JAX_COMPILATION_CACHE_DIR="$RECOVAR_JAX_CACHE_DIR"
export XDG_CACHE_HOME="$POLAR_RUN_ROOT/cache" TMPDIR="$POLAR_RUN_ROOT/tmp"
export RECOVAR_CUDA_LIB="$POLAR_NATIVE_ROOT/libcuda_backproject.so"
export RELAX_CUDA_LIB="$POLAR_NATIVE_ROOT/librelax_cuda.so"
export RECOVAR_RELION_BIND_BUILD_DIR="$POLAR_NATIVE_ROOT/relion_bind"
export RELAX_REQUIRE_CUSTOM_CUDA_FOR_TESTS=1 RELAX_ENABLE_CUSTOM_CUDA=1
mkdir -p "$RECOVAR_JAX_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"

printf 'job=%s node=%s visible_gpu=%s\n' "$SLURM_JOB_ID" "$(hostname)" "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
printf '%s  %s\n' "$POLAR_GIT_BUNDLE_SHA256" "$POLAR_GIT_BUNDLE" | sha256sum -c -
(cd "$POLAR_FIXTURE_ROOT" && sha256sum -c .polar-sha256 >/dev/null)
(cd "$POLAR_NATIVE_ROOT" && sha256sum -c binaries.sha256)

cp -a "$POLAR_SOURCE_ROOT" "$POLAR_RUN_ROOT/source"
chmod -R u+w "$POLAR_RUN_ROOT/source"
mkdir -p "$POLAR_RUN_ROOT/source/.pixi/envs"
ln -s "$POLAR_ENV_ROOT" "$POLAR_RUN_ROOT/source/.pixi/envs/default"
git -C "$POLAR_RUN_ROOT/source" init -q
git -C "$POLAR_RUN_ROOT/source" fetch -q "$POLAR_GIT_BUNDLE" HEAD
git -C "$POLAR_RUN_ROOT/source" reset -q --mixed FETCH_HEAD
test "$(git -C "$POLAR_RUN_ROOT/source" rev-parse HEAD)" = \
    "$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["head"])' "$POLAR_SOURCE_ROOT/.polar-source.json")"

export PYTHONPATH="$POLAR_RUN_ROOT/source"
cd "$POLAR_RUN_ROOT/source"
python scripts/polar/prepare_replay.py "$POLAR_FIXTURE_ROOT" "$PWD" "$POLAR_RUN_ROOT"
python tests/helpers/em_fixtures.py k1_5k128_data k1_5k128_relion_os0
python scripts/native_sources.py check "$POLAR_NATIVE_ROOT" --root "$PWD"
python scripts/native_sources.py imports "$PWD"
git status --short --branch

set +e
python -m pytest -v -s --run-gpu --run-integration --run-slow \
    --basetemp="$POLAR_RUN_ROOT/pytest" \
    tests/integration/test_em_parity_fast.py::test_em_parity_fast_k1_local_replay \
    > "$POLAR_RUN_ROOT/pytest.log" 2>&1
status=$?
set -e
tail -n 120 "$POLAR_RUN_ROOT/pytest.log"
(cd "$POLAR_NATIVE_ROOT" && sha256sum -c binaries.sha256)
printf '%s\n' "$status" > "$POLAR_RUN_ROOT/pytest.exit_code"
exit "$status"
