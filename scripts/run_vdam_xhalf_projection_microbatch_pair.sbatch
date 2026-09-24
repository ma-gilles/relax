#!/usr/bin/env bash
#SBATCH --job-name=vdam-xhalf-pair
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
# Timing A/B pair in ONE job on two GPUs of one node (no --exclusive): the arms run side by
# side, one per GPU, and swap GPUs between the two scored rounds (crossover), so node load and
# GPU-to-GPU variation hit both arms equally. 180G per arm, as the former single-arm job.
#SBATCH --gres=gpu:h100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-xhalf-pair-%j.out

set -euo pipefail

: "${OUTPUT_ROOT:?set a new SAFE_TO_DELETE pair root}"
: "${DATA_DIR:?set the frozen materialized fixture data directory}"
: "${RELION_OPTIMISER:?set the incoming RELION optimiser checkpoint}"
: "${VDAM_SCORECARD:?set the frozen scorecard}"
: "${CASE_ID:?set the frozen K=1 case id}"
: "${EXPECTED_REPO_HEAD:?pin the tracked source head}"
: "${EXPECTED_CUDA_SHA256:?pin the qualified RECOVAR CUDA binary}"
: "${EXPECTED_RELION_BIND_SHA256:?pin the qualified RELION binding binary}"
: "${RELAX_CUDA_LIB_OVERRIDE:?set the qualified CUDA binary path}"
: "${TARGET_RELION_PERTURBATION_SOURCE:?pin the full-precision perturbation source}"
: "${LAYOUT_EVIDENCE_META:?pin the qualified it80 PROFILE layout evidence}"

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
PIXI_PY=${PIXI_PY_OVERRIDE:-${REPO_ROOT}/.pixi/envs/default/bin/python}
REFERENCE_RELION_DIR=${REFERENCE_RELION_DIR:-$(dirname "${RELION_OPTIMISER}")}
TARGET_GPU_UUID=${TARGET_GPU_UUID:-}
TARGET_ITERATION=${TARGET_ITERATION:-80}
NR_ITER_SCHEDULE=${NR_ITER_SCHEDULE:-200}
TARGET_ORIGINAL_INDEX=${TARGET_ORIGINAL_INDEX:-1002}
TARGET_STACK_INDEX=${TARGET_STACK_INDEX:-1003}
CONTROL_ROW_PIXELS=${CONTROL_ROW_PIXELS:-40000000}
CANDIDATE_ROW_PIXELS=${CANDIDATE_ROW_PIXELS:-80000000}
MEMORY_POLL_SECONDS=${MEMORY_POLL_SECONDS:-0.2}
RELION_BIND_BINARY=${REPO_ROOT}/relax/relion_bind/_relion_bind_core.cpython-311-x86_64-linux-gnu.so

for value_name in TARGET_ITERATION NR_ITER_SCHEDULE CONTROL_ROW_PIXELS CANDIDATE_ROW_PIXELS; do
  value=${!value_name}
  case "${value}" in
    ''|*[!0-9]*) echo "${value_name} must be a positive integer" >&2; exit 2 ;;
  esac
  test "${value}" -ge 1
done
test "${NR_ITER_SCHEDULE}" -ge "${TARGET_ITERATION}"
test "${CONTROL_ROW_PIXELS}" -ne "${CANDIDATE_ROW_PIXELS}"
test ! -e "${OUTPUT_ROOT}"
test ! -e "${OUTPUT_ROOT}_jax_cache_control"
test ! -e "${OUTPUT_ROOT}_jax_cache_candidate"
test -x "${PIXI_PY}"
test -s "${DATA_DIR}/particles.star"
test -s "${RELION_OPTIMISER}"
test -s "${VDAM_SCORECARD}"
test -s "${TARGET_RELION_PERTURBATION_SOURCE}"
test -s "${LAYOUT_EVIDENCE_META}"
test -s "${RELAX_CUDA_LIB_OVERRIDE}"
test -s "${RELION_BIND_BINARY}"
ACTUAL_REPO_HEAD=$(git -C "${REPO_ROOT}" rev-parse HEAD)
EXPECTED_REPO_COMMIT=$(git -C "${REPO_ROOT}" rev-parse "${EXPECTED_REPO_HEAD}^{commit}")
test "${ACTUAL_REPO_HEAD}" = "${EXPECTED_REPO_COMMIT}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
test "$(sha256sum "${RELAX_CUDA_LIB_OVERRIDE}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
test "$(sha256sum "${RELION_BIND_BINARY}" | awk '{print $1}')" = "${EXPECTED_RELION_BIND_SHA256}"

for ((iteration = 1; iteration <= TARGET_ITERATION; ++iteration)); do
  tag=$(printf '%03d' "${iteration}")
  test -s "${REFERENCE_RELION_DIR}/run_it${tag}_data.star"
  test -s "${REFERENCE_RELION_DIR}/run_it${tag}_model.star"
  test -s "${REFERENCE_RELION_DIR}/run_it${tag}_class001.mrc"
done

source "${REPO_ROOT}/scripts/vdam_gpu_selection.sh"
# The two allocated GPUs (or TARGET_GPU_UUIDS=uuid_a,uuid_b); both must be visible.
pair_gpus=()
if [[ -n "${TARGET_GPU_UUIDS:-}" ]]; then
  IFS=',' read -r -a pair_gpus <<< "${TARGET_GPU_UUIDS//[[:space:]]/}"
else
  mapfile -t pair_gpus < <(vdam_visible_gpu_uuids)
fi
if [[ "${#pair_gpus[@]}" -ne 2 || "${pair_gpus[0]}" == "${pair_gpus[1]}" ]]; then
  echo "the pair needs exactly two distinct allocated GPUs; observed: ${pair_gpus[*]:-none}" >&2
  exit 2
fi
for pair_gpu in "${pair_gpus[@]}"; do
  vdam_select_target_gpu "${pair_gpu}" 0
done
gpu_a=${pair_gpus[0]}
gpu_b=${pair_gpus[1]}

cache_control=${OUTPUT_ROOT}_jax_cache_control
cache_candidate=${OUTPUT_ROOT}_jax_cache_candidate
mkdir -p "${OUTPUT_ROOT}/provenance" "${OUTPUT_ROOT}/audits" \
  "${cache_control}" "${cache_candidate}"
touch "${OUTPUT_ROOT}/SAFE_TO_DELETE" "${cache_control}/SAFE_TO_DELETE" \
  "${cache_candidate}/SAFE_TO_DELETE"
printf '%s\n' "${ACTUAL_REPO_HEAD}" > "${OUTPUT_ROOT}/provenance/repo_head.txt"
printf '%s\n%s\n' "${gpu_a}" "${gpu_b}" > "${OUTPUT_ROOT}/provenance/pair_gpu_uuids.txt"
printf '%s\n' "${VDAM_VISIBLE_GPU_UUIDS_CSV}" > "${OUTPUT_ROOT}/provenance/allocated_gpu_uuids.csv"
printf 'control\t%s\ncandidate\t%s\n' "${CONTROL_ROW_PIXELS}" "${CANDIDATE_ROW_PIXELS}" \
  > "${OUTPUT_ROOT}/provenance/arms.tsv"
sha256sum "${RELAX_CUDA_LIB_OVERRIDE}" "${RELION_BIND_BINARY}" \
  "${LAYOUT_EVIDENCE_META}" > "${OUTPUT_ROOT}/provenance/input_evidence.sha256"
nvidia-smi -q > "${OUTPUT_ROOT}/provenance/nvidia_smi.txt"
env | LC_ALL=C sort > "${OUTPUT_ROOT}/provenance/submission_environment.txt"

# Retain only the pinned CUDA source override; every RECOVAR science tuning is
# cleared and PROFILE remains unset. The row-pixel target below is the sole
# execution-policy difference between arms.
while IFS='=' read -r name _; do
  case "${name}" in
    RELAX_CUDA_LIB_OVERRIDE) ;;
    RECOVAR_*|RELAX_*) unset "${name}" ;;
  esac
done < <(env)
unset RECOVAR_INITIAL_MODEL_PROFILE
unset RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE
unset RECOVAR_K1_COARSE_FUSED_PROJECTOR
unset RELAX_K1_COARSE_GAUSSIAN_SKIP_PADDED_IMAGES
unset RECOVAR_K1_COARSE_GAUSSIAN_FFI
unset RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF
unset RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS
unset RECOVAR_K1_RELION_F32_COARSE_SUPPORT
unset RECOVAR_RELION_COARSE_CANONICAL_REDUCTION
unset RELAX_EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS
env | LC_ALL=C sort > "${OUTPUT_ROOT}/provenance/production_default_environment.txt"

# Crossover schedule: round 0 warms each arm's compilation cache, rounds 1 and 2 are scored.
# Within a round the two arms run concurrently, one per GPU; between the scored rounds they
# swap GPUs. execution_order.tsv records the GPU of every run.
: > "${OUTPUT_ROOT}/provenance/execution_order.tsv"
run_label() {
  local label=$1 arm=$2 run_gpu=$3
  local row_pixels cache_dir run_root memory_log start_ns run_pid monitor_pid run_status end_ns elapsed_ns
  if [[ "${arm}" == control ]]; then
    row_pixels=${CONTROL_ROW_PIXELS}
    cache_dir=${cache_control}
  else
    row_pixels=${CANDIDATE_ROW_PIXELS}
    cache_dir=${cache_candidate}
  fi
  export RELAX_EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS=${row_pixels}
  unset RECOVAR_INITIAL_MODEL_PROFILE

  run_root=${OUTPUT_ROOT}/${label}
  mkdir -p "${run_root}/telemetry"
  touch "${run_root}/SAFE_TO_DELETE"
  memory_log=${run_root}/telemetry/gpu_memory.tsv
  : > "${memory_log}"
  start_ns=$(date +%s%N)
  OUTPUT_ROOT="${run_root}" \
  DATA_DIR="${DATA_DIR}" \
  RELION_OPTIMISER="${RELION_OPTIMISER}" \
  REPO_ROOT="${REPO_ROOT}" \
  PIXI_PY_OVERRIDE="${PIXI_PY}" \
  EXPECTED_REPO_HEAD="${ACTUAL_REPO_HEAD}" \
  EXPECTED_CUDA_SHA256="${EXPECTED_CUDA_SHA256}" \
  EXPECTED_RELION_BIND_SHA256="${EXPECTED_RELION_BIND_SHA256}" \
  TARGET_GPU_UUID="${run_gpu}" \
  TARGET_ITERATION="${TARGET_ITERATION}" \
  NR_ITER_SCHEDULE="${NR_ITER_SCHEDULE}" \
  VDAM_SCORECARD="${VDAM_SCORECARD}" \
  CASE_ID="${CASE_ID}" \
  TARGET_ORIGINAL_INDEX="${TARGET_ORIGINAL_INDEX}" \
  TARGET_RELION_PART_ID=-1 \
  TARGET_STACK_INDEX="${TARGET_STACK_INDEX}" \
  TARGET_RELION_PERTURBATION_SOURCE="${TARGET_RELION_PERTURBATION_SOURCE}" \
  CAPTURE_LOCAL_SCORE=0 \
  CAPTURE_FUSED_SCORES=0 \
  CAPTURE_COARSE_SCORE=0 \
  CAPTURE_NATIVE_REPLAY=0 \
  BOUNDARY_DETERMINISTIC_CUDA=0 \
  VDAM_JAX_COMPILATION_CACHE_DIR="${cache_dir}" \
    bash "${REPO_ROOT}/scripts/run_vdam_first_state_boundary_capture.sbatch" &
  run_pid=$!
  (
    while kill -0 "${run_pid}" 2>/dev/null; do
      sample_ns=$(date +%s%N)
      memory_values=$(nvidia-smi --id="${run_gpu}" \
        --query-gpu=memory.used,memory.total --format=csv,noheader,nounits)
      printf '%s\t%s\n' "${sample_ns}" "${memory_values//, /$'\t'}"
      sleep "${MEMORY_POLL_SECONDS}"
    done
  ) >> "${memory_log}" &
  monitor_pid=$!
  set +e
  wait "${run_pid}"
  run_status=$?
  set -e
  wait "${monitor_pid}" || true
  end_ns=$(date +%s%N)
  elapsed_ns=$((end_ns - start_ns))
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${label}" "${arm}" "${row_pixels}" "${elapsed_ns}" \
    "${run_gpu}" "${cache_dir}" "${run_status}" \
    >> "${OUTPUT_ROOT}/provenance/execution_order.tsv"
  if [[ "${run_status}" -ne 0 ]]; then
    echo "${label} failed with status ${run_status}; stop the cap escalation" >&2
    return "${run_status}"
  fi
  printf '%s\n' "${row_pixels}" > "${run_root}/provenance/xhalf_target_row_pixels.txt"

  grep -q "RELION fused coarse projector/diff2 enabled (guarded fresh InitialModel default)" \
    "${run_root}/recovar_prefix/runner.stderr"
  if grep -q "native texture coarse scoring enabled" \
      "${run_root}/recovar_prefix/runner.stderr"; then
    echo "native-texture diagnostic unexpectedly replaced the production route" >&2
    return 2
  fi
  if grep -Eq "(coarse Gaussian FFI|coarse CUDA sincosf|exact coarse operands|fused coarse projector/diff2) enabled \(environment override\)" \
      "${run_root}/recovar_prefix/runner.stderr"; then
    echo "production coarse route was forced by an environment override" >&2
    return 2
  fi
  vdam_verify_selected_gpu "${run_gpu}"
}
run_round() {
  local status_a status_b
  run_label "$1" "$2" "${gpu_a}" & local pid_a=$!
  run_label "$3" "$4" "${gpu_b}" & local pid_b=$!
  set +e
  wait "${pid_a}"; status_a=$?
  wait "${pid_b}"; status_b=$?
  set -e
  if [[ "${status_a}" -ne 0 || "${status_b}" -ne 0 ]]; then
    echo "round $1/$3 failed (${status_a}/${status_b}); stop the cap escalation" >&2
    exit 1
  fi
}
run_round warmup-control control warmup-candidate candidate
run_round scored-control-1 control scored-candidate-1 candidate
run_round scored-candidate-2 candidate scored-control-2 control
unset RELAX_EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS

iterations=()
for ((iteration = 1; iteration <= TARGET_ITERATION; ++iteration)); do
  iterations+=("${iteration}")
done
pair_names=(cross-1 cross-2 control-repeat candidate-repeat)
lhs_labels=(scored-candidate-1 scored-candidate-2 scored-control-1 scored-candidate-1)
rhs_labels=(scored-control-1 scored-control-2 scored-control-2 scored-candidate-2)
for index in "${!pair_names[@]}"; do
  name=${pair_names[index]}
  lhs=${lhs_labels[index]}
  rhs=${rhs_labels[index]}
  (
    cd "${REPO_ROOT}"
    "${PIXI_PY}" -m scripts.audit_vdam_particle_state_trajectory \
      --recovar-dir "${OUTPUT_ROOT}/${lhs}/recovar_prefix" \
      --relion-dir "${OUTPUT_ROOT}/${rhs}/recovar_prefix" \
      --output "${OUTPUT_ROOT}/audits/${name}_state.json" \
      --iterations "${iterations[@]}"
  ) > "${OUTPUT_ROOT}/audits/${name}_state.stdout"
done
for label in scored-control-1 scored-control-2 scored-candidate-1 scored-candidate-2; do
  (
    cd "${REPO_ROOT}"
    "${PIXI_PY}" -m scripts.audit_vdam_particle_state_trajectory \
      --recovar-dir "${OUTPUT_ROOT}/${label}/recovar_prefix" \
      --relion-dir "${REFERENCE_RELION_DIR}" \
      --output "${OUTPUT_ROOT}/audits/${label}_vs_relion_state.json" \
      --iterations "${iterations[@]}"
  ) > "${OUTPUT_ROOT}/audits/${label}_vs_relion_state.stdout"
done

(
  cd "${REPO_ROOT}"
  "${PIXI_PY}" -m scripts.summarize_vdam_xhalf_projection_microbatch_pair \
    --root "${OUTPUT_ROOT}" \
    --relion-dir "${REFERENCE_RELION_DIR}" \
    --layout-evidence-meta "${LAYOUT_EVIDENCE_META}" \
    --target-iteration "${TARGET_ITERATION}" \
    --check
) | tee "${OUTPUT_ROOT}/summary.stdout"

touch "${OUTPUT_ROOT}/COMPLETED"
