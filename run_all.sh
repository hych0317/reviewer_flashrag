#!/usr/bin/env bash
set -u

# Run this script from a terminal where the target conda environment is active.
# It will relaunch itself inside a detached screen session if needed.

SCREEN_NAME="reviewer_experiments"
GPU_ID="1"
GENERATOR_MODEL="qwen3-4B-Instruct"
METHOD_NAME="reviewer_reasoning"
MAX_ITER="2"
MAX_DOCS="8"
SAMPLE_SIZE="1000"

# Format: "dataset_name:split"
DATASETS=(
  "hotpotqa:dev"
  "2wikimultihopqa:dev"
  "musique:dev"
  # "nq:test"
  "triviaqa:test"
  # "popqa:test"
  # "strategyqa:train"
  "ambig_qa:dev"
  # "asqa:dev"
  # "fever:dev"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"

if [[ "${1:-}" != "--inside-screen" ]]; then
  if [[ -z "${STY:-}" ]]; then
    mkdir -p "${LOG_DIR}"
    echo "Starting detached screen session: ${SCREEN_NAME}"
    echo "Attach with: screen -r ${SCREEN_NAME}"
    screen -S "${SCREEN_NAME}" -dm bash "$0" --inside-screen
    exit 0
  fi
fi

mkdir -p "${LOG_DIR}"
cd "${SCRIPT_DIR}" || exit 1

RUN_ID="$(date +%Y%m%d_%H%M%S)"
SUMMARY_LOG="${LOG_DIR}/run_all_${RUN_ID}.summary.log"

echo "Run ID: ${RUN_ID}" | tee -a "${SUMMARY_LOG}"
echo "Working directory: ${SCRIPT_DIR}" | tee -a "${SUMMARY_LOG}"
echo "Method: ${METHOD_NAME}" | tee -a "${SUMMARY_LOG}"
echo "Generator model: ${GENERATOR_MODEL}" | tee -a "${SUMMARY_LOG}"
echo "GPU: ${GPU_ID}" | tee -a "${SUMMARY_LOG}"
echo "Max iter: ${MAX_ITER}" | tee -a "${SUMMARY_LOG}"
echo "Max docs: ${MAX_DOCS}" | tee -a "${SUMMARY_LOG}"
echo "Sample size: ${SAMPLE_SIZE}" | tee -a "${SUMMARY_LOG}"
echo "" | tee -a "${SUMMARY_LOG}"

SUCCESS=()
FAILED=()

for entry in "${DATASETS[@]}"; do
  dataset="${entry%%:*}"
  split="${entry#*:}"
  exp_name="${dataset}_${split}_${RUN_ID}"
  exp_log="${LOG_DIR}/${exp_name}.log"

  echo "============================================================" | tee -a "${SUMMARY_LOG}"
  echo "Starting ${dataset} (${split})" | tee -a "${SUMMARY_LOG}"
  echo "Log: ${exp_log}" | tee -a "${SUMMARY_LOG}"
  echo "Start time: $(date '+%F %T')" | tee -a "${SUMMARY_LOG}"

  python ./run_reviewer.py \
    --method_name "${METHOD_NAME}" \
    --dataset_name "${dataset}" \
    --split "${split}" \
    --gpu_id "${GPU_ID}" \
    --max_iter "${MAX_ITER}" \
    --max_docs "${MAX_DOCS}" \
    --generator_model "${GENERATOR_MODEL}" \
    --sample_size "${SAMPLE_SIZE}" \
    > "${exp_log}" 2>&1

  status=$?
  if [[ ${status} -eq 0 ]]; then
    SUCCESS+=("${dataset}:${split}")
    echo "Finished ${dataset} (${split}) successfully." | tee -a "${SUMMARY_LOG}"
  else
    FAILED+=("${dataset}:${split}:exit_${status}")
    echo "Failed ${dataset} (${split}) with exit code ${status}." | tee -a "${SUMMARY_LOG}"
  fi
  echo "End time: $(date '+%F %T')" | tee -a "${SUMMARY_LOG}"
  echo "" | tee -a "${SUMMARY_LOG}"
done

echo "============================================================" | tee -a "${SUMMARY_LOG}"
echo "All configured experiments completed." | tee -a "${SUMMARY_LOG}"
echo "Successful: ${#SUCCESS[@]}" | tee -a "${SUMMARY_LOG}"
for item in "${SUCCESS[@]}"; do
  echo "  OK ${item}" | tee -a "${SUMMARY_LOG}"
done
echo "Failed: ${#FAILED[@]}" | tee -a "${SUMMARY_LOG}"
for item in "${FAILED[@]}"; do
  echo "  FAIL ${item}" | tee -a "${SUMMARY_LOG}"
done
echo "Summary log: ${SUMMARY_LOG}" | tee -a "${SUMMARY_LOG}"
