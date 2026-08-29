#!/usr/bin/env bash
set -euo pipefail

mode=${1:-train}
run_name=${2:-assemble-screwdriver-v1-s0-4xh100-3k}

ROOT=${YAM_TRAIN_ROOT:-/inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORKSPACE=${PI05_WORKSPACE:-${REPO_ROOT}}
POLICY=${WORKSPACE}/XPolicyLab/policy/Pi_05
OPENPI=${POLICY}/openpi
VENV=${OPENPI_ENV_DIR:-${ROOT}/envs/pi05/.venv}
DATASET_NAME=${OPENPI_LEROBOT_REPO_ID:-yam_assemble_screwdriver_20260825_v1}
DATASET=${ROOT}/datasets/lerobot/${DATASET_NAME}
BASE_PARAMS=${OPENPI_BASE_PARAMS:-${ROOT}/weights/base/pi05_base/params}
ASSETS_BASE=${OPENPI_ASSETS_BASE_DIR:-${ROOT}/cache/pi05/assets}
NORM_STATS=${ASSETS_BASE}/pi05_yam/${DATASET_NAME}/norm_stats.json
OUTPUT=${ROOT}/weights/finetuned/pi05/${run_name}
LOG_DIR=${ROOT}/runs/pi05
GPU_IDS=${OPENPI_GPU_IDS:-0,1,2,3}

export PATH="${ROOT}/envs/bin:${PATH}"
export UV_CACHE_DIR=${ROOT}/cache/uv
export UV_PROJECT_ENVIRONMENT=${VENV}
export PYTHONPATH="${OPENPI}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME=${ROOT}/cache/huggingface
export HF_LEROBOT_HOME=${ROOT}/datasets/lerobot
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENPI_DATA_HOME=${ROOT}/cache/openpi
export OPENPI_LOCAL_CACHE_ROOT=${OPENPI_LOCAL_CACHE_ROOT:-${ROOT}/cache/pi05/${HOSTNAME}}
export OPENPI_TRAIN_CONFIG_NAME=pi05_yam
export OPENPI_LEROBOT_REPO_ID=${DATASET_NAME}
export OPENPI_BASE_PARAMS=${BASE_PARAMS}
export OPENPI_ASSETS_BASE_DIR=${ASSETS_BASE}
export OPENPI_FSDP_DEVICES=${OPENPI_FSDP_DEVICES:-4}
export OPENPI_NUM_WORKERS=${OPENPI_NUM_WORKERS:-0}
export OPENPI_WANDB_ENABLED=false

mkdir -p "${LOG_DIR}" "${ROOT}/cache/pi05/${HOSTNAME}"

compute_stats() {
  if [[ -s "${NORM_STATS}" ]]; then
    echo "[Pi_05] reusing norm stats: ${NORM_STATS}"
    return
  fi
  echo "[Pi_05] computing norm stats from ${DATASET}"
  (
    cd "${OPENPI}"
    OPENPI_LEROBOT_REPO_ID=${DATASET_NAME} \
    OPENPI_ASSETS_BASE_DIR=${ASSETS_BASE} \
    OPENPI_BASE_PARAMS=${BASE_PARAMS} \
      "${VENV}/bin/python" scripts/compute_norm_stats.py --config-name pi05_yam
  )
}

case "${mode}" in
  prepare)
    compute_stats
    ;;
  smoke|train)
    compute_stats
    if [[ "${mode}" == "smoke" ]]; then
      export OPENPI_BATCH_SIZE=${OPENPI_BATCH_SIZE:-4}
      export OPENPI_NUM_TRAIN_STEPS=${OPENPI_NUM_TRAIN_STEPS:-1}
      export OPENPI_SAVE_INTERVAL=${OPENPI_SAVE_INTERVAL:-1}
      export OPENPI_MAX_TO_KEEP=${OPENPI_MAX_TO_KEEP:-1}
      export OPENPI_CHECKPOINT_DIR=${OUTPUT}-smoke
    else
      export OPENPI_BATCH_SIZE=${OPENPI_BATCH_SIZE:-32}
      export OPENPI_NUM_TRAIN_STEPS=${OPENPI_NUM_TRAIN_STEPS:-3000}
      export OPENPI_SAVE_INTERVAL=${OPENPI_SAVE_INTERVAL:-500}
      export OPENPI_MAX_TO_KEEP=${OPENPI_MAX_TO_KEEP:-10}
      export OPENPI_CHECKPOINT_DIR=${OUTPUT}
    fi
    bash "${POLICY}/train.sh" \
      yam assemble_the_screwdriver yam_dual joint 0 "${GPU_IDS}" \
      2>&1 | tee "${LOG_DIR}/${run_name}-${mode}.log"
    ;;
  *)
    echo "Usage: $0 [prepare|smoke|train] [run_name]" >&2
    exit 2
    ;;
esac
