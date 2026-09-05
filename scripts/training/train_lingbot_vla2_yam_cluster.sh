#!/usr/bin/env bash
set -euo pipefail

mode=${1:-train}
run_name=${2:-assemble-screwdriver-v1-s0-4xh100-3k}

ROOT=${YAM_TRAIN_ROOT:-/inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORKSPACE=${LINGBOT_VLA2_WORKSPACE:-${REPO_ROOT}}
POLICY=${WORKSPACE}/XPolicyLab/policy/LingBot_VLA2
SOURCE=${POLICY}/lingbot_vla_v2
VENV=${ROOT}/envs/lingbot-vla2/.venv
DATASET=${LINGBOT_VLA2_DATASET_PATH:-${ROOT}/datasets/lerobot/yam_assemble_screwdriver_20260825_v1}
DATASET_NAME=${LINGBOT_VLA2_DATASET_NAME:-$(basename "${DATASET}")}
MODEL=${ROOT}/weights/base/lingbot-vla-v2-6b
TOKENIZER=${ROOT}/weights/base/xiaomi/qwen3_vl_4b_processor
STATS_DIR=${LINGBOT_VLA2_STATS_DIR:-${ROOT}/cache/lingbot-vla2/${DATASET_NAME}}
STATS=${STATS_DIR}/norm_stats.json
OUTPUT=${ROOT}/weights/finetuned/lingbot-vla2/${run_name}
LOG_DIR=${ROOT}/runs/lingbot-vla2
GPU_IDS=${LINGBOT_VLA2_GPU_IDS:-0,1,2,3}
ROBOT_NAME=yam_dual_packed_absolute

export PATH="${ROOT}/envs/bin:${PATH}"
export HF_HOME=${ROOT}/cache/huggingface
export HF_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export UV_CACHE_DIR=${ROOT}/cache/uv
export UV_INDEX_URL=${UV_INDEX_URL:-http://nexus.sii.shaipower.online/repository/pypi/simple}
export UV_INSECURE_HOST=${UV_INSECURE_HOST:-nexus.sii.shaipower.online}
export PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL:-${UV_INDEX_URL}}
export LINGBOT_VLA2_LEROBOT_SPEC=${LINGBOT_VLA2_LEROBOT_SPEC:-lerobot==0.4.2}
export MAX_JOBS=${MAX_JOBS:-16}
export TORCH_HOME=${ROOT}/cache/torch
export LINGBOT_VLA2_ENV_DIR=${VENV}
export LINGBOT_VLA2_MODEL_PATH=${MODEL}
export LINGBOT_VLA2_TOKENIZER_PATH=${TOKENIZER}
export LINGBOT_VLA2_DATASET_PATH=${DATASET}
export LINGBOT_VLA2_NORM_STATS_PATH=${STATS}
export LINGBOT_VLA2_ROBOT_NAME=${ROBOT_NAME}
export LINGBOT_VLA2_ACTION_HORIZON=${LINGBOT_VLA2_ACTION_HORIZON:-50}
export LINGBOT_VLA2_NATIVE_HZ=${LINGBOT_VLA2_NATIVE_HZ:-30}
export LINGBOT_VLA2_TRAIN_WORKERS=${LINGBOT_VLA2_TRAIN_WORKERS:-8}
export LINGBOT_VLA2_MICRO_BATCH_SIZE=${LINGBOT_VLA2_MICRO_BATCH_SIZE:-1}
export LINGBOT_VLA2_GRAD_ACCUM_STEPS=${LINGBOT_VLA2_GRAD_ACCUM_STEPS:-8}
export LINGBOT_VLA2_USE_WANDB=false
export LINGBOT_VLA2_ENABLE_RESUME=false

mkdir -p "${STATS_DIR}" "${LOG_DIR}"

require_file() {
  if [[ ! -e "$1" ]]; then
    echo "Required artifact is missing: $1" >&2
    exit 1
  fi
}

install_environment() {
  if [[ -x "${VENV}/bin/python" ]] && "${VENV}/bin/python" -c \
      'import flash_attn, lingbotvla, mdm, moge, torch, utils3d; assert torch.cuda.is_available()' >/dev/null 2>&1; then
    echo "[LingBot_VLA2] reusing ${VENV}"
    return
  fi
  echo "[LingBot_VLA2] installing environment at ${VENV}"
  bash "${POLICY}/install.sh"
}

compute_stats() {
  if [[ -s "${STATS}" ]]; then
    echo "[LingBot_VLA2] reusing norm stats: ${STATS}"
    return
  fi
  echo "[LingBot_VLA2] computing norm stats from ${DATASET}"
  local stats_gpu=${GPU_IDS%%,*}
  (
    cd "${SOURCE}"
    CUDA_VISIBLE_DEVICES=${stats_gpu} PATH="${VENV}/bin:${PATH}" \
      bash -o pipefail train.sh scripts/compute_norm_stats.py configs/vla/norm_compute/post_data.yaml \
        --data.data_name "${ROBOT_NAME}" \
        --data.robot_name "${ROBOT_NAME}" \
        --data.train_path "${DATASET}" \
        --data.robot_config_root "${POLICY}/robot_configs" \
        --data.norm_path "${STATS}" \
        --data.num_workers "${LINGBOT_VLA2_STATS_WORKERS:-8}" \
        --train.chunk_size "${LINGBOT_VLA2_ACTION_HORIZON}" \
        --train.micro_batch_size "${LINGBOT_VLA2_STATS_BATCH_SIZE:-32}" \
        --train.output_dir "${STATS_DIR}/compute"
  )
  require_file "${STATS}"
}

preflight() {
  require_file "${SOURCE}/tasks/vla/train_lingbotvla.py"
  require_file "${MODEL}/model.safetensors.index.json"
  require_file "${TOKENIZER}/tokenizer.json"
  require_file "${DATASET}/meta/info.json"
  require_file "${POLICY}/robot_configs/${ROBOT_NAME}.yaml"
  require_file "${POLICY}/training/yam_dual.yaml"
  "${VENV}/bin/python" - "${DATASET}" "${STATS}" <<'PY'
import json
import sys
from pathlib import Path

dataset, stats_path = map(Path, sys.argv[1:])
info = json.loads((dataset / "meta/info.json").read_text())
assert info["codebase_version"] == "v3.0"
assert info["total_episodes"] == 19
assert info["total_frames"] == 17789
assert info["features"]["observation.state"]["shape"] == [14]
assert info["features"]["action"]["shape"] == [14]
stats = json.loads(stats_path.read_text())
assert stats["count"] == 17789
for key, width in {
    "observation.state.arm.position": 12,
    "observation.state.effector.position": 2,
    "action.arm.position": 12,
    "action.effector.position": 2,
}.items():
    assert len(stats["norm_stats"][key]["mean"]) == width
print(json.dumps({
    "dataset": dataset.name,
    "episodes": info["total_episodes"],
    "frames": info["total_frames"],
    "fps": info["fps"],
    "gpus": __import__("torch").cuda.device_count(),
}, indent=2))
PY
}

case "${mode}" in
  gate-train)
    LINGBOT_VLA2_MAX_STEPS=1 LINGBOT_VLA2_SAVE_STEPS=1 \
      bash "$0" smoke "${run_name}-smoke"
    require_file "${ROOT}/weights/finetuned/lingbot-vla2/${run_name}-smoke/bundle.yaml"
    LINGBOT_VLA2_MAX_STEPS=3000 LINGBOT_VLA2_SAVE_STEPS=500 \
      bash "$0" train "${run_name}"
    ;;
  prepare)
    install_environment
    preflight
    compute_stats
    ;;
  smoke|train)
    install_environment
    preflight
    compute_stats
    if [[ "${mode}" == "smoke" ]]; then
      export LINGBOT_VLA2_MAX_STEPS=${LINGBOT_VLA2_MAX_STEPS:-1}
      export LINGBOT_VLA2_SAVE_STEPS=${LINGBOT_VLA2_SAVE_STEPS:-1}
    else
      export LINGBOT_VLA2_MAX_STEPS=${LINGBOT_VLA2_MAX_STEPS:-3000}
      export LINGBOT_VLA2_SAVE_STEPS=${LINGBOT_VLA2_SAVE_STEPS:-500}
      if [[ -d "${OUTPUT}" ]] && find "${OUTPUT}" -mindepth 1 -print -quit | grep -q .; then
        echo "Refusing to overwrite existing run: ${OUTPUT}" >&2
        exit 2
      fi
    fi
    export LINGBOT_VLA2_CHECKPOINT_DIR=${OUTPUT}
    CUDA_VISIBLE_DEVICES=${GPU_IDS} bash "${POLICY}/train.sh" \
      yam assemble_the_screwdriver yam_dual joint 0 "${GPU_IDS}" \
      2>&1 | tee "${LOG_DIR}/${run_name}.log"
    ;;
  *)
    echo "Usage: $0 [prepare|smoke|train|gate-train] [run_name]" >&2
    exit 2
    ;;
esac

