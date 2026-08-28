#!/usr/bin/env bash
set -euo pipefail

mode=${1:-train}
run_name=${2:-pick-red-ball-box-v1-s0-4xh100-10k}

ROOT=${YAM_TRAIN_ROOT:-/inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data}
WORKSPACE=${XR1_WORKSPACE:-${ROOT}/operate/manimux-training}
POLICY=${WORKSPACE}/XPolicyLab/policy/Xiaomi_Robotics_1
XR1=${POLICY}/xiaomi_robotics_1/xr1
VENV=${ROOT}/envs/xr1/.venv
DATASET=${XR1_DATASET_PATH:-${ROOT}/datasets/xr1/RoboDojo_real-pick_red_ball_box-yam_dual-ee}
BASE_MODEL=${ROOT}/weights/base/xiaomi/model_states.pt
PROCESSOR=${ROOT}/weights/base/xiaomi/qwen3_vl_4b_processor
OUTPUT=${ROOT}/weights/finetuned/xiaomi-xr1/${run_name}
LOG_DIR=${ROOT}/runs/xiaomi-xr1
GPU_IDS=${XR1_GPU_IDS:-0,1,2,3}
DATA_CONFIG_NAME=${XR1_DATA_CONFIG_NAME:-yam_pick_red_ball_box}
TASK_NAME=${XR1_TASK_NAME:-pick_red_ball_box}
INSTRUCTION=${XR1_INSTRUCTION:-Pick the red ball up and place it into the box.}

export PATH="${ROOT}/envs/bin:${PATH}"
export HF_HOME=${ROOT}/cache/huggingface
export HF_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_CACHE_DIR=${ROOT}/cache/uv
export UV_INDEX_URL=${UV_INDEX_URL:-http://nexus.sii.shaipower.online/repository/pypi/simple}
export UV_INSECURE_HOST=${UV_INSECURE_HOST:-nexus.sii.shaipower.online}
export TORCH_HOME=${ROOT}/cache/torch
export XR1_QWEN_VL_CONFIG_SOURCE=${PROCESSOR}
export XR1_LOGGER=${XR1_LOGGER:-tensorboard}
export MAX_LENGTH=${XR1_MAX_LENGTH:-20000}

mkdir -p "${VENV%/.venv}" "${DATASET}" "${OUTPUT}" "${LOG_DIR}"

require_file() {
  if [[ ! -e "$1" ]]; then
    echo "Required artifact is missing: $1" >&2
    exit 1
  fi
}

find_episodes() {
  if [[ -n "${XR1_YAM_EPISODES:-}" ]]; then
    printf '%s\n' "${XR1_YAM_EPISODES}"
    return
  fi
  mapfile -t candidates < <(
    find "${ROOT}/datasets" -maxdepth 6 -type d \
      -name pick_the_red_ball_up_and_place_it_into_the_box | sort
  )
  if [[ "${#candidates[@]}" -ne 1 ]]; then
    echo "Expected exactly one raw red-ball episode directory, found ${#candidates[@]}:" >&2
    printf '  %s\n' "${candidates[@]}" >&2
    echo "Set XR1_YAM_EPISODES explicitly." >&2
    exit 1
  fi
  printf '%s\n' "${candidates[0]}"
}

install_environment() {
  if [[ -x "${VENV}/bin/python" ]] && "${VENV}/bin/python" -c \
      'import decord, deepspeed, flash_attn, lightning, mmengine, torch, transformers; assert torch.cuda.is_available()' >/dev/null 2>&1; then
    echo "[Xiaomi_Robotics_1] reusing ${VENV}"
    return
  fi
  command -v uv >/dev/null
  if [[ ! -x "${VENV}/bin/python" ]]; then
    uv venv --python 3.12 --system-site-packages "${VENV}"
  elif ! grep -q '^include-system-site-packages = true$' "${VENV}/pyvenv.cfg"; then
    # The official H100 image already provides its CUDA-matched PyTorch build.
    # Keep that binary stack and install only XR1's Python-level dependencies.
    sed -i 's/^include-system-site-packages = false$/include-system-site-packages = true/' \
      "${VENV}/pyvenv.cfg"
  fi
  export UV_LINK_MODE=copy
  export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-600}
  uv pip install --python "${VENV}/bin/python" pip setuptools wheel packaging ninja
  # Nexus mirrors the official PyTorch wheels. Pin the ABI before resolving
  # higher-level packages so a newer torch cannot invalidate FlashAttention.
  uv pip install --python "${VENV}/bin/python" \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
  uv pip install --python "${VENV}/bin/python" \
    lightning==2.5.3 deepspeed==0.18.9 transformers==4.57.1 \
    accelerate==1.11.0 safetensors==0.6.2 liger-kernel==0.6.5 \
    numpy==2.1.3 Pillow==11.3.0 decord==0.6.0 mmengine==0.10.7 \
    omegaconf==2.3.0 hydra-core==1.3.2 wandb==0.23.1 tensorboard==2.20.0
  uv pip install --python "${VENV}/bin/python" \
    flash-attn==2.8.3 --no-build-isolation
  uv pip install --python "${VENV}/bin/python" -e "${WORKSPACE}/XPolicyLab"
  "${VENV}/bin/python" - <<'PY'
import decord
import deepspeed
import flash_attn
import lightning
import mmengine
import torch
import transformers

assert torch.cuda.is_available()
print({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "flash_attn": flash_attn.__version__,
    "transformers": transformers.__version__,
})
PY
}

prepare_data() {
  local episodes
  episodes=$(find_episodes)
  echo "[Xiaomi_Robotics_1] raw episodes=${episodes}"
  PYTHONPATH="${WORKSPACE}/src" "${VENV}/bin/python" \
    "${WORKSPACE}/scripts/prepare_xr1_yam_dataset.py" \
    --episodes "${episodes}" \
    --output "${DATASET}" \
    --instruction "${INSTRUCTION}" \
    --batch-size "${XR1_MICRO_BATCH_SIZE:-1}"
  cp "${DATASET}/yam_pick_red_ball_box.yaml" \
    "${XR1}/configs/data/${DATA_CONFIG_NAME}.yaml"
}

preflight() {
  require_file "${WORKSPACE}/scripts/prepare_xr1_yam_dataset.py"
  require_file "${BASE_MODEL}"
  require_file "${PROCESSOR}/config.json"
  require_file "${PROCESSOR}/tokenizer.json"
  require_file "${DATASET}/manifest.json"
  require_file "${DATASET}/norm_stats.json"
  require_file "${XR1}/configs/data/${DATA_CONFIG_NAME}.yaml"
  "${VENV}/bin/python" - "${DATASET}" "${BASE_MODEL}" <<'PY'
import json
import sys
from pathlib import Path

dataset, model = map(Path, sys.argv[1:])
manifest = json.loads((dataset / "manifest.json").read_text())
assert manifest["schema"] == "manimux.xr1_yam_dataset.v1"
assert manifest["episodes"] > 0
assert manifest["frames"] > 0
assert model.stat().st_size > 9_000_000_000
print(json.dumps({
    "dataset": str(dataset),
    "episodes": manifest["episodes"],
    "frames": manifest["frames"],
    "model_bytes": model.stat().st_size,
    "gpus": __import__("torch").cuda.device_count(),
}, indent=2))
PY
}

case "${mode}" in
  prepare)
    install_environment
    prepare_data
    preflight
    ;;
  smoke|train)
    install_environment
    prepare_data
    preflight
    if [[ "${mode}" == smoke ]]; then
      max_steps=${XR1_MAX_STEPS:-2}
      save_interval=${XR1_SAVE_INTERVAL:-2}
    else
      max_steps=${XR1_MAX_STEPS:-10000}
      save_interval=${XR1_SAVE_INTERVAL:-1000}
    fi
    PATH="${VENV}/bin:${PATH}" \
    OUTPUT_DIR="${DATASET}" \
    DATA_CONFIG_NAME="${DATA_CONFIG_NAME}" \
    PRETRAINED_PATH="${BASE_MODEL}" \
    RUN_ROOT="${OUTPUT}" \
    PROJECT="yam-xiaomi-xr1" \
    MAX_STEPS="${max_steps}" \
    SAVE_INTERVAL="${save_interval}" \
    XR1_LOGGER="${XR1_LOGGER}" \
    bash "${POLICY}/train.sh" \
      RoboDojo_real "${TASK_NAME}" yam_dual ee 0 "${GPU_IDS}" \
      trainer.accumulate_grad_batches="${XR1_GRAD_ACCUM_STEPS:-8}" \
      2>&1 | tee "${LOG_DIR}/${run_name}.log"
    ;;
  *)
    echo "Usage: $0 [prepare|smoke|train] [run_name]" >&2
    exit 2
    ;;
esac
