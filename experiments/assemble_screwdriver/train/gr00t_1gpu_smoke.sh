#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=/inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data
CODE_ROOT=/inspire/hdd2/project/liu-ming-huan/public/ziyang/manimux
POLICY_ROOT=${CODE_ROOT}/XPolicyLab/policy/GR00T_N17
VENV=${DATA_ROOT}/envs/gr00t-n17/.venv
SOURCE_DATASET=yam_assemble_screwdriver_20260825_v1
PROCESSED_DATASET=yam-assemble_screwdriver-yam_dual-joint
RUN_NAME=${GR00T_RUN_NAME:-assemble-screwdriver-gr00t-n17-1xh100-smoke-20260829-v1}
OUTPUT=${DATA_ROOT}/weights/finetuned/gr00t-n17/${RUN_NAME}
LOG=${DATA_ROOT}/runs/gr00t-n17/${RUN_NAME}.log

export CUDA_VISIBLE_DEVICES=0
export PATH=${DATA_ROOT}/envs/bin:${PATH}
export HF_HOME=${DATA_ROOT}/cache/huggingface
export HF_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export TORCH_HOME=${DATA_ROOT}/cache/torch
export GR00T_ENV_DIR=${VENV}
export GR00T_LEROBOT_HOME=${DATA_ROOT}/datasets/lerobot
export GR00T_SRC_DATASET=${SOURCE_DATASET}
export GR00T_BASE_MODEL=${DATA_ROOT}/weights/base/GR00T-N1.7-3B
export GR00T_COSMOS_MODEL=${DATA_ROOT}/weights/base/Cosmos-Reason2-2B
export GR00T_CHECKPOINT_DIR=${OUTPUT}
export GR00T_VIDEO_BACKEND=pyav
export HF_HUB_OFFLINE=1
export NUM_GPUS=1
export MAX_STEPS=1
export SAVE_STEPS=1
export SAVE_TOTAL_LIMIT=1
export GLOBAL_BATCH_SIZE=1
export DATALOADER_NUM_WORKERS=0
export USE_WANDB=0
unset WANDB_API_KEY WANDB_BASE_URL WANDB_ENTITY WANDB_PROJECT WANDB_MODE

mkdir -p "$(dirname "${LOG}")"

if [[ ! -s "${DATA_ROOT}/datasets/lerobot/${PROCESSED_DATASET}/meta/stats.json" ]] || \
   [[ ! -s "${DATA_ROOT}/datasets/lerobot/${PROCESSED_DATASET}/meta/modality.json" ]]; then
  bash "${POLICY_ROOT}/process_data.sh" yam assemble_screwdriver yam_dual joint
fi

bash -o pipefail "${POLICY_ROOT}/train.sh" \
  yam assemble_screwdriver yam_dual joint 0 0 \
  2>&1 | tee "${LOG}"

CHECKPOINT=$(find "${OUTPUT}" -type d -name checkpoint-1 -print -quit)
test -n "${CHECKPOINT}"
"${VENV}/bin/python" - "${CHECKPOINT}" <<'PY'
import sys
import torch
import gr00t.model  # noqa: F401
from transformers import AutoModel

checkpoint = sys.argv[1]
model = AutoModel.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
model.to("cuda:0")
print(f"GR00T_STRICT_RELOAD_OK {checkpoint} {type(model).__name__}")
PY

echo "GR00T_1GPU_SMOKE_OK ${CHECKPOINT}"
