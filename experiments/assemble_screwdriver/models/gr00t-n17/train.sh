#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=/inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data
CODE_ROOT=/inspire/hdd2/project/liu-ming-huan/public/ziyang/manimux
POLICY_ROOT=${CODE_ROOT}/XPolicyLab/policy/GR00T_N17
VENV=${DATA_ROOT}/envs/gr00t-n17/.venv
PROCESSED_DATASET=yam-assemble_screwdriver-yam_dual-joint
RUN_NAME=${GR00T_RUN_NAME:-assemble-screwdriver-gr00t-n17-4xh100-3k-20260829-v1}
OUTPUT=${DATA_ROOT}/weights/finetuned/gr00t-n17/${RUN_NAME}
LOG=${DATA_ROOT}/runs/gr00t-n17/${RUN_NAME}.log

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PATH=${DATA_ROOT}/envs/bin:${PATH}
export HF_HOME=${DATA_ROOT}/cache/huggingface
export HF_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export TORCH_HOME=${DATA_ROOT}/cache/torch
export GR00T_ENV_DIR=${VENV}
export GR00T_LEROBOT_HOME=${DATA_ROOT}/datasets/lerobot
export GR00T_BASE_MODEL=${DATA_ROOT}/weights/base/GR00T-N1.7-3B
export GR00T_COSMOS_MODEL=${DATA_ROOT}/weights/base/Cosmos-Reason2-2B
export GR00T_CHECKPOINT_DIR=${OUTPUT}
export GR00T_VIDEO_BACKEND=pyav
export HF_HUB_OFFLINE=1
export NUM_GPUS=4
export MASTER_PORT=29517
export MAX_STEPS=3000
export SAVE_STEPS=1000
export SAVE_TOTAL_LIMIT=3
export GLOBAL_BATCH_SIZE=32
export DATALOADER_NUM_WORKERS=8
export USE_WANDB=0
unset WANDB_API_KEY WANDB_BASE_URL WANDB_ENTITY WANDB_PROJECT WANDB_MODE

test -s "${DATA_ROOT}/datasets/lerobot/${PROCESSED_DATASET}/meta/stats.json"
test -s "${DATA_ROOT}/datasets/lerobot/${PROCESSED_DATASET}/meta/modality.json"
mkdir -p "$(dirname "${LOG}")"

bash -o pipefail "${POLICY_ROOT}/train.sh" \
  yam assemble_screwdriver yam_dual joint 0 0,1,2,3 \
  2>&1 | tee "${LOG}"

echo "GR00T_4GPU_TRAIN_OK ${OUTPUT}"
