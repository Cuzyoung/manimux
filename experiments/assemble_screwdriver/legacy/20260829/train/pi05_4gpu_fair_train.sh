#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=/inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data
CODE_ROOT=/inspire/hdd2/project/liu-ming-huan/public/ziyang/manimux
RUN_NAME=${PI05_RUN_NAME:-assemble-screwdriver-pi05-4xh100-b32-3k-20260829-v1}

export CUDA_VISIBLE_DEVICES=0,1,2,3
export UV_PYTHON=${DATA_ROOT}/envs/uv-python/cpython-3.11-linux-x86_64-gnu/bin/python
export UV_PYTHON_DOWNLOADS=never
export UV_NO_MANAGED_PYTHON=1
unset WANDB_API_KEY WANDB_BASE_URL WANDB_ENTITY WANDB_PROJECT WANDB_MODE

YAM_TRAIN_ROOT=${DATA_ROOT} \
PI05_WORKSPACE=${CODE_ROOT} \
OPENPI_GPU_IDS=0,1,2,3 \
OPENPI_FSDP_DEVICES=4 \
OPENPI_BATCH_SIZE=32 \
OPENPI_NUM_WORKERS=8 \
OPENPI_NUM_TRAIN_STEPS=3000 \
OPENPI_SAVE_INTERVAL=500 \
OPENPI_MAX_TO_KEEP=10 \
OPENPI_LOCAL_CACHE_ROOT=/tmp/openpi-cache-${RUN_NAME} \
bash "${CODE_ROOT}/scripts/train_pi05_yam_cluster.sh" train "${RUN_NAME}"

echo "PI05_4GPU_FAIR_TRAIN_OK ${DATA_ROOT}/weights/finetuned/pi05/${RUN_NAME}"
