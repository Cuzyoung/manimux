#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=/inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data
CODE_ROOT=/inspire/hdd2/project/liu-ming-huan/public/ziyang/manimux
RUN_NAME=${PI05_RUN_NAME:-assemble-screwdriver-pi05-8xh100-b64-15k-20260829-v1}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export UV_PYTHON=${DATA_ROOT}/envs/uv-python/cpython-3.11-linux-x86_64-gnu/bin/python
export UV_PYTHON_DOWNLOADS=never
export UV_NO_MANAGED_PYTHON=1
unset WANDB_API_KEY WANDB_BASE_URL WANDB_ENTITY WANDB_PROJECT WANDB_MODE

YAM_TRAIN_ROOT=${DATA_ROOT} \
PI05_WORKSPACE=${CODE_ROOT} \
OPENPI_GPU_IDS=0,1,2,3,4,5,6,7 \
OPENPI_FSDP_DEVICES=4 \
OPENPI_BATCH_SIZE=64 \
OPENPI_NUM_WORKERS=8 \
OPENPI_NUM_TRAIN_STEPS=15000 \
OPENPI_SAVE_INTERVAL=1000 \
OPENPI_MAX_TO_KEEP=16 \
OPENPI_RESUME=0 \
OPENPI_LOCAL_CACHE_ROOT=/tmp/openpi-cache-${RUN_NAME} \
bash "${CODE_ROOT}/scripts/train_pi05_yam_cluster.sh" train "${RUN_NAME}"

echo "PI05_8GPU_15K_OK ${DATA_ROOT}/weights/finetuned/pi05/${RUN_NAME}"
