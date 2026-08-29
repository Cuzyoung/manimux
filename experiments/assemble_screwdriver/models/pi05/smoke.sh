#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=/inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data
CODE_ROOT=/inspire/hdd2/project/liu-ming-huan/public/ziyang/manimux
OPENPI=${CODE_ROOT}/XPolicyLab/policy/Pi_05/openpi
VENV=${DATA_ROOT}/envs/pi05/.venv
RUN_NAME=${PI05_RUN_NAME:-assemble-screwdriver-pi05-1xh100-20260829-v4}
OUTPUT=${DATA_ROOT}/weights/finetuned/pi05/${RUN_NAME}-smoke

export CUDA_VISIBLE_DEVICES=0
export UV_PYTHON=${DATA_ROOT}/envs/uv-python/cpython-3.11-linux-x86_64-gnu/bin/python
export UV_PYTHON_DOWNLOADS=never
export UV_NO_MANAGED_PYTHON=1
export PYTHONPATH="${OPENPI}/src${PYTHONPATH:+:${PYTHONPATH}}"
unset WANDB_API_KEY WANDB_BASE_URL WANDB_ENTITY WANDB_PROJECT WANDB_MODE

YAM_TRAIN_ROOT=${DATA_ROOT} \
PI05_WORKSPACE=${CODE_ROOT} \
OPENPI_GPU_IDS=0 \
OPENPI_FSDP_DEVICES=1 \
OPENPI_BATCH_SIZE=1 \
OPENPI_NUM_WORKERS=0 \
OPENPI_NUM_TRAIN_STEPS=1 \
OPENPI_SAVE_INTERVAL=1 \
OPENPI_MAX_TO_KEEP=1 \
bash "${CODE_ROOT}/scripts/train_pi05_yam_cluster.sh" smoke "${RUN_NAME}"

export HF_HOME=${DATA_ROOT}/cache/huggingface
export HF_LEROBOT_HOME=${DATA_ROOT}/datasets/lerobot
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENPI_DATA_HOME=${DATA_ROOT}/cache/openpi
export OPENPI_LEROBOT_REPO_ID=yam_assemble_screwdriver_20260825_v1
export OPENPI_BASE_PARAMS=${DATA_ROOT}/weights/base/pi05_base/params
export OPENPI_ASSETS_BASE_DIR=${DATA_ROOT}/cache/pi05/assets

cd "${OPENPI}"
"${VENV}/bin/python" - "${OUTPUT}/1" <<'PY'
from pathlib import Path
import sys

import jax
from openpi.policies import policy_config
from openpi.training import config as training_config

assert jax.device_count() == 1
assert jax.devices()[0].platform == "gpu"
config = training_config.get_config("pi05_yam")
policy = policy_config.create_trained_policy(config, Path(sys.argv[1]))
assert policy.metadata == (config.policy_metadata or {})
print(f"PI05_SMOKE_RELOAD_OK {sys.argv[1]}")
PY

echo "PI05_1GPU_SMOKE_OK ${OUTPUT}"
