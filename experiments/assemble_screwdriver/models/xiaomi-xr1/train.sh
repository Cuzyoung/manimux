#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=/inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data
CODE_ROOT=/inspire/hdd2/project/liu-ming-huan/public/ziyang/manimux
POLICY_ROOT=${CODE_ROOT}/XPolicyLab/policy/Xiaomi_Robotics_1
XR1_ROOT=${POLICY_ROOT}/xiaomi_robotics_1/xr1
VENV=${DATA_ROOT}/envs/xr1/.venv
DATASET=${DATA_ROOT}/datasets/xr1/RoboDojo_real-assemble_the_screwdriver-yam_dual-ee
# The existing converted screwdriver dataset records this legacy filename in
# manifest.json; the file contents and source episodes are screwdriver data.
DATA_CONFIG_SOURCE=${DATASET}/yam_pick_red_ball_box.yaml
MODEL=${DATA_ROOT}/weights/base/xiaomi/model_states.pt
PROCESSOR=${DATA_ROOT}/weights/base/xiaomi/qwen3_vl_4b_processor
RUN_NAME=${XR1_RUN_NAME:-assemble-screwdriver-xr1-8xh100-b64-15k-20260829-v2}
OUTPUT=${DATA_ROOT}/weights/finetuned/xiaomi-xr1/${RUN_NAME}
LOG=${DATA_ROOT}/runs/xiaomi-xr1/${RUN_NAME}.log
EXP_NAME=RoboDojo_real-assemble_the_screwdriver-yam_dual-ee-0
CONFIG_ROOT=${OUTPUT}/hydra_config

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RESOURCE_GPU=8
export MLP_WORKER_NUM=1
export MLP_WORKER_GPU=8
export MLP_ROLE_INDEX=0
export MLP_WORKER_0_HOST=${MASTER_ADDR:-127.0.0.1}
export MLP_WORKER_0_PORT=${MASTER_PORT:-23456}
export HF_HOME=${DATA_ROOT}/cache/huggingface
export HF_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=${DATA_ROOT}/cache/torch
export XR1_QWEN_VL_CONFIG_SOURCE=${PROCESSOR}
export XR1_LOGGER=tensorboard
export MAX_LENGTH=20000
export PYTHONPATH=${XR1_ROOT}:${CODE_ROOT}/XPolicyLab${PYTHONPATH:+:${PYTHONPATH}}
export HYDRA_FULL_ERROR=1
unset WANDB_API_KEY WANDB_BASE_URL WANDB_ENTITY WANDB_PROJECT WANDB_MODE

mkdir -p "${CONFIG_ROOT}/data" "${OUTPUT}/assets" "$(dirname "${LOG}")"
cp "${DATA_CONFIG_SOURCE}" "${CONFIG_ROOT}/data/yam_assemble_the_screwdriver.yaml"
cd "${OUTPUT}"

PATH="${VENV}/bin:${PATH}" "${VENV}/bin/torchrun" \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  --master_addr="${MLP_WORKER_0_HOST}" \
  --master_port="${MLP_WORKER_0_PORT}" \
  "${XR1_ROOT}/tools/train.py" \
  "hydra.searchpath=[file://${CONFIG_ROOT}]" \
  data=yam_assemble_the_screwdriver \
  trainer.project=yam-xiaomi-xr1 \
  trainer.exp_name="${EXP_NAME}" \
  trainer.default_root_dir="${OUTPUT}" \
  trainer.seed=0 \
  trainer.max_steps=15000 \
  trainer.save_interval=1000 \
  trainer.accumulate_grad_batches=8 \
  +trainer.log_every_n_steps=1 \
  +trainer.strategy.params.stage=2 \
  +trainer.strategy.params.offload_optimizer=false \
  model.params.pretrained="${MODEL}" \
  model.params.model.async_train=true \
  "hydra.run.dir=${OUTPUT}/hydra/rank_\${oc.env:RANK,0}" \
  2>&1 | tee "${LOG}"

echo "XR1_8GPU_15K_OK ${OUTPUT}"
