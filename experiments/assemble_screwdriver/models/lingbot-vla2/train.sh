#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=/inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data
CODE_ROOT=/inspire/hdd2/project/liu-ming-huan/public/ziyang/manimux
POLICY_ROOT=${CODE_ROOT}/XPolicyLab/policy/LingBot_VLA2
SOURCE_ROOT=${POLICY_ROOT}/lingbot_vla_v2
CONFIG=${CODE_ROOT}/experiments/assemble_screwdriver/models/lingbot-vla2/config.yaml
VENV=${DATA_ROOT}/envs/lingbot-vla2/.venv
DATASET=${DATA_ROOT}/datasets/lerobot/yam_assemble_screwdriver_20260825_v1
ROBOT_NAME=yam_dual_packed_relative
STATS_ROOT=${DATA_ROOT}/cache/lingbot-vla2/yam_assemble_screwdriver_20260825_v1/${ROBOT_NAME}
STATS=${STATS_ROOT}/norm_stats.json
MODEL=${DATA_ROOT}/weights/base/lingbot-vla-v2-6b
TOKENIZER=${DATA_ROOT}/weights/base/xiaomi/qwen3_vl_4b_processor
RUN_NAME=${LINGBOT_RUN_NAME:-assemble-screwdriver-lingbot-native-depth-relative-8xh100-b64-15k-20260829-v2}
OUTPUT=${DATA_ROOT}/weights/finetuned/lingbot-vla2-native-depth-relative/${RUN_NAME}
LOG=${DATA_ROOT}/runs/lingbot-vla2-native-depth-relative/${RUN_NAME}.log

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=${DATA_ROOT}/cache/huggingface
export HF_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export TORCH_HOME=${DATA_ROOT}/cache/torch
export PYTHONPATH=${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}
unset WANDB_API_KEY WANDB_BASE_URL WANDB_ENTITY WANDB_PROJECT WANDB_MODE

if [[ ! -s "${STATS}" ]]; then
  mkdir -p "${STATS_ROOT}"
  cd "${SOURCE_ROOT}"
  CUDA_VISIBLE_DEVICES=0 PATH="${VENV}/bin:${PATH}" bash -o pipefail train.sh \
    scripts/compute_norm_stats.py configs/vla/norm_compute/post_data.yaml \
    --data.data_name "${ROBOT_NAME}" \
    --data.robot_name "${ROBOT_NAME}" \
    --data.train_path "${DATASET}" \
    --data.robot_config_root "${POLICY_ROOT}/robot_configs" \
    --data.norm_path "${STATS}" \
    --data.num_workers 8 \
    --train.chunk_size 50 \
    --train.micro_batch_size 32 \
    --train.output_dir "${STATS_ROOT}/compute"
fi

mkdir -p "${OUTPUT}" "$(dirname "${LOG}")"
cd "${OUTPUT}"

PATH="${VENV}/bin:${PATH}" bash -o pipefail "${SOURCE_ROOT}/train.sh" \
  "${SOURCE_ROOT}/tasks/vla/train_lingbotvla.py" "${CONFIG}" \
  --model.model_path "${MODEL}" \
  --model.tokenizer_path "${TOKENIZER}" \
  --data.data_name "${ROBOT_NAME}" \
  --data.train_path "${DATASET}" \
  --data.robot_config_root "${POLICY_ROOT}/robot_configs" \
  --data.norm_stats_file "${STATS}" \
  --data.num_workers 8 \
  --train.output_dir "${OUTPUT}" \
  --train.seed 0 \
  --train.micro_batch_size 1 \
  --train.gradient_accumulation_steps 8 \
  --train.data_parallel_mode fsdp2 \
  --train.enable_full_shard false \
  --train.module_fsdp_enable true \
  --train.vlm_fsdp true \
  --train.freeze_vision_encoder false \
  --train.train_expert_only false \
  --train.max_steps 15000 \
  --train.save_steps 1000 \
  --train.enable_resume false \
  --train.use_wandb false \
  2>&1 | tee "${LOG}"

cp -f "${STATS}" "${OUTPUT}/norm_stats.json"
cp -f "${POLICY_ROOT}/robot_configs/${ROBOT_NAME}.yaml" \
  "${OUTPUT}/robot_config.yaml"

echo "LINGBOT_NATIVE_DEPTH_RELATIVE_8GPU_15K_OK ${OUTPUT}"
