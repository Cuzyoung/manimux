#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=/inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data
CODE_ROOT=/inspire/hdd2/project/liu-ming-huan/public/ziyang/manimux
POLICY_ROOT=${CODE_ROOT}/XPolicyLab/policy/LingBot_VLA2
SOURCE_ROOT=${POLICY_ROOT}/lingbot_vla_v2
CONFIG=${CODE_ROOT}/experiments/assemble_screwdriver/train/lingbot_native_depth.yaml
VENV=${DATA_ROOT}/envs/lingbot-vla2/.venv
DATASET=${DATA_ROOT}/datasets/lerobot/yam_assemble_screwdriver_20260825_v1
STATS=${DATA_ROOT}/cache/lingbot-vla2/yam_assemble_screwdriver_20260825_v1/norm_stats.json
MODEL=${DATA_ROOT}/weights/base/lingbot-vla-v2-6b
TOKENIZER=${DATA_ROOT}/weights/base/xiaomi/qwen3_vl_4b_processor
RUN_NAME=${LINGBOT_RUN_NAME:-assemble-screwdriver-lingbot-native-depth-4xh100-3k-20260829-v1}
OUTPUT=${DATA_ROOT}/weights/finetuned/lingbot-vla2-native-depth/${RUN_NAME}
LOG=${DATA_ROOT}/runs/lingbot-vla2-native-depth/${RUN_NAME}.log

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=${DATA_ROOT}/cache/huggingface
export HF_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export TORCH_HOME=${DATA_ROOT}/cache/torch
export PYTHONPATH=${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}
unset WANDB_API_KEY WANDB_BASE_URL WANDB_ENTITY WANDB_PROJECT WANDB_MODE

mkdir -p "${OUTPUT}" "$(dirname "${LOG}")"
cd "${OUTPUT}"

PATH="${VENV}/bin:${PATH}" bash -o pipefail "${SOURCE_ROOT}/train.sh" \
  "${SOURCE_ROOT}/tasks/vla/train_lingbotvla.py" "${CONFIG}" \
  --model.model_path "${MODEL}" \
  --model.tokenizer_path "${TOKENIZER}" \
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
  --train.max_steps 3000 \
  --train.save_steps 500 \
  --train.use_wandb false \
  2>&1 | tee "${LOG}"

PATH="${VENV}/bin:${PATH}" "${VENV}/bin/python" "${POLICY_ROOT}/prepare_bundle.py" \
  --run-dir "${OUTPUT}" \
  --source-root "${SOURCE_ROOT}" \
  --norm-stats "${STATS}" \
  --robot-config "${POLICY_ROOT}/robot_configs/yam_dual_packed_absolute.yaml" \
  --native-hz 30 \
  --action-horizon 50

echo "LINGBOT_NATIVE_DEPTH_4GPU_TRAIN_OK ${OUTPUT}"
