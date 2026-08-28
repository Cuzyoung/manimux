#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${YAM_TRAIN_ROOT:-/inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data}
CODE_ROOT=${MANIMUX_CODE_ROOT:-/inspire/hdd2/project/liu-ming-huan/public/ziyang/manimux}
PORT=${TENSORBOARD_PORT:-16006}

PI05_RUN=${PI05_RUN_NAME:-assemble-screwdriver-pi05-4xh100-3k-20260829-v2}
LINGBOT_RUN=${LINGBOT_RUN_NAME:-assemble-screwdriver-lingbot-4xh100-3k-20260829-v1}
XR1_RUN=${XR1_RUN_NAME:-assemble-screwdriver-xr1-4xh100-3k-20260829-v1}
GR00T_RUN=${GR00T_RUN_NAME:-assemble-screwdriver-gr00t-n17-4xh100-3k-20260829-v1}

PI05_LOG=${DATA_ROOT}/runs/pi05/${PI05_RUN}-train.log
PI05_EVENTS=${DATA_ROOT}/runs/live-tensorboard/pi05/${PI05_RUN}
LINGBOT_EVENTS=${DATA_ROOT}/weights/finetuned/lingbot-vla2/${LINGBOT_RUN}/runs
XR1_EVENTS=${DATA_ROOT}/weights/finetuned/xiaomi-xr1/${XR1_RUN}/project_yam-xiaomi-xr1/RoboDojo_real-assemble_the_screwdriver-yam_dual-ee-0/project_yam-xiaomi-xr1/RoboDojo_real-assemble_the_screwdriver-yam_dual-ee-0
GR00T_LOG=${DATA_ROOT}/runs/gr00t-n17/${GR00T_RUN}.log
GR00T_EVENTS=${DATA_ROOT}/runs/live-tensorboard/gr00t-n17/${GR00T_RUN}
PYTHON=${DATA_ROOT}/envs/xr1/.venv/bin/python

mkdir -p "${PI05_EVENTS}"
"${PYTHON}" "${CODE_ROOT}/scripts/tail_pi05_metrics.py" \
  --log "${PI05_LOG}" \
  --output "${PI05_EVENTS}" &
pi05_sidecar_pid=$!
"${PYTHON}" "${CODE_ROOT}/scripts/tail_gr00t_metrics.py" \
  --log "${GR00T_LOG}" \
  --output "${GR00T_EVENTS}" &
gr00t_sidecar_pid=$!
trap 'kill "${pi05_sidecar_pid}" "${gr00t_sidecar_pid}" 2>/dev/null || true' EXIT

exec "${PYTHON}" "${CODE_ROOT}/scripts/launch_tensorboard_localhost.py" \
  --logdir-spec "Pi05:${PI05_EVENTS},LingBot:${LINGBOT_EVENTS},XR1:${XR1_EVENTS},GR00T:${GR00T_EVENTS}" \
  --port "${PORT}"
