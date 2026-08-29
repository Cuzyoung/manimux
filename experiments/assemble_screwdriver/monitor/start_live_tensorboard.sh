#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${YAM_TRAIN_ROOT:-/inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data}
CODE_ROOT=${MANIMUX_CODE_ROOT:-/inspire/hdd2/project/liu-ming-huan/public/ziyang/manimux}
PORT=${TENSORBOARD_PORT:-16006}
PYTHON=${DATA_ROOT}/envs/xr1/.venv/bin/python
LIVE_ROOT=${DATA_ROOT}/runs/live-tensorboard
WEIGHT_ROOT=${DATA_ROOT}/weights/finetuned

mkdir -p "${LIVE_ROOT}/pi05" "${LIVE_ROOT}/gr00t-n17"
"${PYTHON}" "${CODE_ROOT}/scripts/discover_training_metrics.py" \
  --data-root "${DATA_ROOT}" \
  --code-root "${CODE_ROOT}" &
metric_supervisor_pid=$!

"${PYTHON}" "${CODE_ROOT}/scripts/launch_tensorboard_localhost.py" \
  --logdir-spec "Pi05:${LIVE_ROOT}/pi05,GR00T:${LIVE_ROOT}/gr00t-n17,LingBot-action:${WEIGHT_ROOT}/lingbot-vla2,LingBot-native-depth:${WEIGHT_ROOT}/lingbot-vla2-native-depth,XR1:${WEIGHT_ROOT}/xiaomi-xr1" \
  --port "${PORT}" \
  --reload-interval 5 &
tensorboard_pid=$!

cleanup() {
  kill "${tensorboard_pid}" "${metric_supervisor_pid}" 2>/dev/null || true
  wait "${tensorboard_pid}" "${metric_supervisor_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
wait "${tensorboard_pid}"
