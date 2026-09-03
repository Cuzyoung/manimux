#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${YAM_TRAIN_ROOT:-/inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data}
CODE_ROOT=${MANIMUX_CODE_ROOT:-/inspire/hdd2/project/liu-ming-huan/public/ziyang/manimux}
PORT=${TENSORBOARD_PORT:-16006}
PYTHON=${DATA_ROOT}/envs/xr1/.venv/bin/python
LIVE_ROOT=${DATA_ROOT}/runs/live-tensorboard
NATIVE_ROOT=${LIVE_ROOT}/native

mkdir -p \
  "${LIVE_ROOT}/pi05" \
  "${LIVE_ROOT}/gr00t-n17" \
  "${NATIVE_ROOT}/lingbot-action" \
  "${NATIVE_ROOT}/lingbot-native-depth" \
  "${NATIVE_ROOT}/xr1"
"${PYTHON}" "${CODE_ROOT}/scripts/discover_training_metrics.py" \
  --data-root "${DATA_ROOT}" \
  --code-root "${CODE_ROOT}" &
metric_supervisor_pid=$!

"${PYTHON}" "${CODE_ROOT}/scripts/launch_training_dashboard_index.py" --port "${PORT}" &
index_pid=$!

event_roots=(
  "${LIVE_ROOT}/pi05"
  "${NATIVE_ROOT}/lingbot-action"
  "${NATIVE_ROOT}/lingbot-native-depth"
  "${NATIVE_ROOT}/xr1"
  "${LIVE_ROOT}/gr00t-n17"
)
tensorboard_pids=()
for index in "${!event_roots[@]}"; do
  model_port=$((PORT + index + 1))
  "${PYTHON}" "${CODE_ROOT}/scripts/launch_tensorboard_localhost.py" \
    --logdir "${event_roots[${index}]}" \
    --port "${model_port}" \
    --reload-interval 5 &
  tensorboard_pids+=("$!")
done

cleanup() {
  kill "${index_pid}" "${metric_supervisor_pid}" "${tensorboard_pids[@]}" 2>/dev/null || true
  wait "${index_pid}" "${metric_supervisor_pid}" "${tensorboard_pids[@]}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
wait -n "${index_pid}" "${metric_supervisor_pid}" "${tensorboard_pids[@]}"
