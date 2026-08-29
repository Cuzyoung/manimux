#!/usr/bin/env bash
set -uo pipefail

DATA_ROOT=${YAM_TRAIN_ROOT:-/inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data}
CODE_ROOT=${MANIMUX_CODE_ROOT:-/inspire/hdd2/project/liu-ming-huan/public/ziyang/manimux}
START_SCRIPT=${CODE_ROOT}/experiments/assemble_screwdriver/monitor/start_live_tensorboard.sh
stopping=0

stop_guard() {
  stopping=1
  if [[ -n "${dashboard_pid:-}" ]]; then
    kill "${dashboard_pid}" 2>/dev/null || true
  fi
}
trap stop_guard INT TERM

while [[ "${stopping}" -eq 0 ]]; do
  bash "${START_SCRIPT}" &
  dashboard_pid=$!
  wait "${dashboard_pid}"
  status=$?
  dashboard_pid=
  if [[ "${stopping}" -eq 0 ]]; then
    echo "dashboard process exited rc=${status}; restarting in 3 seconds" >&2
    sleep 3
  fi
done
