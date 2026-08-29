#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${YAM_TRAIN_ROOT:-/inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data}
CODE_ROOT=${MANIMUX_CODE_ROOT:-/inspire/hdd2/project/liu-ming-huan/public/ziyang/manimux}
STATE_ROOT=${DATA_ROOT}/runs/live-tensorboard
PID_FILE=${STATE_ROOT}/dashboard.pid
LOG_FILE=${STATE_ROOT}/server.log
START_SCRIPT=${CODE_ROOT}/experiments/assemble_screwdriver/monitor/start_live_tensorboard.sh

running_pid() {
  if [[ -f "${PID_FILE}" ]]; then
    local pid
    pid=$(<"${PID_FILE}")
    if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
      printf '%s\n' "${pid}"
      return 0
    fi
  fi
  return 1
}

start_service() {
  mkdir -p "${STATE_ROOT}"
  local pid
  if pid=$(running_pid); then
    echo "training dashboard already running pid=${pid}"
    return
  fi
  nohup setsid bash "${START_SCRIPT}" >>"${LOG_FILE}" 2>&1 &
  pid=$!
  printf '%s\n' "${pid}" >"${PID_FILE}"
  sleep 1
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "training dashboard failed to start; inspect ${LOG_FILE}" >&2
    return 1
  fi
  echo "training dashboard started pid=${pid} port=${TENSORBOARD_PORT:-16006}"
}

stop_service() {
  local pid
  if ! pid=$(running_pid); then
    echo "training dashboard is not running"
    return
  fi
  kill -TERM "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  for _ in $(seq 1 20); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      rm -f "${PID_FILE}"
      echo "training dashboard stopped"
      return
    fi
    sleep 0.25
  done
  kill -KILL "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
  rm -f "${PID_FILE}"
  echo "training dashboard killed after timeout"
}

case "${1:-start}" in
  start)
    start_service
    ;;
  stop)
    stop_service
    ;;
  restart)
    stop_service
    start_service
    ;;
  status)
    if pid=$(running_pid); then
      echo "training dashboard running pid=${pid} port=${TENSORBOARD_PORT:-16006}"
    else
      echo "training dashboard stopped"
      exit 1
    fi
    ;;
  foreground)
    exec bash "${START_SCRIPT}"
    ;;
  *)
    echo "Usage: $0 [start|stop|restart|status|foreground]" >&2
    exit 2
    ;;
esac
