#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST=${YAM_DASHBOARD_REMOTE:-localhost-3338}
REMOTE_CODE_ROOT=${YAM_REMOTE_CODE_ROOT:-/inspire/hdd2/project/liu-ming-huan/public/ziyang/manimux}
LOCAL_PORT=${YAM_DASHBOARD_LOCAL_PORT:-16006}
REMOTE_PORT=${YAM_DASHBOARD_REMOTE_PORT:-16006}
CONTROL_SOCKET=${YAM_DASHBOARD_CONTROL_SOCKET:-/tmp/manimux-yam-dashboard-${UID}.sock}
REMOTE_SERVICE=${REMOTE_CODE_ROOT}/experiments/assemble_screwdriver/monitor/dashboard_service.sh

tunnel_running() {
  ssh -S "${CONTROL_SOCKET}" -O check "${REMOTE_HOST}" >/dev/null 2>&1
}

dashboard_reachable() {
  curl -fsS --max-time 2 "http://127.0.0.1:${LOCAL_PORT}/data/runs" >/dev/null 2>&1
}

start_tunnel() {
  if tunnel_running || dashboard_reachable; then
    return
  fi
  rm -f "${CONTROL_SOCKET}"
  ssh -M -S "${CONTROL_SOCKET}" -fNT \
    -o ExitOnForwardFailure=yes \
    -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
    "${REMOTE_HOST}"
}

case "${1:-start}" in
  start)
    ssh "${REMOTE_HOST}" bash "${REMOTE_SERVICE}" start
    start_tunnel
    echo "Training dashboard: http://127.0.0.1:${LOCAL_PORT}"
    ;;
  stop)
    if tunnel_running; then
      ssh -S "${CONTROL_SOCKET}" -O exit "${REMOTE_HOST}" >/dev/null
    fi
    rm -f "${CONTROL_SOCKET}"
    ssh "${REMOTE_HOST}" bash "${REMOTE_SERVICE}" stop
    ;;
  restart)
    if tunnel_running; then
      ssh -S "${CONTROL_SOCKET}" -O exit "${REMOTE_HOST}" >/dev/null
    fi
    rm -f "${CONTROL_SOCKET}"
    ssh "${REMOTE_HOST}" bash "${REMOTE_SERVICE}" restart
    start_tunnel
    echo "Training dashboard: http://127.0.0.1:${LOCAL_PORT}"
    ;;
  status)
    ssh "${REMOTE_HOST}" bash "${REMOTE_SERVICE}" status
    if tunnel_running || dashboard_reachable; then
      echo "local tunnel running: http://127.0.0.1:${LOCAL_PORT}"
    else
      echo "local tunnel stopped"
      exit 1
    fi
    ;;
  *)
    echo "Usage: $0 [start|stop|restart|status]" >&2
    exit 2
    ;;
esac
