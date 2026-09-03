#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REMOTE_HOST=${YAM_DASHBOARD_REMOTE:-localhost-3338}
REMOTE_CODE_ROOT=${YAM_REMOTE_CODE_ROOT:-/inspire/hdd2/project/liu-ming-huan/public/ziyang/manimux}
LOCAL_PORT=${YAM_DASHBOARD_LOCAL_PORT:-16006}
REMOTE_PORT=${YAM_DASHBOARD_REMOTE_PORT:-16006}
REMOTE_SERVICE=${REMOTE_CODE_ROOT}/experiments/assemble_screwdriver/monitor/dashboard_service.sh
UNIT_SOURCE=${ROOT}/ops/systemd/yam-training-dashboard.service
UNIT_TARGET=${HOME}/.config/systemd/user/yam-training-dashboard.service

dashboard_reachable() {
  curl -fsS --max-time 2 "http://127.0.0.1:${LOCAL_PORT}/healthz" >/dev/null 2>&1
}

install_unit() {
  install -D -m 0644 "${UNIT_SOURCE}" "${UNIT_TARGET}"
  systemctl --user daemon-reload
  systemctl --user enable yam-training-dashboard.service >/dev/null
}

case "${1:-start}" in
  foreground)
    ssh "${REMOTE_HOST}" bash "${REMOTE_SERVICE}" start
    forward_args=()
    for offset in 0 1 2 3 4 5; do
      forward_args+=(
        -L "$((LOCAL_PORT + offset)):127.0.0.1:$((REMOTE_PORT + offset))"
      )
    done
    exec ssh -NT \
      -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=10 \
      -o ServerAliveCountMax=3 \
      "${forward_args[@]}" \
      "${REMOTE_HOST}"
    ;;
  start)
    install_unit
    systemctl --user restart yam-training-dashboard.service
    for _ in $(seq 1 20); do
      if dashboard_reachable; then
        echo "Training dashboard: http://127.0.0.1:${LOCAL_PORT}"
        exit 0
      fi
      sleep 0.5
    done
    echo "dashboard service started but health check is not ready" >&2
    exit 1
    ;;
  stop)
    systemctl --user disable --now yam-training-dashboard.service >/dev/null 2>&1 || true
    ssh "${REMOTE_HOST}" bash "${REMOTE_SERVICE}" stop
    ;;
  restart)
    install_unit
    ssh "${REMOTE_HOST}" bash "${REMOTE_SERVICE}" restart
    systemctl --user restart yam-training-dashboard.service
    echo "Training dashboard: http://127.0.0.1:${LOCAL_PORT}"
    ;;
  status)
    systemctl --user status yam-training-dashboard.service --no-pager
    if dashboard_reachable; then
      echo "dashboard healthy: http://127.0.0.1:${LOCAL_PORT}"
    else
      echo "dashboard health check failed"
      exit 1
    fi
    ;;
  *)
    echo "Usage: $0 [start|stop|restart|status|foreground]" >&2
    exit 2
    ;;
esac
