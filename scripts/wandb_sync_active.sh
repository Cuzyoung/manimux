#!/usr/bin/env bash
set -uo pipefail

ROOT=${YAM_TRAIN_ROOT:-/inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data}
WANDB_ROOT=${WANDB_DIR:-${ROOT}/runs/wandb/wandb}
WANDB_CLI=${WANDB_SYNC_CLI:-${ROOT}/envs/lingbot-vla2/.venv/bin/wandb}
INTERVAL=${WANDB_SYNC_INTERVAL:-60}
ACTIVE_MINUTES=${WANDB_SYNC_ACTIVE_MINUTES:-10}
LOG=${WANDB_SYNC_LOG:-${ROOT}/runs/wandb-sync.log}
SECRET=${WANDB_API_KEY_FILE:-${ROOT}/secrets/wandb_api_key}

if [[ ! -x "${WANDB_CLI}" ]]; then
  echo "W&B CLI is missing: ${WANDB_CLI}" >&2
  exit 1
fi
if [[ ! -s "${SECRET}" ]]; then
  echo "W&B API key file is missing: ${SECRET}" >&2
  exit 1
fi

IFS= read -r WANDB_API_KEY < "${SECRET}"
export WANDB_API_KEY
export WANDB_ENTITY=${WANDB_ENTITY:-ace_experiments}
export WANDB_MODE=online
mkdir -p "$(dirname "${LOG}")"

echo "[$(date -Is)] active W&B syncer started: root=${WANDB_ROOT} interval=${INTERVAL}s" >> "${LOG}"
while true; do
  mapfile -t run_dirs < <(
    find "${WANDB_ROOT}" -maxdepth 2 -type f -name 'run-*.wandb' \
      -mmin "-${ACTIVE_MINUTES}" -printf '%h\n' 2>/dev/null | sort -u
  )
  for run_dir in "${run_dirs[@]}"; do
    case "$(basename "${run_dir}")" in
      offline-run-*)
        echo "[$(date -Is)] syncing ${run_dir}" >> "${LOG}"
        "${WANDB_CLI}" sync --include-offline --include-synced --append \
          --entity "${WANDB_ENTITY}" "${run_dir}" >> "${LOG}" 2>&1 || true
        ;;
    esac
  done
  sleep "${INTERVAL}"
done
