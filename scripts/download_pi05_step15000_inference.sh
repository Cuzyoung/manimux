#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST=root@localhost
REMOTE_PORT=3338
REMOTE_CHECKPOINT=/inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data/weights/finetuned/pi05/assemble-screwdriver-pi05-8xh100-b64-15k-20260829-v2/15000
DESTINATION=/home/ubuntu/manimux/checkpoints/finetuned/ziyang/pi05-yam-assemble-screwdriver-b64-step15000
PARTIAL=${DESTINATION}.partial

mkdir -p "${PARTIAL}"

ssh \
  -o BatchMode=yes \
  -o Compression=no \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=20 \
  -p "${REMOTE_PORT}" \
  "${REMOTE_HOST}" \
  "tar -C '${REMOTE_CHECKPOINT}' -cf - params assets _CHECKPOINT_METADATA" \
  | tar -C "${PARTIAL}" -xf -

test -s "${PARTIAL}/params/_METADATA"
test -s "${PARTIAL}/params/manifest.ocdbt"
test -s "${PARTIAL}/assets/yam_assemble_screwdriver_20260825_v1/norm_stats.json"
mv "${PARTIAL}" "${DESTINATION}"
du -sh "${DESTINATION}"
