# Assemble screwdriver training

Task-specific launch and submission scripts live here. Model training entrypoints under `scripts/` and `XPolicyLab/` remain unchanged.

- `train/pi05_1gpu_smoke.sh`: one H100, one optimizer step, checkpoint save, and strict OpenPI reload.
- `submit/pi05_1gpu_smoke.py`: submits that Pi05 smoke job to `intern-ziyang`.
- `train/pi05_4gpu_train.sh`: four-H100 Pi05 training with global batch 384 for 3000 steps, saving every 500 steps.
- `submit/pi05_4gpu_train.py`: submits the full Pi05 training job to `intern-ziyang`.
- `train/lingbot_1gpu_smoke.sh`: one H100 action-expert smoke, one AdamW optimizer step, checkpoint save, strict HF reload. The four-GPU run trains the full model.
- `submit/lingbot_1gpu_smoke.py`: submits that smoke job to `intern-ziyang`.
- `train/lingbot_4gpu_train.sh`: four-H100 full-model AdamW/FSDP2 training for 3000 steps, saving every 500 steps.
- `submit/lingbot_4gpu_train.py`: submits the full LingBot training job to `intern-ziyang`.
- `train/xr1_1gpu_smoke.sh`: one-H100 full-model XR-1 smoke with native DeepSpeed ZeRO-2/CPUAdam optimizer offload, one optimizer step, checkpoint save, and native checkpoint reload.
- `submit/xr1_1gpu_smoke.py`: submits that XR-1 smoke job to `intern-ziyang`.
- `train/xr1_4gpu_train.sh`: four-H100 full-model XR-1 DeepSpeed ZeRO-2 training for 3000 steps, saving every 500 steps.
- `submit/xr1_4gpu_train.py`: submits the full XR-1 training job to `intern-ziyang`.
