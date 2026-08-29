# Assemble screwdriver training

Task-specific launch and submission scripts live here. Model training entrypoints under `scripts/` and `XPolicyLab/` remain unchanged.

Final 8-H100 / global-batch-64 / 15k launchers:

- `train/pi05_8gpu_15k.sh`
- `train/lingbot_native_depth_relative_8gpu_15k.sh`
- `train/xr1_8gpu_15k.sh`

Their direct QZ submission scripts are under `submit/` with the same basename.
LingBot uses anchor-relative arm qpos and absolute grippers; Pi05 uses its
existing YAM relative-joint transform; XR-1 uses its native anchor-relative EE
action representation.

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
- `train/gr00t_1gpu_smoke.sh`: one-H100 GR00T N1.7 optimizer step, checkpoint save, and strict model reload using the XPolicyLab native launcher.
- `submit/gr00t_1gpu_smoke.py`: submits the GR00T smoke job to `intern-ziyang`.
- `train/gr00t_4gpu_train.sh`: four-H100 GR00T N1.7 training for 3000 steps with the proven global batch 32, saving at 1000/2000/3000.
- `submit/gr00t_4gpu_train.py`: submits the full GR00T training job to `intern-ziyang`.

## GR00T N1.7 setting

The screwdriver run uses XPolicyLab's native `policy/GR00T_N17/train.sh`, which
delegates to the vendored Isaac-GR00T `examples/finetune.sh` and
`gr00t/experiment/launch_finetune.py`. The upstream model recipe is preserved:
AdamW, learning rate `1e-4`, warmup ratio `0.05`, weight decay `1e-5`, state
dropout `0.2`, the native color jitter, frozen LLM/visual backbones, and
trainable projector/diffusion action model. YAM uses the checked-in
`NEW_EMBODIMENT` modality with three RGB views, 14D state/action, absolute joint
actions, and a 16-step horizon.

Only experiment-scale settings differ: the proven four-H100 global batch is
`32`, the requested run is `3000` steps, checkpoints are saved at
`1000/2000/3000`, and W&B is disabled in favor of stdout plus local
TensorBoard. The one-H100 smoke uses batch `1` for one optimizer step, then
strictly reloads the saved model before the four-H100 job is submitted.
