# Assemble screwdriver training

Task-specific launch and submission scripts live here. Model training entrypoints under `scripts/` and `XPolicyLab/` remain unchanged.

- `train/lingbot_1gpu_smoke.sh`: one H100, one optimizer step, checkpoint save, strict HF reload.
- `submit/lingbot_1gpu_smoke.py`: submits that smoke job to `intern-ziyang`.
