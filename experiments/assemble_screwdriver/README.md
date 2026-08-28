# Assemble screwdriver training

Task-specific launch and submission scripts live here. Model training entrypoints under `scripts/` and `XPolicyLab/` remain unchanged.

- `train/lingbot_1gpu_smoke.sh`: one H100 action-expert smoke, one AdamW optimizer step, checkpoint save, strict HF reload. The four-GPU run trains the full model.
- `submit/lingbot_1gpu_smoke.py`: submits that smoke job to `intern-ziyang`.
