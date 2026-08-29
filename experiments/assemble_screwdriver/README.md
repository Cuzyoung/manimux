# Assemble screwdriver training

This directory is the only task-specific entrypoint for the screwdriver
experiments.

## Directory contract

- Cluster code: `/inspire/hdd2/project/liu-ming-huan/public/ziyang/manimux`
- Data and environments: `/inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data`
- Checkpoints: `yam_fintune_data/weights/finetuned/<model>/<run_name>`
- Logs: `yam_fintune_data/runs/<model>/<run_name>.log`

No training code belongs under `yam_fintune_data`.

## Current model entries

Each model directory has one formal launcher, one one-GPU smoke launcher, and
their matching QZ submission files:

```text
models/
├── pi05/{train.sh,smoke.sh,submit.py,submit_smoke.py}
├── lingbot-vla2/{config.yaml,train.sh,smoke.sh,submit.py,submit_smoke.py}
├── xiaomi-xr1/{train.sh,smoke.sh,submit.py,submit_smoke.py}
└── gr00t-n17/{train.sh,smoke.sh,submit.py,submit_smoke.py}
```

The three current formal settings are:

| Model | GPUs | Global batch | Steps | Save interval | Action target |
| --- | ---: | ---: | ---: | ---: | --- |
| Pi0.5 | 8 H100 | 64 | 15000 | 1000 | existing YAM relative-joint transform |
| LingBot-VLA2 | 8 H100 | 64 | 15000 | 1000 | anchor-relative arm qpos, absolute grippers |
| Xiaomi XR-1 | 8 H100 | 64 | 15000 | 1000 | native anchor-relative end-effector action |

LingBot uses native depth supervision and full-model FSDP2 post-training.
Pi0.5 uses the OpenPI JAX trainer. XR-1 uses its native torchrun and
DeepSpeed training entrypoint. W&B and job time limits are disabled.

Run `smoke.sh` before `train.sh`. Each smoke performs one optimizer step,
saves a checkpoint, and reloads it with the model's native loader.

Previous smoke, 3k, 4-GPU, and absolute-action variants are retained under
`legacy/20260829/`. They are not current submission entrypoints.

Monitoring is documented in `monitor/README.md`.
