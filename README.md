<div align="center">

# ManiMux

**Local asynchronous inference and execution infrastructure for real-robot VLAs.**

Models are replaceable; ManiMux owns the control loop, robot, safety, recording and viewer.

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Lint: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://docs.astral.sh/ruff/)
[![Typing: mypy strict](https://img.shields.io/badge/typing-mypy%20strict-2a6db2.svg)](https://mypy-lang.org/)

[English](README.md) · [简体中文](README.zh-CN.md)

</div>

![ManiMux viewer during a MolmoAct rollout on dual YAM arms](assets/viewer-hero.png)

## Purpose

VLA inference is commonly slower than a robot control period and returns actions in chunks.
ManiMux moves inference out of the control loop and owns chunk scheduling, stale-prefix trimming,
atomic dual-arm commits, execution constraints, safety checks, recording and live visualization.

ManiMux is not tied to one model library. MolmoAct, ABC and XR-1 have native service adapters;
additional foundation models enter through `XPolicyLab/`. ManiMux keeps the stable runtime and
wire bridge. Model preprocessing, flow denoising and model-native RTC sampling remain in the
corresponding XPolicyLab model adapter.

## Architecture

```text
CameraServer -> SensorDriver -> ObservationSnapshot
                                  |
                                  v
                    PolicyAdapter.build_observation()
                                  |
                                  v
                    bounded latest-wins queue
                                  |
                                  v
                     PolicyModel worker/client
                       |                  |
                       |                  +-> xpolicylab_ws
                       |                         |
                       v                         v
              native model server        XPolicyLab server
              MolmoAct / ABC / XR-1              |
                                                 v
                                     policy/<MODEL>/model.py
                                       model-native sampler
                       |                         |
                       +-----------+-------------+
                                   v
                              raw action
                                   |
                                   v
                    PolicyAdapter.decode_action()
                     validate / joint map / FK-IK
                                   |
                                   v
              ActionTimeline -> Smooth|MPC -> Safety -> RobotDriver
              trim/blend/atomic commit

EdgeRuntime side channels:
  Recorder <- state / plan / command lineage
  Viewer   <- measured state / committed plan
```

The default path is the ManiMux asynchronous runtime. RTC is an explicit alternative inference
schedule: ManiMux derives a condition and soft mask from real execution progress, then an
RTC-capable model adapter injects them into its native sampler.

## Repository Layout

| Path | Responsibility |
|---|---|
| `src/manimux/runtime/` | Control loop, Timeline, Smooth/MPC, RTC and Safety |
| `src/manimux/policies/` | `PolicyModel` / `PolicyAdapter` contracts and isolated worker |
| `src/manimux/integrations/` | Native integrations and the shared XPolicy WebSocket bridge |
| `src/manimux/robots/` | RobotDriver implementations; dual YAM is the current hardware target |
| `src/manimux/sensors/` | RealSense and the shared camera service |
| `src/manimux/recording/` | Episodes, Zarr, events and action lineage |
| `src/manimux/viewer/` | Generic viewer protocol and robot geometry adapters |
| `XPolicyLab/` | Submodule pointing to our XPolicyLab fork; model-internal changes live here |
| `checkpoints/pretrained/` | Unmodified foundation-model or upstream release weights |
| `checkpoints/finetuned/<publisher>/` | Fine-tuned checkpoints, named after their published repository |
| `configs/<model>/<embodiment>/` | One explicit configuration per model and embodiment experiment |
| `docs/*-runbook.md` | Per-model installation, startup, checks and shutdown |

For example, the Robocurve YAM releases live at
`checkpoints/finetuned/robocurve/{pi05-yam-molmoact2,gr00t-n1.7-yam-molmoact2}`;
the official OpenPI `pi05_base` remains under `checkpoints/pretrained/`.

## Integrations

Status: ✅ running · 🧪 experimental · 🚧 not deployable yet · 🔌 infrastructure

| Status | Model/path | ManiMux canonical action | Config | Runbook |
|---|---|---|---|---|
| ✅ | MolmoAct2 + YAM | 30 × 14 joint positions | `configs/molmoact2/yam/` | [MolmoAct2](docs/molmoact-yam-runbook.md) |
| ✅ | ABC + YAM | 30 × 14 joint positions | `configs/abc/yam/` | [ABC](docs/abc-yam-runbook.md) |
| 🧪 | Native XR-1 + YAM | 30 × 60 EE deltas → 30 × 14 joint positions | `configs/xiaomi-xr1/yam/infra/native.yaml` | [Native XR-1](docs/xr1-yam-runbook.md) |
| ✅ | OpenPI Pi05 + YAM | 16 × 14 absolute joint positions | `configs/pi05/yam/` | [Pi05](docs/pi05-yam-runbook.md) |
| 🧪 | GR00T N1.7 + YAM | 16 × 14 absolute joint positions | `configs/groot/yam/` | [GR00T](docs/gr00t-yam-runbook.md) |
| 🚧 | XPolicy XR-1 + YAM | 30 × 60 EE deltas → 30 × 14 joint positions | `configs/xiaomi-xr1/yam/{server,infra}/` | [XPolicy XR-1](docs/xiaomi-xr1-yam-runbook.md) |
| 🚧 | LingBot-VLA2 + YAM | 50 × 14 absolute joints; YAM post-training bundle required | `configs/lingbot-vla2/yam/` | [LingBot-VLA2](docs/lingbot-vla2-yam-runbook.md) |
| 🔌 | XPolicy bridge | Standard observation/action wire contract | `configs/xpolicylab/yam/infra/smoke.yaml` | [XPolicyLab](docs/xpolicylab-runbook.md) |

OpenPI Pi05 has completed the real three-camera, YAM normalization, XPolicy model-server,
official 10-step flow sampling, default ManiMux and Pi-guided RTC paths on dual YAM. RTC ran with
measured `d=3-5` steps and no post-start chunk gap. The checkpoint produced task-related motion,
but remained hesitant in this scene and has no established success rate; that policy-quality
result is separate from the completed inference infrastructure. GR00T, XPolicy XR-1 and LingBot-VLA2
must not be described as hardware-validated yet.

## Install

```bash
git clone --recursive https://github.com/SII-LiuLab/manimux.git
cd manimux
uv sync --dev
```

For an existing checkout:

```bash
git submodule update --init --recursive
```

The core runtime uses `./.venv`, real YAM runs use `envs/yam/.venv`, and model servers keep
isolated environments. CUDA, Torch and checkpoint setup belongs in
[the environment guide](envs/README.md) and each model runbook.

## Quick Start

### No Hardware

```bash
uv run manimux run --config configs/mock.yaml
uv run manimux-viewer --robot yam --demo --port 8086
```

### Shared YAM Services

The camera service and viewer are model-independent and start once per experiment stack:

```bash
envs/yam/.venv/bin/manimux-camera-server --config configs/cameras.yaml
envs/yam/.venv/bin/manimux-viewer --robot yam --host 0.0.0.0 --port 8086
```

### Native Example: MolmoAct2

```bash
envs/yam/.venv/bin/manimux-molmoact-server --host 127.0.0.1 --port 8202
envs/yam/.venv/bin/manimux run --config configs/molmoact2/yam/infra/manimux.yaml
```

### XPolicy Example: Pi05-YAM

```bash
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  scripts/pi05_yam_server.py \
  --config configs/pi05/yam/server/finetune.yaml

envs/yam/.venv/bin/manimux run --config configs/pi05/yam/infra/manimux.yaml

# Pi-guided RTC uses the same finetuned model server.
envs/yam/.venv/bin/manimux run --config configs/pi05/yam/infra/rtc.yaml
```

These snippets show entry points only. They do not replace checkpoint validation, preflight,
CAN checks or shutdown procedures. Open the matching runbook before any hardware run.

## Configuration Convention

```text
configs/
  <model>/
    <embodiment>/
      server/              # model-service configs
        <checkpoint>.yaml
      infra/               # ManiMux robot, sensors, execution and recording
        <experiment>.yaml
  cameras.yaml
  robots/
  mock.yaml
```

One experiment uses one explicit config. Different checkpoints, runtimes and embodiments do not
share hidden switches. See [configs/README.md](configs/README.md) for naming rules.

## Adding a Model or Embodiment

| Boundary | Responsibility |
|---|---|
| `PolicyModel` | Call a local model or model service; never touch the robot |
| `PolicyAdapter` | Camera/state encoding, action decoding, normalization, joint mapping and FK/IK |
| `RobotDriver` | Vendor SDK, state, commands, home, stop and close |
| `SensorDriver` | Read named, timestamped sensor streams |
| `RobotAdapter` | Viewer URDF, FK, joint groups and appearance |

Changing a model only changes the two Policy layers. Changing a robot only changes the embodiment
adapter, driver and config. Runtime, Timeline, Safety, Recorder and the viewer protocol are not
copied per model. See [Architecture](docs/architecture.md) for the complete contracts.

## Documentation

- [Program progress](docs/program-progress.md)
- [XPolicyLab integration](docs/xpolicylab-runbook.md)
- [MolmoAct2](docs/molmoact-yam-runbook.md) · [ABC](docs/abc-yam-runbook.md) · [Native XR-1](docs/xr1-yam-runbook.md)
- [Pi05](docs/pi05-yam-runbook.md) · [GR00T](docs/gr00t-yam-runbook.md) · [XPolicy XR-1](docs/xiaomi-xr1-yam-runbook.md)
- [LingBot-VLA2](docs/lingbot-vla2-yam-runbook.md) · [CAN bus](docs/can-bus.md)
- [Architecture](docs/architecture.md) · [Development conventions](AGENT.md)

## Safety

ManiMux drives physical robots. Passing tests does not prove hardware safety, and software cannot
replace a physical emergency stop, hardware limits or the vendor watchdog. The runtime starts
paused, faults do not auto-recover, and every hardware entry point requires its runbook preflight.

## License

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [`licenses/`](licenses) for upstream
attribution.
