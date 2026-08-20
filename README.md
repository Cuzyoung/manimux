<div align="center">

# ManiMux

**Local asynchronous inference and execution infrastructure for real-robot VLAs.**

Models are replaceable; ManiMux owns the control loop, robot, safety, recording and viewer.

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230?style=flat-square&logo=ruff&logoColor=D7FF64)](https://docs.astral.sh/ruff/)
[![mypy](https://img.shields.io/badge/types-mypy%20strict-1F5082?style=flat-square)](https://mypy-lang.org/)
[![Hardware](https://img.shields.io/badge/hardware-dual%20YAM-FF7300?style=flat-square)](docs/molmoact-yam-runbook.md)
[![Maintained](https://img.shields.io/badge/maintained-actively-2EA043?style=flat-square)](https://github.com/SII-LiuLab/manimux/commits)

[English](README.md) · [简体中文](README.zh-CN.md)

</div>

![ManiMux viewer during a MolmoAct rollout on dual YAM arms](assets/viewer-hero.png)

## ManiMux in 30 Seconds

**ManiMux is an asynchronous inference runtime between VLA policies and real robots.** The policy
decides what should happen next; ManiMux decides when and how to execute it smoothly and safely,
then records the complete run.

```text
cameras + robot state
        ↓
Policy / XPolicy model     predicts the next 16-50 action steps
        ↓
PolicyAdapter              converts them into executable robot joints
        ↓
ManiMux Runtime            trims stale actions, schedules, blends and checks safety
        ↓
Robot                      submits both arms atomically on one control tick

Side outputs: Recorder episode + 3D Viewer
```

Remember three rules:

1. **The model never commands the robot directly.** It only returns a future action chunk.
2. **The adapter translates semantics.** Joint, EE-pose and delta outputs become one canonical chunk.
3. **ManiMux executes.** Control timing, Timeline, Safety and Recorder are not copied per model.

## Ten-Minute Hardware-Free Start

```bash
git clone --recursive https://github.com/SII-LiuLab/manimux.git
cd manimux
uv sync --dev
uv run manimux run --config configs/mock.yaml
```

`configs/mock.yaml` uses a fake robot, camera and policy while exercising the real worker,
Timeline, SmoothExecutor, SafetyGuard and Recorder. It runs `120` control ticks and writes one
episode under `data/`; it never touches a camera, CAN interface or physical arm.

Launch the independent YAM viewer demo:

```bash
uv run manimux-viewer --robot yam --demo --port 8086
```

Open `http://localhost:8086`. The `--demo` process is a standalone visualization demo, not a live
view of the mock runtime above.

### Try Three Safe Changes

Copy the config first so the repository baseline stays unchanged:

```bash
cp configs/mock.yaml /tmp/manimux-beginner.yaml
```

Change one field at a time and rerun:

| Change | Suggested value | What it demonstrates |
|---|---:|---|
| `run.max_steps` | `120 → 300` | Control ticks and the episode lifecycle |
| `policy.inference_delay_s` | `0.04 → 0.20` | The control loop does not block on a slower model |
| `execution.blend_steps` | `0 / 2 / 8` | The seam between consecutive action chunks |

Read the first config as seven blocks: `run` defines the experiment, `robot` the embodiment,
`sensors` the observation, `policy` the model and chunk, `execution` scheduling and commands,
`viewer` live display, and `recording` the recording policy (the current runtime always records).
See the
[field-by-field config tour](configs/README.md#第一份配置configsmockyaml).

## Where Beginners Should Look

| Path | Think of it as |
|---|---|
| `configs/` | Experiment entry points: model, embodiment, checkpoint and runtime |
| `docs/*-runbook.md` | Operating instructions for installing, starting, checking and stopping a model |
| `src/manimux/` | Shared engine: runtime, robot, sensor, safety, recording and viewer |
| `XPolicyLab/` | Model internals: adapters and samplers for Pi05, GR00T, XR-1 and LingBot |
| `data/` | Run output: resolved config, actions, states, events and episode result |

Running experiments normally requires only `configs/` and one runbook. Enter `src/manimux/` to
change shared infrastructure, and enter `XPolicyLab/` only for model preprocessing, flow denoising
or model-native RTC sampling.

## Purpose

VLA inference is commonly slower than a robot control period and returns actions in chunks.
ManiMux moves inference out of the control loop and owns chunk scheduling, stale-prefix trimming,
atomic dual-arm commits, execution constraints, safety checks, recording and live visualization.

ManiMux is not tied to one model library. MolmoAct and ABC have native service adapters;
additional foundation models enter through `XPolicyLab/`. ManiMux keeps the stable runtime and
wire bridge. Model preprocessing, flow denoising and model-native RTC sampling stay in the
XPolicyLab adapter; embodiment mapping and execution safety stay in ManiMux.

## Architecture

Two loops at different speeds. The whole design is about the hand-off between them.

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 260, "curve": "basis"}}}%%
flowchart TB
    OBS["<b>BUILD ONE OBSERVATION</b><br/>3 cameras + 14 joint angles<br/>which views · what size · what units<br/><br/>PolicyAdapter"]:::stage
    QUEUE(["bounded queue<br/><b>newest wins</b>"]):::pipe

    subgraph THINK["<b>THINK</b> — every ~0.5 s, in its own process, 120-600 ms — <b>swap the model here</b>"]
        direction LR
        MOLMO["<b>MolmoAct2</b><br/>native"]:::molmo
        ABCM["<b>ABC</b><br/>native"]:::abc
        XR1["<b>XR-1</b><br/>XPolicy"]:::xr1
        PI05["<b>Pi05</b><br/>XPolicy"]:::pi
        GROOT["<b>GR00T N1.7</b><br/>XPolicy"]:::groot
        LING["<b>LingBot-VLA2</b><br/>XPolicy"]:::ling
        MOLMO ~~~ ABCM ~~~ XR1 ~~~ PI05 ~~~ GROOT ~~~ LING
    end

    DEC["<b>MAKE IT EXECUTABLE<br/>ON THIS BODY</b><br/>joint order · units · gripper<br/><b>EE→joint IK</b><br/><br/>PolicyAdapter"]:::stage
    TL["<b>PUT THE CHUNK<br/>ON THE CLOCK</b><br/>drop steps already past<br/>blend the seam<br/><b>swap BOTH arms at once</b><br/><br/>ActionTimeline"]:::handoff
    ACT["<b>ACT</b> · every 10-50 ms<br/><b>never waits for anything</b><br/>where should the arms be now?<br/>cap speed and acceleration<br/>refuse out-of-limit commands<br/><br/>Smooth │ MPC · SafetyGuard"]:::stage
    ROBOT(["<b>one command,<br/>both arms together</b><br/>RobotDriver"]):::robot
    SIDE(["episode on disk<br/>live 3D viewer"]):::side

    OBS --> QUEUE --> THINK
    THINK -->|"a chunk:<br/>next 16-50 steps"| DEC
    DEC -->|"canonical<br/>joint-space chunk"| TL
    TL --> ACT --> ROBOT
    ACT -.->|"best-effort"| SIDE

    classDef stage fill:#F6F8FA,stroke:#8C959F,stroke-width:1px,color:#1F2328
    classDef handoff fill:#FFF4E5,stroke:#E36209,stroke-width:2.5px,color:#1F2328
    classDef pipe fill:#FFFFFF,stroke:#8C959F,stroke-dasharray:4 3,color:#57606A
    classDef side fill:#FFFFFF,stroke:#8C959F,stroke-dasharray:4 3,color:#57606A
    classDef robot fill:#1F2328,stroke:#1F2328,color:#FFFFFF
    classDef molmo fill:#2F6FEB,stroke:#1B4DB1,color:#FFFFFF
    classDef abc   fill:#1A7F55,stroke:#125C3D,color:#FFFFFF
    classDef xr1   fill:#8957E5,stroke:#6633B8,color:#FFFFFF
    classDef pi    fill:#E36209,stroke:#A8460A,color:#FFFFFF
    classDef groot fill:#CF222E,stroke:#96101A,color:#FFFFFF
    classDef ling  fill:#9A6700,stroke:#6E4A00,color:#FFFFFF
    style THINK fill:#FFFFFF,stroke:#8C959F,stroke-dasharray:5 4,color:#1F2328
```

**The control loop never waits** — not for the model, disk, viewer, or a log line.
**A stale or invalid chunk never reaches the robot** — wrong session, old sequence,
past deadline, bad shape, non-finite value, or a dual-arm plan the two arms disagree
on drops the *whole* chunk with a logged reason.

RTC is an explicit alternative for the THINK stage: instead of refilling when the
timeline runs low, ManiMux derives a condition and soft mask from real execution
progress, and an RTC-capable adapter injects them into the model's own sampler.

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
| ✅ | OpenPI Pi05 + YAM | 16 × 14 absolute joint positions | `configs/pi05/yam/` | [Pi05](docs/pi05-yam-runbook.md) |
| ✅ | GR00T N1.7 + YAM | 16 × 14 absolute joint positions | `configs/groot/yam/` | [GR00T](docs/gr00t-yam-runbook.md) |
| ✅ | XR-1 + YAM | `30×60` EE delta → `30×14` joints | `configs/xiaomi-xr1/yam/` | [Runbook](docs/xiaomi-xr1-yam-runbook.md) |
| ✅ | LingBot-VLA2 + YAM | `50×14` joints; limited base capability | `configs/lingbot-vla2/yam/` | [Runbook](docs/lingbot-vla2-yam-runbook.md) |
| 🔌 | XPolicy bridge | Observation/action wire contract | `configs/xpolicylab/yam/` | [Runbook](docs/xpolicylab-runbook.md) |

OpenPI Pi05 has completed the real three-camera, YAM normalization, XPolicy model-server,
official 10-step flow sampling, default ManiMux and Pi-guided RTC paths on dual YAM. RTC ran with
measured `d=3-5` steps and no post-start chunk gap. The checkpoint produced task-related motion,
but remained hesitant in this scene and has no established success rate; that policy-quality
result is separate from the completed inference infrastructure. GR00T has also completed GPU,
XPolicy WebSocket, default ManiMux, three-camera, dual-YAM and Recorder execution; its failed pick
rollouts are policy-quality results. XPolicy XR-1 and LingBot-VLA2 have also completed their
default ManiMux hardware paths using `server/base.yaml` with the shared `infra/manimux.yaml`.
Here ✅ means GPU, XPolicy, cameras, scheduling, dual-arm execution and Recorder are connected;
it does not establish YAM task success for a base checkpoint, and projection statistics are not
evidence of YAM post-training.

XR-1 has completed a real 5B GPU forward, XPolicy WebSocket, ManiMux, camera,
dual-YAM and Recorder end-to-end startup, confirming that the robot executed
model output. In the first base rollout the right arm was comparatively normal
while the left arm made persistently abnormal large motions. The infrastructure
path is therefore connected, but the YAM action-semantics gate remains failed;
this was neither startup-pose motion nor evidence of YAM zero-shot capability.

## From Mock to Real Models

Update submodules in an existing checkout:

```bash
git submodule update --init --recursive
```

The core development environment uses `./.venv`, the YAM runtime uses `envs/yam/.venv`, and model
servers use isolated environments. Follow [the environment guide](envs/README.md) and one model
runbook for CUDA, Torch, checkpoints and first startup; do not assemble a hardware command from
README snippets.

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
- [MolmoAct2](docs/molmoact-yam-runbook.md) · [ABC](docs/abc-yam-runbook.md)
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
