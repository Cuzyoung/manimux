<div align="center">

# ManiMux

**A local, asynchronous policy runtime for dual-arm manipulation.**

Run a VLA on real hardware without ever blocking the control loop.

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Lint: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://docs.astral.sh/ruff/)
[![Typing: mypy strict](https://img.shields.io/badge/typing-mypy%20strict-2a6db2.svg)](https://mypy-lang.org/)

[English](README.md) · [简体中文](README.zh-CN.md)

</div>

<!-- 📷 SCREENSHOT — assets/viewer-hero.png
     The Viser viewer during a real rollout: both YAM arms rendered, the purple
     predicted EE trajectory, camera panels, and the GUI column showing
     Policy / Chunk / Inference. Browser full screen, ~1600px wide. -->

![ManiMux viewer during a MolmoAct rollout on dual YAM arms](assets/viewer-hero.png)

## Why

Modern policies are slow and chunked: a VLA takes 120–600 ms to emit 30 future
steps, while a controller needs a fresh command every 10–50 ms. Bolt them
together naively and the arm either stutters waiting for inference, or jumps
when a freshly sampled chunk disagrees with the half-executed one.

ManiMux is the layer in between. Inference runs off the control loop; chunks are
time-aligned, trimmed, and swapped **atomically across both arms**; the result
becomes velocity-limited joint commands, a full action lineage on disk, and a
live 3D view of what the policy intends to do *before* it does it.

Everything is local — one YAML per run, one directory per episode. No cloud
inference, gateway, control plane, or database.

## Pipeline

```text
camera-server ──► SensorDriver ──► Observation ──► bounded latest-wins queue
                                                            │
policy-server ◄── PolicyModel ◄─────────────────────────────┘
      │
      └─► raw chunk ─► PolicyAdapter ─► ActionTimeline ─► Smooth│MPC ─► Safety
                       validate·IK·map    atomic swap           │
                                                                ▼
                                                    RobotDriver.send_command()
                                                                │
                                       Recorder ◄── best-effort ─┴─► Viewer
```

**The control loop never waits** — not for the model, disk, viewer, or a log
line. **A stale or invalid chunk never reaches the robot** — wrong session, old
sequence, past deadline, bad shape, non-finite value, or mismatched dual-arm
plan id drops the *whole* chunk with a logged reason.

## What's in the box

| Part | Path | What it does |
|---|---|---|
| Runtime | `runtime/edge.py` | Fixed-rate control loop, observation building, watchdogs, safety states |
| Timeline | `runtime/timeline.py` | Time-indexed chunk store: stale-prefix trim, blend window, atomic dual-arm commit |
| Executors | `runtime/executors/` | `smooth` (resample + low-pass + limits) and joint-space local `mpc` |
| RTC runtime | `runtime/rtc/` | [Real-Time Chunking](https://arxiv.org/abs/2506.07339) as a parallel runtime — inpainting-style chunk stitching instead of downstream filtering |
| Robots | `robots/` | `RobotDriver` contract · `mock` · ManiUniCon/Meshcat · dual **YAM** over CAN via i2rt |
| Sensors | `sensors/` | `SensorDriver` contract · `mock` · RealSense · standalone ZMQ **camera server** shared by every policy |
| Policies | `policies/` | `PolicyModel` (inference only) + `PolicyAdapter` (all embodiment semantics), bounded worker queue |
| Kinematics | `kinematics/` | Reusable FK/IK so EE-space policies become joint chunks in the adapter, not in the runtime |
| Recording | `recording/` | Zarr + JSONL + MP4 episodes, `.partial` until atomically finalized |
| Viewer | `viewer/` | Viser dashboard, robot-agnostic ZMQ protocol, `RobotAdapter` geometry boundary |
| Integrations | `integrations/` | MolmoAct2 · ABC · XR-1 model servers and adapters |

Three policies ship working, sharing the *same* robot, camera, and viewer layers:

| Policy | Params | Action space | Chunk | Latency | Config |
|---|---|---|---|---|---|
| [MolmoAct2](https://huggingface.co/allenai/MolmoAct2-BimanualYAM) | — | joint positions | 30 × 14 | ~240 ms | `configs/molmoact-yam-live.yaml` |
| ABC-DiT XL | 2.02 B | joint positions | 30 × 14 | ~170 ms | `configs/abc-yam-live.yaml` |
| Xiaomi XR-1 | 5.5 B | EE deltas → IK | 30 × 60 | ~121 ms | `configs/xr1-yam-live.yaml` |

## Install

```bash
git clone https://github.com/SII-LiuLab/manimux.git && cd manimux
uv sync --dev            # core runtime + mock stack + viewer, into ./.venv
```

The three policies are validated against mutually incompatible torch builds
(cu121 / cu128 / cu126), so each hardware stack gets its own venv under `envs/`.
For MolmoAct:

```bash
uv venv envs/yam/.venv --python 3.12
uv pip install --python envs/yam/.venv/bin/python -e ".[molmoact-yam]"
uv pip install --python envs/yam/.venv/bin/python \
  "git+https://github.com/i2rt-robotics/i2rt.git@5d47b358bafb30c65e397f2ece506550a0db4594"
```

`i2rt` is the vendor CAN driver, pinned separately because it is not ManiMux
code. `envs/abc` and `envs/xr1` follow the same pattern with their own torch
installed first — the `abc-yam` and `xr1-yam` extras deliberately omit `torch`.

Which venv does what — the `envs/*` ones are hand-managed and must stay outside
`uv sync`, which would strip their undeclared `i2rt`/`torch`:

| venv | Used for |
|---|---|
| `./.venv` | uv-managed: `make` targets, mock runs, viewer demo. Core + dev only |
| `envs/yam` | **Every real-robot process** — MolmoAct server, camera server, viewer, runtime |
| `envs/abc` · `envs/xr1` | Their own model server only; the runtime still comes from `envs/yam` |

## Quick start — no hardware

```bash
uv run manimux run --config configs/mock.yaml           # full async loop, mock everything
uv run manimux run --config configs/mock.yaml --executor mpc
uv run manimux-viewer --robot yam --demo --port 8086    # viewer with synthetic plans
```

## Real-robot rollout

> ⚠️ Clear the workspace, keep the e-stop in hand, and confirm `can_left` /
> `can_right` are `ERROR-ACTIVE` ([docs/can-bus.md](docs/can-bus.md)).

MolmoAct2 on dual YAM arms. Four processes in **four separate terminals** — each
stays in the foreground, so they cannot be pasted into one shell. All four run
from the same `envs/yam` venv.

**1 · Model server.** MolmoAct on the GPU; every policy has its own binary
(`manimux-{molmoact,abc,xr1}-server`). Wait for `Warmup OK`.

```bash
envs/yam/.venv/bin/manimux-molmoact-server --host 127.0.0.1 --port 8202
```

**2 · Camera server.** Owns the RealSenses, policy-agnostic. Wait for `REP bound`.

```bash
envs/yam/.venv/bin/manimux-camera-server --config configs/cameras.yaml
```

**3 · Viewer.** Policy-agnostic too, then open http://localhost:8086.

```bash
envs/yam/.venv/bin/manimux-viewer --robot yam --host 0.0.0.0 --port 8086
```

**4 · Runtime.** The only process that opens CAN — and it still needs terminals
1 and 2 up. **This one moves the arms.**

```bash
envs/yam/.venv/bin/manimux run --config configs/molmoact-yam-live.yaml
```

The arms move to their start pose, run the rollout, and return home on `Ctrl-C`
or at `run.max_steps`. Speed and timing are config-only — `policy.action_dt_s`,
`execution.smooth.max_velocity` / `max_acceleration`,
`execution.refill_threshold_s`, `robot.options.*_duration_s`. Running ABC, XR-1,
or RTC instead swaps terminal 1's server and terminal 4's config while terminals
2 and 3 stay up — see the runbooks below.

Episodes land in `data/<run-id>/<episode-id>/` as `data.zarr` + `events.jsonl` +
`videos/*.mp4` + `result.json`, with five time-aligned stages always present:

```text
raw_model_action → scheduled_action → optimized_action → command_sent → measured_state
```

## Extending

Four small `Protocol`s, and nothing else has to change. A config names a plugin
by builtin name, entry point, or `module:factory`; the launcher self-checks and
fails fast at startup.

| Boundary | Implement | Owns | Reference |
|---|---|---|---|
| `RobotDriver` | `connect · get_state · send_command · home · stop · close` | Vendor SDK, CAN, named groups (`left_arm`…), joint order and units | `robots/yam/` |
| `SensorDriver` | `start · read · close` | Timestamped frames; may publish several named streams from one connection | `sensors/camera_server/` |
| `PolicyModel` | `reset · infer · close` | Inference only — no robot access, usually a thin HTTP client to its own venv | `integrations/*/policy_plugin.py` |
| `PolicyAdapter` | `build_observation · decode_action · validate` | **All** model×body semantics: camera mapping, normalization, group order, EE→joint IK, gripper convention, shape/finite checks | `integrations/*/policy_plugin.py` |
| `RobotAdapter` (viewer) | `split_actions · pose · visual_configuration · …` | URDF, FK, colors — the dashboard itself stays robot-agnostic | `viewer/robots/yam.py` |

The timeline only ever accepts canonical `joint_position` chunks. An EE-space
policy converts in the adapter via `manimux.kinematics` (XR-1 does exactly
this), so changing arms means changing `policy.options.kinematics` — the model
server never learns the robot exists.

Bring-up order is not optional: **mock first, then low-speed real**, with adapter
unit tests for shapes, units, joint order, and NaN/Inf rejection in between.
Details in [docs/plugin-runtime.md](docs/plugin-runtime.md).

## Docs

[Architecture](docs/architecture.md) ·
[Plugin runtime](docs/plugin-runtime.md) ·
[RTC](docs/rtc-runtime.md) ·
[MolmoAct runbook](docs/molmoact-yam-runbook.md) ·
[ABC runbook](docs/abc-yam-runbook.md) ·
[XR-1 runbook](docs/xr1-yam-runbook.md) ·
[CAN bus](docs/can-bus.md) ·
[Conventions](AGENT.md)

```bash
make format lint typecheck test test-integration mock-run
```

No test touches hardware, a GPU, or the network; hardware-only tests live in
`tests/hardware/` and stay out of CI.

## Safety

ManiMux commands real arms, and passing tests do not prove hardware safety. The
runtime starts **paused** and needs a fresh operator start. Faults never
auto-recover, and either arm faulting faults the whole robot. ManiMux never
masks the vendor e-stop, limits, or watchdog — the physical e-stop does not pass
through software.

## License

Upstream components are attributed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [`licenses/`](licenses):
MolmoAct2 (Apache-2.0), i2rt YAM assets (MIT), CLIP (MIT) and DINOv3,
Xiaomi-Robotics-1 (Apache-2.0). Design ideas borrowed from ManiUniCon's HAL,
Real-Time Chunking, and Universal Viewer's `RobotAdapter`.
