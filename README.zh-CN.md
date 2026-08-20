<div align="center">

# ManiMux

**面向真机 VLA 的本地异步推理与执行基础设施。**

模型可以替换，控制环、机器人、安全、记录和 Viewer 始终由 ManiMux 管理。

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Lint: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://docs.astral.sh/ruff/)
[![Typing: mypy strict](https://img.shields.io/badge/typing-mypy%20strict-2a6db2.svg)](https://mypy-lang.org/)

[English](README.md) · [简体中文](README.zh-CN.md)

</div>

![MolmoAct 驱动双 YAM 臂时的 ManiMux viewer](assets/viewer-hero.png)

## 项目定位

VLA 推理通常比机器人控制周期慢，而且一次输出一段 action chunk。ManiMux 把模型推理
移出控制环，并统一负责 chunk 调度、过期裁剪、双臂原子提交、执行约束、安全检查、
数据记录和实时可视化。

ManiMux 不绑定某个模型库。MolmoAct、ABC、XR-1 可以通过原生服务接入；更多基模通过
`XPolicyLab/` 接入。我们只在 ManiMux 中维护稳定的运行时和 wire bridge，需要修改模型
预处理、flow denoise 或 RTC 采样时，代码留在 XPolicyLab 对应的模型 adapter 内。

## 架构

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

默认使用 ManiMux 异步 runtime。RTC 是显式选择的另一种推理调度方式：ManiMux 根据真实
执行进度生成 condition/soft mask，支持 RTC 的模型 adapter 再把它注入原生采样过程。

## 代码结构

| 路径 | 归属 |
|---|---|
| `src/manimux/runtime/` | 控制环、Timeline、Smooth/MPC、RTC 和 Safety |
| `src/manimux/policies/` | `PolicyModel` / `PolicyAdapter` 契约与隔离 worker |
| `src/manimux/integrations/` | 原生模型集成和统一 XPolicy WebSocket bridge |
| `src/manimux/robots/` | RobotDriver；当前真机实现为双 YAM |
| `src/manimux/sensors/` | RealSense 和共享相机服务 |
| `src/manimux/recording/` | episode、Zarr、事件和动作血缘 |
| `src/manimux/viewer/` | 通用 Viewer 协议和机器人几何 adapter |
| `XPolicyLab/` | 指向我们 XPolicyLab fork 的 submodule；模型内部改动在这里 |
| `configs/<model>/<embodiment>/` | 每个模型 × 本体的独立实验配置 |
| `docs/*-runbook.md` | 每个模型的安装、启动、检查和停止流程 |

## 当前集成

状态：✅ 已跑通 · 🧪 实验性 · 🚧 尚未满足部署条件 · 🔌 基础设施

| 状态 | 模型/链路 | ManiMux canonical action | 配置 | Runbook |
|---|---|---|---|---|
| ✅ | MolmoAct2 + YAM | 30 × 14 关节位置 | `configs/molmoact2/yam/` | [MolmoAct2](docs/molmoact-yam-runbook.md) |
| ✅ | ABC + YAM | 30 × 14 关节位置 | `configs/abc/yam/` | [ABC](docs/abc-yam-runbook.md) |
| 🧪 | XR-1 native + YAM | 30 × 60 末端增量 → 30 × 14 关节位置 | `configs/xiaomi-xr1/yam/infra/native.yaml` | [XR-1 native](docs/xr1-yam-runbook.md) |
| ✅ | OpenPI Pi05 + YAM | 16 × 14 绝对关节位置 | `configs/pi05/yam/` | [Pi05](docs/pi05-yam-runbook.md) |
| 🚧 | GR00T N1.7 + YAM | 16 × 14 绝对关节位置 | `configs/gr00t-n17/yam/` | [GR00T](docs/gr00t-yam-runbook.md) |
| 🧪 | XPolicy XR-1 + YAM | 30 × 60 末端增量 → 30 × 14 关节位置 | `configs/xiaomi-xr1/yam/{server,infra}/` | [XPolicy XR-1](docs/xiaomi-xr1-yam-runbook.md) |
| 🚧 | LingBot-VLA2 + YAM | 等待 55 维语义、映射和 stats | — | [LingBot-VLA2](docs/lingbot-vla2-yam-runbook.md) |
| 🔌 | XPolicy bridge | 标准 observation/action wire contract | `configs/xpolicylab/yam/infra/smoke.yaml` | [XPolicyLab](docs/xpolicylab-runbook.md) |

OpenPI Pi05 已完成真实三相机、YAM norm stats、XPolicy 模型服务、官方 10-step flow、
普通 ManiMux 和 Pi-guided RTC 的双 YAM 全链路运行。RTC 实测 `d=3-5` 步，首个 chunk 后
没有空档。checkpoint 在当前场景能产生任务相关运动，但仍有明显犹豫且没有正式成功率；
这是 policy 质量结论，不代表推理 infra 未完成。GR00T 和 XPolicy XR-1 当前不能视为真机
验证完成。

## 安装

```bash
git clone --recursive https://github.com/SII-LiuLab/manimux.git
cd manimux
uv sync --dev
```

已有 checkout：

```bash
git submodule update --init --recursive
```

核心 runtime 使用 `./.venv`；真机 runtime 使用 `envs/yam/.venv`；不同模型服务使用各自
隔离环境。具体 CUDA、Torch 和 checkpoint 安装只写在
[环境说明](envs/README.md)与对应模型 runbook 中。

## 快速开始

### 无硬件

```bash
uv run manimux run --config configs/mock.yaml
uv run manimux-viewer --robot yam --demo --port 8086
```

### YAM 公共服务

相机和 Viewer 与模型无关，同一组实验只需各启动一次：

```bash
envs/yam/.venv/bin/manimux-camera-server --config configs/cameras.yaml
envs/yam/.venv/bin/manimux-viewer --robot yam --host 0.0.0.0 --port 8086
```

### 原生模型示例：MolmoAct2

```bash
envs/yam/.venv/bin/manimux-molmoact-server --host 127.0.0.1 --port 8202
envs/yam/.venv/bin/manimux run --config configs/molmoact2/yam/infra/manimux.yaml
```

### XPolicy 示例：Pi05-YAM

```bash
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  scripts/pi05_yam_server.py \
  --config configs/pi05/yam/server/finetune.yaml

envs/yam/.venv/bin/manimux run --config configs/pi05/yam/infra/manimux.yaml

# Pi-guided RTC 复用同一个微调模型服务。
envs/yam/.venv/bin/manimux run --config configs/pi05/yam/infra/rtc.yaml
```

这些只展示入口，不代替 checkpoint 检查、preflight、CAN 检查和停止流程。运行真机前
必须打开对应 runbook，并确认急停、工作区和两路 CAN 状态。

## 配置约定

```text
configs/
  <model>/
    <embodiment>/
      server/              # 模型服务配置
        <checkpoint>.yaml
      infra/               # ManiMux 机器人、传感器、执行和记录配置
        <experiment>.yaml
  cameras.yaml
  robots/
  mock.yaml
```

一个实验使用一个明确配置；不同 checkpoint、runtime 或本体不共用隐式开关。详细命名
规则见 [configs/README.md](configs/README.md)。

## 接入新模型或本体

| 边界 | 职责 |
|---|---|
| `PolicyModel` | 调用本地模型或远端模型服务，不接触机器人 |
| `PolicyAdapter` | 相机/状态编码、action 解码、归一化、关节映射和 FK/IK |
| `RobotDriver` | 厂商 SDK、状态读取、命令、home、stop 和 close |
| `SensorDriver` | 读取带时间戳的命名传感器流 |
| `RobotAdapter` | Viewer 中的 URDF、FK、关节分组和外观 |

更换模型只改 Policy 两层；更换机器人只改本体 adapter、driver 和配置。Runtime、Timeline、
Safety、Recorder 与 Viewer 协议不随模型复制。完整接口见
[架构文档](docs/architecture.md)。

## 文档

- [XPolicyLab 集成](docs/xpolicylab-runbook.md)
- [MolmoAct2](docs/molmoact-yam-runbook.md) · [ABC](docs/abc-yam-runbook.md) · [XR-1 native](docs/xr1-yam-runbook.md)
- [Pi05](docs/pi05-yam-runbook.md) · [GR00T](docs/gr00t-yam-runbook.md) · [XPolicy XR-1](docs/xiaomi-xr1-yam-runbook.md)
- [LingBot-VLA2](docs/lingbot-vla2-yam-runbook.md) · [CAN 总线](docs/can-bus.md)
- [架构](docs/architecture.md) · [开发约定](AGENT.md)

## 安全

ManiMux 会驱动真实机械臂。测试全绿不代表硬件安全；物理急停、限位和厂商 watchdog
不能由软件替代。Runtime 启动后默认暂停，fault 不自动恢复，任何真机入口都应先完成
对应 runbook 的 preflight。

## 许可

上游组件归属见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和
[`licenses/`](licenses)。
