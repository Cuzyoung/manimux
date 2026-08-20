<div align="center">

# ManiMux

**面向真机 VLA 的本地异步推理与执行基础设施。**

模型可以替换，控制环、机器人、安全、记录和 Viewer 始终由 ManiMux 管理。

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230?style=flat-square&logo=ruff&logoColor=D7FF64)](https://docs.astral.sh/ruff/)
[![mypy](https://img.shields.io/badge/types-mypy%20strict-1F5082?style=flat-square)](https://mypy-lang.org/)
[![Hardware](https://img.shields.io/badge/hardware-dual%20YAM-FF7300?style=flat-square)](docs/molmoact-yam-runbook.md)
[![RTC](https://img.shields.io/badge/arXiv-2506.07339-B31B1B?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2506.07339)
[![Maintained](https://img.shields.io/badge/maintained-actively-2EA043?style=flat-square)](https://github.com/SII-LiuLab/manimux/commits)

[English](README.md) · [简体中文](README.zh-CN.md)

</div>

![MolmoAct 驱动双 YAM 臂时的 ManiMux viewer](assets/viewer-hero.png)

## 项目定位

VLA 推理通常比机器人控制周期慢，而且一次输出一段 action chunk。ManiMux 把模型推理
移出控制环，并统一负责 chunk 调度、过期裁剪、双臂原子提交、执行约束、安全检查、
数据记录和实时可视化。

ManiMux 不绑定某个模型库。MolmoAct、ABC 通过原生服务接入；更多基模通过
`XPolicyLab/` 接入。我们只在 ManiMux 中维护稳定的运行时和 wire bridge，需要修改模型
预处理、flow denoise 或 RTC 采样时，代码留在 XPolicyLab 对应的模型 adapter 内。

XR-1 使用我们在 XPolicyLab fork 中维护的 `Xiaomi_Robotics_1` adapter，底层是小米
官方 XR-1 源码；它不是 XPolicyLab 上游仓库原本自带的 adapter。ManiMux 只保留 wire
codec 和 YAM FK/IK 映射，模型加载与 denoise 都在 XPolicyLab 内执行。

因此 XR-1 服务返回的是官方 `Xiaomi-Robotics-1-5B` 的真实模型推理动作，不是启动姿态、
预录轨迹或假数据。需要单独验收的是后半段：该 base checkpoint 没有在 YAM 上训练，
当前 YAM stats 只定义投影单位，而 `30 × 60` 末端增量到 `30 × 14` 关节位置的 FK/IK
转换属于 ManiMux 本体 adapter。

## 架构

两个速度不同的循环。整个设计的重点是它们之间的交接。

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 260, "curve": "basis"}}}%%
flowchart TB
    OBS["<b>构造一次 OBSERVATION</b><br/>三路相机 + 14 个关节角<br/>用哪几路视角 · 缩放到多大 · 归一化到什么单位<br/><br/>PolicyAdapter"]:::stage
    QUEUE(["有界队列<br/><b>只留最新一条</b>"]):::pipe

    subgraph THINK["<b>想</b> — 约每 0.5 秒一次，跑在独立进程里，120-600 ms — <b>换模型只换这一层</b>"]
        direction LR
        MOLMO["<b>MolmoAct2</b><br/>原生"]:::molmo
        ABCM["<b>ABC</b><br/>原生"]:::abc
        XR1["<b>XR-1</b><br/>XPolicy"]:::xr1
        PI05["<b>Pi05</b><br/>XPolicy"]:::pi
        GROOT["<b>GR00T N1.7</b><br/>XPolicy"]:::groot
        LING["<b>LingBot-VLA2</b><br/>XPolicy"]:::ling
        MOLMO ~~~ ABCM ~~~ XR1 ~~~ PI05 ~~~ GROOT ~~~ LING
    end

    DEC["<b>让它能在这具身体上执行</b><br/>关节顺序 · 单位 · 夹爪约定<br/><b>末端 → 关节 IK</b><br/><br/>PolicyAdapter"]:::stage
    TL["<b>把 CHUNK 放到时间轴上</b><br/>裁掉已经过去的那几步<br/>接缝处融合<br/><b>双臂在同一步原子切换</b><br/><br/>ActionTimeline"]:::handoff
    ACT["<b>做</b> · 每 10-50 ms 一次<br/><b>永远不等任何东西</b><br/>此刻手应该在哪里？<br/>限制速度和加速度<br/>超出限位的一律拒绝<br/><br/>Smooth │ MPC · SafetyGuard"]:::stage
    ROBOT(["<b>双臂同一 tick<br/>下发一条命令</b><br/>RobotDriver"]):::robot
    SIDE(["落盘的 episode<br/>实时 3D viewer"]):::side

    OBS --> QUEUE --> THINK
    THINK -->|"一个 chunk：<br/>未来 16-50 步"| DEC
    DEC -->|"canonical<br/>关节空间 chunk"| TL
    TL --> ACT --> ROBOT
    ACT -.->|"尽力而为"| SIDE

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

**控制环永不等待** —— 不等模型、磁盘、viewer，也不等一行日志。
**过期或非法的 chunk 绝不会到达机器人** —— session 不符、sequence 过旧、超过
deadline、维度错误、含非有限值、左右臂 plan 不一致，都会**整块**丢弃并写明原因。

RTC 是「想」这一级的另一种选择：不再等时间轴见底才补，而是由 ManiMux 从真实执行
进度推出条件和软掩码，再由支持 RTC 的 adapter 注入模型自己的采样器。

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
| `checkpoints/pretrained/` | 未经本项目微调的基础模型或上游发布权重 |
| `checkpoints/finetuned/<publisher>/` | 按发布者和公开仓库名区分的微调 checkpoint |
| `configs/<model>/<embodiment>/` | 每个模型 × 本体的独立实验配置 |
| `docs/*-runbook.md` | 每个模型的安装、启动、检查和停止流程 |

例如 Robocurve 发布的两个 YAM checkpoint 位于
`checkpoints/finetuned/robocurve/{pi05-yam-molmoact2,gr00t-n1.7-yam-molmoact2}`；
OpenPI 官方 `pi05_base` 仍保留在 `checkpoints/pretrained/`。

## 当前集成

状态：✅ 已跑通 · 🧪 实验性 · 🚧 尚未满足部署条件 · 🔌 基础设施

| 状态 | 模型/链路 | ManiMux canonical action | 配置 | Runbook |
|---|---|---|---|---|
| ✅ | MolmoAct2 + YAM | 30 × 14 关节位置 | `configs/molmoact2/yam/` | [MolmoAct2](docs/molmoact-yam-runbook.md) |
| ✅ | ABC + YAM | 30 × 14 关节位置 | `configs/abc/yam/` | [ABC](docs/abc-yam-runbook.md) |
| ✅ | OpenPI Pi05 + YAM | 16 × 14 绝对关节位置 | `configs/pi05/yam/` | [Pi05](docs/pi05-yam-runbook.md) |
| ✅ | GR00T N1.7 + YAM | 16 × 14 绝对关节位置 | `configs/groot/yam/` | [GR00T](docs/gr00t-yam-runbook.md) |
| ✅ | XR-1 + YAM | `30×60` 末端增量 → `30×14` 关节 | `configs/xiaomi-xr1/yam/` | [运行手册](docs/xiaomi-xr1-yam-runbook.md) |
| ✅ | LingBot-VLA2 + YAM | `50×14` 关节；base 任务能力有限 | `configs/lingbot-vla2/yam/` | [运行手册](docs/lingbot-vla2-yam-runbook.md) |
| 🔌 | XPolicy bridge | observation/action wire contract | `configs/xpolicylab/yam/` | [运行手册](docs/xpolicylab-runbook.md) |

OpenPI Pi05 已完成真实三相机、YAM norm stats、XPolicy 模型服务、官方 10-step flow、
普通 ManiMux 和 Pi-guided RTC 的双 YAM 全链路运行。RTC 实测 `d=3-5` 步，首个 chunk 后
没有空档。checkpoint 在当前场景能产生任务相关运动，但仍有明显犹豫且没有正式成功率；
这是 policy 质量结论，不代表推理 infra 未完成。GR00T 也已经完成 GPU、XPolicy
WebSocket、默认 ManiMux、真实三相机、双臂 YAM 和 Recorder 闭环；其 pick 失败属于
policy 质量结果。XPolicy XR-1 和 LingBot-VLA2 也已完成默认 ManiMux 真机链路；
两者的 base 权重统一使用 `server/base.yaml` 加同一份 `infra/manimux.yaml` 测试。
这里的 ✅ 表示 GPU、XPolicy、相机、调度、双臂执行和 Recorder 已连通，不表示 base
checkpoint 已具备 YAM 任务成功率；YAM projection stats 也不能当作完成后训练的证据。

XR-1 已完成真实 5B GPU forward、XPolicy WebSocket、ManiMux、相机、双臂 YAM 和
Recorder 的全链路启动，确认机械臂执行的是模型输出。首次 base rollout 中右臂相对正常、
左臂持续出现异常大幅运动，因此 infra 链路记为已通，但 YAM 动作语义 Gate 未通过；这既
不能归类为初始姿态移动，也不能当作 checkpoint 已具备 YAM zero-shot 能力。

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

- [项目总进度](docs/program-progress.md)
- [XPolicyLab 集成](docs/xpolicylab-runbook.md)
- [MolmoAct2](docs/molmoact-yam-runbook.md) · [ABC](docs/abc-yam-runbook.md)
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
