# Xiaomi XR-1 + YAM 运行手册

本文只覆盖 XPolicyLab 的 `Xiaomi_Robotics_1` 路线。XPolicyLab 已集成
[小米官方 XR-1 源码](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1)，用户不需要
再 clone 一份官方仓库。原有 native 路线仅作历史对照：
`configs/xiaomi-xr1/yam/infra/native.yaml`。

## 当前权重边界

本地 `model_states.pt` 来自官方
[`Xiaomi-Robotics-1-5B`](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-5B)。
官方 model card 将它定位为继续 post-training 的起点，不是 YAM 策略。
官方另外发布的 RoboCasa / RoboCasa365 / VLABench 权重也都是特定仿真本体，
不能直接驱动双臂 YAM。

## 动作链路

```text
三路 RGB + 14-D YAM state
  -> XPolicy Xiaomi_Robotics_1
  -> 30 x 60 anchor-relative EE deltas
  -> ManiMux xr1_yam adapter FK/IK
  -> 30 x 14 absolute YAM joints
```

模型原生输出不是 joint position。每一步是相对请求时末端锚点的位置、轴角和夹爪增量；
YAM FK/IK 只存在于 ManiMux adapter 中。

## 配置

```text
server:  configs/xiaomi-xr1/yam/server/xpolicy.yaml
ManiMux: configs/xiaomi-xr1/yam/infra/xpolicy.yaml
RTC:     configs/xiaomi-xr1/yam/infra/xpolicy-rtc.yaml
```

RTC 将 ManiMux `30 x 14` overlap condition 通过 FK 反编码到模型原生 `30 x 60` 空间，
再进入五步 flow denoise 的 PiGDM conditioning。这是 ManiMux/XPolicy 扩展，
不是小米官方 XR-1 部署功能；它通过明确 sampler hook 实现，不是运行时
monkeypatch，也没有额外动作幅度阈值。

`yam.json` 只是用 YAM 数据统一了状态与动作单位，并不是当前权重的训练
statistics。只有使用同一份 YAM 数据 fine-tune 并产出配对 stats 后，
权重和归一化才真正匹配。

## 1. 离线检查

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/xiaomi_xr1_yam_server.py --check
```

## 2. 模型服务

```bash
cd /home/ubuntu/manimux
envs/xr1/.venv/bin/python scripts/xiaomi_xr1_yam_server.py \
  --config configs/xiaomi-xr1/yam/server/xpolicy.yaml
```

当前 XPolicy 路线还没有真实 GPU load/forward 证据。完成它之前不要进入真机阶段。

服务加载完成后，先在另一个 terminal 运行无相机、无 CAN 的单次 forward probe。对于
XR-1，它会同时检查模型原生 `30 x 60` EE delta 和 ManiMux FK/IK 后的 `30 x 14`
absolute joint chunk：

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/xpolicylab_yam_forward_probe.py \
  --config configs/xiaomi-xr1/yam/infra/xpolicy.yaml
```

只有输出 `"status": "ok"`、`"native_shape": [30, 60]` 和
`"canonical_shape": [30, 14]`，才算完成 XPolicy GPU forward、WS 和 FK/IK adapter
往返。当前尚未运行该命令。

## 3. ManiMux 实验

普通 ManiMux 与 RTC 使用独立配置：

```bash
envs/yam/.venv/bin/manimux run --config configs/xiaomi-xr1/yam/infra/xpolicy.yaml
envs/yam/.venv/bin/manimux run --config configs/xiaomi-xr1/yam/infra/xpolicy-rtc.yaml
```

不要同时运行两条。相机、Viewer、CAN 检查和停止顺序参考
[MolmoAct + YAM](molmoact-yam-runbook.md)。

## 当前边界

已离线验证配置、权重/processor/stats 路径、动作 codec、FK/IK condition round-trip、RTC
payload 和定向单测。尚未启动 XPolicy 模型服务、GPU forward、真实相机、CAN、
preflight 或真机。`model_states.pt` 是 post-training 起点，不是已经证明能在 YAM 上
完成任务的策略。
