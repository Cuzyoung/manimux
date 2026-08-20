# Xiaomi XR-1 + YAM 运行手册

本文只覆盖 XPolicyLab 的 `Xiaomi_Robotics_1` 路线。原有 native
`configs/xiaomi-xr1/yam/infra/native.yaml` 保留作为独立基线。

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
使用训练 stats 归一化后进入五步 flow denoise 的 PiGDM conditioning。它是正式 sampler
hook，不是 monkeypatch，也没有额外动作幅度阈值。

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

当前还没有真实 GPU load/forward 证据。完成它之前不要进入真机阶段。

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
payload 和定向单测。尚未启动模型服务、GPU forward、真实相机、CAN、preflight 或真机。
`model_states.pt` 是 post-training 起点，不是已经证明能在 YAM 上完成任务的策略。
