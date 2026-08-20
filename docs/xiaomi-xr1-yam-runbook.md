# Xiaomi XR-1 + YAM 运行手册

本文只覆盖 XPolicyLab 的 `Xiaomi_Robotics_1` 路线。XPolicyLab 已集成
[小米官方 XR-1 源码](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1)，用户不需要
再 clone 一份官方仓库。这个 adapter 由我们的 XPolicyLab fork 维护，不是上游
XPolicyLab 原本自带；模型加载、预处理和 denoise 仍完整运行在 XPolicy 标准 server
内。ManiMux 只负责 wire codec、YAM FK/IK 和执行，不提供平行的 native model server。

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

官方公开的 post-training 数据格式没有声明控制 Hz。当前 `action_dt_s=0.033333`
（30 Hz）是 YAM 部署假设，不是已从 XR-1 checkpoint 证明的训练频率。未来 YAM
fine-tune bundle 必须记录原生采样 Hz，并用同一值替换该配置。

## 配置

```text
base server:    configs/xiaomi-xr1/yam/server/base.yaml
ManiMux:        configs/xiaomi-xr1/yam/infra/manimux.yaml
RTC:            configs/xiaomi-xr1/yam/infra/rtc.yaml
```

RTC 将 ManiMux `30 x 14` overlap condition 通过 FK 反编码到模型原生 `30 x 60` 空间，
再进入五步 flow denoise 的 PiGDM conditioning。这是 ManiMux/XPolicy 扩展，
不是小米官方 XR-1 部署功能；它通过明确 sampler hook 实现，不是运行时
monkeypatch，也没有额外动作幅度阈值。

完整 RTC 链路是：

```text
旧的 30 x 14 absolute joint tail + soft mask
  -> 按新 observation 的 FK 锚点重新编码为 30 x 60 EE delta
  -> 用 action mean/std 进入模型归一化空间
  -> XR-1 每一步 Euler denoise 计算 clean estimate 与 VJP guidance
  -> 反归一化 -> FK/IK -> 新的 30 x 14 joint chunk
```

因此它是 sampler-level RTC，不是 chunk splice。但目前只有 CPU 虚拟 flow 证明
guidance 确实在 `_generate` 内生效；还没有 5B 模型的真实 GPU conditioned
forward，所以 I7 仍然只能标记为离线完成。

`yam.json` 已经按官方格式由 `60` 个完整 YAM episode、`23,994` 个 30-step window
计算：action 是 `30 x 60` anchor-relative EE delta 的 mean/std，state 是 `1 x 60`
YAM joint/FK state 的 q01/q99。因此 **base 测试不需要再改数值**；改用官方 washer
demo stats 反而会把另一台机器的单位送给 YAM。

但这份 stats 只让 YAM state/action 映射在数值上有定义，不是官方 5B checkpoint 的
配对 post-training statistics。只有用同一份 YAM 数据 fine-tune 并导出权重后，二者才
真正匹配。重新采集数据或改变 action codec 时再运行：

```bash
cd /home/ubuntu/manimux
PYTHONPATH=src envs/yam/.venv/bin/python -m \
  manimux.integrations.xr1_yam.compute_norm_stats \
  --episodes /path/to/yam/episodes \
  --out src/manimux/integrations/xr1_yam/norm_stats/yam.json
```

## Base 权重能力测试

base 权重使用独立 server config，不覆盖未来的 YAM finetune；执行仍复用标准 ManiMux：

```bash
# offline contract check
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/xiaomi_xr1_yam_server.py \
  --config configs/xiaomi-xr1/yam/server/base.yaml \
  --check

# terminal 1: model server
envs/xr1/.venv/bin/python scripts/xiaomi_xr1_yam_server.py \
  --config configs/xiaomi-xr1/yam/server/base.yaml

# terminal 2: no-CAN GPU/WS/FK/IK probe
envs/yam/.venv/bin/python scripts/xpolicylab_yam_forward_probe.py \
  --config configs/xiaomi-xr1/yam/infra/manimux.yaml
```

probe 必须返回有限的 `native_shape: [30, 60]` 与 `canonical_shape: [30, 14]`。
通过后再启动相机，并由操作者运行：

```bash
envs/yam/.venv/bin/manimux run \
  --config configs/xiaomi-xr1/yam/infra/manimux.yaml
```

30 Hz 仍是 YAM 对照实验假设，不是官方 checkpoint 元数据。base 能否做任务是
policy 实验结果；shape、finite、WS 和 FK/IK 才是本轮 infra 验收项。

## RTC 对照

先用离线脚本验证 XPolicy 使用的 sampler guidance hook：

```bash
envs/xr1/.venv/bin/python scripts/check_xr1_rtc_sampler.py
```

它不构造 5B 模型、不访问 GPU。只有 base server 的 ManiMux forward 和默认 runtime
通过后，才使用同一个 server 做 RTC 对照：

```bash
envs/yam/.venv/bin/manimux run --config configs/xiaomi-xr1/yam/infra/manimux.yaml
envs/yam/.venv/bin/manimux run --config configs/xiaomi-xr1/yam/infra/rtc.yaml
```

不要同时运行 ManiMux 与 RTC。相机、Viewer、CAN 检查和停止顺序参考
[MolmoAct + YAM](molmoact-yam-runbook.md)。

## 当前边界

已离线验证配置、权重/processor/stats 路径、动作 codec、FK/IK condition round-trip、RTC
payload 和两条路径的 sampler guidance 分支。尚未启动 XPolicy 模型服务、GPU
conditioned forward、真实相机、CAN、
preflight 或真机。`model_states.pt` 是 post-training 起点，不是已经证明能在 YAM 上
完成任务的策略。
