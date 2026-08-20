# Pi05 + YAM 运行手册

本文只覆盖 Pi05。默认实验使用 YAM 微调 checkpoint 和普通 ManiMux runtime；同一微调
checkpoint 的 RTC 和 Pi05 base + RTC 都是独立实验，不与默认配置混用。

## 模型契约

YAM 微调权重位于：

```text
checkpoints/pretrained/pi05-yam/
  params/
  assets/yam-bimanual-merged/norm_stats.json
```

- 输入：base、left wrist、right wrist 三路独立 RGB，不拼成一张图；
- state：14 维，左 `6+1` 后右 `6+1`；
- 输出：`16 x 14` absolute joint positions，30Hz；
- flow sampling：OpenPI 默认 10 steps；
- stats：checkpoint 自带 quantile norm stats；
- server：`configs/pi05/yam/server/finetune.yaml`；
- ManiMux 30Hz 默认配置：`configs/pi05/yam/infra/manimux.yaml`；
- Pi-guided RTC 对照：`configs/pi05/yam/infra/rtc.yaml`；
- 50ms 拉伸时序对照：`configs/pi05/yam/infra/stretched-50ms.yaml`。

## 当前验证状态

已完成一次不连接硬件的真实 checkpoint GPU forward，输出为 finite 的 `16 x 14` absolute
joint actions。2026-08-20 又完成了一次真实 YAM rollout：操作者观察到策略明确朝 pick
任务执行，但旧实验配置的 `0.20 rad/s`、`0.40 rad/s²` 低速包络造成慢速追赶和抖动，
因此这次不能记作成功率。

该 episode 位于：

```text
data/run-20260820T061701Z-8cfd2e00/episode-6bcee9a699eb/
```

记录中 143 次推理提交、141 个 plan 接受，常见 stale-prefix 只有 2–3 步，说明 16-step
horizon 没有被推理延迟耗尽。当前 ManiMux infra 配置使用与 MolmoAct 相同的 `smooth` executor
参数：8Hz cutoff、`0.25 rad/s`、`0.5 rad/s²`。Pi05 只保留模型契约要求的 30Hz
和 16-step horizon。这些是执行器
跟踪上限，不是模型动作好坏的阈值判定。

## 1. 离线检查

只检查路径和契约，不加载模型：

```bash
cd /home/ubuntu/manimux
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  scripts/pi05_yam_server.py --check \
  --config configs/pi05/yam/server/finetune.yaml
```

只做合成三相机 observation 的 GPU forward，不启动服务或机器人：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  scripts/pi05_yam_offline_infer.py \
  --config configs/pi05/yam/server/finetune.yaml
```

## 2. 模型服务

```bash
cd /home/ubuntu/manimux
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  scripts/pi05_yam_server.py \
  --config configs/pi05/yam/server/finetune.yaml
```

终端保持前台没有持续日志是正常的。确认 ready：

```bash
ss -ltnp | rg ':8500'
nvidia-smi
```

必须看到 `127.0.0.1:8500` 处于 `LISTEN`。

## 3. 相机与 Viewer

相机服务：

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/manimux-camera-server --config configs/cameras.yaml
```

已有 `5555` 服务时不要重复启动。Viewer 可选：

```bash
envs/yam/.venv/bin/manimux-viewer --robot yam --host 0.0.0.0 --port 8086
```

## 4. Preflight

读取真实三相机和当前关节并推理，但显式禁止 start-position 和退出回 Home，也不发送
模型动作：

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/pi05_base_yam_preflight.py \
  --config configs/pi05/yam/infra/manimux.yaml
```

保存它打印的 measured state、first action、shape、gripper range 和 steady latency。脚本只
报告契约，不给主观 `safe_for_live` 结论。

## 5. 真机 runtime

清空工作区、急停在手，并确认两路 CAN 都是 `ERROR-ACTIVE`：

```bash
for c in can_left can_right; do
  ip -details link show "$c" | grep -o 'ERROR-ACTIVE\|ERROR-PASSIVE\|BUS-OFF'
done
```

默认 ManiMux 实验保留 Pi05-YAM checkpoint 的 30Hz 时序：

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/manimux run --config configs/pi05/yam/infra/manimux.yaml
```

这个 ManiMux 基线将 16 个绝对关节点按约 0.50 秒执行。机器人控制环也是 30Hz，避免
改变 checkpoint 学到的时间语义；异步调度、Timeline、SmoothExecutor 和记录仍由
ManiMux 负责。

50ms 拉伸对照把同一组点按约 0.75 秒执行：

```bash
envs/yam/.venv/bin/manimux run --config configs/pi05/yam/infra/stretched-50ms.yaml
```

两份配置是独立对照实验，不要同时运行。2026-08-20 的 50ms 真实运行中，模型请求和
plan 接受都正常，但 replanning 明显变慢，操作者观察到“向前抖一下又回来”；因此 50ms
只保留为失败对照，不再作为默认真机配置。

连接后先用 3.5 秒移动到 YAML start pose，这不是模型动作。当前执行器比第一次真实 rollout
快很多；第一次使用新参数只做短 rollout，并始终准备急停。

## Chunk 边界诊断

普通 runtime 会在每个新 plan 接受时写一条 `plan_boundary` 到 episode 的
`events.jsonl`。它同时保存模型原始 chunk 首点、ManiMux 拼接后的首点、上一条命令和实测
关节；只增加诊断记录，不改变动作、速度或拼接算法。即使按 `Ctrl-C`，记录也会保留在
`.partial` episode 中。

完成一次 Pi05 和一次 MolmoAct2 短 rollout 后，只需告知“跑完了”；分析端会自动选择两种
policy 最新的有效 episode。也可以手动离线查看：

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/analyze_chunk_boundaries.py
```

## Pi05 YAM finetune + RTC

模型服务仍使用同一个 finetune 配置，不需要启动第二份权重：

```bash
cd /home/ubuntu/manimux
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  scripts/pi05_yam_server.py \
  --config configs/pi05/yam/server/finetune.yaml
```

RTC 使用独立 infra 配置：

```bash
envs/yam/.venv/bin/manimux run --config configs/pi05/yam/infra/rtc.yaml
```

它保留普通 ManiMux 基线的 30Hz、16-step checkpoint 契约和 OpenPI 默认 10-step flow
sampling；当前低速 RTC 对照使用
`0.20 rad/s`、`0.40 rad/s²`，比 MolmoAct2 的执行包络再慢一档。根据真实 rollout 的约
2–3 步延迟设置的初始先验为 `initial_delay_steps: 3`；运行时会用 10-step sampling 的真实
延迟自动更新 `d`。`min_execute_steps: 8`，即 `H=16、s=8`，RTC guidance clip 随
10-step flow 更新为 `beta: 9.1`。只有 RTC 调度和 Pi-guided denoise condition 发生变化，
可以直接与 `infra/manimux.yaml` 做 A/B。

## 停止

在 runtime 前台按一次 `Ctrl-C`，等待 3.5 秒回 Home 并退出；随后再停止相机和模型服务。
不要使用模糊 `pkill`。

## Pi05 base + RTC

base zero-shot 是单独实验：

```text
server: configs/pi05/yam/server/base.yaml
infra:  configs/pi05/yam/infra/base-rtc.yaml
```

它使用 50-step Pi-guided RTC，不代表 YAM 微调版的默认配置。不要用 base config 覆盖本文
的 16-step checkpoint 实验。
