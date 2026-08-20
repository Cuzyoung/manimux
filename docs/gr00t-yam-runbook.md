# GR00T N1.7 + YAM 运行手册

本文只覆盖 `robocurve/gr00t-n1.7-yam-molmoact2`。它是基于 NVIDIA
[Isaac-GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T) 的 YAM 微调权重，不是
`nvidia/GR00T-N1.7-3B` base 直接零样本上 YAM。

当前状态：**XPolicy adapter、checkpoint 契约、ManiMux 配置已完成离线验证；模型环境、
Cosmos 权限、GPU forward、相机和真机尚未验证。**

## 契约

```text
3 RGB cameras + 14D current joint state + instruction
  -> XPolicyLab GR00T_N17 adapter
  -> GR00T N1.7 YAM checkpoint + checkpoint statistics
  -> 16 x 14 absolute joint positions at 30 Hz
  -> ManiMux ActionTimeline -> SmoothExecutor -> YAM
```

| 项目 | 值 |
|---|---|
| 官方模型代码 | `XPolicyLab/policy/GR00T_N17/gr00t_n17/`（vendored Isaac-GR00T N1.7） |
| XPolicy adapter | `XPolicyLab/policy/GR00T_N17/model.py` |
| 权重 | `/home/ubuntu/manimux/checkpoints/pretrained/gr00t-n1.7-yam` |
| normalization | checkpoint 自带 `statistics.json` 的 `new_embodiment` |
| cameras | `base_view`、`left_wrist_view`、`right_wrist_view` |
| state/action | `left_arm[6] + left_gripper[1] + right_arm[6] + right_gripper[1]` |
| action | 16 步、14D、absolute joint position |
| 时间语义 | 训练数据 30 Hz，`action_dt_s = 1/30` |
| denoise | checkpoint `num_inference_timesteps = 4` |

权重目录中的 `processor_config.json` 和 `statistics.json` 是模型契约的一部分。不能把
Pi05 的 stats、XPolicy 原 ARX modality config 或 base 模型默认 embodiment 替换进来。

## 1. 离线检查

这一步不导入 torch、不加载权重到 GPU：

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/gr00t_yam_server.py \
  --config configs/groot/yam/server/finetune.yaml \
  --check
```

检查会确认：两片权重存在、3 相机键一致、state/action 都是 14D、action 是 absolute、
horizon 是 16、30 Hz 声明一致，并且 YAM 的 q01/q99 stats 维度完整。

## 2. 安装模型环境

当前本机尚未创建 `gr00t_n17/.venv`。首次安装由操作者执行：

```bash
cd /home/ubuntu/manimux/XPolicyLab/policy/GR00T_N17
bash install.sh
```

GR00T N1.7 的 processor 还需要 gated `nvidia/Cosmos-Reason2-2B`。先完成 Hugging Face
授权；若使用本地模型，把 `configs/groot/yam/server/finetune.yaml` 中
`cosmos_model_path` 改为本地目录。

## 3. 启动模型服务

```bash
cd /home/ubuntu/manimux
XPolicyLab/policy/GR00T_N17/gr00t_n17/.venv/bin/python \
  scripts/gr00t_yam_server.py \
  --config configs/groot/yam/server/finetune.yaml
```

第一次必须先用一组离线观测完成真实 GPU forward，确认模型加载、显存、三路图像、14D
state、16 x 14 输出和延迟；仅看到端口监听不算模型可用。

## 4. 相机与真机

以下命令会分别打开相机和连接真机，只能由操作者在完成 GPU forward 与 preflight 后执行：

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/manimux-camera-server --config configs/cameras.yaml
```

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/manimux run --config configs/groot/yam/infra/manimux.yaml
```

当前只提供默认 ManiMux 配置，不提供 GR00T RTC 配置。虽然 Isaac-GR00T N1.7 模型内部有
自己的 overlap/frozen-step RTC 分支，但当前 XPolicy `GR00T_N17` adapter 没有实现统一的
`get_action_rtc()` 契约，不能把默认推理包装成 RTC。

## 未验证项

- `gr00t_n17/.venv` 安装和依赖导入；
- gated Cosmos 模型访问或本地缓存；
- checkpoint 的真实 GPU load/forward 和推理延迟；
- XPolicy WebSocket 服务与 ManiMux 的真实往返；
- 相机、preflight、CAN 和真机闭环；
- 任务成功率与动作平滑性。
