# LingBot-VLA2 + YAM 运行手册

## 结论

LingBot-VLA2 已公开完整源码和推理入口，ManiMux 现在也有正式的
`XPolicyLab/policy/LingBot_VLA2/` adapter，不再复用旧的 Qwen2.5
`LingBot_VLA`。

但当前本地 `checkpoints/pretrained/lingbot-vla-v2-6b` 是 foundation
checkpoint，不是 YAM 后训练模型，仍然**不能直接上真机**。正式入口会在加载
GPU 和权重之前检查部署 bundle，缺文件时以状态码 `2` fail closed。

## 上游基线

- 官方源码：<https://github.com/Robbyant/lingbot-vla-v2>
- 本次审计 revision：`951475ae1b1d87553e7dc47c97b53a3d695c0d13`
- 官方 foundation weights：`robbyant/lingbot-vla-v2-6b`
- 官方 RoboTwin 后训练 weights：`robbyant/lingbot-vla-v2-6b-robotwin`
- 源码许可证：Apache-2.0

官方 V2 使用 55 维 canonical state/action：14 arm joint、14 EE pose、2
gripper、12 hand、4 waist、2 head、3 mobile base、4 reserved。真正送进模型的
有效维度由训练 config、robot config 和 norm stats 一起决定，不是把 YAM 的 14
维数据随便塞进 55 维零向量。

## ManiMux / XPolicy 分层

```text
YAM + cameras
      |
      v
ManiMux canonical observation
      |
      v
XPolicyLab WebSocket protocol
      |
      v
LingBot_VLA2 adapter
  - 3 cameras -> official camera feature names
  - left/right 6 joints -> arm.position[12]
  - left/right gripper -> effector.position[2]
      |
      v
official LingbotVLAv2Server
  - FeatureTransform
  - checkpoint-specific norm stats
  - official flow sampling
      |
      v
absolute arm.position[12] + effector.position[2]
      |
      v
XPolicy per-step joint dictionaries -> ManiMux chunk
```

上游源码负责模型、FeatureTransform、normalization 和 flow sampling；XPolicy
adapter 只负责协议与 YAM 字段映射；ManiMux 只负责任务生命周期、调度、执行和
独立安全边界。

## 当前文件

```text
server:  configs/lingbot-vla2/yam/server/xpolicy.yaml
infra:   configs/lingbot-vla2/yam/infra/manimux.yaml
adapter: XPolicyLab/policy/LingBot_VLA2/model.py
profile: XPolicyLab/policy/LingBot_VLA2/robot_configs/yam_dual_absolute.yaml
source:  XPolicyLab/policy/LingBot_VLA2/lingbot_vla_v2/  # pinned nested submodule
check:   scripts/lingbot_vla2_yam_server.py
audit:   scripts/lingbot_vla2_yam_audit.py
```

当前没有 RTC config。官方 V2 sampler 还没有接收 ManiMux 的 soft-mask
conditioning；普通 chunk continuation 不能命名为 RTC。

## 完整部署 bundle

要把 server check 变成 `ready`，必须同时提供：

1. 仓库已经固定的官方 `lingbot-vla-v2` source checkout；
2. 用 YAM 数据 post-train 后的 `hf_ckpt/`；
3. official loader 要求的 `lingbotvla_cli.yaml`；
4. 与这份 YAM 训练数据匹配的 `norm_stats.json`；
5. 与训练完全一致的 YAM robot config；
6. 训练使用的 native control frequency 和 action horizon。

官方 source 已作为 pinned nested submodule 放在
`XPolicyLab/policy/LingBot_VLA2/lingbot_vla_v2/`，revision 为
`951475ae1b1d87553e7dc47c97b53a3d695c0d13`。首次 clone 必须使用
`--recursive`；已有 checkout 使用 `git submodule update --init --recursive`。

官方 loader 固定从 `checkpoint_path.parent.parent.parent / "lingbotvla_cli.yaml"`
读取训练 config。因此 bundle 建议保持如下目录：

```text
checkpoints/pretrained/lingbot-vla-v2-yam/
├── lingbotvla_cli.yaml
├── norm_stats.json
└── runs/yam/hf_ckpt/
    ├── model.safetensors.index.json
    └── model-*.safetensors
```

配置中的 `checkpoint_path` 应指向上面的 `runs/yam/hf_ckpt/`。如果采用其他层级，必须先
修正官方 loader 或保持同样的三层父目录关系，不能只复制一个 `config.json`。

## 离线检查

以下命令只检查文件和 YAML/JSON 契约，不导入官方 torch 模型，不启动服务：

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/lingbot_vla2_yam_server.py
```

当前预期输出是 `status: blocked`、退出码 `2`。这不是脚本失败，而是准确说明
本机还没有 YAM post-training bundle。

审计现有 foundation checkpoint 的 55 维投影：

```bash
envs/yam/.venv/bin/python scripts/lingbot_vla2_yam_audit.py
```

## Bundle 就绪后的命令

先修改 `configs/lingbot-vla2/yam/server/xpolicy.yaml` 中四个路径以及真实
`action_horizon`，再运行离线检查。只有检查显示 `ready` 后，才允许按顺序启动：

```bash
# terminal 1: model server
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/lingbot_vla2_yam_server.py --serve

# terminal 2: cameras
# 使用现场已经验证过的 camera server 命令。

# terminal 3: ManiMux
envs/yam/.venv/bin/manimux run \
  --config configs/lingbot-vla2/yam/infra/manimux.yaml
```

这三条命令当前均未在 LingBot-VLA2 上执行；没有 GPU forward、server handshake、
相机、CAN 或真机证据。

## 后续 RTC

RTC 必须进入官方 `sample_actions` 的 10-step flow denoise loop，接收
`action_condition`、soft mask 和 `beta`，并在每个 denoise step 做 guided
inpainting。完成模型原生 hook、离线数值测试和真实 latency feasibility 之前，
不新增 `configs/lingbot-vla2/yam/infra/rtc.yaml`。
