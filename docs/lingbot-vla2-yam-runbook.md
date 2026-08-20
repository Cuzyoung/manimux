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
schema:  XPolicyLab/policy/LingBot_VLA2/bundle.schema.json
example: configs/lingbot-vla2/yam/bundle.example.yaml
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

这六项由一个 `bundle.yaml` 统一声明。server config 只保存
`bundle_manifest_path`，adapter 不再分别接收 checkpoint、stats、robot config
和 horizon 路径。训练导出只要符合 bundle schema，现有 server 无需改代码即可
回载。

规范文件：

- JSON Schema：`XPolicyLab/policy/LingBot_VLA2/bundle.schema.json`；
- YAML 模板：`configs/lingbot-vla2/yam/bundle.example.yaml`；
- schema version：`manimux.lingbot_vla2_yam_bundle.v1`。

Manifest 的职责固定如下：

| 区域 | 必须声明 | 检查内容 |
| --- | --- | --- |
| `model` | family、官方 source commit | checkout 必须正好位于该 commit |
| `artifacts` | training YAML、checkpoint、stats、robot config | 只能使用 bundle 内相对路径 |
| `control` | native Hz、action horizon、absolute joint | 与训练 chunk 和 ManiMux infra 一致 |
| `embodiment` | YAM 双臂维度、gripper、三相机顺序 | 固定为 6+1 / 6+1 与官方 feature names |

官方 source 已作为 pinned nested submodule 放在
`XPolicyLab/policy/LingBot_VLA2/lingbot_vla_v2/`，revision 为
`951475ae1b1d87553e7dc47c97b53a3d695c0d13`。首次 clone 必须使用
`--recursive`；已有 checkout 使用 `git submodule update --init --recursive`。

官方 loader 固定从 `checkpoint_path.parent.parent.parent / "lingbotvla_cli.yaml"`
读取训练 config。因此 bundle 建议保持如下目录：

```text
checkpoints/pretrained/lingbot-vla-v2-yam/
├── bundle.yaml
├── lingbotvla_cli.yaml
├── norm_stats.json
├── robot_config.yaml
└── runs/yam/hf_ckpt/
    ├── model.safetensors.index.json
    └── model-*.safetensors
```

`bundle.yaml` 中的 checkpoint 固定写成 `runs/yam/hf_ckpt`。这样官方 loader
向上三级后正好得到 bundle root，并读取同目录的 `lingbotvla_cli.yaml`。不接受
绝对 artifact 路径、`..` 逃逸路径或只复制 `config.json` 的不完整 checkpoint。

训练导出完成后，以模板生成 manifest，并把训练时实际使用的 robot config 一起
放入 bundle：

```bash
cp configs/lingbot-vla2/yam/bundle.example.yaml \
  checkpoints/pretrained/lingbot-vla-v2-yam/bundle.yaml
cp XPolicyLab/policy/LingBot_VLA2/robot_configs/yam_dual_absolute.yaml \
  checkpoints/pretrained/lingbot-vla-v2-yam/robot_config.yaml
```

模板中的 `native_hz` 和 `action_horizon` 必须改成数据采集/训练的真实值。
`lingbotvla_cli.yaml` 必须保留 `model.post_training: true`、
`model.config_key: LingbotVLAV2Config`、55 维 canonical 配置、三相机顺序，以及
与 manifest horizon 完全相同的 `train.chunk_size`。

## 离线检查

以下命令只检查文件和 YAML/JSON 契约，不导入官方 torch 模型，不启动服务：

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/lingbot_vla2_yam_server.py
```

当前预期输出是 `status: blocked`、退出码 `2`。这不是脚本失败，而是准确说明
本机还没有 YAM post-training bundle。

检查器同时读取 `configs/lingbot-vla2/yam/infra/manimux.yaml`，要求：

- `robot.control_hz == bundle.control.native_hz`；
- `policy.action_dt_s == 1 / native_hz`；
- `policy.horizon_steps == bundle.control.action_horizon`；
- baseline `execution.runtime == manimux`。

所以训练产物与执行时序不一致时会在模型加载前失败，而不是在真机循环中静默
拉伸动作。

审计现有 foundation checkpoint 的 55 维投影：

```bash
envs/yam/.venv/bin/python scripts/lingbot_vla2_yam_audit.py
```

## Bundle 就绪后的命令

把完整 bundle 放到 server config 已声明的目录，并让 ManiMux infra 的 Hz、dt、
horizon 与 manifest 一致，再运行离线检查。无需修改 adapter 或 server 代码。
只有检查显示 `ready` 后，才允许按顺序启动：

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
