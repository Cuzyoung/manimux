# ManiMux 项目进度

更新时间：2026-08-20

## 项目目标

当前主线覆盖四个 XPolicy 基模：**OpenPI Pi05、GR00T N1.7、Xiaomi XR-1、
LingBot-VLA2**。最终要完成两个闭环：

```text
训练闭环
YAM 数据 -> 模型原生数据格式 -> fine-tune -> checkpoint + stats + training config
        -> 同一个 XPolicy server 直接加载

推理 infra 闭环（ManiMux 负责）
checkpoint -> 模型原生 sampler -> XPolicy WebSocket -> ManiMux ActionChunk
           -> Timeline / Smooth / Safety / Recorder / Viewer -> 双臂 YAM
```

ManiMux 的任务是把第二条链路全部打通，并给训练产物定义稳定的部署入口。训练算法、
数据质量和任务成功率属于 policy 研发，但训练导出的权重必须能够不改 runtime 代码就接入
同一个 server config。

MolmoAct2 和 ABC 是已经跑通的参考基线，不是本轮 XPolicy 模型集成的剩余工作。

## 本阶段责任

我负责把四个模型的 **I0-I6 推理 infra 全部打通**，并在模型原生支持时完成 I7 RTC；
同时负责 T3/T4 的部署 bundle 规范与回载接口。模型训练侧负责 T0-T3 的数据转换、stats、
fine-tune 和权重导出。两边最终在 T4 汇合：训练产物必须由现有 XPolicy server 直接加载，
不能为每个新 checkpoint 临时改 ManiMux core。

## 什么叫“打通”

只有以下 Gate 全部有证据，模型的 infra 才能标记为完成：

| Gate | 验收标准 |
|---|---|
| I0 源码 | 官方源码或固定版本的 fork/submodule 可递归获得 |
| I1 契约 | 相机、state/action 维度、joint/EE、absolute/delta、horizon、频率明确 |
| I2 产物 | checkpoint、stats、processor/training config 路径完整且互相匹配 |
| I3 模型 | 在模型独立环境完成真实 GPU load 和一次 forward |
| I4 协议 | XPolicy WebSocket 完成真实请求/响应，输出 shape 和数值有限 |
| I5 Runtime | 默认 ManiMux 无 monkeypatch 地接收、调度和记录 chunk |
| I6 真机 | 三相机 + 双臂 YAM 完成受控闭环并保存可分析记录 |
| I7 RTC | 仅当 conditioning 真正进入模型原生 sampler 时才验收；不是必选项 |

训练闭环单独使用以下 Gate，不能用“已有公开权重”代替：

| Gate | 验收标准 |
|---|---|
| T0 数据 | YAM episode 能转换为模型原生训练格式，字段和时间频率明确 |
| T1 统计 | 从同一训练集生成与 checkpoint 配套的 norm stats |
| T2 训练 | 最小训练 smoke 能保存 checkpoint，不要求先有高成功率 |
| T3 导出 | 权重、stats、processor/training config 组成完整部署 bundle |
| T4 回载 | XPolicy server 不改代码即可加载新 bundle 并完成 GPU forward |
| T5 评估 | 默认 ManiMux 保存 rollout，任务效果与 infra 指标分开报告 |

状态含义：✅ 已有对应证据；🟡 代码/离线契约完成但缺运行证据；⛔ 缺必要训练产物；
➖ 当前不做或不适用。

## 当前总表

| 模型 | I0-I2 源码/契约/产物 | I3 GPU | I4 WS | I5 ManiMux | I6 YAM | I7 RTC | T0-T4 训练回载 |
|---|---|---|---|---|---|---|---|
| OpenPI Pi05 YAM | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 有训练代码和 YAM checkpoint，尚未在本项目重跑训练导出 |
| GR00T N1.7 YAM | ✅ | ✅ | ✅ | ✅ | ✅ | ➖ | 🟡 有 YAM finetune 和匹配 stats，尚未重跑训练 |
| Xiaomi XR-1 YAM | 🟡 权重不是 YAM policy | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 Native/XPolicy sampler hook 离线通过 | ⛔ 缺 YAM fine-tune 权重和配套 stats |
| LingBot-VLA2 YAM | ⛔ 缺 YAM bundle | ⛔ | ⛔ | 🟡 config/adapter 已完成 | ⛔ | 🟡 sampler RTC 离线完成 | ⛔ 缺 YAM post-training、training YAML 和 stats |

## 分模型进度

### [OpenPI Pi05](pi05-yam-runbook.md)

- 已完成三相机、14D absolute joint、16-step、YAM stats、XPolicy server、默认 ManiMux
  和 Pi-guided RTC 真机链路。
- RTC 已有真实延迟与无 post-start chunk gap 记录。
- 当前动作犹豫属于 checkpoint / policy 质量问题，不是 infra 断路。
- 剩余训练任务：用项目自己的训练命令重跑一次最小 fine-tune，导出权重和 stats，再由现有
  `configs/pi05/yam/server/finetune.yaml` 回载。

### [GR00T N1.7](gr00t-yam-runbook.md)

- 本地权重是 `robocurve/gr00t-n1.7-yam-molmoact2` 的 YAM finetune，不是 base。
- 已确认三相机、14D absolute joint、16-step、30 Hz、4-step denoise 和 checkpoint 自带
  `statistics.json/new_embodiment`。
- XPolicy adapter、server config、ManiMux config、静态 checkpoint 检查和三相机合成输入
  forward probe 已经完成。
- 独立环境、CUDA、FlashAttention 和 gated Cosmos 已完成；真实 GPU forward 与 XPolicy
  WebSocket probe 输出有限的 `16 x 14` absolute joint chunk。
- 第一次冷启动请求为 `632.2 ms`，随后三次稳态为 `101.3 / 91.7 / 89.4 ms`，明显低于
  16 步在 30 Hz 下的 `533.3 ms` chunk 时长；稳态 latency gate 已通过。
- 默认 ManiMux、真实三相机、双臂 YAM 和 Recorder 闭环已经跑通。最近两次完整记录分别有
  91/74 个 chunk 被接受，无 plan rejection 或 invalid action；操作者正常结束 episode。
- 真机没有完成 pick 任务属于当前 Robocurve checkpoint 的 policy 质量结果，不影响 I5/I6
  infra 验收；该模型卡本身也没有闭环真机成功率。
- 当前未实现 GR00T RTC，I7 保持可选的 `➖`，不能用默认 chunk runtime 冒充。

### [Xiaomi XR-1](xiaomi-xr1-yam-runbook.md)

- 已完成官方源码核对、XPolicy adapter、`30 x 60` anchor-relative EE delta、YAM FK/IK、
  默认 ManiMux，以及 Native HTTP / XPolicy 两条 RTC sampler hook。
- RTC 条件链是 `30 x 14` joint -> 以新观测做 FK re-anchor -> `30 x 60` EE delta ->
  5-step Euler 内 PiGDM VJP guidance；条件进入模型原生 sampler，不是 chunk splice。
- 当前 `30 Hz` / `action_dt_s: 0.033333` 是部署假设；官方公开训练格式没有声明控制频率，
  后续 YAM fine-tune bundle 必须携带训练时的 native Hz，不能把当前值当官方参数。
- 当前官方 `Xiaomi-Robotics-1-5B` 是 post-training 起点，不是 YAM policy。
- `yam.json` 只让单位和 codec 有定义，不能替代与权重配套的训练 stats。
- 当前阻塞：真实 5B checkpoint 的 GPU conditioned RTC forward、XPolicy WS 往返，以及
  YAM fine-tune bundle。CPU 虚拟 flow 已证明两条 guidance 分支，但不等于真实模型验收。
- 在没有 YAM 权重前，可以验收 infra shape/latency，但不能把任务失败归为 ManiMux 问题。

### [LingBot-VLA2](lingbot-vla2-yam-runbook.md)

- 官方源码已经固定为 XPolicyLab 内的 nested submodule，正式 V2 adapter、YAM profile、
  server/infra config 和 fail-closed 检查已经完成。
- 已固定 v1 bundle schema、模板、官方 source commit、相对 artifact 路径、训练 Hz/horizon
  和 server 无改码回载接口。
- 当前本地 `lingbot-vla-v2-6b` 是 foundation checkpoint，不是 YAM post-training 权重。
- 缺少 YAM `hf_ckpt`、原始 `lingbotvla_cli.yaml`、匹配 `norm_stats.json`、训练 native Hz。
- 因此 server 当前返回 `blocked` 是正确行为，不允许用零填充 55D 或假 stats 强行启动。
- 官方 sampler 没有原生 RTC API，但公开了 prefix cache 和 `predict_velocity`；XPolicy 已完成
  每个 denoise step 的 VJP soft-mask guidance，不是 chunk splice。
- `get_action_rtc`、14D raw -> normalized/padded 55D、RTC infra config 和 CPU 离线测试已完成。
- RTC 仍缺真实 bundle forward、WS 和 steady latency；静态 delay 参数不能当真机证据。

## ManiMux Infra TODO

按以下顺序推进，不同时改 runtime 设计和模型参数：

1. **XR-1 默认链路**：完成 XPolicy GPU forward -> WS -> EE codec/FK/IK 记录；训练权重另行推进。
2. **LingBot 训练产物**：按现有 v1 bundle schema 完成最小训练导出与回载。
3. **LingBot 默认链路**：真实 GPU forward -> WS -> ManiMux；默认链路通过后测 RTC latency，
   再把现有 sampler RTC config 从 offline-ready 提升为 live-ready。
4. **统一训练交付**：每个模型 runbook 都补齐 T0-T4 的可执行命令和产物检查。
5. **统一验收记录**：每个模型保存同样的 latency、chunk gap、action shape、数值有限性、
   Recorder 输出和真机结果，policy 成功率单独记录。

## 完成定义

本轮任务完成时，四个模型都必须满足：

- 换模型只换 `configs/<model>/yam/{server,infra}/`，不改 ManiMux core；
- 训练产物按 runbook 放入约定目录后，server check 能验证并直接加载；
- 默认 ManiMux 链路完成 GPU、WS、三相机、双臂 YAM 和 Recorder 闭环；
- RTC 未实现时明确写 `not supported`，绝不把普通异步 chunk 冒充 RTC；
- README、总进度和单模型 runbook 的状态一致，离线完成、GPU 完成和真机完成不混写。
