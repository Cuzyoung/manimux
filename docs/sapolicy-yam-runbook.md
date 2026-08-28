# SAPolicy + YAM 接入手册

## 当前结论

SAPolicy 已按 ManiMux 的模块边界接入为两个薄插件：

- `sapolicy_tcp`：只连接独立运行的私有 SAPolicy 服务，不导入模型源码，也不接触机器人；
- `sapolicy_yam`：负责相机名和内参、YAM FK/IK、夹爪单位以及笛卡尔动作到关节动作的转换。

adapter 最终只交付标准 `joint_position ActionChunk`。之后仍走同一套 ManiMux
Timeline、executor、Safety、Recorder、Viewer 和 YAM driver，没有复制控制循环。

Default 路径已经完成真实 checkpoint/EMA/normalizer、真实相机、GPU、IK 和 YAM 真机
rollout；基础设施完整跑完不等于任务成功，首次 60 秒 bottles rollout 的盒子仍为空。
默认 ManiMux 路径现在会在 IK 前按 action 时钟丢弃已过期的源步，以最新实测关节为 seed，
并只提交左右臂共同可解且远离关节限位的前缀。PAINT 路径已完成公式、协议、动作变换、
真实权重 GPU forward 和 mock runtime 契约；最新离线探针满足 `H=16` 的延迟上界，但仍未
进入真机，首轮验证继续使用默认异步 chunk 路径。

## 数据流

```text
YAM joints + named RGB frames
  -> sapolicy_yam: FK + calibrated K + checkpoint gripper units
  -> sapolicy_tcp: private SAPolicy service atomic infer(observation)
  -> sapolicy_yam: trim expired source rows -> measured-state-seeded fail-closed IK
  -> both-arm common safe prefix + joint-limit margin
  -> canonical left_arm/right_arm joint_position ActionChunk
  -> unchanged ManiMux Timeline / executor / Safety / Recorder / YAM driver
```

SAPolicy 服务保留在它自己的私有仓库和 Python 环境中。ManiMux 中没有 vendoring、submodule
或模型依赖；wire client 只使用 Python 标准库和 NumPy。

## PAINT 接入

PAINT 复用 [`runtime/paint.py`](../src/manimux/runtime/paint.py) 的异步 `s/d` 调度，不修改
robot、executor、Safety 或 recorder。由于 SAPolicy 的 DiT 生成归一化 20D 相对笛卡尔动作，
而 ManiMux timeline 保存 14D YAM 关节动作，条件前缀必须走完整可逆契约：

```text
旧 14D joint prefix
  -> YAM FK + task-frame transform
  -> 16D absolute EE wire prefix
  -> 相对当前 observation 的 20D action
  -> checkpoint action normalizer
  -> DiT PAINT: naive forward + backward Euler + repainted forward (3N)
  -> 原有 unnormalize / absolute EE / IK / ActionChunk
```

普通请求仍调用原来的单次 forward sampler。服务只有在 action head 真正实现
`sample_trajectory_paint` 时才在 `backend_info.sampling_modes` 广告 `paint`；旧服务和错误
checkpoint 会在机器人连接前被 capability 检查拒绝。

当前 3cam TCP bottles checkpoint 是 `H=16, N=20, 30 Hz`。2026-08-27 使用历史真机
RGB/关节输入复测：普通 warmed forward 为 `99.6--102.4 ms`，PAINT 在
`d=4/6/8` 时分别为 `182.6/179.8/184.3 ms`，约 5.4 个 action step。PAINT 要求
`d <= s <= H-d`，`H=16` 最多容许 `d=8`，所以当前延迟在离线时序上可行；但 PAINT 的
单次 forward 更慢，它解决 chunk 衔接而不是提高模型吞吐。目前只提供
[`paint-offline-probe.yaml`](../configs/sapolicy/yam/infra/paint-offline-probe.yaml)，其中 robot 和
camera 都是 mock。不要把它改成真机 driver；启用前仍需完成连续离线调度回放、独立命令
安全门验证和只读 preflight。

## 当前支持的精确契约

| 项目 | 当前约束 |
|---|---|
| embodiment | 双臂 YAM，每侧 6 arm joints + 1 gripper |
| observation | 一路或多路命名 RGB；每路必须显式提供经过标定的 `3x3 K` |
| policy state | 每臂 grasp-site `pos3 + rot6d + gripper1`，由私有服务构造 |
| wire action | 每步 `left pose7 + grip1 + right pose7 + grip1`，四元数为 `wxyz` |
| ManiMux action | 两组绝对关节位置，每组 7 维 |
| runtime | 真机基线为 `default`；PAINT 仅离线验证，RTC/AAC 仍 fail closed |
| depth | 暂不支持；`requires_depth: true` 会 fail fast |
| IK failure | 先裁过期步；后段失败时只保留双臂共同前缀，少于配置最小步数则整块拒绝 |

私有服务当前复用了 RoboTwin-compatible `endpose` wire convention：grasp site 沿自身局部
`+x` 前方 0.12 m。adapter 在输入和输出两侧对称处理该偏移。它不是 YAM tool-frame 的
经验修正，不能再叠加一次。

## 配置中不能猜的参数

不要直接把 [`offline-contract.yaml`](../configs/sapolicy/yam/infra/offline-contract.yaml)
改成真机 driver 就运行。至少需要从实际 checkpoint、数据和相机标定中确认：

1. `horizon_steps` 和 `action_dt_s`；SAPolicy 服务端的 `n_action_steps` 必须与这里完全一致。
2. `camera_map`、图像方向/色彩顺序，以及每路真实分辨率对应的 `camera_intrinsics`。
3. checkpoint 是否为 RGB-only；需要 metric depth 的 low-TCP 路线目前没有 ManiMux typed depth contract。
4. `gripper_transforms` 的方向、范围和开合含义。示例中的 `[-1,1] <-> [0,1]` 只是显式模板。
5. SAPolicy 训练 embodiment/tool frame 与 YAM `grasp_site` 是否一致；跨 embodiment 训练或
   仿真成功不自动证明该外参和动作尺度适用于 YAM。
6. IK 位置/姿态收敛精度、迭代次数和关节/执行器限制必须单独验证；删除笛卡尔跳变门限并不
   等于 IK 或硬件安全限制也被删除。

## 私有 SAPolicy 服务

使用 SAPolicy 私有仓库已有的 standalone RoboTwin-compatible server，在 SAPolicy 自己的
环境中加载 config/checkpoint，并保证：

- 监听地址默认只绑定 `127.0.0.1`；
- `n_action_steps == policy.horizon_steps`；
- 暴露不含私有路径的 `backend_info`，让 ManiMux 在启动时校验
  `sapolicy_manimux_v1`、horizon 和 16 维 wire action；
- 使用与训练/eval 相同的 normalizer 和 EMA 选择；
- 真正输出双臂 `(H, 16)` absolute-EE wire action；
- 不把 checkpoint、配置或私有源码复制进 ManiMux。

具体 checkpoint 路径和 SAPolicy 启动命令应只保存在私有仓库的模型 runbook 中，不在 ManiMux
公共文档里记录。

## 分层验证顺序

按以下证据逐层推进，前一层未通过时不启动后一层：

1. **纯离线**：审计 observation、0.12 m wire 偏移、夹爪 affine、动作 shape、跳变阈值和
   IK fail-closed；公开仓库不携带私有 SAPolicy 模型测试资产。
2. **真实模型、无机器人**：启动私有服务，用 mock ManiMux config 完成
   `backend_info -> reset -> RGB/state -> (H,16) -> (H,14)`，记录模型加载、输入 shape、
   forward 延迟和所有 IK 收敛。
3. **只读真机 preflight**：读取相机与关节反馈，检查 K/分辨率、FK pose、夹爪映射、第一条
   action 的 EE/joint delta；不发送命令。
4. **受限执行**：人工确认后才允许 ManiMux 连接 YAM，使用收紧的笛卡尔阈值、joint
   velocity/acceleration limits、短 `max_steps` 和可达的安全起始姿态。

“服务返回有限 `(H,16)`”只证明 transport/forward；“IK 得到有限 `(H,14)`”只证明离线动作
契约。两者都不等于真机任务成功。
