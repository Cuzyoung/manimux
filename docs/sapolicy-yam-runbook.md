# SAPolicy + YAM 接入手册

## 当前结论

SAPolicy 已按 ManiMux 的模块边界接入为两个薄插件：

- `sapolicy_tcp`：只连接独立运行的私有 SAPolicy 服务，不导入模型源码，也不接触机器人；
- `sapolicy_yam`：负责相机名和内参、YAM FK/IK、夹爪单位以及笛卡尔动作到关节动作的转换。

adapter 最终只交付标准 `joint_position ActionChunk`。之后仍走同一套 ManiMux
Timeline、Default 调度、executor、Safety、Recorder、Viewer 和 YAM driver，没有复制控制循环。

当前证据边界是 **本地实现 + 离线契约验证通过**，覆盖 plugin registry、wire codec、
observation 转换、双臂绝对 EE 到 joint chunk、未收敛 IK、笛卡尔跳变和 depth contract
拒绝。尚未验证真实 checkpoint
加载、真实相机 observation、GPU forward、YAM IK 数值或真机执行，因此不能标成“已跑通”或
“真机可用”。

## 数据流

```text
YAM joints + named RGB frames
  -> sapolicy_yam: FK + calibrated K + checkpoint gripper units
  -> sapolicy_tcp: private SAPolicy service atomic infer(observation)
  -> sapolicy_yam: absolute EE waypoints + gripper -> fail-closed IK
  -> canonical left_arm/right_arm joint_position ActionChunk
  -> unchanged ManiMux Timeline / executor / Safety / Recorder / YAM driver
```

SAPolicy 服务保留在它自己的私有仓库和 Python 环境中。ManiMux 中没有 vendoring、submodule
或模型依赖；wire client 只使用 Python 标准库和 NumPy。

## 当前支持的精确契约

| 项目 | 当前约束 |
|---|---|
| embodiment | 双臂 YAM，每侧 6 arm joints + 1 gripper |
| observation | 一路或多路命名 RGB；每路必须显式提供经过标定的 `3x3 K` |
| policy state | 每臂 grasp-site `pos3 + rot6d + gripper1`，由私有服务构造 |
| wire action | 每步 `left pose7 + grip1 + right pose7 + grip1`，四元数为 `wxyz` |
| ManiMux action | 两组绝对关节位置，每组 7 维 |
| runtime | 当前只声明 `default`；RTC/AAC/PAINT 等会在启动 capability 检查时拒绝 |
| depth | 暂不支持；`requires_depth: true` 会 fail fast |
| IK failure | 整个 chunk 拒绝，不执行部分解或未收敛的末态 |

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
6. `max_position_delta_m`、`max_rotation_delta_rad` 应按任务和 checkpoint 收紧，而不是为了
   让某个输出通过而放宽。

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
