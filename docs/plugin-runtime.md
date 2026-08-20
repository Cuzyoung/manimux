# ManiMux Policy / Robot 插件接口

ManiMux 的启动入口保持统一：

```bash
manimux run --config configs/<run>.yaml
```

配置选择一次运行所需的四个组件：

```text
SensorDriver -> PolicyModel + PolicyAdapter -> ActionTimeline -> RobotDriver
                                                        |
                                                        +-> Viewer
```

## 配置选择

`configs/mock.yaml` 是老师原有的本地异步基线；`configs/molmoact-yam-live.yaml`
描述 MolmoAct、YAM、三相机服务和 YAM Viewer 的组合。核心字段如下：

```yaml
robot:
  driver: yam_dual
  config: path/to/left.yaml
  group_dims: {left_arm: 7, right_arm: 7}
  options:
    right_config: path/to/right.yaml

sensors:
  - name: yam_cameras
    driver: camera_server
    options:
      endpoint: tcp://127.0.0.1:5555

policy:
  worker: molmoact_http
  adapter: molmoact_yam
  options:
    server: http://127.0.0.1:8202

viewer:
  enabled: true
  robot_adapter: yam
```

`options` 只保存插件自己的参数；Timeline、Executor、Safety 和 Recorder 仍由
ManiMux core 配置和管理。

## 真机轨迹时间与速度

`configs/molmoact-yam-live.yaml` 将时间和速度都放在配置中，修改时不需要改 Python：

```yaml
robot:
  options:
    start_duration_s: 5.0   # 上电后移动到起始姿态的秒数
    home_duration_s: 5.0    # 单臂回零的秒数

policy:
  horizon_steps: 30
  action_dt_s: 0.05          # 相邻 Policy 点间隔 0.05 秒

execution:
  executor: smooth
  smooth:
    max_velocity: 0.25       # rad/s
    max_acceleration: 0.5    # rad/s^2
```

当前 MolmoAct live 配置使用老师原 ManiMux 的 `policy.action_dt_s`。降低
`max_velocity` 和 `max_acceleration` 会收紧最终下发速度。当前 executor 是
`smooth`，因此真正生效的是
`execution.smooth`；切换到 `mpc` 时应同步配置 `execution.mpc`。

`start_duration_s` 和 `home_duration_s` 只控制 i2rt `move_joints` 的起始/回零过渡，
不控制 Policy 轨迹。YAM 当前使用 5 秒；时间过长会使每周期位置增量过小，在静摩擦和
重力补偿附近可能出现低速抖动。

暂停或等待下一个 action chunk 时，executor 会从最新实测位置重新同步，
与原版 MolmoAct 在 chunk 边界用实测关节状态衔接的逻辑一致。

MolmoAct 配置使用 `execution.inference_schedule: single_inflight`：前一次请求完成后，
当剩余轨迹低于 refill threshold 才提交下一次请求。老师原有配置没有该字段时仍采用
`deadline`，因此原 mock runtime 的调度行为不变。

## 三个稳定边界

### PolicyModel

PolicyModel 只运行模型：`reset -> infer -> close`。它不能下发机器人命令，也不
直接调用 Viewer。模型的原始返回值可以是 ndarray、字典或插件自定义的可序列化对象。

### PolicyAdapter

PolicyAdapter 负责模型与 embodiment 的语义转换：

- 将标准 `ObservationSnapshot` 转成模型输入；
- 将原始输出转成带时间、group 和 action space 的 `ActionChunk`；
- 验证相机名称、左右顺序、维度、单位和 finite values。

ManiMux V1 的执行器接收 `joint_position`。输出末端位姿的 policy 应在 adapter 中通过
该机械臂的 IK 转成 canonical joint groups，不能把厂商 SDK 命令塞进 runtime。

### RobotDriver

RobotDriver 只实现 `connect/get_state/send_command/home/stop/close`。通用
`RobotCommand` 相当于高层的 move；例如 `yam_dual` 将
`left_arm + right_arm` 映射到现有 i2rt `command_joint_state`。其他机械臂只需实现
自己的 driver 映射，不需要修改 policy worker、Viewer 或控制循环。

## 增加新插件

内置名称通过 ManiMux 的轻量 factory 解析。外部包可声明以下 Python entry point：

- `manimux.policies.models`
- `manimux.policies.adapters`
- `manimux.robots`
- `manimux.sensors`

本地开发也可以在配置中直接写 `package.module:factory`。这些只是构造函数发现机制，
不引入服务注册中心、manifest 或 Control Plane。

## 兼容与真机边界

`manimux-molmoact-yam` 兼容入口目前仍保留，四终端手册不变。
`configs/molmoact-yam-live.yaml` 已通过无硬件 HTTP worker、动作拆分、YAM command 映射和
Viewer committed-plan 测试。

`yam_dual` 连接即驱动硬件，没有第二种模式。`robot.options` 是自由字典，所以驱动会
校验其中每一个键（`right_config` / `start_duration_s` / `home_duration_s` /
`move_to_start_on_connect` / `home_on_close`），不认识的一律在构造阶段报错 —— 拼错一个
后缀不该变成「机械臂照动，只是不按配置动」。上真机前的验证靠 mock 配置和 adapter
单测，再做低速真机 smoke test。
