# Configuration layout

模型运行配置按 `configs/<model>/<embodiment>/{server,infra}/` 组织：

```text
configs/
  <model>/
    <embodiment>/
      server/
        <checkpoint-or-backend>.yaml
      infra/
        <experiment>.yaml
```

`server/` 只描述模型服务、checkpoint、norm stats 和模型原生采样参数；`infra/` 描述
ManiMux 的机器人、传感器、policy wire、执行器、Viewer 和记录。文件只创建实际存在的
角色，不再使用 `live` 后缀。名称直接表达实验差异，例如 Pi05 的 `base.yaml` /
`finetune.yaml`，以及统一的 `manimux.yaml` / `rtc.yaml` runtime。

命名轴不能混用：`server/{base,finetune}.yaml` 表示 checkpoint 变体，
`infra/{manimux,rtc}.yaml` 表示推理 runtime。base 权重加本体 projection stats 仍叫
`server/base.yaml`，不能在 infra 层新增 `zero-shot.yaml`。XR-1 只使用 XPolicy server；
不再提供平行的 native runtime 配置。

公共配置不属于任何模型，继续放在顶层：

- `cameras.yaml`：共享相机服务；
- `robots/`：机器人 driver 配置；
- `mock.yaml`：全 mock runtime；
- `maniunicon_meshcat.example.yaml`：通用仿真示例。

新增本体时不要复制模型目录，例如 Pi05 的 ALOHA 配置应放到
`configs/pi05/aloha/`，而不是创建新的 `pi05-aloha` 模型目录。

## 第一份配置：`configs/mock.yaml`

ManiMux 配置使用严格 schema：拼错字段或写入未知字段会直接报错，不会静默忽略。
Beginner 先从全 mock 配置理解七个顶层区域。

### `run`：这次实验是什么

| 字段 | 含义 |
|---|---|
| `task` | 发送给 policy 的任务指令，同时写入 Recorder 和 Viewer。fake policy 不读取语义，但真实 VLA 会读取。 |
| `output_dir` | 每次运行创建 `run-<timestamp>-<id>/` 的根目录，resolved config 和 episode 都写在这里。 |
| `max_steps` | 最多执行多少个 control tick；mock 中 `120 / 100 Hz ≈ 1.2 s`。 |

### `robot`：控制哪具身体

| 字段 | 含义 |
|---|---|
| `driver` | RobotDriver 插件名；`mock_dual_arm` 只在内存里模拟状态和命令。 |
| `control_hz` | ManiMux 主控制环频率；`100 Hz` 表示每 `10 ms` 一个 tick。它不是模型推理频率。 |
| `group_dims` | canonical action 分组和每组维度。名称、顺序和维度必须与 adapter、executor、robot 一致。 |

### `sensors`：policy 能看到什么

`sensors` 是列表，可以声明多路命名传感器。

| 字段 | 含义 |
|---|---|
| `name` | observation 中的传感器键；adapter 按名称查找相机。 |
| `driver` | SensorDriver 插件名；`mock_camera` 生成确定性的 RGB 图。 |
| `width` / `height` | 该传感器输出图像的宽和高。mock 输出 shape 为 `(height, width, 3)`。 |
| `fps` | 传感器声明的目标帧率。当前 `mock_camera` 不自行限频，而是在每个 control tick 被读取；真实 camera-server driver 才按服务数据更新。 |

### `policy`：谁来想、一次想多远

| 字段 | 含义 |
|---|---|
| `worker` | PolicyModel 插件名。模型在独立 spawn 进程中运行；`fake` 生成确定性的双臂波形。 |
| `adapter` | PolicyAdapter 插件名，负责 observation 编码和 action 解码；`identity` 直接接收 fake model 产生的 canonical chunk。 |
| `device` | 模型设备提示；`fake` 不使用它，真实模型插件可以使用。 |
| `action_dt_s` | action chunk 相邻两个点的时间间隔。`0.05 s` 等于 policy 轨迹点 `20 Hz`，不等于 robot 的 `100 Hz` 控制环。 |
| `timeout_s` | 一次推理请求的 deadline；响应晚于 deadline 会被丢弃。 |
| `horizon_steps` | 每个 action chunk 的点数。`20` 点、间隔 `0.05 s`，首末点覆盖 `(20-1)×0.05=0.95 s`。 |
| `inference_delay_s` | fake model 专用的模拟推理耗时；调大它可以观察异步推理和过期响应。 |

### `execution`：什么时候推理、怎样执行

| 字段 | 含义 |
|---|---|
| `executor` | `smooth` 或 `mpc`。mock 默认只使用下面的 `smooth` 参数；`mpc` 块此时不参与执行。 |
| `refill_threshold_s` | 当前 Timeline 剩余时间低于该值时提交下一次推理请求。 |
| `commit_lead_s` | 新 chunk 被接受后，从“当前时刻 + lead”开始生效，给原子切换留出极小调度余量。 |
| `max_plan_age_s` | 从 observation 时间算起，chunk 超过该年龄就以 `plan_too_old` 拒绝。 |
| `underrun_hold_s` | schema 中保留的 underrun 参数；当前 EdgeRuntime 没有读取它，Timeline 无可用动作时会立即保持 measured pose。 |
| `blend_steps` | 新 chunk 裁掉过期前缀后，前多少步从当前 measured command 线性过渡到模型轨迹。`0` 表示不融合。 |

`execution.smooth` 仅在 `executor: smooth` 时生效：

| 字段 | 含义 |
|---|---|
| `cutoff_hz` | 一阶低通截止频率；越低越平滑但跟随越慢。 |
| `max_velocity` | 每个 canonical 标量每秒允许的最大变化量。对关节位置通常理解为 `rad/s`。 |
| `max_acceleration` | 每个 canonical 标量每秒速度允许的最大变化量，关节通常为 `rad/s²`。 |
| `position_limit_abs` | 通用绝对位置包络 `[-limit, +limit]`；SafetyGuard 和 executor 都会使用。它不是机器人逐关节精确限位表。 |

`execution.mpc` 仅在 `executor: mpc` 时生效：

| 字段 | 含义 |
|---|---|
| `horizon_steps` | MPC 每个 tick 向前优化多少个参考点。 |
| `dynamics_a` | 简化一阶动力学中的状态保留系数，必须在 `(0,1)`。越大表示系统响应越慢。 |
| `tracking_weight` | 跟随 policy reference 的代价权重。 |
| `command_delta_weight` | 抑制连续命令突变的代价权重。 |
| `max_velocity` / `max_acceleration` / `position_limit_abs` | MPC 输出之后使用的同类执行限制。 |

### `viewer`：是否实时发布

| 字段 | 含义 |
|---|---|
| `enabled` | 是否把状态、相机、plan 和事件发布给 ManiMux Viewer。mock 默认关闭。 |
| `robot_adapter` | Viewer 采用哪套机器人几何和关节映射；`enabled: false` 时不生效。 |

### `recording`：是否保存 episode

| 字段 | 含义 |
|---|---|
| `enabled` | 配置层已经声明该开关，但当前 EdgeRuntime 始终创建 Recorder，尚未根据该值关闭记录；mock 写 `true` 与实际行为一致。 |

### mock 中没有显式写出的常用字段

这些字段使用 schema 默认值，真实模型配置中经常显式声明：

| 字段 | 默认值 | 含义 |
|---|---:|---|
| `policy.startup_timeout_s` | `30.0` | 等待模型 worker 完成初始化的最长时间。 |
| `policy.trajectory_duration_s` | `null` | 若设置，则用 `duration / (horizon_steps-1)` 覆盖 `action_dt_s`。 |
| `policy.options` | `{}` | 具体模型插件自己的地址、相机映射、stats、group order 等参数。 |
| `robot.config` / `robot.options` | `null` / `{}` | RobotDriver 的外部配置路径和本体专用参数。 |
| `sensor.options` | `{}` | SensorDriver 专用的 endpoint、camera names 等参数。 |
| `execution.runtime` | `manimux` | 选择默认 Timeline runtime 或 `rtc` runtime。 |
| `execution.inference_schedule` | `deadline` | `deadline` 允许旧请求到期后提交新请求；`single_inflight` 始终只保留一个未完成请求。 |
| `execution.rtc` | defaults | 仅 `runtime: rtc` 使用的 delay、最小执行步数和 guidance 参数。 |
| `viewer.policy_label` | `""` | Viewer 与 Recorder 中显示的模型名称。 |
| `viewer.camera_hz` | `5.0` | 向 Viewer 发布相机图像的最高频率；不改变 policy 读取相机的频率。 |
