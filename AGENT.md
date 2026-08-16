# ManiMux V1 开发约定

## 1. 我们要做什么

ManiMux 是一套面向桌面机械臂、以双臂为重点的本地运行与评测基础设施。V1 只解决一个闭环：

```text
本地相机/机器人
      -> 本地模型异步推理
      -> time-indexed action chunk
      -> 安全、连续地执行
      -> 本地记录并可视化/回放
```

V1 必须支持：

- robot、sensor、policy 和 Viewer 以不同频率运行且互不阻塞；
- 通过 adapter 切换不同机器人和本地模型；
- 双臂 action chunk 的延迟对齐、过期丢弃和原子替换；
- 机器人状态、模型输出、实际命令和相机数据写入本地 episode；
- 接入 `SII-LiuLab/universal_viewer` 做实时观察和 pause/resume/home/finish；
- mock/replay 环境下完成无真机、无 GPU 的端到端测试。

详细设计见 [docs/architecture.md](docs/architecture.md)。

## 2. V1 明确不做

- 远端或云端模型推理；
- 数据上传、对象存储和远端 catalog；
- Inference Gateway、认证、配额或多租户 admission control；
- 独立 Control Plane、run-manager、registry、lease 或 scheduler 服务；
- Kubernetes、Kafka、NATS、Prometheus 或分布式 tracing；
- 数据库；本地目录和 manifest 是事实源；
- 模型训练、ONNX export、GGUF runtime 或 CUDA kernel；
- MPC、复杂 trajectory optimizer 和自动策略切换；
- shadow/canary、promotion/rollback 和 fleet 管理；
- Universal Viewer v2 协议。

未来需求出现时再通过 ADR 引入这些能力。不要为尚未发生的扩展建立空服务或抽象层。

## 3. V1 运行边界

顶层只有三个运行单元：

```text
manimux CLI / launcher
├── edge-agent
├── policy-worker
└── universal_viewer
```

- launcher 读取一个 YAML、创建 run/episode 目录并启动/停止进程；
- edge-agent 拥有 HAL、observation、action timeline、controller、safety 和 recorder；
- policy-worker 一次只加载一个本地模型；
- edge-agent 直接使用 Universal Viewer 的现有 ZMQ bridge，不再增加 viewer-bridge 服务。

Recorder 可以是 edge-agent 的独立子进程，但对用户不是第四个服务。

## 4. 不可破坏的约束

### 4.1 控制路径

- controller loop 不能等待模型、阻塞式 IPC、磁盘、Viewer 或日志。
- policy-worker 和 Viewer 不能直接调用机器人 driver。
- controller 每个 tick 只读取本机有界 action timeline，并执行 watchdog/final limits。
- worker 失联、模型超时、plan 过期、state stale 或 schema 不匹配时，必须进入明确的 `hold` 或 `fault`，不能猜测新 action。
- 双臂 action chunk 必须作为一个版本原子替换，左右臂不能执行不同 plan。
- Viewer 只是操作入口；edge-agent 的状态才是权威状态。实体急停永远不经过软件 Viewer。

### 4.2 时间语义

- 本地间隔和调度使用 monotonic clock，禁止用 `time.time()` 计算控制间隔。
- observation、inference request/response、plan、command 和 state 都携带 monotonic timestamp 与递增 sequence。
- action chunk 使用相对时间或固定 `dt`；policy-worker 不决定绝对执行时间。
- edge-agent 根据 observation age、response receive time 和当前 timeline 决定新 chunk 的 commit point。
- 所有时间字段必须在名称中包含单位，例如 `_ns`、`_ms` 或 `_s`。

### 4.3 Action contract

- 每个 run 只使用一个明确的 action schema，不尝试在核心代码里自动推断维度或语义。
- policy adapter 负责 normalization、joint/group mapping、frame/action-space 转换。
- timeline 只接收 adapter 校验后的 canonical action chunk。
- action 必须绑定 `plan_id`、`request_seq`、`action_space`、`dt_ns` 和双臂 group layout。
- NaN/Inf、维度错误、旧 request 或不连续且无法限速的 chunk 必须拒绝并记录 reason。

### 4.4 本地记录

V1 至少记录四条 action/state 数据链：

```text
raw_model_action -> scheduled_action -> command_sent -> measured_state
```

若以后增加 smoothing/MPC，再以可选 named stage 追加，不要求 V1 预先实现七阶段 pipeline。

- numeric arrays 和 timestamps 写入 `data.zarr`；
- 相机写入本地 MP4，逐帧 timestamp 保存在 Zarr；
- 离散事件写入 `events.jsonl`；
- run/episode/result 元数据使用 JSON；
- episode 运行时目录以 `.partial` 结尾，正常结束后原子重命名；
- recorder 失败不能反压 controller，磁盘空间不足时禁止开启新 episode。

## 5. V1 插件接口

只保留三个必要接口：

- `RobotDriver`：连接、读状态、发命令、home、stop；
- `SensorDriver`：启动、读取带时间戳的 frame、停止；
- `PolicyAdapter`：构造 observation、调用/解码 policy、把输出转换成 canonical action chunk。

一个新的 robot + policy 组合如果需要特殊 mapping，在 `PolicyAdapter` 中显式实现。V1 不建立 Robot Registry、Model Registry、Embodiment Registry 或通用 Compatibility Resolver；launcher 在启动时调用 adapter 自检并快速失败。

## 6. 代码组织

V1 使用一个 Python package，不建立多 package monorepo：

```text
src/manimux/
  cli.py
  config.py
  types.py
  runtime/
    edge.py
    timeline.py
    safety.py
  robots/
    base.py
    mock.py
  sensors/
    base.py
    mock.py
  policies/
    base.py
    worker.py
    fake.py
  recording/
    episode.py
  viewer/
    bridge.py
configs/
tests/
  unit/
  integration/
  hardware/
docs/
```

当 C++ backend 真正接入时，再增加 protobuf/Unix socket adapter；V1 Python 进程间优先使用 bounded multiprocessing queue 和 shared-memory image slot。

## 7. 开发要求

- Python 3.11+，公共类型使用 dataclass/Pydantic 或明确的 typed structure；禁止在核心 contract 中传无约束 `dict[str, Any]`。
- 配置使用一个经严格校验的 YAML；未知字段报错，resolved config 保存到 run 目录。
- 队列必须有固定上限；禁止无限 queue、无限 retry 和静默 fallback。
- 捕获异常后必须记录分类 reason 并触发定义好的状态变化；禁止 `except Exception: pass`。
- joint order、坐标系、四元数顺序、单位、normalization 和 gripper 语义必须在配置或 adapter 中显式写明。
- 新依赖必须服务于 V1 已实现功能，不能只为未来扩展引入。

根目录在代码落地后提供：

```bash
make format
make lint
make typecheck
make test
make test-integration
make mock-run
```

`make test` 不访问真机、外网或云服务。

## 8. 测试要求

- timeline 和状态机使用 fake clock，不用真实 `sleep` 验证时序；
- adapter 有 shape、unit、joint order 和 NaN/Inf contract tests；
- integration 至少覆盖 response 超时、乱序、重复、stale、双臂原子替换和 buffer underrun；
- recorder 测试 partial episode、磁盘写失败和进程重启后的可发现性；
- Viewer 退出不能影响 controller deadline；
- 新 robot 先通过 mock/replay，再通过 dry-run，最后才进行低速真机测试；
- 真机测试放在 `tests/hardware/`，默认 CI 不运行。

## 9. 真机安全

- 启动默认 `PAUSED`，必须收到新鲜 operator start/resume 才能执行 policy。
- `Home` 是受限运动命令，不是 fault recovery。
- 软件不得屏蔽厂商急停、限位、错误或 watchdog。
- safety fault 不自动恢复；需要人工确认并重新建立新鲜 observation/plan。
- 真机入口必须显式指定 real-robot 配置，并显示 robot identity、action schema 和 limits。

## 10. 完成定义

变更只有在以下条件满足时才完成：

- 正常、超时、stale、崩溃和磁盘失败路径均有定义；
- 相关 unit/integration tests、lint 和 typecheck 通过；
- mock-run 仍能完成 inference -> timeline -> control -> record -> Viewer 闭环；
- 配置与文档同步；
- 没有新增 V1 范围之外的服务、依赖或抽象。

参考项目只借鉴其边界清晰的思想：ManiUniCon 的 HAL/shared memory、Realtime-VLA V2 的异步 timeline、Embodied.cpp 的本地 backend、Tether 的 parity/record-replay 方法，以及 Universal Viewer 的 `RobotAdapter`。不直接复制它们的完整平台结构。
