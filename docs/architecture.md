# 桌面双臂机器人推理与评测平台架构设计

状态：Draft v0.1  
日期：2026-08-16  
适用范围：桌面双臂为主，兼容单臂；单实验室起步，可扩展到多机器人与多推理节点

## 1. 结论先行

平台应采用四个相互独立的平面，而不是把模型、机器人和 Viewer 放进同一个 loop：

```text
                         Control Plane
         registry / compatibility / run / lease / rollout
                              │
              ┌───────────────┼────────────────┐
              │               │                │
              ▼               ▼                ▼
       Robot Edge Plane  Inference Plane   Data Plane
       state + control   model workers     record/catalog/eval
              │               │                │
              └──── events / typed contracts ──┘
                              │
                              ▼
                    Universal Viewer Plane
```

首期真正需要自研的核心不是新的模型 runtime，而是 Robot Edge Plane：

```text
timestamped observation
        │
        ├──────── asynchronous inference ────────┐
        │                                        │
        ▼                                        ▼
measured state ──► action timeline ◄── aligned action chunk
                        │
               smoother / local MPC
                        │
                 final safety gate
                        │
                  high-rate control
```

关键决定如下：

1. 机器人侧控制永不等待 inference、Viewer、数据库或远端服务。
2. 模型后端和机器人硬件分别插件化，中间通过版本化 observation/action contract 连接。
3. action chunk 是带时间语义、可替换的 plan，不是普通数组队列。
4. 双臂 chunk 作为一个 coordination unit 原子发布和替换。
5. 原始预测、适配、对齐、优化、安全处理、实际下发和实测状态全链路记录。
6. Universal Viewer 是观察和操作入口，不是机器人控制或安全的权威来源。
7. 首期采用 modular monolith + 多进程 + 单机友好的依赖；不预先引入 Kubernetes/Kafka。

## 2. 目标与边界

### 2.1 功能目标

- 一个机器人可以在不修改控制实现的情况下切换多个模型和 runtime；
- 一个模型可以通过 embodiment adapter 部署到多个兼容机器人；
- inference 可运行在机器人本机、同一局域网 GPU 主机或后续的云端；
- sensor、policy、controller、recorder 和 Viewer 以不同频率运行且互不阻塞；
- 支持 joint/EE action、absolute/delta action、单/双臂、不同 gripper 语义；
- 支持 action chunk overlap、延迟补偿、time parameterization、平滑与可选 MPC；
- 支持 mock、replay、simulation 和 real robot 使用同一上层接口；
- 支持 run/episode 管理、数据落盘、后台上传、人工/自动评分、离线 replay 和 A/B 对比；
- 能定位“模型慢”“网络慢”“buffer 空”“controller 抖”“动作被 safety 修改”等不同故障。

### 2.2 非目标

- 不承担机械臂厂商的认证安全功能；
- 不在首期训练 VLA；
- 不在首期重做 Tether 的 export/promotion 产品或 Embodied.cpp 的 kernel/runtime；
- 不要求一个统一的内部张量布局覆盖所有未来模型；统一的是带 schema 的边界；
- 不承诺普通 Linux/Python 进程提供硬实时，只提供隔离、deadline 监控和可测的 soft real-time 行为。

## 3. 参考项目核对与吸收边界

本设计基于 2026-08-16 获取的上游 HEAD。commit 固定用于说明调研基线，后续集成应重新核对。

| 项目 | 调研 commit | 吸收的能力 | 不直接照搬的部分 |
|---|---|---|---|
| [FastCrest/tether](https://github.com/FastCrest/tether) | `97d2a87` | artifact proof、parity、deadline、guard、record/replay、shadow/canary、低延迟 transport | 不在首期重建完整 deployment/MLOps 产品；其 BSL 1.1 代码不能未经审查直接并入 |
| [SEU-PAISys/Embodied.cpp](https://github.com/SEU-PAISys/Embodied.cpp) | `c5a96a2` | C++/GGUF backend、typed embodied adapter、VLA protobuf、分阶段 inference timing | 不让它承担 robot execution；不让其模型 ABI 成为平台唯一 action schema |
| [dexmal/realtime-vla-v2](https://github.com/dexmal/realtime-vla-v2) | `a36d02a` | inference/execution overlap、time-axis planning、delay alignment、local MPC/smoothing、分阶段轨迹日志 | 不复制 pickle-over-HTTP、全局 inference lock 和 AIRBOT/Pi0.5 专用结构 |
| [Universal-Control/ManiUniCon](https://github.com/Universal-Control/ManiUniCon) | `85c6f2e` | multi-process、shared-memory ring/queue、HAL、sensor/policy 解耦、Zarr/LeRobot 数据路线 | 不把共享内存 layout 当跨服务协议；补上 session、plan version、stale response、统一时钟与安全状态机 |
| [SII-LiuLab/universal_viewer](https://github.com/SII-LiuLab/universal_viewer) | `5726829` | `RobotAdapter`、`PolicyPlan`/`RobotSnapshot`、ZMQ bridge、3D/FK、动态相机、operator controls | v1 base64 JSON 只保留兼容 bridge；Viewer control 不直接成为执行授权 |

五者在目标架构中的位置是：

```text
checkpoint ── Tether proof/export ─┐
                                  ├── PolicyBackend workers
GGUF ──────── Embodied.cpp ───────┘
                                               │
                        Realtime-VLA ideas ────┤ action timeline/executor
                                               │
                        ManiUniCon ideas ──────┤ HAL/shared-memory edge
                                               │
                     universal_viewer ─────────┘ projection + operator UI
```

## 4. 核心设计原则

### 4.1 实时 lane 与异步 lane 分离

实时 lane 只包含：state read、timeline sample、local controller、final safety、driver write、watchdog。异步 lane 包含 camera processing、inference、record、upload、metrics、Viewer 和 run orchestration。

任何异步组件退出都不应拖死控制 loop。控制 loop 根据本地状态机和 plan TTL 决定 hold/stop。

### 4.2 用语义 schema 解耦，不用固定维度解耦

不能假设“双臂就是 14 维”。平台用 `EmbodimentSpec` 描述：

- group：`left_arm`、`right_arm`、`left_gripper`、`right_gripper` 等；
- 每组 DoF 名称、顺序、单位和 limits；
- action space：joint position/delta/velocity、EE pose/delta、gripper；
- frame：`world`、`robot_base`、`left_base`、`right_base`、`tool`；
- quaternion convention、normalization、control frequency；
- model tensor 与 canonical action group 的映射。

fast path 仍可以传连续 ndarray，但 ndarray 必须绑定 `schema_id + schema_version + embodiment_revision`，不能靠双方默契解释。

### 4.3 时间属于数据 contract

每个消息至少有：

- `source_monotonic_ns`：源主机采样/产生时间；
- `source_clock_id`：该 monotonic clock 的身份；
- `wall_time_ns`：检索和跨机粗对齐；
- `sequence`：同一 stream 内严格递增；
- `observed_at_edge_ns`：edge 收到远端结果的本地时间，适用时填写。

跨主机不能比较裸 monotonic timestamp。推理请求携带 observation 的 edge 时间；响应只携带 action offset、服务端 phase latency 和 echo id。edge 用本地 send/receive time 计算 action 放置位置。需要更准确的相机跨机对齐时使用 PTP/硬件时间戳，并记录 clock quality。

### 4.4 Latest-value、bounded、lossless 三类通道分开

| 数据 | 语义 | 背压策略 |
|---|---|---|
| robot state / camera preview | 最新值更重要 | latest-only ring，可丢旧帧 |
| inference observation | 最新完整对齐快照 | 新请求可 supersede 未开始旧请求 |
| inference response | 只接受当前 session/request | stale/duplicate 明确丢弃并记录 |
| active action timeline | 有界、按版本原子更新 | 不允许局部写入或无限增长 |
| control command | 有状态且需确认 | bounded reliable queue + idempotency key |
| episode raw data | 期望 lossless | 本地 spool；空间不足时终止/标记 episode，不能阻塞控制 |
| preview/metrics | best effort | 限频、采样、可丢弃 |

## 5. 系统组件

### 5.1 Robot Edge Plane

每台物理机器人对应一个 `edge-agent` deployment。双臂应视为同一个 `robot_id` 下的多个 kinematic group，而不是两个互不相关 robot session。

```text
Robot/Sensor Drivers
        │
        ▼
State Hub ───► Observation Synchronizer ───► Inference Session Client
   │                                              │
   │                                              ▼
   │                                      Plan Ingress Validator
   │                                              │
   ▼                                              ▼
Safety Supervisor ◄── Final Gate ◄── Local Executor ◄── Action Timeline
   │                                                        │
   └──────────────────────► Robot Driver ◄──────────────────┘

All stages ── non-blocking event tap ──► Recorder / Viewer Bridge / Metrics
```

组件职责：

#### Robot/Sensor Drivers

- 封装厂商 SDK、相机 SDK、teleop device；
- 暴露 capability、health、state、command 和 stop；
- 不加载 policy，不做业务层 action chunk 管理；
- 所有 driver 都需 mock/replay 实现和 contract tests。

#### State Hub

- 用共享内存 ring buffer 保存高频 robot state 与 sensor descriptor；
- 每个 stream 单独 sequence/timestamp；
- 支持 latest、last-k 和按时间邻近读取；
- 大图像存在固定 slot pool，消息只传 slot descriptor，避免多份复制。

#### Observation Synchronizer

- 根据 policy 的 `ObservationSpec` 从各 stream 取时间对齐快照；
- 明确记录 image age、state interpolation、缺帧、相机时间质量；
- 对 inference 输出 immutable `ObservationEnvelope`；
- 不在 control loop 内做 resize、JPEG、tokenize。

#### Inference Session Client

- 维护 `session_id`、递增 `request_seq`、deadline 和 cancellation；
- 默认同一 session 只允许一个 actively-computing request，可配置 speculative/pipelined 模式；
- 新 observation 可取消或标记旧请求为 superseded；
- 对 server timeout、disconnect 和 overload 做有界重试，不复用过期 action。

#### Plan Ingress Validator

- 校验 session/request、artifact、schema、shape、有限值、action dt/horizon、TTL；
- 调用 `ModelOutputAdapter` 完成 denormalization、joint/group mapping 和 frame conversion；
- 远端返回不能直接进入 driver。

#### Action Timeline

- 保存当前 `PlanBundle` 及少量历史，以 robot monotonic time 为索引；
- 同一 `coordination_group` 的 action 原子 commit；
- 新 plan 只替换尚未执行的时间区间，保留必要 continuity prefix；
- 拒绝 stale、时间倒退、过短 lookahead 或 schema 不匹配的 plan；
- buffer underrun 时返回定义好的 hold/stop policy。

#### Local Executor

可组合 stages：

1. latency alignment；
2. chunk stitching/blending；
3. time parameterization；
4. interpolation/resampling；
5. optional smoothing 或 MPC；
6. gripper discrete/continuous handling；
7. final command generation。

各 stage 输入输出都可记录。MVP 先实现 deterministic interpolator + velocity/acceleration limiter；MPC 在基础时序和日志验证后加入。

#### Safety Supervisor

- 维护 robot lifecycle state；
- 监视 driver、controller、plan age、state age、process heartbeat、joint/workspace/self-collision/cross-arm constraints；
- 最终决定 `allow / clamp / reject / fault`；
- safety fault 需要 operator acknowledgement，不能靠下一帧正常 action 自动清除。

#### Recorder Agent

- 订阅 edge event tap，不参与控制判定；
- 先写本地 append-only spool 和 WAL，再异步整理/upload；
- 即使上传、对象存储或 metadata DB 不可用也能继续有限时间运行；
- 达到 disk high-water mark 时按配置停止开始新 episode，正在运行的 episode给出明确 terminal reason。

### 5.2 Inference Plane

```text
Edge clients
    │
    ▼
Inference Gateway
  auth / admission / sticky session / deadline / metrics
    │
    ├── PythonPolicyBackend (PyTorch/JAX wrapper)
    ├── EmbodiedCppBackend (protobuf/ZMQ adapter)
    ├── TetherBackend (HTTP/ZMQ adapter)
    ├── TensorRT/ONNX backend
    └── Fake/Replay backend
```

`Inference Gateway` 只负责通用调度，不解释机器人厂商接口。`PolicyBackend` contract 至少包括：

- `load(artifact_manifest)` / `unload()`；
- `open_session(session_context)` / `reset_session()` / `close_session()`；
- `predict(observation, deadline)`；
- `health()` / `capabilities()`；
- 可选 `cancel(request_id)`、prefix cache、local conditioning、streaming chunk。

必须支持有状态模型：同一 episode sticky 到同一个 artifact/backend replica；reset 是一等 API。模型 worker crash 后不能在未知 cache 状态下继续 session，应使 session epoch 变化并通知 edge 重新建立。

Gateway 的 deadline 只决定“是否还值得计算/返回”，机器人安全由 edge 的本地 deadline 决定。服务端不得用 last-good action 冒充当前有效响应；若需要 fallback，必须返回显式 `fallback_kind`，edge 再决定是否接受。

### 5.3 Control Plane

MVP 采用一个 `run-manager` modular service，包含：

- Robot Registry：robot type、instance、capability、online status；
- Model Registry：artifact manifest、digest、runtime、validation status；
- Embodiment Registry：schema、normalization、calibration、safety revision；
- Compatibility Resolver：判断 model × embodiment × runtime 是否可组合；
- Resource/Lease Manager：robot 与 GPU worker 的互斥 lease；
- Run/Episode Manager：状态机、task spec、seed、operator、终止原因；
- Deployment Session Manager：将已验证 artifact 绑定到 inference worker 和 robot；
- Command API：start/pause/resume/step/home/finish/fault-ack；
- Audit Log：所有有副作用的 operator/control-plane 操作。

不要让 scheduler 直接控制机械臂。它只能向 edge 提交带 id、precondition 和 TTL 的命令，等待 edge acknowledgement。

### 5.4 Data Plane

建议按四类存储分工：

| 类型 | MVP 选型 | 内容 |
|---|---|---|
| 本地实时 spool | MCAP 或等价 append-only chunk file | timestamped state/action/event、图像索引、schema |
| 大对象存储 | S3/MinIO | MCAP、MP4、深度/Zarr、模型 artifact、报告、calibration |
| metadata/catalog | PostgreSQL；单机开发可 SQLite | robot/model/run/episode/artifact/annotation 索引 |
| metrics/traces | Prometheus + OpenTelemetry/OTLP | 在线健康、性能、分布式 trace |

选择 MCAP 是为了把异构时间流、schema 与索引放在一个可恢复容器；它不是训练数据最终格式。后台 exporter 可生成：

- LeRobot dataset；
- RLDS；
- Zarr；
- Parquet summary；
- MP4 preview；
- inference record/replay fixture。

原始数据只写一次，多种训练/分析格式由可版本化 exporter 派生，避免在线 loop 同时维护多套格式。

### 5.5 Universal Viewer Plane

现有 Universal Viewer 已正确将机器人几何差异隔离进 `RobotAdapter`。建议保留该项目为独立包，通过 `viewer-bridge` 接平台事件，不把 Viewer 源码并入 edge control。

Viewer 接入分两阶段：

#### Compatibility bridge（MVP）

- 将 platform `PlanBundle` 投影成现有 `PolicyPlan`；
- 将 `MeasuredState` 和 camera preview 投影成 `RobotSnapshot`；
- 继续使用 ZMQ localhost 端口连接现有 Viewer；
- edge/bridge 限频并只发送 preview，不把该通道作为 recorder；
- 现有 `paused=True` fail-closed 行为保留，但真正 pause 状态以 edge 为准。

#### Viewer Protocol v2

新增统一 `EventEnvelope`：

```text
schema_version, event_id, run_id, episode_id, robot_id,
session_id, sequence, source_monotonic_ns, wall_time_ns,
kind, payload
```

Viewer v2 应展示：

- raw/adapted/aligned/optimized plan、safe command 的切换与差异；
- actual command 与 measured state；
- inference phase latency、RTT、observation age、timeline lookahead；
- plan replacement、buffer underrun、safety clamp/reject/fault；
- model artifact、embodiment/calibration/safety revision；
- live 和 historical episode 的同一套 timeline。

图像通过 binary frame、shared-memory descriptor 或 artifact URL 传输，不进入 base64 JSON event。

Viewer command 流改为：

```text
UI intent
  -> Run Manager command(command_id, expected_state, ttl)
  -> Edge Safety Supervisor
  -> accepted/rejected acknowledgement
  -> Viewer renders authoritative state
```

实体急停永远不经 Viewer。

## 6. 核心数据契约

### 6.1 公共 Envelope

所有 RPC/event 复用公共 metadata：

```proto
message EnvelopeMeta {
  uint32 schema_version = 1;
  string event_id = 2;
  string robot_id = 3;
  string run_id = 4;
  string episode_id = 5;
  string session_id = 6;
  uint64 sequence = 7;
  uint64 source_monotonic_ns = 8;
  string source_clock_id = 9;
  int64 wall_time_ns = 10;
  string trace_id = 11;
}
```

这是方向性示例；正式 field number 需要在 `packages/contracts` 锁定并加兼容性测试。

### 6.2 ObservationEnvelope

包含：

- meta；
- `observation_schema_id/revision`；
- named robot state groups；
- named image/depth references 与每路 capture timestamp；
- language/task context；
- 当前 timeline cursor、pending horizon、上一 plan id；
- 可选 previous action/history condition；
- 每个输入的 validity/age/alignment quality。

Inference worker 返回的是与请求绑定的 `RawPolicyPlan`，不能只返回裸 list。

### 6.3 PlanBundle

```text
plan_id
request_id / request_seq / session_epoch
model_artifact_digest
embodiment_revision
action_schema_id / revision
coordination_group = "dual_arm"
time_base = relative_offsets
offsets_ns[T] or fixed_dt_ns
groups {
  left_arm:      [T, D_l]
  right_arm:     [T, D_r]
  left_gripper:  [T, D_lg]
  right_gripper: [T, D_rg]
}
stage
valid_for_ns
inference_profile
quality/fallback flags
```

一个双臂 plan 的所有 group 必须共享 `plan_id` 和时间网格。需要异步 gripper 时可以内部使用不同采样保持，但 commit 仍是原子的。

### 6.4 ModelArtifactManifest

artifact 不能只是 checkpoint 路径。manifest 至少固定：

- immutable digest、model family/version、source commit；
- runtime kind/version、container/build digest；
- observation/action schema；
- state/action normalization artifact digest；
-支持的 embodiment 与 adapter revision；
- input camera names、shape、color order；
- chunk horizon/dt、是否 stateful、reset semantics；
- precision、device requirements、estimated VRAM；
- parity/benchmark report 与 validation status；
- license/provenance。

通过 Tether export/proof 的模型可把 proof packet 作为 manifest artifact；Embodied.cpp GGUF 则记录 GGUF 和 projector/action-head 的完整 digest。

### 6.5 RobotManifest 与 EmbodimentSpec

`RobotManifest` 描述硬件实例和 capability；`EmbodimentSpec` 描述 model/control 共同理解的语义。两者分开是因为同型号机器人也可能有不同相机、gripper、标定和安全区域。

重要 revision：

- driver build；
- joint/group ordering；
- URDF；
- camera intrinsics/extrinsics；
- tool/TCP；
- workspace/collision geometry；
- safety limits；
- action normalization。

任一关键 revision 变化都应生成新的 compatibility decision，不能沿用旧 session。

## 7. 异步推理与 action chunk 执行流程

### 7.1 正常流程

```text
t0  camera/state snapshot aligned
t1  request N sent (edge monotonic timestamp retained)
    current plan N-1 continues executing
t2  response N received
    validate IDs/schema/TTL/finiteness
    adapt + denormalize
    estimate observation-to-now delay
    discard already-obsolete prefix
    stitch against currently commanded trajectory
    time-parameterize and run safety precheck
t3  atomically commit future segment of plan N
    controller samples new timeline at next tick
```

推理线程不应在“整个旧 chunk 执行完”后才请求下一 chunk。触发条件可组合：

- timeline lookahead 低于 `refill_threshold`；
- 距上次 inference 超过 cadence；
- scene/task context 变化；
- safety/contact event 请求 replan；
- operator step 模式。

### 7.2 Plan replacement 规则

新 plan 只有同时满足以下条件才可 commit：

1. session 和 session epoch 匹配；
2. request sequence 不旧于已接受请求；
3. observation age 与 response age 未超过上限；
4. schema、artifact、embodiment revision 匹配；
5. action 全为有限值且维度/单位合法；
6. continuity window 内的位置/速度/加速度跳变可处理；
7. safety precheck 没有 reject；
8. 具有足够未来 horizon，不会 commit 后立即 underrun。

commit point 应在 `now + control_lead_time` 之后。旧 plan 在 commit point 前保留，新 plan 前缀根据 latency 和当前实测/指令状态裁剪，再用短 blend window 连接。

### 7.3 超时与失败

| 故障 | Edge 行为 | 数据记录 |
|---|---|---|
| inference deadline miss | 继续当前有效 timeline；触发下一请求或降级 backend | timeout、lookahead、backend |
| stale response | 丢弃，不更新 timeline | stale reason、request/session id |
| timeline underrun | hold last safe target；超过 hold TTL 后 stop/fault | underrun duration、terminal reason |
| camera stale | 停止发新 inference；按 task safety policy hold/stop | per-camera age |
| state stale | 立即禁止新 command，driver stop/fault | watchdog reason |
| executor/MPC failure | 使用已验证 fallback limiter；无 fallback 则 hold/fault | solver status、fallback kind |
| recorder failure | control 继续；标记 episode degraded；spool 满则阻止新 episode | storage error |
| Viewer disconnect | execution 不自动继续或停止，由当前 edge state 和 policy 决定 | viewer connectivity event |

“last-good action”不能无限复用。任何 fallback 都必须带 expiry。

## 8. Robot lifecycle 与安全状态机

建议 edge 维护以下权威状态：

```text
OFFLINE -> CONNECTING -> DISARMED -> HOMING -> READY -> ARMED -> RUNNING
                 │          │          │        │         │
                 └──────────┴──────────┴────────┴──────► FAULT
                                              RUNNING -> PAUSED
                                              PAUSED  -> RUNNING
```

规则：

- `RUNNING` 需要有效 robot lease、operator arm、健康 driver、fresh state、合法 active session 和 safety config；
- pause 后 timeline 冻结/清空的策略由 controller 类型定义，但 resume 必须从新鲜 observation/replan 开始；
- `HOME` 只能从允许的状态进入 `HOMING`，使用独立受限轨迹，不执行 policy plan；
- `FAULT` 不能自动恢复到 RUNNING；
- 单臂 driver fault 默认使整个 dual-arm coordination group fault，除非 task 显式声明安全的单臂降级；
- Viewer 的按钮只发 intent，最终状态取 edge acknowledgement。

Safety checks 分三层：

1. artifact/plan ingest：schema、NaN、粗 limits、workspace、trajectory feasibility；
2. timeline/executor：速度、加速度、jerk、continuity、cross-arm collision lookahead；
3. final tick：freshness、joint/workspace clamp/reject、driver watchdog、fault latch。

## 9. Eval 与数据模型

### 9.1 领域对象

```text
EvaluationSuite
  └── EvaluationRun
        ├── model artifact + robot/embodiment/config/calibration revisions
        ├── task set + seed/repetition protocol
        └── Episode[]
              ├── raw streams and event ledger
              ├── outcome / terminal reason
              ├── annotations[]
              └── derived artifacts / metrics[]
```

`Run` 固定实验条件；`Episode` 是一次从 reset/home 后开始到 success/failure/abort/fault 的尝试。不能为了修正评分覆盖 episode 原记录。

### 9.2 Episode 必记内容

- robot/model/runtime/container/source/config digest；
- task、instruction、seed、operator、开始结束时间；
- robot state、camera timestamps、raw sensor artifact references；
- 每个 inference request/response 与 phase latency；
- 七阶段 action lineage：raw、adapted、aligned、optimized、safe、executed、measured；
- control/safety/operator event；
- success、terminal reason、steps、wall time；
- recorder completeness、clock quality、dropped stream count；
- annotation revision 和 evaluator version。

标准 `terminal_reason` 采用有界 enum，例如：

```text
success, task_failure, timeout, operator_abort, safety_fault,
robot_fault, inference_failure, observation_failure,
timeline_underrun, storage_degraded, infrastructure_error
```

错误文本另存，不能把任意字符串当主要分类。

### 9.3 Online 与 offline eval

- Online evaluator：读取 task signal/operator annotation，结束 episode 并生成初始 outcome；
- Offline evaluator：从不可变 artifact 重算轨迹质量、动作误差、碰撞/越界、模型对比；
- Replay evaluator：将相同 observation trace 发给不同 artifact，比较 raw/adapted plan 和 latency，不驱动真机；
- Shadow evaluator：候选模型收到相同 observation，但输出只记录、不进入 active timeline。

Shadow 与 active policy 不能共享未知状态 cache；每个 backend 保持独立 session state。

## 10. 调度、兼容性与多机器人

### 10.1 Compatibility Resolver

在创建 run 前检查：

```text
model ObservationSpec  <= robot/sensor capabilities
model ActionSpec       -> embodiment adapter -> controller capabilities
runtime requirements   <= worker device/VRAM/software
safety revision        compatible with robot/tool/calibration
viewer adapter         optional, never blocks run compatibility
```

结果是带理由的 `compatible / incompatible / requires_override`，并保存到 run manifest。

### 10.2 Lease

- robot hardware 使用 exclusive lease；
- inference worker 可按显存和并发能力共享；
- session sticky 到 worker replica；
- lease 带 TTL 和 owner，edge 在 control-plane 短暂断开时可按本地 policy 完成/暂停当前 episode，但不能无 lease 开新 episode；
- operator command 带 expected state，避免旧 Viewer tab 重放命令。

### 10.3 首期不做复杂集群调度

MVP 使用静态 worker registration 和简单 least-loaded/affinity routing。只有出现真实的多节点资源竞争后，再考虑 NATS JetStream、专用 scheduler 或 Kubernetes。消息系统不进入控制 loop。

## 11. API 与 transport 建议

| 边界 | MVP transport | 理由 |
|---|---|---|
| 同机 driver/state/controller | shared memory + bounded queue/eventfd | 低复制、确定背压 |
| edge ↔ inference gateway | gRPC async unary 或 bidi stream + protobuf | 多语言、deadline/cancel、二进制图像、schema |
| edge/control plane command | gRPC/HTTP + command ack | 低频、可审计、易调试 |
| event/recorder local tap | Unix socket/IPC pub-sub + bounded subscriber queues | 观察者不阻塞生产者 |
| Viewer v1 bridge | 现有 ZMQ | 最小改造 |
| object upload | S3 multipart | 可恢复、通用 |

不采用 Python pickle 作为网络协议。ZMQ 可以作为低延迟 transport，但消息体仍应是版本化 protobuf/flat binary，生产网络需认证与加密。

## 12. 可观测性与 SLO

### 12.1 统一 trace

一次 action 的 trace chain：

```text
sensor capture
 -> observation sync
 -> inference queue/serialize/network/compute
 -> response validate/adapt
 -> timeline commit
 -> controller sample/driver send
 -> measured state
```

trace/event 通过 `trace_id + request_id + plan_id` 关联。recording ledger 是可复现事实源，OTel 是在线诊断视图，两者不能相互替代。

### 12.2 初始性能目标

目标应按具体机器人 profile 配置，以下是首期工程验收基线，不是所有硬件承诺：

| 指标 | 初始目标 |
|---|---|
| edge control loop | 100 Hz profile 下 p99 tick < 5 ms，deadline miss 可见 |
| state freshness | 50 Hz source 下 p99 age < 40 ms |
| plan commit | 收到有效 response 后 p99 < 10 ms，不含模型适配的重计算 |
| action timeline | 正常运行 lookahead 不低于 2 个 control tick；目标阈值按模型 chunk 配置 |
| viewer/recorder outage | 不增加 control deadline miss |
| stale response | 100% 被拒绝且产生 bounded-reason event |
| data completeness | 正常结束 episode 的必需 stream/schema/manifest 100% 可校验 |

推理 SLO 按 model/runtime/hardware profile 分开，不能用一个“30 Hz”数字覆盖所有模型。

## 13. 建议代码组织

```text
apps/
  edge_agent/                # edge composition root 与 lifecycle
  inference_gateway/         # routing/admission/backend hosting
  run_manager/               # registry/run/lease/command API
  viewer_bridge/             # platform events -> universal_viewer
packages/
  contracts/
    proto/
    generated/
    schemas/
  edge_runtime/
    state_hub/
    observation/
    timeline/
    executor/
    safety/
    lifecycle/
  robot_hal/
    base/
    mock/
    replay/
    robots/
    sensors/
  policy_runtime/
    base/
    fake/
    python/
    embodied_cpp/
    tether/
  eval_runtime/
  data_runtime/
  common/
configs/
  robots/
  embodiments/
  policies/
  safety/
  evals/
deploy/
  compose/
  systemd/
docs/
  adr/
  protocols/
tests/
  contract/
  integration/
  hardware/
```

部署进程可以少于 package 数量。MVP 建议进程：

1. `edge-agent`；
2. `inference-gateway` + 一个或多个 worker；
3. `run-manager`；
4. `viewer-bridge`/Universal Viewer；
5. PostgreSQL/MinIO/Prometheus 等可选 supporting services。

Recorder 可先作为 edge-agent 的隔离子进程，后续独立部署。

## 14. 分阶段落地

### Phase 0：contracts 与可执行骨架

- monorepo、标准命令、CI；
- protobuf/schema、ID 与 clock utilities；
- mock dual-arm robot、fake camera、fake policy；
- lifecycle、PlanBundle、Action Timeline 的纯状态机；
- 端到端 `mock-run`；
- Viewer v1 compatibility bridge。

验收：无 GPU、无真机时能看到双臂预测/实测轨迹，pause/resume 有 edge ack，记录产生完整 episode。

### Phase 1：单机真机最小闭环

- 首个双臂 driver 与 RealSense；
- shared-memory State Hub；
- deterministic executor、limits、watchdog；
- PythonPolicyBackend；
- 本地 spool + catalog + artifact upload；
- manual eval annotation。

验收：推理进程、Viewer、数据库分别被杀掉时，控制行为符合故障表；能 replay 同一 episode。

### Phase 2：异步实时执行

- latency estimator、chunk trimming/stitching、atomic replacement；
- time parameterization；
- optional smoother/local MPC；
- raw 到 measured 的全阶段 Viewer/record；
- inference timeout、cancellation、session affinity。

验收：用可控延迟/抖动注入证明无 stale plan 执行，比较同步 chunk baseline 与异步 timeline 的速度、平滑度和成功率。

### Phase 3：多 runtime 与标准 eval

- EmbodiedCppBackend；
- TetherBackend/artifact proof attachment；
- evaluation suite、offline diff、shadow policy；
- LeRobot/RLDS/Zarr exporter；
- model × robot compatibility resolver。

验收：同一 robot/run spec 可切换两个 backend；同一 trace 可对两个 artifact 生成可比较报告。

### Phase 4：多机器人生产化

- worker resource scheduling、robot lease、RBAC；
- artifact promotion/rollback；
- canary/shadow 自动门控；
- fleet dashboard、retention、备份和审计。

只有当 2–3 台机器人真实并发暴露瓶颈后才进入该阶段。

## 15. MVP 验收场景

以下场景应成为 integration tests 或可重复 test procedure：

1. Fake model 每 200 ms 返回 1 秒双臂 chunk，controller 100 Hz 连续执行；
2. 人为注入 50–500 ms 抖动，timeline 不执行过期 response；
3. response 乱序、重复、维度错误、含 NaN，均被拒绝；
4. 新 chunk 在未来 commit point 原子替换，左右臂 plan id 始终相同；
5. 推理 worker crash 时 robot 先完成有效 lookahead，再 hold，TTL 后 fault；
6. Viewer crash/重启不影响控制 deadline；
7. recorder disk error 不阻塞控制，episode 标为 degraded；
8. state stream stale 触发 stop/fault；camera stale 不再发 inference；
9. pause/home/finish 使用 command id 和 edge acknowledgement，旧命令不能重放；
10. episode 可完整 replay，并能导出一个训练格式和一个 eval report。

## 16. 需要通过 ADR 尽早确认的问题

以下问题不阻塞 Phase 0，但在相关实现开始前应记录 ADR：

1. 首个真实双臂型号、厂商 SDK 与控制模式；
2. MCAP 是否作为 canonical raw episode container，或使用自定义 chunked Zarr；
3. edge 主实现使用 Python 多进程、C++ sidecar，还是 controller 部分单独 C++；
4. local IPC 采用现成共享内存库还是自研固定 layout；
5. gRPC image payload 是内联压缩帧、共享对象引用还是视频流；
6. 首期 trajectory action space 以 joint position 还是 relative EE pose 为规范入口；
7. cross-arm collision checker 与 MPC solver 的具体选择；
8. Universal Viewer v2 协议是在原仓库演进还是由本仓库维护独立 bridge contract。

无论这些选择如何变化，第 4 节的实时隔离、显式时间、版本化 schema、双臂原子 plan 和 append-only data lineage 都应保持不变。
