# 桌面双臂机器人基础设施开发约定

## 1. 项目使命

本仓库用于构建一套面向桌面机械臂、以双臂为重点的机器人运行与评测基础设施。平台需要让不同机器人、传感器、模型 runtime 和评测任务通过稳定契约组合，并支持：

- 机器人本地高频控制与低频、可变延迟的模型推理解耦；
- 多种 VLA、普通 policy 和多种推理后端按统一接口运行；
- action chunk 的时间对齐、异步替换、平滑、约束和安全执行；
- 可复现的在线评测、完整 episode 数据记录、离线 replay 和模型对比；
- 接入 `SII-LiuLab/universal_viewer` 做实时观察、回放与受控操作；
- 从单机单机器人逐步扩展到多机器人、多 GPU worker，而不牺牲单机开发体验。

详细设计见 [docs/architecture.md](docs/architecture.md)。若代码实现与设计文档冲突，应先明确冲突并通过 ADR 更新设计，不能静默偏离。

## 2. 当前范围

优先完成：

1. 双臂机器人 HAL、传感器采集和本地 edge runtime；
2. 异步 inference session 与可插拔 model backend；
3. time-indexed action buffer、延迟对齐和安全执行；
4. run/episode 数据记录、评测结果和 artifact catalog；
5. Universal Viewer bridge；
6. mock robot、replay robot 和仿真端到端测试。

当前不做：

- 模型训练框架；
- 通用的 Kubernetes/fleet 管理平台；
- 自研 CUDA kernel、ONNX exporter 或 GGUF inference engine；
- 用 Web UI、Python GC、网络 RPC 或数据库承担硬实时保证；
- 用软件 pause 替代实体急停或厂商安全控制器。

Tether、Embodied.cpp、Realtime-VLA V2 和 ManiUniCon 是参考或适配目标，不是必须 fork 到本仓库的依赖。引入上游代码前必须检查许可证、维护成本和接口边界，并记录 ADR。

## 3. 不可破坏的架构约束

### 3.1 控制路径

- 机器人控制 loop 不能等待模型、网络、磁盘、数据库、Viewer 或日志系统。
- 控制 loop 只读取本机有界内存结构中的最新有效 plan，并在每个 tick 做 watchdog 和 safety 检查。
- 网络断开、模型超时、plan 过期、schema 不兼容或进程失联时必须进入明确的 fail-safe 行为，默认 `hold`，无法可靠 hold 时 `stop`。
- policy、inference worker 和 Viewer 都不能绕过 edge safety supervisor 直接调用机器人驱动。
- 双臂协调 plan 必须原子替换；不能让左、右臂分别落入不同 chunk 版本。
- MPC、插值器和平滑器属于执行质量层，不是安全认证边界。其输出仍需经过最终限制与 watchdog。

### 3.2 时间语义

- 控制与调度使用 monotonic clock；UTC 只用于跨机器检索和展示。
- 所有 observation、plan、command、state 和 event 都必须携带时间戳、序号以及 `run_id`/`episode_id`/`robot_id` 中适用的标识。
- 时间单位在协议层固定使用整数纳秒，在配置与用户界面可使用带单位名称的秒或毫秒字段；禁止无单位的 `timeout`、`latency`、`timestamp`。
- 远端 inference server 不能决定机器人绝对执行时间。它只返回 action offsets/dt 和服务端耗时；edge runtime 根据本地收发时间、observation age 和延迟估计落入 action timeline。
- 禁止用 `time.time()` 计算本地控制间隔。

### 3.3 数据与 schema

- wire contract、持久化 schema 和插件 contract 都要显式版本化。
- 已发布 schema 采用 additive evolution；重命名、改变单位或改变语义需要新 major version 和迁移工具。
- 记录链必须保留 `raw_model_plan -> adapted_plan -> aligned_plan -> optimized_plan -> safe_command -> executed_command -> measured_state`，其中 `optimized_plan` 还应标明 time parameterization、smoothing 或 MPC 子阶段；不能只保存最终 action。
- 原始 episode artifact 一经完成即视为不可变；人工标注和自动评分作为新的 revision 追加。
- 控制路径数据与分析数据分离：控制使用有界 ring buffer/queue，分析使用 recorder 的异步副本。
- 图像不得以 base64 JSON 作为平台主数据格式；本机用共享内存/二进制 buffer，跨机用二进制 RPC，持久化用视频、图像或 chunked array artifact。

### 3.4 插件边界

- 新机器人通过 `RobotDriver`、`SensorDriver`、`EmbodimentSpec` 和 capability manifest 接入。
- 新模型通过 `PolicyBackend` 与 model artifact manifest 接入。
- 模型输出先由 model/embodiment adapter 转换为规范 action schema，再进入通用执行管线。
- UI 的几何/FK 适配器与控制 HAL 是不同 contract；可以复用元数据，不能让 Viewer adapter 成为控制实现。
- 插件不得通过 import 私有实现或读取其他模块内部共享内存布局来耦合。

## 4. 计划中的仓库边界

在代码尚未落地前，以以下目录作为目标结构。新增顶层目录应先说明为什么现有边界不够。

```text
apps/                       # 可部署进程的入口
  edge_agent/
  inference_gateway/
  run_manager/
  viewer_bridge/
packages/
  contracts/                # protobuf、schema、生成代码
  edge_runtime/             # timeline、arbiter、safety、controller
  robot_hal/                # robot/sensor driver contract 与插件
  policy_runtime/           # backend contract、router、artifact loader
  eval_runtime/             # run/episode lifecycle、evaluator
  data_runtime/             # recorder、spool、uploader、catalog client
  common/                   # 少量无领域归属的基础工具
configs/                    # 经 schema 校验的配置
deploy/                     # compose/systemd/容器和部署配置
docs/
  adr/                      # Architecture Decision Records
  protocols/                # 稳定协议说明
tests/
  contract/
  integration/
  hardware/                 # 默认不在 CI 和普通 test 中运行
```

不要创建一个包含 driver、模型加载、控制 loop、记录和 UI 的“万能进程”。单机部署也应保持进程边界，只是可由同一个 launcher 启动。

## 5. 开发基线

### 5.1 语言与依赖

- 平台 Python 代码使用 Python 3.11+，完整类型标注，公共数据结构不得依赖无约束 `dict[str, Any]`。
- C++ adapter/runtime 使用其所属组件支持的标准；跨语言接口以 protobuf/C ABI 为边界，不暴露 STL 或 Python object。
- 配置必须经过结构化 schema 校验。未知字段默认报错，不能静默忽略拼写错误。
- 锁文件必须提交。新增重量级依赖前说明用途、替代方案、平台支持和 license。
- 禁止复制第三方仓库大段实现来规避依赖管理。

### 5.2 代码质量

- 名称使用英文，文档可使用中文；协议字段和日志 key 必须为稳定英文标识。
- 核心逻辑使用小而可组合的模块；控制 loop 中避免运行时分配、无限队列和无界重试。
- 捕获异常时必须分类、记录上下文并进入定义好的状态；禁止 `except Exception: pass`。
- 不允许静默 CPU fallback、静默 action 截断、静默 schema downgrade 或静默改用旧 plan。
- 任何 fallback 都要有 metric、event 和可测试的触发条件。
- 日志不得写入 token、凭据或未经策略允许的原始图像/任务文本。

### 5.3 标准命令

代码引入后，根目录必须提供以下稳定入口；底层工具可更换，但入口名称不变：

```bash
make format
make lint
make typecheck
make test
make test-integration
make proto
make mock-run
```

`make test` 不得访问真机、外网或云服务。`make mock-run` 应启动 mock robot、fake policy、recorder 和 Viewer bridge 的最小闭环。

## 6. 测试要求

每次变更按风险提供相称验证：

- 纯函数和状态机：单元测试覆盖正常、边界、超时、乱序和重复消息；
- wire/storage schema：golden fixture 与向后兼容测试；
- robot/model 插件：统一 contract test，不只测试单一实现；
- 异步执行：使用可控 fake clock，禁止依赖真实 `sleep` 验证时序；
- recorder：测试进程崩溃、磁盘满、上传中断和恢复，不得反压控制 loop；
- integration：至少覆盖 stale response 丢弃、chunk 原子替换、inference timeout、Viewer 断开、单臂故障触发双臂安全策略；
- 性能敏感路径：提交基准值与机器配置，至少报告 p50/p95/p99 和 deadline miss；
- 真机测试：放在 `tests/hardware/`，必须显式指定 robot、配置和人工 arming，不可由默认 CI 触发。

修复 bug 时先增加能复现问题的测试。无法自动化的真机行为，需要提交可重复的 test procedure 和记录 artifact。

## 7. 真机安全开发规则

- 所有新 driver、action adapter、控制器和 safety rule 先通过 mock/replay，再通过仿真或 dry-run，最后才允许低速真机验证。
- 真机命令必须显式指定 `--real-robot` 或等价开关，并打印 robot identity、配置 hash、安全限制和控制模式。
- 启动默认状态为 `DISARMED` 或 `PAUSED`；恢复执行必须是新鲜、可审计的 operator command。
- `Home` 是受约束运动请求，不等同于急停恢复；发生 safety fault 后不能自动 home。
- 软件不得屏蔽厂商错误、实体急停、限位或 watchdog。
- 修改 joint order、坐标系、四元数顺序、单位、normalization 或 gripper 语义属于高风险变更，必须有 contract fixture 和 migration note。

## 8. 可观测性要求

每个进程提供 health/readiness 状态，但 readiness 不能代表机器人安全可执行。核心 metric 至少包括：

- sensor/state age 与 drop count；
- inference queue、端到端 latency、server compute latency、timeout 和 stale response；
- timeline lookahead、buffer underrun、plan replacement 和 controller deadline miss；
- safety clamp/reject/fault，按有界 reason 分类；
- recorder backlog、spool 使用量、upload lag 和 dropped observability events；
- episode success、terminal reason、人工 override 和模型/配置版本。

metric label 必须低基数。`run_id`、`episode_id`、instruction 和任意异常文本放在 trace/event，不作为 Prometheus label。

## 9. 变更流程与完成定义

开始实现前：

1. 阅读本文件和相关设计/ADR；
2. 检查工作区已有修改，不覆盖无关内容；
3. 说明本次变更影响的实时路径、schema、安全与数据兼容性；
4. 若引入新边界或推翻既有决定，在 `docs/adr/` 添加 ADR。

一个变更只有在以下条件满足时才算完成：

- 功能、错误路径和降级行为已实现；
- 相应单元/contract/integration 测试通过；
- lint、typecheck 和相关基准通过；
- 配置示例、协议和用户文档同步更新；
- 新增 metric/event 可用于判断功能是否健康；
- 明确说明未运行的真机或硬件测试；
- 没有把实时风险转移给 UI、网络、数据库或 operator 猜测。

## 10. 参考项目的使用原则

- ManiUniCon：借鉴 multi-process、shared-memory、HAL 和数据采集结构；补强显式时间语义、双臂原子 plan、安全状态机与跨主机 contract。
- Realtime-VLA V2：借鉴异步 inference/execution overlap、delay alignment、time parameterization、local smoothing/MPC 和分阶段 trajectory logging。
- Embodied.cpp：作为 C++/GGUF `PolicyBackend` 候选，通过其服务协议适配，不在 edge runtime 中重写模型。
- Tether：借鉴 artifact parity、proof、shadow/canary、deadline 和 record/replay；首期只定义兼容的 artifact/validation 接口。
- Universal Viewer：作为 projection/read model 和 operator console；沿用 `RobotAdapter` 的可视化解耦，但控制请求必须走平台 command API 并等待 edge acknowledgement。

引用参考实现时在代码或 ADR 中固定上游 URL、commit 和许可证，不依赖未经固定的 `main` 行为。
