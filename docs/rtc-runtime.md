# RTC runtime（实时分块）

论文：*Real-Time Execution of Action Chunking Flow Policies*（Physical Intelligence，
arXiv:2506.07339）。用 `execution.runtime` 选择：

```yaml
execution:
  runtime: manimux   # 默认：老师原版 runtime，行为逐字节不变
  # runtime: rtc
```

## 它解决什么

策略一次输出 `H` 步，但推理要花 `d` 个控制周期。

- **同步执行**（等推理回来）：机械臂停顿，且训练时没见过这种停顿。
- **朴素异步**（算好就换）：新 chunk 是从多峰分布里重新采样的，可能选了完全不同的
  策略（从障碍上方绕 vs 下方绕），接缝处动作跳变。
- **RTC**：把它当成 **inpainting** 问题。生成新 chunk 时，把旧 chunk 未执行的部分
  作为条件：前 `d` 步**冻结**（那些时刻必然已经执行掉了），之后权重指数衰减到 0。
  连续性来自引导，不是来自下游滤波。

## 权重公式（论文 Eq. 5）

```
        ⎧ 1                              i < d
W_i  =  ⎨ c_i · (e^{c_i} − 1)/(e − 1)     d ≤ i < H − s
        ⎩ 0                              i ≥ H − s

c_i = (H − s − i) / (H − s − d + 1)
```

`s` = 本轮已执行步数，`d` = 保守的延迟预测。可行性约束 **`d ≤ s ≤ H − d`**。

引导本身（PiGDM，含 VJP）在**模型服务端**：MolmoAct 是
`molmoact_rtc.py`，ABC 是 `abc_minimal/dit.py::sample_actions_pi_rtc`。runtime 只负责
算出条件和权重发过去。

## 配置

```yaml
execution:
  runtime: rtc
  rtc:
    # 开始下一次推理前至少执行多少步；实际执行步长是 max(min_execute_steps, d)。
    # 留空则默认 H/2。
    min_execute_steps: 15
    # 第一次推理还没有实测延迟时用的初值（控制步数）。
    initial_delay_steps: 4
    # 延迟预测取最近这么多次实测值的**最大值**（保守，宁可多冻结几步）。
    delay_buffer_size: 10
    # 引导权重上限。在 t=1/steps 处按 (t²+(1−t)²)/(t(1−t)) 取值：
    # 5 步采样 → 约 4.25，10 步 → 约 9.1。太大会抖，太小引导太弱。
    beta: 5.0
    # 每个控制步允许的最大关节变化（弧度）。见下方"为什么不走 executor"。
    max_joint_step: 0.02
```

## 为什么 RTC 不走 smooth executor

RTC 的保证成立的前提是：**真正发给机器人的动作，就是下一个 chunk 所条件化的那个动作**。
下游任何依赖状态的钳位、低通滤波或插值都会让这个前提失效 —— 模型以为机器人会走
`A_cur[i]`，实际走的是滤波后的值，接缝对不上。

所以 RTC 把速度包络**折进 chunk 本身**（`project_chunk_to_joint_speed`），
而且推理在途时冻结不动，保证被条件化的那份 chunk 不会在denoise 过程中变。
夹爪列不参与限速。

这也是为什么 RTC 是**并列的 runtime 而不是一个 executor 选项** —— 两者对"chunk 是什么"
的定义就不同：ManiMux runtime 把它当参考轨迹去跟踪，RTC 把它当指令流本身。

## 支持情况

| policy | 状态 |
|---|---|
| MolmoAct | ✅ 服务端 PiGDM 引导已接（RTC 时自动禁用 CUDA graph，两者不兼容） |
| ABC | ✅ `sample_actions_pi_rtc` |
| XR-1 | ⚠️ 只有硬前缀变体（`action_prefix`），软引导未接；用 RTC 时条件会被服务端忽略 |

## 诊断

`events.jsonl` 里的 RTC 相关事件：

| 事件 | 含义 |
|---|---|
| `inference_submitted` | 带 `executed_steps`(s) 和 `forecast_delay`(d) |
| `plan_accepted` | 带 `measured_delay`（新 chunk 实际从第几步接上）和 `forecast_delay` |
| `rtc_chunk_projected` | 速度包络改写了多少行、最大修正量 |
| `rtc_chunk_underrun` | 替换 chunk 没按时到，保持当前位姿（不重放最后一个目标） |
| `rtc_delay_infeasible` | 推理慢过半个 chunk，`d ≤ s ≤ H−d` 无解，停止提交 |

出现 `rtc_delay_infeasible` 说明推理太慢或 horizon 太短：降低采样步数，或换更长的 chunk。
