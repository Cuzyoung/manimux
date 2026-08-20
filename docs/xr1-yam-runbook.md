# XR-1 + YAM 运行手册

和 MolmoAct / ABC 一样的四个服务，只有第 1 步（模型服务）和第 4 步的配置文件不同。

> **先读这一节再上真机。** XR-1 是唯一一个输出末端笛卡尔增量的模型，而且我们手上
> 这份权重是官方的**微调起点**，不是可直接部署的策略。零样本的限制写在文末。

## 配置位置

```text
Native ManiMux: configs/xiaomi-xr1/yam/infra/native.yaml
```

XPolicy 和 RTC 路线使用同一模型目录下的 `xpolicy-*` 配置，单独见
[Xiaomi XR-1 + YAM](xiaomi-xr1-yam-runbook.md)。以后其他本体放在
`configs/xiaomi-xr1/<embodiment>/`。

## 0. 一次性准备

XR-1 要独立的 venv（torch 2.8 / CUDA 12.6 + 配套的 flash-attn，和 MolmoAct 的
cu121、ABC 的 cu128 都装不到一起）：

```bash
cd /home/ubuntu/manimux
uv venv envs/xr1/.venv --python 3.12
uv pip install --python envs/xr1/.venv/bin/python torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu126
uv pip install --python envs/xr1/.venv/bin/python -e ".[xr1-yam]"
uv pip install --python envs/xr1/.venv/bin/python flash-attn==2.8.3 --no-build-isolation
```

权重和 Qwen3-VL 的 processor 都已经在
`checkpoints/pretrained/xiaomi/`，不需要联网：模型里 1135 个张量（5.50 B 参数）
全部来自 `model_states.pt`，Qwen3-VL 只读本地的 `config.json`。

## XR-1 是怎么推理的

```
三相机 RGB                        左右臂关节 (14,)
  ego / 左腕 / 右腕                      │
        │  等比缩放到 32 的倍数            │  compose_state: 填进 60 维槽位
        │  面积上限 160k 像素              │  分位数归一化到 [-1,1]
        ▼                                ▼
   Qwen3-VL 4B (bf16, flash-attn2)  ┌──────────────┐
        │  prompt 里带 # Ego View /   │  XR-1 5.5B   │
        │  # Left-Wrist View 等标题   │  VLM + DiT   │
   任务指令 ──────────────────────►  │   (MoT)      │
                                    └──────┬───────┘
                                           │ 5 步 Euler 流积分
                                           ▼
                                  (30, 60) 归一化动作
                                           │ 反归一化 + action_mask
                                           ▼
                        末端笛卡尔增量（相对当前末端坐标系）
                                           │
                        ══════ 以下在 adapter 里做，不在服务端 ══════
                                           │
                   FK(当前关节) ──► 锚点位姿 (R₀, p₀)
                                           │
                   绝对目标 = p₀ + R₀·Δp ,  R₀·aa2rotm(Δaa)
                                           │
                   IK（i2rt / mink，逐步以上一步为种子）
                                           ▼
                    左臂 (30,7) ◄── 关节轨迹 ──► 右臂 (30,7)
```

实测：模型加载约 87 秒，单次推理 **121 ms**；IK 单次 0.4 ms，一个 chunk 60 次
求解约 25 ms。

**为什么 FK/IK 放在 adapter 而不是服务端**：把笛卡尔目标变成关节是机器人的知识，
不是模型的知识。换一台机械臂只要换 `policy.options.kinematics`，服务端不动。
运动学实现在 `manimux/kinematics/`，任何输出末端位姿的模型都能复用。

`prompt` 直接进 Qwen3-VL，用自然语言写任务即可（不像 ABC 要贴合训练措辞）。

## 1. XR-1 模型服务

```bash
cd /home/ubuntu/manimux
envs/xr1/.venv/bin/manimux-xr1-server \
  --host 127.0.0.1 \
  --port 8400 \
  --checkpoint checkpoints/pretrained/xiaomi/model_states.pt \
  --device cuda:0
```

看到 `Warmup OK` 和 `Uvicorn running` 后继续。启动时会打印一条归一化统计量来源的
警告，那不是错误，是提醒（见文末）。

## 2. 相机服务

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/manimux-camera-server --config configs/cameras.yaml
```

## 3. Viewer

```bash
cd /home/ubuntu/manimux
source .venv/bin/activate
manimux-viewer --robot yam --host 0.0.0.0 --port 8086
```

## 4. ManiMux runtime

先确认 CAN（检查和重开见 [CAN 总线](can-bus.md)）：

```bash
for c in can_left can_right; do printf '%s: ' "$c"; ip -details link show "$c" | grep -o 'ERROR-ACTIVE\|ERROR-PASSIVE\|BUS-OFF'; done
```

**这一条会让机械臂真的动**，清空工作区、急停在手：

```bash
envs/yam/.venv/bin/manimux run --config configs/xiaomi-xr1/yam/infra/native.yaml
```

## 三个 policy 的对照

| | MolmoAct | ABC | XR-1 |
|---|---|---|---|
| 端口 | 8202 | 8300 | 8400 |
| `policy.worker` / `adapter` | `molmoact_http` / `molmoact_yam` | `abc_http` / `abc_yam` | `xr1_http` / `xr1_yam` |
| 动作空间 | 关节绝对位置 | 关节绝对位置 | **末端笛卡尔增量 → IK** |
| chunk | 30 × 14 | 30 × 14 | 30 × 60（有效 18 列） |
| 指令编码 | VLM prompt | CLIP 文本向量 | VLM prompt |
| 推理延迟 | ~240 ms | ~170 ms | ~121 ms |
| venv | `envs/yam` (cu121) | `envs/abc` (cu128) | `envs/xr1` (cu126) |

机器人、相机、Viewer 三层三者共用，一行代码都没改。

## 零样本的已知限制

**这份权重是 `Xiaomi-Robotics-1-5B`，官方定位是「微调起点」**，不是能直接部署的
策略。官方部署文档里那个真机通用权重 `XiaomiRobotics/Xiaomi-Robotics-1` 没有公开
（HTTP 401）。已发布的 RoboCasa / VLABench 三个权重是仿真环境的单臂微调版，驱动
不了双臂 YAM。

**上游不发归一化统计量。** 模型跑起来必须要 `mean/std/q01/q99`，而仓库里唯一
一份来自 5 个 episode 的洗衣机 demo（另一台机器人）。目前打包在
`src/manimux/integrations/xr1_yam/norm_stats/washer_demo.json`，用
`--norm-stats` 可以换。实测影响：

| | 情况 |
|---|---|
| 12 个手臂关节 | 落在 demo 的分位数区间内，归一化结果合理（−0.86 ~ +0.50） |
| 右臂 j2 | 略微越界，饱和到 −1.0 |
| 末端位移尺度 | std 约 0.051 m / 30 步，量级正常 |
| **左右夹爪** | **坏的**：demo 的夹爪是弧度（区间 `[-3.05,-0.05]` / `[-7.14,-0.06]`），YAM 是归一化 `[0,1]`。输入端永远饱和到 +1，模型看不见真实开合；输出端增量 std 达 0.87，会把 `[0,1]` 的夹爪打飞 |

所以：**手臂运动可能出得来，夹爪一定不对。** 第一次上真机务必低速、
清空工作区、急停在手；在拿到 YAM 微调权重和对应统计量之前，不要指望它完成需要
精细抓取的任务。

## 计算 YAM 的归一化统计量

上游的 `deploy.py` 从**训练输出目录的 `config.py`** 里读 `mean/std/q01/q99`，所以统计量
永远和权重来自同一次训练。我们手上只有裸权重，没有那个目录，所以要自己算一份。

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python -m manimux.integrations.xr1_yam.compute_norm_stats \
  --episodes /home/ubuntu/yam-abc-reproduce/data/episodes \
  --out src/manimux/integrations/xr1_yam/norm_stats/yam.json
```

已经用 60 个 episode / 23994 个窗口算好，产物在
`src/manimux/integrations/xr1_yam/norm_stats/yam.json`。有新数据重跑一遍即可，不用改代码。

脚本严格照上游 `json_dataset.py` 的构造方式做增量，末端位姿由 `manimux.kinematics` 的
FK 算出（和录制数据时用的 FK 逐位相同）：

```
rotm  = proprio.{arm}_ee_rotm[t]
pos   = proprio.{arm}_ee_pos[t]
dpos  = rotm.T @ (action.{arm}_ee_pos[t:t+30] - pos)
daa   = rotm2aa(rotm.T @ action.{arm}_ee_rotm[t:t+30])
dgrip = action.{arm}_gripper[t:t+30] - proprio.{arm}_gripper[t]
```

两个必须注意的点：

- **动作的 mean/std 是逐步的 `(30, 60)`，不能拍平成 `(60,)`。** 第 1 步位移是毫米级、
  第 30 步是厘米级，共用一组 std 会让近端在归一化空间接近 0，模型学不动近端 ——
  而近端正是马上要执行的几步。上游 `validate_stats` 会强制检查形状。状态没有
  horizon，所以 `q01/q99` 是 `(1, 60)`。
- **恒定不动的维度要显式置零。** `std ≈ 0` 除下去会炸。分位数路线的做法是让
  `q99 == q01`，服务端会把该维直接置零；脚本对 waist、底盘速度和保留位就是这么处理的
  （YAM 没有这些自由度）。

### 用它启动

`yam.json` 已经是服务的默认值，第 1 步那条命令就在用它。要换回上游的洗衣机 demo
统计量做对照，才需要显式指定：

```bash
envs/xr1/.venv/bin/manimux-xr1-server --host 127.0.0.1 --port 8400 \
  --checkpoint checkpoints/pretrained/xiaomi/model_states.pt \
  --norm-stats src/manimux/integrations/xr1_yam/norm_stats/washer_demo.json
```

实测改善：模型输出幅度从 16σ 回到 4σ，右夹爪输入从 1 个值恢复到 507 个值，右臂 IK
步间跳变从 57° 降到 21°。但同输入的采样一致性只从 −0.016 升到 +0.269（有把握的策略
应接近 +1）—— **算统计量只能修好单位换算，修不好「模型没在 YAM 数据上训练过」**，
所以上真机仍要低速、急停在手。
