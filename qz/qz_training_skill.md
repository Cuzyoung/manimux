# QZ 训练提交技能

这份文档只记录 QZ 训练任务的登录、资源核对、payload 生成和提交流程，供后续 agent 复用。
它不保存 token、密码或完整实验 payload，也不自动提交任务。

## 1. 路径约定

必须先分清本地和远端：

```text
本地代码 checkout：/home/ubuntu/manimux
远端登录别名：    localhost-3338
远端训练根目录：  /inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data
远端代码目录：    $YAM_TRAIN_ROOT/operate/manimux
远端数据目录：    $YAM_TRAIN_ROOT/datasets
远端权重目录：    $YAM_TRAIN_ROOT/weights
远端日志目录：    $YAM_TRAIN_ROOT/runs
```

远端路径只能放在 `ziyang/yam_fintune_data` 下面。不要把训练 checkout、权重或日志写到
`/home/ubuntu`，也不要默认使用历史目录 `operate/manimux-training` 或
`operate/manimux-training-clean`。如果远端尚未完成统一 checkout，先确认实际目录，再设置
`WORKSPACE`，不要在 payload 里猜路径。

远端只读检查：

```bash
ssh localhost-3338 '
  ROOT=/inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data
  cd "$ROOT/operate/manimux"
  git rev-parse --show-toplevel
  git rev-parse HEAD
  git -C XPolicyLab rev-parse HEAD
  nvidia-smi -L
'
```

## 2. QZ CLI 登录

QZ CLI 默认位置是 `/home/ubuntu/.local/bin/qz`，建议先确认版本和帮助：

```bash
/home/ubuntu/.local/bin/qz --help
```

首次登录或 token 过期：

```bash
qz config set server https://qz.sii.edu.cn
qz config set auth_url https://keycloak-inspire-prod.sii.edu.cn
qz login
qz config get
```

登录 token 由 QZ CLI 本地保存。不要把 `INSPIRE_TOKEN`、密码或配置文件复制到仓库，也不要
写入 JSON payload。

如果 API schema 缓存异常，可清理本地 discovery 缓存后重新加载：

```bash
rm -f ~/.config/inspire-cli/cache/discovery-*.json
qz spec
```

## 3. 当前身份和项目

这些是已核对过的 Ziyang 项目信息：

```text
project_name: intern-ziyang
project_id:   project-f34ef3ad-b8b5-4c42-bd6e-47b6ed5e6020
user_id:      user-3ea382df-a19b-413a-931e-e8b31312ac0e
workspace_id: ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6
```

提交前只读确认当前账号和项目，不要只相信历史文件：

```bash
qz user GetUserDetail
qz project GetProjectForPage --data \
  '{"page":1,"page_size":100,"filter":{"name":"intern-ziyang"}}'
```

必须确认返回的 user/project ID 与上面一致。项目状态必须是平台允许创建训练任务的状态；
如果状态不是可提交状态，停止，不要通过脚本绕过。

## 4. 计算资源组和 GPU 规格

后续任务指定使用的逻辑计算资源组是：

```text
logic_compute_group_id = lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e
```

旧实验使用过的 `lcg-79b2ad0e-a375-43f3-a0b1-b4ce79710fd7` 不要再用于新任务。

`spec_id` 不能凭记忆填写。它和 GPU 数量、集群资源组绑定，提交前必须从当前 workspace 的
schedule 配置查询：

```bash
qz train GetTrainScheduleConfig \
  --set workspace_id=ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6
```

从返回的 `predef_train_spec` 中选择：

1. `logic_compute_group_ids` 包含 `lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e`；
2. `gpu_count` 与任务要求一致；
3. 名称/架构确实是 H100（若任务要求 H100）；
4. 规格仍处于可用状态。

不要把历史 4 卡 spec `8a53ac21-299a-4dee-85e9-9c04a544cf8d` 自动套到新资源组，除非上面的
查询明确证明它仍然有效。

## 5. 提交前检查

训练命令在远端执行，代码、数据、权重和日志都使用远端绝对路径。一个任务的基本变量应
明确写出：

```bash
ROOT=/inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data
WORKSPACE=$ROOT/operate/manimux
GPU_IDS=0,1,2,3,4,5,6,7       # 8 卡示例；按 spec 修改
```

提交前手动检查：

```bash
ssh localhost-3338 "
  ROOT=$ROOT
  test -d \"\$ROOT/operate/manimux\"
  test -d \"\$ROOT/datasets\"
  test -d \"\$ROOT/weights\"
  nvidia-smi -L
"
```

训练脚本必须使用统一 checkout 中的路径，例如：

```text
$ROOT/operate/manimux/scripts/training/train_pi05_yam_cluster.sh
$ROOT/operate/manimux/scripts/training/train_lingbot_vla2_yam_cluster.sh
$ROOT/operate/manimux/scripts/training/train_xr1_yam_cluster.sh
$ROOT/operate/manimux/scripts/training/train_gr00t_n17_yam_cluster.sh
```

如果这些文件不存在，不要提交；先完成远端 checkout 同步。不要在 QZ command 里引用本地
`/home/ubuntu/manimux`。

## 6. CreateJob payload 最小结构

先通过 schema 查看当前 API，而不是照抄旧 JSON：

```bash
qz schema train.CreateJob
```

payload 至少需要这些字段（具体字段类型以当前 schema 为准）：

```json
{
  "name": "pi05-yam-assemble-8xh100-15k-YYYYMMDD-v1",
  "logic_compute_group_id": "lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e",
  "project_id": "project-f34ef3ad-b8b5-4c42-bd6e-47b6ed5e6020",
  "workspace_id": "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6",
  "framework": "pytorch",
  "task_priority": 10,
  "auto_fault_tolerance": false,
  "enable_notification": false,
  "name": "<unique-job-name>",
  "command": "<remote bash command>",
  "framework_config": [
    {
      "image": "docker.sii.shaipower.online/inspire-studio/pytorch:25.06-py3",
      "image_type": "SOURCE_OFFICIAL",
      "instance_count": 1,
      "shm_gi": 200,
      "spec_id": "<queried-spec-id>"
    }
  ]
}
```

注意：上面是模板，不是可直接提交的任务。`spec_id`、任务名、训练步数、GPU 数和脚本路径
必须按本次实验填写。

## 7. 生成、dry-run、正式提交

推荐先把 payload 保存到本地临时目录或 `qz/generated/`（该目录不应提交），然后检查 JSON：

```bash
python -m json.tool /tmp/pi05_job.json >/dev/null
qz train CreateJob --dry-run --data @/tmp/pi05_job.json
```

`--dry-run` 只打印请求，不会创建任务。确认以下内容后，才允许正式调用：

- user/project/workspace 身份正确；
- logic compute group 是 `lcg-71...`；
- 当前查询到的 spec 与 GPU 数一致；
- command 使用远端 `ziyang/yam_fintune_data` 路径；
- 数据集、基础权重、tokenizer、norm stats 都已存在；
- 任务名是新名字，不会覆盖已有输出；
- 没有 `WANDB_API_KEY` 等秘密被写进 command。

正式提交命令：

```bash
qz train CreateJob --data @/tmp/pi05_job.json
```

QZ 的 `CreateJob` 会改变外部状态。任何新任务或失败后的重试，都必须先展示完整 payload，
确认本次提交的项目、资源组、spec、路径和命令，再执行正式 CreateJob；不要让脚本无条件自动
重试或批量重复提交。

## 8. Pi05 / LingBot / XR-1 command 约定

command 只负责在远端启动对应官方训练入口，模型训练细节由各模型自己的 config 控制。
示意：

```bash
set -euo pipefail
ROOT=/inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data
WORK=$ROOT/operate/manimux
RUN=<unique-run-name>
mkdir -p "$ROOT/runs/pi05" "$ROOT/weights/finetuned/pi05"
YAM_TRAIN_ROOT="$ROOT" \
OPENPI_GPU_IDS=0,1,2,3,4,5,6,7 \
OPENPI_FSDP_DEVICES=8 \
OPENPI_NUM_TRAIN_STEPS=15000 \
OPENPI_SAVE_INTERVAL=1000 \
bash "$WORK/scripts/training/train_pi05_yam_cluster.sh" train "$RUN"
```

LingBot 和 XR-1 使用各自的 `LINGBOT_VLA2_*` / `XR1_*` 环境变量以及对应训练 wrapper。
不要把 Pi05 的 14 维 joint 数据目录直接作为 XR-1 的 EE 数据目录；XR-1 必须使用它自己的
转换产物和 data config。

三个 wrapper 的核心启动形式如下（实际 GPU 数、run 名和步数按实验 payload 填写）：

```bash
# LingBot-VLA2
YAM_TRAIN_ROOT="$ROOT" \
LINGBOT_VLA2_GPU_IDS=0,1,2,3,4,5,6,7 \
LINGBOT_VLA2_MAX_STEPS=15000 LINGBOT_VLA2_SAVE_STEPS=1000 \
bash "$WORK/scripts/training/train_lingbot_vla2_yam_cluster.sh" train "$RUN"

# Xiaomi Robotics 1 / XR-1
YAM_TRAIN_ROOT="$ROOT" \
XR1_GPU_IDS=0,1,2,3,4,5,6,7 \
XR1_MAX_STEPS=15000 XR1_SAVE_INTERVAL=1000 \
bash "$WORK/scripts/training/train_xr1_yam_cluster.sh" train "$RUN"

# GR00T N1.7（若本次实验包含 GR00T）
YAM_TRAIN_ROOT="$ROOT" \
GR00T_GPU_IDS=0,1,2,3,4,5,6,7 \
GR00T_MAX_STEPS=15000 GR00T_SAVE_STEPS=1000 \
bash "$WORK/scripts/training/train_gr00t_n17_yam_cluster.sh" train "$RUN"
```

常用远端数据/权重约定：

```text
Pi05/LingBot LeRobot：$ROOT/datasets/lerobot/<dataset_name>
XR-1 转换数据：       $ROOT/datasets/xr1/<dataset_name>
基础模型：             $ROOT/weights/base/
微调输出：             $ROOT/weights/finetuned/<model>/<run_name>
训练日志：             $ROOT/runs/<model>/<run_name>.log
```

## 9. 查看任务和日志

```bash
qz train ListJobs --workspace-id ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6 --page-size 100
qz train GetJob --job-id <job-id>
qz train GetJobLog --job-id <job-id> --page-size 200
qz train GetTaskMetric --job-id <job-id> --metric-mode REAL_USED
```

拿到 job ID 后，优先通过 QZ 的当前 schema/help 查询详情命令：

```bash
qz train --help
qz schema train.GetJob
```

远端日志：

```bash
ssh localhost-3338 \
  'tail -f /inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data/runs/pi05/<run>.log'
```

必须区分：`CreateJob 成功`、`任务排队`、`分配 GPU`、`训练进程启动`、`出现 smoke checkpoint`、
`正式训练达到目标步数`。只有最后两项之后，才能说训练确实开始并产生了有效 checkpoint。

## 10. 停止和失败处理

先确认 job ID 和状态，再停止排队中的错误任务：

```bash
qz train --help
qz train StopJob --job-id <job-id>
```

不要停止正在正常训练的任务。失败重试前先判断原因：

- `AUTH` / token：重新 `qz login`；
- `INVALID_ARGUMENT`：对照 `qz schema train.CreateJob` 检查字段；
- 资源不可用：重新查询 schedule/spec，不要继续使用旧 spec；
- 远端路径不存在：修正 checkout 或 payload 路径；
- OOM：先记录实际 GPU 数、micro batch、gradient accumulation 和模型 config，不能只重复提交；
- 训练脚本异常：在远端用相同 command 做只读/离线 preflight，修好后再提交新 job。

不要加入“finished 拦截”、哈希门禁或无限自动重试来掩盖训练问题。训练脚本应直接报告
真实错误，任务状态由 QZ 和训练日志判断。

## 11. 常用命令速查

```text
qz login                                      登录/刷新 token
qz config get                                 查看 CLI 配置
qz spec                                       查看 API 服务
qz schema train.CreateJob                     查看 CreateJob schema
qz user GetUserDetail                         查看当前用户信息
qz project GetProjectForPage ...              查看项目
qz train GetTrainScheduleConfig ...           查询 workspace 资源/spec
qz train CreateJob --dry-run --data @job.json 模拟请求
qz train CreateJob --data @job.json           正式创建任务
qz train ListJobs                              查看任务列表
qz train GetJob --job-id <id>                  查看任务详情
qz train GetJobLog --job-id <id>               查看平台日志
qz train GetTaskMetric --job-id <id>           查看资源指标
qz train StopJob --job-id <id>                 停止指定任务
ssh localhost-3338 '...'                       检查远端 checkout/数据/日志
```

每次任务都应保存：job name、job ID、提交时间、project/workspace、logic compute group、spec、
远端 checkout commit、XPolicyLab commit、数据集路径、训练 config 和日志路径。不要保存 token。
