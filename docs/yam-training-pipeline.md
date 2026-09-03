# YAM 三个模型训练流程

本文记录从 YAM 遥操作数据到 Pi05、LingBot-VLA2、Xiaomi Robotics 1（XR-1）训练的完整命令。
训练命令在训练服务器执行；`/inspire/.../yam_fintune_data` 是共享训练盘，
`/home/ubuntu/manimux` 是代码 checkout。

## 0. 环境和数据

```bash
export CODE=/home/ubuntu/manimux
export DATA=/inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data
cd "$CODE"
```

当前螺丝刀数据集：

```text
$DATA/datasets/raw/assemble_the_screwdriver_20260825/
$DATA/datasets/lerobot/yam_assemble_screwdriver_20260825_v1/
```

数据采样率为 30 Hz；state/action 顺序为 `left arm 6 + left gripper 1 + right arm 6 +
right gripper 1 = 14 维`。原始记录保持绝对值，Pi05/LingBot loader 再按配置计算相对量和
normalization。

## 1. 数据转换

### 1.1 Pi05/LingBot：YAM episode 转 LeRobot

```bash
cd "$CODE"
envs/yam/.venv/bin/python scripts/datasets/convert_yam_to_lerobot.py \
  "$DATA/datasets/raw/assemble_the_screwdriver_20260825" \
  --repo-id yam_assemble_screwdriver_20260825_v1 \
  --output-root "$DATA/datasets/lerobot/yam_assemble_screwdriver_20260825_v1"
```

该脚本只封装视频、state、action 和 task，不做相邻帧差分，也不把 joint 改成 EE pose。

### 1.2 XR-1：YAM 数据转原生 JSON

```bash
cd "$CODE"
PYTHONPATH="$CODE/src" envs/xr1/.venv/bin/python \
  scripts/datasets/prepare_xr1_yam_dataset.py \
  --episodes "$DATA/datasets/raw/assemble_the_screwdriver_20260825" \
  --output "$DATA/xr1/yam_assemble_screwdriver_20260825_v1" \
  --action-length 30 --batch-size 16 \
  --config-name yam_assemble_screwdriver_20260825_v1
```

XR-1 的官方 `train.sh` 会按 `configs/data/<name>.yaml` 查找 data config，因此把刚生成的
config 放到它的配置目录：

```bash
mkdir -p "$CODE/XPolicyLab/policy/Xiaomi_Robotics_1/xiaomi_robotics_1/xr1/configs/data"
cp "$DATA/xr1/yam_assemble_screwdriver_20260825_v1/yam_assemble_screwdriver_20260825_v1.yaml" \
  "$CODE/XPolicyLab/policy/Xiaomi_Robotics_1/xiaomi_robotics_1/xr1/configs/data/yam_assemble_screwdriver_20260825_v1.yaml"
```

XR-1 使用 EE pose delta 数据接口，不能直接把 Pi05 的 14 维 joint LeRobot 目录当作 XR-1
数据集。

## 2. normalization

Pi05 的统计由训练 wrapper 在不存在时自动生成；已有匹配文件会复用：

```bash
cd "$CODE"
PI05_WORKSPACE="$CODE" bash .local/training/tools/train_pi05_yam_cluster.sh \
  prepare assemble-screwdriver-v1-s0-8xh100-15k
```

LingBot 转换并统计：

```bash
cd "$CODE/XPolicyLab/policy/LingBot_VLA2"
bash process_data.sh yam assemble_screwdriver yam_dual joint \
  "" "$DATA/datasets/raw/assemble_the_screwdriver_20260825"
```

XR-1 的 `prepare_xr1_yam_dataset.py` 会在输出目录生成匹配的 `norm_stats.json` 和 data
config；不要混用其他机器人统计。

## 3. Pi05 训练

下面是 8 卡、global batch 64、15000 steps、每 1000 steps 保存的示例：

```bash
cd "$CODE"
OPENPI_GPU_IDS=0,1,2,3,4,5,6,7 \
OPENPI_FSDP_DEVICES=8 OPENPI_BATCH_SIZE=64 \
OPENPI_NUM_TRAIN_STEPS=15000 OPENPI_SAVE_INTERVAL=1000 OPENPI_MAX_TO_KEEP=15 \
PI05_WORKSPACE="$CODE" bash .local/training/tools/train_pi05_yam_cluster.sh \
  train assemble-screwdriver-v1-s0-8xh100-15k
```

底层入口是 `XPolicyLab/policy/Pi_05/train.sh`，输出在 `$DATA/weights/finetuned/pi05/`。

## 4. LingBot-VLA2 训练

`LINGBOT_VLA2_ENABLE_RESUME=false` 确保从 foundation checkpoint 开始，不读取旧 optimizer
状态：

```bash
cd "$CODE/XPolicyLab/policy/LingBot_VLA2"
LINGBOT_VLA2_DATASET_PATH="$DATA/datasets/lingbot/yam-assemble-screwdriver" \
LINGBOT_VLA2_NORM_STATS_PATH="$DATA/datasets/lingbot/yam-assemble-screwdriver/norm_stats.json" \
LINGBOT_VLA2_MODEL_PATH="$CODE/checkpoints/pretrained/lingbot-vla-v2-6b" \
LINGBOT_VLA2_TOKENIZER_PATH="$CODE/checkpoints/pretrained/qwen3_vl_4b_processor" \
LINGBOT_VLA2_MICRO_BATCH_SIZE=1 LINGBOT_VLA2_GRAD_ACCUM_STEPS=64 \
LINGBOT_VLA2_MAX_STEPS=15000 LINGBOT_VLA2_SAVE_STEPS=1000 \
LINGBOT_VLA2_ENABLE_RESUME=false \
bash train.sh yam assemble_screwdriver yam_dual joint 0 0,1,2,3,4,5,6,7
```

wrapper 会把 `lingbotvla_cli.yaml`、`robot_config.yaml` 和 `norm_stats.json` 复制到 checkpoint。

## 5. Xiaomi Robotics 1 / XR-1 训练

XR-1 官方 wrapper 的 `action_type` 必须是 `ee`。对于当前 YAM 数据，不要再运行官方
`process_data.sh`（它是 RoboDojo HDF5 入口）；使用上面的 `prepare_xr1_yam_dataset.py`
后直接训练：

```bash
cd "$CODE/XPolicyLab/policy/Xiaomi_Robotics_1"
OUTPUT_DIR="$DATA/xr1/yam_assemble_screwdriver_20260825_v1" \
DATA_CONFIG_NAME=yam_assemble_screwdriver_20260825_v1 \
PRETRAINED_PATH="$CODE/checkpoints/pretrained_ckpt/model_states.pt" \
RUN_ROOT="$DATA/weights/finetuned/xr1/assemble-screwdriver-v1-s0-8xh100-15k" \
MAX_STEPS=15000 SAVE_INTERVAL=1000 ASYNC_TRAIN=true \
bash train.sh RoboDojo_real assemble_the_screwdriver yam_dual ee 0 0,1,2,3,4,5,6,7
```

## 6. TensorBoard

训练服务器启动持续发现新 run 的 dashboard：

```bash
cd "$CODE"
bash .local/training/dashboard/training_dashboard.sh start
```

本地打开 `http://127.0.0.1:16006`。日志按 Pi05、GR00T、LingBot action/native-depth、XR-1
分组；dashboard 不改变训练进程。

推理部署不属于训练流程：普通 runtime 使用 `manimux-...yaml`，RTC 使用 `rtc-...yaml`，
具体命令见根目录 README 和各模型 runbook。
