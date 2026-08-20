# GR00T N1.7 + YAM 运行手册

本文只覆盖 `gr00t-n1.7-yam` checkpoint。当前已完成静态契约和 adapter 离线测试，尚未
安装模型环境、完成 Cosmos 权限、GPU forward、相机或真机验证。

## 模型契约

```text
checkpoints/pretrained/gr00t-n1.7-yam/
  model-00001-of-00002.safetensors
  model-00002-of-00002.safetensors
  model.safetensors.index.json
  processor_config.json
  statistics.json
```

- cameras：`base_view`、`left_wrist_view`、`right_wrist_view`；
- state/action：`left_arm`、`left_gripper`、`right_arm`、`right_gripper`；
- 输出：`16 x 14` absolute joint positions；
- server：`configs/gr00t-n17/yam/server/finetune.yaml`；
- ManiMux infra：`configs/gr00t-n17/yam/infra/manimux.yaml`。

adapter 使用 checkpoint 自己的 `new_embodiment` modality config，不能换回 XPolicy 原来
的 ARX config。

## 1. 模型环境

GR00T 使用独立 Python 3.10 环境。首次安装由操作者执行：

```bash
cd /home/ubuntu/manimux/XPolicyLab/policy/GR00T_N17
bash install.sh
```

processor 依赖 gated `nvidia/Cosmos-Reason2-2B`。需要先完成 Hugging Face 授权，或把
`cosmos_model_path` 指向已经下载的本地目录。

## 2. 离线检查

不加载模型：

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/gr00t_yam_server.py --check
```

## 3. 模型服务

环境和 Cosmos 权限准备完成后：

```bash
cd /home/ubuntu/manimux
XPolicyLab/policy/GR00T_N17/gr00t_n17/.venv/bin/python \
  scripts/gr00t_yam_server.py --config configs/gr00t-n17/yam/server/finetune.yaml
```

未完成一次真实 GPU forward 前，不进入相机和真机阶段。

## 4. 相机、Preflight 与真机

相机和 Viewer 命令与其他 YAM 模型相同：

```bash
envs/yam/.venv/bin/manimux-camera-server --config configs/cameras.yaml
envs/yam/.venv/bin/manimux-viewer --robot yam --host 0.0.0.0 --port 8086
```

当前还没有 GR00T 专用真实 preflight 证据。完成模型 forward 和 preflight 后，真机入口才是：

```bash
envs/yam/.venv/bin/manimux run --config configs/gr00t-n17/yam/infra/manimux.yaml
```

该命令会连接 CAN 并移动机械臂；当前阶段不要把“配置可加载”等同于“可以直接上真机”。
