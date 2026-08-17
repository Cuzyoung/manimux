# MolmoAct + YAM 运行手册

运行前清空机械臂工作区并准备好急停。以下四个服务分别在四个终端中按顺序启动。

## 1. MolmoAct 模型服务

```bash
cd /home/ubuntu/manimux
source envs/yam/.venv/bin/activate
manimux-molmoact-server \
  --host 127.0.0.1 \
  --port 8202 \
  --repo-id allenai/MolmoAct2-BimanualYAM \
  --device cuda:0 \
  --dtype bfloat16
```

看到 `Warmup OK` 和 `Uvicorn running` 后继续。

## 2. RealSense 相机服务

```bash
cd /home/ubuntu/manimux
source envs/yam/.venv/bin/activate
cd src/manimux/integrations/molmoact_yam
manimux-molmoact-camera --config configs/molmoact_yam_left.yaml
```

确认三台相机均已打开，并看到 `REP bound` 和 `PUB bound`。

## 3. Viewer

```bash
cd /home/ubuntu/manimux
source .venv/bin/activate
manimux-viewer --robot yam --host 0.0.0.0 --port 8086
```

浏览器打开 `http://localhost:8086`。

## 4. MolmoAct/YAM integration

执行后机械臂会移动到配置中的起始姿态。

```bash
cd /home/ubuntu/manimux
source envs/yam/.venv/bin/activate
cd src/manimux/integrations/molmoact_yam
LEFT_CFG=configs/molmoact_yam_left.yaml
RIGHT_CFG=configs/molmoact_yam_right.yaml
manimux-molmoact-yam \
  --left-config-path "$LEFT_CFG" \
  --right-config-path "$RIGHT_CFG" \
  -n 1
```

## 停止

先在 integration 终端按一次 `Ctrl-C`，等待 rollout 保存、机械臂回零并退出；随后再停止相机、模型服务和 Viewer。

Rollout 默认保存在 `data/molmoact_yam_eval_runs/`。
