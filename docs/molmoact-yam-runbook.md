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

## 4. ManiMux 真机 runtime

先确认 `can_left` 和 `can_right` 都是 `ERROR-ACTIVE`。执行后机械臂会按配置的
`start_duration_s` 移动到起始姿态，然后保持 `PAUSED`；在 Viewer 确认轨迹后再点
`Start / Resume`。

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/manimux run --config configs/molmoact-yam-live.yaml
```

真机速度和时间直接修改 `configs/molmoact-yam-live.yaml`：

- `policy.action_dt_s`：相邻 Policy 轨迹点的时间间隔（秒）；
- `execution.smooth.max_velocity`：关节最大速度（rad/s）；
- `execution.smooth.max_acceleration`：关节最大加速度（rad/s²）；
- `robot.options.start_duration_s` / `home_duration_s`：起始姿态和回零秒数。

当前 live 配置使用老师原 ManiMux 的 `action_dt_s: 0.05`；30 个轨迹点约覆盖
1.45 秒。关节限制为 `0.25 rad/s`、`0.5 rad/s²`。首次恢复真机时仍应只做
一次短距离、工作区清空且急停就绪的低速验证。CAN 处于 `ERROR-PASSIVE` 或
`BUS-OFF` 时禁止启动。

## 停止

启动 runtime 后会自动推理和执行，不需要在 Viewer 中点击 Start。达到
`run.max_steps` 后，机械臂自动回 Home，然后 runtime 退出。

正常运行时在 runtime 终端按一次 `Ctrl-C`，等待机械臂回零并退出；随后再停止
相机、模型服务和 Viewer。

Rollout 默认保存在 `data/run-*/episode-*`；未完整结束的记录保留 `.partial` 后缀。
