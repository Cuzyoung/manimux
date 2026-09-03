# SAPolicy + YAM 接入手册

## 当前结论

SAPolicy 经 **XPolicyLab WebSocket** 接入 ManiMux：

- `XPolicyLab/policy/SAPolicy`：薄封装，推理走本机 `~/sa/SpatialAlignPolicy`；
- ManiMux `worker: xpolicylab_ws`：通用 WS 传输；
- ManiMux `adapter: sapolicy_yam`：相机/内参、YAM FK/IK，以及绝对 EE wire → `joint_position ActionChunk`。

## 数据流

```text
YAM joints + named RGB frames
  -> sapolicy_yam: FK + K → additional_info.sapolicy
  -> xpolicylab_ws → XPolicyLab/policy/SAPolicy (SpatialAlign infer)
  -> packed_ee_wire (H,16) absolute EE
  -> sapolicy_yam: measured-state-seeded IK (fail → hold; Timeline 裁过期步)
  -> canonical left_arm/right_arm joint_position ActionChunk
  -> ManiMux Timeline / executor / Safety / Recorder / YAM driver
```

SpatialAlign 代码与依赖保留在独立仓库/环境；ManiMux 不 import torch。

## 启动

```bash
# SpatialAlign venv（加载 3cam_tcp.ckpt）
cd /home/ubuntu/sa/SpatialAlignPolicy && source .venv/bin/activate
pip install -e /home/ubuntu/manimux/XPolicyLab
python /home/ubuntu/manimux/scripts/servers/sapolicy_yam_server.py \
  --config /home/ubuntu/manimux/configs/sapolicy/yam/server/abc-bottles.yaml

# 相机必须 640x480；模型内部再裁到训练分辨率
envs/yam/.venv/bin/manimux-camera-server --config configs/sapolicy/yam/cameras.yaml
envs/yam/.venv/bin/manimux-viewer --robot yam --host 0.0.0.0 --port 8086
envs/yam/.venv/bin/manimux run --config configs/sapolicy/yam/infra/manimux-xpl.yaml
```

权重默认 `3cam_tcp.ckpt`（仓库根目录）。`cfg_file` 必须与该 checkpoint 的训练架构一致。

## 契约

| 项目 | 当前约束 |
|---|---|
| embodiment | 双臂 YAM，每侧 6 arm joints + 1 gripper |
| observation | 命名 RGB + 标定 `3x3 K` |
| DiT 原生动作 | 相对 `[pose18 \| grip2]`：`pos3+rot6d` ×2 + grip ×2 = 20D |
| wire → ManiMux | 绝对 `pos3+quat_xyzw+grip` ×2 = 16D |
| ManiMux action | 两组绝对关节，每组 7 维 |
| depth | 不支持 |
| IK | 失败 hold 上一步；过期步由 Timeline 裁切 |

Wire endpose 是 YAM `grasp_site` / ABC TCP，不再做 RoboTwin 的 0.12 m 前向偏移。

## 本机 mock（无真机）

无权重联调用 `server/mock.yaml`（`dry_run: true`，回放当前 EE）。

```bash
# 缺 mink/i2rt 时脚本会把 IK 退化成 hold-seed
python scripts/validation/sapolicy_yam_mock_run.py

# 已有 envs/yam 时也可拆成两进程
python scripts/servers/sapolicy_yam_server.py --config configs/sapolicy/yam/server/mock.yaml
envs/yam/.venv/bin/manimux run --config configs/sapolicy/yam/infra/mock.yaml
```

## 分层验证

1. 离线：wire / IK 契约与 adapter 单测
2. 本机 mock：`scripts/validation/sapolicy_yam_mock_run.py`
3. GPU：server `--check` 后真实 forward（非 dry_run）
4. 只读 preflight → 短真机
