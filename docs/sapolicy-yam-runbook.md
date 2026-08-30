# SAPolicy + YAM 接入手册

## 当前结论

SAPolicy 经 **XPolicyLab WebSocket** 接入 ManiMux：

- `XPolicyLab/policy/SAPolicy`：加载 SpatialAlign 权重并推理；
- ManiMux `worker: xpolicylab_ws`：通用 WS 传输；
- ManiMux `adapter: sapolicy_yam`：相机/内参、YAM FK/IK、0.12 m endpose，以及绝对 EE wire → `joint_position ActionChunk`。

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
# SpatialAlign venv
cd /path/to/SpatialAlignVLA && source .venv/bin/activate
pip install -e /path/to/manimux/XPolicyLab
# 编辑 configs/sapolicy/yam/server/abc-bottles.yaml：填 cfg_file，dry_run: false
python /path/to/manimux/scripts/sapolicy_yam_server.py \
  --config configs/sapolicy/yam/server/abc-bottles.yaml

# ManiMux yam venv
manimux run --config configs/sapolicy/yam/infra/manimux-xpl.yaml
```

`abc_tcp.ckpt` 的 Hydra `cfg_file` 必须与训练架构一致。

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

RoboTwin-compatible endpose：grasp site 沿局部 `+x` 前方 0.12 m（`endpose_forward_offset_m`）。

## 本机 mock（无真机）

`abc-bottles.yaml` 默认 `dry_run: true`，策略服务回放当前 EE（xyzw hold），不加载权重。

```bash
# 缺 mink/i2rt 时脚本会把 IK 退化成 hold-seed
python scripts/sapolicy_yam_mock_run.py

# 已有 envs/yam 时也可拆成两进程
python scripts/sapolicy_yam_server.py --config configs/sapolicy/yam/server/abc-bottles.yaml
envs/yam/.venv/bin/manimux run --config configs/sapolicy/yam/infra/mock.yaml
```

## 分层验证

1. 离线：wire / IK 契约与 adapter 单测
2. 本机 mock：`scripts/sapolicy_yam_mock_run.py`
3. GPU：server `--check` 后真实 forward（非 dry_run）
4. 只读 preflight → 短真机
