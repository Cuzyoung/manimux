# LingBot-VLA2 + YAM 运行手册

本文只覆盖本地 `lingbot-vla-v2-6b`。当前状态是**部署契约阻塞**，不是可启动的真机模型。
因此当前没有 `configs/lingbot-vla2/yam/`；拿到完整模型契约后才创建该目录。

## 已确认

- 六个 safetensors shard 已下载完整；
- 本地 V2 是 Qwen3-VL + MoE action expert；
- `state_proj` 输入和 `action_out_proj` 输出使用 55 维统一表示；
- XPolicy 当前 `LingBot_VLA` 是旧 Qwen2.5 adapter，不能加载这份 V2 权重。

## 缺失契约

- YAM 的 14 维 joint/gripper 对应 55 个槽位中的哪些位置；
- action horizon 和训练控制频率；
- 55 维输出中 joint、EE 或其他信号的具体语义；
- YAM 对应 norm stats；
- V2 正式推理和 flow sampling 接口。

缺少这些信息时不能伪造 server/infra/RTC config。

## 可重复审计

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/lingbot_vla2_yam_audit.py
```

脚本故意以状态码 `2` 退出，含义是“权重存在，但部署契约不完整”。

## 后续正确路径

1. 获得 LingBot-VLA2 官方推理源码或公开兼容实现；
2. 获得 YAM fine-tune 的 55-slot embodiment mapping、horizon、频率和 stats；
3. 在 XPolicy 新增正式 `policy/LingBot_VLA2/` adapter；
4. 先做真实 checkpoint GPU forward；
5. 再创建独立 ManiMux config；
6. 只有 conditioning 进入原生 sampler 后才创建 RTC config。

当前没有 LingBot-VLA2 真机命令，这是有意的 fail-closed 状态。
