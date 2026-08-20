# Configuration layout

模型运行配置按 `configs/<model>/<embodiment>/{server,infra}/` 组织：

```text
configs/
  <model>/
    <embodiment>/
      server/
        <checkpoint-or-backend>.yaml
      infra/
        <experiment>.yaml
```

`server/` 只描述模型服务、checkpoint、norm stats 和模型原生采样参数；`infra/` 描述
ManiMux 的机器人、传感器、policy wire、执行器、Viewer 和记录。文件只创建实际存在的
角色，不再使用 `live` 后缀。名称直接表达实验差异，例如 Pi05 的 `base.yaml` /
`finetune.yaml`，以及统一的 `manimux.yaml` / `rtc.yaml` runtime。

命名轴不能混用：`server/{base,finetune}.yaml` 表示 checkpoint 变体，
`infra/{manimux,rtc}.yaml` 表示推理 runtime。base 权重加本体 projection stats 仍叫
`server/base.yaml`，不能在 infra 层新增 `zero-shot.yaml`。XR-1 只使用 XPolicy server；
不再提供平行的 native runtime 配置。

公共配置不属于任何模型，继续放在顶层：

- `cameras.yaml`：共享相机服务；
- `robots/`：机器人 driver 配置；
- `mock.yaml`：全 mock runtime；
- `maniunicon_meshcat.example.yaml`：通用仿真示例。

新增本体时不要复制模型目录，例如 Pi05 的 ALOHA 配置应放到
`configs/pi05/aloha/`，而不是创建新的 `pi05-aloha` 模型目录。
