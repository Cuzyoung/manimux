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
`finetune.yaml` 和 `manimux.yaml` / `stretched-50ms.yaml`，以及 Xiaomi XR-1 的
`native.yaml` / `xpolicy.yaml` / `xpolicy-rtc.yaml`。

公共配置不属于任何模型，继续放在顶层：

- `cameras.yaml`：共享相机服务；
- `robots/`：机器人 driver 配置；
- `mock.yaml`：全 mock runtime；
- `maniunicon_meshcat.example.yaml`：通用仿真示例。

新增本体时不要复制模型目录，例如 Pi05 的 ALOHA 配置应放到
`configs/pi05/aloha/`，而不是创建新的 `pi05-aloha` 模型目录。
