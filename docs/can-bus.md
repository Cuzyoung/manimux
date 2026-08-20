# CAN 总线：检查与重开

YAM 的两条臂各走一条 CAN 总线（`can_left` / `can_right`）。跑真机前必须两条都是
`ERROR-ACTIVE`；`ERROR-PASSIVE` 或 `BUS-OFF` 时禁止启动。

## 检查状态

```bash
ip -details link show can_left  | grep "can state"
ip -details link show can_right | grep "can state"
```

期望输出：

```
    can state ERROR-ACTIVE restart-ms 0
```

## 重开（最常用）

程序异常退出、电机报错、或状态不是 `ERROR-ACTIVE` 时，down 再 up 即可：

```bash
sudo ip link set can_left down  && sudo ip link set can_left up
sudo ip link set can_right down && sudo ip link set can_right up
```

一条命令搞定两条并顺带确认：

```bash
for c in can_left can_right; do
  sudo ip link set $c down && sudo ip link set $c up
  printf '%s: ' "$c"; ip -details link show "$c" | grep -o 'ERROR-ACTIVE\|ERROR-PASSIVE\|BUS-OFF'
done
```

## 重开之后还是不对

- `BUS-OFF` 反复出现：多半是线缆接触或终端电阻问题，不是软件问题；先断电检查接线。
- 完全没有 `can_left` / `can_right` 设备：CAN 适配器没识别到，检查 USB 连接后重新插拔。
- 重开后马上又掉：确认机械臂已经上电，而且没有其它进程还占着总线
  （`pgrep -af manimux` 看看有没有残留的 runtime）。

## 什么时候需要重开

- runtime 被 `kill -9`，或进程崩溃没走正常关闭流程；
- 上一次运行出现 `fail to communicate with the motor` / `DM Error in control loop`；
- 机械臂重新上电之后。

正常按 Ctrl-C 退出（手臂回零后自动退出）不需要重开总线。
