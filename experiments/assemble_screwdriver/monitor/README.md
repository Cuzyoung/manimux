# Live training curves

The monitor is a sidecar. It does not change any training process or optimizer.
LingBot-VLA2 and XR-1 already emit TensorBoard events. The sidecar mirrors Pi05's
stdout `loss`, `grad_norm`, and `param_norm` into the same dashboard.

On the training host:

```bash
bash /inspire/hdd2/project/liu-ming-huan/public/ziyang/manimux/experiments/assemble_screwdriver/monitor/start_live_tensorboard.sh
```

On the local workstation:

```bash
ssh -N -L 16006:127.0.0.1:16006 localhost-3338
```

Open <http://127.0.0.1:16006>. TensorBoard reloads new event data every ten
seconds. Use these primary curves:

- Pi05: `training/loss`
- LingBot-VLA2: `training/vla_loss`
- XR-1 flow matching: `train/loss_mse`
