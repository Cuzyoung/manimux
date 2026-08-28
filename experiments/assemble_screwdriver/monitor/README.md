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
- GR00T N1.7: `training/loss`

The GR00T checkpoint downloader waits for complete deployment bundles and then
resumes each file into the local checkpoint directory. It intentionally omits
DeepSpeed optimizer shards, which are only needed for resuming training:

```bash
python experiments/assemble_screwdriver/monitor/download_gr00t_checkpoints.py \
  --remote-root /inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data/weights/finetuned/gr00t-n17/assemble-screwdriver-gr00t-n17-4xh100-3k-20260829-v1 \
  --local-root /home/ubuntu/manimux/checkpoints/finetuned/gr00t-n17/assemble-screwdriver-20260829-v1
```
