# Live training curves

The monitor is a sidecar. It does not change any training process or optimizer.
LingBot-VLA2 and XR-1 already emit TensorBoard events. The sidecar mirrors Pi05's
stdout `loss`, `grad_norm`, and `param_norm` into the same dashboard.

From the ManiMux repository root on the local workstation:

```bash
./training_dashboard.sh
```

Open <http://127.0.0.1:16006>. The first page selects a model; each model opens
an independent TensorBoard containing only that model's runs. The command
installs a user systemd service for the local tunnel, so an SSH disconnect is
retried automatically. The remote dashboard has its own restart guard.
`status`, `restart`, and `stop` are also supported:

```bash
./training_dashboard.sh status
```

The dashboard maintains a two-level `model/run_name` event catalog. Native
LingBot and XR1 event files appear automatically with their long internal
checkpoint paths hidden. A supervisor discovers new Pi05 and GR00T stdout logs
and mirrors their metrics into TensorBoard. New runs show up in the selected
model's run list within roughly five seconds, without a service restart. Use
these primary curves:

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
