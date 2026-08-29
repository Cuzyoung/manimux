# Live training curves

The monitor is a sidecar. It does not change any training process or optimizer.
LingBot-VLA2 and XR-1 already emit TensorBoard events. The sidecar mirrors Pi05's
stdout `loss`, `grad_norm`, and `param_norm` into the same dashboard.

From the shared training-code checkout on the CPU server:

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

All training outputs and inference bundles stay under:

```text
/inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data/weights
```

The dashboard reads the shared run directories directly. No checkpoint download
to `/home/ubuntu/manimux` is part of the current training workflow.
