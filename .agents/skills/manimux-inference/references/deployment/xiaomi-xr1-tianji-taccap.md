# Xiaomi Robotics 1 / Tianji–TacCap ball-passing integration

> 2026-10-09: The old screwdriver, red-object, pass-ball and offline experiment entries below were retired. These commands are historical; select a current entry from the [experiment index](../../../../../manimux/configs/experiments/README.md). Diagnostic tools now require an explicit experiment argument.


This entry point uses `XPolicyLab/policy/Xiaomi_Robotics_1` to load the local step-50000
DeepSpeed model file and serves ManiMux through the shared `xpolicylab_ws` worker. The model receives two
real wrist RGB views. The third training view, ego, was black, so XPolicyLab creates
an all-zero `uint8 RGB` image during preprocessing; it neither registers a fake camera nor reuses another live view.

## Fixed contract

- Checkpoint: `mp_rank_00_model_states.pt`; the eight optimizer states are not used for inference.
- Normalization: `training_metadata/normalize.json` beside the checkpoint, with shapes
  `mean/std=(30,60)`, `q01/q99=(1,60)`.
- Observation: seven joints and one `[0,1]` TacCap opening value per Tianji arm, plus two wrist RGB views.
  The third view is black at the left wrist resolution, then passes through the shared XR-1 resize.
- Action: 30×60 anchor-relative EE delta. Every row is relative to the TCP pose at request time,
  not accumulated row by row. The adapter uses assembled Tianji TCP FK/IK to produce two 30×8 joint-position arrays.
- TacCap retains the checkpoint's continuous `[0,1]` opening values without replacing them with close-latch semantics.
- Waist/base slots `[16:20]` have no Tianji actuators; they are explicitly discarded and their maximum absolute value is recorded.
- The initial integration supports ordinary Async single-inflight only; the Tianji XR-1 RTC condition codec is not implemented.

## Files

- Model service: `manimux/configs/policy/xiaomi-xr1/tianji/pass_ball/step50000.yaml`
- Experiment template with execution safely disabled:
  `manimux/configs/experiments/pass_ball/xiaomi-xr1/tianji_taccap_xiaomi_xr1_step50000.yaml`
- server launcher: `manimux/servers/xiaomi_xr1_tianji_server.py`

The template keeps `robot.options.execute: false` and RoboGUI disabled. Loading the configuration, checking the checkpoint, or starting
the service does not move the robot.

## 1. Install the isolated model environment

Do not install model dependencies into the Tianji hardware runtime environment:

```bash
cd /path/to/manimux/XPolicyLab/policy/Xiaomi_Robotics_1
MIBOT_CONDA_ENV=mibot bash install.sh
```

The Qwen processor defaults to `Qwen/Qwen3-VL-4B-Instruct`; its first start requires Hugging Face
access or a populated cache. On offline machines, use `--processor /absolute/local/processor`.

## 2. Check artifacts and configuration only

```bash
cd /path/to/manimux
conda run -n mibot --no-capture-output python \
  manimux/servers/xiaomi_xr1_tianji_server.py \
  --experiment manimux/configs/experiments/pass_ball/xiaomi-xr1/tianji_taccap_xiaomi_xr1_step50000.yaml \
  --local .local/tianji_taccap.yaml \
  --checkpoint /path/to/posttrain_pass_ball_50k_8gpu_epoch0_step50000_checkpoint \
  --check
```

When given a checkpoint directory, the launcher resolves the unique
`mp_rank_00_model_states.pt` and automatically uses the adjacent
`training_metadata/normalize.json`. `--check` verifies only files, shapes, protocol, and the black-image contract;
it does not load the 5B model.

## 3. Start the model service

```bash
cd /path/to/manimux
conda run -n mibot --no-capture-output python \
  manimux/servers/xiaomi_xr1_tianji_server.py \
  --experiment manimux/configs/experiments/pass_ball/xiaomi-xr1/tianji_taccap_xiaomi_xr1_step50000.yaml \
  --local .local/tianji_taccap.yaml \
  --checkpoint /path/to/posttrain_pass_ball_50k_8gpu_epoch0_step50000_checkpoint
```

The service publishes its checkpoint, normalization, `packed_ee_delta`, black ego profile, and
action semantics. ManiMux checks each before rollout to avoid connecting to UMI-DP or the wrong XR-1
checkpoint.

## 4. Cameras and safe dry run

In another terminal, start the existing TacCap camera server with only the two wrist cameras:

```bash
cd /path/to/manimux
conda run -n xense-taccap --no-capture-output python \
  -m manimux.servers.camera.server \
  --experiment manimux/configs/experiments/pass_ball/xiaomi-xr1/tianji_taccap_xiaomi_xr1_step50000.yaml \
  --local .local/tianji_taccap.yaml
```

Keep `execute: false` in the template and run the runtime to inspect the handshake, inference latency, IK, and recordings:

```bash
conda run -n xense-taccap --no-capture-output python -m manimux run \
  --config manimux/configs/experiments/pass_ball/xiaomi-xr1/tianji_taccap_xiaomi_xr1_step50000.yaml \
  --local .local/tianji_taccap.yaml
```

For physical execution, first validate the dry run, initial pose, both wrist camera orientations, TCP, gripper `[0,1]` convention, and the full
30-step IK. Copy the template into Git-ignored `.local/`, review it manually, and set
`execute: true` only in that copy. This integration did not start hardware or claim successful physical ball passing.
