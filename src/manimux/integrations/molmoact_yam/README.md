# MolmoAct2 + YAM integration

This package contains the robot-side launcher, asynchronous action-chunk loop,
camera service, MolmoAct client/server, rollout recorder, RTC helper, and YAM
runtime needed by ManiMux. It imports only modules inside `manimux` plus normal
Python dependencies; it does not load Python files from another checkout.

Model checkpoints are intentionally not stored in Git. Server mode downloads
`allenai/MolmoAct2-BimanualYAM` through Hugging Face, or accepts a local model
directory through `--repo-id`. The real YAM motor layer uses a separately
pinned `i2rt` hardware driver.

Install the integration dependencies:

```bash
uv sync --dev --extra molmoact-yam
uv pip install \
  "git+https://github.com/i2rt-robotics/i2rt.git@5d47b358bafb30c65e397f2ece506550a0db4594"
```

The checked-in hardware configuration is under `configs/`. Review the camera
serials, CAN channels, start joints, storage directory, and server endpoint
before connecting hardware.

Start the four processes from the ManiMux repository root:

```bash
# Terminal 1: MolmoAct inference server.
uv run --extra molmoact-yam manimux-molmoact-server --port 8202

# Terminal 2: long-lived RealSense owner.
uv run --extra molmoact-yam manimux-camera-server \
  --config configs/robots/yam_left.yaml

# Terminal 3: Viser dashboard.
uv run manimux-viewer --robot yam --port 8086

# Terminal 4: robot rollout with viewer telemetry.
uv run --extra molmoact-yam manimux-molmoact-yam \
  --left-config-path configs/robots/yam_left.yaml \
  --right-config-path configs/robots/yam_right.yaml
```

The viewer is observational for this integration: MolmoAct/YAM remains the
owner of inference, action stitching, robot commands, recording, and parking.
Ctrl-C saves the incomplete rollout, waits for zero-home, and exits; it does
not start another rollout in the same process.

LeRobot conversion is optional. Raw rollouts remain available if the separate
`lerobot` package is absent or conversion fails.
