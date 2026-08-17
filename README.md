# ManiMux

ManiMux is a local asynchronous policy runtime for dual-arm robot evaluation.
It keeps model inference off the control loop, time-aligns action chunks, executes
them through either Smooth or local MPC, and records the complete action lineage
to local storage.

V1 is intentionally local-only. It has no cloud inference, auth, gateway, fleet
control plane, database, or data upload.

## Current runnable slice

```text
mock/ManiUniCon robot + mock camera
                 |
          observation snapshot
                 |
       local policy worker process
                 |
        atomic ActionTimeline
                 |
        SmoothExecutor | MPCExecutor
                 |
      safety -> robot command -> Zarr/JSONL + built-in Viewer
```

Implemented now:

- strict single-file YAML configuration;
- bounded latest-wins local inference queues;
- canonical named dual-arm groups;
- time-indexed, stale-trimming atomic action timeline;
- Smooth and joint-position MPC executors;
- deterministic mock robot, camera, and policy;
- optional ManiUniCon dual-Meshcat adapter;
- local partial-to-final episode recording;
- built-in, robot-adapter-based Viser policy viewer;
- self-contained MolmoAct2/YAM async rollout integration and camera service.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev
uv run manimux run --config configs/mock.yaml
uv run manimux run --config configs/mock.yaml --executor mpc
```

Run the viewer by itself with synthetic YAM state and policy plans:

```bash
uv run manimux-viewer --robot yam --demo --port 8086
```

The viewer renders the achieved robot state and the currently pending
end-effector trajectory. The trajectory starts at the gripper center and uses
a deep-to-light purple gradient from the nearest action toward the future.
The GUI can switch between only the current plan and retained plan history.

## MolmoAct2 + YAM

The MolmoAct robot launcher, asynchronous chunk execution, inference server,
camera server, rollout recording, and viewer observer now live inside
`src/manimux/integrations/molmoact_yam/`. They do not import Python files from
the previous checkout. YAM URDF/XML/STL assets used by the viewer are also
bundled under `src/manimux/assets/`.

Install the optional hardware-policy dependencies and start the four-process
setup from this repository:

```bash
uv sync --dev --extra molmoact-yam
uv pip install \
  "git+https://github.com/i2rt-robotics/i2rt.git@5d47b358bafb30c65e397f2ece506550a0db4594"

# Terminal 1
uv run --extra molmoact-yam manimux-molmoact-server --port 8202

# Terminal 2
uv run --extra molmoact-yam manimux-molmoact-camera \
  --config src/manimux/integrations/molmoact_yam/configs/molmoact_yam_left.yaml

# Terminal 3
uv run manimux-viewer --robot yam --port 8086

# Terminal 4
uv run --extra molmoact-yam manimux-molmoact-yam \
  --left-config-path src/manimux/integrations/molmoact_yam/configs/molmoact_yam_left.yaml \
  --right-config-path src/manimux/integrations/molmoact_yam/configs/molmoact_yam_right.yaml
```

Before a hardware run, review camera serials, CAN channel names, start joints,
and storage paths in the two checked-in YAML files. `i2rt` is pinned separately
because it is the hardware driver rather than ManiMux application code. Model
weights and rollout data are intentionally not committed.

For the concise four-terminal startup sequence, see
[docs/molmoact-yam-runbook.md](docs/molmoact-yam-runbook.md).

Episodes are written under `data/<run-id>/`. Run all local checks with:

```bash
make lint
make typecheck
make test
make test-integration
```

For the current scope and interface contracts, see
[docs/architecture.md](docs/architecture.md). For the simulator setup used on
`jw`, see [docs/maniunicon-sim.md](docs/maniunicon-sim.md).

## Repository status

This branch includes the current YAM integration, but automated tests do not
prove real-hardware safety. Before commanding a robot, validate hardware
limits, watchdogs, emergency stop, CAN interfaces, current joint state, and the
exact target trajectory on the workstation.
