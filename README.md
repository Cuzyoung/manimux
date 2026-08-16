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
      safety -> robot command -> Zarr/JSONL
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
- lazy Universal Viewer compatibility bridge.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev
uv run manimux run --config configs/mock.yaml
uv run manimux run --config configs/mock.yaml --executor mpc
```

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

This is the Milestone 0 implementation. Do not use it to command real hardware
without adding hardware limits, watchdog validation, emergency-stop integration,
and a robot-specific acceptance test.
