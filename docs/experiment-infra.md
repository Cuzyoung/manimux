# Experiment Infrastructure

Operator-facing screenshots, button meanings and the complete state flow are in the
[Viewer visual tutorial](viewer-tutorial.html). This document keeps the experiment data contract and
fair-comparison rules.

This document defines the operational contract for repeated real-robot evaluation. Model loading,
camera services and hardware preflight remain in each model runbook; the experiment layer does not
start or modify them.

## 1. Two runtime entry points

```bash
# One rollout, controlled from the terminal
manimux run --config <experiment.yaml>

# Persistent runtime service, controlled from Viewer
manimux serve --config <experiment.yaml>
```

`run` preserves the original CLI workflow. `serve` keeps the selected config available while Viewer
creates isolated rollouts. It does not launch a Policy Server, camera server or Viewer.

## 2. Viewer modes

Viewer exposes a prominent `Experiment mode` switch before each rollout:

| Mode | Intended use | Human reward |
|---|---|---|
| **OFF** | Deployment, debugging and demonstrations | Optional; the next rollout is not blocked |
| **ON** | Formal pilot or benchmark collection | Required after every finalized rollout |

The switch is locked after `Prepare new rollout` so one rollout cannot change modes midway. When
experiment mode is ON, also set a readable `Layout / condition ID` such as `red-ball-left-01`.

The task text shown in Viewer is not decorative: the value present when `Prepare new rollout` is
clicked is copied into that rollout config and sent to the policy.

## 3. Operator flow

After the model server, camera server, Viewer and `manimux serve` are independently ready:

1. Confirm the task command; for an experiment rollout, fill the layout or condition ID.
2. Click `Prepare normal rollout` or `Prepare experiment rollout`.
3. Wait for `PAUSED`, inspect the physical setup, then click `Start rollout`.
4. Use `Pause / Hold` only when execution must stop without ending the rollout.
5. Click `Finish & Home` after success, failure or timeout.
6. Wait for Recorder finalization and the robot's configured shutdown/home sequence.
7. If experiment mode is ON, select `success`, `failure` or `invalid`, add the smoothness score and
   failure tags, then click `Save evaluation`.
8. Prepare the next rollout only after the service reports ready.

`Pause / Hold` holds the current commanded position; it does not return home. The advanced recovery
control can request the configured home path without ending the rollout. `Finish & Home` finalizes
the episode and then follows the runtime's configured shutdown sequence.

## 4. Evidence contract

```text
data/experiments/<campaign>/<algorithm>/session-*/
├── session-manifest.json
└── rollout-001/
    ├── meta.json
    ├── events.jsonl
    ├── result.json
    ├── data.zarr/
    │   ├── ticks/
    │   └── plans/000000/
    │       ├── canonical_raw/
    │       ├── infra_output/
    │       └── committed/
    ├── videos/
    │   ├── <camera>.mp4
    │   └── index.json
    └── evaluation/
        └── human-label.json
```

- `session-manifest.json` freezes the resolved config, its SHA256, ManiMux git SHA and XPolicyLab git
  SHA.
- `meta.json` records task, layout, algorithm, experiment mode and the Policy Server fingerprint.
- `canonical_raw` is the decoded policy chunk before the inference strategy.
- `infra_output` is the chunk after the selected inference strategy.
- `committed` is the final horizon accepted by Timeline after trimming or blending.
- `ticks` stores measured state, scheduled reference, executor output and command.
- `videos/index.json` stores camera timestamps, frame counts, dropped bundles and encoder errors.
- `human-label.json` exists only when an operator saves an evaluation.
- `result.json.success` means the runtime finalized normally; it is never task success.

Video recording is best-effort and asynchronous. A full video queue drops video bundles rather than
blocking the robot control loop. Formal analysis must inspect `dropped_bundles` and `error`; a damaged
recording should be marked `invalid`, not silently counted as failure.

## 5. Fair pilot checklist

Before comparing algorithms:

- use the same checkpoint, norm stats, task text, home/start state and physical layout definition;
- warm the model server before timed rollouts;
- assign an explicit layout ID and randomize algorithm order;
- freeze each algorithm config and preserve its config hash;
- count attempts, valid rollouts, invalid rollouts and safety stops separately;
- inspect the backend fingerprint so a restarted service did not load another checkpoint;
- tune on development layouts, then stop changing parameters on test layouts;
- derive automatic metrics only after matching trajectories, videos and human labels.

See [experiment design](experiment-design.md) for the study matrix and reporting rules.
