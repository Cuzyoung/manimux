#!/usr/bin/env python3
"""Run one hardware-free XPolicyLab inference through the ManiMux YAM adapter."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import yaml

from manimux.config import load_config
from manimux.policies import build_policy_adapter, build_policy_model
from manimux.types import (
    ActionContext,
    InferenceRequest,
    ObservationSnapshot,
    RobotState,
    SensorFrame,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _repo_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _start_joints(path: Path, expected_dim: int) -> np.ndarray:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    values = np.asarray(payload["agent"]["start_joints"], dtype=np.float64)
    if values.shape != (expected_dim,) or not np.isfinite(values).all():
        raise ValueError(
            f"{path} agent.start_joints must contain {expected_dim} finite values"
        )
    return values


def _synthetic_rgb(height: int, width: int, offset: int) -> np.ndarray:
    horizontal = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    vertical = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    red = np.broadcast_to((horizontal + offset).astype(np.uint8), (height, width))
    green = np.broadcast_to((vertical + offset).astype(np.uint8), (height, width))
    blue = ((red.astype(np.uint16) + green.astype(np.uint16)) // 2).astype(np.uint8)
    return np.ascontiguousarray(np.stack([red, green, blue], axis=-1))


def _snapshot(config_path: Path, height: int, width: int) -> ObservationSnapshot:
    config = load_config(config_path)
    group_order = list(config.policy.options["group_order"])
    if group_order != ["left_arm", "right_arm"]:
        raise ValueError(f"YAM probe requires left_arm/right_arm, got {group_order}")
    robot_configs = {
        "left_arm": _repo_path(config.robot.config),
        "right_arm": _repo_path(config.robot.options["right_config"]),
    }
    groups = {
        name: _start_joints(robot_configs[name], int(config.robot.group_dims[name]))
        for name in group_order
    }

    camera_map = config.policy.options["camera_map"]
    camera_names = list(dict.fromkeys(str(name) for name in camera_map.values()))
    if len(camera_names) != 3:
        raise ValueError(f"YAM probe requires three camera roles, got {camera_names}")
    monotonic_ns = time.monotonic_ns()
    frames = {
        name: SensorFrame(
            name=name,
            data=_synthetic_rgb(height, width, 17 + 56 * index),
            capture_monotonic_ns=monotonic_ns,
            sequence=1,
        )
        for index, name in enumerate(camera_names)
    }
    return ObservationSnapshot(
        state=RobotState(groups=groups, monotonic_ns=monotonic_ns, sequence=1),
        frames=frames,
    )


def _native_action_summary(raw: object) -> dict[str, object]:
    if isinstance(raw, Mapping) and "actions" in raw:
        actions = np.asarray(raw["actions"])
        native_format = "packed_actions"
    elif isinstance(raw, Sequence) and raw and all(isinstance(step, Mapping) for step in raw):
        rows = [
            np.concatenate([np.asarray(value).reshape(-1) for value in step.values()])
            for step in raw
        ]
        actions = np.stack(rows)
        native_format = "action_step_dicts"
    else:
        return {"native_shape": None, "native_format": "unknown"}
    return {
        "native_shape": list(actions.shape),
        "native_finite": bool(np.isfinite(actions).all()),
        "native_format": native_format,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--instruction", default="Pick the red block up.")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()
    if args.height <= 0 or args.width <= 0:
        raise ValueError("height and width must be positive")

    config_path = args.config.resolve()
    config = load_config(config_path)
    if config.policy.worker != "xpolicylab_ws":
        raise ValueError("forward probe requires policy.worker: xpolicylab_ws")
    snapshot = _snapshot(config_path, args.height, args.width)
    session_id = f"xpolicy-probe-{uuid.uuid4().hex[:8]}"
    observation_time_ns = time.monotonic_ns()
    request = InferenceRequest(
        session_id=session_id,
        request_seq=1,
        observation_time_ns=observation_time_ns,
        deadline_ns=observation_time_ns + int(config.policy.timeout_s * 1_000_000_000),
        observation=snapshot,
        instruction=args.instruction,
    )

    model = build_policy_model(config.policy)
    adapter = build_policy_adapter(config.robot, config.policy)
    started = time.perf_counter()
    try:
        model.reset(session_id)
        raw = model.infer(request)
    finally:
        model.close()
    round_trip_ms = (time.perf_counter() - started) * 1000.0
    native_summary = _native_action_summary(raw)
    if native_summary.get("native_finite") is False:
        raise ValueError("model returned non-finite native actions")

    chunk = adapter.decode_action(
        raw,
        ActionContext(
            request_seq=1,
            observation_time_ns=observation_time_ns,
            created_time_ns=time.monotonic_ns(),
        ),
    )
    group_order = list(config.policy.options["group_order"])
    packed = np.concatenate([chunk.groups[name] for name in group_order], axis=1)
    expected_shape = (
        config.policy.horizon_steps,
        sum(int(config.robot.group_dims[name]) for name in group_order),
    )
    if packed.shape != expected_shape:
        raise ValueError(f"expected a {expected_shape} chunk, got {packed.shape}")
    if not np.isfinite(packed).all():
        raise ValueError("ManiMux adapter returned non-finite joint targets")

    print(
        json.dumps(
            {
                "status": "ok",
                "config": str(config_path),
                "server": config.policy.options["server"],
                "action_codec": config.policy.options.get("action_codec", "joint_position"),
                **native_summary,
                "action_space": chunk.action_space,
                "canonical_shape": list(packed.shape),
                "dt_s": chunk.dt_ns / 1_000_000_000,
                "round_trip_ms": round(round_trip_ms, 1),
                "minimum": float(packed.min()),
                "maximum": float(packed.max()),
                "first_action": packed[0].tolist(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
