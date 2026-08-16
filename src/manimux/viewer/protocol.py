"""Policy- and robot-independent messages sent to the viewer."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np

PROTOCOL_VERSION = 1


@dataclass(slots=True)
class PolicyPlan:
    """One predicted action chunk in the representation a robot adapter accepts."""

    policy: str
    instruction: str
    actions: np.ndarray
    action_dt: float
    inference_ms: float
    chunk_id: int
    robot: str = ""
    action_space: str = "joint_position"
    metadata: dict[str, Any] = field(default_factory=dict)
    start_index: int = 0

    def __post_init__(self) -> None:
        self.actions = np.asarray(self.actions, dtype=np.float64)
        if self.actions.ndim != 2:
            raise ValueError("actions must have shape (horizon, action_dim)")
        if not self.actions.shape[0] or not self.actions.shape[1]:
            raise ValueError("actions must have a non-empty horizon and action dimension")
        if not np.isfinite(self.actions).all():
            raise ValueError("actions must contain only finite values")
        if not np.isfinite(self.action_dt) or self.action_dt <= 0:
            raise ValueError("action_dt must be finite and positive")
        if not np.isfinite(self.inference_ms) or self.inference_ms < 0:
            raise ValueError("inference_ms must be finite and non-negative")
        if self.start_index < 0 or self.start_index > len(self.actions):
            raise ValueError("start_index must be within the action horizon")

    def to_wire(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "kind": "plan",
            "robot": self.robot,
            "policy": self.policy,
            "instruction": self.instruction,
            "actions": self.actions.tolist(),
            "action_space": self.action_space,
            "action_dt": float(self.action_dt),
            "inference_ms": float(self.inference_ms),
            "chunk_id": int(self.chunk_id),
            "start_index": int(self.start_index),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class RobotSnapshot:
    """Latest achieved joint state, camera images, and execution cursor."""

    joint_positions: np.ndarray
    cameras: dict[str, np.ndarray]
    step: int
    max_steps: int
    chunk_index: int = 0
    connected: bool = True
    robot: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    active_chunk_id: int | None = None

    def __post_init__(self) -> None:
        self.joint_positions = np.asarray(self.joint_positions, dtype=np.float64).reshape(-1)
        if not np.isfinite(self.joint_positions).all():
            raise ValueError("joint_positions must contain only finite values")

    def to_wire(self, jpeg_quality: int = 75) -> dict[str, Any]:
        from PIL import Image

        encoded: dict[str, str] = {}
        for name, frame in self.cameras.items():
            rgb = np.asarray(frame, dtype=np.uint8)
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError(f"camera {name!r} must have shape (height, width, 3)")
            buffer = io.BytesIO()
            Image.fromarray(rgb).save(buffer, format="JPEG", quality=jpeg_quality)
            encoded[name] = base64.b64encode(buffer.getvalue()).decode("ascii")
        return {
            "protocol_version": PROTOCOL_VERSION,
            "kind": "state",
            "robot": self.robot,
            "joint_positions": self.joint_positions.tolist(),
            "cameras_jpeg": encoded,
            "step": int(self.step),
            "max_steps": int(self.max_steps),
            "chunk_index": int(self.chunk_index),
            "active_chunk_id": self.active_chunk_id,
            "connected": bool(self.connected),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class RuntimeEvent:
    """Policy-executor lifecycle or asynchronous-inference status event."""

    event: str
    robot: str = ""
    policy: str = ""
    step: int = 0
    chunk_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "kind": "event",
            "event": self.event,
            "robot": self.robot,
            "policy": self.policy,
            "step": int(self.step),
            "chunk_id": self.chunk_id,
            "metadata": self.metadata,
        }
