"""Small policy-side SDK for publishing rollout observations to the viewer."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from .protocol import PolicyPlan, RobotSnapshot, RuntimeEvent
from .transport import ViewerPublisher


class ViewerClient:
    """Non-owning observer used by a policy executor.

    The client never runs inference or commands a robot. The executor remains the
    source of truth and calls these methods at its existing lifecycle boundaries.
    """

    def __init__(
        self,
        *,
        robot: str,
        policy: str,
        endpoint: str = "tcp://127.0.0.1:5568",
        camera_hz: float = 5.0,
        publisher: ViewerPublisher | None = None,
        control_mode: str = "observe",
    ) -> None:
        if camera_hz < 0:
            raise ValueError("camera_hz must be non-negative")
        if control_mode not in {"observe", "managed"}:
            raise ValueError("control_mode must be 'observe' or 'managed'")
        self.robot = robot
        self.policy = policy
        self.control_mode = control_mode
        self._publisher = publisher or ViewerPublisher(endpoint)
        self._camera_period_s = 0.0 if camera_hz == 0 else 1.0 / camera_hz
        self._last_camera_publish = float("-inf")

    def _event(
        self,
        event: str,
        *,
        step: int = 0,
        chunk_id: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._publisher.publish(
            RuntimeEvent(
                event=event,
                robot=self.robot,
                policy=self.policy,
                step=step,
                chunk_id=chunk_id,
                metadata=dict(metadata or {}),
            )
        )

    def episode_started(
        self,
        *,
        instruction: str,
        max_steps: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        details = dict(metadata or {})
        details.update(
            instruction=instruction,
            max_steps=int(max_steps),
            control_mode=self.control_mode,
        )
        self._event("episode_started", metadata=details)

    def inference_submitted(
        self,
        *,
        step: int,
        chunk_id: int,
        planned_switch_step: int | None = None,
    ) -> None:
        self._event(
            "inference_submitted",
            step=step,
            chunk_id=chunk_id,
            metadata={"planned_switch_step": planned_switch_step},
        )

    def plan_activated(
        self,
        *,
        actions: np.ndarray,
        action_index: int,
        chunk_id: int,
        step: int,
        action_dt: float,
        inference_ms: float,
        instruction: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._publisher.publish(
            PolicyPlan(
                policy=self.policy,
                instruction=instruction,
                actions=np.asarray(actions, dtype=np.float64),
                action_dt=action_dt,
                inference_ms=max(0.0, inference_ms),
                chunk_id=chunk_id,
                robot=self.robot,
                metadata=dict(metadata or {}),
                start_index=action_index,
            )
        )

    def step_executed(
        self,
        *,
        joint_positions: np.ndarray,
        cameras: Mapping[str, np.ndarray],
        step: int,
        max_steps: int,
        action_index: int,
        chunk_id: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        now = time.monotonic()
        frames: dict[str, np.ndarray] = {}
        if self._camera_period_s > 0 and (now - self._last_camera_publish >= self._camera_period_s):
            frames = {
                name: np.asarray(frame, dtype=np.uint8).copy() for name, frame in cameras.items()
            }
            if frames:
                self._last_camera_publish = now
        self._publisher.publish(
            RobotSnapshot(
                joint_positions=np.asarray(joint_positions, dtype=np.float64),
                cameras=frames,
                step=step,
                max_steps=max_steps,
                chunk_index=action_index,
                robot=self.robot,
                metadata=dict(metadata or {}),
                active_chunk_id=chunk_id,
            )
        )

    def episode_finished(
        self,
        *,
        reason: str,
        step: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        details = dict(metadata or {})
        details["reason"] = reason
        self._event("episode_finished", step=step, metadata=details)

    def close(self) -> None:
        self._publisher.close()
