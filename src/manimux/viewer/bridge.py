from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from manimux.types import ActionChunk, RobotState, SensorFrame


@dataclass(frozen=True, slots=True)
class ViewerControl:
    paused: bool
    step_once: bool = False
    home_requested: bool = False
    finish_requested: bool = False


class ViewerBridge:
    """Lazy compatibility bridge to SII-LiuLab/universal_viewer protocol v1."""

    def __init__(self, enabled: bool, robot_adapter: str, group_order: list[str]) -> None:
        self._enabled = enabled
        self._robot_adapter = robot_adapter
        self._group_order = group_order
        self._publisher: Any | None = None
        self._controls: Any | None = None
        self._policy_plan_type: Any | None = None
        self._snapshot_type: Any | None = None
        if not enabled:
            return
        try:
            from universal_policy_viewer import PolicyPlan, RobotSnapshot
            from universal_policy_viewer.bridge import ControlClient, ViewerPublisher
        except ImportError as exc:
            raise RuntimeError(
                "viewer.enabled=true requires SII-LiuLab/universal_viewer on PYTHONPATH"
            ) from exc
        self._publisher = ViewerPublisher()
        self._controls = ControlClient()
        self._policy_plan_type = PolicyPlan
        self._snapshot_type = RobotSnapshot

    def poll_control(self) -> ViewerControl:
        if not self._enabled:
            return ViewerControl(paused=False)
        assert self._controls is not None
        state = self._controls.poll()
        return ViewerControl(
            paused=bool(state.get("paused", True)),
            step_once=bool(state.get("step_once", False)),
            home_requested=bool(state.get("home_requested", False)),
            finish_requested=bool(state.get("finish_requested", False)),
        )

    def publish_plan(self, chunk: ActionChunk, inference_ms: float) -> None:
        if not self._enabled:
            return
        assert self._publisher is not None
        assert self._policy_plan_type is not None
        actions = np.concatenate([chunk.groups[name] for name in self._group_order], axis=1)
        message = self._policy_plan_type(
            robot=self._robot_adapter,
            policy="manimux-local",
            instruction="",
            actions=actions,
            action_space=chunk.action_space,
            action_dt=chunk.dt_ns / 1_000_000_000,
            inference_ms=inference_ms,
            chunk_id=chunk.request_seq,
            metadata={"plan_id": chunk.plan_id},
        )
        self._publisher.publish(message)

    def publish_state(
        self,
        state: RobotState,
        frames: dict[str, SensorFrame],
        *,
        step: int,
        max_steps: int,
    ) -> None:
        if not self._enabled:
            return
        assert self._publisher is not None
        assert self._snapshot_type is not None
        joint_positions = np.concatenate([state.groups[name] for name in self._group_order])
        message = self._snapshot_type(
            robot=self._robot_adapter,
            joint_positions=joint_positions,
            cameras={name: frame.data for name, frame in frames.items()},
            step=step,
            max_steps=max_steps,
            connected=True,
        )
        self._publisher.publish(message)

    def close(self) -> None:
        for component in (self._publisher, self._controls):
            if component is not None:
                component.close()
