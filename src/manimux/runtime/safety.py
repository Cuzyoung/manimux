from __future__ import annotations

from enum import StrEnum

import numpy as np

from manimux.types import RobotCommand, RobotState


class RuntimeState(StrEnum):
    DISCONNECTED = "disconnected"
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    FAULT = "fault"


class SafetyGuard:
    def __init__(self, group_dims: dict[str, int], position_limit_abs: float) -> None:
        self._group_dims = dict(group_dims)
        self._position_limit_abs = position_limit_abs

    def validate_state(self, state: RobotState) -> None:
        if set(state.groups) != set(self._group_dims):
            raise ValueError("state groups do not match robot configuration")
        for name, dim in self._group_dims.items():
            if state.groups[name].shape != (dim,):
                raise ValueError(f"state group {name!r} has the wrong dimension")

    def validate_command(self, command: RobotCommand) -> None:
        if set(command.groups) != set(self._group_dims):
            raise ValueError("command groups do not match robot configuration")
        for name, dim in self._group_dims.items():
            values = command.groups[name]
            if values.shape != (dim,) or not np.isfinite(values).all():
                raise ValueError(f"command group {name!r} is invalid")
            if np.any(np.abs(values) > self._position_limit_abs):
                raise ValueError(f"command group {name!r} exceeds the position limit")
