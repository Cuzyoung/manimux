from __future__ import annotations

import math

import numpy as np

from manimux.config import SmoothConfig
from manimux.runtime.executors.limits import ScalarLimits, limit_step
from manimux.types import (
    ActionHorizon,
    GroupVector,
    RobotCommand,
    RobotState,
    copy_group_vector,
)


class SmoothExecutor:
    def __init__(self, config: SmoothConfig, control_dt_s: float) -> None:
        self._dt_s = control_dt_s
        rc = 1.0 / (2.0 * math.pi * config.cutoff_hz)
        self._alpha = control_dt_s / (rc + control_dt_s)
        self._limits = ScalarLimits(
            max_velocity=config.max_velocity,
            max_acceleration=config.max_acceleration,
            position_limit_abs=config.position_limit_abs,
        )
        self._previous: GroupVector | None = None
        self._previous_velocity: GroupVector | None = None

    @property
    def horizon_steps(self) -> int:
        return 2

    def reset(self, state: RobotState) -> None:
        self._previous = copy_group_vector(state.groups)
        self._previous_velocity = {
            name: np.zeros_like(value) for name, value in state.groups.items()
        }

    def step(
        self,
        now_ns: int,
        state: RobotState,
        reference: ActionHorizon,
    ) -> RobotCommand:
        if self._previous is None or self._previous_velocity is None:
            self.reset(state)
        assert self._previous is not None
        assert self._previous_velocity is not None
        target = {
            name: self._previous[name] + self._alpha * (values[0] - self._previous[name])
            for name, values in reference.groups.items()
        }
        output, velocities = limit_step(
            target,
            self._previous,
            self._previous_velocity,
            dt_s=self._dt_s,
            limits=self._limits,
        )
        self._previous = copy_group_vector(output)
        self._previous_velocity = copy_group_vector(velocities)
        return RobotCommand(groups=output, monotonic_ns=now_ns, plan_id=reference.plan_id)
