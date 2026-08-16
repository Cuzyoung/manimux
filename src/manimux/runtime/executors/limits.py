from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manimux.types import GroupVector


@dataclass(frozen=True, slots=True)
class ScalarLimits:
    max_velocity: float
    max_acceleration: float
    position_limit_abs: float


def limit_step(
    target: GroupVector,
    previous: GroupVector,
    previous_velocity: GroupVector,
    *,
    dt_s: float,
    limits: ScalarLimits,
) -> tuple[GroupVector, GroupVector]:
    output: GroupVector = {}
    velocities: GroupVector = {}
    for name, desired in target.items():
        velocity = np.clip(
            (desired - previous[name]) / dt_s,
            -limits.max_velocity,
            limits.max_velocity,
        )
        velocity = np.clip(
            velocity,
            previous_velocity[name] - limits.max_acceleration * dt_s,
            previous_velocity[name] + limits.max_acceleration * dt_s,
        )
        command = previous[name] + velocity * dt_s
        output[name] = np.clip(
            command,
            -limits.position_limit_abs,
            limits.position_limit_abs,
        )
        velocities[name] = velocity
    return output, velocities
