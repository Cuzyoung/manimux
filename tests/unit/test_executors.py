from __future__ import annotations

import numpy as np

from manimux.config import MPCConfig, SmoothConfig
from manimux.runtime.executors import MPCExecutor, SmoothExecutor
from manimux.types import ActionHorizon, RobotState


def _state() -> RobotState:
    return RobotState(
        groups={"left_arm": np.zeros(2), "right_arm": np.zeros(2)},
        monotonic_ns=0,
        sequence=0,
    )


def _reference(target: float, horizon_steps: int = 20) -> ActionHorizon:
    values = np.full((horizon_steps, 2), target, dtype=np.float64)
    return ActionHorizon(
        start_time_ns=0,
        dt_ns=10_000_000,
        plan_id="plan",
        groups={"left_arm": values, "right_arm": -values},
    )


def test_smooth_executor_obeys_acceleration_and_velocity_limits() -> None:
    executor = SmoothExecutor(
        SmoothConfig(
            cutoff_hz=100.0,
            max_velocity=1.0,
            max_acceleration=2.0,
            position_limit_abs=3.0,
        ),
        control_dt_s=0.01,
    )
    state = _state()
    executor.reset(state)
    first = executor.step(0, state, _reference(2.0))
    second = executor.step(10_000_000, state, _reference(2.0))
    np.testing.assert_allclose(first.groups["left_arm"], [0.0002, 0.0002])
    np.testing.assert_allclose(second.groups["left_arm"], [0.0006, 0.0006])


def test_mpc_executor_tracks_reference_with_bounded_first_step() -> None:
    executor = MPCExecutor(
        MPCConfig(
            horizon_steps=10,
            dynamics_a=0.85,
            tracking_weight=10.0,
            command_delta_weight=1.0,
            max_velocity=2.0,
            max_acceleration=8.0,
            position_limit_abs=3.0,
        ),
        control_dt_s=0.01,
    )
    state = _state()
    executor.reset(state)
    command = executor.step(0, state, _reference(1.0))
    assert np.all(command.groups["left_arm"] > 0)
    assert np.all(command.groups["right_arm"] < 0)
    assert np.max(np.abs(command.groups["left_arm"])) <= 0.0008 + 1e-12
