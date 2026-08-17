from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from manimux.config import ManiMuxConfig, load_config


def test_mock_config_loads() -> None:
    config = load_config(Path("configs/mock.yaml"))
    assert config.robot.driver == "mock_dual_arm"
    assert config.execution.executor == "smooth"
    assert config.execution.inference_schedule == "deadline"
    assert config.robot.group_dims["left_arm"] == 6


def test_total_trajectory_duration_overrides_point_spacing() -> None:
    config = load_config(Path("configs/molmoact-yam-live.yaml"))

    assert config.policy.trajectory_duration_s is None
    assert config.policy.effective_action_dt_s == pytest.approx(0.05)
    assert config.execution.smooth.max_velocity == 0.25
    assert config.execution.smooth.max_acceleration == 0.5


def test_unknown_config_field_fails() -> None:
    with pytest.raises(ValidationError):
        ManiMuxConfig.model_validate(
            {
                "run": {"task": "x", "unknown": True},
                "robot": {"driver": "mock", "group_dims": {"arm": 1}},
                "policy": {"worker": "fake", "adapter": "identity"},
            }
        )
