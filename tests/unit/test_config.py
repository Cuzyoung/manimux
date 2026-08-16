from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from manimux.config import ManiMuxConfig, load_config


def test_mock_config_loads() -> None:
    config = load_config(Path("configs/mock.yaml"))
    assert config.robot.driver == "mock_dual_arm"
    assert config.execution.executor == "smooth"
    assert config.robot.group_dims["left_arm"] == 6


def test_unknown_config_field_fails() -> None:
    with pytest.raises(ValidationError):
        ManiMuxConfig.model_validate(
            {
                "run": {"task": "x", "unknown": True},
                "robot": {"driver": "mock", "group_dims": {"arm": 1}},
                "policy": {"worker": "fake", "adapter": "identity"},
            }
        )
