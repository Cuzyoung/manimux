from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from manimux.config import ManiMuxConfig, load_config
from manimux.plugins import PluginError
from manimux.runtime import build_runtime


def test_mock_config_loads() -> None:
    config = load_config(Path("configs/mock.yaml"))
    assert config.robot.driver == "mock_dual_arm"
    assert config.execution.executor == "smooth"
    assert config.execution.inference_schedule == "deadline"
    assert config.robot.group_dims["left_arm"] == 6


def test_total_trajectory_duration_overrides_point_spacing() -> None:
    config = load_config(Path("configs/molmoact2/yam/infra/manimux.yaml"))

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


def test_all_infra_configs_load() -> None:
    for path in sorted(Path("configs").glob("*/yam/infra/*.yaml")):
        load_config(path)


def test_rtc_rejects_default_scheduler_fields_that_it_does_not_use() -> None:
    config = load_config(Path("configs/mock.yaml"))
    payload = config.model_dump(mode="python")
    payload["execution"]["runtime"] = "rtc"
    payload["execution"]["refill_threshold_s"] = 0.2

    with pytest.raises(ValidationError, match="not used by RTC"):
        ManiMuxConfig.model_validate(payload)


def test_recording_cannot_be_silently_disabled() -> None:
    config = load_config(Path("configs/mock.yaml"))
    payload = config.model_dump(mode="python")
    payload["recording"]["enabled"] = False

    with pytest.raises(ValidationError):
        ManiMuxConfig.model_validate(payload)


def test_unknown_inference_strategy_fails_before_runtime_construction(tmp_path: Path) -> None:
    config = load_config(Path("configs/mock.yaml"))
    config.execution.runtime = "missing_strategy"

    with pytest.raises(PluginError, match="manimux.inference_strategies"):
        build_runtime(config, tmp_path)
