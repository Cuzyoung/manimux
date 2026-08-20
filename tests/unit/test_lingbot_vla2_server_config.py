from __future__ import annotations

import pytest

from scripts.lingbot_vla2_yam_server import _validate


def _minimal_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "policy_name": "LingBot_VLA2",
        "protocol": "ws",
        "action_type": "joint",
        "action_horizon": 50,
        "lingbot_vla2_root": "/missing/source",
        "checkpoint_path": "/missing/checkpoint",
        "robot_config_path": "/missing/robot.yaml",
        "norm_stats_path": "/missing/stats.json",
    }
    config.update(overrides)
    return config


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"policy_name": "LingBot_VLA"}, "policy_name must be LingBot_VLA2"),
        ({"protocol": "http"}, "protocol must be ws"),
        ({"action_type": "ee"}, "action_type must be joint"),
        ({"action_horizon": 0}, "action_horizon must be positive"),
    ],
)
def test_server_rejects_contract_drift(override, message) -> None:
    with pytest.raises(ValueError, match=message):
        _validate(_minimal_config(**override))


def test_missing_posttraining_bundle_is_blocked() -> None:
    report = _validate(_minimal_config())
    assert report["status"] == "blocked"
    assert report["rtc_capability"] == "not_integrated"
    assert "missing files" in report["errors"][0]
