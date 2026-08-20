from __future__ import annotations

import pytest

from scripts.lingbot_vla2_yam_server import _validate


def _minimal_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "policy_name": "LingBot_VLA2",
        "protocol": "ws",
        "action_type": "joint",
        "lingbot_vla2_root": "/missing/source",
        "bundle_manifest_path": "/missing/bundle.yaml",
    }
    config.update(overrides)
    return config


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"policy_name": "LingBot_VLA"}, "policy_name must be LingBot_VLA2"),
        ({"protocol": "http"}, "protocol must be ws"),
        ({"action_type": "ee"}, "action_type must be joint"),
    ],
)
def test_server_rejects_contract_drift(override, message) -> None:
    with pytest.raises(ValueError, match=message):
        _validate(_minimal_config(**override))


def test_missing_posttraining_bundle_is_blocked() -> None:
    report = _validate(_minimal_config())
    assert report["status"] == "blocked"
    assert report["rtc_capability"] == "pi_guided_v1_sampler"
    assert "missing files" in report["errors"][0]


def test_infra_must_match_bundle_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    import XPolicyLab.policy.LingBot_VLA2.model as adapter

    monkeypatch.setattr(
        adapter,
        "validate_bundle",
        lambda _: {
            "status": "ready",
            "errors": [],
            "native_hz": 20.0,
            "action_horizon": 25,
        },
    )
    infra = {
        "robot": {"control_hz": 30.0},
        "policy": {"horizon_steps": 50, "action_dt_s": 1 / 30},
        "execution": {"runtime": "manimux"},
    }
    report = _validate(_minimal_config(), infra)
    assert report["status"] == "blocked"
    assert len(report["errors"]) == 3


def test_structurally_feasible_rtc_config_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import XPolicyLab.policy.LingBot_VLA2.model as adapter

    monkeypatch.setattr(
        adapter,
        "validate_bundle",
        lambda _: {
            "status": "ready",
            "errors": [],
            "native_hz": 30.0,
            "action_horizon": 50,
            "rtc_capability": "pi_guided_v1_sampler",
        },
    )
    infra = {
        "robot": {"control_hz": 30.0},
        "policy": {"horizon_steps": 50, "action_dt_s": 1 / 30},
        "execution": {
            "runtime": "rtc",
            "rtc": {
                "initial_delay_steps": 12,
                "min_execute_steps": 20,
                "delay_buffer_size": 10,
                "beta": 5.0,
            },
        },
    }
    report = _validate(_minimal_config(), infra)
    assert report["status"] == "ready"
    assert report["rtc_capability"] == "pi_guided_v1_sampler"
