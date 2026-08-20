from __future__ import annotations

import pytest

import scripts.xiaomi_xr1_yam_server as xr1_server
from scripts.xiaomi_xr1_yam_server import _validate


def _minimal_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "policy_name": "Xiaomi_Robotics_1",
        "protocol": "ws",
        "action_type": "ee",
        "output_format": "packed_ee_delta",
        "action_length": 30,
    }
    config.update(overrides)
    return config


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"protocol": "http"}, "protocol must be ws"),
        ({"action_type": "joint"}, "action_type must be ee"),
        ({"action_length": 16}, "action_length must be 30"),
    ],
)
def test_xr1_server_rejects_contract_drift(
    override: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _validate(_minimal_config(**override))


def test_xr1_runtime_status_does_not_claim_inference_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    python = tmp_path / "env/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(xr1_server, "MODEL_PYTHON", python)

    config = xr1_server._load_config(
        xr1_server.DEFAULT_CONFIG
    )
    report = _validate(config)

    assert report["contract_status"] == "ready"
    assert report["runtime_status"] == "environment_present_gpu_forward_not_verified"
    assert report["inference_status"] == "not_verified"
    assert report["policy_status"] == "blocked_for_yam_task_without_finetune"
