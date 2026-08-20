from __future__ import annotations

import pytest

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
