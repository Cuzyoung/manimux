from __future__ import annotations

import json
from pathlib import Path

import pytest

from manimux.config import load_config
from scripts.gr00t_yam_server import _validate_checkpoint

STATE_KEYS = ("left_arm", "left_gripper", "right_arm", "right_gripper")
DIMS = {"left_arm": 6, "left_gripper": 1, "right_arm": 6, "right_gripper": 1}


def _write_checkpoint(root: Path) -> None:
    root.mkdir()
    model_config = {"architectures": ["Gr00tN1d7"], "num_inference_timesteps": 4}
    (root / "config.json").write_text(json.dumps(model_config), encoding="utf-8")
    modality = {
        "video": {
            "modality_keys": ["base_view", "left_wrist_view", "right_wrist_view"]
        },
        "state": {"modality_keys": list(STATE_KEYS)},
        "action": {
            "modality_keys": list(STATE_KEYS),
            "delta_indices": list(range(16)),
            "action_configs": [{"rep": "ABSOLUTE"} for _ in STATE_KEYS],
        },
    }
    processor = {"processor_kwargs": {"modality_configs": {"new_embodiment": modality}}}
    (root / "processor_config.json").write_text(json.dumps(processor), encoding="utf-8")
    field_names = ("min", "max", "mean", "std", "q01", "q99")
    statistics = {
        "new_embodiment": {
            scope: {
                key: {field: [0.0] * dim for field in field_names}
                for key, dim in DIMS.items()
            }
            for scope in ("state", "action")
        }
    }
    (root / "statistics.json").write_text(json.dumps(statistics), encoding="utf-8")
    (root / "model-00001-of-00001.safetensors").write_bytes(b"test")
    index = {"weight_map": {"model.test": "model-00001-of-00001.safetensors"}}
    (root / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")


def test_gr00t_checkpoint_contract(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)

    resolved, horizon = _validate_checkpoint(
        {"model_dir": str(checkpoint), "native_frequency_hz": 30.0}
    )

    assert resolved == checkpoint.resolve()
    assert horizon == 16


def test_gr00t_checkpoint_rejects_frequency_drift(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)

    with pytest.raises(ValueError, match="native_frequency_hz must be 30.0"):
        _validate_checkpoint({"model_dir": str(checkpoint), "native_frequency_hz": 20.0})


def test_gr00t_manimux_config_preserves_native_contract() -> None:
    config = load_config("configs/groot/yam/infra/manimux.yaml")

    assert config.policy.worker == "xpolicylab_ws"
    assert config.policy.adapter == "xpolicylab"
    assert config.policy.horizon_steps == 16
    assert config.policy.effective_action_dt_s == pytest.approx(1.0 / 30.0)
    assert config.robot.group_dims == {"left_arm": 7, "right_arm": 7}
    assert config.execution.runtime == "manimux"
