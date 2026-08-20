from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

MODEL_PATH = Path(__file__).resolve().parents[2] / "XPolicyLab/policy/LingBot_VLA2/model.py"


def _load_model_module():
    spec = importlib.util.spec_from_file_location("lingbot_vla2_adapter_test", MODEL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observation():
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    return {
        "vision": {
            "cam_head": {"color": image},
            "cam_left_wrist": {"color": image},
            "cam_right_wrist": {"color": image},
        },
        "state": {
            "left_arm_joint_state": np.arange(6, dtype=np.float32),
            "left_ee_joint_state": np.array([6], dtype=np.float32),
            "right_arm_joint_state": np.arange(10, 16, dtype=np.float32),
            "right_ee_joint_state": np.array([16], dtype=np.float32),
        },
        "instruction": "pick the block",
    }


def test_encode_yam_observation_uses_official_feature_names() -> None:
    module = _load_model_module()
    encoded = module.encode_observation(_observation(), "fallback")
    np.testing.assert_array_equal(
        encoded["observation.state.arm.position"],
        np.array([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        encoded["observation.state.effector.position"],
        np.array([6, 16], dtype=np.float32),
    )
    assert encoded["task"] == "pick the block"


def test_decode_absolute_joint_chunk() -> None:
    module = _load_model_module()
    arms = np.arange(36, dtype=np.float32).reshape(3, 12)
    effectors = np.arange(6, dtype=np.float32).reshape(3, 2)
    actions = module.decode_actions(
        {
            "action.arm.position": arms,
            "action.effector.position": effectors,
        }
    )
    assert len(actions) == 3
    np.testing.assert_array_equal(actions[1]["left_arm_joint_state"], arms[1, :6])
    np.testing.assert_array_equal(actions[1]["right_arm_joint_state"], arms[1, 6:])
    np.testing.assert_array_equal(actions[1]["left_ee_joint_state"], effectors[1, :1])
    np.testing.assert_array_equal(actions[1]["right_ee_joint_state"], effectors[1, 1:])


def test_decode_rejects_wrong_action_width() -> None:
    module = _load_model_module()
    with pytest.raises(ValueError, match="must be \\(H, 12\\)"):
        module.decode_actions(
            {
                "action.arm.position": np.zeros((2, 14), dtype=np.float32),
                "action.effector.position": np.zeros((2, 2), dtype=np.float32),
            }
        )


def test_validate_complete_bundle_without_loading_model(tmp_path: Path) -> None:
    module = _load_model_module()
    source_root = tmp_path / "source"
    (source_root / "deploy").mkdir(parents=True)
    (source_root / "lingbotvla/data/vla_data").mkdir(parents=True)
    (source_root / "deploy/lingbot_vla_v2_policy.py").touch()
    (source_root / "lingbotvla/data/vla_data/utils.py").touch()

    bundle_root = tmp_path / "bundle"
    checkpoint = bundle_root / "runs/yam/hf_ckpt"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model-00001-of-00001.safetensors").touch()
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.weight": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )
    (bundle_root / "lingbotvla_cli.yaml").write_text(
        yaml.safe_dump(
            {
                "data": {"cameras": ["camera_top", "camera_wrist_left", "camera_wrist_right"]},
                "train": {
                    "action_dim": 55,
                    "max_action_dim": 55,
                    "max_state_dim": 55,
                    "chunk_size": 50,
                },
            }
        ),
        encoding="utf-8",
    )
    stats_path = bundle_root / "norm_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "norm_stats": {
                    "observation.state.arm.position": {"mean": [0] * 12, "std": [1] * 12},
                    "observation.state.effector.position": {"mean": [0] * 2, "std": [1] * 2},
                    "action.arm.position": {"mean": [0] * 12, "std": [1] * 12},
                    "action.effector.position": {"mean": [0] * 2, "std": [1] * 2},
                }
            }
        ),
        encoding="utf-8",
    )

    report = module.validate_bundle(
        {
            "lingbot_vla2_root": source_root,
            "checkpoint_path": checkpoint,
            "robot_config_path": MODEL_PATH.parent / "robot_configs/yam_dual_absolute.yaml",
            "norm_stats_path": stats_path,
            "action_horizon": 50,
        }
    )
    assert report["status"] == "ready"
    assert report["errors"] == []
