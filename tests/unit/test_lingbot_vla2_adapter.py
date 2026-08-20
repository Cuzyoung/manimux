from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from jsonschema import Draft202012Validator

MODEL_PATH = Path(__file__).resolve().parents[2] / "XPolicyLab/policy/LingBot_VLA2/model.py"
REPO_ROOT = Path(__file__).resolve().parents[2]


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


def test_validate_complete_bundle_without_loading_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_model_module()
    source_root = tmp_path / "source"
    (source_root / "deploy").mkdir(parents=True)
    (source_root / "lingbotvla/data/vla_data").mkdir(parents=True)
    (source_root / "deploy/lingbot_vla_v2_policy.py").touch()
    (source_root / "lingbotvla/data/vla_data/utils.py").touch()
    revision = "951475ae1b1d87553e7dc47c97b53a3d695c0d13"
    monkeypatch.setattr(module, "_source_revision", lambda _: revision)

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
                "model": {
                    "config_key": "LingbotVLAV2Config",
                    "post_training": True,
                },
                "data": {
                    "cameras": ["camera_top", "camera_wrist_left", "camera_wrist_right"],
                    "joints": module.EXPECTED_JOINTS,
                },
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
    robot_config = bundle_root / "robot_config.yaml"
    robot_config.write_text(
        (MODEL_PATH.parent / "robot_configs/yam_dual_absolute.yaml").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    manifest_path = bundle_root / "bundle.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": module.BUNDLE_SCHEMA_VERSION,
                "model": {
                    "family": "lingbot-vla-v2",
                    "official_source_revision": revision,
                },
                "artifacts": {
                    "training_config": "lingbotvla_cli.yaml",
                    "checkpoint": "runs/yam/hf_ckpt",
                    "norm_stats": "norm_stats.json",
                    "robot_config": "robot_config.yaml",
                },
                "control": {
                    "native_hz": 30.0,
                    "action_horizon": 50,
                    "action_space": "absolute_joint_position",
                },
                "embodiment": {
                    "name": "yam_dual",
                    "arm_dofs": [6, 6],
                    "gripper_dofs": [1, 1],
                    "cameras": module.EXPECTED_CAMERAS,
                },
            }
        ),
        encoding="utf-8",
    )

    report = module.validate_bundle(
        {
            "lingbot_vla2_root": source_root,
            "bundle_manifest_path": manifest_path,
        }
    )
    assert report["status"] == "ready"
    assert report["errors"] == []
    assert report["native_hz"] == 30.0
    assert report["action_horizon"] == 50
    assert report["checkpoint_path"] == str(checkpoint)


def test_bundle_artifact_cannot_escape_root(tmp_path: Path) -> None:
    module = _load_model_module()
    with pytest.raises(ValueError, match="escapes the bundle root"):
        module._bundle_artifact(tmp_path, "../weights", name="checkpoint")


def test_bundle_schema_and_yaml_example_stay_in_sync() -> None:
    module = _load_model_module()
    schema = json.loads(
        (MODEL_PATH.parent / "bundle.schema.json").read_text(encoding="utf-8")
    )
    example = yaml.safe_load(
        (REPO_ROOT / "configs/lingbot-vla2/yam/bundle.example.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert schema["properties"]["schema_version"]["const"] == module.BUNDLE_SCHEMA_VERSION
    assert example["schema_version"] == module.BUNDLE_SCHEMA_VERSION
    assert example["embodiment"]["cameras"] == module.EXPECTED_CAMERAS
    assert example["artifacts"]["checkpoint"] == "runs/yam/hf_ckpt"
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(example)
