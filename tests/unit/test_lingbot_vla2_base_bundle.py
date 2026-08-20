from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from manimux.integrations.lingbot_vla2_yam.compute_norm_stats import (
    compute_stats,
    episode_features,
    usable_episodes,
)
from scripts.prepare_lingbot_vla2_base_bundle import prepare_bundle


def _write_episode(path: Path) -> None:
    path.mkdir(parents=True)
    path.joinpath("write_complete.flag").touch()
    arrays = {
        "left-joint_pos.npy": np.arange(18, dtype=np.float32).reshape(3, 6),
        "right-joint_pos.npy": np.arange(18, 36, dtype=np.float32).reshape(3, 6),
        "left-gripper_pos.npy": np.array([[0.0], [0.5], [1.0]], dtype=np.float32),
        "right-gripper_pos.npy": np.array([[1.0], [0.5], [0.0]], dtype=np.float32),
        "action-left-joint.npy": np.arange(1, 19, dtype=np.float32).reshape(3, 6),
        "action-right-joint.npy": np.arange(19, 37, dtype=np.float32).reshape(3, 6),
        "action-left-gripper.npy": np.array([[0.1], [0.6], [0.9]], dtype=np.float32),
        "action-right-gripper.npy": np.array([[0.9], [0.4], [0.1]], dtype=np.float32),
    }
    for name, values in arrays.items():
        np.save(path / name, values)


def test_lingbot_stats_use_absolute_yam_joint_features(tmp_path: Path) -> None:
    episode = tmp_path / "task/episode"
    _write_episode(episode)

    assert usable_episodes(tmp_path) == [episode]
    features = episode_features(episode)
    assert features["observation.state.arm.position"].shape == (3, 12)
    assert features["action.effector.position"].shape == (3, 2)

    stats = compute_stats([episode])
    assert stats["count"] == 3
    assert len(stats["norm_stats"]["action.arm.position"]["mean"]) == 12
    assert len(stats["norm_stats"]["action.effector.position"]["std"]) == 2


def test_prepare_base_bundle_reuses_checkpoint_with_symlinks(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "base"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"tensor": "model-00001.safetensors"}})
    )
    (checkpoint / "model-00001.safetensors").write_bytes(b"weights")
    processor = tmp_path / "qwen3_vl_processor"
    processor.mkdir()
    for name in ("config.json", "tokenizer.json"):
        (processor / name).write_text("{}")
    stats = tmp_path / "norm_stats.json"
    stats.write_text(json.dumps({"norm_stats": {}, "count": 0}))

    output = tmp_path / "bundle"
    manifest_path = prepare_bundle(
        checkpoint=checkpoint,
        processor=processor,
        stats=stats,
        output=output,
    )

    manifest = yaml.safe_load(manifest_path.read_text())
    training = yaml.safe_load((output / "lingbotvla_cli.yaml").read_text())
    linked_shard = output / "runs/yam/hf_ckpt/model-00001.safetensors"
    assert manifest["artifacts"]["checkpoint"] == "runs/yam/hf_ckpt"
    assert training["data"]["data_name"] == "yam_dual"
    assert training["train"]["chunk_size"] == 50
    assert linked_shard.is_symlink()
    assert linked_shard.resolve() == checkpoint / "model-00001.safetensors"
