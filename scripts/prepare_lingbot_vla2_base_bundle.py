#!/usr/bin/env python3
"""Build the LingBot-VLA2 base-checkpoint bundle with YAM projection stats."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "XPolicyLab/policy/LingBot_VLA2/lingbot_vla_v2"
SOURCE_REVISION = "951475ae1b1d87553e7dc47c97b53a3d695c0d13"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/pretrained/lingbot-vla-v2-6b"
DEFAULT_PROCESSOR = REPO_ROOT / "checkpoints/pretrained/xiaomi/qwen3_vl_4b_processor"
DEFAULT_STATS = (
    REPO_ROOT
    / "src/manimux/integrations/lingbot_vla2_yam/norm_stats/yam_60ep.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "checkpoints/pretrained/lingbot-vla-v2-6b-yam-projection"
ROBOT_CONFIG = REPO_ROOT / "XPolicyLab/policy/LingBot_VLA2/robot_configs/yam_dual_absolute.yaml"
TRAINING_TEMPLATE = SOURCE_ROOT / "configs/vla/real_robot/real_robot.yaml"


def _relative_symlink(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(f"refusing to replace non-symlink artifact: {destination}")
    destination.symlink_to(os.path.relpath(source, destination.parent))


def prepare_bundle(
    *,
    checkpoint: Path,
    processor: Path,
    stats: Path,
    output: Path,
) -> Path:
    checkpoint = checkpoint.resolve()
    processor = processor.resolve()
    stats = stats.resolve()
    output = output.resolve()
    for required in (
        checkpoint / "model.safetensors.index.json",
        processor / "config.json",
        processor / "tokenizer.json",
        stats,
        ROBOT_CONFIG,
        TRAINING_TEMPLATE,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    linked_checkpoint = output / "runs/yam/hf_ckpt"
    linked_checkpoint.mkdir(parents=True, exist_ok=True)
    for artifact in checkpoint.iterdir():
        _relative_symlink(artifact, linked_checkpoint / artifact.name)

    training_config = yaml.safe_load(TRAINING_TEMPLATE.read_text(encoding="utf-8"))
    training_config["model"]["model_path"] = str(checkpoint)
    training_config["model"]["tokenizer_path"] = str(processor)
    training_config["data"]["data_name"] = "yam_dual"
    training_config["data"]["train_path"] = "base-inference-no-training-data"
    training_config["data"]["norm_stats_file"] = "norm_stats.json"
    training_config["train"]["chunk_size"] = 50
    (output / "lingbotvla_cli.yaml").write_text(
        yaml.safe_dump(training_config, sort_keys=False), encoding="utf-8"
    )
    shutil.copy2(stats, output / "norm_stats.json")
    shutil.copy2(ROBOT_CONFIG, output / "robot_config.yaml")

    manifest = {
        "schema_version": "xpolicylab.lingbot_vla2_yam_bundle.v1",
        "model": {
            "family": "lingbot-vla-v2",
            "official_source_revision": SOURCE_REVISION,
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
            "cameras": ["camera_top", "camera_wrist_left", "camera_wrist_right"],
        },
    }
    manifest_path = output / "bundle.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--processor", type=Path, default=DEFAULT_PROCESSOR)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = prepare_bundle(
        checkpoint=args.checkpoint,
        processor=args.processor,
        stats=args.stats,
        output=args.output,
    )
    print(f"wrote {manifest}")
    print("checkpoint_variant=lingbot_vla2_6b_base_with_yam_stats")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
