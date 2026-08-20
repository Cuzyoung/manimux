#!/usr/bin/env python3
"""Launch XPolicyLab GR00T_N17 with the YAM-finetuned checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
XPOLICY_ROOT = REPO_ROOT / "XPolicyLab"
GR00T_ROOT = XPOLICY_ROOT / "policy/GR00T_N17/gr00t_n17"
DEFAULT_CONFIG = REPO_ROOT / "configs/groot/yam/server/finetune.yaml"

EXPECTED_VIDEO_KEYS = ("base_view", "left_wrist_view", "right_wrist_view")
EXPECTED_STATE_KEYS = ("left_arm", "left_gripper", "right_arm", "right_gripper")
EXPECTED_DIMS = {
    "left_arm": 6,
    "left_gripper": 1,
    "right_arm": 6,
    "right_gripper": 1,
}
EXPECTED_HORIZON = 16
EXPECTED_FREQUENCY_HZ = 30.0


def _prepare_imports() -> None:
    for path in (REPO_ROOT, XPOLICY_ROOT, GR00T_ROOT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"server config must be a mapping: {path}")
    return loaded


def _validate_statistics(model_dir: Path) -> None:
    statistics = json.loads((model_dir / "statistics.json").read_text(encoding="utf-8"))
    yam = statistics.get("new_embodiment")
    if not isinstance(yam, dict):
        raise ValueError("GR00T checkpoint statistics have no new_embodiment entry")
    for modality in ("state", "action"):
        values = yam.get(modality)
        if not isinstance(values, dict):
            raise ValueError(f"GR00T checkpoint statistics have no {modality} mapping")
        if set(values) != set(EXPECTED_STATE_KEYS):
            raise ValueError(
                f"GR00T YAM {modality} statistics must be {EXPECTED_STATE_KEYS}, "
                f"got {tuple(values)}"
            )
        for key, expected_dim in EXPECTED_DIMS.items():
            record = values.get(key)
            if not isinstance(record, dict):
                raise ValueError(f"GR00T YAM {modality}.{key} statistics are missing")
            for field in ("q01", "q99"):
                field_values = record.get(field)
                if not isinstance(field_values, list) or len(field_values) != expected_dim:
                    raise ValueError(
                        f"GR00T YAM {modality}.{key}.{field} must have "
                        f"{expected_dim} values"
                    )


def _validate_checkpoint(config: dict[str, Any]) -> tuple[Path, int]:
    if not (XPOLICY_ROOT / ".git").exists():
        raise FileNotFoundError(
            f"XPolicyLab submodule is missing under {XPOLICY_ROOT}; initialize submodules first"
        )
    model_dir = Path(str(config["model_dir"])).expanduser().resolve()
    for name in ("config.json", "processor_config.json", "statistics.json"):
        if not (model_dir / name).is_file():
            raise FileNotFoundError(f"GR00T checkpoint is missing {model_dir / name}")

    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"GR00T checkpoint is missing {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = sorted(set(index.get("weight_map", {}).values()))
    if not shards:
        raise ValueError(f"GR00T checkpoint has an empty weight map: {index_path}")
    missing_shards = [name for name in shards if not (model_dir / name).is_file()]
    if missing_shards:
        raise FileNotFoundError(f"GR00T checkpoint is missing shards: {missing_shards}")

    model_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if model_config.get("architectures") != ["Gr00tN1d7"]:
        raise ValueError("GR00T YAM checkpoint architecture must be Gr00tN1d7")
    if model_config.get("num_inference_timesteps") != 4:
        raise ValueError("GR00T YAM checkpoint must use 4 inference timesteps")

    processor = json.loads((model_dir / "processor_config.json").read_text(encoding="utf-8"))
    modality_configs = processor.get("processor_kwargs", {}).get("modality_configs", {})
    yam = modality_configs.get("new_embodiment")
    if not isinstance(yam, dict):
        raise ValueError("GR00T checkpoint has no new_embodiment modality config")
    video_keys = tuple(yam["video"]["modality_keys"])
    state_keys = tuple(yam["state"]["modality_keys"])
    action_keys = tuple(yam["action"]["modality_keys"])
    if video_keys != EXPECTED_VIDEO_KEYS:
        raise ValueError(f"GR00T YAM video keys must be {EXPECTED_VIDEO_KEYS}, got {video_keys}")
    if state_keys != EXPECTED_STATE_KEYS or action_keys != EXPECTED_STATE_KEYS:
        raise ValueError(
            f"GR00T YAM state/action keys must be {EXPECTED_STATE_KEYS}, "
            f"got state={state_keys}, action={action_keys}"
        )
    action_configs = yam["action"].get("action_configs", [])
    if len(action_configs) != len(EXPECTED_STATE_KEYS):
        raise ValueError("GR00T YAM checkpoint must define four action configs")
    representations = {
        str(action_config.get("rep", "")).lower()
        for action_config in action_configs
    }
    if representations != {"absolute"}:
        raise ValueError(f"GR00T YAM actions must be absolute, got {sorted(representations)}")
    horizon = len(yam["action"]["delta_indices"])
    if horizon != EXPECTED_HORIZON:
        raise ValueError(
            f"GR00T YAM action horizon must be {EXPECTED_HORIZON}, got {horizon}"
        )
    frequency_hz = float(config.get("native_frequency_hz", 0.0))
    if frequency_hz != EXPECTED_FREQUENCY_HZ:
        raise ValueError(
            f"GR00T YAM native_frequency_hz must be {EXPECTED_FREQUENCY_HZ}, "
            f"got {frequency_hz}"
        )
    _validate_statistics(model_dir)
    return model_dir, horizon


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check", action="store_true", help="validate and print resolved setup")
    args = parser.parse_args()

    config = _load_config(args.config.resolve())
    model_dir, horizon = _validate_checkpoint(config)
    contract = {
        "xpolicylab_root": str(XPOLICY_ROOT),
        "policy_name": "GR00T_N17",
        "model_root": str(model_dir),
        "embodiment_tag": config.get("embodiment_tag", "NEW_EMBODIMENT"),
        "action_space": "absolute_joint_position",
        "action_dimension": sum(EXPECTED_DIMS.values()),
        "action_horizon": horizon,
        "native_frequency_hz": EXPECTED_FREQUENCY_HZ,
        "inference_timesteps": 4,
        "checkpoint_role": "yam_finetune_on_molmoact2_data",
        "normalization": str(model_dir / "statistics.json"),
        "rtc": False,
        "cosmos_model": config.get("cosmos_model_path", "nvidia/Cosmos-Reason2-2B"),
    }
    if args.check:
        print(json.dumps(contract, indent=2))
        return 0

    _prepare_imports()
    import setup_policy_server

    setup_policy_server.main(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
