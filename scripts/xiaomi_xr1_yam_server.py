#!/usr/bin/env python3
"""Launch Xiaomi-Robotics-1 through XPolicyLab's managed WebSocket server."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
XPOLICY_ROOT = REPO_ROOT / "XPolicyLab"
XR1_ROOT = XPOLICY_ROOT / "policy/Xiaomi_Robotics_1/xiaomi_robotics_1/xr1"
DEFAULT_CONFIG = REPO_ROOT / "configs/xiaomi-xr1/yam/server/xpolicy.yaml"


def _load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"server config must be a mapping: {path}")
    return loaded


def _shape(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    if not value:
        return (0,)
    child = _shape(value[0])
    if any(_shape(item) != child for item in value):
        raise ValueError("normalization arrays must be rectangular")
    return (len(value), *child)


def _validate(config: dict[str, Any]) -> dict[str, Any]:
    if not (XPOLICY_ROOT / ".git").exists():
        raise FileNotFoundError(f"XPolicyLab submodule is missing under {XPOLICY_ROOT}")
    if config.get("policy_name") != "Xiaomi_Robotics_1":
        raise ValueError("policy_name must be Xiaomi_Robotics_1")
    if config.get("protocol") != "ws":
        raise ValueError("protocol must be ws")
    if config.get("action_type") != "ee":
        raise ValueError("action_type must be ee; XR-1 does not emit arm joint targets")
    if config.get("output_format") != "packed_ee_delta":
        raise ValueError("output_format must be packed_ee_delta for ManiMux XR-1")
    if int(config.get("action_length", 0)) != 30:
        raise ValueError("action_length must be 30 for the released XR-1 checkpoint")

    checkpoint = Path(str(config["checkpoint_path"])).expanduser().resolve()
    if not checkpoint.is_file() or not zipfile.is_zipfile(checkpoint):
        raise FileNotFoundError(f"XR-1 checkpoint is missing or invalid: {checkpoint}")
    with zipfile.ZipFile(checkpoint) as archive:
        tensor_records = [name for name in archive.namelist() if "/data/" in name]
    if not tensor_records:
        raise ValueError(f"XR-1 checkpoint has no tensor records: {checkpoint}")

    processor = Path(str(config["vlm_processor_path"])).expanduser().resolve()
    for name in ("config.json", "tokenizer.json", "preprocessor_config.json"):
        if not (processor / name).is_file():
            raise FileNotFoundError(f"XR-1 processor is missing {processor / name}")

    stats_path = Path(str(config["norm_stats_path"])).expanduser().resolve()
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    expected = {
        "mean": (30, 60),
        "std": (30, 60),
        "q01": (1, 60),
        "q99": (1, 60),
    }
    shapes = {name: _shape(stats.get(name)) for name in expected}
    if shapes != expected:
        raise ValueError(f"XR-1 normalization shapes must be {expected}, got {shapes}")

    return {
        "xpolicylab_root": str(XPOLICY_ROOT),
        "policy_name": "Xiaomi_Robotics_1",
        "checkpoint": str(checkpoint),
        "checkpoint_tensor_records": len(tensor_records),
        "processor": str(processor),
        "norm_stats": str(stats_path),
        "model_action_space": "anchor_relative_ee_delta",
        "model_action_shape": [30, 60],
        "manimux_action_space": "absolute_joint_position",
        "manimux_action_shape": [30, 14],
        "denoise_steps": 5,
        "checkpoint_role": "post_training_start_point_not_yam_policy",
        "rtc_capability": "manimux_pi_guided_v1_extension",
    }


def _prepare_imports() -> None:
    for path in (REPO_ROOT, XPOLICY_ROOT, XR1_ROOT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check", action="store_true", help="validate without loading the model")
    args = parser.parse_args()

    config = _load_config(args.config.resolve())
    contract = _validate(config)
    if args.check:
        print(json.dumps(contract, indent=2))
        return 0

    _prepare_imports()
    import setup_policy_server

    setup_policy_server.main(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
