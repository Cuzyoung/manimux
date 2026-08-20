#!/usr/bin/env python3
"""Check or launch the official LingBot-VLA2 XPolicyLab adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
XPOLICY_ROOT = REPO_ROOT / "XPolicyLab"
DEFAULT_CONFIG = REPO_ROOT / "configs/lingbot-vla2/yam/server/xpolicy.yaml"


def _load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"server config must be a mapping: {path}")
    return loaded


def _prepare_imports() -> None:
    for path in (REPO_ROOT, XPOLICY_ROOT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _validate(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("policy_name") != "LingBot_VLA2":
        raise ValueError("policy_name must be LingBot_VLA2")
    if config.get("protocol") != "ws":
        raise ValueError("protocol must be ws")
    if config.get("action_type") != "joint":
        raise ValueError("action_type must be joint")
    if int(config.get("action_horizon", 0)) <= 0:
        raise ValueError("action_horizon must be positive")
    _prepare_imports()
    from XPolicyLab.policy.LingBot_VLA2.model import validate_bundle

    report = validate_bundle(config)
    report.update(
        {
            "policy_name": "LingBot_VLA2",
            "official_repository": "https://github.com/Robbyant/lingbot-vla-v2",
            "model_action_shape": [int(config["action_horizon"]), 14],
            "manimux_action_space": "absolute_joint_position",
            "rtc_capability": "not_integrated",
        }
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--serve",
        action="store_true",
        help="load the model and start XPolicyLab; without this flag only check files",
    )
    args = parser.parse_args()

    config = _load_config(args.config.resolve())
    report = _validate(config)
    print(json.dumps(report, indent=2))
    if report["status"] != "ready":
        return 2
    if not args.serve:
        return 0

    import setup_policy_server

    setup_policy_server.main(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
