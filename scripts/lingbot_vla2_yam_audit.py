#!/usr/bin/env python3
"""Audit the local LingBot-VLA2 foundation checkpoint and official V2 source."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/pretrained/lingbot-vla-v2-6b"
DEFAULT_SOURCE = REPO_ROOT / "XPolicyLab/policy/LingBot_VLA2/lingbot_vla_v2"


def _tensor_shape(checkpoint: Path, tensor_name: str) -> list[int]:
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    shard = checkpoint / index["weight_map"][tensor_name]
    with shard.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    return list(header[tensor_name]["shape"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()

    contract = {
        "checkpoint": str(checkpoint),
        "model_family": json.loads((checkpoint / "config.json").read_text()).get(
            "vlm_family"
        ),
        "state_projection_shape": _tensor_shape(checkpoint, "model.state_proj.weight"),
        "action_projection_shape": _tensor_shape(
            checkpoint, "model.action_out_proj.weight"
        ),
        "official_repository": "https://github.com/Robbyant/lingbot-vla-v2",
        "official_source_present": (
            args.source_root / "deploy/lingbot_vla_v2_policy.py"
        ).is_file(),
        "xpolicylab_adapter": "XPolicyLab/policy/LingBot_VLA2",
        "yam_norm_stats": None,
        "yam_action_mapping": "absolute 12 arm joints + 2 grippers",
        "status": "blocked",
        "reason": (
            "The official V2 source is public, but this local foundation checkpoint does "
            "not contain the lingbotvla_cli.yaml expected by the official loader, a "
            "YAM post-training checkpoint, or matched YAM normalization statistics."
        ),
    }
    print(json.dumps(contract, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
