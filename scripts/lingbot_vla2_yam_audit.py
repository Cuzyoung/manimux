#!/usr/bin/env python3
"""Audit why the local LingBot-VLA2 foundation checkpoint is not deployable on YAM yet."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/pretrained/lingbot-vla-v2-6b"


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
        "xpolicylab_adapter": None,
        "yam_norm_stats": None,
        "yam_action_mapping": None,
        "status": "blocked",
        "reason": (
            "XPolicyLab LingBot_VLA is the older Qwen2.5 implementation; this checkpoint "
            "is LingBot-VLA2/Qwen3 with a 55-D unified state/action interface and does "
            "not publish the YAM embodiment mapping, horizon, or normalization statistics."
        ),
    }
    print(json.dumps(contract, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
