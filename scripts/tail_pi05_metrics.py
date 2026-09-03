#!/usr/bin/env python3
"""Mirror Pi05 stdout metrics into a live TensorBoard event stream."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


STEP_RE = re.compile(
    r"Step\s+(?P<step>\d+):\s+"
    r"grad_norm=(?P<grad_norm>[-+0-9.eE]+),\s+"
    r"loss=(?P<loss>[-+0-9.eE]+),\s+"
    r"param_norm=(?P<param_norm>[-+0-9.eE]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "last_step.txt"
    last_step = int(state_path.read_text()) if state_path.exists() else -1

    with args.log.open("r", errors="replace") as stream, SummaryWriter(
        log_dir=args.output
    ) as writer:
        while True:
            wrote = False
            for line in stream:
                match = STEP_RE.search(line)
                if match is None:
                    continue
                step = int(match.group("step"))
                if step <= last_step:
                    continue
                writer.add_scalar("training/loss", float(match.group("loss")), step)
                writer.add_scalar(
                    "training/grad_norm", float(match.group("grad_norm")), step
                )
                writer.add_scalar(
                    "training/param_norm", float(match.group("param_norm")), step
                )
                last_step = step
                wrote = True
            if wrote:
                writer.flush()
                state_path.write_text(f"{last_step}\n")
            if args.once:
                return
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
