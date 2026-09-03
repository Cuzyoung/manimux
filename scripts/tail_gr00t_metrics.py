#!/usr/bin/env python3
"""Mirror GR00T Trainer stdout metrics into TensorBoard events."""

from __future__ import annotations

import argparse
import ast
import re
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


PROGRESS_RE = re.compile(r"(?P<step>\d+)/(?P<total>\d+)")
METRICS_RE = re.compile(r"\{[^\n]*'loss':\s*[-+0-9.eE]+[^\n]*\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "last_step.txt"
    last_written = int(state_path.read_text()) if state_path.exists() else -1

    while not args.log.exists():
        time.sleep(args.poll_seconds)

    current_step = -1
    with args.log.open("r", errors="replace") as stream, SummaryWriter(
        log_dir=args.output
    ) as writer:
        while True:
            wrote = False
            for line in stream:
                for progress in PROGRESS_RE.finditer(line):
                    current_step = max(current_step, int(progress.group("step")))
                for raw in METRICS_RE.findall(line):
                    metrics = ast.literal_eval(raw)
                    if current_step <= last_written:
                        continue
                    for key in ("loss", "grad_norm", "learning_rate"):
                        if key in metrics:
                            writer.add_scalar(f"training/{key}", float(metrics[key]), current_step)
                    last_written = current_step
                    wrote = True
            if wrote:
                writer.flush()
                state_path.write_text(f"{last_written}\n")
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
