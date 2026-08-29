#!/usr/bin/env python3
"""Discover stdout-based training runs and mirror them into TensorBoard.

LingBot and XR1 write native TensorBoard events. Pi05 and GR00T currently log
their scalar metrics to stdout, so this supervisor starts one existing tailer
per recognizable log file and keeps watching for new runs.
"""

from __future__ import annotations

import argparse
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PI05_METRIC_RE = re.compile(
    rb"Step\s+\d+:\s+grad_norm=[-+0-9.eE]+,\s+loss=[-+0-9.eE]+,\s+param_norm=[-+0-9.eE]+"
)
GR00T_METRIC_RE = re.compile(rb"\{[^\n]*'loss':\s*[-+0-9.eE]+[^\n]*\}")
PROBE_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class MetricRun:
    model: str
    log_path: Path
    output_path: Path
    tail_script: Path


def _tail_bytes(path: Path) -> bytes:
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - PROBE_BYTES))
        return stream.read()


def _contains_metrics(model: str, path: Path) -> bool:
    try:
        data = _tail_bytes(path)
    except (FileNotFoundError, PermissionError, OSError):
        return False
    pattern = PI05_METRIC_RE if model == "pi05" else GR00T_METRIC_RE
    return pattern.search(data) is not None


def discover_runs(data_root: Path, code_root: Path) -> list[MetricRun]:
    live_root = data_root / "runs/live-tensorboard"
    sources = (
        (
            "pi05",
            data_root / "runs/pi05",
            code_root / "scripts/tail_pi05_metrics.py",
        ),
        (
            "gr00t-n17",
            data_root / "runs/gr00t-n17",
            code_root / "scripts/tail_gr00t_metrics.py",
        ),
    )
    runs: list[MetricRun] = []
    for model, log_root, tail_script in sources:
        if not log_root.is_dir():
            continue
        for log_path in sorted(log_root.glob("*.log")):
            if log_path.name.endswith(".launch.log"):
                continue
            if not _contains_metrics(model, log_path):
                continue
            runs.append(
                MetricRun(
                    model=model,
                    log_path=log_path,
                    output_path=live_root / model / log_path.stem,
                    tail_script=tail_script,
                )
            )
    return runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.once:
        for run in discover_runs(args.data_root, args.code_root):
            print(f"{run.model}\t{run.log_path}\t{run.output_path}")
        return

    children: dict[Path, subprocess.Popen[bytes]] = {}
    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        while not stopping:
            for path, child in list(children.items()):
                if child.poll() is not None:
                    print(f"metric tailer exited rc={child.returncode}: {path}", flush=True)
                    del children[path]

            for run in discover_runs(args.data_root, args.code_root):
                if run.log_path in children:
                    continue
                run.output_path.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable,
                    str(run.tail_script),
                    "--log",
                    str(run.log_path),
                    "--output",
                    str(run.output_path),
                ]
                children[run.log_path] = subprocess.Popen(command)
                print(
                    f"discovered {run.model} run={run.log_path.stem} "
                    f"output={run.output_path}",
                    flush=True,
                )
            time.sleep(args.poll_seconds)
    finally:
        for child in children.values():
            child.terminate()
        for child in children.values():
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()


if __name__ == "__main__":
    main()
