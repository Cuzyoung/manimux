#!/usr/bin/env python3
"""Wait for GR00T deployment checkpoints and download each one immediately."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_FILES = (
    "config.json",
    "embodiment_id.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "processor_config.json",
    "statistics.json",
    "trainer_state.json",
    "training_args.bin",
    "experiment_cfg/conf.yaml",
    "experiment_cfg/config.yaml",
    "experiment_cfg/dataset_statistics.json",
    "experiment_cfg/final_model_config.json",
    "experiment_cfg/final_processor_config.json",
)


def run(*args: str, input_text: str | None = None) -> str:
    result = subprocess.run(
        args,
        input=input_text,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout


def remote(host: str, command: str) -> str:
    return run("ssh", "-o", "BatchMode=yes", host, command).strip()


def find_ready_checkpoint(host: str, root: str, step: int) -> tuple[str, dict[str, int]] | None:
    command = f"find {shlex.quote(root)} -type d -name checkpoint-{step} -print -quit"
    checkpoint = remote(host, command)
    if not checkpoint:
        return None

    sizes: dict[str, int] = {}
    for relative in REQUIRED_FILES:
        path = f"{checkpoint}/{relative}"
        output = remote(host, f"test -s {shlex.quote(path)} && stat -c %s {shlex.quote(path)} || true")
        if not output:
            return None
        sizes[relative] = int(output)
    return checkpoint, sizes


def resume_file(host: str, remote_path: str, local_path: Path, expected_size: int) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    while not local_path.exists() or local_path.stat().st_size != expected_size:
        batch = f'reget "{remote_path}" "{local_path}"\n'
        try:
            run("sftp", "-b", "-", host, input_text=batch)
        except subprocess.CalledProcessError as error:
            print(error.stdout, flush=True)
            time.sleep(10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost-3338")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=(1000, 2000, 3000))
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for step in args.steps:
        destination = args.local_root / f"checkpoint-{step}"
        marker = destination / "DOWNLOAD_COMPLETE.json"
        if marker.exists():
            print(f"checkpoint-{step} already downloaded", flush=True)
            continue

        print(f"waiting for checkpoint-{step}", flush=True)
        ready = None
        while ready is None:
            try:
                ready = find_ready_checkpoint(args.host, args.remote_root, step)
            except (subprocess.CalledProcessError, ValueError) as error:
                print(f"checkpoint-{step} monitor retry: {error}", flush=True)
            if ready is None:
                time.sleep(args.poll_seconds)

        checkpoint, sizes = ready
        print(f"checkpoint-{step} complete remotely; downloading deployment bundle", flush=True)
        for relative, expected_size in sizes.items():
            resume_file(
                args.host,
                f"{checkpoint}/{relative}",
                destination / relative,
                expected_size,
            )

        marker.write_text(
            json.dumps(
                {
                    "remote_checkpoint": checkpoint,
                    "step": step,
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                    "files": sizes,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"checkpoint-{step} downloaded to {destination}", flush=True)


if __name__ == "__main__":
    main()
