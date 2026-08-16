from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from manimux.config import load_config
from manimux.runtime.edge import EdgeRuntime


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _run(config_path: Path, executor: str | None = None) -> int:
    config = load_config(config_path)
    if executor is not None:
        if executor not in {"smooth", "mpc"}:
            raise ValueError(f"unsupported executor override {executor!r}")
        config.execution.executor = cast(Literal["smooth", "mpc"], executor)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"run-{timestamp}-{uuid.uuid4().hex[:8]}"
    run_dir = config.run.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "run.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "git_sha": _git_sha(),
                "config": config.model_dump(mode="json"),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    result = EdgeRuntime(config, run_dir).run()
    print(
        f"completed {result.steps} steps; accepted={result.accepted_plans} "
        f"rejected={result.rejected_plans}; episode={result.episode_dir}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="manimux")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run one local robot-policy session")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--executor", choices=("smooth", "mpc"))
    return parser


def main(argv: list[str] | None = None) -> int:
    mp.freeze_support()
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return _run(args.config, args.executor)
    raise AssertionError(f"unhandled command {args.command}")
