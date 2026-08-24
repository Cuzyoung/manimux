from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from manimux.config import ManiMuxConfig, load_config
from manimux.runtime import build_runtime
from manimux.session import RuntimeSessionService


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_config(config_path: Path, executor: str | None = None) -> ManiMuxConfig:
    config = load_config(config_path)
    if executor is not None:
        if executor not in {"smooth", "mpc"}:
            raise ValueError(f"unsupported executor override {executor!r}")
        config.execution.executor = cast(Literal["smooth", "mpc"], executor)
    return config


def _create_run_dir(config: ManiMuxConfig, *, mode: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"run-{timestamp}-{uuid.uuid4().hex[:8]}"
    run_dir = config.run.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "run.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_id": run_id,
                "mode": mode,
                "created_at": datetime.now(UTC).isoformat(),
                "git_sha": _git_sha(),
                "config": config.model_dump(mode="json"),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    return run_dir


def _run(config_path: Path, executor: str | None = None) -> int:
    config = _load_config(config_path, executor)
    run_dir = _create_run_dir(config, mode="run")
    try:
        result = build_runtime(config, run_dir).run()
    except KeyboardInterrupt:
        print("interrupted; robot shutdown and partial episode save completed")
        return 130
    status = "completed" if result.success else "FAULT"
    print(
        f"{status} {result.steps} steps; reason={result.terminal_reason}; "
        f"accepted={result.accepted_plans} rejected={result.rejected_plans}; "
        f"episode={result.episode_dir}"
    )
    return 0 if result.success else 2


def _serve(config_path: Path, executor: str | None = None) -> int:
    config = _load_config(config_path, executor)
    run_dir = _create_run_dir(config, mode="serve")
    service = RuntimeSessionService(config, run_dir)
    try:
        service.serve()
    except KeyboardInterrupt:
        print("runtime service stopped by operator")
        return 130
    return 0


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--executor", choices=("smooth", "mpc"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="manimux")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run one local robot-policy session")
    _add_runtime_arguments(run_parser)
    serve_parser = subparsers.add_parser(
        "serve", help="keep one experiment session available for Viewer-controlled rollouts"
    )
    _add_runtime_arguments(serve_parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    mp.freeze_support()
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return _run(args.config, args.executor)
    if args.command == "serve":
        return _serve(args.config, args.executor)
    raise AssertionError(f"unhandled command {args.command}")
