"""Execution runtimes.

``manimux`` is the original time-indexed runtime with a smoothing executor.
``rtc`` is Physical Intelligence real-time chunking. A run picks one with
``execution.runtime``; the default keeps the original behaviour byte for byte.
"""

from __future__ import annotations

from pathlib import Path

from manimux.config import ManiMuxConfig
from manimux.runtime.edge import EdgeRuntime, RunResult


def build_runtime(config: ManiMuxConfig, run_dir: Path) -> EdgeRuntime:
    if config.execution.runtime == "manimux":
        return EdgeRuntime(config, run_dir)
    from manimux.runtime.rtc import RtcRuntime

    return RtcRuntime(config, run_dir)


__all__ = ["EdgeRuntime", "RunResult", "build_runtime"]
