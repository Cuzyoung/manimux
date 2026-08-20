#!/usr/bin/env python3
"""Compatibility entry for the former Pi05-base server command."""

from __future__ import annotations

import sys
from pathlib import Path

from pi05_yam_server import main

if __name__ == "__main__":
    if not any(arg == "--config" or arg.startswith("--config=") for arg in sys.argv[1:]):
        default_config = Path(__file__).resolve().parents[1] / "configs/pi05/yam/server/base.yaml"
        sys.argv.extend(("--config", str(default_config)))
    raise SystemExit(main())
