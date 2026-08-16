"""Run ManiMux's bundled MolmoAct YAM integration with the viewer observer.

The policy, asynchronous inference, action stitching, robot execution, saving,
labeling, and parking all remain owned by the official launcher. This module
only creates a :class:`ViewerClient` and passes it into the launcher's optional
runtime hook.
"""

from __future__ import annotations

import argparse
import sys

from manimux.integrations.molmoact_yam.launch_yaml_eval_molmoact import (
    main as launch_molmoact,
)

from ..client import ViewerClient


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--viewer-endpoint", default="tcp://127.0.0.1:5568")
    parser.add_argument("--viewer-camera-hz", type=float, default=5.0)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    client = ViewerClient(
        robot="yam",
        policy="MolmoAct",
        endpoint=known.viewer_endpoint,
        camera_hz=known.viewer_camera_hz,
        control_mode="observe",
    )
    try:
        launch_molmoact(runtime=client)
    finally:
        client.close()


if __name__ == "__main__":
    main()
