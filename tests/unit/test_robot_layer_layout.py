from __future__ import annotations

from pathlib import Path

import manimux.robots as robots_pkg


def test_robot_layer_does_not_import_any_policy_integration() -> None:
    """Every policy drives the same body, so the body must not depend on one."""
    root = Path(robots_pkg.__file__).resolve().parent
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        assert "manimux.integrations" not in source, path
        assert "molmoact" not in source.lower(), path


def test_every_run_config_shares_one_yam_body() -> None:
    """The YAM config is embodiment state; it must not live under a policy."""
    import yaml

    repo = Path(robots_pkg.__file__).resolve().parents[2].parent
    seen = set()
    for config_path in sorted((repo / "configs").glob("*-yam*.yaml")):
        config = yaml.safe_load(config_path.read_text())
        seen.add((config["robot"]["config"], config["robot"]["options"]["right_config"]))
    assert seen == {("configs/robots/yam_left.yaml", "configs/robots/yam_right.yaml")}

    for side in ("left", "right"):
        body = yaml.safe_load((repo / f"configs/robots/yam_{side}.yaml").read_text())
        assert body["robot"]["_target_"] == "manimux.robots.yam.arm.YAMRobot"
        assert len(body["agent"]["start_joints"]) == 7
