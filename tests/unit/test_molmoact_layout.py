from __future__ import annotations

import ast
from pathlib import Path

import manimux.integrations.molmoact_yam as molmoact_yam

INTEGRATION_ROOT = Path(molmoact_yam.__file__).resolve().parent


def test_molmoact_integration_has_no_old_checkout_file_reference() -> None:
    for path in INTEGRATION_ROOT.rglob("*"):
        if path.suffix not in {".py", ".yaml"}:
            continue
        content = path.read_text(encoding="utf-8")
        assert "/home/ubuntu" not in content
        assert "spec_from_file_location" not in content
        assert "universal_policy_viewer" not in content


def test_keyboard_interrupt_path_breaks_after_home_instead_of_reusing_state() -> None:
    launcher = INTEGRATION_ROOT / "launch_yaml_eval_molmoact.py"
    tree = ast.parse(launcher.read_text(encoding="utf-8"))
    handlers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler)
        and isinstance(node.type, ast.Name)
        and node.type.id == "KeyboardInterrupt"
    ]
    assert handlers
    handler_nodes = list(ast.walk(handlers[0]))
    assert any(isinstance(node, ast.Break) for node in handler_nodes)
    assert not any(isinstance(node, ast.Continue) for node in handler_nodes)
