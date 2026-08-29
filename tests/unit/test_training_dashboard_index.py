from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/launch_training_dashboard_index.py"
SPEC = importlib.util.spec_from_file_location("launch_training_dashboard_index", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_index_links_each_model_to_its_own_port():
    html = MODULE.render_index(16006).decode()

    assert "YAM 训练曲线" in html
    assert "Pi05" in html
    assert "LingBot Action" in html
    assert "Xiaomi Robotics 1" in html
    assert 'data-port="16007"' in html
    assert 'data-port="16011"' in html
