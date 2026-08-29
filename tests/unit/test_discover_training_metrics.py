from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/discover_training_metrics.py"
SPEC = importlib.util.spec_from_file_location("discover_training_metrics", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_discovers_supported_metric_logs_and_keeps_run_names(tmp_path: Path):
    data_root = tmp_path / "data"
    code_root = tmp_path / "code"
    pi_root = data_root / "runs/pi05"
    gr00t_root = data_root / "runs/gr00t-n17"
    pi_root.mkdir(parents=True)
    gr00t_root.mkdir(parents=True)

    pi_log = pi_root / "task-pi05-b32-v1-train.log"
    pi_log.write_text("Step 7: grad_norm=1.2, loss=0.3, param_norm=4.5\n")
    gr00t_log = gr00t_root / "task-gr00t-v1.log"
    gr00t_log.write_text("12/3000 {'loss': 0.25, 'grad_norm': 0.7}\n")
    (gr00t_root / "ignored.launch.log").write_text("{'loss': 9.9}\n")
    (pi_root / "not-ready.log").write_text("environment setup only\n")

    runs = MODULE.discover_runs(data_root, code_root)

    assert [(run.model, run.log_path.name) for run in runs] == [
        ("pi05", pi_log.name),
        ("gr00t-n17", gr00t_log.name),
    ]
    assert runs[0].output_path == (
        data_root / "runs/live-tensorboard/pi05/task-pi05-b32-v1-train"
    )
    assert runs[1].output_path == (
        data_root / "runs/live-tensorboard/gr00t-n17/task-gr00t-v1"
    )


def test_new_log_appears_on_next_discovery(tmp_path: Path):
    data_root = tmp_path / "data"
    code_root = tmp_path / "code"
    pi_root = data_root / "runs/pi05"
    pi_root.mkdir(parents=True)

    assert MODULE.discover_runs(data_root, code_root) == []

    new_log = pi_root / "new-run.log"
    new_log.write_text("Step 1: grad_norm=2e-1, loss=4e-2, param_norm=3e+0\n")

    runs = MODULE.discover_runs(data_root, code_root)
    assert len(runs) == 1
    assert runs[0].log_path == new_log
