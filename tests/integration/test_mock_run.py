from __future__ import annotations

from pathlib import Path

import zarr

from manimux.config import load_config
from manimux.runtime.edge import EdgeRuntime


def test_mock_run_records_async_episode(tmp_path: Path) -> None:
    config = load_config("configs/mock.yaml")
    config.run.output_dir = tmp_path
    config.run.max_steps = 80
    config.policy.inference_delay_s = 0.02
    run_dir = tmp_path / "run-test"
    run_dir.mkdir()

    result = EdgeRuntime(config, run_dir).run()

    assert result.steps == 80
    assert result.accepted_plans >= 1
    assert result.episode_dir.is_dir()
    assert not result.episode_dir.name.endswith(".partial")
    root = zarr.open_group(str(result.episode_dir / "data.zarr"), mode="r")
    assert root["ticks/monotonic_ns"].shape == (80,)
    assert root["ticks/state/left_arm"].shape == (80, 6)
    assert len(list(root["plans"].group_keys())) >= 1
