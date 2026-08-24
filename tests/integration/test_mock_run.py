from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from manimux.config import load_config
from manimux.policies.capabilities import PolicyCapabilities
from manimux.runtime.edge import EdgeRuntime
from manimux.types import (
    ActionChunk,
    InferenceResponse,
    RobotCommand,
    RobotState,
    SensorFrame,
)
from manimux.viewer import ViewerControl


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
    first_plan = root["plans/000000"]
    assert set(first_plan.group_keys()) == {"canonical_raw", "infra_output", "committed"}
    assert first_plan["canonical_raw/left_arm"].shape[1] == 6
    assert first_plan["infra_output/left_arm"].shape[1] == 6
    assert first_plan["committed/left_arm"].shape[1] == 6
    metadata = json.loads((result.episode_dir / "meta.json").read_text(encoding="utf-8"))
    assert metadata["policy_backend"] == {}
    assert metadata["launch_mode"] == "run"
    events = [
        json.loads(line)
        for line in (result.episode_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    boundaries = [event for event in events if event["kind"] == "plan_boundary"]
    assert boundaries
    assert boundaries[0]["blend_anchor_source"] == "measured_state"
    assert set(boundaries[0]["raw_first"]) == set(config.robot.group_dims)
    assert set(boundaries[0]["committed_first"]) == set(config.robot.group_dims)


def test_single_inflight_schedule_refills_after_each_response(tmp_path: Path) -> None:
    config = load_config("configs/mock.yaml")
    config.run.output_dir = tmp_path
    config.run.max_steps = 120
    config.policy.inference_delay_s = 0.01
    config.execution.inference_schedule = "single_inflight"
    run_dir = tmp_path / "run-single-inflight"
    run_dir.mkdir()

    result = EdgeRuntime(config, run_dir).run()

    assert result.steps == 120
    assert result.accepted_plans >= 2


class _HomeTestRobot:
    def __init__(self, group_dims: dict[str, int]) -> None:
        self.groups = {name: np.ones(dim) for name, dim in group_dims.items()}
        self.commands: list[dict[str, np.ndarray]] = []
        self.home_calls = 0
        self.sequence = 0

    def connect(self) -> None:
        pass

    def get_state(self) -> RobotState:
        self.sequence += 1
        return RobotState(
            groups={name: value.copy() for name, value in self.groups.items()},
            monotonic_ns=__import__("time").monotonic_ns(),
            sequence=self.sequence,
        )

    def send_command(self, command: RobotCommand) -> None:
        self.commands.append({name: value.copy() for name, value in command.groups.items()})

    def home(self) -> None:
        self.home_calls += 1
        self.groups = {name: np.zeros_like(value) for name, value in self.groups.items()}

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


class _HomeTestWorker:
    def __init__(self) -> None:
        self.request = None
        self.poll_count = 0
        self.started = False

    def start(self) -> None:
        self.started = True

    @property
    def is_alive(self) -> bool:
        return self.started

    def submit_latest(self, request) -> None:
        self.request = request

    def poll(self) -> InferenceResponse | None:
        self.poll_count += 1
        if self.poll_count != 2 or self.request is None:
            return None
        request = self.request
        groups = {
            name: np.full((4, len(values)), 1.5)
            for name, values in request.observation.state.groups.items()
        }
        chunk = ActionChunk(
            plan_id="pre-home-plan",
            request_seq=request.request_seq,
            observation_time_ns=request.observation_time_ns,
            created_time_ns=request.observation_time_ns,
            action_space="joint_position",
            dt_ns=50_000_000,
            groups=groups,
        )
        return InferenceResponse(
            session_id=request.session_id,
            request_seq=request.request_seq,
            finished_time_ns=request.observation_time_ns,
            inference_ms=1.0,
            raw_action=chunk,
            observation_time_ns=request.observation_time_ns,
        )

    def close(self) -> None:
        self.started = False


class _HomeTestViewer:
    def __init__(self) -> None:
        self.poll_count = 0

    def poll_control(self) -> ViewerControl:
        self.poll_count += 1
        if self.poll_count == 1:
            return ViewerControl(paused=True)
        if self.poll_count == 2:
            return ViewerControl(paused=True, home_requested=True)
        if self.poll_count == 4:
            return ViewerControl(paused=True, finish_requested=True)
        return ViewerControl(paused=True)

    def set_state_metadata(self, _metadata) -> None:
        pass

    def publish_plan(self, *_args, **_kwargs) -> None:
        pass

    def publish_state(self, *_args, **_kwargs) -> None:
        pass

    def publish_event(self, *_args, **_kwargs) -> None:
        pass

    def close(self) -> None:
        pass


class _AutoRunningViewer:
    def set_state_metadata(self, _metadata) -> None:
        pass

    def poll_control(self) -> ViewerControl:
        return ViewerControl(paused=False)

    def publish_plan(self, *_args, **_kwargs) -> None:
        pass

    def publish_state(self, *_args, **_kwargs) -> None:
        pass

    def publish_event(self, *_args, **_kwargs) -> None:
        pass

    def close(self) -> None:
        pass


class _FingerprintWorker(_HomeTestWorker):
    capabilities = PolicyCapabilities(
        backend_metadata={
            "server_revision": "xpolicy-sha",
            "model": {"model_root": "/checkpoints/pi05-step-1000"},
        }
    )


def test_policy_backend_fingerprint_is_written_after_worker_start(tmp_path: Path) -> None:
    config = load_config("configs/mock.yaml")
    config.run.max_steps = 1
    run_dir = tmp_path / "run-policy-fingerprint"
    run_dir.mkdir()
    runtime = EdgeRuntime(config, run_dir)
    runtime._robot = _HomeTestRobot(config.robot.group_dims)
    runtime._sensors = []
    runtime._worker = _FingerprintWorker()
    runtime._viewer = _AutoRunningViewer()

    result = runtime.run()

    metadata = json.loads((result.episode_dir / "meta.json").read_text(encoding="utf-8"))
    assert metadata["policy_backend"]["server_revision"] == "xpolicy-sha"
    assert metadata["policy_backend"]["model"]["model_root"].endswith("pi05-step-1000")


def test_max_steps_automatically_homes_and_exits(tmp_path: Path) -> None:
    config = load_config("configs/mock.yaml")
    config.viewer.enabled = True
    config.run.max_steps = 2
    config.robot.options["home_on_close"] = True
    run_dir = tmp_path / "run-max-steps-home"
    run_dir.mkdir()
    runtime = EdgeRuntime(config, run_dir)
    robot = _HomeTestRobot(config.robot.group_dims)
    runtime._robot = robot
    runtime._sensors = []
    runtime._worker = _HomeTestWorker()
    runtime._viewer = _AutoRunningViewer()

    result = runtime.run()

    assert result.steps == 2
    assert robot.home_calls == 1
    assert result.terminal_reason == "completed"


def test_home_discards_pre_home_response_and_commands_fresh_measured_state(tmp_path: Path) -> None:
    config = load_config("configs/mock.yaml")
    config.viewer.enabled = True
    run_dir = tmp_path / "run-home-reset"
    run_dir.mkdir()
    runtime = EdgeRuntime(config, run_dir)
    robot = _HomeTestRobot(config.robot.group_dims)
    runtime._robot = robot
    runtime._sensors = []
    runtime._worker = _HomeTestWorker()
    runtime._viewer = _HomeTestViewer()

    result = runtime.run()

    assert result.steps == 0
    assert result.accepted_plans == 0
    assert result.rejected_plans == 1
    assert robot.home_calls == 1
    assert robot.commands
    for command in robot.commands[1:]:
        for values in command.values():
            np.testing.assert_array_equal(values, np.zeros_like(values))


class _OrderedRobot(_HomeTestRobot):
    def __init__(self, group_dims: dict[str, int], events: list[str]) -> None:
        super().__init__(group_dims)
        self.events = events

    def connect(self) -> None:
        self.events.append("robot.connect")

    def stop(self) -> None:
        self.events.append("robot.stop")

    def close(self) -> None:
        self.events.append("robot.close")


class _OrderedWorker(_HomeTestWorker):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    def start(self) -> None:
        self.events.append("worker.start")
        super().start()

    def close(self) -> None:
        self.events.append("worker.close")
        super().close()


class _InterruptingSensor:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.read_count = 0

    def start(self) -> None:
        self.events.append("sensor.start")

    def read(self) -> SensorFrame:
        self.read_count += 1
        self.events.append(f"sensor.read.{self.read_count}")
        if self.read_count == 2:
            raise KeyboardInterrupt
        return SensorFrame(
            name="overhead",
            data=np.zeros((2, 2, 3), dtype=np.uint8),
            capture_monotonic_ns=1,
            sequence=self.read_count,
        )

    def close(self) -> None:
        self.events.append("sensor.close")


def test_interrupt_stops_robot_first_then_saves_partial_episode(tmp_path: Path) -> None:
    config = load_config("configs/mock.yaml")
    run_dir = tmp_path / "run-interrupt"
    run_dir.mkdir()
    runtime = EdgeRuntime(config, run_dir)
    events: list[str] = []
    runtime._robot = _OrderedRobot(config.robot.group_dims, events)
    runtime._worker = _OrderedWorker(events)
    runtime._sensors = [_InterruptingSensor(events)]

    with pytest.raises(KeyboardInterrupt):
        runtime.run()

    assert events[:4] == ["sensor.start", "sensor.read.1", "worker.start", "robot.connect"]
    assert events.index("robot.stop") < events.index("robot.close")
    assert events.index("robot.close") < events.index("worker.close")
    assert events.index("worker.close") < events.index("sensor.close")
    partials = list(run_dir.glob("*.partial"))
    assert len(partials) == 1
    result = json.loads((partials[0] / "result.json").read_text())
    assert result["terminal_reason"] == "KeyboardInterrupt"


def test_interrupt_homes_only_when_configured(tmp_path: Path) -> None:
    config = load_config("configs/mock.yaml")
    config.robot.options["home_on_close"] = True
    run_dir = tmp_path / "run-interrupt-home"
    run_dir.mkdir()
    runtime = EdgeRuntime(config, run_dir)
    events: list[str] = []
    robot = _OrderedRobot(config.robot.group_dims, events)
    runtime._robot = robot
    runtime._worker = _OrderedWorker(events)
    runtime._sensors = [_InterruptingSensor(events)]

    with pytest.raises(KeyboardInterrupt):
        runtime.run()

    assert robot.home_calls == 1
    assert events.index("robot.stop") < events.index("robot.close")
