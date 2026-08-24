from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from manimux.types import ActionChunk, ActionHorizon, RobotState, SensorFrame
from manimux.viewer.bridge import ViewerBridge
from manimux.viewer.client import ViewerClient
from manimux.viewer.dashboard import (
    PolicyViewer,
    _compose_camera_wall,
    _instruction_markdown,
    _prefill_task,
    _trajectory_colors,
)
from manimux.viewer.protocol import PolicyPlan, RobotSnapshot, RuntimeEvent
from manimux.viewer.robots import available_robot_adapters, load_robot_adapter
from manimux.viewer.robots.yam import DEFAULT_I2RT_ROOT, YamAdapter


@pytest.mark.parametrize("experiment_mode", [False, True])
def test_prepare_button_atomically_selects_rollout_mode(experiment_mode: bool) -> None:
    viewer = PolicyViewer.__new__(PolicyViewer)
    viewer.experiment_mode = not experiment_mode
    viewer.evaluation_saved = True
    viewer.service_ready = True
    viewer.new_rollout_requested = False
    viewer.preparing_rollout = False
    viewer.paused = False
    viewer.step_once = False
    viewer.home_requested = False
    viewer.finish_requested = False
    viewer.prepare_normal_btn = SimpleNamespace(disabled=False, visible=True)
    viewer.prepare_experiment_btn = SimpleNamespace(disabled=False, visible=True)
    viewer.layout_id = SimpleNamespace(disabled=False, value="layout-02")
    viewer.task = SimpleNamespace(disabled=False, value="fold the towel")
    viewer.status = SimpleNamespace(content="")
    viewer.rollout_setup_status = SimpleNamespace(content="")
    viewer.new_rollout_folder = SimpleNamespace(visible=True)
    viewer.policy_control_folder = SimpleNamespace(visible=False)
    viewer.evaluation_folder = SimpleNamespace(visible=False)
    viewer.overlay_folder = SimpleNamespace(visible=False)
    viewer.run_folder = SimpleNamespace(visible=True)

    viewer._prepare_rollout(experiment_mode=experiment_mode)
    request = viewer.control_state()

    assert request["new_rollout_requested"] is True
    assert request["experiment_mode"] is experiment_mode
    assert request["task_command"] == "fold the towel"
    assert request["layout_id"] == "layout-02"
    assert viewer.prepare_normal_btn.disabled
    assert viewer.prepare_experiment_btn.disabled
    assert not viewer.prepare_normal_btn.visible
    assert not viewer.prepare_experiment_btn.visible
    assert viewer.task.disabled
    assert viewer.layout_id.disabled
    assert viewer.control_state()["new_rollout_requested"] is False


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("waiting", (True, False, False, False, False)),
        ("setup", (True, False, False, False, True)),
        ("preparing", (True, False, False, False, True)),
        ("control", (False, True, False, True, True)),
        ("evaluation", (False, False, True, False, True)),
        ("complete", (False, False, False, False, True)),
    ],
)
def test_viewer_stage_exposes_only_the_current_action_area(
    stage: str, expected: tuple[bool, bool, bool, bool, bool]
) -> None:
    viewer = PolicyViewer.__new__(PolicyViewer)
    handles = [SimpleNamespace(visible=False) for _ in range(5)]
    (
        viewer.new_rollout_folder,
        viewer.policy_control_folder,
        viewer.evaluation_folder,
        viewer.overlay_folder,
        viewer.run_folder,
    ) = handles

    viewer._set_stage(stage)  # type: ignore[arg-type]

    assert tuple(handle.visible for handle in handles) == expected


def test_camera_wall_keeps_top_left_and_right_in_fixed_tiles() -> None:
    frames = {
        "top": np.full((360, 960, 3), (255, 0, 0), dtype=np.uint8),
        "left": np.full((360, 480, 3), (0, 255, 0), dtype=np.uint8),
        "right": np.full((360, 480, 3), (0, 0, 255), dtype=np.uint8),
    }

    wall = _compose_camera_wall(frames)

    assert wall.shape == (720, 960, 3)
    assert tuple(wall[200, 600]) == (255, 0, 0)
    assert tuple(wall[550, 240]) == (0, 255, 0)
    assert tuple(wall[550, 720]) == (0, 0, 255)


def test_protocol_is_not_tied_to_yam_dimensions() -> None:
    actions = np.zeros((25, 6))
    plan = PolicyPlan("example", "task", actions, 1 / 30, 500, 2, robot="custom")
    wire = plan.to_wire()
    assert wire["actions"] == actions.tolist()
    assert wire["robot"] == "custom"
    assert wire["protocol_version"] == 1


def test_protocol_rejects_malformed_actions() -> None:
    with pytest.raises(ValueError, match="shape"):
        PolicyPlan("example", "task", np.zeros(6), 1 / 30, 1, 0)
    with pytest.raises(ValueError, match="finite"):
        PolicyPlan("example", "task", np.array([[np.nan]]), 1 / 30, 1, 0)


def test_snapshot_encodes_generic_robot_state() -> None:
    state = RobotSnapshot(
        np.zeros(4),
        {},
        step=3,
        max_steps=10,
        chunk_index=2,
        robot="custom",
        active_chunk_id=7,
    )
    wire = state.to_wire()
    assert wire["joint_positions"] == [0.0] * 4
    assert wire["robot"] == "custom"
    assert wire["active_chunk_id"] == 7
    assert wire["chunk_index"] == 2


def test_plan_can_start_inside_an_activated_async_chunk() -> None:
    plan = PolicyPlan(
        "example",
        "task",
        np.zeros((10, 6)),
        1 / 30,
        250,
        4,
        start_index=3,
    )
    assert plan.to_wire()["start_index"] == 3
    with pytest.raises(ValueError, match="start_index"):
        PolicyPlan(
            "example",
            "task",
            np.zeros((10, 6)),
            1 / 30,
            250,
            4,
            start_index=11,
        )


def test_runtime_event_is_policy_independent() -> None:
    event = RuntimeEvent(
        "inference_submitted",
        robot="custom",
        policy="example",
        step=12,
        chunk_id=3,
        metadata={"planned_switch_step": 19},
    ).to_wire()
    assert event["kind"] == "event"
    assert event["event"] == "inference_submitted"
    assert event["metadata"]["planned_switch_step"] == 19


class _RecordingPublisher:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.closed = False

    def publish(self, message: Any) -> None:
        self.messages.append(message.to_wire())

    def close(self) -> None:
        self.closed = True


def test_viewer_client_observes_without_owning_policy_execution() -> None:
    publisher = _RecordingPublisher()
    client = ViewerClient(
        robot="custom",
        policy="example",
        camera_hz=0,
        publisher=publisher,  # type: ignore[arg-type]
    )
    client.episode_started(instruction="task", max_steps=100)
    client.inference_submitted(step=5, chunk_id=2, planned_switch_step=12)
    client.plan_activated(
        actions=np.zeros((10, 6)),
        action_index=3,
        chunk_id=2,
        step=12,
        action_dt=1 / 30,
        inference_ms=400,
        instruction="task",
    )
    client.step_executed(
        joint_positions=np.zeros(6),
        cameras={},
        step=13,
        max_steps=100,
        action_index=4,
        chunk_id=2,
    )
    client.close()

    assert [message["kind"] for message in publisher.messages] == [
        "event",
        "event",
        "plan",
        "state",
    ]
    assert publisher.messages[2]["start_index"] == 3
    assert publisher.messages[3]["active_chunk_id"] == 2
    assert publisher.closed


def test_runtime_bridge_publishes_the_exact_committed_plan() -> None:
    publisher = _RecordingPublisher()
    bridge = ViewerBridge(
        enabled=False,
        robot_adapter="custom",
        group_order=["left", "right"],
        policy="molmoact_http",
        instruction="task",
    )
    bridge._enabled = True
    bridge._publisher = publisher
    bridge._policy_plan_type = PolicyPlan
    raw = ActionChunk(
        plan_id="plan-1",
        request_seq=1,
        observation_time_ns=0,
        created_time_ns=1,
        action_space="joint_position",
        dt_ns=10,
        groups={"left": np.full((4, 1), 9.0), "right": np.full((4, 1), -9.0)},
    )
    committed = ActionHorizon(
        start_time_ns=20,
        dt_ns=10,
        plan_id="plan-1",
        groups={"left": np.array([[1.0], [2.0]]), "right": np.array([[-1.0], [-2.0]])},
    )

    bridge.publish_plan(raw, 250.0, committed=committed)

    assert len(publisher.messages) == 1
    message = publisher.messages[0]
    assert message["policy"] == "molmoact_http"
    assert message["instruction"] == "task"
    assert message["actions"] == [[1.0, -1.0], [2.0, -2.0]]
    assert message["metadata"]["committed_start_time_ns"] == 20


def test_runtime_bridge_publishes_managed_lifecycle_event() -> None:
    publisher = _RecordingPublisher()
    bridge = ViewerBridge(
        enabled=False,
        robot_adapter="custom",
        group_order=["arm"],
        policy="policy",
    )
    bridge._enabled = True
    bridge._publisher = publisher
    bridge._runtime_event_type = RuntimeEvent

    bridge.publish_event(
        "episode_started",
        metadata={"control_mode": "managed", "instruction": "task"},
    )

    assert publisher.messages[0]["metadata"]["control_mode"] == "managed"


def test_runtime_bridge_throttles_camera_encoding_off_the_control_rate() -> None:
    publisher = _RecordingPublisher()
    bridge = ViewerBridge(
        enabled=False,
        robot_adapter="custom",
        group_order=["arm"],
        camera_hz=1.0,
    )
    bridge._enabled = True
    bridge._publisher = publisher
    bridge._snapshot_type = RobotSnapshot
    state = RobotState(groups={"arm": np.zeros(2)}, monotonic_ns=1, sequence=1)
    frame = SensorFrame(
        name="camera",
        data=np.zeros((4, 4, 3), dtype=np.uint8),
        capture_monotonic_ns=1,
        sequence=1,
    )

    bridge.publish_state(state, {"camera": frame}, step=0, max_steps=2)
    bridge.publish_state(state, {"camera": frame}, step=1, max_steps=2)

    assert set(publisher.messages[0]["cameras_jpeg"]) == {"camera"}
    assert publisher.messages[1]["cameras_jpeg"] == {}


def test_yam_is_a_discovered_adapter() -> None:
    assert "yam" in available_robot_adapters()
    assert isinstance(load_robot_adapter("yam"), YamAdapter)


def test_yam_adapter_splits_single_and_dual_arm_vectors() -> None:
    adapter = YamAdapter()
    assert set(adapter.split_actions(np.zeros((2, 7)), "joint_position")) == {"left"}
    assert set(adapter.split_actions(np.zeros((2, 14)), "joint_position")) == {
        "left",
        "right",
    }
    with pytest.raises(ValueError, match="action_space"):
        adapter.split_actions(np.zeros((2, 7)), "cartesian_delta")
    with pytest.raises(ValueError, match="7 or 14"):
        adapter.split_joint_positions(np.zeros(8))


def test_yam_fk_and_assets_are_self_contained() -> None:
    assert (DEFAULT_I2RT_ROOT / "i2rt/robot_models/arm/yam/yam.urdf").is_file()
    assert (DEFAULT_I2RT_ROOT / "i2rt/robot_models/gripper/linear_4310/linear_4310.xml").is_file()
    adapter = YamAdapter()
    joints = np.array([0.0, 0.8, 1.2, -0.3, 0.0, 0.0, 0.5])
    poses = adapter.positions("left", np.stack([joints, joints]))
    assert poses.shape == (2, 3)
    assert np.isfinite(poses).all()


def test_trajectory_gradient_uses_deep_to_light_purple() -> None:
    colors = _trajectory_colors(9)
    assert colors.shape == (9, 3)
    assert colors.dtype == np.uint8
    assert colors[0].tolist() == [67, 20, 133]
    assert colors[-1].tolist() == [210, 150, 255]
    assert len(np.unique(colors, axis=0)) == 9


def test_the_task_prompt_is_rendered_for_the_always_visible_header() -> None:
    assert "pick up the red ball" in _instruction_markdown("  pick up the red ball  ")
    assert "waiting" in _instruction_markdown("   ")


def test_republished_instructions_do_not_erase_an_operator_mid_edit() -> None:
    """The runtime resends its instruction on every chunk; typing must survive."""

    assert _prefill_task("", "  fold the towel  ") == "fold the towel"
    assert _prefill_task("my own comm", "fold the towel") == "my own comm"
