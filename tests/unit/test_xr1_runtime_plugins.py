"""XR-1 is the first policy whose actions are Cartesian, so the adapter does real
geometry. These tests pin that geometry: a zero delta must be an exact no-op, a
known delta must reproduce exactly through FK, and the dimensions YAM does not
have (waist, mobile base) must never reach the arms.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from manimux.clock import SystemClock
from manimux.config import load_config
from manimux.integrations.xr1_yam.policy_plugin import (
    ACTION_DIM,
    XR1HttpPolicyModel,
    XR1YamAdapter,
    _axis_angle_to_rotation,
)
from manimux.policies import build_policy_adapter, build_policy_model
from manimux.robots import build_robot
from manimux.runtime.rtc import RtcInferenceRequest
from manimux.sensors import build_sensor
from manimux.sensors.camera_server import CameraServerSensorDriver
from manimux.types import (
    ActionContext,
    InferenceRequest,
    ObservationSnapshot,
    RobotState,
    SensorFrame,
)

pytest.importorskip("mujoco")
pytest.importorskip("mink")
pytest.importorskip("i2rt")

ANCHOR = np.array(
    [
        -0.6094,
        0.5835,
        0.8425,
        -1.0168,
        -0.1108,
        -0.4580,
        0.55,
        -0.5000,
        0.4000,
        0.7000,
        -0.9000,
        0.2000,
        -0.3000,
        0.35,
    ]
)


@pytest.fixture(scope="module")
def adapter() -> XR1YamAdapter:
    config = load_config("configs/xiaomi-xr1/yam/infra/native.yaml")
    return build_policy_adapter(config.robot, config.policy)


def test_xr1_run_config_swaps_only_the_policy_layer() -> None:
    from manimux.robots.yam import YamDualArmDriver

    xr1 = load_config(Path("configs/xiaomi-xr1/yam/infra/native.yaml"))
    molmoact = load_config(Path("configs/molmoact2/yam/infra/manimux.yaml"))

    assert isinstance(build_robot(xr1.robot, SystemClock()), YamDualArmDriver)
    assert isinstance(build_sensor(xr1.sensors[0], SystemClock()), CameraServerSensorDriver)
    assert isinstance(build_policy_model(xr1.policy), XR1HttpPolicyModel)

    assert xr1.robot.driver == molmoact.robot.driver
    assert xr1.sensors[0].driver == molmoact.sensors[0].driver
    assert xr1.viewer.robot_adapter == molmoact.viewer.robot_adapter


def test_zero_delta_holds_the_measured_pose(adapter: XR1YamAdapter) -> None:
    """Every step is a delta against the same anchor, so all-zero must not move."""
    chunk = adapter.decode_action(
        {"actions": np.zeros((30, ACTION_DIM)), "state": ANCHOR},
        ActionContext(request_seq=1, observation_time_ns=0, created_time_ns=0),
    )

    for group, offset in (("left_arm", 0), ("right_arm", 7)):
        joints = chunk.groups[group]
        assert joints.shape == (30, 7)
        assert np.abs(joints[:, :6] - ANCHOR[offset : offset + 6]).max() < 2e-3
        assert np.abs(joints[:, 6] - ANCHOR[offset + 6]).max() < 1e-12


def test_known_delta_round_trips_through_forward_kinematics(adapter: XR1YamAdapter) -> None:
    from manimux.kinematics import build_kinematics

    rng = np.random.default_rng(3)
    actions = np.zeros((30, ACTION_DIM))
    d_pos = rng.uniform(-0.05, 0.05, (30, 3))
    d_aa = rng.uniform(-0.15, 0.15, (30, 3))
    actions[:, 0:3] = d_pos
    actions[:, 3:6] = d_aa
    actions[:, 6] = np.linspace(0.0, 0.3, 30)
    # Waist and mobile-base columns: YAM has neither, they must be dropped.
    actions[:, 16] = 9.9
    actions[:, 17:20] = 9.9

    chunk = adapter.decode_action(
        {"actions": actions, "state": ANCHOR},
        ActionContext(request_seq=2, observation_time_ns=0, created_time_ns=0),
    )

    kinematics = build_kinematics("yam")
    anchor_pose = kinematics.fk(ANCHOR[:6], ANCHOR[6])
    rotation, position = anchor_pose[:3, :3], anchor_pose[:3, 3]

    for step in range(30):
        expected_p = position + rotation @ d_pos[step]
        expected_r = rotation @ _axis_angle_to_rotation(d_aa[step])
        achieved = kinematics.fk(chunk.groups["left_arm"][step, :6], ANCHOR[6])
        assert np.linalg.norm(achieved[:3, 3] - expected_p) < 2e-3
        residual = expected_r.T @ achieved[:3, :3]
        angle = abs(np.arccos(np.clip((np.trace(residual) - 1) / 2, -1.0, 1.0)))
        assert angle < np.radians(0.2)

    gripper = chunk.groups["left_arm"][:, 6]
    assert gripper[0] == pytest.approx(ANCHOR[6])
    assert gripper[-1] == pytest.approx(ANCHOR[6] + 0.3)
    assert chunk.action_space == "joint_position"
    assert chunk.plan_id.startswith("xr1-")


def test_adapter_rejects_a_bare_action_matrix(adapter: XR1YamAdapter) -> None:
    with pytest.raises(TypeError, match="state"):
        adapter.decode_action(
            np.zeros((30, ACTION_DIM)),
            ActionContext(request_seq=1, observation_time_ns=0, created_time_ns=0),
        )


def test_adapter_rejects_wrong_action_width(adapter: XR1YamAdapter) -> None:
    with pytest.raises(ValueError, match="shape"):
        adapter.decode_action(
            {"actions": np.zeros((30, 14)), "state": ANCHOR},
            ActionContext(request_seq=1, observation_time_ns=0, created_time_ns=0),
        )


class _StubResponse:
    def __init__(self, body: str) -> None:
        self.status_code = 200
        self.text = body


def test_http_model_sends_cameras_and_carries_the_anchor_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json_numpy

    config = load_config("configs/xiaomi-xr1/yam/infra/native.yaml")
    model = build_policy_model(config.policy)
    model._session_id = "session"

    captured: dict[str, object] = {}

    def _post(url: str, **kwargs: object) -> _StubResponse:
        captured["url"] = url
        captured["payload"] = json_numpy.loads(kwargs["data"])
        return _StubResponse(
            json_numpy.dumps({"actions": np.zeros((30, ACTION_DIM), dtype=np.float32)})
        )

    stub = types.ModuleType("requests")
    stub.post = _post  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "requests", stub)

    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    snapshot = ObservationSnapshot(
        state=RobotState(
            groups={"left_arm": ANCHOR[:7], "right_arm": ANCHOR[7:]},
            monotonic_ns=1,
            sequence=1,
        ),
        frames={
            name: SensorFrame(name=name, data=frame, capture_monotonic_ns=1, sequence=1)
            for name in ("left_camera", "front_camera", "right_camera")
        },
    )
    raw = model.infer(
        InferenceRequest(
            session_id="session",
            request_seq=1,
            observation_time_ns=0,
            deadline_ns=2**62,
            observation=snapshot,
            instruction="pick up the red ball",
        )
    )

    assert captured["url"] == "http://127.0.0.1:8400/act"
    payload = captured["payload"]
    assert set(payload) == {"top_cam", "left_cam", "right_cam", "timestamp", "instruction", "state"}
    np.testing.assert_array_equal(payload["state"], ANCHOR)

    # The deltas are anchored on the state that produced them, so the model
    # plugin must hand that state to the adapter alongside the matrix.
    assert isinstance(raw, dict)
    np.testing.assert_array_equal(raw["state"], ANCHOR)
    assert np.asarray(raw["actions"]).shape == (30, ACTION_DIM)


def test_http_model_encodes_rtc_joint_condition_in_native_action_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json_numpy

    config = load_config("configs/xiaomi-xr1/yam/infra/native-rtc.yaml")
    model = build_policy_model(config.policy)
    model._session_id = "session"
    captured: dict[str, object] = {}

    def _post(url: str, **kwargs: object) -> _StubResponse:
        del url
        captured["payload"] = json_numpy.loads(kwargs["data"])
        return _StubResponse(
            json_numpy.dumps({"actions": np.zeros((30, ACTION_DIM), dtype=np.float32)})
        )

    stub = types.ModuleType("requests")
    stub.post = _post  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "requests", stub)

    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    snapshot = ObservationSnapshot(
        state=RobotState(
            groups={"left_arm": ANCHOR[:7], "right_arm": ANCHOR[7:]},
            monotonic_ns=1,
            sequence=1,
        ),
        frames={
            name: SensorFrame(name=name, data=frame, capture_monotonic_ns=1, sequence=1)
            for name in ("left_camera", "front_camera", "right_camera")
        },
    )
    joint_condition = np.tile(ANCHOR, (30, 1))
    model.infer(
        RtcInferenceRequest(
            session_id="session",
            request_seq=1,
            observation_time_ns=0,
            deadline_ns=2**62,
            observation=snapshot,
            instruction="pick up the red ball",
            action_condition=joint_condition,
            condition_weights=np.ones(30),
            rtc_beta=4.25,
        )
    )

    payload = captured["payload"]
    assert np.asarray(payload["action_condition"]).shape == (30, 60)
    np.testing.assert_allclose(payload["action_condition"], 0.0, atol=1e-5)
    np.testing.assert_array_equal(payload["action_condition_weights"], np.ones(30))
    assert payload["rtc_beta"] == pytest.approx(4.25)
