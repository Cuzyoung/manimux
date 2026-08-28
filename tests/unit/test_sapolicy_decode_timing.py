from __future__ import annotations

import numpy as np
import pytest

import manimux.kinematics
from manimux.config import load_config
from manimux.integrations.sapolicy_yam.policy_plugin import SAPolicyYamAdapter
from manimux.types import (
    ActionContext,
    InferenceRequest,
    ObservationSnapshot,
    RobotState,
    SensorFrame,
)

CONFIG = "configs/sapolicy/yam/infra/real-hardware.yaml"


class _FakeKinematics:
    num_arm_joints = 6

    def __init__(self) -> None:
        self.fail_x: list[float] = []
        self.low_margin_x: list[float] = []
        self.ik_calls: list[tuple[np.ndarray, np.ndarray]] = []

    def fk(self, joints, gripper):
        del gripper
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = float(np.asarray(joints)[0])
        pose[1, 3] = float(np.asarray(joints)[1])
        return pose

    def ik(self, target, seed, gripper):
        del gripper
        target = np.asarray(target, dtype=np.float64)
        seed = np.asarray(seed, dtype=np.float64)
        self.ik_calls.append((target.copy(), seed.copy()))
        solved = seed.copy()
        solved[0] = target[0, 3]
        solved[1] = target[1, 3]
        converged = not any(
            np.isclose(target[0, 3], value, atol=1e-8) for value in self.fail_x
        )
        return converged, solved

    def pose_error(self, target, joints, gripper):
        actual = self.fk(joints, gripper)
        return float(np.linalg.norm(actual[:3, 3] - target[:3, 3])), 0.0

    def joint_limit_margins(self, joints):
        values = np.asarray(joints, dtype=np.float64)
        margins = np.ones(6, dtype=np.float64)
        if any(np.isclose(values[0], value, atol=1e-8) for value in self.low_margin_x):
            margins[3] = 0.01
        return margins


def _build_adapter(monkeypatch, *, margin: float = 0.0):
    config = load_config(CONFIG)
    config.policy.options["min_valid_horizon_steps"] = 4
    config.policy.options["joint_limit_margin_rad"] = margin
    fake = _FakeKinematics()
    monkeypatch.setattr(
        manimux.kinematics,
        "build_kinematics",
        lambda name, **options: fake,
    )
    adapter = SAPolicyYamAdapter(config.robot, config.policy)
    return adapter, fake


def _snapshot(now_ns: int, *, left_seed: float = 0.7, right_seed: float = -0.7):
    return ObservationSnapshot(
        state=RobotState(
            groups={
                "left_arm": np.array([left_seed, 0.2, 0.0, 0.0, 0.0, 0.0, 0.5]),
                "right_arm": np.array([right_seed, -0.2, 0.0, 0.0, 0.0, 0.0, 0.5]),
            },
            monotonic_ns=now_ns,
            sequence=1,
        ),
        frames={
            "front_camera": SensorFrame(
                name="front_camera",
                data=np.zeros((8, 8, 3), dtype=np.uint8),
                capture_monotonic_ns=now_ns,
                sequence=1,
            )
        },
    )


def _wire_actions(adapter: SAPolicyYamAdapter) -> np.ndarray:
    packed = np.zeros((16, 14), dtype=np.float64)
    for step in range(16):
        packed[step, 0] = step * 0.01
        packed[step, 1] = 0.2
        packed[step, 6] = 0.5
        packed[step, 7] = -0.2 - step * 0.01
        packed[step, 8] = -0.2
        packed[step, 13] = 0.5
    return adapter._joint_prefix_to_wire(packed)


def _anchor(adapter: SAPolicyYamAdapter, seq: int, snapshot: ObservationSnapshot) -> None:
    adapter.prepare_request(
        InferenceRequest(
            session_id="session",
            request_seq=seq,
            observation_time_ns=snapshot.state.monotonic_ns,
            deadline_ns=snapshot.state.monotonic_ns + 1_000_000_000,
            observation=snapshot,
        )
    )


def test_sapolicy_discards_expired_source_steps_before_ik(monkeypatch) -> None:
    adapter, fake = _build_adapter(monkeypatch)
    observation_ns = 1_000_000_000
    snapshot = _snapshot(observation_ns)
    _anchor(adapter, 1, snapshot)
    actions = _wire_actions(adapter)
    fake.ik_calls.clear()
    # These targets would fail if the adapter still solved the stale prefix.
    fake.fail_x = [0.0, 0.01, 0.02, -0.2, -0.21, -0.22]

    chunk = adapter.decode_action(
        actions,
        ActionContext(
            request_seq=1,
            observation_time_ns=observation_ns,
            created_time_ns=observation_ns + 100,
            execution_time_ns=observation_ns + 3 * adapter._action_dt_ns,
            measured_state=_snapshot(
                observation_ns + 3 * adapter._action_dt_ns,
                left_seed=0.8,
                right_seed=-0.8,
            ).state,
        ),
    )

    assert chunk.horizon_steps == 13
    assert chunk.observation_time_ns == observation_ns
    assert chunk.source_offset_steps == 3
    assert chunk.metadata["adapter_source_offset_steps"] == 3
    assert chunk.metadata["adapter_ik_valid_steps"] == 13
    assert fake.ik_calls[0][0][0, 3] == pytest.approx(0.03)
    assert fake.ik_calls[0][1][0] == pytest.approx(0.8)
    assert fake.ik_calls[1][0][0, 3] == pytest.approx(-0.23)
    assert fake.ik_calls[1][1][0] == pytest.approx(-0.8)


def test_sapolicy_keeps_only_dual_arm_common_valid_prefix(monkeypatch) -> None:
    adapter, fake = _build_adapter(monkeypatch)
    observation_ns = 2_000_000_000
    snapshot = _snapshot(observation_ns)
    _anchor(adapter, 2, snapshot)
    actions = _wire_actions(adapter)
    fake.ik_calls.clear()
    fake.fail_x = [-0.26]

    chunk = adapter.decode_action(
        actions,
        ActionContext(
            request_seq=2,
            observation_time_ns=observation_ns,
            created_time_ns=observation_ns + 100,
            measured_state=snapshot.state,
        ),
    )

    assert chunk.horizon_steps == 6
    assert chunk.groups["left_arm"].shape == (6, 7)
    assert chunk.groups["right_arm"].shape == (6, 7)
    assert chunk.metadata["adapter_ik_truncated"] is True
    assert "right_arm" in chunk.metadata["adapter_ik_truncation_reason"]
    assert "source step 6" in chunk.metadata["adapter_ik_truncation_reason"]


def test_sapolicy_rejects_prefix_shorter_than_configured_minimum(monkeypatch) -> None:
    adapter, fake = _build_adapter(monkeypatch)
    observation_ns = 3_000_000_000
    snapshot = _snapshot(observation_ns)
    _anchor(adapter, 3, snapshot)
    actions = _wire_actions(adapter)
    fake.ik_calls.clear()
    fake.fail_x = [0.02]

    with pytest.raises(ValueError, match="common_valid_prefix_steps=2") as exc_info:
        adapter.decode_action(
            actions,
            ActionContext(
                request_seq=3,
                observation_time_ns=observation_ns,
                created_time_ns=observation_ns + 100,
                measured_state=snapshot.state,
            ),
        )

    message = str(exc_info.value)
    assert "source step 2" in message
    assert "position_error_m=" in message
    assert "joint_limit_margin_rad=" in message


def test_sapolicy_truncates_before_joint_limit_margin(monkeypatch) -> None:
    adapter, fake = _build_adapter(monkeypatch, margin=0.05)
    observation_ns = 4_000_000_000
    snapshot = _snapshot(observation_ns)
    _anchor(adapter, 4, snapshot)
    actions = _wire_actions(adapter)
    fake.ik_calls.clear()
    fake.low_margin_x = [0.08]

    chunk = adapter.decode_action(
        actions,
        ActionContext(
            request_seq=4,
            observation_time_ns=observation_ns,
            created_time_ns=observation_ns + 100,
            measured_state=snapshot.state,
        ),
    )

    assert chunk.horizon_steps == 8
    reason = chunk.metadata["adapter_ik_truncation_reason"]
    assert "joint J4 margin 0.010000 rad" in reason
    assert "below 0.050000 rad" in reason
