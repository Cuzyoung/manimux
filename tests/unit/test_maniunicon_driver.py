from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manimux.robots.maniunicon import ManiUniConMeshcatDualArmDriver
from manimux.types import RobotCommand


class _Clock:
    def now_ns(self) -> int:
        return 123

    def sleep_until_ns(self, target_ns: int) -> None:
        del target_ns


@dataclass
class _State:
    joint_positions: np.ndarray
    gripper_state: np.ndarray


class _Arm:
    def __init__(self, offset: float) -> None:
        self.state = _State(np.full(6, offset), np.asarray([offset]))
        self.actions: list[object] = []
        self.connected = False

    def connect(self) -> bool:
        self.connected = True
        return True

    def disconnect(self) -> bool:
        self.connected = False
        return True

    def get_state(self) -> _State:
        return self.state

    def send_action(self, action: object) -> bool:
        self.actions.append(action)
        return True

    def reset_to_init(self) -> bool:
        return True

    def stop(self) -> bool:
        return True


def _groups() -> dict[str, int]:
    return {"left_arm": 6, "right_arm": 6, "left_gripper": 1, "right_gripper": 1}


def test_maniunicon_driver_combines_state_and_splits_command() -> None:
    left = _Arm(1.0)
    right = _Arm(2.0)
    driver = ManiUniConMeshcatDualArmDriver(
        left=left,
        right=right,
        action_factory=lambda **fields: fields,
        group_dims=_groups(),
        clock=_Clock(),
    )

    driver.connect()
    state = driver.get_state()
    np.testing.assert_array_equal(state.groups["left_arm"], np.ones(6))
    np.testing.assert_array_equal(state.groups["right_gripper"], [2.0])
    assert state.monotonic_ns == 123

    command_groups = {
        "left_arm": np.full(6, 0.1),
        "right_arm": np.full(6, -0.2),
        "left_gripper": np.asarray([0.3]),
        "right_gripper": np.asarray([0.4]),
    }
    driver.send_command(RobotCommand(command_groups, monotonic_ns=123, plan_id="p1"))

    assert len(left.actions) == len(right.actions) == 1
    left_action = left.actions[0]
    right_action = right.actions[0]
    assert isinstance(left_action, dict) and isinstance(right_action, dict)
    np.testing.assert_array_equal(left_action["joint_positions"], command_groups["left_arm"])
    np.testing.assert_array_equal(right_action["gripper_state"], command_groups["right_gripper"])

    driver.stop()
    driver.close()
    assert not left.connected and not right.connected


def test_maniunicon_driver_rejects_noncanonical_groups() -> None:
    left = _Arm(0.0)
    right = _Arm(0.0)
    groups = {"left_arm": 6, "right_arm": 6}
    try:
        ManiUniConMeshcatDualArmDriver(
            left=left,
            right=right,
            action_factory=lambda **fields: fields,
            group_dims=groups,
            clock=_Clock(),
        )
    except ValueError as exc:
        assert "groups must be" in str(exc)
    else:
        raise AssertionError("expected noncanonical groups to be rejected")
