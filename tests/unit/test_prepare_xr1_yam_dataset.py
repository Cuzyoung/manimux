from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[2] / "scripts/prepare_xr1_yam_dataset.py"
SPEC = importlib.util.spec_from_file_location("prepare_xr1_yam_dataset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _transform(length: int, offset: tuple[float, float, float]) -> np.ndarray:
    result = np.repeat(np.eye(4, dtype=np.float64)[None], length, axis=0)
    result[:, :3, 3] = np.asarray(offset)
    return result


def test_episode_payload_uses_recorded_ee_transforms_without_kinematics(tmp_path: Path) -> None:
    length = 2
    arrays: dict[str, np.ndarray] = {}
    for arm, sign in (("left", 1.0), ("right", -1.0)):
        arrays[f"{arm}-joint_pos"] = np.zeros((length, 6), dtype=np.float64)
        arrays[f"{arm}-gripper_pos"] = np.zeros((length, 1), dtype=np.float64)
        arrays[f"action-{arm}-joint"] = np.zeros((length, 6), dtype=np.float64)
        arrays[f"action-{arm}-gripper"] = np.full((length, 1), 0.25, dtype=np.float64)
        state = _transform(length, (0.1 * sign, 0.0, 0.2))
        action = state.copy()
        action[:, 0, 3] += 0.03 * sign
        arrays[f"{arm}-ee_transform"] = state
        arrays[f"action-{arm}-ee_transform"] = action

    payload, states, action_by_step = MODULE._episode_payload(
        tmp_path,
        arrays,
        {"num_frames": length},
        "Assemble the screwdriver.",
    )

    assert states.shape == (length, MODULE.STATE_DIM)
    np.testing.assert_allclose(action_by_step[0][0][0:3], [0.03, 0.0, 0.0])
    np.testing.assert_allclose(action_by_step[0][0][8:11], [-0.03, 0.0, 0.0])
    assert payload["proprios"]["left_ee_pos"][0] == [0.1, 0.0, 0.2]
    assert payload["actions"]["right_ee_pos"][0] == [-0.13, 0.0, 0.2]
    assert "Assemble the screwdriver." in payload["instruction"]["general"][0][
        "conversations"
    ][0]["value"]
