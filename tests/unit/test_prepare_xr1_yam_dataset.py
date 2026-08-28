from __future__ import annotations

import importlib.util
import json
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
    assert "build_kinematics" not in SCRIPT.read_text()
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
        action[:, :3, :3] = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        arrays[f"{arm}-ee_transform"] = state
        arrays[f"action-{arm}-ee_transform"] = action

    payload, states, action_by_step = MODULE._episode_payload(
        tmp_path, arrays, {"num_frames": length}, "Assemble the screwdriver."
    )

    assert states.shape == (length, MODULE.STATE_DIM)
    np.testing.assert_allclose(action_by_step[0][0][0:3], [0.03, 0.0, 0.0])
    np.testing.assert_allclose(action_by_step[0][0][3:6], [0.0, 0.0, np.pi / 2])
    np.testing.assert_allclose(action_by_step[0][0][8:11], [-0.03, 0.0, 0.0])
    np.testing.assert_allclose(action_by_step[0][0][11:14], [0.0, 0.0, np.pi / 2])
    assert payload["proprios"]["left_ee_pos"][0] == [0.1, 0.0, 0.2]
    assert payload["actions"]["right_ee_pos"][0] == [-0.13, 0.0, 0.2]


def test_load_episode_requires_consistent_recorded_eepose(tmp_path: Path) -> None:
    length = 2
    for arm in ("left", "right"):
        np.save(tmp_path / f"{arm}-joint_pos.npy", np.zeros((length, 6)))
        np.save(tmp_path / f"{arm}-gripper_pos.npy", np.zeros((length, 1)))
        np.save(tmp_path / f"action-{arm}-joint.npy", np.zeros((length, 6)))
        np.save(tmp_path / f"action-{arm}-gripper.npy", np.zeros((length, 1)))
        for prefix in ("", "action-"):
            transform = _transform(length, (0.1, 0.2, 0.3))
            np.save(tmp_path / f"{prefix}{arm}-ee_pos.npy", transform[:, :3, 3])
            np.save(tmp_path / f"{prefix}{arm}-ee_rotm.npy", transform[:, :3, :3].reshape(length, 9))
            np.save(tmp_path / f"{prefix}{arm}-ee_transform.npy", transform)
    (tmp_path / "metadata.json").write_text(
        json.dumps({"num_frames": length, "extra": {"eepose": {"enabled": True}}})
    )

    arrays, metadata = MODULE._load_episode(tmp_path)

    assert arrays["left-ee_transform"].shape == (length, 4, 4)
    assert metadata["extra"]["eepose"]["enabled"] is True
