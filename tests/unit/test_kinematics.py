"""YAM FK/IK: accurate enough to drive the arm, and identical to the recorder.

Policies that emit end-effector poses (XR-1 and friends) go through this, so a
silent convention drift here would show up as a wrong-looking arm rather than an
error. The round-trip test pins accuracy; the equivalence test pins the
convention against the FK that produced the recorded ``ee_pos`` / ``ee_rotm``.
"""

from __future__ import annotations

import numpy as np
import pytest

kinematics = pytest.importorskip("manimux.kinematics")

# First frame of a real red-ball episode (configs/molmoact_yam_left.yaml).
START_JOINTS = np.array([-0.6094, 0.5835, 0.8425, -1.0168, -0.1108, -0.4580])


@pytest.fixture(scope="module")
def yam():
    pytest.importorskip("mujoco")
    pytest.importorskip("mink")
    pytest.importorskip("i2rt")
    return kinematics.build_kinematics("yam")


def test_fk_ik_round_trip_is_sub_millimetre(yam) -> None:
    rng = np.random.default_rng(0)
    for _ in range(8):
        joints = START_JOINTS + rng.uniform(-0.25, 0.25, size=6)
        gripper = 0.5
        target = yam.fk(joints, gripper)

        # Seed from a perturbed pose, the way a chunk step seeds off the last one.
        seed = joints + rng.uniform(-0.08, 0.08, size=6)
        converged, solved = yam.ik(target, seed, gripper)
        assert converged

        achieved = yam.fk(solved, gripper)
        assert np.linalg.norm(achieved[:3, 3] - target[:3, 3]) < 1e-3  # < 1 mm
        rotation = target[:3, :3].T @ achieved[:3, :3]
        angle = abs(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0)))
        assert angle < np.radians(0.05)


def test_fk_matches_the_recorded_episode_convention(yam) -> None:
    """Same convention as yam_abc_reproduce's ForwardKinematics.

    That implementation evaluates the site through mink; this one through
    ``mj_forward`` on the same spliced model, so the two agree to float noise
    (~1e-15) rather than bit-for-bit. ``qpos`` is still exactly equal, which is
    where a convention drift would actually show up.
    """
    import ast
    from pathlib import Path

    source = Path("/home/ubuntu/yam-abc-reproduce/yam_abc_reproduce/data/eepose.py")
    if not source.exists():
        pytest.skip("yam-abc-reproduce checkout is not available")

    import mujoco
    from i2rt.robots.kinematics import Kinematics
    from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

    tree = ast.parse(source.read_text())
    kept = [
        node
        for node in tree.body
        if (isinstance(node, ast.ClassDef) and node.name == "ForwardKinematics")
        or (isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "EE_SITE")
    ]
    namespace: dict[str, object] = {
        "mujoco": mujoco,
        "np": np,
        "Path": Path,
        "Kinematics": Kinematics,
        "ArmType": ArmType,
        "GripperType": GripperType,
        "combine_arm_and_gripper_xml": combine_arm_and_gripper_xml,
    }
    module = ast.Module(body=kept, type_ignores=[])
    exec(compile(module, str(source), "exec"), namespace, namespace)  # noqa: S102
    reference = namespace["ForwardKinematics"]("yam", "linear_4310", 6)  # type: ignore[operator]

    rng = np.random.default_rng(7)
    for _ in range(6):
        joints = START_JOINTS + rng.uniform(-0.4, 0.4, size=6)
        gripper = float(rng.uniform(0.0, 1.0))
        expected = reference.batch(joints[None, :], np.array([[gripper]]))["ee_transform"][0]
        np.testing.assert_allclose(yam.fk(joints, gripper), expected, atol=1e-12)
        np.testing.assert_array_equal(
            yam.robot_state_to_qpos(joints, gripper),
            reference.robot_state_to_qpos(joints, gripper),
        )


def test_ik_rejects_a_malformed_target(yam) -> None:
    with pytest.raises(ValueError, match="4x4"):
        yam.ik(np.eye(3), START_JOINTS, 0.5)


def test_clip_arm_joints_projects_mink_overshoot_onto_model_stops(yam) -> None:
    lower, upper = yam.joint_position_limits()
    overshot = lower.copy()
    overshot[3] = lower[3] - 8.53e-4
    clipped = yam.clip_arm_joints(overshot)
    assert clipped[3] == lower[3]
    np.testing.assert_allclose(yam.clip_arm_joints(upper + 1e-3), upper)
