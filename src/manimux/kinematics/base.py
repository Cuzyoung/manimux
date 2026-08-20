"""Arm kinematics contract shared by every policy that speaks end-effector poses.

A growing number of VLAs (XR-1 and friends) emit Cartesian end-effector targets
instead of joint positions. ManiMux executors only ever command joint groups, so
a policy adapter converts with this contract. Keeping it here rather than inside
one integration means each embodiment implements it once and every such policy
reuses it.

Poses are 4x4 homogeneous transforms of the arm's grasp site in the arm base
frame -- the same convention the recorded YAM episodes use.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class ArmKinematics(Protocol):
    """Forward/inverse kinematics for one arm."""

    @property
    def num_arm_joints(self) -> int: ...

    def fk(self, joints: FloatArray, gripper: float) -> FloatArray:
        """Joint positions -> 4x4 end-effector transform."""
        ...

    def ik(
        self,
        target_pose: FloatArray,
        init_joints: FloatArray,
        gripper: float,
    ) -> tuple[bool, FloatArray]:
        """4x4 end-effector target -> (converged, joint positions).

        ``init_joints`` seeds the solver; passing the current measured joints
        keeps the solution on the branch the arm is already on.
        """
        ...
