"""YAM bimanual arm: the ManiMux RobotDriver and the i2rt hardware wrapper.

Lives outside any policy integration because every policy that drives a YAM
uses the same body. A run selects it with ``robot.driver: yam_dual``.
"""

from manimux.robots.yam.arm import YAMRobot
from manimux.robots.yam.base import BimanualRobot, Robot
from manimux.robots.yam.driver import YamDualArmDriver, build_robot

__all__ = ["BimanualRobot", "Robot", "YAMRobot", "YamDualArmDriver", "build_robot"]
