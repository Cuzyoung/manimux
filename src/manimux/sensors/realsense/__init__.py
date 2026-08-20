"""RealSense camera hardware drivers.

Vendored from https://github.com/williamtsai726/YAM. These are plain camera
drivers with no policy or embodiment coupling, so any integration can use them.
"""

from manimux.sensors.realsense.base import CameraDriver, DummyCamera
from manimux.sensors.realsense.camera import RealSenseCamera, get_device_ids

__all__ = ["CameraDriver", "DummyCamera", "RealSenseCamera", "get_device_ids"]
