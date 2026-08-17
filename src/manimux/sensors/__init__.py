from __future__ import annotations

from collections.abc import Callable

from manimux.clock import Clock
from manimux.config import SensorConfig
from manimux.plugins import load_plugin
from manimux.sensors.base import SensorDriver
from manimux.sensors.mock import MockCameraDriver

SensorFactory = Callable[[SensorConfig, Clock], SensorDriver]


def _mock_camera_factory(config: SensorConfig, clock: Clock) -> SensorDriver:
    return MockCameraDriver(config.name, config.width, config.height, clock)


def _camera_server_factory(config: SensorConfig, clock: Clock) -> SensorDriver:
    from manimux.integrations.molmoact_yam.sensor_plugin import build_sensor as build_camera_sensor

    return build_camera_sensor(config, clock)


_BUILTINS: dict[str, SensorFactory] = {
    "mock_camera": _mock_camera_factory,
    "camera_server": _camera_server_factory,
}


def build_sensor(config: SensorConfig, clock: Clock) -> SensorDriver:
    factory = load_plugin(
        config.driver,
        group="manimux.sensors",
        builtins=_BUILTINS,
    )
    return factory(config, clock)


__all__ = [
    "MockCameraDriver",
    "SensorDriver",
    "SensorFactory",
    "build_sensor",
]
