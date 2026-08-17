from __future__ import annotations

from typing import Protocol

from manimux.types import SensorFrame

SensorRead = SensorFrame | dict[str, SensorFrame]


class SensorDriver(Protocol):
    def start(self) -> None: ...

    def read(self) -> SensorRead: ...

    def close(self) -> None: ...
