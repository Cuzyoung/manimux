from __future__ import annotations

from typing import Protocol

from manimux.types import SensorFrame


class SensorDriver(Protocol):
    def start(self) -> None: ...

    def read(self) -> SensorFrame: ...

    def close(self) -> None: ...
