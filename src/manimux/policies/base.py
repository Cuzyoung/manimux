from __future__ import annotations

from typing import Protocol

from manimux.config import PolicyConfig, RobotConfig
from manimux.types import ActionChunk, InferenceRequest, ObservationSnapshot


class PolicyModel(Protocol):
    def reset(self, session_id: str) -> None: ...

    def infer(self, request: InferenceRequest) -> ActionChunk: ...

    def close(self) -> None: ...


class PolicyAdapter(Protocol):
    def build_observation(self, snapshot: ObservationSnapshot) -> ObservationSnapshot: ...

    def decode_action(self, raw: ActionChunk) -> ActionChunk: ...

    def validate(self, robot: RobotConfig, policy: PolicyConfig) -> None: ...
