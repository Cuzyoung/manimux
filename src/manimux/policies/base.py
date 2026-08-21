from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Protocol, cast

from manimux.config import PolicyConfig, RobotConfig
from manimux.policies.capabilities import PolicyCapabilities
from manimux.types import ActionChunk, ActionContext, InferenceRequest, ObservationSnapshot


class PolicyModel(Protocol):
    def reset(self, session_id: str) -> None: ...

    def infer(self, request: InferenceRequest) -> object: ...

    def capabilities(self) -> PolicyCapabilities: ...

    def close(self) -> None: ...


class PolicyAdapter(Protocol):
    def build_observation(self, snapshot: ObservationSnapshot) -> ObservationSnapshot: ...

    def prepare_request(self, request: InferenceRequest) -> InferenceRequest: ...

    def decode_action(self, raw: object, context: ActionContext) -> ActionChunk: ...

    def validate(self, robot: RobotConfig, policy: PolicyConfig) -> None: ...


def decode_policy_action(
    adapter: PolicyAdapter,
    raw: object,
    context: ActionContext,
) -> ActionChunk:
    """Call a context-aware adapter while retaining the original one-argument API."""
    method = adapter.decode_action
    if len(inspect.signature(method).parameters) == 1:
        legacy = cast(Callable[[object], ActionChunk], method)
        return legacy(raw)
    return method(raw, context)


def prepare_policy_request(
    adapter: PolicyAdapter,
    request: InferenceRequest,
) -> InferenceRequest:
    method = getattr(adapter, "prepare_request", None)
    if not callable(method):
        return request
    prepared = method(request)
    if not isinstance(prepared, InferenceRequest):
        raise TypeError("policy adapter prepare_request must return InferenceRequest")
    return prepared
