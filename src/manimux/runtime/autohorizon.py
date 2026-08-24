from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from manimux.config import ManiMuxConfig
from manimux.policies.base import PolicyAdapter
from manimux.runtime.inference import CommitSettings, InferenceSubmission, RequestState
from manimux.runtime.safety import RuntimeState
from manimux.runtime.timeline import ActionTimeline, CommitResult
from manimux.types import (
    ActionChunk,
    GroupVector,
    InferenceRequest,
    InferenceResponse,
    ObservationSnapshot,
    copy_group_vector,
)


@dataclass(slots=True)
class AutoHorizonInferenceRequest(InferenceRequest):
    autohorizon: bool = True


class AutoHorizonInferenceStrategy:
    """Official synchronous execution cadence around an XPolicy attention hook."""

    def __init__(self, config: ManiMuxConfig) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return "autohorizon"

    @property
    def control_mode(self) -> str:
        return "autohorizon"

    @property
    def required_sampling_modes(self) -> frozenset[str]:
        return frozenset({"autohorizon"})

    def reset(self) -> None:
        return None

    def build_submission(
        self,
        *,
        session_id: str,
        request_seq: int,
        now_ns: int,
        snapshot: ObservationSnapshot,
        adapter: PolicyAdapter,
        timeline: ActionTimeline,
        request_state: RequestState,
        runtime_state: RuntimeState,
    ) -> InferenceSubmission | None:
        if (
            request_state.in_flight
            or runtime_state != RuntimeState.RUNNING
            or timeline.remaining_ns(now_ns) > 0
        ):
            return None
        deadline_ns = now_ns + int(self._config.policy.timeout_s * 1_000_000_000)
        return InferenceSubmission(
            request=AutoHorizonInferenceRequest(
                session_id=session_id,
                request_seq=request_seq,
                observation_time_ns=snapshot.state.monotonic_ns,
                deadline_ns=deadline_ns,
                observation=adapter.build_observation(snapshot),
                instruction=self._config.run.task,
            )
        )

    def commit_settings(
        self,
        *,
        response: InferenceResponse,
        measured: GroupVector,
        last_command: GroupVector,
    ) -> CommitSettings:
        del response, last_command
        return CommitSettings(
            current_command=copy_group_vector(measured),
            blend_steps=0,
            anchor_source="measured_state",
        )

    def prepare_chunk(
        self,
        *,
        chunk: ActionChunk,
        response: InferenceResponse,
        now_ns: int,
    ) -> ActionChunk:
        raw = response.raw_action
        metadata = raw.get("autohorizon") if isinstance(raw, Mapping) else None
        if not isinstance(metadata, Mapping):
            raise ValueError("AutoHorizon response is missing metadata")
        execution_steps = metadata.get("execution_steps")
        if (
            not isinstance(execution_steps, int)
            or isinstance(execution_steps, bool)
            or not 1 <= execution_steps <= chunk.horizon_steps
        ):
            raise ValueError(
                "AutoHorizon execution_steps must satisfy "
                f"1 <= e <= {chunk.horizon_steps}, got {execution_steps!r}"
            )
        return ActionChunk(
            plan_id=chunk.plan_id,
            request_seq=chunk.request_seq,
            observation_time_ns=now_ns,
            created_time_ns=chunk.created_time_ns,
            action_space=chunk.action_space,
            dt_ns=chunk.dt_ns,
            groups={name: values[:execution_steps].copy() for name, values in chunk.groups.items()},
        )

    def on_plan_accepted(
        self,
        *,
        chunk: ActionChunk,
        result: CommitResult,
        response: InferenceResponse,
        now_ns: int,
    ) -> dict[str, object]:
        del now_ns
        fields: dict[str, object] = {
            "selected_horizon_steps": chunk.horizon_steps,
            "trimmed_steps": result.trimmed_steps,
        }
        raw = response.raw_action
        metadata = raw.get("autohorizon") if isinstance(raw, Mapping) else None
        if isinstance(metadata, Mapping):
            for key in (
                "attention_step",
                "forward_horizon",
                "backward_horizon",
                "join_row",
                "method",
                "framework",
                "upstream_commit",
            ):
                value = metadata.get(key)
                if isinstance(value, int | float | str) and not isinstance(value, bool):
                    fields[key] = value
        return fields

    def on_response_rejected(self, response: InferenceResponse) -> None:
        del response

    def take_runtime_events(self, *, step: int) -> list[tuple[str, dict[str, object]]]:
        del step
        return []

    def on_tick(self, *, steps: int, loop_ms: float, control_dt_ns: int) -> None:
        del steps, loop_ms, control_dt_ns
