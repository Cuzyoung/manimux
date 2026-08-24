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
class DvacInferenceRequest(InferenceRequest):
    dvac: bool = True
    dvac_tail_steps: int = 5
    dvac_alpha: float = 2.0
    dvac_rolling_window_size: int = 5
    dvac_min_execution_steps: int = 1
    dvac_max_execution_steps: int = 50


class DvacInferenceStrategy:
    """Paper-synchronous execution around an XPolicy denoising-variance hook."""

    def __init__(self, config: ManiMuxConfig) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return "dvac"

    @property
    def control_mode(self) -> str:
        return "dvac"

    @property
    def required_sampling_modes(self) -> frozenset[str]:
        return frozenset({"dvac"})

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
        settings = self._config.execution.dvac
        maximum = settings.max_execution_steps or self._config.policy.horizon_steps
        deadline_ns = now_ns + int(self._config.policy.timeout_s * 1_000_000_000)
        return InferenceSubmission(
            request=DvacInferenceRequest(
                session_id=session_id,
                request_seq=request_seq,
                observation_time_ns=snapshot.state.monotonic_ns,
                deadline_ns=deadline_ns,
                observation=adapter.build_observation(snapshot),
                instruction=self._config.run.task,
                dvac_tail_steps=settings.tail_steps,
                dvac_alpha=settings.alpha,
                dvac_rolling_window_size=settings.rolling_window_size,
                dvac_min_execution_steps=settings.min_execution_steps,
                dvac_max_execution_steps=maximum,
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
        metadata = raw.get("dvac") if isinstance(raw, Mapping) else None
        if not isinstance(metadata, Mapping):
            raise ValueError("DVAC response is missing metadata")
        execution_steps = metadata.get("execution_steps")
        if (
            not isinstance(execution_steps, int)
            or isinstance(execution_steps, bool)
            or not 1 <= execution_steps <= chunk.horizon_steps
        ):
            raise ValueError(
                "DVAC execution_steps must satisfy "
                f"1 <= e <= {chunk.horizon_steps}, got {execution_steps!r}"
            )
        return ActionChunk(
            plan_id=chunk.plan_id,
            request_seq=chunk.request_seq,
            observation_time_ns=now_ns,
            created_time_ns=chunk.created_time_ns,
            action_space=chunk.action_space,
            dt_ns=chunk.dt_ns,
            groups={
                name: values[:execution_steps].copy()
                for name, values in chunk.groups.items()
            },
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
        metadata = raw.get("dvac") if isinstance(raw, Mapping) else None
        if isinstance(metadata, Mapping):
            for key in (
                "threshold",
                "rolling_mean",
                "rolling_std",
                "total_variance",
                "first_threshold_crossing",
                "rolling_states",
                "cold_start",
                "tail_steps",
                "action_dim",
                "method",
                "source",
            ):
                value = metadata.get(key)
                if value is None or isinstance(value, int | float | str | bool):
                    fields[key] = value
        return fields

    def on_response_rejected(self, response: InferenceResponse) -> None:
        del response

    def take_runtime_events(self, *, step: int) -> list[tuple[str, dict[str, object]]]:
        del step
        return []

    def on_tick(self, *, steps: int, loop_ms: float, control_dt_ns: int) -> None:
        del steps, loop_ms, control_dt_ns
