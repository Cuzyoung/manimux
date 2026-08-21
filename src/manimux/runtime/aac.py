from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from manimux.config import ManiMuxConfig
from manimux.policies.base import PolicyAdapter
from manimux.runtime.inference import (
    CommitSettings,
    InferenceSubmission,
    RequestState,
)
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
class AacInferenceRequest(InferenceRequest):
    aac_num_samples: int = 20
    aac_motion_threshold: float = 3.0
    aac_ee_stats_path: str | None = None
    aac_chunk_id_selector: str = "0"
    aac_backward_beta: float = 0.99


class AacInferenceStrategy:
    """Official AAC synchronous cadence around an XPolicy multi-sample hook."""

    def __init__(self, config: ManiMuxConfig) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return "aac"

    @property
    def control_mode(self) -> str:
        return "aac"

    @property
    def required_sampling_modes(self) -> frozenset[str]:
        return frozenset({"aac"})

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
        aac = self._config.execution.aac
        deadline_ns = now_ns + int(self._config.policy.timeout_s * 1_000_000_000)
        request = AacInferenceRequest(
            session_id=session_id,
            request_seq=request_seq,
            observation_time_ns=snapshot.state.monotonic_ns,
            deadline_ns=deadline_ns,
            observation=adapter.build_observation(snapshot),
            instruction=self._config.run.task,
            aac_num_samples=aac.num_samples,
            aac_motion_threshold=aac.motion_threshold,
            aac_ee_stats_path=aac.ee_stats_path,
            aac_chunk_id_selector=aac.chunk_id_selector,
            aac_backward_beta=aac.backward_beta,
        )
        return InferenceSubmission(
            request=request,
            event_fields={
                "num_samples": aac.num_samples,
                "motion_threshold": aac.motion_threshold,
                "ee_stats_path": aac.ee_stats_path,
                "chunk_id_selector": aac.chunk_id_selector,
            },
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
        del response
        return ActionChunk(
            plan_id=chunk.plan_id,
            request_seq=chunk.request_seq,
            observation_time_ns=now_ns,
            created_time_ns=chunk.created_time_ns,
            action_space=chunk.action_space,
            dt_ns=chunk.dt_ns,
            groups={name: values.copy() for name, values in chunk.groups.items()},
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
        metadata = raw.get("aac") if isinstance(raw, Mapping) else None
        if isinstance(metadata, Mapping):
            for key in ("chunk_id", "entropy_elbow", "motion_floor"):
                value = metadata.get(key)
                if isinstance(value, int | float) and not isinstance(value, bool):
                    fields[key] = value
        return fields

    def on_response_rejected(self, response: InferenceResponse) -> None:
        del response

    def take_runtime_events(self, *, step: int) -> list[tuple[str, dict[str, object]]]:
        del step
        return []

    def on_tick(self, *, steps: int, loop_ms: float, control_dt_ns: int) -> None:
        del steps, loop_ms, control_dt_ns
