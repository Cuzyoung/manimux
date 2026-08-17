from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from manimux.clock import Clock, SystemClock
from manimux.config import ManiMuxConfig
from manimux.policies import build_policy_adapter
from manimux.policies.base import decode_policy_action
from manimux.policies.worker import PolicyWorkerClient
from manimux.recording import EpisodeRecorder
from manimux.robots import build_robot
from manimux.robots.base import RobotDriver
from manimux.runtime.executors import Executor, MPCExecutor, SmoothExecutor
from manimux.runtime.safety import RuntimeState, SafetyGuard
from manimux.runtime.timeline import ActionTimeline
from manimux.sensors import build_sensor
from manimux.types import (
    ActionContext,
    GroupVector,
    InferenceRequest,
    ObservationSnapshot,
    RobotCommand,
    SensorFrame,
    copy_group_vector,
)
from manimux.viewer import ViewerBridge


@dataclass(frozen=True, slots=True)
class RunResult:
    episode_dir: Path
    steps: int
    accepted_plans: int
    rejected_plans: int
    success: bool
    terminal_reason: str


class EdgeRuntime:
    def __init__(
        self,
        config: ManiMuxConfig,
        run_dir: Path,
        *,
        clock: Clock | None = None,
    ) -> None:
        self._config = config
        self._run_dir = run_dir
        self._clock = clock or SystemClock()
        self._control_dt_ns = int(1_000_000_000 / config.robot.control_hz)
        self._robot = self._build_robot()
        self._sensors = [build_sensor(sensor, self._clock) for sensor in config.sensors]
        self._adapter = build_policy_adapter(config.robot, config.policy)
        self._adapter.validate(config.robot, config.policy)
        self._session_id = f"session-{uuid.uuid4().hex}"
        self._worker = PolicyWorkerClient(config.policy, self._session_id)
        self._timeline = ActionTimeline(config.robot.group_dims)
        self._executor = self._build_executor()
        limits = (
            config.execution.smooth
            if config.execution.executor == "smooth"
            else config.execution.mpc
        )
        self._safety = SafetyGuard(config.robot.group_dims, limits.position_limit_abs)
        self._viewer = ViewerBridge(
            enabled=config.viewer.enabled,
            robot_adapter=config.viewer.robot_adapter,
            group_order=list(config.robot.group_dims),
            policy=config.viewer.policy_label or "manimux-local",
            instruction=config.run.task if config.viewer.policy_label else "",
            camera_hz=config.viewer.camera_hz,
        )
        self._state = RuntimeState.DISCONNECTED

    def _build_robot(self) -> RobotDriver:
        return build_robot(self._config.robot, self._clock)

    def _build_executor(self) -> Executor:
        control_dt_s = self._control_dt_ns / 1_000_000_000
        if self._config.execution.executor == "smooth":
            return SmoothExecutor(self._config.execution.smooth, control_dt_s)
        return MPCExecutor(self._config.execution.mpc, control_dt_s)

    def _hold_command(self, now_ns: int, groups: GroupVector) -> RobotCommand:
        return RobotCommand(
            groups=copy_group_vector(groups),
            monotonic_ns=now_ns,
            plan_id=self._timeline.active_plan_id,
        )

    def run(self) -> RunResult:
        started_wall = time.perf_counter()
        episode_id = f"episode-{uuid.uuid4().hex[:12]}"
        recorder = EpisodeRecorder(
            self._run_dir,
            episode_id,
            self._config.robot.group_dims,
            metadata={
                "episode_id": episode_id,
                "session_id": self._session_id,
                "task": self._config.run.task,
                "executor_kind": self._config.execution.executor,
            },
        )
        accepted_plans = 0
        rejected_plans = 0
        request_seq = 0
        last_submitted_seq = -1
        last_request_deadline_ns = 0
        request_in_flight = False
        last_inference_ms: float | None = None
        discard_responses_through = -1
        worker_failure_reported = False
        robot_connected = False
        steps = 0
        completed = False
        terminal_reason = "completed"
        abort_reason = "runtime_exception"
        try:
            for sensor in self._sensors:
                sensor.start()
                # A ping only proves the server socket is alive. Pull one real
                # observation before enabling motors so stale/missing cameras fail closed.
                sensor.read()
            self._worker.start()
            self._robot.connect()
            robot_connected = True
            initial_state = self._robot.get_state()
            self._safety.validate_state(initial_state)
            self._executor.reset(initial_state)
            self._state = RuntimeState.RUNNING
            self._viewer.publish_event(
                "episode_started",
                metadata={
                    "instruction": self._config.run.task,
                    "max_steps": self._config.run.max_steps,
                    "control_mode": "managed",
                },
            )
            next_tick_ns = self._clock.now_ns()

            while steps < self._config.run.max_steps:
                now_ns = self._clock.now_ns()
                state = self._robot.get_state()
                self._safety.validate_state(state)
                frames: dict[str, SensorFrame] = {}
                for sensor in self._sensors:
                    reading = sensor.read()
                    batch = {reading.name: reading} if isinstance(reading, SensorFrame) else reading
                    overlap = set(frames).intersection(batch)
                    if overlap:
                        raise RuntimeError(f"duplicate sensor frames: {sorted(overlap)}")
                    frames.update(batch)
                viewer_control = self._viewer.poll_control()
                if viewer_control.finish_requested:
                    terminal_reason = "viewer_finish_requested"
                    recorder.event("viewer_finish_requested", step=steps)
                    break
                if viewer_control.home_requested:
                    self._robot.home()
                    state = self._robot.get_state()
                    self._safety.validate_state(state)
                    self._timeline = ActionTimeline(self._config.robot.group_dims)
                    self._executor.reset(state)
                    discard_responses_through = max(discard_responses_through, request_seq)
                    self._state = RuntimeState.PAUSED
                    terminal_reason = "completed"
                    recorder.event("viewer_home_requested", step=steps)
                    # home() can take seconds. Rebase the controller clock instead
                    # of issuing a burst of catch-up commands from stale state.
                    next_tick_ns = self._clock.now_ns()
                    continue
                self._state = (
                    RuntimeState.RUNNING
                    if viewer_control.step_once or not viewer_control.paused
                    else RuntimeState.PAUSED
                )

                if not self._worker.is_alive and not worker_failure_reported:
                    worker_failure_reported = True
                    recorder.event("policy_worker_stopped", step=steps)

                response = self._worker.poll()
                if response is not None:
                    request_in_flight = False
                    if response.error is not None:
                        rejected_plans += 1
                        recorder.event(
                            "inference_rejected",
                            request_seq=response.request_seq,
                            reason=response.error,
                        )
                    elif (
                        response.session_id != self._session_id
                        or response.request_seq <= discard_responses_through
                        or response.request_seq < last_submitted_seq
                        or response.finished_time_ns > last_request_deadline_ns
                        or response.raw_action is None
                    ):
                        rejected_plans += 1
                        recorder.event(
                            "inference_rejected",
                            request_seq=response.request_seq,
                            reason="stale_or_expired_response",
                        )
                    else:
                        try:
                            chunk = decode_policy_action(
                                self._adapter,
                                response.raw_action,
                                ActionContext(
                                    request_seq=response.request_seq,
                                    observation_time_ns=response.observation_time_ns,
                                    created_time_ns=response.finished_time_ns,
                                ),
                            )
                        except (TypeError, ValueError) as exc:
                            rejected_plans += 1
                            recorder.event(
                                "plan_rejected",
                                request_seq=response.request_seq,
                                reason=f"invalid_action:{type(exc).__name__}:{exc}",
                            )
                            chunk = None
                        if chunk is not None:
                            result = self._timeline.commit(
                                chunk,
                                now_ns=now_ns,
                                commit_lead_ns=int(
                                    self._config.execution.commit_lead_s * 1_000_000_000
                                ),
                                max_plan_age_ns=int(
                                    self._config.execution.max_plan_age_s * 1_000_000_000
                                ),
                                current_command=state.groups,
                                blend_steps=self._config.execution.blend_steps,
                            )
                            if result.accepted:
                                accepted_plans += 1
                                last_inference_ms = response.inference_ms
                                recorder.record_plan(chunk)
                                recorder.event(
                                    "plan_accepted",
                                    plan_id=chunk.plan_id,
                                    request_seq=chunk.request_seq,
                                    trimmed_steps=result.trimmed_steps,
                                )
                                self._viewer.publish_plan(
                                    chunk,
                                    response.inference_ms,
                                    committed=self._timeline.active_horizon(),
                                )
                            else:
                                rejected_plans += 1
                                recorder.event(
                                    "plan_rejected",
                                    plan_id=chunk.plan_id,
                                    request_seq=chunk.request_seq,
                                    reason=result.reason,
                                )

                refill_ns = int(self._config.execution.refill_threshold_s * 1_000_000_000)
                request_expired = now_ns > last_request_deadline_ns
                request_ready = (
                    not request_in_flight
                    if self._config.execution.inference_schedule == "single_inflight"
                    else last_submitted_seq < 0 or request_expired
                )
                if (
                    self._timeline.remaining_ns(now_ns) < refill_ns
                    and request_ready
                    and self._worker.is_alive
                ):
                    request_seq += 1
                    deadline_ns = now_ns + int(self._config.policy.timeout_s * 1_000_000_000)
                    snapshot = ObservationSnapshot(state=state, frames=frames)
                    request = InferenceRequest(
                        session_id=self._session_id,
                        request_seq=request_seq,
                        observation_time_ns=state.monotonic_ns,
                        deadline_ns=deadline_ns,
                        observation=self._adapter.build_observation(snapshot),
                        instruction=self._config.run.task,
                    )
                    self._worker.submit_latest(request)
                    request_in_flight = True
                    last_submitted_seq = request_seq
                    last_request_deadline_ns = deadline_ns
                    recorder.event("inference_submitted", request_seq=request_seq)

                reference = self._timeline.reference_horizon(
                    now_ns=now_ns,
                    dt_ns=self._control_dt_ns,
                    horizon_steps=self._executor.horizon_steps,
                )
                scheduled = copy_group_vector(state.groups)
                if self._state == RuntimeState.RUNNING and reference is not None:
                    scheduled = {
                        name: values[0].copy() for name, values in reference.groups.items()
                    }
                    command = self._executor.step(now_ns, state, reference)
                elif self._state == RuntimeState.RUNNING:
                    # Match MolmoAct's measured-state chunk stitching: while a new
                    # chunk is unavailable, hold the measured pose and resume from it.
                    self._executor.reset(state)
                    command = self._hold_command(now_ns, state.groups)
                else:
                    self._executor.reset(state)
                    command = self._hold_command(now_ns, state.groups)
                self._safety.validate_command(command)
                self._robot.send_command(command)
                self._viewer.publish_state(
                    state,
                    frames,
                    step=steps,
                    max_steps=self._config.run.max_steps,
                    chunk_index=self._timeline.cursor(now_ns),
                    active_chunk_id=(
                        None
                        if self._timeline.accepted_request_seq < 0
                        else self._timeline.accepted_request_seq
                    ),
                )
                if self._state == RuntimeState.RUNNING:
                    recorder.record_tick(
                        monotonic_ns=now_ns,
                        state=state,
                        scheduled=scheduled,
                        optimized=command.groups,
                        command=command.groups,
                        plan_id=command.plan_id,
                        inference_ms=last_inference_ms,
                        camera_times_ns={
                            name: frame.capture_monotonic_ns for name, frame in frames.items()
                        },
                    )
                if self._state == RuntimeState.RUNNING:
                    steps += 1
                next_tick_ns += self._control_dt_ns
                finished_tick_ns = self._clock.now_ns()
                if next_tick_ns <= finished_tick_ns:
                    recorder.event(
                        "control_overrun",
                        lag_ns=finished_tick_ns - next_tick_ns,
                        step=steps,
                    )
                    next_tick_ns = finished_tick_ns + self._control_dt_ns
                self._clock.sleep_until_ns(next_tick_ns)

            success = True
            episode_dir = recorder.finish(
                success=success,
                terminal_reason=terminal_reason,
                steps=steps,
                wall_time_s=time.perf_counter() - started_wall,
            )
            completed = True
            return RunResult(
                episode_dir=episode_dir,
                steps=steps,
                accepted_plans=accepted_plans,
                rejected_plans=rejected_plans,
                success=success,
                terminal_reason=terminal_reason,
            )
        except KeyboardInterrupt:
            abort_reason = "KeyboardInterrupt"
            raise
        except BaseException as exc:
            abort_reason = type(exc).__name__
            raise
        finally:
            faulted = not completed and abort_reason != "KeyboardInterrupt"
            self._state = RuntimeState.IDLE
            cleanup_errors: list[BaseException] = []
            stopped_cleanly = True
            try:
                self._robot.stop()
            except BaseException as exc:
                stopped_cleanly = False
                cleanup_errors.append(exc)
            if (
                robot_connected
                and stopped_cleanly
                and not faulted
                and bool(self._config.robot.options.get("home_on_close", False))
            ):
                try:
                    self._robot.home()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            try:
                self._robot.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
            try:
                self._worker.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
            for sensor in self._sensors:
                try:
                    sensor.close()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            try:
                self._viewer.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
            if not completed:
                try:
                    recorder.abort(abort_reason)
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if cleanup_errors:
                raise BaseExceptionGroup("runtime cleanup failed", cleanup_errors)
