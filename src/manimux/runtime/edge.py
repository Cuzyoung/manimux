from __future__ import annotations

import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from manimux.clock import Clock, SystemClock
from manimux.config import ManiMuxConfig
from manimux.policies.fake import FakePolicyAdapter
from manimux.policies.worker import PolicyWorkerClient
from manimux.recording import EpisodeRecorder
from manimux.robots.base import RobotDriver
from manimux.robots.maniunicon import ManiUniConMeshcatDualArmDriver
from manimux.robots.mock import MockDualArmDriver
from manimux.runtime.executors import Executor, ExecutorError, MPCExecutor, SmoothExecutor
from manimux.runtime.safety import RuntimeState, SafetyGuard
from manimux.runtime.timeline import ActionTimeline
from manimux.sensors.mock import MockCameraDriver
from manimux.types import (
    GroupVector,
    InferenceRequest,
    ObservationSnapshot,
    RobotCommand,
    copy_group_vector,
)
from manimux.viewer import ViewerBridge


@dataclass(frozen=True, slots=True)
class RunResult:
    episode_dir: Path
    steps: int
    accepted_plans: int
    rejected_plans: int


class EdgeRuntime:
    def __init__(
        self,
        config: ManiMuxConfig,
        run_dir: Path,
        *,
        clock: Clock | None = None,
    ) -> None:
        if any(sensor.driver != "mock_camera" for sensor in config.sensors):
            raise ValueError("V1 currently supports only mock_camera sensors")
        self._config = config
        self._run_dir = run_dir
        self._clock = clock or SystemClock()
        self._control_dt_ns = int(1_000_000_000 / config.robot.control_hz)
        self._robot = self._build_robot()
        self._sensors = [
            MockCameraDriver(sensor.name, sensor.width, sensor.height, self._clock)
            for sensor in config.sensors
        ]
        self._adapter = FakePolicyAdapter()
        self._adapter.validate(config.robot, config.policy)
        self._session_id = f"session-{uuid.uuid4().hex}"
        self._worker = PolicyWorkerClient(config.policy, self._session_id)
        self._timeline = ActionTimeline(config.robot.group_dims)
        self._executor = self._build_executor()
        position_limit = (
            config.execution.smooth.position_limit_abs
            if config.execution.executor == "smooth"
            else config.execution.mpc.position_limit_abs
        )
        self._safety = SafetyGuard(config.robot.group_dims, position_limit)
        self._viewer = ViewerBridge(
            enabled=config.viewer.enabled,
            robot_adapter=config.viewer.robot_adapter,
            group_order=list(config.robot.group_dims),
        )
        self._state = RuntimeState.DISCONNECTED

    def _build_robot(self) -> RobotDriver:
        if self._config.robot.driver == "mock_dual_arm":
            return MockDualArmDriver(self._config.robot.group_dims, self._clock)
        if self._config.robot.driver == "maniunicon_meshcat_dual_arm":
            if self._config.robot.config is None:
                raise ValueError("maniunicon_meshcat_dual_arm requires robot.config")
            return ManiUniConMeshcatDualArmDriver.from_config_file(
                self._config.robot.config,
                self._config.robot.group_dims,
                self._clock,
            )
        raise ValueError(f"unsupported V1 robot driver {self._config.robot.driver!r}")

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
        last_inference_ms: float | None = None
        last_command: RobotCommand | None = None
        steps = 0
        completed = False
        try:
            self._robot.connect()
            for sensor in self._sensors:
                sensor.start()
            self._worker.start()
            initial_state = self._robot.get_state()
            self._safety.validate_state(initial_state)
            self._executor.reset(initial_state)
            self._state = RuntimeState.RUNNING
            next_tick_ns = self._clock.now_ns()

            for step in range(self._config.run.max_steps):
                now_ns = self._clock.now_ns()
                state = self._robot.get_state()
                self._safety.validate_state(state)
                frames = {
                    frame.name: frame for frame in (sensor.read() for sensor in self._sensors)
                }
                viewer_control = self._viewer.poll_control()
                if viewer_control.finish_requested:
                    recorder.event("viewer_finish_requested", step=step)
                    break
                if viewer_control.home_requested:
                    self._robot.home()
                    self._timeline = ActionTimeline(self._config.robot.group_dims)
                    recorder.event("viewer_home_requested", step=step)
                self._state = RuntimeState.PAUSED if viewer_control.paused else RuntimeState.RUNNING

                response = self._worker.poll()
                if response is not None:
                    if response.error is not None:
                        rejected_plans += 1
                        recorder.event(
                            "inference_rejected",
                            request_seq=response.request_seq,
                            reason=response.error,
                        )
                    elif (
                        response.session_id != self._session_id
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
                        chunk = self._adapter.decode_action(response.raw_action)
                        current = last_command.groups if last_command is not None else state.groups
                        result = self._timeline.commit(
                            chunk,
                            now_ns=now_ns,
                            commit_lead_ns=int(
                                self._config.execution.commit_lead_s * 1_000_000_000
                            ),
                            max_plan_age_ns=int(
                                self._config.execution.max_plan_age_s * 1_000_000_000
                            ),
                            current_command=current,
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
                            self._viewer.publish_plan(chunk, response.inference_ms)
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
                if self._timeline.remaining_ns(now_ns) < refill_ns and (
                    last_submitted_seq < 0 or request_expired
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
                    )
                    self._worker.submit_latest(request)
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
                    try:
                        command = self._executor.step(now_ns, state, reference)
                    except ExecutorError as exc:
                        self._state = RuntimeState.FAULT
                        recorder.event("executor_fault", reason=str(exc))
                        command = self._hold_command(now_ns, state.groups)
                else:
                    command = self._hold_command(now_ns, state.groups)
                self._safety.validate_command(command)
                self._robot.send_command(command)
                last_command = command
                self._viewer.publish_state(
                    state,
                    frames,
                    step=step,
                    max_steps=self._config.run.max_steps,
                )
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
                steps = step + 1
                next_tick_ns += self._control_dt_ns
                self._clock.sleep_until_ns(next_tick_ns)

            episode_dir = recorder.finish(
                success=self._state != RuntimeState.FAULT,
                terminal_reason=(
                    "completed" if self._state != RuntimeState.FAULT else "executor_fault"
                ),
                steps=steps,
                wall_time_s=time.perf_counter() - started_wall,
            )
            completed = True
            return RunResult(
                episode_dir=episode_dir,
                steps=steps,
                accepted_plans=accepted_plans,
                rejected_plans=rejected_plans,
            )
        finally:
            if not completed:
                recorder.abort("runtime_exception")
            self._state = RuntimeState.IDLE
            self._worker.close()
            with suppress(Exception):
                self._robot.stop()
            with suppress(Exception):
                self._robot.close()
            for sensor in self._sensors:
                with suppress(Exception):
                    sensor.close()
            with suppress(Exception):
                self._viewer.close()
            for sensor in self._sensors:
                sensor.close()
            self._viewer.close()
