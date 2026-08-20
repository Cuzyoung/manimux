"""Real-time chunking runtime (arXiv:2506.07339).

Selected with ``execution.runtime: rtc``. The default runtime in
``manimux.runtime.edge`` is untouched, and so is the way a chunk is *executed*:
the timeline, the smoothing executor, safety, recording and the viewer all
behave exactly as they do on the default path.

RTC is two changes and nothing else, because the paper is about generation, not
execution:

1. **When to infer.** Instead of refilling when the timeline runs low, start the
   next inference once ``s`` steps of the current chunk have been consumed,
   where ``s = max(s_min, d)`` and ``d`` is a conservative forecast (the max of
   recent observed delays). Feasibility requires ``d <= s <= H - d``.

2. **What to condition on.** Send the unexecuted tail of the committed chunk,
   left-shifted so index 0 lines up with the new chunk's index 0, together with
   the paper's soft mask: the first ``d`` steps frozen, then an exponential
   decay to zero. The policy server applies the guidance while denoising.

Everything downstream of the produced chunk stays the default runtime's job.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from pathlib import Path

import numpy as np

from manimux.config import ManiMuxConfig
from manimux.policies.base import decode_policy_action
from manimux.recording.episode import EpisodeRecorder
from manimux.runtime.edge import EdgeRuntime, RunResult, RuntimeState
from manimux.runtime.rtc.mask import inpainting_condition
from manimux.runtime.rtc.request import RtcInferenceRequest
from manimux.runtime.timeline import ActionTimeline
from manimux.types import (
    ActionContext,
    InferenceRequest,
    ObservationSnapshot,
    SensorFrame,
    copy_group_vector,
)


class RtcRuntime(EdgeRuntime):
    """The default runtime's control loop with RTC's inference schedule."""

    def __init__(self, config: ManiMuxConfig, run_dir: Path) -> None:
        super().__init__(config, run_dir)
        rtc = config.execution.rtc
        self._group_order = tuple(config.robot.group_dims)
        self._rtc_beta = float(rtc.beta)
        self._min_execute_steps = rtc.min_execute_steps
        self._delay_forecast: deque[int] = deque(
            [int(rtc.initial_delay_steps)], maxlen=int(rtc.delay_buffer_size)
        )

    def _execution_horizon(self, horizon: int, delay: int) -> int:
        """``s`` for this cycle, honouring the paper's ``d <= s <= H - d``."""
        floor = self._min_execute_steps or max(1, horizon // 2)
        return int(min(max(floor, delay), horizon - delay))

    def _report(
        self,
        steps: int,
        loop_ms: list[float],
        latency: list[tuple[int, float, float]],
        horizon: int,
    ) -> None:
        if not loop_ms:
            return
        recent = latency[-5:]
        if recent:
            delay = sum(d for d, _, _ in recent) / len(recent)
            server = sum(s for _, s, _ in recent) / len(recent)
            trip = sum(r for _, _, r in recent) / len(recent)
            margin = horizon - max(self._delay_forecast) - delay
            infer = (
                f"| infer server {server:5.0f}ms trip {trip:5.0f}ms "
                f"d={delay:4.1f} margin={margin:+5.1f}步"
            )
        else:
            infer = "| infer --"
        print(
            f"[rtc] step {steps:5d} | loop {sum(loop_ms) / len(loop_ms):5.1f}/"
            f"{self._control_dt_ns / 1e6:.0f}ms max {max(loop_ms):5.1f} {infer}",
            flush=True,
        )

    def run(self) -> RunResult:  # noqa: C901 - one control loop, kept linear
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
                "executor_kind": f"rtc+{self._config.execution.executor}",
            },
        )
        accepted_plans = 0
        rejected_plans = 0
        request_seq = 0
        request_in_flight = False
        inference_started_ns = 0
        forecast_used = 0
        last_inference_ms: float | None = None
        robot_connected = False
        steps = 0
        completed = False
        terminal_reason = "completed"
        abort_reason = "runtime_exception"
        infeasible_reported = False
        latency: list[tuple[int, float, float]] = []
        loop_ms: list[float] = []
        last_report_step = 0
        horizon_steps = 0
        last_command: dict | None = None

        try:
            for sensor in self._sensors:
                sensor.start()
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
                    "control_mode": "rtc",
                },
            )
            next_tick_ns = self._clock.now_ns()

            while steps < self._config.run.max_steps:
                loop_start_ns = self._clock.now_ns()
                now_ns = loop_start_ns
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
                    self._state = RuntimeState.PAUSED
                    recorder.event("viewer_home_requested", step=steps)
                    next_tick_ns = self._clock.now_ns()
                    continue
                self._state = (
                    RuntimeState.RUNNING
                    if viewer_control.step_once or not viewer_control.paused
                    else RuntimeState.PAUSED
                )

                response = self._worker.poll()
                if response is not None:
                    request_in_flight = False
                    chunk = None
                    if response.error is not None or response.raw_action is None:
                        rejected_plans += 1
                        recorder.event(
                            "inference_rejected",
                            request_seq=response.request_seq,
                            reason=response.error or "empty_response",
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
                            # The blend seeds the new plan at this value, so it
                            # has to be the last *command*. Seeding it with the
                            # measurement yanks the plan back by the servo's
                            # tracking error at every commit -- barely visible on
                            # the default runtime, which refills rarely, but RTC
                            # re-plans every s steps and it turns into a stutter.
                            current_command=(
                                last_command
                                if last_command is not None
                                else state.groups
                            ),
                            blend_steps=self._config.execution.blend_steps,
                        )
                        if result.accepted:
                            accepted_plans += 1
                            last_inference_ms = response.inference_ms
                            # The mask indexes chunk rows, so the delay has to be
                            # counted in chunk steps too. They only coincide when
                            # control_hz happens to equal 1/action_dt_s.
                            measured_delay = max(
                                0, round((now_ns - inference_started_ns) / chunk.dt_ns)
                            )
                            self._delay_forecast.append(measured_delay)
                            latency.append(
                                (
                                    measured_delay,
                                    response.inference_ms,
                                    (now_ns - inference_started_ns) / 1e6,
                                )
                            )
                            recorder.record_plan(chunk)
                            recorder.event(
                                "plan_accepted",
                                plan_id=chunk.plan_id,
                                request_seq=chunk.request_seq,
                                measured_delay=measured_delay,
                                forecast_delay=forecast_used,
                                server_ms=round(response.inference_ms, 1),
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

                # ---- RTC change 1: when to infer ----------------------------
                committed = self._timeline.active_horizon()
                if committed is not None:
                    horizon_steps = committed.horizon_steps
                if (
                    not request_in_flight
                    and self._worker.is_alive
                    and self._state == RuntimeState.RUNNING
                ):
                    condition = None
                    weights = None
                    ready = committed is None  # the first chunk is unconditioned
                    if committed is not None:
                        horizon = committed.horizon_steps
                        executed = self._timeline.cursor(now_ns)
                        delay = max(self._delay_forecast)
                        if 2 * delay > horizon and not infeasible_reported:
                            infeasible_reported = True
                            recorder.event(
                                "rtc_delay_infeasible",
                                step=steps,
                                delay=delay,
                                horizon=horizon,
                            )
                        if executed >= self._execution_horizon(horizon, delay):
                            ready = True
                            if delay <= executed <= horizon - delay:
                                # ---- RTC change 2: what to condition on -----
                                rows = np.concatenate(
                                    [committed.groups[n] for n in self._group_order],
                                    axis=1,
                                )
                                condition, weights = inpainting_condition(
                                    rows, executed_steps=executed, delay_steps=delay
                                )
                                forecast_used = delay
                    if ready:
                        request_seq += 1
                        snapshot = ObservationSnapshot(state=state, frames=frames)
                        request: InferenceRequest = RtcInferenceRequest(
                            session_id=self._session_id,
                            request_seq=request_seq,
                            observation_time_ns=state.monotonic_ns,
                            deadline_ns=now_ns
                            + int(self._config.policy.timeout_s * 1_000_000_000),
                            observation=self._adapter.build_observation(snapshot),
                            instruction=self._config.run.task,
                            action_condition=(
                                None if condition is None else condition.astype(np.float64)
                            ),
                            condition_weights=(
                                None if weights is None else weights.astype(np.float64)
                            ),
                            rtc_beta=self._rtc_beta,
                        )
                        self._worker.submit_latest(request)
                        request_in_flight = True
                        inference_started_ns = now_ns
                        recorder.event(
                            "inference_submitted",
                            request_seq=request_seq,
                            executed_steps=self._timeline.cursor(now_ns),
                            forecast_delay=forecast_used,
                            conditioned=condition is not None,
                        )

                # ---- everything below is the default runtime, verbatim ------
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
                else:
                    self._executor.reset(state)
                    command = self._hold_command(now_ns, state.groups)
                self._safety.validate_command(command)
                self._robot.send_command(command)
                last_command = copy_group_vector(command.groups)
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
                    steps += 1

                loop_ms.append((self._clock.now_ns() - loop_start_ns) / 1e6)
                report_every = max(1, int(round(2e9 / self._control_dt_ns)))
                if steps - last_report_step >= report_every:
                    last_report_step = steps
                    self._report(steps, loop_ms, latency, horizon_steps)
                    loop_ms.clear()

                next_tick_ns += self._control_dt_ns
                finished_tick_ns = self._clock.now_ns()
                if next_tick_ns <= finished_tick_ns:
                    recorder.event(
                        "control_overrun", lag_ns=finished_tick_ns - next_tick_ns, step=steps
                    )
                    next_tick_ns = finished_tick_ns + self._control_dt_ns
                self._clock.sleep_until_ns(next_tick_ns)

            episode_dir = recorder.finish(
                success=True,
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
                success=True,
                terminal_reason=terminal_reason,
            )
        except KeyboardInterrupt:
            abort_reason = "KeyboardInterrupt"
            raise
        except BaseException as exc:
            abort_reason = type(exc).__name__
            raise
        finally:
            # Same teardown contract as the default runtime.
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
            for closer in (self._robot.close, self._worker.close):
                try:
                    closer()
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
