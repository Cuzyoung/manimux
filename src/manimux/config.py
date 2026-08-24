from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunConfig(StrictModel):
    task: str
    output_dir: Path = Path("./data")
    max_steps: int = Field(default=500, gt=0)


class RobotConfig(StrictModel):
    driver: str
    config: Path | None = None
    control_hz: float = Field(default=100.0, gt=0)
    group_dims: dict[str, int]
    options: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_group_dims(self) -> RobotConfig:
        if not self.group_dims or any(dim <= 0 for dim in self.group_dims.values()):
            raise ValueError("robot group_dims must contain positive dimensions")
        return self


class SensorConfig(StrictModel):
    name: str
    driver: str
    width: int = Field(default=64, gt=0)
    height: int = Field(default=48, gt=0)
    fps: float = Field(default=30.0, gt=0)
    options: dict[str, object] = Field(default_factory=dict)


class PolicyConfig(StrictModel):
    worker: str
    adapter: str
    device: str = "cpu"
    action_dt_s: float = Field(default=0.05, gt=0)
    trajectory_duration_s: float | None = Field(default=None, gt=0)
    timeout_s: float = Field(default=1.0, gt=0)
    horizon_steps: int = Field(default=20, gt=1)
    inference_delay_s: float = Field(default=0.04, ge=0)
    startup_timeout_s: float = Field(default=30.0, gt=0)
    options: dict[str, object] = Field(default_factory=dict)

    @property
    def effective_action_dt_s(self) -> float:
        """Spacing between policy points, optionally derived from a total duration."""

        if self.trajectory_duration_s is None:
            return self.action_dt_s
        return self.trajectory_duration_s / (self.horizon_steps - 1)


class ExecutorLimitsConfig(StrictModel):
    max_velocity: float = Field(default=2.0, gt=0)
    max_acceleration: float = Field(default=8.0, gt=0)
    position_limit_abs: float = Field(default=3.14, gt=0)


class SmoothConfig(ExecutorLimitsConfig):
    cutoff_hz: float = Field(default=8.0, gt=0)


class MPCConfig(ExecutorLimitsConfig):
    horizon_steps: int = Field(default=15, gt=1)
    dynamics_a: float = Field(default=0.85, gt=0, lt=1)
    tracking_weight: float = Field(default=10.0, gt=0)
    command_delta_weight: float = Field(default=1.0, ge=0)


class RtcConfig(StrictModel):
    """Physical Intelligence real-time chunking (arXiv:2506.07339).

    ``s_min`` is the smallest number of chunk steps to execute before starting
    the next inference; the effective execution horizon is ``max(s_min, d)``
    where ``d`` is the delay forecast. ``beta`` clips the guidance weight, which
    is unbounded at flow time 0; scale it with the sampler's step count using
    ``(t^2+(1-t)^2)/(t(1-t))`` at ``t = 1/steps`` (5 steps -> ~4.25, 10 -> ~9.1).
    """

    min_execute_steps: int | None = Field(default=None, gt=0)
    initial_delay_steps: int = Field(default=4, ge=0)
    delay_buffer_size: int = Field(default=10, gt=0)
    beta: float = Field(default=5.0, gt=0)
    # RTC changes only *when* to infer and *what* to condition on. How a chunk
    # is executed -- timeline, smoothing, limits -- stays the default runtime's,
    # configured by ``execution.executor`` and ``execution.smooth`` as usual.


class TemporalEnsembleConfig(StrictModel):
    """ACT temporal ensembling with an asynchronous query cadence."""

    coefficient: float = Field(default=0.01, ge=0)
    query_interval_steps: int = Field(default=1, gt=0)


class AacConfig(StrictModel):
    """Adaptive Action Chunking sampling and selection parameters."""

    num_samples: int = Field(default=20, gt=1)
    motion_threshold: float = Field(default=3.0, ge=0)
    ee_stats_path: str | None = None
    chunk_id_selector: Literal["0", "mean", "backward"] = "0"
    backward_beta: float = Field(default=0.99, gt=0, le=1)


class PaintConfig(StrictModel):
    """PAINT asynchronous execution parameters (arXiv:2606.19774)."""

    execution_steps: int = Field(default=10, gt=0)
    initial_delay_steps: int = Field(default=4, gt=0)
    delay_buffer_size: int = Field(default=10, gt=0)


class DvacConfig(StrictModel):
    """Denoising-Variance Adaptive Chunking (arXiv:2606.03847v1)."""

    tail_steps: int = Field(default=5, gt=1)
    alpha: float = Field(default=2.0, ge=0)
    rolling_window_size: int = Field(default=5, gt=0)
    min_execution_steps: int = Field(default=1, gt=0)
    max_execution_steps: int | None = Field(default=None, gt=0)


class ExecutionConfig(StrictModel):
    runtime: str = Field(default="manimux", min_length=1)
    executor: Literal["smooth", "mpc"] = "smooth"
    inference_schedule: Literal["deadline", "single_inflight"] = "deadline"
    refill_threshold_s: float = Field(default=0.4, gt=0)
    commit_lead_s: float = Field(default=0.02, ge=0)
    max_plan_age_s: float = Field(default=1.0, gt=0)
    blend_steps: int = Field(default=2, ge=0)
    smooth: SmoothConfig = SmoothConfig()
    mpc: MPCConfig = MPCConfig()
    rtc: RtcConfig = RtcConfig()
    temporal_ensemble: TemporalEnsembleConfig = TemporalEnsembleConfig()
    aac: AacConfig = AacConfig()
    paint: PaintConfig = PaintConfig()
    dvac: DvacConfig = DvacConfig()

    @model_validator(mode="after")
    def validate_strategy_fields(self) -> ExecutionConfig:
        if self.runtime == "rtc":
            ignored = {"inference_schedule", "refill_threshold_s"}.intersection(
                self.model_fields_set
            )
            if ignored:
                names = ", ".join(sorted(ignored))
                raise ValueError(f"execution fields are not used by RTC: {names}")
        if self.runtime == "act_temporal_ensemble":
            ignored = {"inference_schedule", "refill_threshold_s"}.intersection(
                self.model_fields_set
            )
            if ignored:
                names = ", ".join(sorted(ignored))
                raise ValueError(
                    f"execution fields are not used by ACT temporal ensembling: {names}"
                )
            if self.blend_steps != 0:
                raise ValueError(
                    "ACT temporal ensembling requires execution.blend_steps=0 "
                    "to avoid modifying the ensembled trajectory"
                )
        if self.runtime == "aac":
            if not self.aac.ee_stats_path:
                raise ValueError("execution.aac.ee_stats_path is required by AAC")
            ignored = {"inference_schedule", "refill_threshold_s"}.intersection(
                self.model_fields_set
            )
            if ignored:
                names = ", ".join(sorted(ignored))
                raise ValueError(f"execution fields are not used by AAC: {names}")
            if self.blend_steps != 0:
                raise ValueError(
                    "AAC requires execution.blend_steps=0 so the selected chunk "
                    "is not rewritten at commit"
                )
        if self.runtime == "paint":
            ignored = {"inference_schedule", "refill_threshold_s"}.intersection(
                self.model_fields_set
            )
            if ignored:
                names = ", ".join(sorted(ignored))
                raise ValueError(f"execution fields are not used by PAINT: {names}")
            if self.blend_steps != 0:
                raise ValueError(
                    "PAINT requires execution.blend_steps=0 so A[s:s+d] is not "
                    "rewritten before it becomes the next prefix condition"
                )
        if self.runtime == "autohorizon":
            ignored = {"inference_schedule", "refill_threshold_s"}.intersection(
                self.model_fields_set
            )
            if ignored:
                names = ", ".join(sorted(ignored))
                raise ValueError(f"execution fields are not used by AutoHorizon: {names}")
            if self.blend_steps != 0:
                raise ValueError(
                    "AutoHorizon requires execution.blend_steps=0 so the selected "
                    "official execution horizon is not rewritten at commit"
                )
        if self.runtime == "dvac":
            ignored = {"inference_schedule", "refill_threshold_s"}.intersection(
                self.model_fields_set
            )
            if ignored:
                names = ", ".join(sorted(ignored))
                raise ValueError(f"execution fields are not used by DVAC: {names}")
            if self.blend_steps != 0:
                raise ValueError(
                    "DVAC requires execution.blend_steps=0 so the selected "
                    "paper execution horizon is not rewritten at commit"
                )
        return self


class ViewerConfig(StrictModel):
    enabled: bool = False
    robot_adapter: str = ""
    policy_label: str = ""
    camera_hz: float = Field(default=5.0, ge=0)


class RecordingConfig(StrictModel):
    enabled: Literal[True] = True


class ManiMuxConfig(StrictModel):
    run: RunConfig
    robot: RobotConfig
    sensors: list[SensorConfig] = []
    policy: PolicyConfig
    execution: ExecutionConfig = ExecutionConfig()
    viewer: ViewerConfig = ViewerConfig()
    recording: RecordingConfig = RecordingConfig()

    @model_validator(mode="after")
    def validate_runtime_contract(self) -> ManiMuxConfig:
        if self.execution.runtime == "rtc":
            delay = self.execution.rtc.initial_delay_steps
            if 2 * delay > self.policy.horizon_steps:
                raise ValueError("RTC requires 2 * initial_delay_steps <= policy.horizon_steps")
        if self.execution.runtime == "act_temporal_ensemble":
            query_interval = self.execution.temporal_ensemble.query_interval_steps
            if query_interval >= self.policy.horizon_steps:
                raise ValueError(
                    "ACT temporal ensembling requires query_interval_steps "
                    "< policy.horizon_steps so consecutive chunks overlap"
                )
        if self.execution.runtime == "paint":
            execution = self.execution.paint.execution_steps
            delay = self.execution.paint.initial_delay_steps
            horizon = self.policy.horizon_steps
            if not delay <= execution <= horizon - delay:
                raise ValueError(
                    "PAINT requires initial_delay_steps <= execution_steps "
                    "<= horizon_steps - initial_delay_steps"
                )
        if self.execution.runtime == "dvac":
            dvac = self.execution.dvac
            maximum = dvac.max_execution_steps or self.policy.horizon_steps
            if not dvac.min_execution_steps <= maximum <= self.policy.horizon_steps:
                raise ValueError(
                    "DVAC requires min_execution_steps <= max_execution_steps "
                    "<= policy.horizon_steps"
                )
        return self


def load_config(path: str | Path) -> ManiMuxConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return ManiMuxConfig.model_validate(raw)
