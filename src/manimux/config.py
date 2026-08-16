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


class PolicyConfig(StrictModel):
    worker: str
    adapter: str
    device: str = "cpu"
    action_dt_s: float = Field(default=0.05, gt=0)
    timeout_s: float = Field(default=1.0, gt=0)
    horizon_steps: int = Field(default=20, gt=1)
    inference_delay_s: float = Field(default=0.04, ge=0)


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


class ExecutionConfig(StrictModel):
    executor: Literal["smooth", "mpc"] = "smooth"
    refill_threshold_s: float = Field(default=0.4, gt=0)
    commit_lead_s: float = Field(default=0.02, ge=0)
    max_plan_age_s: float = Field(default=1.0, gt=0)
    underrun_hold_s: float = Field(default=0.5, ge=0)
    blend_steps: int = Field(default=2, ge=0)
    smooth: SmoothConfig = SmoothConfig()
    mpc: MPCConfig = MPCConfig()


class ViewerConfig(StrictModel):
    enabled: bool = False
    robot_adapter: str = ""


class RecordingConfig(StrictModel):
    enabled: bool = True


class ManiMuxConfig(StrictModel):
    run: RunConfig
    robot: RobotConfig
    sensors: list[SensorConfig] = []
    policy: PolicyConfig
    execution: ExecutionConfig = ExecutionConfig()
    viewer: ViewerConfig = ViewerConfig()
    recording: RecordingConfig = RecordingConfig()


def load_config(path: str | Path) -> ManiMuxConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return ManiMuxConfig.model_validate(raw)
