from __future__ import annotations

import time

import numpy as np

from manimux.config import load_config
from manimux.integrations.sapolicy_yam.policy_plugin import (
    SAPolicyInferenceRequest,
    SAPolicyTcpPolicyModel,
    build_adapter,
)
from manimux.runtime.inference import build_inference_strategy
from manimux.runtime.paint import PaintInferenceRequest
from manimux.types import ObservationSnapshot, RobotState, SensorFrame

CONFIG = "configs/sapolicy/yam/infra/paint-offline-probe.yaml"


def _snapshot(now_ns: int) -> ObservationSnapshot:
    return ObservationSnapshot(
        state=RobotState(
            groups={
                "left_arm": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]),
                "right_arm": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]),
            },
            monotonic_ns=now_ns,
            sequence=1,
        ),
        frames={
            "front_camera": SensorFrame(
                name="front_camera",
                data=np.zeros((8, 8, 3), dtype=np.uint8),
                capture_monotonic_ns=now_ns,
                sequence=1,
            )
        },
    )


def test_sapolicy_paint_config_uses_shared_strategy() -> None:
    config = load_config(CONFIG)

    assert build_inference_strategy(config).name == "paint"
    assert config.execution.paint.execution_steps == 8
    assert config.execution.paint.initial_delay_steps == 8
    assert config.execution.executor == "direct"
    assert config.execution.blend_steps == 0


def test_sapolicy_adapter_converts_joint_prefix_to_wire_ee() -> None:
    config = load_config(CONFIG)
    adapter = build_adapter(config.robot, config.policy)
    now_ns = time.monotonic_ns()
    prefix = np.tile(
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5] * 2),
        (3, 1),
    )
    request = PaintInferenceRequest(
        session_id="session",
        request_seq=2,
        observation_time_ns=now_ns,
        deadline_ns=now_ns + 1_000_000_000,
        observation=_snapshot(now_ns),
        instruction="put bottles in bin",
        paint_action_prefix=prefix,
        paint_delay_steps=3,
        paint_execution_steps=8,
    )

    prepared = adapter.prepare_request(request)

    assert isinstance(prepared, SAPolicyInferenceRequest)
    assert prepared.sapolicy_paint_delay_steps == 3
    assert prepared.sapolicy_paint_action_prefix is not None
    assert prepared.sapolicy_paint_action_prefix.shape == (3, 16)
    assert np.isfinite(prepared.sapolicy_paint_action_prefix).all()
    np.testing.assert_allclose(
        prepared.sapolicy_paint_action_prefix[:, [7, 15]], 0.5
    )


def test_sapolicy_worker_advertises_and_dispatches_paint() -> None:
    config = load_config(CONFIG)
    model = SAPolicyTcpPolicyModel(config.policy)
    calls: list[tuple[str, object]] = []

    class _Client:
        def call(self, command, argument=None, *, timeout_s):
            del timeout_s
            calls.append((command, argument))
            if command == "backend_info":
                return {
                    "protocol": "sapolicy_manimux_v1",
                    "wire_action_dim": 16,
                    "horizon_steps": 16,
                    "observation_history": 3,
                    "sampling_modes": ["default", "paint"],
                }
            if command == "reset_model":
                return True
            if command == "infer_paint":
                return {
                    "actions": np.zeros((16, 16), dtype=np.float64),
                    "paint": {
                        "delay_steps": 3,
                        "num_steps": 10,
                        "model_evaluations": 30,
                        "inversion": "backward_euler",
                    },
                }
            raise AssertionError(command)

        def close(self):
            return None

    model._client = _Client()
    model.reset("session")
    now_ns = time.monotonic_ns()
    request = SAPolicyInferenceRequest(
        session_id="session",
        request_seq=2,
        observation_time_ns=now_ns,
        deadline_ns=now_ns + 1_000_000_000,
        observation=_snapshot(now_ns),
        instruction="put bottles in bin",
        sapolicy_observation={"image": np.zeros((2, 2, 3), dtype=np.uint8)},
        sapolicy_paint_action_prefix=np.zeros((3, 16), dtype=np.float64),
        sapolicy_paint_delay_steps=3,
    )

    result = model.infer(request)

    assert model.capabilities().sampling_modes == frozenset({"default", "paint"})
    assert isinstance(result, dict)
    assert result["actions"].shape == (16, 16)
    assert result["paint"]["model_evaluations"] == 30
    assert calls[-1][0] == "infer_paint"
    payload = calls[-1][1]
    assert payload["action_prefix"].shape == (3, 16)
