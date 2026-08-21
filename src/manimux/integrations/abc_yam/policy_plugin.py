"""ManiMux policy plugins for the ABC-DiT Bimanual YAM checkpoint.

Same split as the MolmoAct plugin: the model plugin only speaks the ``/act``
HTTP protocol, and every robot semantic (group order, dimensions, chunk timing)
lives in the adapter.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from manimux.config import PolicyConfig, RobotConfig
from manimux.policies.capabilities import PolicyCapabilities
from manimux.types import ActionChunk, ActionContext, InferenceRequest, ObservationSnapshot

DEFAULT_SERVER = "http://127.0.0.1:8300"
DEFAULT_GROUP_ORDER = ("left_arm", "right_arm")
DEFAULT_CAMERA_MAP = {
    "left_cam": "left_camera",
    "top_cam": "front_camera",
    "right_cam": "right_camera",
}


def _string_option(options: Mapping[str, object], name: str, default: str) -> str:
    value = options.get(name, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"policy.options.{name} must be a non-empty string")
    return value


def _string_sequence(
    options: Mapping[str, object],
    name: str,
    default: Sequence[str],
) -> tuple[str, ...]:
    value = options.get(name, list(default))
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"policy.options.{name} must be a non-empty list of strings")
    return tuple(value)


def _server_url(server: str) -> str:
    normalized = server.strip().rstrip("/")
    if "://" not in normalized:
        normalized = f"http://{normalized}"
    if not normalized.endswith("/act"):
        normalized += "/act"
    return normalized


class AbcHttpPolicyModel:
    """HTTP inference backend; robot semantics stay in ``AbcYamAdapter``."""

    def __init__(self, config: PolicyConfig) -> None:
        self._url = _server_url(_string_option(config.options, "server", DEFAULT_SERVER))
        self._health_url = self._url.removesuffix("/act") + "/healthz"
        self._group_order = _string_sequence(
            config.options,
            "group_order",
            DEFAULT_GROUP_ORDER,
        )
        self._camera_map = dict(DEFAULT_CAMERA_MAP)
        camera_map = config.options.get("camera_map")
        if camera_map is not None:
            if not isinstance(camera_map, dict) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in camera_map.items()
            ):
                raise ValueError("policy.options.camera_map must map strings to strings")
            self._camera_map = dict(camera_map)
        diffusion_steps = config.options.get("diffusion_steps")
        if diffusion_steps is not None and (
            not isinstance(diffusion_steps, int) or diffusion_steps <= 0
        ):
            raise ValueError("policy.options.diffusion_steps must be a positive integer")
        self._diffusion_steps = diffusion_steps
        self._timeout_s = float(config.options.get("http_timeout_s", config.timeout_s))
        if self._timeout_s <= 0:
            raise ValueError("policy.options.http_timeout_s must be positive")
        self._session_id: str | None = None

    def reset(self, session_id: str) -> None:
        import requests

        response = requests.get(self._health_url, timeout=self._timeout_s)
        if response.status_code != 200:
            raise RuntimeError(f"ABC health check failed with status {response.status_code}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("ABC health check returned invalid JSON") from exc
        if payload.get("status") != "ok":
            raise RuntimeError(f"ABC health check is not ready: {payload!r}")
        self._session_id = session_id

    def infer(self, request: InferenceRequest) -> object:
        if request.session_id != self._session_id:
            raise RuntimeError("ABC session is not initialized")
        snapshot = request.observation
        missing_groups = [name for name in self._group_order if name not in snapshot.state.groups]
        if missing_groups:
            raise ValueError(f"ABC observation is missing groups: {missing_groups}")
        missing_frames = [name for name in self._camera_map.values() if name not in snapshot.frames]
        if missing_frames:
            raise ValueError(f"ABC observation is missing cameras: {missing_frames}")

        payload: dict[str, Any] = {
            wire_name: snapshot.frames[source_name].data
            for wire_name, source_name in self._camera_map.items()
        }
        payload.update(
            {
                "timestamp": time.time(),
                "instruction": request.instruction,
                "state": np.concatenate(
                    [snapshot.state.groups[name] for name in self._group_order]
                ),
            }
        )
        if self._diffusion_steps is not None:
            payload["num_steps"] = self._diffusion_steps

        # An RTC runtime sends the inpainting condition on a request subclass;
        # the default runtime never sets it and the payload is unchanged.
        condition = getattr(request, "action_condition", None)
        weights = getattr(request, "condition_weights", None)
        if condition is not None and weights is not None:
            payload["action_condition"] = np.asarray(condition, dtype=np.float32)
            payload["action_condition_weights"] = np.asarray(weights, dtype=np.float32)
            payload["rtc_beta"] = float(getattr(request, "rtc_beta", 5.0))

        import json_numpy
        import requests

        remaining_s = max(0.001, (request.deadline_ns - time.monotonic_ns()) / 1e9)
        response = requests.post(
            self._url,
            headers={"Content-Type": "application/json"},
            data=json_numpy.dumps(payload),
            timeout=min(self._timeout_s, remaining_s),
        )
        if response.status_code != 200:
            raise RuntimeError(f"ABC server error {response.status_code}: {response.text}")
        decoded = json_numpy.loads(response.text)
        if not isinstance(decoded, dict) or "actions" not in decoded:
            raise ValueError("ABC response must contain actions")
        actions = np.asarray(decoded["actions"], dtype=np.float64)
        if actions.ndim != 2 or not actions.shape[0] or not np.isfinite(actions).all():
            raise ValueError("ABC actions must be a non-empty finite matrix")
        return np.ascontiguousarray(actions)

    def close(self) -> None:
        self._session_id = None

    def capabilities(self) -> PolicyCapabilities:
        return PolicyCapabilities(sampling_modes=frozenset({"default", "rtc"}))


class AbcYamAdapter:
    """Translate canonical YAM snapshots and raw ABC matrices."""

    def __init__(self, robot: RobotConfig, policy: PolicyConfig) -> None:
        self._group_order = _string_sequence(
            policy.options,
            "group_order",
            DEFAULT_GROUP_ORDER,
        )
        self._group_dims = dict(robot.group_dims)
        self._action_dt_ns = int(policy.effective_action_dt_s * 1_000_000_000)
        camera_map = policy.options.get("camera_map", DEFAULT_CAMERA_MAP)
        if not isinstance(camera_map, dict):
            raise ValueError("policy.options.camera_map must be a mapping")
        self._required_cameras = tuple(str(value) for value in camera_map.values())

    def build_observation(self, snapshot: ObservationSnapshot) -> ObservationSnapshot:
        missing = [name for name in self._required_cameras if name not in snapshot.frames]
        if missing:
            raise ValueError(f"ABC YAM adapter is missing cameras: {missing}")
        return snapshot

    def decode_action(self, raw: object, context: ActionContext) -> ActionChunk:
        actions = np.asarray(raw, dtype=np.float64)
        expected_dim = sum(self._group_dims[name] for name in self._group_order)
        if actions.ndim != 2 or actions.shape[1] != expected_dim:
            raise ValueError(
                f"ABC actions must have shape (horizon, {expected_dim}), got {actions.shape}"
            )
        if not actions.shape[0] or not np.isfinite(actions).all():
            raise ValueError("ABC actions must be non-empty and finite")
        groups: dict[str, np.ndarray] = {}
        start = 0
        for name in self._group_order:
            end = start + self._group_dims[name]
            groups[name] = np.ascontiguousarray(actions[:, start:end])
            start = end
        return ActionChunk(
            plan_id=f"abc-{context.request_seq}-{uuid.uuid4().hex[:8]}",
            request_seq=context.request_seq,
            observation_time_ns=context.observation_time_ns,
            created_time_ns=context.created_time_ns,
            action_space="joint_position",
            dt_ns=self._action_dt_ns,
            groups=groups,
        )

    def validate(self, robot: RobotConfig, policy: PolicyConfig) -> None:
        del policy
        if tuple(robot.group_dims) != self._group_order:
            raise ValueError(
                "ABC YAM requires robot groups in order "
                f"{list(self._group_order)}, got {list(robot.group_dims)}"
            )
        if any(robot.group_dims[name] != 7 for name in self._group_order):
            raise ValueError("ABC YAM requires two 7-value arm+gripper groups")


def build_model(config: PolicyConfig) -> AbcHttpPolicyModel:
    return AbcHttpPolicyModel(config)


def build_adapter(robot: RobotConfig, policy: PolicyConfig) -> AbcYamAdapter:
    return AbcYamAdapter(robot, policy)
