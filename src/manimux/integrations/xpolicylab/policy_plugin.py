"""ManiMux policy plugins backed by an XPolicyLab WebSocket policy server.

Same split as the MolmoAct/ABC/XR-1 plugins: the model plugin only speaks the
wire protocol, and every robot semantic -- group order, arm/gripper layout,
camera naming, chunk timing -- lives in the adapter.

What is different is that the server on the other end is not ours. XPolicyLab
serves one ``Model`` per policy out of its own environment, so a single plugin
pair reaches every policy in its zoo; which policy is running is decided when
that server is launched, not here.

Each request sends the observation and sampling parameters together through
XPolicyLab's ``INFER`` message. The server updates the model observation and
computes its action under one model lock, so observations cannot be interleaved
between separate RPCs.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping, Sequence

import numpy as np

from manimux.config import PolicyConfig, RobotConfig
from manimux.integrations.xpolicylab.obs_codec import (
    GroupLayout,
    build_layouts,
    decode_action_steps,
    encode_observation,
)
from manimux.integrations.xpolicylab.ws_client import XPolicyLabWsClient
from manimux.types import ActionChunk, ActionContext, InferenceRequest, ObservationSnapshot

DEFAULT_SERVER = "ws://127.0.0.1:8500"
DEFAULT_GROUP_ORDER = ("left_arm", "right_arm")
DEFAULT_GROUP_PREFIXES = {"left_arm": "left", "right_arm": "right"}
DEFAULT_CAMERA_MAP = {
    "cam_head": "front_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}
DEFAULT_GRIPPER_DOFS = 1
DEFAULT_ACTION_CODEC = "joint_position"
XR1_ACTION_CODEC = "xr1_ee_delta"


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


def _string_mapping(
    options: Mapping[str, object],
    name: str,
    default: Mapping[str, str],
) -> dict[str, str]:
    value = options.get(name, dict(default))
    if not isinstance(value, dict) or not value:
        raise ValueError(f"policy.options.{name} must be a non-empty mapping")
    if not all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()):
        raise ValueError(f"policy.options.{name} must map strings to strings")
    return dict(value)


def _positive_float_option(options: Mapping[str, object], name: str, default: float) -> float:
    value = options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"policy.options.{name} must be a number")
    if value <= 0:
        raise ValueError(f"policy.options.{name} must be positive")
    return float(value)


def _positive_int_option(options: Mapping[str, object], name: str, default: int) -> int:
    value = options.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"policy.options.{name} must be a non-negative integer")
    return value


def _layouts_from_options(
    options: Mapping[str, object],
    group_dims: Mapping[str, int],
) -> tuple[GroupLayout, ...]:
    return build_layouts(
        _string_sequence(options, "group_order", DEFAULT_GROUP_ORDER),
        _string_mapping(options, "group_prefixes", DEFAULT_GROUP_PREFIXES),
        group_dims,
        gripper_dofs=_positive_int_option(options, "gripper_dofs", DEFAULT_GRIPPER_DOFS),
    )


class XPolicyLabWsPolicyModel:
    """WebSocket inference backend; robot semantics stay in the adapter."""

    def __init__(self, config: PolicyConfig) -> None:
        options = config.options
        self._url = _string_option(options, "server", DEFAULT_SERVER)
        self._camera_map = _string_mapping(options, "camera_map", DEFAULT_CAMERA_MAP)
        # The observation carries the rate *we* run at. XPolicyLab treats the
        # environment as the owner of the control rate, so this is ours to
        # declare, and it comes from the same value the adapter uses for dt.
        self._frequency = 1.0 / config.effective_action_dt_s
        self._request_timeout_s = _positive_float_option(
            options, "request_timeout_s", config.timeout_s
        )
        self._connect_timeout_s = _positive_float_option(
            options, "connect_timeout_s", config.startup_timeout_s
        )
        # The model plugin never sees RobotConfig, so the arm/gripper split is
        # resolved from the first observation and then held fixed.
        self._group_order = _string_sequence(options, "group_order", DEFAULT_GROUP_ORDER)
        self._group_prefixes = _string_mapping(options, "group_prefixes", DEFAULT_GROUP_PREFIXES)
        self._gripper_dofs = _positive_int_option(options, "gripper_dofs", DEFAULT_GRIPPER_DOFS)
        self._action_codec = _string_option(options, "action_codec", DEFAULT_ACTION_CODEC)
        if self._action_codec not in {DEFAULT_ACTION_CODEC, XR1_ACTION_CODEC}:
            raise ValueError(
                "policy.options.action_codec must be 'joint_position' or 'xr1_ee_delta'"
            )
        self._kinematics = None
        if self._action_codec == XR1_ACTION_CODEC:
            from manimux.kinematics import build_kinematics

            kinematics_name = _string_option(options, "kinematics", "yam")
            kinematics_options = options.get("kinematics_options", {})
            if not isinstance(kinematics_options, dict):
                raise ValueError("policy.options.kinematics_options must be a mapping")
            self._kinematics = build_kinematics(kinematics_name, **kinematics_options)
        self._horizon_steps = config.horizon_steps
        self._layouts: tuple[GroupLayout, ...] = ()
        self._session_id: str | None = None
        self._client: XPolicyLabWsClient | None = None

    def reset(self, session_id: str) -> None:
        self.close()
        client = XPolicyLabWsClient(
            url=self._url,
            evaluation_id=session_id,
            trial_id=session_id,
            connect_timeout_s=self._connect_timeout_s,
            request_timeout_s=self._request_timeout_s,
        )
        client.connect()
        client.reset()
        self._client = client
        self._session_id = session_id

    def infer(self, request: InferenceRequest) -> object:
        if request.session_id != self._session_id:
            raise RuntimeError("XPolicyLab session is not initialized")
        client = self._client
        if client is None:
            raise RuntimeError("XPolicyLab client is not connected")

        snapshot = request.observation
        if not self._layouts:
            self._layouts = build_layouts(
                self._group_order,
                self._group_prefixes,
                {name: int(values.size) for name, values in snapshot.state.groups.items()},
                gripper_dofs=self._gripper_dofs,
            )
        observation = encode_observation(
            snapshot,
            layouts=self._layouts,
            camera_map=self._camera_map,
            instruction=request.instruction,
            frequency=self._frequency,
        )
        condition = getattr(request, "action_condition", None)
        weights = getattr(request, "condition_weights", None)
        if (condition is None) != (weights is None):
            raise ValueError("RTC action_condition and condition_weights must be provided together")
        if condition is None:
            raw = client.infer(observation, sampling={"mode": "default"})
            return self._response_with_anchor(raw, snapshot)

        beta = float(getattr(request, "rtc_beta", 5.0))
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError(f"rtc_beta must be finite and positive, got {beta}")
        condition_array = np.asarray(condition, dtype=np.float32)
        weights_array = np.asarray(weights, dtype=np.float32)
        expected_condition = (
            self._horizon_steps,
            sum(layout.dim for layout in self._layouts),
        )
        if condition_array.shape != expected_condition:
            raise ValueError(
                f"RTC action_condition must have shape {expected_condition}, "
                f"got {condition_array.shape}"
            )
        if weights_array.shape != (condition_array.shape[0],):
            raise ValueError(
                f"RTC condition_weights must have shape {(condition_array.shape[0],)}, "
                f"got {weights_array.shape}"
            )
        sampling_condition = condition_array
        if self._action_codec == XR1_ACTION_CODEC:
            from manimux.integrations.xr1_yam.policy_plugin import (
                joint_condition_to_xr1_actions,
            )

            assert self._kinematics is not None
            sampling_condition = joint_condition_to_xr1_actions(
                condition_array,
                snapshot.state.groups,
                group_order=self._group_order,
                kinematics=self._kinematics,
            )
        raw = client.infer(
            observation,
            sampling={
                "mode": "rtc",
                "action_condition": sampling_condition,
                "condition_weights": weights_array,
                "beta": beta,
            },
        )
        return self._response_with_anchor(raw, snapshot)

    def _response_with_anchor(self, raw: object, snapshot: ObservationSnapshot) -> object:
        if self._action_codec != XR1_ACTION_CODEC:
            return raw
        anchor = np.concatenate(
            [
                np.asarray(snapshot.state.groups[name], dtype=np.float64)
                for name in self._group_order
            ]
        )
        return {"actions": raw, "state": np.ascontiguousarray(anchor)}

    def close(self) -> None:
        client, self._client = self._client, None
        self._session_id = None
        if client is not None:
            client.close()


class XPolicyLabAdapter:
    """Translate canonical snapshots and XPolicyLab per-step action dictionaries."""

    def __init__(self, robot: RobotConfig, policy: PolicyConfig) -> None:
        self._layouts = _layouts_from_options(policy.options, robot.group_dims)
        self._camera_map = _string_mapping(policy.options, "camera_map", DEFAULT_CAMERA_MAP)
        self._required_cameras = tuple(self._camera_map.values())
        self._action_dt_ns = int(policy.effective_action_dt_s * 1_000_000_000)
        self._horizon_steps = policy.horizon_steps

    def build_observation(self, snapshot: ObservationSnapshot) -> ObservationSnapshot:
        missing = [name for name in self._required_cameras if name not in snapshot.frames]
        if missing:
            raise ValueError(f"XPolicyLab adapter is missing cameras: {missing}")
        return snapshot

    def decode_action(self, raw: object, context: ActionContext) -> ActionChunk:
        groups = decode_action_steps(raw, layouts=self._layouts)
        horizons = {values.shape[0] for values in groups.values()}
        if horizons != {self._horizon_steps}:
            raise ValueError(
                f"XPolicyLab action horizon must be {self._horizon_steps}, got {sorted(horizons)}"
            )
        return ActionChunk(
            plan_id=f"xpolicylab-{context.request_seq}-{uuid.uuid4().hex[:8]}",
            request_seq=context.request_seq,
            observation_time_ns=context.observation_time_ns,
            created_time_ns=context.created_time_ns,
            action_space="joint_position",
            dt_ns=self._action_dt_ns,
            groups=groups,
        )

    def validate(self, robot: RobotConfig, policy: PolicyConfig) -> None:
        del policy
        expected = tuple(layout.group for layout in self._layouts)
        if tuple(robot.group_dims) != expected:
            raise ValueError(
                "XPolicyLab requires robot groups in order "
                f"{list(expected)}, got {list(robot.group_dims)}"
            )
        for layout in self._layouts:
            if robot.group_dims[layout.group] != layout.dim:
                raise ValueError(
                    f"group {layout.group!r} is {robot.group_dims[layout.group]} values "
                    f"but the layout describes {layout.dim}"
                )


def build_model(config: PolicyConfig) -> XPolicyLabWsPolicyModel:
    return XPolicyLabWsPolicyModel(config)


def build_adapter(robot: RobotConfig, policy: PolicyConfig) -> XPolicyLabAdapter:
    return XPolicyLabAdapter(robot, policy)
