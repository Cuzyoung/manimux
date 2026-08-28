"""SAPolicy service and YAM embodiment plugins.

The policy worker only transports already-encoded observations to SAPolicy's
private peer process. The adapter owns all embodiment knowledge: camera aliases,
camera intrinsics, YAM FK/IK, gripper units, and conversion of SAPolicy's absolute
end-effector waypoints into ManiMux's canonical joint-position ``ActionChunk``.

SAPolicy's current RoboTwin-compatible service represents its grasp-site pose as
an ``endpose`` located 12 cm behind that site. ``server_tcp_forward_offset_m``
makes that wire convention explicit on both encode and decode; it is not a YAM
tool-frame correction.
"""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from manimux.config import PolicyConfig, RobotConfig
from manimux.integrations.sapolicy_yam.tcp_client import SAPolicyTcpClient
from manimux.policies.capabilities import PolicyCapabilities
from manimux.types import (
    ActionChunk,
    ActionContext,
    FloatArray,
    InferenceRequest,
    ObservationSnapshot,
)

DEFAULT_GROUP_ORDER = ("left_arm", "right_arm")
ARM_JOINTS = 6
GROUP_DIM = 7
WIRE_ACTION_DIM = 16


def _string_option(options: Mapping[str, object], name: str, default: str) -> str:
    value = options.get(name, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"policy.options.{name} must be a non-empty string")
    return value


def _string_sequence(
    options: Mapping[str, object], name: str, default: Sequence[str]
) -> tuple[str, ...]:
    value = options.get(name, list(default))
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"policy.options.{name} must be a non-empty list of strings")
    return tuple(value)


def _mapping_option(options: Mapping[str, object], name: str) -> dict[str, object]:
    value = options.get(name)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"policy.options.{name} must be a non-empty mapping")
    if not all(isinstance(key, str) and key for key in value):
        raise ValueError(f"policy.options.{name} keys must be non-empty strings")
    return dict(value)


@dataclass(slots=True)
class SAPolicyInferenceRequest(InferenceRequest):
    """Canonical request plus the adapter-produced private-service payload."""

    sapolicy_observation: dict[str, object] = field(default_factory=dict)
    sapolicy_paint_action_prefix: np.ndarray | None = None
    sapolicy_paint_delay_steps: int | None = None


@dataclass(frozen=True, slots=True)
class GripperTransform:
    """Affine map from YAM normalized position to the checkpoint's gripper unit."""

    scale: float
    offset: float
    robot_min: float
    robot_max: float
    clip_output: bool

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.scale, self.offset, self.robot_min, self.robot_max], dtype=np.float64
        )
        if not np.isfinite(values).all() or self.scale == 0:
            raise ValueError("SAPolicy gripper affine values must be finite and scale non-zero")
        if self.robot_min >= self.robot_max:
            raise ValueError("SAPolicy gripper robot_min must be less than robot_max")

    def to_policy(self, robot_value: float) -> float:
        value = float(robot_value)
        if not self.robot_min <= value <= self.robot_max:
            raise ValueError(
                f"YAM gripper state {value:.6f} is outside "
                f"[{self.robot_min:.6f}, {self.robot_max:.6f}]"
            )
        return self.scale * value + self.offset

    def to_robot(self, policy_value: float) -> float:
        value = (float(policy_value) - self.offset) / self.scale
        if self.clip_output:
            return float(np.clip(value, self.robot_min, self.robot_max))
        if not self.robot_min <= value <= self.robot_max:
            raise ValueError(
                f"SAPolicy gripper output maps to {value:.6f}, outside "
                f"[{self.robot_min:.6f}, {self.robot_max:.6f}]"
            )
        return value


def _parse_gripper_transforms(
    options: Mapping[str, object], group_order: Sequence[str]
) -> dict[str, GripperTransform]:
    raw = _mapping_option(options, "gripper_transforms")
    if set(raw) != set(group_order):
        raise ValueError(
            "policy.options.gripper_transforms must contain exactly "
            f"{list(group_order)}"
        )
    transforms: dict[str, GripperTransform] = {}
    for group in group_order:
        value = raw[group]
        if not isinstance(value, dict):
            raise ValueError(f"gripper transform for {group!r} must be a mapping")
        required = {"scale", "offset", "robot_min", "robot_max", "clip_output"}
        if set(value) != required:
            raise ValueError(
                f"gripper transform for {group!r} must contain exactly {sorted(required)}"
            )
        if not isinstance(value["clip_output"], bool):
            raise ValueError(f"gripper transform clip_output for {group!r} must be boolean")
        transforms[group] = GripperTransform(
            scale=float(value["scale"]),
            offset=float(value["offset"]),
            robot_min=float(value["robot_min"]),
            robot_max=float(value["robot_max"]),
            clip_output=value["clip_output"],
        )
    return transforms


def _parse_camera_map(options: Mapping[str, object]) -> dict[str, str]:
    raw = _mapping_option(options, "camera_map")
    if not all(isinstance(value, str) and value for value in raw.values()):
        raise ValueError("policy.options.camera_map must map model names to sensor names")
    result = {key: str(value) for key, value in raw.items()}
    if len(set(result.values())) != len(result):
        raise ValueError("policy.options.camera_map sensor names must be unique")
    return result


def _parse_intrinsics(
    options: Mapping[str, object], camera_names: Sequence[str]
) -> dict[str, np.ndarray]:
    raw = _mapping_option(options, "camera_intrinsics")
    if set(raw) != set(camera_names):
        raise ValueError(
            "policy.options.camera_intrinsics must contain exactly the model-facing "
            f"camera names {list(camera_names)}"
        )
    result: dict[str, np.ndarray] = {}
    for name in camera_names:
        matrix = np.asarray(raw[name], dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError(f"camera intrinsics for {name!r} must be a finite 3x3 matrix")
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.isclose(matrix[2, 2], 1.0):
            raise ValueError(f"camera intrinsics for {name!r} are not a valid pinhole matrix")
        result[name] = np.ascontiguousarray(matrix)
    return result


def _parse_model_frame_transforms(
    options: Mapping[str, object], group_order: Sequence[str]
) -> dict[str, np.ndarray]:
    """Parse transforms from each IK base frame into the checkpoint state frame.

    The ABC bottles checkpoint was trained from the two-arm station MJCF, whose
    state positions are expressed in the station/world frame.  ManiMux solves IK
    with one standalone arm model per side.  Relative body-frame actions are
    invariant to this fixed transform, but the checkpoint's *state conditioning*
    is not, so the transform must be applied before observations reach SAPolicy
    and inverted again before local-arm IK.
    """
    raw = options.get("model_from_kinematics")
    if raw is None:
        return {group: np.eye(4, dtype=np.float64) for group in group_order}
    if not isinstance(raw, dict) or set(raw) != set(group_order):
        raise ValueError(
            "policy.options.model_from_kinematics must contain exactly "
            f"{list(group_order)}"
        )
    result: dict[str, np.ndarray] = {}
    for group in group_order:
        transform = np.asarray(raw[group], dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError(
                f"model_from_kinematics for {group!r} must be a finite 4x4 matrix"
            )
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise ValueError(
                f"model_from_kinematics for {group!r} must have homogeneous last row"
            )
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-6
        ):
            raise ValueError(
                f"model_from_kinematics for {group!r} must contain an SO(3) rotation"
            )
        result[group] = np.ascontiguousarray(transform)
    return result


def _pose_to_wire_endpose(pose: np.ndarray, offset_m: float) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"SAPolicy FK pose must be a finite 4x4 matrix, got {pose.shape}")
    rotation = pose[:3, :3]
    position = pose[:3, 3] - rotation @ np.array([offset_m, 0.0, 0.0])
    quaternion_xyzw = Rotation.from_matrix(rotation).as_quat()
    quaternion_wxyz = quaternion_xyzw[[3, 0, 1, 2]]
    return np.concatenate([position, quaternion_wxyz]).astype(np.float64)


def _wire_endpose_to_pose(endpose: np.ndarray, offset_m: float) -> np.ndarray:
    values = np.asarray(endpose, dtype=np.float64).reshape(-1)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError(f"SAPolicy endpose must have 7 finite values, got {values.shape}")
    quaternion_xyzw = values[[4, 5, 6, 3]]
    rotation = Rotation.from_quat(quaternion_xyzw).as_matrix()
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = values[:3] + rotation @ np.array([offset_m, 0.0, 0.0])
    return pose


class SAPolicyTcpPolicyModel:
    """Protocol-only model plugin for an already-running private SAPolicy service."""

    def __init__(self, config: PolicyConfig) -> None:
        host = _string_option(config.options, "host", "127.0.0.1")
        port = config.options.get("port", 9977)
        if not isinstance(port, int):
            raise ValueError("policy.options.port must be an integer")
        connect_timeout_s = float(config.options.get("connect_timeout_s", config.timeout_s))
        self._request_timeout_s = float(
            config.options.get("request_timeout_s", config.timeout_s)
        )
        self._expected_horizon = config.horizon_steps
        if self._request_timeout_s <= 0:
            raise ValueError("policy.options.request_timeout_s must be positive")
        self._client = SAPolicyTcpClient(
            host, port, connect_timeout_s=connect_timeout_s
        )
        self._session_id: str | None = None
        self._backend_metadata: dict[str, object] = {}
        self._sampling_modes = frozenset({"default"})

    def reset(self, session_id: str) -> None:
        info = self._client.call("backend_info", timeout_s=self._request_timeout_s)
        if not isinstance(info, dict):
            raise ValueError("SAPolicy backend_info response must be a mapping")
        if info.get("protocol") != "sapolicy_manimux_v1":
            raise RuntimeError(f"unexpected SAPolicy service protocol: {info!r}")
        if info.get("wire_action_dim") != WIRE_ACTION_DIM:
            raise RuntimeError(f"unexpected SAPolicy wire action dimension: {info!r}")
        if info.get("horizon_steps") != self._expected_horizon:
            raise RuntimeError(
                "SAPolicy service horizon does not match ManiMux policy.horizon_steps: "
                f"server={info.get('horizon_steps')} config={self._expected_horizon}"
            )
        raw_modes = info.get("sampling_modes", ["default"])
        if (
            not isinstance(raw_modes, list)
            or not raw_modes
            or not all(isinstance(mode, str) and mode for mode in raw_modes)
        ):
            raise RuntimeError(f"invalid SAPolicy sampling_modes: {raw_modes!r}")
        self._sampling_modes = frozenset(raw_modes)
        if "default" not in self._sampling_modes:
            raise RuntimeError("SAPolicy service must advertise default sampling")
        self._backend_metadata = {
            "server": "sapolicy_manimux_v1",
            "model": {
                "wire_action_dim": info["wire_action_dim"],
                "horizon_steps": info["horizon_steps"],
                "observation_history": info.get("observation_history"),
                "sampling_modes": sorted(self._sampling_modes),
            },
        }
        self._client.call("reset_model", timeout_s=self._request_timeout_s)
        self._session_id = session_id

    def infer(self, request: InferenceRequest) -> object:
        if request.session_id != self._session_id:
            raise RuntimeError("SAPolicy session is not initialized")
        if not isinstance(request, SAPolicyInferenceRequest):
            raise TypeError("SAPolicy worker requires an adapter-prepared request")

        timeout_s = min(
            self._request_timeout_s,
            max(0.0, (request.deadline_ns - time.monotonic_ns()) / 1e9),
        )
        paint_prefix = request.sapolicy_paint_action_prefix
        paint_delay = request.sapolicy_paint_delay_steps
        if (paint_prefix is None) != (paint_delay is None):
            raise ValueError("SAPolicy PAINT prefix and delay must be provided together")
        if paint_prefix is None:
            result = self._client.call(
                "infer", request.sapolicy_observation, timeout_s=timeout_s
            )
            raw_actions = result
        else:
            result = self._client.call(
                "infer_paint",
                {
                    "observation": request.sapolicy_observation,
                    "action_prefix": paint_prefix,
                    "delay_steps": paint_delay,
                },
                timeout_s=timeout_s,
            )
            if not isinstance(result, Mapping):
                raise ValueError("SAPolicy PAINT response must be a mapping")
            raw_actions = result.get("actions")

        actions = np.asarray(raw_actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != WIRE_ACTION_DIM or not actions.shape[0]:
            raise ValueError(
                "SAPolicy response must have shape (horizon, 16), "
                f"got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("SAPolicy response contains non-finite values")
        actions = np.ascontiguousarray(actions)
        if paint_prefix is None:
            return actions
        metadata = result.get("paint")
        if not isinstance(metadata, Mapping):
            raise ValueError("SAPolicy PAINT response is missing metadata")
        return {"actions": actions, "paint": dict(metadata)}

    def close(self) -> None:
        self._session_id = None
        self._backend_metadata = {}
        self._sampling_modes = frozenset({"default"})
        self._client.close()

    def capabilities(self) -> PolicyCapabilities:
        return PolicyCapabilities(
            sampling_modes=self._sampling_modes,
            backend_metadata=dict(self._backend_metadata),
        )


class SAPolicyYamAdapter:
    """Encode YAM observations and solve SAPolicy Cartesian chunks back to joints."""

    def __init__(self, robot: RobotConfig, policy: PolicyConfig) -> None:
        from manimux.kinematics import build_kinematics

        self._group_order = _string_sequence(
            policy.options, "group_order", DEFAULT_GROUP_ORDER
        )
        self._group_dims = dict(robot.group_dims)
        self._horizon_steps = policy.horizon_steps
        self._action_dt_ns = int(policy.effective_action_dt_s * 1_000_000_000)
        self._camera_map = _parse_camera_map(policy.options)
        self._intrinsics = _parse_intrinsics(
            policy.options, tuple(self._camera_map)
        )
        self._grippers = _parse_gripper_transforms(
            policy.options, self._group_order
        )
        self._model_from_kinematics = _parse_model_frame_transforms(
            policy.options, self._group_order
        )
        self._kinematics_from_model = {
            group: np.linalg.inv(transform)
            for group, transform in self._model_from_kinematics.items()
        }
        self._server_offset_m = float(
            policy.options.get("server_tcp_forward_offset_m", 0.12)
        )
        if self._server_offset_m < 0 or not np.isfinite(self._server_offset_m):
            raise ValueError("policy.options.server_tcp_forward_offset_m must be finite and >= 0")
        requires_depth = policy.options.get("requires_depth", False)
        if not isinstance(requires_depth, bool):
            raise ValueError("policy.options.requires_depth must be boolean")
        if requires_depth:
            raise ValueError(
                "SAPolicy depth checkpoints are not supported by the current ManiMux SensorFrame; "
                "use an RGB checkpoint or add a typed metric-depth sensor contract first"
            )

        kinematics_name = _string_option(policy.options, "kinematics", "yam")
        kinematics_options = policy.options.get("kinematics_options", {})
        if not isinstance(kinematics_options, dict):
            raise ValueError("policy.options.kinematics_options must be a mapping")
        self._kinematics = build_kinematics(kinematics_name, **kinematics_options)
        self._anchors: OrderedDict[int, np.ndarray] = OrderedDict()
        if self._kinematics.num_arm_joints != ARM_JOINTS:
            raise ValueError(
                f"SAPolicy YAM requires {ARM_JOINTS} arm joints, kinematics reports "
                f"{self._kinematics.num_arm_joints}"
            )
        raw_minimum = policy.options.get(
            "min_valid_horizon_steps", min(4, self._horizon_steps)
        )
        if isinstance(raw_minimum, bool) or not isinstance(raw_minimum, int):
            raise ValueError("policy.options.min_valid_horizon_steps must be an integer")
        self._min_valid_horizon_steps = int(raw_minimum)
        if not 1 <= self._min_valid_horizon_steps <= self._horizon_steps:
            raise ValueError(
                "policy.options.min_valid_horizon_steps must satisfy "
                f"1 <= value <= {self._horizon_steps}"
            )
        self._joint_limit_margin_rad = float(
            policy.options.get("joint_limit_margin_rad", 0.0)
        )
        if (
            not np.isfinite(self._joint_limit_margin_rad)
            or self._joint_limit_margin_rad < 0
        ):
            raise ValueError(
                "policy.options.joint_limit_margin_rad must be finite and >= 0"
            )
        if self._joint_limit_margin_rad > 0 and not callable(
            getattr(self._kinematics, "joint_limit_margins", None)
        ):
            raise ValueError(
                "configured SAPolicy joint_limit_margin_rad requires kinematics "
                "joint-limit diagnostics"
            )

    def build_observation(self, snapshot: ObservationSnapshot) -> ObservationSnapshot:
        missing = [name for name in self._camera_map.values() if name not in snapshot.frames]
        if missing:
            raise ValueError(f"SAPolicy YAM adapter is missing cameras: {missing}")
        return snapshot

    def _joint_prefix_to_wire(self, prefix: np.ndarray) -> np.ndarray:
        """Convert packed YAM joint waypoints into SAPolicy absolute EE wire actions."""
        rows = np.asarray(prefix, dtype=np.float64)
        packed_dim = sum(self._group_dims[name] for name in self._group_order)
        if rows.ndim != 2 or rows.shape[1] != packed_dim or not len(rows):
            raise ValueError(
                f"SAPolicy PAINT joint prefix must have shape [d, {packed_dim}], "
                f"got {rows.shape}"
            )
        if not np.isfinite(rows).all():
            raise ValueError("SAPolicy PAINT joint prefix contains non-finite values")

        wire = np.empty((len(rows), WIRE_ACTION_DIM), dtype=np.float64)
        for step, row in enumerate(rows):
            packed_offset = 0
            wire_offset = 0
            for group in self._group_order:
                state = row[packed_offset : packed_offset + GROUP_DIM]
                local_pose = self._kinematics.fk(
                    state[:ARM_JOINTS], float(state[-1])
                )
                model_pose = self._model_from_kinematics[group] @ local_pose
                wire[step, wire_offset : wire_offset + 7] = _pose_to_wire_endpose(
                    model_pose, self._server_offset_m
                )
                wire[step, wire_offset + 7] = self._grippers[group].to_policy(
                    float(state[-1])
                )
                packed_offset += GROUP_DIM
                wire_offset += 8
        return np.ascontiguousarray(wire)

    def prepare_request(self, request: InferenceRequest) -> InferenceRequest:
        snapshot = request.observation
        anchor = np.concatenate(
            [
                np.asarray(snapshot.state.groups[name], dtype=np.float64)
                for name in self._group_order
            ]
        )
        self._anchors[request.request_seq] = np.ascontiguousarray(anchor)
        while len(self._anchors) > 8:
            self._anchors.popitem(last=False)

        payload: dict[str, object] = {}
        for side, group in zip(("left", "right"), self._group_order, strict=True):
            state = np.asarray(snapshot.state.groups[group], dtype=np.float64)
            local_pose = self._kinematics.fk(state[:ARM_JOINTS], float(state[-1]))
            pose = self._model_from_kinematics[group] @ local_pose
            payload[f"{side}_endpose"] = _pose_to_wire_endpose(
                pose, self._server_offset_m
            )
            payload[f"{side}_gripper"] = self._grippers[group].to_policy(float(state[-1]))

        model_cameras = tuple(self._camera_map)
        images = {
            model_name: np.ascontiguousarray(snapshot.frames[source_name].data)
            for model_name, source_name in self._camera_map.items()
        }
        first = model_cameras[0]
        payload["image"] = images[first]
        payload["intrinsic_cv"] = self._intrinsics[first]
        if len(model_cameras) > 1:
            payload["camera_names"] = list(model_cameras)
            payload["images"] = images
            payload["intrinsics"] = dict(self._intrinsics)

        paint_prefix = getattr(request, "paint_action_prefix", None)
        paint_delay = getattr(request, "paint_delay_steps", None)
        if (paint_prefix is None) != (paint_delay is None):
            raise ValueError("SAPolicy PAINT action prefix and delay must be provided together")
        wire_prefix = None
        if paint_prefix is not None:
            delay = int(paint_delay)
            packed_prefix = np.asarray(paint_prefix, dtype=np.float64)
            packed_dim = sum(self._group_dims[name] for name in self._group_order)
            if delay <= 0 or packed_prefix.shape != (delay, packed_dim):
                raise ValueError(
                    f"SAPolicy PAINT prefix must have shape ({delay}, {packed_dim}), "
                    f"got {packed_prefix.shape}"
                )
            wire_prefix = self._joint_prefix_to_wire(packed_prefix)

        return SAPolicyInferenceRequest(
            session_id=request.session_id,
            request_seq=request.request_seq,
            observation_time_ns=request.observation_time_ns,
            deadline_ns=request.deadline_ns,
            observation=request.observation,
            instruction=request.instruction,
            sapolicy_observation=payload,
            sapolicy_paint_action_prefix=wire_prefix,
            sapolicy_paint_delay_steps=None if paint_delay is None else int(paint_delay),
        )

    def decode_action(self, raw: object, context: ActionContext) -> ActionChunk:
        raw_actions = raw.get("actions") if isinstance(raw, Mapping) else raw
        actions = np.asarray(raw_actions, dtype=np.float64)
        if actions.shape != (self._horizon_steps, WIRE_ACTION_DIM):
            raise ValueError(
                "SAPolicy actions must have shape "
                f"({self._horizon_steps}, {WIRE_ACTION_DIM}), got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("SAPolicy actions contain non-finite values")
        anchor = self._anchors.pop(context.request_seq, None)
        if anchor is None:
            raise ValueError(
                f"SAPolicy adapter has no observation anchor for request {context.request_seq}"
            )

        source_offset = 0
        if context.execution_time_ns is not None:
            age_ns = max(0, context.execution_time_ns - context.observation_time_ns)
            source_offset = int(age_ns // self._action_dt_ns)
        if source_offset >= self._horizon_steps:
            raise ValueError(
                "SAPolicy response has no executable source waypoints: "
                f"source_offset={source_offset}, horizon={self._horizon_steps}"
            )

        seeds: dict[str, FloatArray] = {}
        for arm_index, group in enumerate(self._group_order):
            state_start = arm_index * GROUP_DIM
            seed_state = anchor[state_start : state_start + GROUP_DIM]
            if context.measured_state is not None:
                measured = context.measured_state.groups.get(group)
                if measured is None:
                    raise ValueError(
                        f"SAPolicy measured state is missing group {group!r}"
                    )
                seed_state = np.asarray(measured, dtype=np.float64)
                if seed_state.shape != (GROUP_DIM,) or not np.isfinite(seed_state).all():
                    raise ValueError(
                        f"SAPolicy measured state group {group!r} is invalid"
                    )
            seeds[group] = np.asarray(
                seed_state[:ARM_JOINTS], dtype=np.float64
            ).copy()

        solved_rows: dict[str, list[FloatArray]] = {
            group: [] for group in self._group_order
        }
        truncation_reason: str | None = None
        for source_step in range(source_offset, self._horizon_steps):
            decoded_step = source_step - source_offset
            step_rows: dict[str, FloatArray] = {}
            for arm_index, group in enumerate(self._group_order):
                action_start = arm_index * 8
                row = actions[source_step, action_start : action_start + 8]
                try:
                    solved, failure = self._solve_waypoint(
                        group=group,
                        row=row,
                        seed=seeds[group],
                        source_step=source_step,
                        decoded_step=decoded_step,
                    )
                except ValueError as exc:
                    solved = None
                    failure = (
                        f"SAPolicy {group} invalid waypoint at source step "
                        f"{source_step} (decoded step {decoded_step}): {exc}"
                    )
                if failure is not None:
                    truncation_reason = failure
                    break
                assert solved is not None
                step_rows[group] = solved
            if truncation_reason is not None:
                break
            for group, solved in step_rows.items():
                solved_rows[group].append(solved)
                seeds[group] = solved[:ARM_JOINTS].copy()

        valid_steps = len(solved_rows[self._group_order[0]])
        if valid_steps < self._min_valid_horizon_steps:
            reason = truncation_reason or "source horizon is too short after time trimming"
            raise ValueError(
                f"{reason}; common_valid_prefix_steps={valid_steps}, "
                f"required={self._min_valid_horizon_steps}, "
                f"source_offset={source_offset}"
            )
        groups = {
            group: np.ascontiguousarray(np.stack(rows, axis=0))
            for group, rows in solved_rows.items()
        }

        metadata: dict[str, object] = {
            "adapter_source_offset_steps": source_offset,
            "adapter_ik_valid_steps": valid_steps,
            "adapter_ik_truncated": truncation_reason is not None,
        }
        if truncation_reason is not None:
            metadata["adapter_ik_truncation_reason"] = truncation_reason

        return ActionChunk(
            plan_id=f"sapolicy-{context.request_seq}-{uuid.uuid4().hex[:8]}",
            request_seq=context.request_seq,
            observation_time_ns=context.observation_time_ns,
            created_time_ns=context.created_time_ns,
            action_space="joint_position",
            dt_ns=self._action_dt_ns,
            groups=groups,
            source_offset_steps=source_offset,
            metadata=metadata,
        )

    def _solve_waypoint(
        self,
        *,
        group: str,
        row: FloatArray,
        seed: FloatArray,
        source_step: int,
        decoded_step: int,
    ) -> tuple[FloatArray | None, str | None]:
        gripper = self._grippers[group].to_robot(float(row[7]))
        model_target = _wire_endpose_to_pose(row[:7], self._server_offset_m)
        target = self._kinematics_from_model[group] @ model_target
        converged, raw_solved = self._kinematics.ik(target, seed, float(gripper))
        solved = np.asarray(raw_solved, dtype=np.float64)
        if solved.shape != (ARM_JOINTS,) or not np.isfinite(solved).all():
            return None, (
                f"SAPolicy {group} IK returned invalid joints at source step "
                f"{source_step} (decoded step {decoded_step})"
            )

        diagnostics = self._ik_diagnostics(target, solved, float(gripper))
        if not converged:
            return None, (
                f"SAPolicy {group} IK did not converge at source step {source_step} "
                f"(decoded step {decoded_step}); {diagnostics}"
            )

        margins_method = getattr(self._kinematics, "joint_limit_margins", None)
        if callable(margins_method):
            margins = np.asarray(margins_method(solved), dtype=np.float64).reshape(-1)
            if margins.shape != (ARM_JOINTS,) or not np.isfinite(margins).all():
                raise ValueError("kinematics returned invalid joint-limit margins")
            limiting_joint = int(np.argmin(margins))
            minimum_margin = float(margins[limiting_joint])
            if minimum_margin < self._joint_limit_margin_rad:
                return None, (
                    f"SAPolicy {group} joint J{limiting_joint + 1} margin "
                    f"{minimum_margin:.6f} rad is below "
                    f"{self._joint_limit_margin_rad:.6f} rad at source step "
                    f"{source_step} (decoded step {decoded_step}); {diagnostics}"
                )

        packed = np.empty(GROUP_DIM, dtype=np.float64)
        packed[:ARM_JOINTS] = solved
        packed[ARM_JOINTS] = float(gripper)
        return packed, None

    def _ik_diagnostics(
        self, target: FloatArray, solved: FloatArray, gripper: float
    ) -> str:
        details = [
            "target_xyz_m="
            + np.array2string(
                target[:3, 3], precision=5, separator=",", suppress_small=False
            )
        ]
        pose_error_method = getattr(self._kinematics, "pose_error", None)
        if callable(pose_error_method):
            try:
                position_error, orientation_error = pose_error_method(
                    target, solved, gripper
                )
                details.append(f"position_error_m={float(position_error):.6f}")
                details.append(
                    f"orientation_error_rad={float(orientation_error):.6f}"
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                details.append(f"pose_diagnostic_error={type(exc).__name__}")
        margins_method = getattr(self._kinematics, "joint_limit_margins", None)
        if callable(margins_method):
            try:
                margins = np.asarray(margins_method(solved), dtype=np.float64).reshape(-1)
                if margins.shape == (ARM_JOINTS,) and np.isfinite(margins).all():
                    limiting_joint = int(np.argmin(margins))
                    details.append(f"limiting_joint=J{limiting_joint + 1}")
                    details.append(
                        f"joint_limit_margin_rad={float(margins[limiting_joint]):.6f}"
                    )
            except (RuntimeError, TypeError, ValueError) as exc:
                details.append(f"limit_diagnostic_error={type(exc).__name__}")
        details.append(
            "solved_joints_rad="
            + np.array2string(
                solved, precision=5, separator=",", suppress_small=False
            )
        )
        return "; ".join(details)

    def validate(self, robot: RobotConfig, policy: PolicyConfig) -> None:
        del policy
        if tuple(robot.group_dims) != self._group_order:
            raise ValueError(
                "SAPolicy YAM requires robot groups in order "
                f"{list(self._group_order)}, got {list(robot.group_dims)}"
            )
        if any(robot.group_dims[name] != GROUP_DIM for name in self._group_order):
            raise ValueError("SAPolicy YAM requires two 7-value arm+gripper groups")


def build_model(config: PolicyConfig) -> SAPolicyTcpPolicyModel:
    return SAPolicyTcpPolicyModel(config)


def build_adapter(robot: RobotConfig, policy: PolicyConfig) -> SAPolicyYamAdapter:
    return SAPolicyYamAdapter(robot, policy)
