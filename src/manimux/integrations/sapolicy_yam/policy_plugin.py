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
        self._backend_metadata = {
            "server": "sapolicy_manimux_v1",
            "model": {
                "wire_action_dim": info["wire_action_dim"],
                "horizon_steps": info["horizon_steps"],
                "observation_history": info.get("observation_history"),
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
        result = self._client.call(
            "infer", request.sapolicy_observation, timeout_s=timeout_s
        )
        actions = np.asarray(result, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != WIRE_ACTION_DIM or not actions.shape[0]:
            raise ValueError(
                "SAPolicy response must have shape (horizon, 16), "
                f"got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("SAPolicy response contains non-finite values")
        return np.ascontiguousarray(actions)

    def close(self) -> None:
        self._session_id = None
        self._backend_metadata = {}
        self._client.close()

    def capabilities(self) -> PolicyCapabilities:
        # The current private service exposes one forward path and no native
        # inpainting/multi-sample hook. ManiMux must reject RTC/AAC/PAINT early.
        return PolicyCapabilities(
            sampling_modes=frozenset({"default"}),
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
        self._server_offset_m = float(
            policy.options.get("server_tcp_forward_offset_m", 0.12)
        )
        self._max_position_delta_m = float(
            policy.options.get("max_position_delta_m", 0.0)
        )
        self._max_rotation_delta_rad = float(
            policy.options.get("max_rotation_delta_rad", 0.0)
        )
        if self._server_offset_m < 0 or not np.isfinite(self._server_offset_m):
            raise ValueError("policy.options.server_tcp_forward_offset_m must be finite and >= 0")
        if not np.isfinite(self._max_position_delta_m) or self._max_position_delta_m <= 0:
            raise ValueError("policy.options.max_position_delta_m must be positive")
        if not np.isfinite(self._max_rotation_delta_rad) or self._max_rotation_delta_rad <= 0:
            raise ValueError("policy.options.max_rotation_delta_rad must be positive")
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

    def build_observation(self, snapshot: ObservationSnapshot) -> ObservationSnapshot:
        missing = [name for name in self._camera_map.values() if name not in snapshot.frames]
        if missing:
            raise ValueError(f"SAPolicy YAM adapter is missing cameras: {missing}")
        return snapshot

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
            pose = self._kinematics.fk(state[:ARM_JOINTS], float(state[-1]))
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

        return SAPolicyInferenceRequest(
            session_id=request.session_id,
            request_seq=request.request_seq,
            observation_time_ns=request.observation_time_ns,
            deadline_ns=request.deadline_ns,
            observation=request.observation,
            instruction=request.instruction,
            sapolicy_observation=payload,
        )

    def decode_action(self, raw: object, context: ActionContext) -> ActionChunk:
        actions = np.asarray(raw, dtype=np.float64)
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

        groups: dict[str, np.ndarray] = {}
        for arm_index, group in enumerate(self._group_order):
            state_start = arm_index * GROUP_DIM
            state = anchor[state_start : state_start + GROUP_DIM]
            action_start = arm_index * 8
            arm_actions = actions[:, action_start : action_start + 8]
            groups[group] = self._solve_arm(group, arm_actions, state)

        return ActionChunk(
            plan_id=f"sapolicy-{context.request_seq}-{uuid.uuid4().hex[:8]}",
            request_seq=context.request_seq,
            observation_time_ns=context.observation_time_ns,
            created_time_ns=context.created_time_ns,
            action_space="joint_position",
            dt_ns=self._action_dt_ns,
            groups=groups,
        )

    def _solve_arm(
        self, group: str, actions: np.ndarray, anchor_state: np.ndarray
    ) -> np.ndarray:
        anchor_joints = anchor_state[:ARM_JOINTS]
        anchor_gripper = float(anchor_state[-1])
        anchor_pose = self._kinematics.fk(anchor_joints, anchor_gripper)
        seed = anchor_joints.copy()
        out = np.empty((actions.shape[0], GROUP_DIM), dtype=np.float64)

        for step, row in enumerate(actions):
            target = _wire_endpose_to_pose(row[:7], self._server_offset_m)
            position_delta = float(np.linalg.norm(target[:3, 3] - anchor_pose[:3, 3]))
            rotation_delta = float(
                Rotation.from_matrix(anchor_pose[:3, :3].T @ target[:3, :3]).magnitude()
            )
            if position_delta > self._max_position_delta_m:
                raise ValueError(
                    f"SAPolicy {group} step {step} position delta {position_delta:.4f} m "
                    f"exceeds {self._max_position_delta_m:.4f} m"
                )
            if rotation_delta > self._max_rotation_delta_rad:
                raise ValueError(
                    f"SAPolicy {group} step {step} rotation delta {rotation_delta:.4f} rad "
                    f"exceeds {self._max_rotation_delta_rad:.4f} rad"
                )
            gripper = self._grippers[group].to_robot(float(row[7]))
            converged, solved = self._kinematics.ik(target, seed, gripper)
            if not converged:
                raise ValueError(f"SAPolicy {group} IK did not converge at chunk step {step}")
            seed = np.asarray(solved, dtype=np.float64)
            if seed.shape != (ARM_JOINTS,) or not np.isfinite(seed).all():
                raise ValueError(f"SAPolicy {group} IK returned invalid joints at step {step}")
            out[step, :ARM_JOINTS] = seed
            out[step, ARM_JOINTS] = gripper
        return np.ascontiguousarray(out)

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
