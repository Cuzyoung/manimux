"""ManiMux policy plugins for the Xiaomi Robotics 1 (XR-1) checkpoint.

The model plugin only speaks the ``/act`` HTTP protocol. Everything that is
embodiment knowledge lives in the adapter, and for XR-1 that is more than a
column split: the model emits Cartesian deltas, so the adapter runs forward
kinematics on the measured joints, reconstructs absolute end-effector targets,
and solves IK back to the joint groups the executors command.

Action layout (``mibot.utils.io.ACTION_PARTS``), 60 columns per step::

     0: 3  left  end-effector position delta, in the current left EE frame
     3: 6  left  end-effector rotation delta, axis-angle, same frame
     6: 7  left  gripper delta
     8:11  right end-effector position delta
    11:14  right end-effector rotation delta
    14:15  right gripper delta
    16:17  waist delta          -- YAM has no waist, dropped
    17:20  base velocity        -- YAM has no base,  dropped
    rest   reserved, always zero

All 30 steps are deltas against the *same* anchor: the pose measured when the
chunk was requested. They are absolute waypoints once reconstructed, not
incremental step-to-step offsets.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from manimux.config import PolicyConfig, RobotConfig
from manimux.types import ActionChunk, ActionContext, InferenceRequest, ObservationSnapshot

log = logging.getLogger("manimux.policies.xr1")

DEFAULT_SERVER = "http://127.0.0.1:8400"
DEFAULT_GROUP_ORDER = ("left_arm", "right_arm")
DEFAULT_CAMERA_MAP = {
    "top_cam": "front_camera",
    "left_cam": "left_camera",
    "right_cam": "right_camera",
}
ACTION_DIM = 60
ARM_JOINTS = 6
GROUP_DIM = ARM_JOINTS + 1

# Column slices, per arm, inside one 60-wide action row.
ARM_SLICES = {
    "left_arm": {"pos": slice(0, 3), "aa": slice(3, 6), "gripper": slice(6, 7)},
    "right_arm": {"pos": slice(8, 11), "aa": slice(11, 14), "gripper": slice(14, 15)},
}


def joint_condition_to_xr1_actions(
    condition: np.ndarray,
    anchor_groups: Mapping[str, np.ndarray],
    *,
    group_order: Sequence[str],
    kinematics: Any,
) -> np.ndarray:
    """Encode joint-position waypoints as XR-1's anchor-relative 60-D actions."""
    from manimux.integrations.xr1_yam.mibot.utils.io import rotm2aa_batch

    condition = np.asarray(condition, dtype=np.float64)
    expected_dim = sum(np.asarray(anchor_groups[name]).size for name in group_order)
    if condition.ndim != 2 or condition.shape[1] != expected_dim:
        raise ValueError(
            f"XR-1 joint condition must have shape (horizon, {expected_dim}), "
            f"got {condition.shape}"
        )
    if not np.isfinite(condition).all():
        raise ValueError("XR-1 joint condition must be finite")
    if tuple(group_order) != DEFAULT_GROUP_ORDER:
        raise ValueError(f"XR-1 condition codec requires group order {list(DEFAULT_GROUP_ORDER)}")
    if kinematics.num_arm_joints != ARM_JOINTS:
        raise ValueError(f"XR-1 condition codec requires {ARM_JOINTS} arm joints")

    actions = np.zeros((condition.shape[0], ACTION_DIM), dtype=np.float64)
    offset = 0
    for group in group_order:
        anchor = np.asarray(anchor_groups[group], dtype=np.float64).reshape(-1)
        if anchor.shape != (GROUP_DIM,):
            raise ValueError(f"XR-1 anchor group {group!r} must have shape ({GROUP_DIM},)")
        targets = condition[:, offset : offset + GROUP_DIM]
        offset += GROUP_DIM

        anchor_pose = kinematics.fk(anchor[:ARM_JOINTS], float(anchor[-1]))
        target_poses = np.stack(
            [kinematics.fk(row[:ARM_JOINTS], float(row[-1])) for row in targets]
        )
        anchor_rotation = anchor_pose[:3, :3]
        columns = ARM_SLICES[group]
        actions[:, columns["pos"]] = (
            anchor_rotation.T @ (target_poses[:, :3, 3] - anchor_pose[:3, 3]).T
        ).T
        delta_rotations = anchor_rotation.T @ target_poses[:, :3, :3]
        actions[:, columns["aa"]] = rotm2aa_batch(delta_rotations)
        actions[:, columns["gripper"]] = targets[:, -1:] - anchor[-1]

    return np.ascontiguousarray(actions, dtype=np.float32)


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


def _axis_angle_to_rotation(axis_angle: np.ndarray) -> np.ndarray:
    """Rodrigues' formula, matching ``mibot.utils.io.aa2rotm``."""
    theta = float(np.linalg.norm(axis_angle))
    if theta < 1e-8:
        return np.eye(3)
    axis = axis_angle / theta
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(theta) * cross + (1.0 - np.cos(theta)) * (cross @ cross)


class XR1HttpPolicyModel:
    """HTTP inference backend; robot semantics stay in ``XR1YamAdapter``."""

    def __init__(self, config: PolicyConfig) -> None:
        self._url = _server_url(_string_option(config.options, "server", DEFAULT_SERVER))
        self._health_url = self._url.removesuffix("/act") + "/healthz"
        self._group_order = _string_sequence(config.options, "group_order", DEFAULT_GROUP_ORDER)
        self._camera_map = dict(DEFAULT_CAMERA_MAP)
        camera_map = config.options.get("camera_map")
        if camera_map is not None:
            if not isinstance(camera_map, dict) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in camera_map.items()
            ):
                raise ValueError("policy.options.camera_map must map strings to strings")
            self._camera_map = dict(camera_map)
        self._timeout_s = float(config.options.get("http_timeout_s", config.timeout_s))
        if self._timeout_s <= 0:
            raise ValueError("policy.options.http_timeout_s must be positive")
        self._horizon_steps = config.horizon_steps
        from manimux.kinematics import build_kinematics

        kinematics_name = _string_option(config.options, "kinematics", "yam")
        kinematics_options = config.options.get("kinematics_options", {})
        if not isinstance(kinematics_options, dict):
            raise ValueError("policy.options.kinematics_options must be a mapping")
        self._kinematics = build_kinematics(kinematics_name, **kinematics_options)
        self._session_id: str | None = None

    def reset(self, session_id: str) -> None:
        import requests

        response = requests.get(self._health_url, timeout=self._timeout_s)
        if response.status_code != 200:
            raise RuntimeError(f"XR-1 health check failed with status {response.status_code}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("XR-1 health check returned invalid JSON") from exc
        if payload.get("status") != "ok":
            raise RuntimeError(f"XR-1 health check is not ready: {payload!r}")
        self._session_id = session_id

    def infer(self, request: InferenceRequest) -> object:
        if request.session_id != self._session_id:
            raise RuntimeError("XR-1 session is not initialized")
        snapshot = request.observation
        missing_groups = [name for name in self._group_order if name not in snapshot.state.groups]
        if missing_groups:
            raise ValueError(f"XR-1 observation is missing groups: {missing_groups}")
        missing_frames = [name for name in self._camera_map.values() if name not in snapshot.frames]
        if missing_frames:
            raise ValueError(f"XR-1 observation is missing cameras: {missing_frames}")

        state = np.concatenate([snapshot.state.groups[name] for name in self._group_order])
        payload: dict[str, Any] = {
            wire_name: snapshot.frames[source_name].data
            for wire_name, source_name in self._camera_map.items()
        }
        payload.update(
            {
                "timestamp": time.time(),
                "instruction": request.instruction,
                "state": state,
            }
        )

        # An RTC runtime sends the inpainting condition on a request subclass;
        # the default runtime never sets it and the payload is unchanged.
        condition = getattr(request, "action_condition", None)
        weights = getattr(request, "condition_weights", None)
        if (condition is None) != (weights is None):
            raise ValueError("RTC action_condition and condition_weights must be provided together")
        if condition is not None:
            condition_array = np.asarray(condition, dtype=np.float32)
            expected_shape = (
                self._horizon_steps,
                sum(np.asarray(snapshot.state.groups[name]).size for name in self._group_order),
            )
            if condition_array.shape != expected_shape:
                raise ValueError(
                    f"RTC action_condition must have shape {expected_shape}, "
                    f"got {condition_array.shape}"
                )
            weights_array = np.asarray(weights, dtype=np.float32)
            if weights_array.shape != (self._horizon_steps,):
                raise ValueError(
                    f"RTC condition_weights must have shape {(self._horizon_steps,)}, "
                    f"got {weights_array.shape}"
                )
            if not np.isfinite(weights_array).all() or np.any(
                (weights_array < 0.0) | (weights_array > 1.0)
            ):
                raise ValueError("RTC condition_weights must be finite and in [0, 1]")
            beta = float(getattr(request, "rtc_beta", 5.0))
            if not math.isfinite(beta) or beta <= 0:
                raise ValueError(f"rtc_beta must be finite and positive, got {beta}")
            payload["action_condition"] = joint_condition_to_xr1_actions(
                condition_array,
                snapshot.state.groups,
                group_order=self._group_order,
                kinematics=self._kinematics,
            )
            payload["action_condition_weights"] = weights_array
            payload["rtc_beta"] = beta

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
            raise RuntimeError(f"XR-1 server error {response.status_code}: {response.text}")
        decoded = json_numpy.loads(response.text)
        if not isinstance(decoded, dict) or "actions" not in decoded:
            raise ValueError("XR-1 response must contain actions")
        actions = np.asarray(decoded["actions"], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or not np.isfinite(actions).all():
            raise ValueError(
                f"XR-1 actions must be a finite (horizon, {ACTION_DIM}) matrix, got {actions.shape}"
            )
        # The deltas are anchored on the state that produced them, so the anchor
        # has to travel with them to the adapter.
        return {"actions": np.ascontiguousarray(actions), "state": state}

    def close(self) -> None:
        self._session_id = None


class XR1YamAdapter:
    """End-effector deltas -> absolute poses -> YAM joint groups via IK."""

    def __init__(self, robot: RobotConfig, policy: PolicyConfig) -> None:
        from manimux.kinematics import build_kinematics

        self._group_order = _string_sequence(policy.options, "group_order", DEFAULT_GROUP_ORDER)
        self._group_dims = dict(robot.group_dims)
        self._action_dt_ns = int(policy.effective_action_dt_s * 1_000_000_000)
        camera_map = policy.options.get("camera_map", DEFAULT_CAMERA_MAP)
        if not isinstance(camera_map, dict):
            raise ValueError("policy.options.camera_map must be a mapping")
        self._required_cameras = tuple(str(value) for value in camera_map.values())

        kinematics_name = _string_option(policy.options, "kinematics", "yam")
        options = policy.options.get("kinematics_options", {})
        if not isinstance(options, dict):
            raise ValueError("policy.options.kinematics_options must be a mapping")
        self._kinematics = build_kinematics(kinematics_name, **options)
        if self._kinematics.num_arm_joints != ARM_JOINTS:
            raise ValueError(
                f"XR-1 assumes {ARM_JOINTS} arm joints, kinematics reports "
                f"{self._kinematics.num_arm_joints}"
            )

    def build_observation(self, snapshot: ObservationSnapshot) -> ObservationSnapshot:
        missing = [name for name in self._required_cameras if name not in snapshot.frames]
        if missing:
            raise ValueError(f"XR-1 YAM adapter is missing cameras: {missing}")
        return snapshot

    def decode_action(self, raw: object, context: ActionContext) -> ActionChunk:
        if not isinstance(raw, dict) or "actions" not in raw or "state" not in raw:
            raise TypeError("XR-1 adapter expects {'actions': ndarray, 'state': ndarray}")
        actions = np.asarray(raw["actions"], dtype=np.float64)
        anchor = np.asarray(raw["state"], dtype=np.float64).reshape(-1)
        expected_state = sum(self._group_dims[name] for name in self._group_order)
        if anchor.shape != (expected_state,):
            raise ValueError(f"XR-1 anchor state must have shape ({expected_state},)")
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or not actions.shape[0]:
            raise ValueError(
                f"XR-1 actions must have shape (horizon, {ACTION_DIM}), got {actions.shape}"
            )

        groups: dict[str, np.ndarray] = {}
        start = 0
        for name in self._group_order:
            width = self._group_dims[name]
            joints = anchor[start : start + ARM_JOINTS]
            gripper = float(anchor[start + ARM_JOINTS])
            groups[name] = self._solve_arm(name, actions, joints, gripper, width)
            start += width

        return ActionChunk(
            plan_id=f"xr1-{context.request_seq}-{uuid.uuid4().hex[:8]}",
            request_seq=context.request_seq,
            observation_time_ns=context.observation_time_ns,
            created_time_ns=context.created_time_ns,
            action_space="joint_position",
            dt_ns=self._action_dt_ns,
            groups=groups,
        )

    def _solve_arm(
        self,
        group: str,
        actions: np.ndarray,
        joints: np.ndarray,
        gripper: float,
        width: int,
    ) -> np.ndarray:
        columns = ARM_SLICES[group]
        anchor_pose = self._kinematics.fk(joints, gripper)
        anchor_rotation = anchor_pose[:3, :3]
        anchor_position = anchor_pose[:3, 3]

        horizon = actions.shape[0]
        out = np.empty((horizon, width), dtype=np.float64)
        seed = joints.astype(np.float64).copy()
        failures = 0

        for step in range(horizon):
            row = actions[step]
            target = np.eye(4)
            # Deltas are expressed in the anchor end-effector frame.
            target[:3, 3] = anchor_position + anchor_rotation @ row[columns["pos"]]
            target[:3, :3] = anchor_rotation @ _axis_angle_to_rotation(row[columns["aa"]])

            converged, solved = self._kinematics.ik(target, seed, gripper)
            if converged:
                # Seeding the next step with this solution keeps the chunk on one
                # IK branch, so the joint trajectory stays continuous.
                seed = solved
            else:
                # A non-converged solve is wherever the differential solver ran out
                # of iterations, not a pose the model asked for. Hold the last good
                # configuration instead of commanding it, and keep the seed clean so
                # one unreachable step cannot poison the rest of the chunk.
                failures += 1
                solved = seed
            out[step, :ARM_JOINTS] = solved
            out[step, ARM_JOINTS] = gripper + float(row[columns["gripper"]][0])

        if failures:
            log.warning("%s: IK did not converge on %d/%d chunk steps", group, failures, horizon)
        if not np.isfinite(out).all():
            raise ValueError(f"XR-1 IK produced non-finite joints for {group}")
        return np.ascontiguousarray(out)

    def validate(self, robot: RobotConfig, policy: PolicyConfig) -> None:
        del policy
        if tuple(robot.group_dims) != self._group_order:
            raise ValueError(
                "XR-1 YAM requires robot groups in order "
                f"{list(self._group_order)}, got {list(robot.group_dims)}"
            )
        if any(robot.group_dims[name] != GROUP_DIM for name in self._group_order):
            raise ValueError(f"XR-1 YAM requires two {GROUP_DIM}-value arm+gripper groups")


def build_model(config: PolicyConfig) -> XR1HttpPolicyModel:
    return XR1HttpPolicyModel(config)


def build_adapter(robot: RobotConfig, policy: PolicyConfig) -> XR1YamAdapter:
    return XR1YamAdapter(robot, policy)
