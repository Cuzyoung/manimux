"""Built-in YAM robot adapter using the official i2rt model assets."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np

from .base import RobotAdapter, RobotGroup, SceneBox

DEFAULT_I2RT_ROOT = Path(__file__).resolve().parents[2] / "assets"


class YamKinematics:
    """Read-only FK assembled from the official YAM arm and gripper XML."""

    def __init__(self, i2rt_root: Path) -> None:
        models = i2rt_root / "i2rt/robot_models"
        arm_path = models / "arm/yam/yam.xml"
        gripper_path = models / "gripper/linear_4310/linear_4310.xml"
        for path in (arm_path, gripper_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"YAM model asset not found: {path}. "
                    "Pass --robot-model-root with the i2rt checkout path."
                )
        self.xml_path = arm_path
        self.model = mujoco.MjModel.from_xml_string(self._combined_xml(arm_path, gripper_path))
        self.data = mujoco.MjData(self.model)
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        if self.site_id < 0:
            raise RuntimeError(f"grasp_site is missing from {self.xml_path}")

    @staticmethod
    def _combined_xml(arm_path: Path, gripper_path: Path) -> str:
        arm_root = ET.parse(arm_path).getroot()
        gripper_root = ET.parse(gripper_path).getroot()
        arm_asset = arm_root.find("asset")
        if arm_asset is None:
            raise RuntimeError(f"asset section is missing from {arm_path}")
        for mesh in arm_asset.findall("mesh"):
            filename = Path(mesh.get("file", "")).name
            mesh.set("file", str((arm_path.parent / "assets" / filename).resolve()))
        gripper_asset = gripper_root.find("asset")
        if gripper_asset is not None:
            for mesh in gripper_asset.findall("mesh"):
                copy = deepcopy(mesh)
                filename = Path(mesh.get("file", "")).name
                copy.set(
                    "file",
                    str((gripper_path.parent / "assets" / filename).resolve()),
                )
                arm_asset.append(copy)
        compiler = arm_root.find("compiler")
        if compiler is not None:
            compiler.attrib.pop("meshdir", None)
        body = arm_root.find("worldbody/body")
        if body is None:
            raise RuntimeError(f"root body is missing from {arm_path}")
        while body.find("body") is not None:
            child = body.find("body")
            assert child is not None
            body = child
        gripper_body = gripper_root.find("worldbody/body[@name='gripper']")
        if gripper_body is None:
            raise RuntimeError(f"gripper body is missing from {gripper_path}")
        body.append(deepcopy(gripper_body))
        for tag in ("equality", "contact"):
            element = gripper_root.find(tag)
            if element is not None:
                arm_root.append(deepcopy(element))
        return ET.tostring(arm_root, encoding="unicode")

    def qpos(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size != 7:
            raise ValueError(f"expected 7 YAM values, got {action.size}")
        qpos = np.zeros(self.model.nq, dtype=np.float64)
        qpos[:7] = action
        qpos[6] = np.clip(qpos[6], 0.0, 1.0)
        for joint_id in range(self.model.njnt):
            address = int(self.model.jnt_qposadr[joint_id])
            if address < 7 and self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_SLIDE:
                lo, hi = self.model.jnt_range[joint_id]
                qpos[address] = lo + qpos[address] * (hi - lo)
        for equality_id in range(self.model.neq):
            if self.model.eq_type[equality_id] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            joint_1 = int(self.model.eq_obj1id[equality_id])
            joint_2 = int(self.model.eq_obj2id[equality_id])
            address_1 = int(self.model.jnt_qposadr[joint_1])
            address_2 = int(self.model.jnt_qposadr[joint_2])
            coefficients = self.model.eq_data[equality_id, :5][::-1]
            qpos[address_2] = np.polyval(coefficients, qpos[address_1])
        return qpos

    def pose(self, action: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = self.qpos(action)
        mujoco.mj_forward(self.model, self.data)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = self.data.site_xmat[self.site_id].reshape(3, 3)
        transform[:3, 3] = self.data.site_xpos[self.site_id]
        return transform


class YamAdapter(RobotAdapter):
    """First bundled robot: one or two 7-DoF YAM arms."""

    name = "yam"
    label = "YAM"

    def __init__(
        self,
        model_root: Path | str | None = None,
        i2rt_root: Path | str | None = None,
    ) -> None:
        root_value = model_root if model_root is not None else i2rt_root
        self.i2rt_root = Path(root_value or DEFAULT_I2RT_ROOT).expanduser().resolve()
        urdf_path = self.i2rt_root / "i2rt/robot_models/arm/yam/yam.urdf"
        if not urdf_path.is_file():
            raise FileNotFoundError(
                f"YAM URDF not found: {urdf_path}. "
                "Pass --robot-model-root with the i2rt checkout path."
            )
        self.groups = (
            RobotGroup(
                "left",
                "Left arm",
                (0.0, 0.32, 0.0),
                (52, 111, 255),
                (33, 180, 106),
                urdf_path,
            ),
            RobotGroup(
                "right",
                "Right arm",
                (0.0, -0.32, 0.0),
                (255, 94, 87),
                (255, 170, 0),
                urdf_path,
            ),
        )
        self.scene_boxes = (SceneBox("table", (0.75, 0.82, 0.035), (0.38, 0.0, 0.02)),)
        self.kinematics = YamKinematics(self.i2rt_root)

    @staticmethod
    def _split(values: np.ndarray, *, sequence: bool) -> dict[str, np.ndarray]:
        array = np.asarray(values, dtype=np.float64)
        expected_ndim = 2 if sequence else 1
        if array.ndim != expected_ndim:
            kind = "action chunk" if sequence else "joint state"
            raise ValueError(f"YAM {kind} must be {expected_ndim}D, got {array.ndim}D")
        width = array.shape[-1]
        if width == 7:
            return {"left": array}
        if width == 14:
            return {"left": array[..., :7], "right": array[..., 7:14]}
        raise ValueError(f"YAM expects 7 or 14 values, got {width}")

    def split_actions(
        self, actions: np.ndarray, action_space: str = "joint_position"
    ) -> dict[str, np.ndarray]:
        if action_space != "joint_position":
            raise ValueError(
                f"YAM adapter supports action_space='joint_position', got {action_space!r}"
            )
        return self._split(actions, sequence=True)

    def split_joint_positions(self, joint_positions: np.ndarray) -> dict[str, np.ndarray]:
        return self._split(joint_positions, sequence=False)

    def pose(self, group: str, configuration: np.ndarray) -> np.ndarray:
        self.group(group)
        return self.kinematics.pose(configuration)

    def visual_configuration(self, group: str, configuration: np.ndarray) -> np.ndarray:
        self.group(group)
        action = np.asarray(configuration, dtype=np.float64).reshape(-1)
        if action.size != 7:
            raise ValueError(f"YAM group configuration must contain 7 values, got {action.size}")
        gripper = float(np.clip(action[6], 0.0, 1.0)) * -0.04695
        return np.concatenate((action[:6], np.array([gripper, gripper])))

    def initial_configuration(self, group: str) -> np.ndarray:
        self.group(group)
        return np.zeros(8, dtype=np.float64)

    def camera_slot(self, source_name: str) -> str:
        aliases = {"top_camera": "top", "front_camera": "top"}
        normalized = source_name.removesuffix("_rgb")
        return aliases.get(normalized, normalized.removesuffix("_camera"))

    def demo_sample(self, elapsed_s: float, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        left = np.array(
            [
                0.15 * np.sin(elapsed_s),
                0.8,
                1.25,
                -0.35 + 0.15 * np.sin(elapsed_s * 1.3),
                0.1,
                -0.2,
                0.7,
            ]
        )
        right = np.array(
            [
                -0.15 * np.sin(elapsed_s),
                0.75,
                1.20,
                -0.25 + 0.12 * np.cos(elapsed_s),
                -0.1,
                0.2,
                0.4,
            ]
        )
        chunk = []
        denominator = max(1, horizon - 1)
        for index in range(horizon):
            left_delta = np.array(
                [0.1 * index / denominator, 0, 0, 0.05 * np.sin(index / 4), 0, 0, 0]
            )
            right_delta = np.array(
                [-0.1 * index / denominator, 0, 0, -0.05 * np.sin(index / 4), 0, 0, 0]
            )
            chunk.append(np.r_[left + left_delta, right + right_delta])
        return np.r_[left, right], np.stack(chunk)
