"""Universal Viser dashboard for live robot-policy inference."""

from __future__ import annotations

import argparse
import base64
import io
import signal
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, cast

import numpy as np
import viser
from PIL import Image

from manimux.evaluation import write_manual_evaluation
from manimux.types import FloatArray, UInt8Array

from .protocol import PolicyPlan, RobotSnapshot
from .robots import available_robot_adapters, load_robot_adapter
from .robots.base import RobotAdapter, RobotGroup
from .transport import ControlServer, ViewerReceiver

MAX_PLAN_HISTORY = 16
_TRAJECTORY_COLOR_STOPS = np.asarray(
    [
        (67, 20, 133),
        (101, 31, 183),
        (139, 52, 230),
        (177, 92, 255),
        (210, 150, 255),
    ],
    dtype=np.float64,
)


def _trajectory_colors(point_count: int) -> UInt8Array:
    """Return deep-to-light purple from the current pose into the future."""

    if point_count <= 0:
        return np.empty((0, 3), dtype=np.uint8)
    progress = np.linspace(0.0, 1.0, point_count)
    stop_positions = np.linspace(0.0, 1.0, len(_TRAJECTORY_COLOR_STOPS))
    return np.stack(
        [
            np.interp(progress, stop_positions, _TRAJECTORY_COLOR_STOPS[:, channel])
            for channel in range(3)
        ],
        axis=1,
    ).astype(np.uint8)


def _instruction_markdown(instruction: str) -> str:
    """Render the live task prompt for the always-visible header."""

    text = instruction.strip()
    return f"### 📋 {text}" if text else "### 📋 _waiting for a task prompt_"


def _prefill_task(current: str, incoming: str) -> str:
    """Seed the operator's editable command without overwriting their typing.

    The runtime republishes its instruction on every chunk, so assigning it each
    time would erase a command the operator is halfway through composing.
    """

    return incoming.strip() if not current.strip() else current


class PolicyViewer:
    """Robot-independent dashboard backed by one selected robot adapter."""

    def __init__(
        self,
        host: str,
        port: int,
        bridge_endpoint: str,
        control_endpoint: str,
        robot: RobotAdapter,
    ) -> None:
        self.robot = robot
        self.server = viser.ViserServer(host=host, port=port, label="Universal Policy Viewer")
        self.server.gui.configure_theme(
            control_layout="fixed",
            control_width="medium",
            dark_mode=False,
            show_logo=False,
            show_share_button=False,
            brand_color=(70, 103, 190),
        )
        self.server.gui.set_panel_label("UNIVERSAL · POLICY VIEWER")
        self.lock = threading.RLock()
        self.running = True
        self.paused = True
        self.step_once = False
        self.finish_requested = False
        self.home_requested = False
        self.new_rollout_requested = False
        self.preparing_rollout = False
        self.service_ready = False
        self.experiment_mode = False
        self.evaluation_saved = True
        self.episode_active = False
        self.launch_mode = "unknown"
        self.last_state_time = 0.0
        self.tails: dict[str, deque[FloatArray]] = {
            group.name: deque(maxlen=300) for group in self.robot.groups
        }
        self.robot_handles: dict[str, Any] = {}
        self.plan_actions: dict[str, FloatArray] = {}
        self.plan_chunk_id: int | None = None
        self.plan_start_index = 0
        self.plan_history_serial = 0
        self.show_plan_history = False
        self.current_plan_handles: dict[str, list[Any]] = {
            group.name: [] for group in self.robot.groups
        }
        self.plan_history_handles: dict[str, deque[Any]] = {
            group.name: deque() for group in self.robot.groups
        }
        self.observe_only = False
        self.current_episode_dir: Path | None = None
        self.episode_finalized = False
        self.camera_images: dict[str, Any] = {}
        self._build_scene()
        self._build_gui()
        self.receiver = ViewerReceiver(bridge_endpoint, self.on_message)
        self.control_server = ControlServer(control_endpoint, self.control_state)

    @staticmethod
    def _root(group: RobotGroup) -> str:
        return f"/robot/{group.name}"

    def _build_scene(self) -> None:
        self.server.scene.set_up_direction("+z")
        self.server.scene.add_grid(
            "/floor",
            width=2.0,
            height=1.6,
            cell_size=0.1,
            section_size=0.5,
            position=(0.25, 0.0, -0.01),
        )
        self.server.scene.add_frame("/world", axes_length=0.15, axes_radius=0.006)
        for box in self.robot.scene_boxes:
            self.server.scene.add_box(
                f"/environment/{box.name}",
                color=box.color,
                dimensions=box.dimensions,
                position=box.position,
            )
        for group in self.robot.groups:
            root = self._root(group)
            self.server.scene.add_frame(root, show_axes=False, position=group.base_position)
            self.server.scene.add_frame(f"{root}/current_ee", axes_length=0.08, axes_radius=0.004)
            if group.urdf_path is None:
                continue
            try:
                from viser.extras import ViserUrdf

                current = ViserUrdf(self.server, group.urdf_path, root_node_name=root)
                current.update_cfg(self.robot.initial_configuration(group.name))
                self.robot_handles[group.name] = current
            # A third-party visualizer may raise backend-specific exceptions;
            # URDF rendering is optional, so preserve trajectory-only operation.
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[viewer] {group.label} URDF unavailable ({exc}); "
                    "trajectory rendering remains enabled"
                )

    def _build_gui(self) -> None:
        self.status = self.server.gui.add_markdown("🟠 **Waiting for policy executor**")
        self.instruction = self.server.gui.add_markdown(_instruction_markdown(""))
        with self.server.gui.add_folder("New rollout", expand_by_default=True):
            self.rollout_setup_status = self.server.gui.add_markdown(
                "⚪ Waiting for a ManiMux runtime service."
            )
            self.task = self.server.gui.add_text(
                "Task command", "", multiline=True, disabled=True
            )
            self.layout_id = self.server.gui.add_text(
                "Experiment layout / condition ID", "default", disabled=True
            )
            self.prepare_normal_btn = self.server.gui.add_button(
                "Prepare normal rollout", color="blue", disabled=True
            )
            self.prepare_experiment_btn = self.server.gui.add_button(
                "🧪 Prepare experiment rollout", color="green", disabled=True
            )
        with self.server.gui.add_folder("Policy control", expand_by_default=True):
            self.start_btn = self.server.gui.add_button(
                "Start / Resume", color="blue", disabled=True
            )
            self.pause_btn = self.server.gui.add_button("Pause", color="gray", disabled=True)
            self.home_btn = self.server.gui.add_button("Home", color="gray", disabled=True)
            self.step_btn = self.server.gui.add_button("Step once", color="gray", disabled=True)
            self.finish_btn = self.server.gui.add_button(
                "Finish rollout", color="red", disabled=True
            )
        with self.server.gui.add_folder("Run", expand_by_default=True):
            self.robot_name = self.server.gui.add_text("Robot", self.robot.label, disabled=True)
            self.policy_name = self.server.gui.add_text("Policy", "waiting", disabled=True)
            self.action_space = self.server.gui.add_text("Action space", "waiting", disabled=True)
            self.chunk_info = self.server.gui.add_text("Chunk", "—", disabled=True)
            self.executor_info = self.server.gui.add_text("Executor", "waiting", disabled=True)
            self.latency = self.server.gui.add_text("Inference", "—", disabled=True)
            self.progress = self.server.gui.add_number("Step", 0, disabled=True)
            self.runtime_name = self.server.gui.add_text("Runtime", "waiting", disabled=True)
            self.episode_path = self.server.gui.add_text("Episode", "waiting", disabled=True)
        with self.server.gui.add_folder("Post-rollout evaluation", expand_by_default=True):
            self.evaluation_status = self.server.gui.add_markdown(
                "⚪ Finish the rollout before saving an evaluation."
            )
            self.task_result = self.server.gui.add_dropdown(
                "Task result",
                ("unlabeled", "success", "failure", "invalid"),
                initial_value="unlabeled",
                disabled=True,
            )
            self.smoothness_score = self.server.gui.add_dropdown(
                "Smoothness (1-5)",
                ("1", "2", "3", "4", "5"),
                initial_value="3",
                disabled=True,
            )
            self.reviewer_id = self.server.gui.add_text(
                "Reviewer", "operator", disabled=True
            )
            self.operator_note = self.server.gui.add_text(
                "Operator note", "", multiline=True, disabled=True
            )
            self.failure_tag_inputs = {
                "replay_backtrack": self.server.gui.add_checkbox(
                    "Replay / backtrack", False, disabled=True
                ),
                "hold_stall": self.server.gui.add_checkbox(
                    "Hold / stall", False, disabled=True
                ),
                "collision": self.server.gui.add_checkbox(
                    "Collision", False, disabled=True
                ),
                "drop_spill": self.server.gui.add_checkbox(
                    "Drop / spill", False, disabled=True
                ),
                "perception": self.server.gui.add_checkbox(
                    "Perception error", False, disabled=True
                ),
                "policy_semantics": self.server.gui.add_checkbox(
                    "Policy / task error", False, disabled=True
                ),
                "safety_stop": self.server.gui.add_checkbox(
                    "Safety stop", False, disabled=True
                ),
                "other": self.server.gui.add_checkbox("Other", False, disabled=True),
            }
            self.save_evaluation_btn = self.server.gui.add_button(
                "Save evaluation", color="blue", disabled=True
            )
        with self.server.gui.add_folder("Overlay controls", expand_by_default=True):
            self.show_plan = self.server.gui.add_checkbox("Predicted EE trajectory", True)
            self.show_tail = self.server.gui.add_checkbox("Achieved EE trail", True)
            self.show_frames = self.server.gui.add_checkbox("EE coordinate frames", True)
            self.show_history_btn = self.server.gui.add_button(
                "Show trajectory history", color="gray"
            )
            self.hide_history_btn = self.server.gui.add_button(
                "Current trajectory only", color="blue", visible=False
            )
            self.clear_history_btn = self.server.gui.add_button("Clear trajectory history")
            self.clear_btn = self.server.gui.add_button("Clear trails")
        self.images_folder = self.server.gui.add_folder("Images", expand_by_default=True)

        @self.start_btn.on_click
        def _start(_event: Any) -> None:
            self.paused = False
            self.step_once = False
            self.status.content = "🟢 **Connected · RUNNING**"

        @self.prepare_normal_btn.on_click
        def _prepare_normal(_event: Any) -> None:
            self._prepare_rollout(experiment_mode=False)

        @self.prepare_experiment_btn.on_click
        def _prepare_experiment(_event: Any) -> None:
            self._prepare_rollout(experiment_mode=True)

        @self.pause_btn.on_click
        def _pause(_event: Any) -> None:
            self.paused = True
            self.status.content = "🟡 **Connected · PAUSED**"

        @self.step_btn.on_click
        def _step(_event: Any) -> None:
            self.paused = True
            self.step_once = True
            self.status.content = "🟡 **Single step requested**"

        @self.home_btn.on_click
        def _home(_event: Any) -> None:
            self.paused = True
            self.home_requested = True
            self.status.content = "🟡 **Returning home · PAUSED**"

        @self.finish_btn.on_click
        def _finish(_event: Any) -> None:
            self.finish_requested = True
            self.finish_btn.disabled = True
            self.status.content = "🟠 **Finishing rollout and saving episode**"

        @self.clear_btn.on_click
        def _clear(_event: Any) -> None:
            self._clear_achieved_tails()

        @self.show_history_btn.on_click
        def _show_history(_event: Any) -> None:
            self.show_plan_history = True
            self.show_history_btn.visible = False
            self.hide_history_btn.visible = True
            self._refresh_plan_visibility()

        @self.hide_history_btn.on_click
        def _hide_history(_event: Any) -> None:
            self.show_plan_history = False
            self.show_history_btn.visible = True
            self.hide_history_btn.visible = False
            self._refresh_plan_visibility()

        @self.clear_history_btn.on_click
        def _clear_history(_event: Any) -> None:
            self._clear_plan_history()

        @self.save_evaluation_btn.on_click
        def _save_evaluation(_event: Any) -> None:
            self._save_manual_evaluation()

        @self.show_plan.on_update
        def _show_plan(_event: Any) -> None:
            self._refresh_plan_visibility()

    def _set_instruction(self, instruction: str) -> None:
        self.instruction.content = _instruction_markdown(instruction)
        self.task.value = _prefill_task(self.task.value, instruction)

    def _set_evaluation_enabled(self, enabled: bool) -> None:
        for handle in (
            self.task_result,
            self.smoothness_score,
            self.reviewer_id,
            self.operator_note,
            *self.failure_tag_inputs.values(),
        ):
            handle.disabled = not enabled
        self.save_evaluation_btn.disabled = not enabled

    def _set_experiment_mode(self, enabled: bool) -> None:
        self.experiment_mode = enabled
        self.rollout_setup_status.content = (
            "🟢 **Experiment rollout** · a human label is required after Finish."
            if enabled
            else "🔵 **Normal rollout** · no human label is required."
        )

    def _set_setup_controls_enabled(self, enabled: bool) -> None:
        allowed = enabled and self.evaluation_saved
        self.prepare_normal_btn.disabled = not allowed
        self.prepare_experiment_btn.disabled = not allowed
        self.layout_id.disabled = not allowed
        self.task.disabled = not allowed

    def _prepare_rollout(self, *, experiment_mode: bool) -> None:
        self._set_experiment_mode(experiment_mode)
        self.new_rollout_requested = True
        self.preparing_rollout = True
        self.service_ready = False
        self.paused = True
        self._set_setup_controls_enabled(False)
        kind = "experiment" if experiment_mode else "normal"
        self.status.content = f"🟠 **Preparing a new {kind} rollout**"

    def _set_policy_controls_enabled(self, enabled: bool) -> None:
        for button in (
            self.start_btn,
            self.pause_btn,
            self.home_btn,
            self.step_btn,
            self.finish_btn,
        ):
            button.disabled = not enabled or self.observe_only

    def _update_prepare_enabled(self) -> None:
        self._set_setup_controls_enabled(self.service_ready)

    def _reset_evaluation(self, episode_dir: str) -> None:
        self.current_episode_dir = Path(episode_dir).expanduser() if episode_dir else None
        self.episode_finalized = False
        self.evaluation_saved = not self.experiment_mode
        self.episode_path.value = episode_dir or "not published"
        self.task_result.value = "unlabeled"
        self.smoothness_score.value = "3"
        self.reviewer_id.value = "operator"
        self.operator_note.value = ""
        for handle in self.failure_tag_inputs.values():
            handle.value = False
        self._set_evaluation_enabled(False)
        self.evaluation_status.content = "⚪ Finish the rollout before saving an evaluation."

    def _save_manual_evaluation(self) -> None:
        if not self.episode_finalized or self.current_episode_dir is None:
            self.evaluation_status.content = "🔴 Episode is not finalized."
            return
        result = str(self.task_result.value)
        if result == "unlabeled":
            self.evaluation_status.content = "🔴 Select success, failure, or invalid."
            return
        try:
            target = write_manual_evaluation(
                self.current_episode_dir,
                task_result=cast(Any, result),
                smoothness_score=int(self.smoothness_score.value),
                failure_tags=[
                    name for name, handle in self.failure_tag_inputs.items() if handle.value
                ],
                operator_note=self.operator_note.value,
                reviewer_id=self.reviewer_id.value,
            )
        except (OSError, TypeError, ValueError) as exc:
            self.evaluation_status.content = f"🔴 Evaluation was not saved: {exc}"
            return
        self.evaluation_status.content = f"🟢 Saved `{target}`"
        self.evaluation_saved = True
        self._update_prepare_enabled()

    def control_state(self) -> dict[str, Any]:
        state = {
            "paused": self.paused,
            "step_once": self.step_once,
            "home_requested": self.home_requested,
            "finish_requested": self.finish_requested,
            "new_rollout_requested": self.new_rollout_requested,
            "task_command": self.task.value.strip(),
            "experiment_mode": self.experiment_mode,
            "layout_id": self.layout_id.value.strip(),
        }
        self.step_once = False
        self.home_requested = False
        self.finish_requested = False
        self.new_rollout_requested = False
        return state

    @staticmethod
    def _image(payload: str) -> UInt8Array:
        raw = base64.b64decode(payload)
        return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))

    def _matches_selected_robot(self, message: dict[str, Any]) -> bool:
        robot_name = str(message.get("robot", ""))
        if not robot_name or robot_name == self.robot.name:
            return True
        self.status.content = (
            f"🔴 **Message targets robot `{robot_name}`; viewer uses `{self.robot.name}`**"
        )
        return False

    def on_message(self, message: dict[str, Any]) -> None:
        with self.lock:
            if not self._matches_selected_robot(message):
                return
            try:
                kind = message.get("kind")
                if kind == "plan":
                    self._update_plan(message)
                elif kind == "state":
                    self._update_state(message)
                elif kind == "event":
                    self._update_event(message)
            except (KeyError, TypeError, ValueError) as exc:
                self.status.content = (
                    f"🔴 **Rejected malformed {message.get('kind')} message: {exc}**"
                )

    def _update_plan(self, message: dict[str, Any]) -> None:
        # ``joint_actions`` keeps old local publishers readable during migration.
        raw_actions = message.get("actions", message.get("joint_actions"))
        if raw_actions is None:
            raise ValueError("plan message is missing actions")
        actions = np.asarray(raw_actions, dtype=np.float64)
        action_space = str(message.get("action_space", "joint_position"))
        grouped_actions = self.robot.split_actions(actions, action_space)
        start_index = int(message.get("start_index", 0))
        if start_index < 0 or start_index > len(actions):
            raise ValueError("plan start_index is outside the action horizon")
        chunk_id = int(message.get("chunk_id", 0))
        if self.plan_chunk_id is not None and chunk_id != self.plan_chunk_id:
            self._archive_current_plan()
        self.plan_actions = {
            group_name: np.asarray(group_actions, dtype=np.float64)
            for group_name, group_actions in grouped_actions.items()
        }
        self.plan_chunk_id = chunk_id
        self.plan_start_index = start_index
        self.policy_name.value = str(message.get("policy", "unknown"))
        self._set_instruction(str(message.get("instruction", "")))
        self.action_space.value = action_space
        self.latency.value = f"{float(message.get('inference_ms', 0.0)):.0f} ms"
        self.chunk_info.value = (
            f"#{message.get('chunk_id', 0)} · {len(actions)} actions · "
            f"{float(message.get('action_dt', 0.0)):.3f}s"
        )
        for group in self.robot.groups:
            group_actions = grouped_actions.get(group.name)
            if group_actions is None:
                self._draw_plan(group, np.empty((0, 0)))
            else:
                self._draw_plan(group, group_actions[start_index:])

    def _draw_plan(self, group: RobotGroup, actions: FloatArray) -> None:
        root = f"{self._root(group)}/predicted_ee"
        if len(actions) < 2:
            for handle in self.current_plan_handles[group.name]:
                handle.visible = False
            self.current_plan_handles[group.name] = []
            return
        # This line lives below the group root, which already carries the arm's
        # base_position. Keep FK output in that local frame to avoid applying
        # the left/right base offset twice.
        points = self.robot.positions(group.name, actions)
        segments = np.stack((points[:-1], points[1:]), axis=1)
        point_colors = _trajectory_colors(len(points))
        segment_colors = np.stack((point_colors[:-1], point_colors[1:]), axis=1)
        visible = bool(self.show_plan.value)
        path = self.server.scene.add_line_segments(
            f"{root}/path",
            segments,
            segment_colors,
            line_width=10,
            visible=visible,
        )
        start = self.server.scene.add_icosphere(
            f"{root}/start",
            radius=0.012,
            color=cast(tuple[int, int, int], tuple(int(value) for value in point_colors[0])),
            position=points[0],
            visible=visible,
        )
        end = self.server.scene.add_icosphere(
            f"{root}/end",
            radius=0.015,
            color=cast(tuple[int, int, int], tuple(int(value) for value in point_colors[-1])),
            position=points[-1],
            visible=visible,
        )
        self.current_plan_handles[group.name] = [path, start, end]

    def _archive_current_plan(self) -> None:
        """Keep the previous activated chunk as a dim trajectory overlay."""

        self.plan_history_serial += 1
        for group in self.robot.groups:
            actions = self.plan_actions.get(group.name)
            if actions is None:
                continue
            actions = actions[self.plan_start_index :]
            if len(actions) < 2:
                continue
            points = self.robot.positions(group.name, actions)
            point_colors = _trajectory_colors(len(points)).astype(np.float64)
            muted_colors = (0.55 * point_colors + 0.45 * 190.0).astype(np.uint8)
            history_colors = np.stack((muted_colors[:-1], muted_colors[1:]), axis=1)
            handle = self.server.scene.add_line_segments(
                f"{self._root(group)}/predicted_history/{self.plan_history_serial}",
                np.stack((points[:-1], points[1:]), axis=1),
                history_colors,
                line_width=3.5,
                visible=bool(self.show_plan.value) and self.show_plan_history,
            )
            history = self.plan_history_handles[group.name]
            history.append(handle)
            while len(history) > MAX_PLAN_HISTORY:
                history.popleft().remove()

    def _clear_plan_history(self) -> None:
        for history in self.plan_history_handles.values():
            while history:
                history.popleft().remove()

    def _refresh_plan_visibility(self) -> None:
        show_current = bool(self.show_plan.value)
        for handles in self.current_plan_handles.values():
            for handle in handles:
                handle.visible = show_current
        show_history = show_current and self.show_plan_history
        for history in self.plan_history_handles.values():
            for handle in history:
                handle.visible = show_history

    def _reset_plan_overlay(self) -> None:
        for handles in self.current_plan_handles.values():
            for handle in handles:
                handle.visible = False
        self.current_plan_handles = {group.name: [] for group in self.robot.groups}
        self._clear_plan_history()
        self.plan_actions = {}
        self.plan_chunk_id = None
        self.plan_start_index = 0

    def _clear_achieved_tails(self) -> None:
        for group in self.robot.groups:
            self.tails[group.name].clear()
            self._draw_tail(group, np.empty((0, 3)))

    def _update_state(self, message: dict[str, Any]) -> None:
        joint_positions = np.asarray(message.get("joint_positions", []), dtype=np.float64)
        grouped_positions = self.robot.split_joint_positions(joint_positions)
        self.last_state_time = time.time()
        self.progress.value = int(message.get("step", 0))
        if not message.get("connected", True):
            self.status.content = "🟠 **Executor disconnected**"
        elif self.paused:
            self.status.content = (
                "🔵 **Connected · OBSERVE ONLY**"
                if self.observe_only
                else "🟡 **Connected · PAUSED**"
            )
        else:
            self.status.content = "🟢 **Connected · RUNNING**"
        active_chunk_id = message.get("active_chunk_id")
        action_index = int(message.get("chunk_index", 0))
        if (
            active_chunk_id is not None
            and self.plan_chunk_id is not None
            and int(active_chunk_id) == self.plan_chunk_id
        ):
            for group in self.robot.groups:
                actions = self.plan_actions.get(group.name)
                self._draw_plan(
                    group,
                    actions[action_index:] if actions is not None else np.empty((0, 0)),
                )
        for group_name, configuration in grouped_positions.items():
            self._update_group(self.robot.group(group_name), configuration)
        for source_name, payload in message.get("cameras_jpeg", {}).items():
            slot = self.robot.camera_slot(source_name)
            image = self._image(payload)
            if slot not in self.camera_images:
                with self.images_folder:
                    self.camera_images[slot] = self.server.gui.add_image(
                        image,
                        label=slot,
                        format="jpeg",
                        jpeg_quality=70,
                    )
            else:
                self.camera_images[slot].image = image

    def _update_event(self, message: dict[str, Any]) -> None:
        event = str(message.get("event", "unknown"))
        metadata = message.get("metadata") or {}
        if event == "episode_started":
            self._reset_plan_overlay()
            self._clear_achieved_tails()
            self.episode_active = True
            self.launch_mode = str(metadata.get("launch_mode", "run"))
            self.observe_only = metadata.get("control_mode", "observe") == "observe"
            self.paused = True
            self.service_ready = False
            self.preparing_rollout = False
            self._set_experiment_mode(bool(metadata.get("experiment_mode", False)))
            self.layout_id.value = str(metadata.get("layout_id", "")) or "default"
            self._set_setup_controls_enabled(False)
            self._set_policy_controls_enabled(True)
            self._set_instruction(str(metadata.get("instruction", self.task.value)))
            self.policy_name.value = str(metadata.get("policy_label", "waiting"))
            self.runtime_name.value = str(metadata.get("runtime", "waiting"))
            self._reset_evaluation(str(metadata.get("episode_dir", "")))
            self.executor_info.value = "observe only" if self.observe_only else "managed"
            self.status.content = (
                "🔵 **Connected · OBSERVE ONLY**"
                if self.observe_only
                else "🟡 **Connected · PAUSED · press Start / Resume**"
            )
        elif event == "inference_submitted":
            planned = metadata.get("planned_switch_step")
            suffix = f" → switch {planned}" if planned is not None else ""
            self.executor_info.value = f"chunk #{message.get('chunk_id')} pending{suffix}"
        elif event == "episode_finished":
            self.episode_active = False
            self.executor_info.value = str(metadata.get("reason", "finished"))
            episode_dir = str(metadata.get("episode_dir", ""))
            if episode_dir:
                self.current_episode_dir = Path(episode_dir).expanduser()
                self.episode_path.value = episode_dir
            self.episode_finalized = self.current_episode_dir is not None
            self.evaluation_saved = not self.experiment_mode
            self.paused = True
            self._set_policy_controls_enabled(False)
            self._set_evaluation_enabled(self.episode_finalized and self.experiment_mode)
            if not self.episode_finalized:
                self.evaluation_status.content = "🔴 Runtime did not publish a rollout path."
                self.status.content = "🔴 **Rollout finished without a saved path**"
            elif self.experiment_mode:
                self.evaluation_status.content = (
                    "🟡 Select the task result and smoothness score, then save."
                )
                self.status.content = (
                    "⚪ **Rollout finished · awaiting human reward**"
                    if self.launch_mode == "serve"
                    else "⚪ **One-shot run finished · awaiting human reward**"
                )
            else:
                self.evaluation_status.content = (
                    "⚪ Experiment mode was OFF; no human reward is required."
                )
                self.status.content = (
                    "⚪ **Rollout finished · preparing for the next rollout**"
                    if self.launch_mode == "serve"
                    else "⚪ **One-shot run finished · use `manimux serve` for UI rollouts**"
                )
        elif event == "runtime_service_ready":
            first_service_announcement = self.launch_mode != "serve"
            self.launch_mode = "serve"
            self.policy_name.value = str(metadata.get("policy_label", "waiting"))
            self.runtime_name.value = str(metadata.get("runtime", "waiting"))
            self._set_instruction(str(metadata.get("task", self.task.value)))
            last_error = str(metadata.get("last_error", ""))
            if self.preparing_rollout and not last_error:
                return
            self.preparing_rollout = False
            self.service_ready = True
            if first_service_announcement:
                self.layout_id.value = (
                    str(metadata.get("default_layout_id", "")) or "default"
                )
            self.rollout_setup_status.content = (
                "Choose a normal rollout, or an experiment rollout that requires a label."
            )
            if self.current_episode_dir is None or self.evaluation_saved:
                self.episode_path.value = str(metadata.get("last_episode_dir", "")) or "ready"
            self._update_prepare_enabled()
            if last_error:
                self.status.content = (
                    f"🔴 **Last rollout failed · service idle**  \\n`{last_error}`"
                )
            elif self.evaluation_saved:
                self.status.content = "🟡 **Runtime service ready · prepare a rollout**"
        elif event == "episode_failed":
            self.episode_active = False
            self.preparing_rollout = False
            self.service_ready = True
            self.evaluation_saved = True
            self._set_policy_controls_enabled(False)
            self._update_prepare_enabled()
            error = str(metadata.get("error", "unknown startup error"))
            self.status.content = f"🔴 **Rollout failed before execution**  \\n`{error}`"

    def _update_group(self, group: RobotGroup, configuration: FloatArray) -> None:
        robot_handle = self.robot_handles.get(group.name)
        if robot_handle is not None:
            robot_handle.update_cfg(self.robot.visual_configuration(group.name, configuration))
        transform = self.robot.pose(group.name, configuration)
        # current_ee is also a child of the translated group root.
        position = transform[:3, 3]
        from scipy.spatial.transform import Rotation

        xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
        self.server.scene.add_frame(
            f"{self._root(group)}/current_ee",
            axes_length=0.08,
            axes_radius=0.004,
            position=position,
            wxyz=(xyzw[3], *xyzw[:3]),
            visible=bool(self.show_frames.value),
        )
        self.tails[group.name].append(position)
        self._draw_tail(group, np.asarray(self.tails[group.name]))

    def _draw_tail(self, group: RobotGroup, points: FloatArray) -> None:
        name = f"{self._root(group)}/achieved_tail"
        if len(points) < 2:
            self.server.scene.add_line_segments(
                name,
                np.empty((0, 2, 3)),
                group.trail_color,
                visible=False,
            )
            return
        self.server.scene.add_line_segments(
            name,
            np.stack((points[:-1], points[1:]), axis=1),
            group.trail_color,
            line_width=3,
            visible=bool(self.show_tail.value),
        )

    def close(self) -> None:
        self.running = False
        self.receiver.close()
        self.control_server.close()
        self.server.stop()


def _demo(viewer: PolicyViewer) -> None:
    elapsed_s = 0.0
    chunk_id = 0
    next_plan_s = 0.0
    while viewer.running:
        joints, actions = viewer.robot.demo_sample(elapsed_s, horizon=25)
        if elapsed_s >= next_plan_s:
            viewer.on_message(
                PolicyPlan(
                    policy="Synthetic demo",
                    instruction="Inspect a predicted action chunk",
                    actions=actions,
                    action_dt=1 / 30,
                    inference_ms=824,
                    chunk_id=chunk_id,
                    robot=viewer.robot.name,
                ).to_wire()
            )
            chunk_id += 1
            next_plan_s = elapsed_s + 2.0
        height, width = 180, 320
        x = np.broadcast_to(np.linspace(0, 1, width)[None, :], (height, width))
        y = np.broadcast_to(np.linspace(0, 1, height)[:, None], (height, width))
        camera = np.stack((x, y, np.full_like(x, 0.3)), axis=-1)
        viewer.on_message(
            RobotSnapshot(
                joint_positions=joints,
                cameras={"overview": (camera * 255).astype(np.uint8)},
                step=int(elapsed_s * 30),
                max_steps=1000,
                robot=viewer.robot.name,
            ).to_wire()
        )
        elapsed_s += 0.05
        time.sleep(0.05)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8086)
    parser.add_argument("--bridge-endpoint", default="tcp://127.0.0.1:5568")
    parser.add_argument("--control-endpoint", default="tcp://127.0.0.1:5569")
    parser.add_argument(
        "--robot",
        default="yam",
        help="built-in name, installed entry point, or module:factory",
    )
    parser.add_argument(
        "--robot-model-root",
        type=Path,
        help="optional model/dependency root forwarded to the robot adapter",
    )
    parser.add_argument(
        "--list-robots",
        action="store_true",
        help="list bundled/discovered adapters and exit",
    )
    parser.add_argument("--demo", action="store_true", help="show synthetic data without hardware")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.list_robots:
        print("\n".join(available_robot_adapters()))
        return
    robot = load_robot_adapter(args.robot, args.robot_model_root)
    viewer = PolicyViewer(
        args.host,
        args.port,
        args.bridge_endpoint,
        args.control_endpoint,
        robot,
    )
    if args.demo:
        threading.Thread(target=_demo, args=(viewer,), daemon=True).start()
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    print(f"Robot adapter: {robot.name} ({robot.label})")
    print(f"Open http://localhost:{args.port}")
    try:
        while not stop.wait(0.25):
            if (
                viewer.episode_active
                and viewer.last_state_time
                and time.time() - viewer.last_state_time > 2
            ):
                viewer.status.content = "🟠 **Executor heartbeat lost**"
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
