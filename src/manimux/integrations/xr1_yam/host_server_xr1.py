"""Xiaomi Robotics 1 (XR-1) inference server.

Same ``/act`` wire protocol as the MolmoAct2 and ABC servers, so every ManiMux
policy looks alike from the runtime's side:

  * 3 cameras, roles [ego, left wrist, right wrist]
  * raw robot state is shape (14,)  (per-arm 6 joints + 1 gripper, two arms)

What differs is the *output*: XR-1 emits ``(30, 60)`` end-effector deltas
expressed in the current end-effector frame, not joint positions. The server
returns them raw; ``policy_plugin.XR1YamAdapter`` reconstructs absolute poses and
solves IK, because turning Cartesian targets into joints is embodiment
knowledge, not model knowledge.

Wire protocol:

    GET  /act        -> health check, returns {"status": "ok", ...}
    POST /act        -> action inference
        request body  (json_numpy):
            {
              "top_cam":     ndarray(H, W, 3) uint8 RGB,   # ego / scene view
              "left_cam":    ndarray(H, W, 3) uint8 RGB,   # left wrist
              "right_cam":   ndarray(H, W, 3) uint8 RGB,   # right wrist
              "instruction": str,
              "state":       ndarray(14,) float32,
              "timestamp":   float (optional),
              "action_prefix": ndarray(N, 60) float32 (optional, async execution),
            }
        response body (json_numpy):
            {"actions": ndarray(30, 60) float32, "dt_ms": float}

Run:

    manimux-xr1-server --host 0.0.0.0 --port 8400 \
        --checkpoint checkpoints/pretrained/xiaomi/model_states.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import json_numpy
import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from manimux.integrations.xr1_yam.mibot.utils.io import (
    ACTION_DIM,
    ACTION_EPS,
    STATE_DIM,
    build_action_mask,
    compose_state,
    denormalize_action,
    resize_image,
    validate_quantiles,
    validate_stats,
)

# Patches the stdlib `json` module so np.ndarray round-trips through JSON.
json_numpy.patch()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("xr1.yam.server")

_HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = "checkpoints/pretrained/xiaomi/model_states.pt"
DEFAULT_PROCESSOR = "checkpoints/pretrained/xiaomi/qwen3_vl_4b_processor"
# Computed from our own YAM recordings. The upstream demo statistics that used
# to sit here belong to a different robot: they saturate YAM's [0,1] gripper to a
# constant and scale the gripper action by ~4x. washer_demo.json is kept for
# reference and can still be selected with --norm-stats.
DEFAULT_NORM_STATS = str(_HERE / "norm_stats" / "yam.json")
ACTION_LENGTH = 30
ARM_JOINTS = 6
YAM_STATE_DIM = 14
MAX_PIXELS = 160_000
RESIZE_FACTOR = 32


def _load_norm_stats(path: str, device: torch.device) -> dict[str, torch.Tensor]:
    payload = json.loads(Path(path).expanduser().read_text())
    action_length = int(payload.get("action_length", ACTION_LENGTH))
    mean, std = validate_stats(payload["mean"], payload["std"], action_length)
    q01, q99 = validate_quantiles(payload["q01"], payload["q99"])
    source = payload.get("_source")
    if source:
        log.warning("normalization statistics source: %s", source)
    return {
        "mean": torch.tensor(np.asarray(mean, dtype=np.float32), device=device),
        "std": torch.tensor(np.asarray(std, dtype=np.float32), device=device),
        "q01": torch.tensor(np.asarray(q01, dtype=np.float32), device=device),
        "q99": torch.tensor(np.asarray(q99, dtype=np.float32), device=device),
        "action_mask": torch.from_numpy(build_action_mask(action_length)).to(device),
    }


class Policy:
    """Load XR-1 once and serve single-observation action chunks."""

    def __init__(
        self,
        checkpoint: str,
        processor_dir: str = DEFAULT_PROCESSOR,
        norm_stats_path: str = DEFAULT_NORM_STATS,
        device: str = "cuda:0",
    ) -> None:
        from transformers import AutoProcessor

        from manimux.integrations.xr1_yam.mibot.models import MIMODEL
        from manimux.integrations.xr1_yam.mibot.models.VLA import XR1 as xr1_module

        self.device = torch.device(device)
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.processor_dir = str(Path(processor_dir).expanduser().resolve())

        # Build the VLM from the local Qwen3-VL config so nothing hits the network.
        # `_from_config` random-initializes; every weight comes from the checkpoint.
        xr1_module.QWEN_VL_CONFIG_SOURCE = self.processor_dir
        log.info("Building XR-1 with Qwen3-VL config from %s", self.processor_dir)
        model = MIMODEL.build(
            {
                "type": "xr1",
                "freq_coefficient": 1.0,
                "freq_excluded_dims": [17, 18, 19],
                "ffn_gradient_checkpointing": False,
                "async_train": True,
            }
        ).to(torch.bfloat16)

        log.info("Loading %s", self.checkpoint)
        state_dict = torch.load(
            self.checkpoint, map_location="cpu", mmap=True, weights_only=False
        )["module"]
        prefix = "model."
        stripped = {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        model.load_state_dict(stripped, strict=True)
        self.model = model.eval().to(self.device)
        log.info("Loaded %d tensors", len(stripped))

        self.stats = _load_norm_stats(norm_stats_path, self.device)
        self.norm_stats_path = norm_stats_path

        self.processor = AutoProcessor.from_pretrained(self.processor_dir)
        self.processor.tokenizer.padding_side = "right"

        # Flow sampling is not concurrency-safe and clients poll at ~30 Hz.
        self._lock = threading.Lock()

    @staticmethod
    def _messages(instruction: str, ego: Any, left_wrist: Any, right_wrist: Any) -> list[dict]:
        """Verbatim from the upstream client: view headers and the /no_cot suffix
        are part of the trained prompt, not cosmetics."""
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "The following observations are captured from multiple "
                            "views.\n# Ego View\n"
                        ),
                    },
                    {"type": "image", "image": ego},
                    {"type": "text", "text": "\n# Left-Wrist View\n"},
                    {"type": "image", "image": left_wrist},
                    {"type": "text", "text": "\n# Right-Wrist View\n"},
                    {"type": "image", "image": right_wrist},
                    {
                        "type": "text",
                        "text": (
                            "\nGenerate robot actions for the task:\n"
                            f"{instruction} /no_cot"
                        ),
                    },
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
        ]

    def predict(
        self,
        *,
        top_cam: Any,
        left_cam: Any,
        right_cam: Any,
        instruction: str,
        state: Any,
        action_prefix: Any = None,
        action_condition: Any = None,
        condition_weights: Any = None,
        rtc_beta: float = 5.0,
    ) -> np.ndarray:
        from PIL import Image

        state_np = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_np.shape != (YAM_STATE_DIM,):
            raise ValueError(f"state must have shape ({YAM_STATE_DIM},), got {state_np.shape}")

        images = [
            resize_image(
                Image.fromarray(_as_rgb_uint8(frame), mode="RGB"),
                factor=RESIZE_FACTOR,
                max_pixels=MAX_PIXELS,
            )
            for frame in (top_cam, left_cam, right_cam)
        ]

        # YAM state is [left 6 joints, left gripper, right 6 joints, right gripper];
        # compose_state left-aligns each arm inside its 7-wide slot and zero-pads
        # the rest of the 60-D vector.
        composed = compose_state(
            left_joint=state_np[:ARM_JOINTS],
            left_gripper=state_np[ARM_JOINTS],
            right_joint=state_np[ARM_JOINTS + 1 : 2 * ARM_JOINTS + 1],
            right_gripper=state_np[-1],
        )

        with self._lock, torch.no_grad():
            batch = self.processor.apply_chat_template(
                [self._messages(instruction, *images)],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
                images_kwargs={"do_resize": False},
            )
            batch = {
                key: (value.to(self.device) if isinstance(value, torch.Tensor) else value)
                for key, value in batch.items()
            }
            batch_size = batch["input_ids"].shape[0]
            mask = self.stats["action_mask"].unsqueeze(0).expand(batch_size, -1, -1)

            if action_prefix is None:
                batch["action"] = torch.zeros(
                    (batch_size, *self.stats["mean"].shape),
                    device=self.device,
                    dtype=torch.bfloat16,
                )
            else:
                prefix = np.asarray(action_prefix, dtype=np.float32)
                if prefix.ndim != 2 or prefix.shape[1] != ACTION_DIM:
                    raise ValueError(
                        f"action_prefix must have shape (N, {ACTION_DIM}), got {prefix.shape}"
                    )
                padded = np.zeros((ACTION_LENGTH, ACTION_DIM), dtype=np.float32)
                padded[: len(prefix)] = prefix[:ACTION_LENGTH]
                raw = torch.from_numpy(padded)[None].to(self.device)
                batch["action"] = ((raw - self.stats["mean"]) / (self.stats["std"] + ACTION_EPS)) * mask
                batch["prefix_length"] = min(len(prefix), ACTION_LENGTH)

            batch["action_mask"] = mask
            batch["state"] = self._normalize_state(composed)

            if (action_condition is None) != (condition_weights is None):
                raise ValueError(
                    "action_condition and action_condition_weights must be provided together"
                )
            if action_condition is None:
                action = self.model.generate(batch)
            else:
                condition_t, weights_t = self._rtc_tensors(action_condition, condition_weights)
                with self.model.rtc_condition(condition_t, weights_t, beta=float(rtc_beta)):
                    action = self.model.generate(batch)
            action = denormalize_action(action * mask, self.stats["mean"], self.stats["std"]) * mask

        out = action.float().detach().cpu().numpy()
        if out.ndim == 3:
            out = out[0]
        if out.shape != (ACTION_LENGTH, ACTION_DIM):
            raise ValueError(f"expected ({ACTION_LENGTH}, {ACTION_DIM}) actions, got {out.shape}")
        return out.astype(np.float32)

    def _rtc_tensors(self, action_condition: Any, condition_weights: Any):
        """Validate the condition and lift it into the model's normalized space.

        The wire carries raw robot units; guidance is applied in the same space
        the model denoises in, so this normalization is not optional.
        """
        condition = np.asarray(action_condition, dtype=np.float32)
        if condition.shape != (ACTION_LENGTH, ACTION_DIM):
            raise ValueError(
                f"action_condition must have shape {(ACTION_LENGTH, ACTION_DIM)}, "
                f"got {condition.shape}"
            )
        weights = np.asarray(condition_weights, dtype=np.float32).reshape(-1)
        if weights.shape != (ACTION_LENGTH,):
            raise ValueError(
                f"action_condition_weights must have shape {(ACTION_LENGTH,)}, "
                f"got {weights.shape}"
            )
        if not np.isfinite(condition).all():
            raise ValueError("action_condition contains non-finite values")
        if not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)):
            raise ValueError("action_condition_weights must be finite values in [0, 1]")

        mean = self.stats["mean"].float().cpu().numpy()
        std = self.stats["std"].float().cpu().numpy()
        normalized = (condition - mean) / (std + ACTION_EPS)
        # Columns this embodiment never drives carry no signal; the mask zeroes
        # them in the model, so keep the guidance target zero there too.
        normalized = normalized * self.stats["action_mask"].float().cpu().numpy()
        condition_t = torch.from_numpy(normalized)[None].to(
            device=self.device, dtype=torch.bfloat16
        )
        weights_t = torch.from_numpy(weights)[None].to(
            device=self.device, dtype=torch.bfloat16
        )
        return condition_t, weights_t

    def _normalize_state(self, composed: np.ndarray) -> torch.Tensor:
        """Quantile-normalize to [-1, 1]; dims with q99 == q01 stay at zero."""
        state = torch.from_numpy(composed).to(device=self.device, dtype=torch.float32)[None]
        q01, q99 = self.stats["q01"].float(), self.stats["q99"].float()
        valid = (q99 > q01)[0]
        normalized = torch.zeros_like(state)
        normalized[..., valid] = (
            2.0
            * (state[..., valid] - q01[..., valid])
            / (q99[..., valid] - q01[..., valid] + ACTION_EPS)
            - 1.0
        )
        return normalized.clamp(-1.0, 1.0).to(torch.bfloat16)


def _as_rgb_uint8(arr: Any) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"image must be HxWx3, got shape {a.shape}")
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)


def build_app(policy: Policy) -> FastAPI:
    app = FastAPI(title="Xiaomi-Robotics-1 Bimanual YAM server", version="0.1.0")

    @app.get("/act")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "checkpoint": policy.checkpoint,
                "device": str(policy.device),
                "dtype": "bfloat16",
                "num_cameras": 3,
                "camera_order": ["ego", "left_wrist", "right_wrist"],
                "state_dim": STATE_DIM,
                "robot_state_dim": YAM_STATE_DIM,
                "action_dim": ACTION_DIM,
                "chunk_length": ACTION_LENGTH,
                "action_space": "ee_delta",
                "rtc_mode": "pi_guided_v1",
                "norm_stats": policy.norm_stats_path,
            }
        )

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.post("/act")
    async def act(request: Request) -> Response:
        raw = await request.body()
        try:
            payload = json_numpy.loads(raw.decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            return _error_response(400, f"failed to decode json_numpy body: {e}")

        try:
            top_cam = payload["top_cam"]
            left_cam = payload["left_cam"]
            right_cam = payload["right_cam"]
            instruction = str(payload["instruction"])
            state = payload["state"]
        except KeyError as e:
            return _error_response(400, f"missing required field: {e}")

        t0 = time.perf_counter()
        try:
            actions = policy.predict(
                top_cam=top_cam,
                left_cam=left_cam,
                right_cam=right_cam,
                instruction=instruction,
                state=state,
                action_prefix=payload.get("action_prefix"),
                action_condition=payload.get("action_condition"),
                condition_weights=payload.get("action_condition_weights"),
                rtc_beta=float(payload.get("rtc_beta", 5.0)),
            )
        except Exception as e:  # noqa: BLE001
            log.exception("inference failed")
            return _error_response(500, f"inference failed: {e}")
        dt_ms = (time.perf_counter() - t0) * 1000.0

        body = json_numpy.dumps({"actions": actions, "dt_ms": dt_ms})
        return Response(content=body, media_type="application/json")

    return app


def _error_response(status: int, message: str) -> Response:
    body = json_numpy.dumps({"error": message})
    return Response(content=body, status_code=status, media_type="application/json")


def warmup(policy: Policy) -> None:
    log.info("Warming up model with dummy frames ...")
    dummy_img = np.zeros((360, 640, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    try:
        actions = policy.predict(
            top_cam=dummy_img,
            left_cam=dummy_img,
            right_cam=dummy_img,
            instruction="warmup",
            state=np.zeros(YAM_STATE_DIM, dtype=np.float32),
        )
    except Exception:  # noqa: BLE001
        log.exception("warmup inference failed (server will still start)")
        return
    log.info(
        "Warmup OK (%.1f ms, actions %s)", (time.perf_counter() - t0) * 1000.0, actions.shape
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Xiaomi-Robotics-1 Bimanual YAM inference server")
    p.add_argument("--host", default="0.0.0.0", help="bind address (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8400, help="bind port (default: 8400)")
    p.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"XR-1 model_states.pt (default: {DEFAULT_CHECKPOINT})",
    )
    p.add_argument(
        "--processor",
        default=DEFAULT_PROCESSOR,
        help="local Qwen3-VL processor/config directory",
    )
    p.add_argument(
        "--norm-stats",
        default=DEFAULT_NORM_STATS,
        help="JSON with mean/std/q01/q99; the bundled default is NOT YAM's",
    )
    p.add_argument("--device", default="cuda:0", help="torch device (default: cuda:0)")
    p.add_argument("--no-warmup", action="store_true", help="skip warmup pass")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    policy = Policy(
        checkpoint=args.checkpoint,
        processor_dir=args.processor,
        norm_stats_path=args.norm_stats,
        device=args.device,
    )
    if not args.no_warmup:
        warmup(policy)

    app = build_app(policy)

    import uvicorn

    log.info("Listening on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
