#!/usr/bin/env python3
"""Launch XPolicyLab Pi_05 with a YAM zero-shot config and Pi-RTC sampling."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

XPOLICY_ROOT = Path("/home/ubuntu/XPolicyLab")
OPENPI_SRC = XPOLICY_ROOT / "policy/Pi_05/openpi/src"
DEFAULT_CONFIG = Path("/home/ubuntu/manimux/configs/pi05-base-yam-server.yaml")
CONFIG_NAME = "pi05_base_yam_zero_shot"
BASE_CONFIG_NAME = "pi05_base_aloha_full_sim_arx-x5_seed_0"


def _prepare_imports() -> None:
    for path in (XPOLICY_ROOT, OPENPI_SRC):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _register_yam_config(norm_stats_dir: Path) -> None:
    from openpi.training import config as openpi_config

    base = openpi_config.get_config(BASE_CONFIG_NAME)
    yam_data = dataclasses.replace(
        base.data,
        repo_id="yam-bimanual-merged",
        assets=openpi_config.AssetsConfig(
            assets_dir=str(norm_stats_dir.parent),
            asset_id=norm_stats_dir.name,
        ),
        use_delta_joint_actions=False,
        adapt_to_pi=False,
    )
    openpi_config._CONFIGS_DICT[CONFIG_NAME] = dataclasses.replace(  # noqa: SLF001
        base,
        name=CONFIG_NAME,
        data=yam_data,
    )


def _install_rtc_sampler() -> None:
    import einops
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_module
    from openpi.models import pi0 as pi0_module

    def sample_actions(
        self: Any,
        rng: Any,
        observation: Any,
        *,
        num_steps: int = 10,
        noise: Any = None,
        action_condition: Any = None,
        condition_weights: Any = None,
        rtc_beta: float = 5.0,
    ) -> Any:
        observation = model_module.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(
                rng,
                (batch_size, self.action_horizon, self.action_dim),
            )

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = pi0_module.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
        )

        def predict_velocity(x_t: Any, time_value: Any) -> Any:
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation,
                x_t,
                jnp.broadcast_to(time_value, batch_size),
            )
            suffix_attn_mask = pi0_module.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_to_suffix_mask = einops.repeat(
                prefix_mask,
                "b p -> b s p",
                s=suffix_tokens.shape[1],
            )
            full_attn_mask = jnp.concatenate(
                [prefix_to_suffix_mask, suffix_attn_mask],
                axis=-1,
            )
            suffix_positions = (
                jnp.sum(prefix_mask, axis=-1)[:, None]
                + jnp.cumsum(suffix_mask, axis=-1)
                - 1
            )
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=suffix_positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            if prefix_out is not None:
                raise AssertionError("cached Pi0 prefix unexpectedly produced an output")
            return self.action_out_proj(suffix_out[:, -self.action_horizon :])

        target = None if action_condition is None else jnp.asarray(action_condition)
        weights = None if condition_weights is None else jnp.asarray(condition_weights)

        def step(carry: tuple[Any, Any]) -> tuple[Any, Any]:
            x_t, time_value = carry
            if target is None:
                velocity = predict_velocity(x_t, time_value)
            else:

                def clean_estimate(sample: Any) -> tuple[Any, Any]:
                    raw_velocity = predict_velocity(sample, time_value)
                    return sample - time_value * raw_velocity, raw_velocity

                clean, pullback, velocity = jax.vjp(clean_estimate, x_t, has_aux=True)
                weighted_error = (target - clean) * weights
                guidance = pullback(weighted_error)[0]
                tau = 1.0 - time_value
                denominator = jnp.maximum(time_value * tau, jnp.finfo(x_t.dtype).eps)
                raw_scale = (time_value**2 + tau**2) / denominator
                guidance_scale = jnp.minimum(jnp.asarray(rtc_beta), raw_scale)
                velocity = velocity - guidance_scale * guidance
            return x_t + dt * velocity, time_value + dt

        def cond(carry: tuple[Any, Any]) -> Any:
            return carry[1] >= -dt / 2

        actions, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return actions

    pi0_module.Pi0.sample_actions = sample_actions


def _install_policy_infer() -> None:
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_module
    from openpi.policies import policy as policy_module

    def infer(
        self: Any,
        obs: dict[str, Any],
        *,
        noise: np.ndarray | None = None,
        num_steps: int = 10,
        action_condition: np.ndarray | None = None,
        condition_weights: np.ndarray | None = None,
        rtc_beta: float = 5.0,
    ) -> dict[str, Any]:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if not math.isfinite(rtc_beta) or rtc_beta <= 0:
            raise ValueError(f"rtc_beta must be finite and positive, got {rtc_beta}")
        if (action_condition is None) != (condition_weights is None):
            raise ValueError("action_condition and condition_weights must be provided together")

        raw_inputs = jax.tree.map(lambda value: value, obs)
        if action_condition is not None:
            raw_inputs["actions"] = np.asarray(action_condition, dtype=np.float32)
        inputs = self._input_transform(raw_inputs)  # noqa: SLF001
        transformed_condition = inputs.pop("actions", None)

        is_batched = np.asarray(inputs["state"]).ndim > 1
        if not is_batched:
            inputs = jax.tree.map(lambda value: jnp.asarray(value)[np.newaxis, ...], inputs)
        else:
            inputs = jax.tree.map(jnp.asarray, inputs)
        self._rng, sample_rng = jax.random.split(self._rng)  # noqa: SLF001

        sample_kwargs = dict(self._sample_kwargs)  # noqa: SLF001
        sample_kwargs["num_steps"] = int(num_steps)
        if noise is not None:
            noise_array = jnp.asarray(noise)
            if not is_batched and noise_array.ndim == 2:
                noise_array = noise_array[None, ...]
            sample_kwargs["noise"] = noise_array
        if transformed_condition is not None:
            target = jnp.asarray(transformed_condition)
            if not is_batched and target.ndim == 2:
                target = target[None, ...]
            weights = jnp.asarray(condition_weights, dtype=target.dtype)
            if not is_batched and weights.ndim == 1:
                weights = weights[None, ...]
            if weights.ndim == 2:
                weights = weights[..., None]
            expected_target = (
                inputs["state"].shape[0],
                self._model.action_horizon,  # noqa: SLF001
                self._model.action_dim,  # noqa: SLF001
            )
            expected_weights = expected_target[:2] + (1,)
            if target.shape != expected_target:
                raise ValueError(
                    "transformed action_condition must have shape "
                    f"{expected_target}, got {target.shape}"
                )
            if weights.shape != expected_weights:
                raise ValueError(
                    f"condition_weights must have shape {expected_weights}, got {weights.shape}"
                )
            sample_kwargs.update(
                action_condition=target,
                condition_weights=weights,
                rtc_beta=float(rtc_beta),
            )

        observation = model_module.Observation.from_dict(inputs)
        started = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng, observation, **sample_kwargs),  # noqa: SLF001
        }
        model_time = time.monotonic() - started
        if not is_batched:
            outputs = jax.tree.map(lambda value: np.asarray(value[0, ...]), outputs)
        else:
            outputs = jax.tree.map(np.asarray, outputs)
        outputs = self._output_transform(outputs)  # noqa: SLF001
        outputs["policy_timing"] = {"infer_ms": model_time * 1000.0}
        return outputs

    policy_module.Policy.infer = infer


def _install_xpolicy_model(config: dict[str, Any]) -> None:
    from openpi.policies import policy_config
    from openpi.shared import normalize
    from openpi.training import config as openpi_config
    from XPolicyLab.policy.Pi_05 import model as xpolicy_model

    original_init = xpolicy_model.Model.__init__
    original_get_action = xpolicy_model.Model.get_action

    def get_model(self: Any, model_cfg: dict[str, Any]) -> Any:
        model_root = xpolicy_model._resolve_pi05_model_root(model_cfg)  # noqa: SLF001
        norm_stats_dir = Path(str(model_cfg["norm_stats_path"])).expanduser().resolve()
        stats = normalize.load(norm_stats_dir)
        train_config = openpi_config.get_config(str(model_cfg["train_config_name"]))
        return policy_config.create_trained_policy(
            train_config,
            str(model_root),
            norm_stats=stats,
        )

    def init(self: Any, model_cfg: dict[str, Any]) -> None:
        original_init(self, model_cfg)
        self._pi05_num_steps = int(model_cfg.get("num_steps", 5))

    def get_action(self: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("num_steps", self._pi05_num_steps)
        return original_get_action(self, **kwargs)

    def get_action_rtc(self: Any, payload: dict[str, Any]) -> Any:
        required = {"action_condition", "condition_weights", "rtc_beta"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"RTC payload missing fields: {missing}")
        return self.get_action(
            action_condition=np.asarray(payload["action_condition"], dtype=np.float32),
            condition_weights=np.asarray(payload["condition_weights"], dtype=np.float32),
            rtc_beta=float(payload["rtc_beta"]),
        )

    xpolicy_model.Model.get_model = get_model
    xpolicy_model.Model.__init__ = init
    xpolicy_model.Model.get_action = get_action
    xpolicy_model.Model.get_action_rtc = get_action_rtc

    expected_horizon = openpi_config.get_config(CONFIG_NAME).model.action_horizon
    if expected_horizon != 50:
        raise ValueError(f"expected pi05_base horizon 50, got {expected_horizon}")
    if int(config.get("num_steps", 0)) <= 0:
        raise ValueError("server config num_steps must be positive")


def _load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"server config must be a mapping: {path}")
    return loaded


def _validate_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    model_root = Path(str(config["model_path"])).expanduser().resolve()
    stats_dir = Path(str(config["norm_stats_path"])).expanduser().resolve()
    if not (model_root / "params").is_dir():
        raise FileNotFoundError(f"pi05_base params not found under {model_root}")
    if not (stats_dir / "norm_stats.json").is_file():
        raise FileNotFoundError(f"YAM norm_stats.json not found under {stats_dir}")
    return model_root, stats_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check", action="store_true", help="validate and print resolved setup")
    args = parser.parse_args()

    config = _load_config(args.config.resolve())
    model_root, stats_dir = _validate_paths(config)
    _prepare_imports()
    _register_yam_config(stats_dir)
    _install_rtc_sampler()
    _install_policy_infer()
    _install_xpolicy_model(config)

    if args.check:
        print(
            json.dumps(
                {
                    "train_config_name": CONFIG_NAME,
                    "model_root": str(model_root),
                    "norm_stats": str(stats_dir / "norm_stats.json"),
                    "action_space": "absolute_joint_position",
                    "action_horizon": 50,
                    "num_steps": int(config["num_steps"]),
                    "rtc": "pi_guided_v1",
                },
                indent=2,
            )
        )
        return 0

    import setup_policy_server

    setup_policy_server.main(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
