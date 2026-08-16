"""Built-in and third-party robot adapter discovery."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import cast

from .base import RobotAdapter, RobotGroup, SceneBox

BUILTIN_ROBOTS = ("yam",)


def available_robot_adapters() -> tuple[str, ...]:
    discovered = set(BUILTIN_ROBOTS)
    discovered.update(entry.name for entry in metadata.entry_points(group="manimux.viewer.robots"))
    return tuple(sorted(discovered))


def _factory_from_reference(reference: str) -> Callable[..., RobotAdapter]:
    if ":" not in reference:
        raise ValueError(
            f"unknown robot adapter {reference!r}; use a built-in name or module:factory"
        )
    module_name, attribute_name = reference.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name)
    if not callable(factory):
        raise TypeError(f"robot adapter factory is not callable: {reference}")
    return cast(Callable[..., RobotAdapter], factory)


def load_robot_adapter(
    name_or_reference: str, model_root: Path | str | None = None
) -> RobotAdapter:
    """Load a built-in, installed entry-point, or ``module:factory`` adapter."""

    adapter: RobotAdapter
    if name_or_reference == "yam":
        from .yam import YamAdapter

        adapter = YamAdapter(model_root=model_root)
    else:
        matching = [
            entry
            for entry in metadata.entry_points(group="manimux.viewer.robots")
            if entry.name == name_or_reference
        ]
        factory = cast(
            Callable[..., RobotAdapter],
            matching[0].load() if matching else _factory_from_reference(name_or_reference),
        )
        adapter = factory(model_root=model_root) if model_root is not None else factory()
    if not isinstance(adapter, RobotAdapter):
        raise TypeError(
            f"adapter {name_or_reference!r} returned {type(adapter).__name__}, "
            "which does not implement RobotAdapter"
        )
    return adapter


__all__ = [
    "RobotAdapter",
    "RobotGroup",
    "SceneBox",
    "available_robot_adapters",
    "load_robot_adapter",
]
