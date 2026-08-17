from __future__ import annotations

import importlib
from importlib import metadata
from typing import TypeVar, cast


class PluginError(ValueError):
    """Raised when a configured ManiMux plugin cannot be resolved."""


PluginT = TypeVar("PluginT")


def load_plugin(
    name: str,
    *,
    group: str,
    builtins: dict[str, PluginT],
) -> PluginT:
    """Resolve a built-in, entry-point, or ``module:attribute`` plugin.

    Built-ins keep the checked-in V1 configuration stable. Entry points make
    separately installed policy/robot packages discoverable, while the explicit
    module form is useful during local development without an installation step.
    """

    if name in builtins:
        return builtins[name]

    if ":" in name:
        module_name, attribute = name.split(":", 1)
        if not module_name or not attribute:
            raise PluginError(f"invalid plugin reference {name!r}; expected module:attribute")
        try:
            module = importlib.import_module(module_name)
            return cast(PluginT, getattr(module, attribute))
        except (ImportError, AttributeError) as exc:
            raise PluginError(f"cannot load plugin {name!r}: {exc}") from exc

    matches = [entry for entry in metadata.entry_points(group=group) if entry.name == name]
    if not matches:
        available = sorted(
            {*builtins, *(entry.name for entry in metadata.entry_points(group=group))}
        )
        choices = ", ".join(available) or "none"
        raise PluginError(f"unknown {group} plugin {name!r}; available: {choices}")
    if len(matches) > 1:
        raise PluginError(f"multiple {group} entry points are registered as {name!r}")
    try:
        return cast(PluginT, matches[0].load())
    except Exception as exc:
        raise PluginError(f"cannot load {group} plugin {name!r}: {exc}") from exc
