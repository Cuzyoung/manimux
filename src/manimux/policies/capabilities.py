from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PolicyCapabilities:
    """Capabilities of the live policy backend, reported after startup."""

    sampling_modes: frozenset[str] = frozenset({"default"})

    def supports(self, sampling_mode: str) -> bool:
        return sampling_mode in self.sampling_modes
