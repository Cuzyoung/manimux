from __future__ import annotations

from manimux.types import ActionHorizon, RobotCommand, RobotState


class DirectExecutor:
    """Forward the time-aligned policy reference without shaping or clipping."""

    @property
    def horizon_steps(self) -> int:
        return 2

    def reset(self, state: RobotState) -> None:
        del state

    def step(
        self,
        now_ns: int,
        state: RobotState,
        reference: ActionHorizon,
    ) -> RobotCommand:
        del state
        return RobotCommand(
            groups={name: values[0].copy() for name, values in reference.groups.items()},
            monotonic_ns=now_ns,
            plan_id=reference.plan_id,
        )
