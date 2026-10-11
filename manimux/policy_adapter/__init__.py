"""Load adapters from experiment configuration without legacy aliases or model-specific registries."""

from manimux.plugins import load_plugin
from manimux.policy_adapter.base import PolicyAdapter


def build_policy_adapter(
    robot: dict, policy: dict, *, kinematics=None, motion_limits=None
) -> PolicyAdapter:
    adapter_class = load_plugin(
        policy["adapter"]["type"], group="manimux.policy_adapter", builtins={}
    )
    options = {"kinematics": kinematics}
    if getattr(adapter_class, "uses_motion_limits", False):
        options["motion_limits"] = motion_limits
    return adapter_class(robot, policy, **options)
