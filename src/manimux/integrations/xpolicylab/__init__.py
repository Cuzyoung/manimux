"""Bridge to an XPolicyLab WebSocket policy server.

XPolicyLab (https://github.com/XPolicyLab/XPolicyLab) standardises how a policy
is served: one ``Model`` per policy, launched from its own environment behind a
WebSocket. This package lets ManiMux drive any of them without adopting its
evaluation loop, which steps synchronously and would defeat the point of an
asynchronous runtime.

Nothing here imports XPolicyLab. It is a peer process reached over a socket;
its source is pinned by the ``XPolicyLab/`` submodule and runs in its own model
environment.
"""

from manimux.integrations.xpolicylab.policy_plugin import (
    XPolicyLabAdapter,
    XPolicyLabWsPolicyModel,
    build_adapter,
    build_model,
)

__all__ = [
    "XPolicyLabAdapter",
    "XPolicyLabWsPolicyModel",
    "build_adapter",
    "build_model",
]
