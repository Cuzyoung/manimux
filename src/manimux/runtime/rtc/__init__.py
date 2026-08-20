"""Real-time chunking runtime (arXiv:2506.07339)."""

from manimux.runtime.rtc.mask import inpainting_condition, soft_mask
from manimux.runtime.rtc.request import RtcInferenceRequest
from manimux.runtime.rtc.runtime import RtcRuntime

__all__ = [
    "RtcInferenceRequest",
    "RtcRuntime",
    "inpainting_condition",
    "soft_mask",
]
